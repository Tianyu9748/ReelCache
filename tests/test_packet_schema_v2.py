"""Packet schema v2: legacy v1 decode, explicit v2 headers and validated tile maps."""

import base64
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

from reelcache import _hevc
from reelcache.codecs import (
    HEVCProvider, NodeEncoding, decode_packed, decode_packed_pixels, decode_tensor, encode_node,
    encode_packed, encode_tensor, encoded_nbytes, packed_keys,
)
from reelcache.errors import ConfigurationError, IntegrityError, UnsupportedError
from reelcache.nvdec import NVDECDecoder
from reelcache.types import CodecSpec, EncodedTensor, MatrixKey, ModelIdentity, TensorContext

FIXTURE = Path(__file__).resolve().parent / "fixtures/legacy_v1_packets.json"


def load_packet(row):
    return EncodedTensor(
        row["metadata"], {name: base64.b64decode(raw) for name, raw in row["payloads"].items()},
        None if row["basis"] is None else base64.b64decode(row["basis"]),
    )


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def rehash(packet, metadata=None, payloads=None):
    metadata = copy.deepcopy(dict(packet.metadata) if metadata is None else metadata)
    payloads = dict(packet.payloads if payloads is None else payloads)
    for name, raw in payloads.items():
        if name in metadata["payloads"]:
            metadata["payloads"][name]["sha256"] = hashlib.sha256(raw).hexdigest()
    return EncodedTensor(metadata, payloads, packet.basis)


def node(layers=2, *, frames=3, height=6, width=10, ranks=(5, 7), pix_fmt="yuv444p", seed=0,
         token_start=90, streams=(None,)):
    model = ModelIdentity("schema-v2", "fixture-v2", layers, {"K": 12, "V": 16}, "tokens_channels", "float32")
    generator = torch.Generator().manual_seed(seed)
    matrices, contexts, specs = {}, {}, {}
    for layer in range(layers):
        for target, rank in zip(("K", "V"), ranks):
            for stream in streams:
                key = MatrixKey(layer, target)
                public = key if stream is None else (key, stream)
                tokens = frames * height * width
                channels = model.channels_for(target)
                ramp = torch.linspace(-1, 1, tokens)[:, None]
                matrices[public] = (
                    ramp * torch.randn(1, channels, generator=generator)
                    + torch.randn(tokens, 3, generator=generator) @ torch.randn(3, channels, generator=generator)
                )
                contexts[public] = TensorContext(
                    model, key, token_start, frames, height, width, "pre_rope" if target == "K" else "none",
                )
                specs[public] = CodecSpec("svd_hevc", rank, pix_fmt=pix_fmt)
    return matrices, contexts, specs


class LegacyVersionOnePacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text())
        if cls.fixture["schema"] != "reelcache.test_fixture.legacy_v1_packets.v1":
            raise AssertionError("unexpected legacy fixture schema")

    def test_real_historical_store_packets_decode_unchanged(self):
        rows = self.fixture["historical_store_packets"]
        self.assertEqual({row["metadata"]["context"]["key"]["target"] for row in rows}, {"K", "V"})
        for row in rows:
            with self.subTest(source=row["source"]):
                packet = load_packet(row)
                before = json.dumps(packet.metadata, sort_keys=True)
                size = encoded_nbytes(packet)
                self.assertEqual(packet.metadata["schema"], "reelcache.tensor.v1")
                self.assertNotIn("inter_prediction", packet.metadata["codec"])
                for verify in (False, True):
                    decoded = decode_tensor(packet, verify_packet=verify)
                    self.assertEqual(tensor_sha(decoded), row["baseline_decoded_sha256"])
                self.assertEqual(json.dumps(packet.metadata, sort_keys=True), before)
                self.assertEqual(encoded_nbytes(packet), size)
                spec = CodecSpec.from_dict(packet.metadata["codec"])
                self.assertIsNone(spec.inter_prediction)
                self.assertEqual(spec.to_dict() | {"ffmpeg_path": None}, {**packet.metadata["codec"], "ffmpeg_path": None})
                self.assertEqual(CodecSpec.from_dict(row["spec"]), spec)
                trace = _hevc.trace_headers(packet.payloads["coefficients.000.hevc"])
                # Historical semantics: x265 preset-default GOP (I then P/B pictures, I at QP5).
                self.assertEqual(trace["slice_type_counts"]["I"], 1)
                self.assertGreaterEqual(trace["slice_type_counts"]["P"] + trace["slice_type_counts"]["B"], 1)
                self.assertIn(5, trace["slice_qps"])
                with self.assertRaisesRegex(ConfigurationError, "decode-only"):
                    encode_tensor(decode_tensor(packet), TensorContext.from_dict(packet.metadata["context"]), spec)

    def test_baseline_generated_packets_decode_bit_identically(self):
        rows = self.fixture["baseline_generated_packets"]
        self.assertEqual({row["pix_fmt"] for row in rows if row["kind"] == "svd_hevc"}, {"yuv444p", "yuv420p"})
        self.assertEqual({row["frames"] for row in rows if row["kind"] == "svd_hevc"}, {1, 3, 8})
        for row in rows:
            with self.subTest(case=row["case"]):
                packet = load_packet(row)
                self.assertEqual(packet.metadata["schema"], "reelcache.tensor.v1")
                for verify in (False, True):
                    self.assertEqual(tensor_sha(decode_tensor(packet, verify_packet=verify)), row["baseline_decoded_sha256"])
                if row["kind"] == "svd_hevc":
                    self.assertTrue(CodecSpec.from_dict(packet.metadata["codec"]).legacy_gop)
                    counts = _hevc.trace_headers(packet.payloads["coefficients.000.hevc"])["slice_type_counts"]
                    if row["frames"] > 1:
                        self.assertGreaterEqual(counts["P"] + counts["B"], 1)
                    else:
                        self.assertEqual(counts, {"I": 1, "P": 0, "B": 0})
                else:
                    self.assertIs(CodecSpec.from_dict(packet.metadata["codec"]).inter_prediction, False)

    def test_new_raw_and_svd_packets_keep_the_v1_format(self):
        for row in self.fixture["baseline_generated_packets"]:
            if row["kind"] == "svd_hevc":
                continue
            packet = load_packet(row)
            context = TensorContext.from_dict(packet.metadata["context"])
            spec = CodecSpec.from_dict(packet.metadata["codec"])
            reencoded = encode_tensor(decode_tensor(packet), context, spec)
            self.assertEqual(reencoded.metadata["schema"], "reelcache.tensor.v1")
            self.assertEqual(set(reencoded.metadata), set(packet.metadata))
            self.assertEqual(reencoded.metadata["codec"], packet.metadata["codec"])

    def test_version_one_headers_cannot_be_relabelled(self):
        row = next(row for row in self.fixture["baseline_generated_packets"] if row["case"] == "causvid_chunk_444_per_channel")
        packet = load_packet(row)
        metadata = copy.deepcopy(packet.metadata)
        metadata["codec"]["inter_prediction"] = False
        with self.assertRaises(IntegrityError):
            decode_tensor(EncodedTensor(metadata, packet.payloads, packet.basis), verify_packet=True)
        metadata = copy.deepcopy(packet.metadata)
        metadata["schema"] = "reelcache.tensor.v2"
        with self.assertRaises(IntegrityError):
            decode_tensor(EncodedTensor(metadata, packet.payloads, packet.basis), verify_packet=True)
        metadata["hevc"] = _hevc.encoder_record(False)
        with self.assertRaises(IntegrityError):
            decode_tensor(EncodedTensor(metadata, packet.payloads, packet.basis), verify_packet=True)


class SingleMatrixVersionTwoTests(unittest.TestCase):
    def test_header_fields_and_strict_validation(self):
        matrices, contexts, specs = node(layers=1)
        key = MatrixKey(0, "K")
        packet = encode_tensor(matrices[key], contexts[key], specs[key], validate_hevc=True)
        self.assertEqual(packet.metadata["schema"], "reelcache.tensor.v2")
        self.assertEqual(packet.metadata["video"]["schema"], "reelcache.channel_tiles.v1")
        decode_tensor(packet, verify_packet=True)
        for mutate in (
            lambda m: m.pop("hevc"),
            lambda m: m.update(hevc={**m["hevc"], "x265_params": {}}),
            lambda m: m["codec"].pop("inter_prediction"),
            lambda m: m["codec"].update(inter_prediction=None),
            lambda m: m.update(schema="reelcache.tensor.v1"),
            lambda m: m.update(schema="reelcache.tensor.v3"),
        ):
            metadata = copy.deepcopy(packet.metadata)
            mutate(metadata)
            with self.assertRaises(IntegrityError):
                decode_tensor(EncodedTensor(metadata, packet.payloads, packet.basis), verify_packet=True)


class PackedStreamTests(unittest.TestCase):
    def test_node_packets_round_trip_both_formats_with_validated_tile_maps(self):
        for pix_fmt in _hevc.PIXEL_FORMATS:
            matrices, contexts, specs = node(pix_fmt=pix_fmt, seed=1)
            encoding = encode_node(matrices, contexts, specs, packing="node", node_id="chunk-0", validate_hevc=True)
            self.assertIsInstance(encoding, NodeEncoding)
            self.assertEqual(len(encoding.packets), 1)
            packet = encoding.packets[0]
            metadata = packet.metadata
            with self.subTest(pix_fmt=pix_fmt):
                self.assertEqual(metadata["schema"], "reelcache.tensor_pack.v2")
                self.assertIsNone(packet.basis)
                self.assertEqual(metadata["node"]["node_id"], "chunk-0")
                self.assertEqual(metadata["codec"]["inter_prediction"], False)
                self.assertEqual(metadata["hevc"], _hevc.encoder_record(False))
                canvas = metadata["canvas"]
                self.assertEqual((canvas["width"] % 32, canvas["height"] % 32), (0, 0))
                self.assertEqual(canvas["pictures"], 4 * 3)
                self.assertEqual(packed_keys(packet), sorted(matrices, key=lambda k: (k.layer, k.target)))
                cursor = 0
                for tile in metadata["tiles"]:
                    self.assertEqual(tile["pictures"], [cursor, 3])
                    self.assertEqual(tile["rect"][:2], [0, 0])
                    cursor += 3
                    self.assertIsNotNone(tile["decoded_coefficients_sha256"])
                padding = metadata["padding"]
                self.assertEqual(padding["canvas_padding_bytes"], padding["canvas_samples"] - padding["tile_samples"])
                trace = _hevc.trace_headers(packet.payloads["stream.hevc"])
                self.assertEqual(trace["slice_type_counts"], {"I": 12, "P": 0, "B": 0})
                decoded = decode_packed(packet, verify_packet=True)
                self.assertEqual(list(decoded), packed_keys(packet))
                for key, value in decoded.items():
                    reference = decode_tensor(encode_tensor(matrices[key], contexts[key], specs[key]))
                    error = ((value - matrices[key]).norm() / matrices[key].norm()).item()
                    self.assertLess(error, 0.05)
                    self.assertLess(((reference - matrices[key]).norm() / matrices[key].norm()).item(), 0.05)

    def test_temporal_tiles_identical_to_same_canvas_single_matrix_streams(self):
        for pix_fmt in _hevc.PIXEL_FORMATS:
            for frames, ranks in ((3, (5, 7)), (8, (4, 9))):
                matrices, contexts, specs = node(frames=frames, ranks=ranks, pix_fmt=pix_fmt, seed=frames)
                packed = encode_node(matrices, contexts, specs, packing="node").packets[0]
                canvas = (packed.metadata["canvas"]["width"], packed.metadata["canvas"]["height"])
                packed_pixels = decode_packed_pixels(packed, verify_packet=True)
                packed_values = decode_packed(packed)
                for key in matrices:
                    with self.subTest(pix_fmt=pix_fmt, frames=frames, key=key):
                        single = encode_packed(
                            {key: matrices[key]}, {key: contexts[key]}, {key: specs[key]},
                            packing="explicit", canvas=canvas,
                        )
                        self.assertEqual(len(single.metadata["tiles"]), 1)
                        self.assertTrue(np.array_equal(decode_packed_pixels(single)[key], packed_pixels[key]))
                        self.assertTrue(torch.equal(decode_tensor(single), packed_values[key]))
                        own = encode_tensor(matrices[key], contexts[key], specs[key])
                        own_pixels, _ = _hevc.decode_pixels(
                            [own.payloads["coefficients.000.hevc"]], own.metadata["video"],
                        )
                        self.assertEqual(own_pixels.shape, packed_pixels[key].shape)

    def test_partial_decode_uses_tile_byte_ranges(self):
        matrices, contexts, specs = node(layers=3, seed=5)
        packet = encode_node(matrices, contexts, specs, packing="node").packets[0]
        self.assertEqual(packet.metadata["packing"]["random_access"], "tile_byte_ranges")
        ranges = [tile["byte_range"] for tile in packet.metadata["tiles"]]
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(sum(length for _, length in ranges), len(packet.payloads["stream.hevc"]))
        full = decode_packed_pixels(packet)
        wanted = [MatrixKey(1, "V"), MatrixKey(2, "K")]
        with mock.patch("reelcache._hevc.decode_pictures", wraps=_hevc.decode_pictures) as spy:
            partial = decode_packed_pixels(packet, keys=wanted)
        self.assertEqual(spy.call_args.kwargs["pictures"], 6)
        for key in wanted:
            self.assertTrue(np.array_equal(partial[key], full[key]))
        values = decode_packed(packet, keys=[MatrixKey(0, "K")])
        self.assertEqual(list(values), [MatrixKey(0, "K")])

    def test_cfg_streams_layer_groups_and_mixed_kinds(self):
        matrices, contexts, specs = node(layers=4, streams=("cond", "uncond"), seed=9)
        cond_raw = (MatrixKey(0, "K"), "raw")
        matrices[cond_raw] = torch.randn(180, 12)
        contexts[cond_raw] = TensorContext(contexts[(MatrixKey(0, "K"), "cond")].model, MatrixKey(0, "K"), 90, 3, 6, 10, "post_rope")
        specs[cond_raw] = CodecSpec()
        encoding = encode_node(matrices, contexts, specs, packing="layer_group", layer_group_size=2)
        self.assertEqual(len(encoding.packets), 3)
        self.assertEqual(encoding.locations[cond_raw], (2, None))
        self.assertEqual(encoding.packets[2].metadata["schema"], "reelcache.tensor.v1")
        first = encoding.packets[0]
        self.assertEqual({key[0].layer for key in packed_keys(first)}, {0, 1})
        self.assertEqual(packed_keys(first)[:2], [(MatrixKey(0, "K"), "cond"), (MatrixKey(0, "K"), "uncond")])
        self.assertEqual(encoding.locations[(MatrixKey(3, "V"), "uncond")], (1, 7))
        decoded = decode_packed(encoding.packets[1], keys=[(MatrixKey(3, "V"), "uncond")])
        self.assertEqual(list(decoded), [(MatrixKey(3, "V"), "uncond")])
        raw = decode_tensor(encoding.packets[2])
        self.assertTrue(torch.equal(raw, matrices[cond_raw]))
        per_matrix = encode_node(matrices, contexts, specs, packing="per_matrix")
        self.assertEqual(len(per_matrix.packets), len(matrices))
        self.assertTrue(all(location[1] is None for location in per_matrix.locations.values()))
        hevc = {key: value for key, value in matrices.items() if specs[key].kind == "svd_hevc"}
        default = encode_node(hevc, {key: contexts[key] for key in hevc}, {key: specs[key] for key in hevc})
        self.assertEqual((default.packing, len(default.packets)), ("node", 1))
        self.assertEqual(HEVCProvider().packing, "node")

    def test_inter_prediction_and_cross_node_packets_are_rejected(self):
        matrices, contexts, specs = node(seed=3)
        inter = {key: CodecSpec("svd_hevc", spec.rank, inter_prediction=True) for key, spec in specs.items()}
        for packing, size in (("node", None), ("layer_group", 1)):
            with self.assertRaisesRegex(ConfigurationError, "inter prediction is rejected"):
                encode_node(matrices, contexts, inter, packing=packing, layer_group_size=size)
        with self.assertRaisesRegex(ConfigurationError, "inter prediction is rejected"):
            encode_packed(matrices, contexts, inter)
        mixed = dict(specs)
        mixed[MatrixKey(1, "V")] = inter[MatrixKey(1, "V")]
        with self.assertRaisesRegex(ConfigurationError, "inter prediction is rejected"):
            encode_node(matrices, contexts, mixed, packing="node")
        self.assertEqual(len(encode_node(matrices, contexts, inter, packing="per_matrix").packets), 4)
        with self.assertRaisesRegex(ConfigurationError, "packing='per_matrix'"):
            encode_node(matrices, contexts, inter)
        other = dict(contexts)
        key = MatrixKey(1, "K")
        other[key] = TensorContext(contexts[key].model, key, 270, 3, 6, 10, "pre_rope")
        for packing in ("per_matrix", "node"):
            with self.assertRaisesRegex(ConfigurationError, "two nodes"):
                encode_node(matrices, other, specs, packing=packing)
        legacy = {name: CodecSpec.from_dict({k: v for k, v in spec.to_dict().items() if k != "inter_prediction"})
                  for name, spec in specs.items()}
        with self.assertRaisesRegex(ConfigurationError, "decode-only"):
            encode_node(matrices, contexts, legacy, packing="node")
        with self.assertRaises(ConfigurationError):
            encode_node(matrices, contexts, specs, packing="stripes")
        with self.assertRaises(ConfigurationError):
            encode_node(matrices, contexts, specs, packing="layer_group")
        with self.assertRaises(ConfigurationError):
            encode_packed(matrices, contexts, specs, canvas=(48, 48))

    def test_spatial_packing_is_explicit_and_not_labelled_identical(self):
        matrices, contexts, specs = node(seed=4)
        packet = encode_node(matrices, contexts, specs, packing="node", layout="spatial").packets[0]
        packing = packet.metadata["packing"]
        self.assertEqual(packing["layout"], "spatial")
        self.assertFalse(packing["pixel_identical_to_same_canvas_matrix"])
        self.assertEqual(packing["random_access"], "whole_stream")
        self.assertEqual(packet.metadata["canvas"]["pictures"], 3)
        self.assertTrue(all(tile["byte_range"] is None for tile in packet.metadata["tiles"]))
        rects = [tuple(tile["rect"]) for tile in packet.metadata["tiles"]]
        self.assertEqual(len(set((x, y) for x, y, _, _ in rects)), len(rects))
        decoded = decode_packed(packet, verify_packet=True)
        for key, value in decoded.items():
            self.assertLess(((value - matrices[key]).norm() / matrices[key].norm()).item(), 0.05)

    def test_tile_map_tampering_is_rejected(self):
        matrices, contexts, specs = node(seed=6)
        packet = encode_node(matrices, contexts, specs, packing="node", validate_hevc=True).packets[0]
        spatial = encode_node(matrices, contexts, specs, packing="node", layout="spatial").packets[0]
        decode_packed(packet, verify_packet=True)

        def tampered(base, change):
            metadata = copy.deepcopy(base.metadata)
            change(metadata)
            return EncodedTensor(metadata, base.payloads, None)

        cases = {
            "rect outside canvas": lambda m: m["tiles"][0].update(rect=[0, 0, m["canvas"]["width"] + 32, 30]),
            "misaligned canvas": lambda m: m["canvas"].update(width=m["canvas"]["width"] + 8),
            "misaligned origin": lambda m: m["tiles"][1].update(rect=[8, 0, *m["tiles"][1]["rect"][2:]]),
            "picture overlap": lambda m: m["tiles"][1].update(pictures=[0, 3]),
            "picture count": lambda m: m["tiles"][0].update(pictures=[0, 2]),
            "inter codec": lambda m: m["codec"].update(inter_prediction=True),
            "inter tile": lambda m: m["tiles"][0]["codec"].update(inter_prediction=True),
            "other node": lambda m: m["tiles"][2]["context"].update(token_start=0),
            "other node key": lambda m: m["tiles"][2].update(key={"layer": 1, "target": "V"}),
            "order": lambda m: m["tiles"].reverse(),
            "duplicate": lambda m: m["tiles"].__setitem__(1, copy.deepcopy(m["tiles"][0])),
            "byte range": lambda m: m["tiles"][0].update(byte_range=[0, m["tiles"][0]["byte_range"][1] - 1]),
            "padding": lambda m: m["padding"].update(canvas_padding_bytes=0),
            "encoder": lambda m: m.update(hevc=_hevc.encoder_record(True)),
            "schema": lambda m: m.update(schema="reelcache.tensor.v2"),
            "normalization": lambda m: m["tiles"][0]["normalization"].update(mode="global"),
            "missing field": lambda m: m["tiles"][0].pop("byte_range"),
        }
        for label, change in cases.items():
            with self.subTest(case=label), self.assertRaises((IntegrityError, ConfigurationError)):
                decode_packed(tampered(packet, change), verify_packet=True)
        # Spatial packets exercise each rectangle rule on its own, with the tile's true mosaic size.
        canvas_width = spatial.metadata["canvas"]["width"]
        alignment = spatial.metadata["canvas"]["alignment"]
        rects = [tile["rect"] for tile in spatial.metadata["tiles"]]
        self.assertTrue(all(rect[1] == 0 and rect[0] % alignment == 0 for rect in rects))
        last = len(rects) - 1
        past_edge = tampered(spatial, lambda m: m["tiles"][last].update(rect=[canvas_width, 0, *rects[last][2:]]))
        self.assertEqual(canvas_width % alignment, 0)
        with self.assertRaisesRegex(IntegrityError, "extends past the canvas"):
            decode_packed(past_edge)
        shifted = rects[1][0] + 8
        self.assertLessEqual(shifted + rects[1][2], rects[2][0])
        misaligned = tampered(spatial, lambda m: m["tiles"][1].update(rect=[shifted, 0, *rects[1][2:]]))
        with self.assertRaisesRegex(IntegrityError, "origin is not CTU-aligned"):
            decode_packed(misaligned)
        resized = tampered(spatial, lambda m: m["tiles"][1].update(rect=[*rects[1][:2], rects[1][2] + 1, rects[1][3]]))
        with self.assertRaisesRegex(IntegrityError, "size differs from its channel-tile mosaic"):
            decode_packed(resized)
        overlap = tampered(spatial, lambda m: m["tiles"][1].update(
            rect=[*m["tiles"][0]["rect"][:2], *m["tiles"][1]["rect"][2:]],
        ))
        with self.assertRaisesRegex(IntegrityError, "overlap"):
            decode_packed(overlap)
        stream = bytearray(packet.payloads["stream.hevc"])
        stream[len(stream) // 2] ^= 0xFF
        with self.assertRaises(IntegrityError):
            decode_packed(EncodedTensor(packet.metadata, {**packet.payloads, "stream.hevc": bytes(stream)}, None),
                          verify_packet=True)
        with self.assertRaises(IntegrityError):
            decode_packed(EncodedTensor(packet.metadata, {**packet.payloads, "extra": b"x"}, None))
        with self.assertRaises(ConfigurationError):
            decode_tensor(packet)

    def test_single_tile_packets_decode_through_decode_tensor_and_provider(self):
        matrices, contexts, specs = node(layers=1, seed=8)
        key = MatrixKey(0, "V")
        single = encode_packed({key: matrices[key]}, {key: contexts[key]}, {key: specs[key]}, packing="explicit")
        decoded = decode_tensor(single, verify_packet=True)
        self.assertTrue(torch.equal(decoded, decode_packed(single)[key]))
        torch.testing.assert_close(decode_tensor(single, columns=slice(2, 5)), decoded[:, 2:5])
        provider = HEVCProvider(packing="layer_group", layer_group_size=1, x265_threads=1)
        encoding = provider.encode_node(matrices, contexts, specs)
        self.assertEqual(encoding.packing, "layer_group")
        self.assertEqual(set(provider.decode_packed(encoding.packets[0])), set(matrices))
        for kwargs in ({"packing": "bad"}, {"packing": "layer_group"}, {"layer_group_size": 2},
                       {"packed_layout": "diagonal"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigurationError):
                HEVCProvider(**kwargs)
        # Packed NVDEC decoding exists (Step 5a); an unavailable NVDEC fails explicitly, never in software.
        nvdec = NVDECDecoder(gpu_id=0, gpu_uuid="GPU-00000000-test")
        unavailable = UnsupportedError("NVDEC is unavailable on this host")
        with mock.patch.object(NVDECDecoder, "_initialize", side_effect=unavailable), \
                mock.patch.object(_hevc, "decode_pictures", side_effect=AssertionError("software fallback")):
            with self.assertRaisesRegex(UnsupportedError, "unavailable"):
                HEVCProvider(nvdec=nvdec).decode_packed(single)
            with self.assertRaisesRegex(UnsupportedError, "unavailable"):
                decode_tensor(single, nvdec=nvdec)


if __name__ == "__main__":
    unittest.main()
