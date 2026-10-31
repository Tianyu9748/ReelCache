"""Step 6: packed node streams (encode side), canonical canvases, device quantization, external bases."""

from __future__ import annotations

import hashlib
import math
import unittest

import numpy as np
import torch

from reelcache import _hevc
from reelcache.codecs import _json_bytes, decode_packed, encode_packed, factor_binding
from reelcache.errors import ConfigurationError, IntegrityError
from reelcache.factorization import Factors, project_onto_basis
from reelcache import node_packets as npk
from reelcache.packed_decode import prepare_packet
from reelcache.types import CodecSpec, MatrixKey, ModelIdentity, TensorContext

MODEL = ModelIdentity("step6-pack", "step6-pack-source", 2, 64, "tokens_channels", "bfloat16")
FRAMES, HEIGHT, WIDTH = 3, 6, 8
TOKENS = FRAMES * HEIGHT * WIDTH
RANKS = {MatrixKey(0, "K"): 5, MatrixKey(0, "V"): 17, MatrixKey(1, "K"): 9, MatrixKey(1, "V"): 40}


def context(key, token_start=0):
    return TensorContext(MODEL, key, token_start, FRAMES, HEIGHT, WIDTH, "pre_rope" if key.target == "K" else "none")


def matrices(seed=0):
    generator = torch.Generator().manual_seed(seed)
    result = {}
    for key in RANKS:
        value = torch.randn(TOKENS, 12, generator=generator) @ torch.randn(12, 64, generator=generator)
        value = value + 0.05 * torch.randn(TOKENS, 64, generator=generator)
        if key.target == "K":
            value = torch.nn.functional.layer_norm(value, (64,))
        result[key] = value.to(torch.bfloat16)
    return result


def basis(rank, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(64, rank, generator=generator)).Q.T.contiguous().to(torch.float16)


class CanonicalCanvasTests(unittest.TestCase):
    def test_ladder_is_monotone_covering_and_cached(self):
        ladder = npk.canvas_ladder(3, 30, 52, "yuv444p", max_rank=1536)
        self.assertEqual(ladder, npk.canvas_ladder(3, 30, 52, "yuv444p", max_rank=1536))
        widths = [row[1] for row in ladder]
        heights = [row[2] for row in ladder]
        self.assertEqual(widths, sorted(widths))
        self.assertEqual(heights, sorted(heights))
        for rank in (1, 31, 32, 33, 200, 262, 288, 500, 1536):
            canvas = npk.canonical_canvas(3, 30, 52, "yuv444p", rank, channels=1536)
            layout = _hevc.video_layout(3, 30, 52, rank, "yuv444p")
            self.assertGreaterEqual(canvas["width"], layout["image_width"])
            self.assertGreaterEqual(canvas["height"], layout["image_height"])
            self.assertGreaterEqual(min(canvas["width"], canvas["height"]), _hevc.MIN_CANVAS)
            self.assertEqual(canvas["width"] % _hevc.CANVAS_ALIGNMENT, 0)
            self.assertGreaterEqual(canvas["rank_step"], rank)
        # Canonical sizes are few: every CausVid K98/V95-like node rank in 257..320 shares one canvas.
        sizes = {(item["width"], item["height"]) for item in
                 (npk.canonical_canvas(3, 30, 52, "yuv444p", rank, channels=1536) for rank in range(257, 321))}
        self.assertEqual(len(sizes), 1)
        own = npk.canonical_canvas(3, 30, 52, "yuv444p", 262, channels=1536, ladder=None)
        self.assertEqual(own["policy"], "own_size")
        with self.assertRaises(ConfigurationError):
            npk.canonical_canvas(3, 30, 52, "yuv444p", 2000, channels=1536)


class DeviceQuantizationTests(unittest.TestCase):
    def test_quantization_matches_hevc_reference_bitwise(self):
        generator = torch.Generator().manual_seed(3)
        for normalization, fraction in (("per_channel", 1.0), ("global", 1.0), ("leading_columns", 0.25)):
            spec = CodecSpec(kind="svd_hevc", rank=17, normalization=normalization, protected_fraction=fraction,
                             inter_prediction=False)
            values = torch.randn(TOKENS, 17, generator=generator) * torch.logspace(1, -3, 17)
            values[:, 3] = 0.25  # constant column: zero-span affine
            pixels, affine, layout = npk.quantize_device(values, spec)
            reference = _hevc.quantize_coefficients(values.numpy(), normalization, fraction)
            self.assertTrue(np.array_equal(pixels.numpy(), reference[0]))
            self.assertEqual(affine.tobytes(), reference[1].astype("<f4").tobytes())
            self.assertEqual(layout, reference[2])

    def test_mosaic_matches_reference_layout(self):
        generator = torch.Generator().manual_seed(4)
        for rank, pix_fmt in ((5, "yuv444p"), (40, "yuv444p"), (7, "yuv420p")):
            video = _hevc.video_layout(FRAMES, HEIGHT, WIDTH, rank, pix_fmt)
            pixels = torch.randint(0, 256, (TOKENS, rank), generator=generator, dtype=torch.uint8)
            mosaic = npk.mosaic_device(pixels, video).numpy()
            values = pixels.numpy().reshape(FRAMES, HEIGHT, WIDTH, rank)
            for segment in range(video["segments"]):
                expected = _hevc._mosaic(values, video, segment)
                block = mosaic[segment * FRAMES:(segment + 1) * FRAMES]
                self.assertTrue(np.array_equal(block, expected[:, :, :block.shape[2], :block.shape[3]]))
                self.assertFalse(expected[:, :, block.shape[2]:, :].any())


class NodePackEncodeTests(unittest.TestCase):
    def setUp(self):
        self.matrices = matrices()
        self.contexts = {key: context(key) for key in RANKS}
        self.bases = {key: basis(rank, seed) for seed, (key, rank) in enumerate(RANKS.items())}

    def finalized(self, ladder=npk.CANVAS_RANK_LADDER, keep=False):
        tiles = []
        for key, rank in RANKS.items():
            spec = CodecSpec(kind="svd_hevc", rank=rank, inter_prediction=False)
            tiles.append(npk.finalize_tile(self.matrices[key], self.bases[key], self.contexts[key], spec, entry=key,
                                           target=0.0, diagnostic=True, tile_rows=1024, keep_coefficients=keep))
        node = npk.assemble_canvas("n0", tiles, codec=npk.pack_codec(CodecSpec(kind="svd_hevc", rank=1,
                                                                              inter_prediction=False)),
                                   ladder=ladder, channels=64)
        node.to_host()
        return node

    def test_pack_is_byte_identical_to_codecs_encode_packed_for_any_thread_count(self):
        prepared, specs = {}, {}
        for key, rank in RANKS.items():
            spec = CodecSpec(kind="svd_hevc", rank=rank, inter_prediction=False)
            projection = project_onto_basis(self.matrices[key], self.bases[key], rank=rank, tile_rows=1024)
            factors = Factors(self.bases[key], projection.coefficients, rank, metrics={
                "projection_relative_residual": projection.relative_residual,
                "basis_condition_bound": projection.condition_bound, "zero_input": projection.zero_input,
                "tile_rows": 1024})
            prepared[key] = (factors, factor_binding(self.matrices[key], self.contexts[key], spec, factors))
            specs[key] = spec
        node = self.finalized()
        reference = encode_packed(self.matrices, self.contexts, specs, prepared=prepared, node_id="n0",
                                  canvas=(node.canvas["width"], node.canvas["height"]), hevc_threads=1)
        for threads in (1, 4):
            with self.subTest(threads=threads):
                packet = npk.encode_node_pack(self.finalized(), threads=threads, validate_decode=True)
                self.assertEqual(_json_bytes(packet.metadata), _json_bytes(reference.metadata))
                self.assertEqual(set(packet.payloads), set(reference.payloads))
                for name, value in reference.payloads.items():
                    self.assertEqual(packet.payloads[name], value, name)
        # Without the validation decode only the decoded digests are absent.
        unvalidated = npk.encode_node_pack(self.finalized(), threads=2, validate_decode=False)
        self.assertEqual(unvalidated.payloads["stream.hevc"], reference.payloads["stream.hevc"])
        self.assertTrue(all(tile["decoded_coefficients_sha256"] is None for tile in unvalidated.metadata["tiles"]))

    def test_canonical_canvas_costs_only_padding_and_decodes_identically(self):
        own = npk.encode_node_pack(self.finalized(ladder=None), threads=2)
        canonical = npk.encode_node_pack(self.finalized(), threads=2)
        self.assertLessEqual(own.metadata["canvas"]["width"] * own.metadata["canvas"]["height"],
                             canonical.metadata["canvas"]["width"] * canonical.metadata["canvas"]["height"])
        left = decode_packed(own, dtype=torch.float32)
        right = decode_packed(canonical, dtype=torch.float32)
        for key in RANKS:
            self.assertTrue(torch.equal(left[key], right[key]))
            self.assertEqual([t["decoded_coefficients_sha256"] for t in own.metadata["tiles"]],
                             [t["decoded_coefficients_sha256"] for t in canonical.metadata["tiles"]])

    def test_post_codec_error_formula_matches_direct_reconstruction(self):
        packet = npk.encode_node_pack(self.finalized(keep=True), threads=2, validate_decode=True)
        decoded = decode_packed(packet, dtype=torch.float64)
        for key in RANKS:
            source = self.matrices[key].double()
            direct = float((source - decoded[key]).norm() / source.norm())
            label = f"{key.layer}:{key.target}"
            # The formula excludes only the final native-dtype cast of decode_packed (here FP64 output).
            self.assertAlmostEqual(packet.metrics["post_codec_relative_error"][label], direct, delta=1e-5)

    def test_external_bases_detach_attach_and_reject_substitution(self):
        packet = npk.encode_node_pack(self.finalized(), threads=1)
        metadata, payloads, bases = npk.detach_pack_bases(packet)
        self.assertFalse(any(name.startswith("basis.") for name in payloads))
        restored = npk.attach_pack_bases(metadata, payloads, bases)
        prepared = prepare_packet(restored, verify=True)
        self.assertEqual(len(prepared.tiles), len(RANKS))
        forged = list(bases)
        forged[1] = bytes(len(forged[1]))
        with self.assertRaises(IntegrityError):
            npk.attach_pack_bases(metadata, payloads, forged)
        with self.assertRaises(IntegrityError):
            npk.attach_pack_bases(metadata, payloads, bases[:-1])
        index = npk.pack_tile_index(metadata)
        self.assertEqual(sorted(index.values()), list(range(len(RANKS))))

    def test_finalize_refuses_a_representation_below_target(self):
        key = MatrixKey(0, "V")
        spec = CodecSpec(kind="svd_hevc", rank=1, inter_prediction=False)
        with self.assertRaises(IntegrityError):
            npk.finalize_tile(self.matrices[key], self.bases[key][:1], self.contexts[key], spec, entry=key,
                              target=0.999, diagnostic=False, tile_rows=1024)


if __name__ == "__main__":
    unittest.main()
