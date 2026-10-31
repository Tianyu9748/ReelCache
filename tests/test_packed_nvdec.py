"""Step 5a: packed NVDEC restore logic, tested on CPU with a software stand-in for the decoder pool.

Real NVDEC behavior (bitwise identity, A100/RTX profiles, pool scaling and event safety under
load) is validated by the GPU harness in
artifacts/sglang_integration/benchmarks/packed_nvdec/scripts/gpu_validate.py; restores inside
torch.inference_mode, the submit write-ordering race and its rejection by
artifacts/sglang_integration/benchmarks/restore_e2e/scripts/inference_mode_gpu.py.
"""

import base64
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

import numpy as np
import torch

import reelcache
from reelcache import _hevc
from reelcache import decoder_pool as dp
from reelcache import restore_kernels as rk
from reelcache.codecs import (
    _unpack_tile, decode_packed, decode_packed_pixels, encode_node, encode_packed, encode_tensor, packed_keys,
)
from reelcache.decoder_pool import (
    COLD_SESSION_BYTES, CanvasArena, DecodeTicket, DecoderPool, check_ordering, cold_session_bytes, synthetic_unit,
)
from reelcache.errors import CapacityError, ConfigurationError, IntegrityError, StateError, UnsupportedError
from reelcache.nvdec import (
    CanvasSession, NVDECDecoder, check_canvas, check_units, normal_tensors, unpack_device_tile, write_mode,
)
from reelcache.packed_decode import NodeRef, PackedRestorer, PreparedPacket, canonical_canvas_shapes, prepare_packet
from reelcache.prefetch import PackedLayerPrefetcher
from reelcache.types import CodecSpec, EncodedTensor, MatrixKey, ModelIdentity, TensorContext

FIXTURE = Path(__file__).resolve().parent / "fixtures/legacy_v1_packets.json"
UUID = "GPU-bf621a31-0ece-332e-d43d-d20e0d4a42be"
HEADS = {"K": (2, 6), "V": (2, 8)}


def node(layers=3, *, ranks=(5, 7), pix_fmt="yuv444p", seed=0, token_start=180):
    model = ModelIdentity("packed-nvdec", "fixture-5a", layers, {"K": 12, "V": 16}, "tokens_channels", "float32")
    generator = torch.Generator().manual_seed(seed)
    matrices, contexts, specs = {}, {}, {}
    frames, height, width = 3, 6, 10
    for layer in range(layers):
        for target, rank in zip(("K", "V"), ranks):
            key = MatrixKey(layer, target)
            channels = model.channels_for(target)
            tokens = frames * height * width
            ramp = torch.linspace(-1, 1, tokens)[:, None]
            matrices[key] = (ramp * torch.randn(1, channels, generator=generator)
                             + torch.randn(tokens, 4, generator=generator) @ torch.randn(4, channels,
                                                                                          generator=generator))
            contexts[key] = TensorContext(model, key, token_start, frames, height, width,
                                          "pre_rope" if target == "K" else "none")
            specs[key] = CodecSpec("svd_hevc", rank + layer, pix_fmt=pix_fmt)
    return matrices, contexts, specs


class FakeEvent:
    def __init__(self, done=True):
        self.done = done

    def query(self):
        return self.done


class SoftwarePool(DecoderPool):
    """CPU stand-in honouring DecoderPool.submit's contract (FFmpeg decode into CPU canvases)."""

    def __init__(self):  # noqa: D107 - deliberately bypasses the CUDA session setup
        self.device = torch.device("cpu")
        self.jobs = []
        self.warmed = []

    def prewarm(self, samples, timeout=None):
        self.warmed.extend(samples)
        return 0.0

    def submit(self, units, canvas, first=0, *, pix_fmt="yuv444p", worker=None, after="current"):
        units, pictures, nbytes = check_units(units)
        if pictures != len(units):
            raise ConfigurationError("one access unit per unit")
        height, width = check_canvas(canvas, device=self.device, pix_fmt=pix_fmt, pictures=first + pictures)
        if check_ordering(canvas, after) not in ("current", None):  # the pool's contract; events are not modelled
            raise ConfigurationError("unexpected ordering events")
        stream = b"".join(data[offset:offset + length] for data, offset, length, _ in units)
        images, _ = _hevc.decode_pictures(stream, pictures=pictures, height=height, width=width, pix_fmt=pix_fmt)
        canvas[first:first + pictures].copy_(torch.from_numpy(np.array(images)))
        ticket = DecodeTicket(nbytes, pictures)
        ticket.slot = 0
        ticket._finish(event=None)
        self.jobs.append({"units": units, "canvas": tuple(canvas.shape)})
        return ticket

    def stats(self):
        return {"jobs": len(self.jobs)}

    def close(self):
        pass


def legacy_packets():
    rows = json.loads(FIXTURE.read_text())["historical_store_packets"]
    return [EncodedTensor(row["metadata"], {name: base64.b64decode(raw) for name, raw in row["payloads"].items()},
                          None if row["basis"] is None else base64.b64decode(row["basis"])) for row in rows]


def software_factors(packet):
    """{key: (FP16 basis, FFmpeg-decoded FP32 coefficients, uint8 pixels)} of a v2 packet."""
    from reelcache.codecs import _from_bytes, _is_pack

    result = {}
    if _is_pack(packet):
        pixels = decode_packed_pixels(packet)
        for key, row in zip(packed_keys(packet), packet.metadata["tiles"]):
            shape = packet.metadata["payloads"][row["basis"]["payload"]]["shape"]
            affine = np.frombuffer(packet.payloads[row["affine"]], dtype="<f4").reshape(
                row["normalization"]["affine_shape"])
            values = _hevc.dequantize_coefficients(pixels[key], affine, row["normalization"])
            result[key] = (_from_bytes(packet.payloads[row["basis"]["payload"]], "float16", shape),
                           torch.from_numpy(values), pixels[key])
        return result
    metadata = packet.metadata
    video = metadata["video"]
    streams = [packet.payloads[f"coefficients.{index:03d}.hevc"] for index in range(video["segments"])]
    pixels = _hevc.decode_pixels(streams, video)[0]
    affine = np.frombuffer(packet.payloads["affine.f32"], dtype="<f4").reshape(
        metadata["normalization"]["affine_shape"])
    values = _hevc.dequantize_coefficients(pixels, affine, metadata["normalization"])
    key = MatrixKey.from_dict(metadata["context"]["key"])
    result[key] = (_from_bytes(packet.basis, "float16", metadata["basis"]["shape"]), torch.from_numpy(values),
                   pixels)
    return result


class PictureUnitTests(unittest.TestCase):
    def test_fast_scan_equals_full_access_unit_parse(self):
        matrices, contexts, specs = node()
        for pix_fmt in ("yuv444p", "yuv420p"):
            specs = {key: CodecSpec("svd_hevc", spec.rank, pix_fmt=pix_fmt) for key, spec in specs.items()}
            packet = encode_node(matrices, contexts, specs, packing="node").packets[0]
            stream = packet.payloads["stream.hevc"]
            pictures = packet.metadata["canvas"]["pictures"]
            full = [(unit["offset"], unit["length"]) for unit in _hevc.all_intra_access_units(stream, pictures)]
            self.assertEqual(_hevc.picture_units(stream, 0, len(stream), pictures), full)
            for tile in packet.metadata["tiles"]:
                first, count = tile["pictures"]
                self.assertEqual(_hevc.picture_units(stream, *tile["byte_range"], count), full[first:first + count])
            with self.assertRaises(IntegrityError):
                _hevc.picture_units(stream, 0, len(stream), pictures + 1)
            with self.assertRaises(IntegrityError):
                _hevc.picture_units(stream, 0, len(stream), pictures - 1)
            with self.assertRaises(IntegrityError):
                _hevc.picture_units(stream, full[1][0] + 2, full[1][1] - 2, 1)
            with self.assertRaises(IntegrityError):
                _hevc.picture_units(stream, 0, len(stream) + 1, pictures)

    def test_packed_canvases_respect_the_nvdec_minimum(self):
        self.assertEqual(_hevc.MIN_CANVAS, 160)
        matrices, contexts, specs = node()
        for layout in ("temporal", "spatial"):
            packet = encode_node(matrices, contexts, specs, packing="node", layout=layout).packets[0]
            canvas = packet.metadata["canvas"]
            self.assertGreaterEqual(min(canvas["width"], canvas["height"]), _hevc.NVDEC_MIN_DIMENSION)
            self.assertEqual((canvas["width"] % 32, canvas["height"] % 32), (0, 0))
        with self.assertRaisesRegex(ConfigurationError, "NVDEC minimum"):
            encode_packed(matrices, contexts, specs, canvas=(128, 160))
        key = MatrixKey(0, "K")
        single = encode_packed({key: matrices[key]}, {key: contexts[key]}, {key: specs[key]}, packing="explicit")
        self.assertEqual((single.metadata["canvas"]["width"], single.metadata["canvas"]["height"]), (160, 160))


class DeviceTileTests(unittest.TestCase):
    def test_tile_extraction_matches_cpu_cells_for_segments_and_formats(self):
        rng = np.random.default_rng(3)
        with mock.patch.object(_hevc, "MAX_DIMENSION", 64):
            for pix_fmt in ("yuv444p", "yuv420p"):
                for rank in (1, 7, 40, 97):
                    video = _hevc.video_layout(3, 6, 10, rank, pix_fmt)
                    rect = [32, 64, video["image_width"], video["image_height"]]
                    pictures = 2 + video["frames"] * video["segments"]
                    canvas = rng.integers(0, 256, (pictures, video["planes"], 64 + video["image_height"],
                                                   32 + video["image_width"]), dtype=np.uint8)
                    expected = _unpack_tile(canvas, {"rect": rect}, 2, video)
                    actual = unpack_device_tile(torch.from_numpy(canvas), 2, rect, video)
                    self.assertTrue(np.array_equal(actual.numpy(), expected), (pix_fmt, rank, video["segments"]))
                    with self.assertRaises(IntegrityError):
                        unpack_device_tile(torch.from_numpy(canvas[:-1]), 2, rect, video)


class PrepareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrices, cls.contexts, cls.specs = node()

    def test_temporal_pack_units_tiles_and_bases(self):
        packet = encode_node(self.matrices, self.contexts, self.specs, packing="node", node_id="n").packets[0]
        prepared = prepare_packet(packet, verify=True)
        self.assertIsInstance(prepared, PreparedPacket)
        self.assertTrue(prepared.random_access)
        self.assertEqual(prepared.node["node_id"], "n")
        stream = packet.payloads["stream.hevc"]
        full = _hevc.all_intra_access_units(stream, prepared.pictures)
        self.assertEqual([unit[1:] for unit in prepared.stream_units],
                         [(unit["offset"], unit["length"], 1) for unit in full])
        self.assertEqual([tile.key for tile in prepared.tiles], packed_keys(packet))
        for tile, row in zip(prepared.tiles, packet.metadata["tiles"]):
            self.assertEqual(list(tile.pictures), row["pictures"])
            self.assertEqual(list(tile.rect), row["rect"])
            self.assertEqual(len(tile.units), row["pictures"][1])
            self.assertEqual(sum(unit[2] for unit in tile.units), row["byte_range"][1])
            self.assertEqual(tile.basis.dtype, torch.float16)
            self.assertEqual(tile.rank, row["codec"]["rank"])
            self.assertEqual(tile.scale.dtype, np.float32)
        self.assertIs(prepared.tile(MatrixKey(1, "V")), prepared.tiles[3])
        with self.assertRaises(ConfigurationError):
            prepared.tile(MatrixKey(9, "V"))

    def test_spatial_pack_and_single_packets(self):
        spatial = encode_node(self.matrices, self.contexts, self.specs, packing="node", layout="spatial").packets[0]
        prepared = prepare_packet(spatial)
        self.assertFalse(prepared.random_access)
        self.assertEqual(len(prepared.stream_units), prepared.pictures)
        self.assertTrue(all(tile.units == () for tile in prepared.tiles))
        key = MatrixKey(2, "V")
        single = encode_tensor(self.matrices[key], self.contexts[key], self.specs[key])
        prepared = prepare_packet(single, key=(key, "cond"))
        (tile,) = prepared.tiles
        self.assertEqual(tile.key, (key, "cond"))
        self.assertEqual(len(tile.units), tile.layout["frames"] * tile.layout["segments"])
        self.assertEqual((prepared.width, prepared.height), (tile.layout["image_width"], tile.layout["image_height"]))
        with self.assertRaises(ConfigurationError):
            prepare_packet(single, key=(MatrixKey(0, "V"), "cond"))
        with self.assertRaises(ConfigurationError):
            prepare_packet(spatial, key=(key, "cond"))

    def test_inter_legacy_and_non_hevc_packets_are_rejected_explicitly(self):
        key = MatrixKey(0, "V")
        inter = encode_tensor(self.matrices[key], self.contexts[key], CodecSpec("svd_hevc", 5, inter_prediction=True))
        with self.assertRaisesRegex(UnsupportedError, "all-intra"):
            prepare_packet(inter)
        for legacy in legacy_packets():
            with self.assertRaisesRegex(UnsupportedError, "NVDECDecoder"):
                prepare_packet(legacy)
        raw = encode_tensor(self.matrices[key], self.contexts[key], CodecSpec("raw"))
        with self.assertRaises(UnsupportedError):
            prepare_packet(raw)
        with self.assertRaises(ConfigurationError):
            prepare_packet(b"not a packet")


class ArenaTicketTests(unittest.TestCase):
    def test_canvas_reuse_waits_for_every_consumer_event(self):
        arena = CanvasArena("cpu", 2 * 1000, wait_seconds=0.05, poll_seconds=0.001)
        first = arena.acquire((1, 1, 10, 100))
        second = arena.acquire((1, 1, 10, 100))
        pending = [FakeEvent(True), FakeEvent(False)]
        arena.release(first, after=pending)
        with self.assertRaisesRegex(CapacityError, "exhausted"):
            arena.acquire((1, 1, 10, 100))
        pending[1].done = True
        self.assertIs(arena.acquire((1, 1, 10, 100)), first)
        arena.release(second, after=FakeEvent(True))
        other = arena.acquire((1, 1, 20, 50))  # evicts the free canvas of another shape
        self.assertEqual(tuple(other.shape), (1, 1, 20, 50))
        report = arena.report()
        self.assertEqual((report["reuses"], report["evictions"]), (1, 1))
        self.assertLessEqual(report["peak_bytes"], 2000)
        with self.assertRaises(CapacityError):
            arena.acquire((1, 1, 100, 100))
        with self.assertRaises(ConfigurationError):
            arena.release(other, after=[object()])
        with self.assertRaises(ConfigurationError):
            CanvasArena("cpu", 0)

    def test_arena_waits_for_retiring_canvases(self):
        arena = CanvasArena("cpu", 1000, wait_seconds=5, poll_seconds=0.001)
        canvas = arena.acquire((1, 1, 10, 100))
        event = FakeEvent(False)
        arena.release(canvas, after=event)
        timer = threading.Timer(0.05, lambda: setattr(event, "done", True))
        timer.start()
        self.assertIs(arena.acquire((1, 1, 10, 100)), canvas)
        timer.join()
        self.assertEqual(arena.report()["waits"], 1)

    def test_tickets_deliver_events_and_errors(self):
        ticket = DecodeTicket(10, 1)
        self.assertFalse(ticket.done())
        with self.assertRaises(TimeoutError):
            ticket.event(timeout=0.01)
        marker = FakeEvent()
        ticket._finish(event=marker)
        self.assertIs(ticket.event(), marker)
        failed = DecodeTicket(10, 1)
        failed._finish(error=IntegrityError("bad stream"))
        with self.assertRaisesRegex(IntegrityError, "bad stream"):
            failed.event()


class HardwareGuardTests(unittest.TestCase):
    def test_pool_and_sessions_never_fall_back_without_nvdec(self):
        with self.assertRaises(ConfigurationError):
            DecoderPool(gpu_id=0, gpu_uuid="not-a-uuid", workers=1)
        with self.assertRaises(ConfigurationError):
            DecoderPool(gpu_id=0, gpu_uuid=UUID, workers=0)
        with self.assertRaises(ConfigurationError):
            DecoderPool(gpu_id=0, gpu_uuid=UUID, workers=2, max_sessions_per_worker=0)
        with self.assertRaises(ConfigurationError):
            CanvasSession(gpu_id=0, width=8, height=256, pix_fmt="yuv444p", api=object())
        with self.assertRaises(UnsupportedError):
            CanvasSession(gpu_id=0, width=256, height=256, pix_fmt="yuv422p", api=object())
        with mock.patch("torch.cuda.device_count", return_value=0):
            with self.assertRaisesRegex(UnsupportedError, "not visible"):
                DecoderPool(gpu_id=0, gpu_uuid=UUID, workers=1)
            with self.assertRaisesRegex(UnsupportedError, "not visible"):
                CanvasSession(gpu_id=0, width=256, height=256, pix_fmt="yuv444p", api=object())

    def test_units_and_canvases_are_validated(self):
        data = b"\x00\x00\x01" * 10
        self.assertEqual(check_units([(data, 0, 3, 1)])[1:], (1, 3))
        for bad in ([], [(data, 0, 3)], [(bytearray(data), 0, 3, 1)], [(data, 0, 0, 1)], [(data, 0, 3, 0)]):
            with self.assertRaises((ConfigurationError, IntegrityError)):
                check_units(bad)
        with self.assertRaises(IntegrityError):
            check_units([(data, 28, 3, 1)])
        canvas = torch.empty((2, 3, 16, 16), dtype=torch.uint8)
        self.assertEqual(check_canvas(canvas, device=torch.device("cpu"), pix_fmt="yuv444p", pictures=2), (16, 16))
        for bad, fmt, pictures in ((canvas, "yuv420p", 2), (canvas, "yuv444p", 3), (canvas.float(), "yuv444p", 1),
                                   (canvas.transpose(2, 3), "yuv444p", 1)):
            with self.assertRaises((ConfigurationError, UnsupportedError)):
                check_canvas(bad, device=torch.device("cpu"), pix_fmt=fmt, pictures=pictures)


class CodecsNVDECWiringTests(unittest.TestCase):
    def test_decode_packed_with_nvdec_uses_access_units_and_matches_software(self):
        matrices, contexts, specs = node(seed=2)
        packet = encode_node(matrices, contexts, specs, packing="node").packets[0]
        canvas = packet.metadata["canvas"]
        calls = []

        def fake_canvas(self, units, *, width, height, pix_fmt):
            calls.append([unit[1:] for unit in units])
            self.assertEqual((width, height, pix_fmt), (canvas["width"], canvas["height"], canvas["pix_fmt"]))
            stream = b"".join(data[offset:offset + length] for data, offset, length, _ in units)
            images, _ = _hevc.decode_pictures(stream, pictures=len(units), height=height, width=width,
                                              pix_fmt=pix_fmt)
            return torch.from_numpy(np.array(images)) if self.device_output else images

        keys = [MatrixKey(1, "K"), MatrixKey(2, "V")]
        expected = decode_packed(packet, keys=keys)
        stream = packet.payloads["stream.hevc"]
        for device_output in (True, False):
            decoder = NVDECDecoder(gpu_id=0, gpu_uuid=UUID, device_output=device_output)
            decoder.assertEqual = self.assertEqual
            with mock.patch.object(NVDECDecoder, "decode_canvas", fake_canvas):
                actual = decode_packed(packet, keys=keys, nvdec=decoder)
            self.assertEqual(list(actual), keys)
            for key in keys:
                self.assertTrue(torch.equal(actual[key], expected[key]), key)
            tiles = [row for row in packet.metadata["tiles"] if MatrixKey.from_dict(row["key"]) in keys]
            expected_units = [(start, size, 1) for row in tiles
                              for start, size in _hevc.picture_units(stream, *row["byte_range"], row["pictures"][1])]
            self.assertEqual(calls[-1], expected_units)

    def test_decode_canvas_admission_precedes_hardware(self):
        decoder = NVDECDecoder(gpu_id=0, gpu_uuid=UUID, max_bitstream_bytes=8, max_output_bytes=1000)
        data = b"\x00" * 64
        with mock.patch.object(NVDECDecoder, "_initialize", lambda self: None):
            with self.assertRaisesRegex(CapacityError, "compressed input"):
                decoder.decode_canvas([(data, 0, 16, 1)], width=160, height=160, pix_fmt="yuv444p")
            with self.assertRaisesRegex(CapacityError, "canvas output"):
                decoder.decode_canvas([(data, 0, 8, 1)], width=160, height=160, pix_fmt="yuv444p")
            with self.assertRaises(UnsupportedError):
                decoder.decode_canvas([(data, 0, 8, 1)], width=16, height=16, pix_fmt="yuv422p")
        self.assertEqual(decoder.stats()["sessions_created"], 0)


class RestorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrices, cls.contexts, cls.specs = node(seed=5)
        cls.node = encode_node(cls.matrices, cls.contexts, cls.specs, packing="node", node_id="a").packets[0]
        cls.groups = encode_node(cls.matrices, cls.contexts, cls.specs, packing="layer_group",
                                 layer_group_size=2, node_id="b").packets
        cls.singles = encode_node(cls.matrices, cls.contexts, cls.specs, packing="per_matrix").packets
        cls.spatial = encode_node(cls.matrices, cls.contexts, cls.specs, packing="node", layout="spatial").packets[0]
        cls.coefficients = {}
        for node_id, packets in (("a", [cls.node]), ("b", cls.groups), ("c", cls.singles), ("d", [cls.spatial])):
            for packet in packets:
                for key, factors in software_factors(packet).items():
                    cls.coefficients[(node_id, key)] = factors

    def restorer(self, granularity="layer", budget=64 * 1024**2):
        return PackedRestorer(SoftwarePool(), arena_bytes=budget, granularity=granularity)

    def nodes(self):
        return [
            NodeRef("a", [prepare_packet(self.node)]),
            NodeRef("b", [prepare_packet(packet) for packet in self.groups]),
            NodeRef("c", lambda: self.singles),
            NodeRef("d", [prepare_packet(self.spatial)]),
        ]

    def test_picture_sources_are_bitwise_tile_map_extractions(self):
        restorer = self.restorer()
        batch = restorer.submit(self.nodes())
        self.assertEqual(len(batch), 4 * len(self.matrices))
        for node_id, key in batch.entries():
            source = batch.source((node_id, key))
            expected = self.coefficients[(node_id, key)][2]
            self.assertTrue(np.array_equal(rk.unpack_pictures(source).numpy(), expected), (node_id, key))
        batch.close()
        batch.close()
        with self.assertRaises(StateError):
            batch.reconstruct(batch.entries(0), None)

    def test_prewarm_covers_each_canvas_geometry_once(self):
        restorer = self.restorer()
        self.assertEqual(restorer.prewarm(self.nodes()), 0.0)
        warmed = restorer.pool.warmed
        geometries = [sample[:3] for sample in warmed]
        self.assertEqual(len(geometries), len(set(geometries)))
        expected = {(p.width, p.height, p.pix_fmt) for node in self.nodes() for p in node.load()}
        self.assertEqual(set(geometries), expected)
        for width, height, pix_fmt, unit in warmed:
            self.assertEqual(unit[3], 1)
            self.assertEqual(unit[0][unit[1] + 3:unit[1] + 5] if unit[0][unit[1] + 2] == 1 else
                             unit[0][unit[1] + 4:unit[1] + 6], b"\x40\x01")
        with self.assertRaises(ConfigurationError):
            restorer.prewarm([object()])

    def test_jobs_follow_layer_order_granularity_and_skip_resident_entries(self):
        loads = []

        def loader():
            loads.append("c")
            return self.singles

        nodes = [NodeRef("a", [prepare_packet(self.node)]), NodeRef("c", loader)]
        for granularity, expected in (("layer", 3 + 6), ("tile", 6 + 6), ("packet", 1 + 6), (2, 2 + 6)):
            batch = self.restorer(granularity).plan(nodes)
            self.assertEqual(len(batch.jobs), expected, granularity)
            self.assertEqual([job.first_layer for job in batch.jobs], sorted(job.first_layer for job in batch.jobs))
            batch.close()
        loads.clear()
        restorer = self.restorer()
        batch = restorer.plan(nodes, resident={"c", ("a", MatrixKey(0, "K")), ("a", MatrixKey(2, "V"))})
        self.assertEqual(loads, [])
        self.assertEqual(batch.skipped, 2)
        self.assertEqual(len(batch), len(self.matrices) - 2)
        self.assertNotIn(("a", MatrixKey(0, "K")), batch.entries())
        units = [unit for job in batch.jobs for unit in job.units]
        tiles = {tile.key: tile for tile in batch.jobs[0].packet.tiles}
        self.assertEqual(len(units), sum(len(tiles[key].units) for _, key in batch.entries()))
        batch.close()
        batch = restorer.plan(nodes, keys=[MatrixKey(1, "K"), MatrixKey(1, "V")])
        self.assertEqual(loads, ["c"])
        self.assertEqual(sorted(batch.layers), [1])
        self.assertEqual(len(batch.entries()), 4)
        batch.close()
        with self.assertRaises(ConfigurationError):
            restorer.plan([NodeRef("a", [prepare_packet(self.node)])] * 2)
        with self.assertRaises(ConfigurationError):
            restorer.plan(nodes, resident=[3])
        with self.assertRaises(ConfigurationError):
            PackedRestorer(SoftwarePool(), granularity="node", arena_bytes=1)

    def destination_factory(self):
        caches = {}

        def destination(node_id, key, tile):
            heads, head_dim = HEADS[key.target]
            name = (node_id, key)
            if name not in caches:
                caches[name] = torch.zeros((1, 180, heads, head_dim), dtype=torch.float32)
            return rk.NativeView(caches[name], "BTHD", token_offset=0)

        return caches, destination

    def test_reconstruction_matches_software_coefficients_and_recycles_canvases(self):
        freqs = torch.polar(torch.ones(64, 3, dtype=torch.float64),
                            torch.outer(torch.arange(64, dtype=torch.float64), torch.tensor([1.0, 0.1, 0.01],
                                                                                            dtype=torch.float64)))
        table = rk.RopeTable.from_complex(freqs)

        def rope(node_id, key, tile):
            return rk.RopePositions(table, (tile.token_start // 180 * 3, 0, 0)) if key.target == "K" else None

        restorer = self.restorer(budget=4 * 1024**2)
        batch = restorer.submit(self.nodes())
        caches, destination = self.destination_factory()
        for layer in batch.layers:
            handle = batch.reconstruct(batch.entries(layer), destination, rope)
            self.assertEqual(handle.stats["jobs"], 8)
        references, reference_destination = self.destination_factory()
        jobs = []
        for node_id, key in batch.entries():
            basis, values, _ = self.coefficients[(node_id, key)]
            tile = batch._entries[(node_id, key)][1]
            jobs.append(rk.ReconstructionJob(rk.CoefficientSource(values), reference_destination(node_id, key, tile),
                                             tile.grid, basis=basis, rank=tile.rank, rope=rope(node_id, key, tile)))
        rk.reconstruct(jobs)
        for name, value in caches.items():
            torch.testing.assert_close(value, references[name], rtol=1e-6, atol=1e-6, msg=str(name))
        report = restorer.arena.report()
        self.assertEqual(report["released"], len(batch.jobs))
        self.assertEqual(report["free"] + report["retiring"], len(batch.jobs) - report["reuses"])
        batch.close()

    def test_bounded_arena_and_layer_ahead_prefetch(self):
        prepared = prepare_packet(self.node)
        canvas_bytes = 2 * prepared.tiles[0].pictures[1] * 3 * prepared.height * prepared.width
        restorer = self.restorer(budget=2 * canvas_bytes)
        prefetch = PackedLayerPrefetcher(restorer, depth=1)
        batch = prefetch.begin([NodeRef("a", [prepared])], resident={("a", MatrixKey(1, "V"))})
        self.assertEqual(sum(job.submitted for job in batch.jobs), 2)
        caches, destination = self.destination_factory()
        for layer in prefetch.layers:
            prefetch.restore(layer, destination)
            self.assertLessEqual(restorer.arena.report()["peak_bytes"], 2 * canvas_bytes)
        self.assertIsNone(prefetch.restore(7, destination))
        prefetch.end()
        prefetch.end()
        self.assertEqual(prefetch.stats()["skipped_resident"], 1)
        self.assertEqual(len(caches), len(self.matrices) - 1)
        with self.assertRaises(StateError):
            prefetch.restore(0, destination)
        with self.assertRaises(ConfigurationError):
            PackedLayerPrefetcher(object())


class FakeStream:
    def __init__(self):
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event)


class SoftwareSession:
    """CPU stand-in for CanvasSession inside the real DecoderPool: FFmpeg decode, plain in-place canvas copies.

    It deliberately leaves ``torch.inference_mode`` alone, so the pool worker has to provide the canvas's mode.
    ``gate`` (a threading.Event) holds every decode until set, so a test can route jobs while workers are busy.
    ``log`` (a list) records ``(worker thread, geometry, units)`` per decode, in the order workers start them.
    """

    gate = None
    log = None
    # Session creation: ``create_seconds`` of work per session; ``creating`` / ``max_creating`` count concurrent ones.
    create_seconds = 0.0
    creating = 0
    max_creating = 0
    creation_lock = threading.Lock()

    def __init__(self, *, gpu_id, width, height, pix_fmt, api=None):
        with SoftwareSession.creation_lock:
            SoftwareSession.creating += 1
            SoftwareSession.max_creating = max(SoftwareSession.max_creating, SoftwareSession.creating)
        try:
            if SoftwareSession.create_seconds:
                time.sleep(SoftwareSession.create_seconds)
        finally:
            with SoftwareSession.creation_lock:
                SoftwareSession.creating -= 1
        self.width, self.height, self.pix_fmt = width, height, pix_fmt
        self.stream = FakeStream()
        self.setup_seconds = {"caps": 0.0, "stream": 0.0, "create_decoder": 0.0}
        self.stats = {"first_picture_seconds": None}

    @property
    def geometry(self):
        return self.width, self.height, self.pix_fmt

    def decode(self, units, canvas, first=0):
        if SoftwareSession.log is not None:
            SoftwareSession.log.append((threading.current_thread().name, self.geometry, len(units)))
        if SoftwareSession.gate is not None and not SoftwareSession.gate.wait(60):
            raise TimeoutError("the test gate was never opened")
        units, pictures, _ = check_units(units)
        height, width = check_canvas(canvas, device=torch.device("cpu"), pix_fmt=self.pix_fmt,
                                     pictures=first + pictures)
        stream = b"".join(data[offset:offset + length] for data, offset, length, _ in units)
        images, _ = _hevc.decode_pictures(stream, pictures=pictures, height=height, width=width, pix_fmt=self.pix_fmt)
        canvas[first:first + pictures].copy_(torch.from_numpy(np.array(images)))
        if self.stats["first_picture_seconds"] is None:
            self.stats["first_picture_seconds"] = 0.0
        return None

    def close(self):
        pass


@contextmanager
def threaded_software_pool(workers=2, **options):
    """The real DecoderPool (worker threads, routing, tickets, session LRU) decoding in software on CPU.

    Like the NVDEC pool, its worker threads never inherit the submitter's thread-local inference mode.
    ``pool.recorded`` lists ``(canvas is an inference tensor, inference mode enabled)`` per ``record_stream``.
    """
    recorded = []

    def record_stream(tensor, stream):
        recorded.append((tensor.is_inference(), torch.is_inference_mode_enabled()))

    properties = SimpleNamespace(uuid=UUID.removeprefix("GPU-"))
    with mock.patch("torch.cuda.set_device"), mock.patch("reelcache.decoder_pool.CanvasSession", SoftwareSession), \
            mock.patch.object(torch.Tensor, "record_stream", record_stream):
        with mock.patch("torch.cuda.device_count", return_value=1), \
                mock.patch("torch.cuda.get_device_properties", return_value=properties):
            pool = DecoderPool(gpu_id=0, gpu_uuid=UUID, workers=workers, api=object(), **options)
        pool.device = torch.device("cpu")
        pool.recorded = recorded
        try:
            yield pool
        finally:
            pool.close()


class InferenceModeTests(unittest.TestCase):
    """Restores issued inside ``torch.inference_mode`` (how SGLang serves requests) through decoder threads.

    The GPU counterpart (real NVDEC sessions, Triton reconstruction, real K98/V95 packets) is
    artifacts/sglang_integration/benchmarks/restore_e2e/scripts/inference_mode_gpu.py.
    """

    @classmethod
    def setUpClass(cls):
        cls.matrices, cls.contexts, cls.specs = node(seed=11)
        cls.node = encode_node(cls.matrices, cls.contexts, cls.specs, packing="node", node_id="a").packets[0]
        cls.groups = encode_node(cls.matrices, cls.contexts, cls.specs, packing="layer_group",
                                 layer_group_size=2, node_id="b").packets
        cls.singles = encode_node(cls.matrices, cls.contexts, cls.specs, packing="per_matrix").packets

    def nodes(self):
        return [NodeRef("a", [prepare_packet(self.node)]),
                NodeRef("b", [prepare_packet(packet) for packet in self.groups]),
                NodeRef("c", [prepare_packet(packet) for packet in self.singles])]

    def test_helpers_make_normal_buffers_and_write_inference_tensors_on_any_thread(self):
        with torch.inference_mode():
            inference, probe = torch.zeros(4), torch.zeros(4)
            with normal_tensors():
                normal = torch.zeros(4)
                grad = torch.is_grad_enabled()
        self.assertTrue(inference.is_inference())
        self.assertFalse(normal.is_inference())
        self.assertFalse(grad)
        outcomes = {}

        def worker():
            outcomes["inherited"] = torch.is_inference_mode_enabled()
            try:
                probe.add_(1)
            except RuntimeError as exc:
                outcomes["unguarded"] = str(exc)
            with write_mode(inference):
                inference.add_(1)
            with write_mode(normal):
                outcomes["normal_write_mode"] = torch.is_inference_mode_enabled()
                normal.add_(1)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        self.assertFalse(outcomes["inherited"])
        self.assertIn("Inplace update to inference tensor outside InferenceMode", outcomes["unguarded"])
        self.assertFalse(outcomes["normal_write_mode"])
        self.assertTrue(torch.equal(inference, torch.ones(4)))
        self.assertTrue(torch.equal(normal, torch.ones(4)))

    def test_arena_canvases_and_prepared_bases_are_normal_tensors_inside_inference_mode(self):
        with torch.inference_mode():
            arena = CanvasArena("cpu", 4000)
            canvas = arena.acquire((1, 1, 10, 100))
            prepared = [prepare_packet(self.node)] + [prepare_packet(packet) for packet in self.singles]
        self.assertFalse(canvas.is_inference())
        self.assertFalse(any(tile.basis.is_inference() for packet in prepared for tile in packet.tiles))
        canvas.fill_(7)  # writable outside inference mode, as decoder threads write it
        arena.release(canvas)
        with torch.inference_mode():
            again = arena.acquire((1, 1, 10, 100))
            again.fill_(3)
        self.assertIs(again, canvas)
        self.assertFalse(again.is_inference())

    def test_pool_workers_write_caller_inference_canvases_in_their_mode(self):
        prepared = prepare_packet(self.node)
        shape = (prepared.pictures, prepared.planes, prepared.height, prepared.width)
        units = list(prepared.stream_units)
        joined = b"".join(data[offset:offset + length] for data, offset, length, _ in units)
        expected = np.array(_hevc.decode_pictures(joined, pictures=len(units), height=prepared.height,
                                                  width=prepared.width, pix_fmt=prepared.pix_fmt)[0])
        ready = FakeEvent()  # caller canvases are ordered explicitly (unordered decodes need arena canvases)
        with threaded_software_pool(2) as pool:
            with torch.inference_mode():
                canvas = torch.zeros(shape, dtype=torch.uint8)
                ticket = pool.submit(units, canvas, pix_fmt=prepared.pix_fmt, after=[ready])
            ticket.event(timeout=120)
            normal = torch.zeros(shape, dtype=torch.uint8)
            pool.submit(units, normal, pix_fmt=prepared.pix_fmt, after=[ready]).event(timeout=120)
            # Without the worker's mode handling the write is exactly the error SGLang hit.
            with mock.patch("reelcache.decoder_pool.write_mode", lambda tensor: nullcontext()):
                with torch.inference_mode():
                    unguarded = torch.zeros(shape, dtype=torch.uint8)
                failed = pool.submit(units, unguarded, pix_fmt=prepared.pix_fmt, after=[ready])
                with self.assertRaisesRegex(RuntimeError, "Inplace update to inference tensor outside InferenceMode"):
                    failed.event(timeout=120)
            recorded = set(pool.recorded)
            stats = pool.stats()
        self.assertTrue(canvas.is_inference())
        self.assertTrue(np.array_equal(canvas.numpy(), expected))
        self.assertTrue(torch.equal(canvas, normal))
        self.assertEqual(recorded, {(False, False), (True, True)})
        self.assertEqual((stats["jobs"], stats["failures"]), (3, 1))

    def test_restore_inside_inference_mode_matches_normal_mode_bitwise(self):
        def restore(restorer, nodes, inference, layer_ahead=False):
            with torch.inference_mode(inference):
                # SGLang's native caches and engine RoPE tables are inference tensors inside its requests.
                table = rk.RopeTable.from_complex(torch.polar(
                    torch.ones(64, 3, dtype=torch.float64),
                    torch.outer(torch.arange(64, dtype=torch.float64), torch.tensor([1.0, 0.1, 0.01],
                                                                                    dtype=torch.float64))))
                caches = {}

                def destination(node_id, key, tile):
                    heads, head_dim = HEADS[key.target]
                    if (node_id, key) not in caches:
                        caches[(node_id, key)] = torch.zeros((1, 180, heads, head_dim), dtype=torch.float32)
                    return rk.NativeView(caches[(node_id, key)], "BTHD", token_offset=0)

                def rope(node_id, key, tile):
                    return rk.RopePositions(table, (tile.token_start // 180 * 3, 0, 0)) if key.target == "K" else None

                if layer_ahead:
                    prefetch = PackedLayerPrefetcher(restorer, depth=1)
                    prefetch.begin(nodes)
                    for layer in prefetch.layers:
                        prefetch.restore(layer, destination, rope)
                    prefetch.end()
                else:
                    with restorer.submit(nodes) as batch:
                        for layer in batch.layers:
                            batch.reconstruct(batch.entries(layer), destination, rope)
            return caches

        with threaded_software_pool(2) as pool:
            reference = restore(PackedRestorer(pool, arena_bytes=64 * 1024**2), self.nodes(), False)
            with torch.inference_mode():
                nodes = self.nodes()  # prepared inside the request, reused by every later restore
            restorer = PackedRestorer(pool, arena_bytes=64 * 1024**2, cache_bases=True)
            runs = {
                "inference": restore(restorer, nodes, True),
                "normal_after_inference": restore(restorer, nodes, False),  # canvases/bases of the inference run
                "inference_after_normal": restore(restorer, nodes, True),
                "inference_layer_ahead": restore(restorer, nodes, True, layer_ahead=True),
            }
            report = restorer.arena.report()
            recorded = set(pool.recorded)
            cached = [value for _, value in restorer._bases.values()]
        self.assertEqual(len(reference), 3 * len(self.matrices))
        for name, caches in runs.items():
            self.assertEqual(caches.keys(), reference.keys(), name)
            for entry, value in caches.items():
                self.assertTrue(torch.equal(value, reference[entry]), (name, entry))
            self.assertEqual(all(value.is_inference() for value in caches.values()), name != "normal_after_inference")
        self.assertGreater(report["reuses"], 0)
        self.assertEqual(report["released"], report["allocations"] + report["reuses"])
        self.assertEqual(recorded, {(False, False)})  # arena canvases are normal tensors
        self.assertTrue(cached and not any(value.is_inference() for value in cached))


class StagedReadinessTests(unittest.TestCase):
    """Tiles whose bases are still being staged (``reelcache.restore_staging`` tickets): decode does not wait,
    each reconstruction waits for its own tiles' tickets, a failed ticket fails the reconstruction before any
    write, and parallel access-unit scans equal the sequential ones."""

    @classmethod
    def setUpClass(cls):
        cls.matrices, cls.contexts, cls.specs = node(seed=13)
        cls.pack = encode_node(cls.matrices, cls.contexts, cls.specs, packing="node", node_id="s").packets[0]

    def restore(self, prepared, on_layer=None):
        caches = {}

        def destination(node_id, key, tile):
            heads, head_dim = HEADS[key.target]
            return rk.NativeView(caches.setdefault((node_id, key), torch.zeros((1, 180, heads, head_dim))),
                                 "BTHD", token_offset=0)

        with threaded_software_pool(2) as pool:
            restorer = PackedRestorer(pool, arena_bytes=64 * 1024**2)
            with restorer.submit([NodeRef("s", [prepared])]) as batch:
                for layer in batch.layers:
                    if on_layer is not None:
                        on_layer(layer)
                    batch.reconstruct(batch.entries(layer), destination)
        return caches

    def staged(self, tickets):
        from dataclasses import replace

        prepared = prepare_packet(self.pack)
        return replace(prepared, tiles=tuple(replace(tile, ready=tickets(tile)) for tile in prepared.tiles))

    def test_reconstruction_waits_for_its_tickets_and_matches_the_unstaged_restore(self):
        from reelcache.restore_staging import StagingTicket

        reference = self.restore(prepare_packet(self.pack))
        tickets = {}

        def ticket(tile):
            return tickets.setdefault(tile.layer, StagingTicket(f"layer {tile.layer}"))

        prepared = self.staged(ticket)
        released, order = {}, []

        def on_layer(layer):
            order.append(layer)
            threading.Timer(0.05, lambda: (released.setdefault(layer, time.perf_counter()),
                                           tickets[layer]._set(None))).start()

        started = []
        original = rk.reconstruct

        def timed(jobs, **options):
            started.append(time.perf_counter())
            return original(jobs, **options)

        with mock.patch.object(rk, "reconstruct", timed):
            caches = self.restore(prepared, on_layer)
        self.assertEqual(caches.keys(), reference.keys())
        for entry, value in caches.items():
            self.assertTrue(torch.equal(value, reference[entry]), entry)
        self.assertEqual(len(started), len(order))
        for position, layer in enumerate(order):
            self.assertGreaterEqual(started[position], released[layer])

    def test_a_failed_ticket_fails_its_reconstruction_before_any_write(self):
        from reelcache.restore_staging import StagingTicket

        good, bad = StagingTicket("good"), StagingTicket("bad")
        good._set(None)
        bad._fail(IntegrityError("staged rows differ from the tile descriptor"))
        prepared = self.staged(lambda tile: bad if tile.layer == 1 else good)
        caches = {}

        def destination(node_id, key, tile):
            heads, head_dim = HEADS[key.target]
            return rk.NativeView(caches.setdefault((node_id, key), torch.zeros((1, 180, heads, head_dim))),
                                 "BTHD", token_offset=0)

        with threaded_software_pool(2) as pool:
            restorer = PackedRestorer(pool, arena_bytes=64 * 1024**2)
            with restorer.submit([NodeRef("s", [prepared])]) as batch:
                batch.reconstruct(batch.entries(0), destination)
                with self.assertRaisesRegex(IntegrityError, "differ from the tile descriptor"):
                    batch.reconstruct(batch.entries(1), destination)
        written = {key.layer for (_, key), value in caches.items() if value.abs().sum() > 0}
        self.assertEqual(written, {0})

    def test_parallel_access_unit_scans_equal_sequential(self):
        from concurrent.futures import ThreadPoolExecutor

        sequential = prepare_packet(self.pack)
        with ThreadPoolExecutor(3) as executor:
            parallel = prepare_packet(self.pack, executor=executor)
        self.assertEqual([tile.units for tile in parallel.tiles], [tile.units for tile in sequential.tiles])
        self.assertEqual(parallel.stream_units, sequential.stream_units)
        with self.assertRaises(IntegrityError):  # an embedded-basis pack cannot take external bases
            prepare_packet(self.pack, bases=[])
        # Scans start before validation; a malformed tile map still fails validation (not a scan error).
        metadata = json.loads(json.dumps(self.pack.metadata))
        metadata["tiles"][1]["byte_range"][0] += 1
        broken = EncodedTensor(metadata, dict(self.pack.payloads))
        with ThreadPoolExecutor(3) as executor:
            with self.assertRaisesRegex(IntegrityError, "not contiguous"):
                prepare_packet(broken, executor=executor)


class OrderingContractTests(unittest.TestCase):
    """DecoderPool.submit write ordering. Step 16 decoded garbage with ``after=None`` into a plain ``torch.empty``
    canvas (NVDEC wrote memory the allocator had recycled from work still queued on the compute stream); such
    unordered submissions now need a canvas leased from a CanvasArena. The GPU race itself is reproduced by
    artifacts/sglang_integration/benchmarks/restore_e2e/scripts/inference_mode_gpu.py (``after_contract``)."""

    @classmethod
    def setUpClass(cls):
        matrices, contexts, specs = node(seed=13)
        cls.prepared = prepare_packet(encode_node(matrices, contexts, specs, packing="node", node_id="o").packets[0])

    def test_unordered_decodes_need_a_leased_arena_canvas(self):
        prepared = self.prepared
        units = list(prepared.stream_units)
        shape = (prepared.pictures, prepared.planes, prepared.height, prepared.width)
        joined = b"".join(data[offset:offset + length] for data, offset, length, _ in units)
        expected = np.array(_hevc.decode_pictures(joined, pictures=len(units), height=prepared.height,
                                                  width=prepared.width, pix_fmt=prepared.pix_fmt)[0])
        size = int(np.prod(shape))
        arena, other = CanvasArena("cpu", 2 * size), CanvasArena("cpu", 2 * size)
        with threaded_software_pool(2) as pool:
            for after in (None, (), [], [None]):  # no ordering at all
                with self.assertRaisesRegex(ConfigurationError, "leased from a CanvasArena"):
                    pool.submit(units, torch.empty(shape, dtype=torch.uint8), pix_fmt=prepared.pix_fmt, after=after)
            for after in ("later", [object()], 3):
                with self.assertRaisesRegex(ConfigurationError, "after must be"):
                    pool.submit(units, torch.empty(shape, dtype=torch.uint8), pix_fmt=prepared.pix_fmt, after=after)
            leased = arena.acquire(shape)
            pool.submit(units, leased, pix_fmt=prepared.pix_fmt, after=None).event(timeout=120)
            self.assertTrue(np.array_equal(leased.numpy(), expected))
            with self.assertRaisesRegex(ConfigurationError, "leased from a CanvasArena"):
                pool.submit(units[:1], leased[:1], pix_fmt=prepared.pix_fmt, after=None)  # a view carries no lease
            ordered = torch.empty(shape, dtype=torch.uint8)
            pool.submit(units, ordered, pix_fmt=prepared.pix_fmt, after=[FakeEvent()]).event(timeout=120)
            self.assertTrue(np.array_equal(ordered.numpy(), expected))
            arena.release(leased)
            with self.assertRaisesRegex(ConfigurationError, "leased from a CanvasArena"):
                pool.submit(units, leased, pix_fmt=prepared.pix_fmt, after=None)  # released: no longer leased
            stats = pool.stats()
        self.assertEqual((stats["jobs"], stats["failures"]), (2, 0))  # rejected submissions never reach a worker
        with self.assertRaisesRegex(StateError, "not leased"):
            arena.release(leased)  # double release would hand the canvas out twice
        again = arena.acquire(shape)
        self.assertIs(again, leased)
        with self.assertRaisesRegex(StateError, "not leased"):
            other.release(again)
        with self.assertRaisesRegex(StateError, "not leased"):
            arena.release(torch.empty(shape, dtype=torch.uint8))
        arena.release(again)
        self.assertEqual(arena.report()["released"], 2)
        self.assertEqual(check_ordering(torch.empty(1), "current"), "current")
        self.assertEqual(len(check_ordering(torch.empty(1), [FakeEvent(), None])), 1)


@contextmanager
def held_decodes():
    """Hold every SoftwareSession decode until the block ends (routing then sees all queued work)."""
    gate = threading.Event()
    SoftwareSession.gate = gate
    try:
        yield gate
    finally:
        gate.set()
        SoftwareSession.gate = None


@contextmanager
def logged_decodes():
    """Record ``(worker thread, geometry, units)`` of every SoftwareSession decode in start order."""
    log = []
    SoftwareSession.log = log
    try:
        yield log
    finally:
        SoftwareSession.log = None


def wait_until(predicate, timeout=60.0):
    deadline = time.perf_counter() + timeout
    while not predicate():
        if time.perf_counter() > deadline:
            raise TimeoutError("condition not reached")
        time.sleep(0.002)


def settled(pool):
    """No work is queued or running on any worker."""
    return all(worker["outstanding_bytes"] == 0 and not worker["queued"] for worker in pool.stats()["workers"])


class RoutingTests(unittest.TestCase):
    """DecoderPool routing. Step 16: with ~7.8 MB packs and an 8 MiB cold-session charge, a geometry that was not
    prewarmed stayed on the worker that decoded it first, so every later restore decoded its packs serially there
    (LongLive 1.0: 98 ms instead of 64 ms); queued session creations are now charged, so the first jobs of a geometry
    spread. CR-17-5: background warm-ups of a geometry on the other workers delayed restores in every form (queued
    behind a restore's jobs, idle-priority, one at a time after an idle grace period: one creation overlapping a
    restore cost it ~0.3 s on A100), so the pool creates sessions only synchronously -- prewarm, or a job's own
    creation -- and a session creation is charged once whatever the canvas (calibrated)."""

    @classmethod
    def setUpClass(cls):
        matrices, contexts, specs = node(seed=19)
        cls.yuv444 = prepare_packet(encode_node(matrices, contexts, specs, packing="node", node_id="r").packets[0])
        matrices, contexts, specs = node(seed=23, pix_fmt="yuv420p")
        cls.yuv420 = prepare_packet(encode_node(matrices, contexts, specs, packing="node", node_id="s").packets[0])

    @staticmethod
    def geometry(packet):
        return packet.width, packet.height, packet.pix_fmt

    @staticmethod
    def expected(packet):
        units = list(packet.stream_units)
        joined = b"".join(data[offset:offset + length] for data, offset, length, _ in units)
        return np.array(_hevc.decode_pictures(joined, pictures=len(units), height=packet.height, width=packet.width,
                                              pix_fmt=packet.pix_fmt)[0])

    @staticmethod
    def submit(pool, packet, worker=None):
        canvas = torch.zeros((packet.pictures, packet.planes, packet.height, packet.width), dtype=torch.uint8)
        ticket = pool.submit(list(packet.stream_units), canvas, pix_fmt=packet.pix_fmt, after=[FakeEvent()],
                             worker=worker)
        return ticket, canvas

    def restore(self, pool, packet, count, tickets=None):
        """``count`` whole-canvas jobs submitted while decodes are held; returns their worker slots."""
        shape = (packet.pictures, packet.planes, packet.height, packet.width)
        with held_decodes():
            submitted = [pool.submit(list(packet.stream_units), torch.empty(shape, dtype=torch.uint8),
                                     pix_fmt=packet.pix_fmt, after=[FakeEvent()]) for _ in range(count)]
        for ticket in submitted:
            ticket.event(timeout=120)
        if tickets is not None:
            tickets.extend(submitted)
        return [ticket.slot for ticket in submitted]

    def test_a_new_geometry_spreads_over_the_workers_through_its_own_jobs(self):
        geometry = self.geometry(self.yuv444)
        with threaded_software_pool(4) as pool:
            first = self.restore(pool, self.yuv444, 4)
            created = pool.stats()
            second = self.restore(pool, self.yuv444, 4)
            third = self.restore(pool, self.yuv444, 8)
            stats = pool.stats()
        # Each first job pays its own worker's session creation (charged), so the next job goes to a cold idle worker
        # rather than queueing on a warm one; later restores find the geometry warm on every worker.
        self.assertEqual(first, [0, 1, 2, 3])
        self.assertEqual(second, [0, 1, 2, 3])
        self.assertEqual(sorted(third), [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual((created["cold_jobs"], created["auto_warms"], created["background_warm_ups"]), (4, 0, "none"))
        for worker in stats["workers"]:
            self.assertEqual(worker["geometries"], [geometry])
            self.assertEqual((worker["sessions_created"], worker["outstanding_bytes"], worker["jobs"]), (1, 0, 4))
        self.assertEqual((stats["cold_jobs"], stats["auto_warms"], stats["jobs"]), (4, 0, 16))

    def test_no_session_is_ever_created_in_the_background(self):
        a, b = self.geometry(self.yuv444), self.geometry(self.yuv420)
        with threaded_software_pool(4) as pool, logged_decodes() as log:
            with held_decodes():
                first, _ = self.submit(pool, self.yuv444, worker=0)
                second, canvas = self.submit(pool, self.yuv420, worker=0)
                pending = pool.stats()["workers"]
            for ticket in (first, second):
                ticket.event(timeout=120)
            time.sleep(0.6)  # idle for longer than any former warm-up grace period (0.25 s)
            stats = pool.stats()
        # Queued jobs charge their session creation on their own worker only; nothing is queued anywhere else, then
        # or after the pool idled.
        self.assertEqual([worker["geometries"] for worker in pending], [sorted([a, b]), [], [], []])
        self.assertEqual([worker["outstanding_bytes"] > 0 for worker in pending], [True, False, False, False])
        self.assertEqual(log, [("reelcache-nvdec-0", a, len(self.yuv444.stream_units)),
                               ("reelcache-nvdec-0", b, len(self.yuv420.stream_units))])
        self.assertEqual([worker["sessions_created"] for worker in stats["workers"]], [2, 0, 0, 0])
        self.assertEqual([worker["geometries"] for worker in stats["workers"]], [sorted([a, b]), [], [], []])
        self.assertEqual((stats["cold_jobs"], stats["auto_warms"], stats["jobs"]), (2, 0, 2))
        self.assertTrue(np.array_equal(canvas.numpy(), self.expected(self.yuv420)))

    def test_a_restore_never_waits_on_a_session_creation_it_did_not_route(self):
        # CR-17-5 guarantee (coordinator 17:27): the C2 pool's one background warm-up at a time, started after a
        # 0.25 s idle grace period, still met restores (A100: +0.32 / +0.34 s). Session creations are slow here
        # (1 s each); a watcher records every moment one is in progress, and the pool's wait records must agree.
        observed = []
        stop = threading.Event()

        def watch():  # records every moment a session creation is in progress
            while not stop.is_set():
                if SoftwareSession.creating:
                    observed.append(time.perf_counter())
                time.sleep(0.001)

        watcher = threading.Thread(target=watch, daemon=True)
        try:
            SoftwareSession.create_seconds, SoftwareSession.max_creating = 1.0, 0
            watcher.start()
            with threaded_software_pool(4) as pool:
                first = self.restore(pool, self.yuv444, 2)  # a geometry new to the pool: two workers create it
                cold = pool.stats()
                for gap in (0.6, 0.0, 0.6):  # an idle pool between restores, then back to back, then idle again
                    time.sleep(gap)
                    tickets = []
                    before = pool.stats()
                    started = time.perf_counter()
                    slots = self.restore(pool, self.yuv444, 6, tickets)
                    finished = time.perf_counter()
                    after = pool.stats()
                    self.assertEqual(sorted(set(slots)), sorted(set(first)), gap)  # stays on its warm workers
                    self.assertFalse([moment for moment in observed if started <= moment <= finished], gap)
                    waits = [ticket.started - ticket.submitted for ticket in tickets]
                    self.assertEqual(sum(ticket.cold for ticket in tickets), 0, gap)
                    # The records say the same: no warm-up in flight at submit, no wait behind a creation or warm-up.
                    self.assertEqual((before["warm_ups_in_flight"], before["queued_jobs"]), (0, 0), gap)
                    self.assertEqual(after["jobs_started"] - before["jobs_started"], 6, gap)
                    self.assertAlmostEqual(after["queue_wait_seconds"] - before["queue_wait_seconds"], sum(waits),
                                           places=6, msg=gap)
                    for name in ("creation_wait_seconds", "warm_up_wait_seconds", "sessions_created", "cold_jobs"):
                        self.assertEqual(after[name], before[name], (gap, name))
                time.sleep(0.6)
                stats = pool.stats()
        finally:
            stop.set()
            watcher.join(10)
            SoftwareSession.create_seconds = 0.0
        self.assertEqual(len(set(first)), 2)
        self.assertEqual(cold["cold_jobs"], 2)
        # The only creations were the first restore's own two (concurrent, inside that restore).
        self.assertEqual(sum(worker["sessions_created"] for worker in stats["workers"]), 2)
        self.assertEqual((stats["cold_jobs"], stats["auto_warms"], stats["failures"]), (2, 0, 0))
        self.assertLessEqual(SoftwareSession.max_creating, 2)

    def test_stats_record_queue_creation_and_warm_up_waits(self):
        # Records only (Step 11's per-restore bridge records): cumulative waits, so a restore's share is a difference.
        a = self.geometry(self.yuv444)
        b = (*self.geometry(self.yuv420), self.yuv420.stream_units[0])
        try:
            SoftwareSession.create_seconds = 1.0
            with threaded_software_pool(1) as pool:
                start = pool.stats()
                tickets = []
                self.restore(pool, self.yuv444, 2, tickets)  # the 1st job creates the session, the 2nd waits behind
                cold = pool.stats()
                warming = threading.Thread(target=pool.prewarm, args=([b],), kwargs={"timeout": 120})
                warming.start()
                wait_until(lambda: pool.stats()["warm_up_running"] == 1)
                behind = pool.stats()
                # One worker: nowhere else to route, so the job waits out the running warm-up (and it is recorded).
                ticket, _ = self.submit(pool, self.yuv444)
                ticket.event(timeout=120)
                warming.join(60)
                end = pool.stats()
        finally:
            SoftwareSession.create_seconds = 0.0
        waited = [ticket.started - ticket.submitted for ticket in tickets]
        self.assertEqual((start["jobs_started"], start["queue_wait_seconds"], start["queue_wait_max_seconds"]),
                         (0, 0.0, 0.0))
        self.assertEqual((cold["jobs_started"], cold["queued_jobs"], cold["sessions_created"]), (2, 0, 1))
        self.assertAlmostEqual(cold["queue_wait_seconds"], sum(waited), places=6)
        self.assertAlmostEqual(cold["queue_wait_max_seconds"], max(waited), places=6)
        self.assertGreaterEqual(cold["creation_wait_seconds"], 0.9)  # the 2nd job waited out the 1 s creation
        self.assertLessEqual(cold["creation_wait_seconds"], cold["queue_wait_seconds"] + 1e-9)
        self.assertEqual(cold["warm_up_wait_seconds"], 0.0)
        self.assertEqual(behind["warm_ups_in_flight"], 1)  # what a restore submitted now would record
        warm_up = end["warm_up_wait_seconds"] - cold["warm_up_wait_seconds"]
        self.assertGreaterEqual(warm_up, 0.5)
        self.assertLessEqual(warm_up, end["queue_wait_seconds"] - cold["queue_wait_seconds"] + 1e-9)
        self.assertGreaterEqual(end["creation_wait_seconds"] - cold["creation_wait_seconds"], 0.5)
        self.assertEqual((end["warm_ups_in_flight"], end["queued_jobs"], end["jobs_started"], end["jobs"]),
                         (0, 0, 3, 3))
        worker = end["workers"][0]
        for name in ("jobs_started", "queue_wait_seconds", "creation_wait_seconds", "warm_up_wait_seconds",
                     "queue_wait_max_seconds", "sessions_created"):
            self.assertEqual(worker[name], end[name], name)  # one worker: its records are the pool's

    def test_a_concurrent_prewarm_never_delays_a_restore(self):
        # Coordinator 18:5x (A6): an explicit prewarm can run on another thread while a restore decodes (seal-time
        # preparation on an archival worker). Its creations are slow here (0.5 s each, 4 workers: ~2 s in all); the
        # restores of a warm geometry must never wait on one.
        a = (*self.geometry(self.yuv444), self.yuv444.stream_units[0])
        b = (*self.geometry(self.yuv420), self.yuv420.stream_units[0])
        with threaded_software_pool(4) as pool:
            pool.prewarm([a], timeout=120)
            try:
                SoftwareSession.create_seconds, SoftwareSession.max_creating = 0.5, 0
                warming = threading.Thread(target=pool.prewarm, args=([b],), kwargs={"timeout": 120})
                warming.start()
                wait_until(lambda: pool.stats()["warm_up_running"] == 1)
                before = pool.stats()
                tickets = []
                while warming.is_alive():
                    self.restore(pool, self.yuv444, 6, tickets)
                    time.sleep(0.05)
                warming.join(60)
                after = pool.stats()
                spans = {worker.index: list(worker.warm_up_spans) for worker in pool._workers}
            finally:
                SoftwareSession.create_seconds = 0.0
            ticket, canvas = self.submit(pool, self.yuv444)
            ticket.event(timeout=120)
            final = pool.stats()
        self.assertGreaterEqual(len(tickets), 12)  # at least two restores overlapped the prewarm
        self.assertEqual(SoftwareSession.max_creating, 1)  # the prewarm created one session at a time
        # No job waited while a warm-up ran on its worker, nor was any routed to a worker while it warmed (exact;
        # wall-clock waits depend on the host's load).
        self.assertEqual(after["warm_up_wait_seconds"] - before["warm_up_wait_seconds"], 0.0)
        self.assertEqual(after["creation_wait_seconds"] - before["creation_wait_seconds"], 0.0)
        self.assertFalse([ticket for ticket in tickets
                          if any(start <= ticket.submitted <= end for start, end in spans[ticket.slot])])
        self.assertEqual(sum(ticket.cold for ticket in tickets), 0)
        self.assertEqual((after["warm_ups"] - before["warm_ups"] + 1, after["warm_ups_in_flight"],
                          after["warm_up_running"]), (4, 0, 0))  # ``before`` already counted the running one
        for worker in final["workers"]:  # synchronous: every session exists when prewarm returned
            self.assertEqual(sorted(worker["geometries"]), sorted([a[:3], b[:3]]))
            self.assertEqual((worker["sessions_created"], worker["outstanding_bytes"]), (2, 0))
        self.assertTrue(np.array_equal(canvas.numpy(), self.expected(self.yuv444)))

    def test_jobs_go_ahead_of_queued_warm_ups_and_moot_ones_are_dropped(self):
        a, b = self.geometry(self.yuv444), self.geometry(self.yuv420)
        sample = (*b, self.yuv420.stream_units[0])
        with threaded_software_pool(1) as pool, logged_decodes() as log:
            with held_decodes():
                first, _ = self.submit(pool, self.yuv444)  # the worker is busy (held)
                warming = threading.Thread(target=pool.prewarm, args=([sample],), kwargs={"timeout": 120})
                warming.start()
                wait_until(lambda: pool.stats()["workers"][0]["warm_queued"] == 1)
                queued = pool.stats()
                second, _ = self.submit(pool, self.yuv444)  # submitted after the warm-up: decoded before it
                third, canvas = self.submit(pool, self.yuv420)  # creates b itself: the queued warm-up becomes moot
            for ticket in (first, second, third):
                ticket.event(timeout=120)
            warming.join(60)
            stats = pool.stats()
        # Queued warm-ups are neither charged nor warm for routing.
        self.assertEqual((queued["workers"][0]["geometries"], queued["warm_up_running"]), ([a], 0))
        self.assertEqual(log, [("reelcache-nvdec-0", a, len(self.yuv444.stream_units)),
                               ("reelcache-nvdec-0", a, len(self.yuv444.stream_units)),
                               ("reelcache-nvdec-0", b, len(self.yuv420.stream_units))])  # no warm-up decode at all
        self.assertFalse(warming.is_alive())  # prewarm returned: its session exists
        self.assertEqual((stats["warm_ups"], stats["warm_ups_moot"], stats["sessions_created"], stats["cold_jobs"]),
                         (1, 1, 2, 2))
        self.assertEqual((stats["warm_ups_in_flight"], stats["workers"][0]["outstanding_bytes"]), (0, 0))
        self.assertTrue(np.array_equal(canvas.numpy(), self.expected(self.yuv420)))

    def test_warm_ups_run_one_at_a_time_across_the_pool(self):
        samples = [(*self.geometry(packet), packet.stream_units[0]) for packet in (self.yuv444, self.yuv420)]
        try:
            SoftwareSession.create_seconds, SoftwareSession.max_creating = 0.05, 0
            with threaded_software_pool(4) as pool:
                seconds = pool.prewarm(samples, timeout=120)
                stats = pool.stats()
        finally:
            SoftwareSession.create_seconds = 0.0
        self.assertEqual(SoftwareSession.max_creating, 1)
        self.assertGreaterEqual(seconds, 8 * 0.05)
        self.assertEqual((stats["warm_ups"], stats["warm_ups_moot"], stats["sessions_created"]), (8, 0, 8))
        with threaded_software_pool(2) as pool:  # a pool closed with warm-ups still queued fails them explicitly
            gate = threading.Event()
            SoftwareSession.gate = gate
            failures = []

            def prewarm():
                try:
                    pool.prewarm([samples[1]], timeout=120)
                except StateError as exc:
                    failures.append(exc)
            try:
                self.submit(pool, self.yuv444, worker=0)
                self.submit(pool, self.yuv444, worker=1)  # both workers busy: the warm-ups stay queued
                thread = threading.Thread(target=prewarm)
                thread.start()
                wait_until(lambda: pool.stats()["warm_ups_in_flight"] == 2)
                closer = threading.Thread(target=pool.close)
                closer.start()
            finally:
                gate.set()
                SoftwareSession.gate = None
            closer.join(60)
            thread.join(60)
        self.assertEqual(len(failures), 1)
        self.assertIn("closed before this warm-up ran", str(failures[0]))

    def test_prewarm_is_synchronous_and_leaves_nothing_behind(self):
        samples = [(*self.geometry(packet), packet.stream_units[0]) for packet in (self.yuv444, self.yuv420)]
        try:
            SoftwareSession.create_seconds = 0.05
            with threaded_software_pool(4) as pool:
                pool.prewarm(samples, timeout=120)
                returned = pool.stats()
                time.sleep(0.4)
                later = pool.stats()
        finally:
            SoftwareSession.create_seconds = 0.0
        # Every (worker, geometry) session exists when prewarm returns; nothing is queued, nothing is created later.
        for stats in (returned, later):
            self.assertEqual([worker["sessions_created"] for worker in stats["workers"]], [2, 2, 2, 2])
            self.assertTrue(all(worker["outstanding_bytes"] == 0 and not worker["queued"]
                                for worker in stats["workers"]))
        self.assertEqual((returned["cold_jobs"], returned["jobs"]), (0, 0))

    def test_session_charge_is_one_flat_creation_cost(self):
        # Calibrated (Step 5c jobs 4403103 / 4413408): creating a session costs the same for a 160x160 and a 3040x1632
        # canvas (A100 41-52 ms, RTX PRO 6000 30-32 ms), so the routing charge does not grow with the canvas.
        for shape in ((160, 160, "yuv444p"), (640, 352, "yuv444p"), (1568, 800, "yuv444p"), (3040, 1632, "yuv444p"),
                      (1568, 800, "yuv420p")):
            self.assertEqual(cold_session_bytes(*shape), COLD_SESSION_BYTES, shape)
        with self.assertRaises(UnsupportedError):
            cold_session_bytes(640, 352, "rgb24")
        for bad in ((0, 352), (640, 2.0)):
            with self.assertRaises(ConfigurationError):
                cold_session_bytes(*bad, "yuv444p")
        big, small = (1568, 800, "yuv444p"), (640, 352, "yuv444p")
        with threaded_software_pool(2) as pool:
            busy, idle = pool._workers
            try:
                for backlog, expected in ((COLD_SESSION_BYTES // 2, busy), (3 * COLD_SESSION_BYTES, idle)):
                    with pool._lock:
                        busy.geometries.update({big, small})
                        busy.outstanding_bytes = backlog
                    # A warm worker keeps the job while its queued work is below one session creation, for every canvas.
                    self.assertIs(pool._choose(big), expected, backlog)
                    self.assertIs(pool._choose(small), expected, backlog)
            finally:
                with pool._lock:
                    busy.geometries.clear()
                    busy.outstanding_bytes = 0

    def test_restored_values_do_not_depend_on_routing_or_prewarm(self):
        matrices, contexts, specs = node(seed=29)
        packets = (encode_node(matrices, contexts, specs, packing="node", node_id="a").packets
                   + encode_node(matrices, contexts, specs, packing="layer_group", layer_group_size=2,
                                 node_id="b").packets
                   + encode_node(matrices, contexts, specs, packing="per_matrix").packets)

        def restore(pool, prewarm=False):
            nodes = [NodeRef(f"n{index}", [prepare_packet(packet)]) for index, packet in enumerate(packets)]
            restorer = PackedRestorer(pool, arena_bytes=64 * 1024**2)
            if prewarm:
                restorer.prewarm(nodes)
            caches = {}

            def destination(node_id, key, tile):
                heads, head_dim = HEADS[key.target]
                if (node_id, key) not in caches:
                    caches[(node_id, key)] = torch.zeros((1, 180, heads, head_dim), dtype=torch.float32)
                return rk.NativeView(caches[(node_id, key)], "BTHD", token_offset=0)

            with restorer.submit(nodes) as batch:
                for layer in batch.layers:
                    batch.reconstruct(batch.entries(layer), destination, None)
            return caches

        runs = {}
        with threaded_software_pool(1) as pool:
            reference = restore(pool)
        geometries = {(prepared.width, prepared.height, prepared.pix_fmt)
                      for prepared in (prepare_packet(packet) for packet in packets)}
        self.assertGreater(len(geometries), 1)  # several canvas geometries meet a fresh pool
        with threaded_software_pool(4) as pool:
            runs["fresh_pool"] = restore(pool)
            runs["warm_pool"] = restore(pool)
            fresh = pool.stats()
        with threaded_software_pool(4, warm_all_workers=False) as pool:
            runs["explicit_no_warm_ups"] = restore(pool)
        with threaded_software_pool(4) as pool:
            runs["prewarmed"] = restore(pool, prewarm=True)
            prewarmed = pool.stats()
        self.assertGreater(fresh["cold_jobs"], 0)
        self.assertEqual((fresh["failures"], fresh["auto_warms"], prewarmed["cold_jobs"]), (0, 0, 0))
        for name, caches in runs.items():
            self.assertEqual(caches.keys(), reference.keys(), name)
            for entry, value in caches.items():
                self.assertTrue(torch.equal(value, reference[entry]), (name, entry))

    def test_a_geometry_met_without_prewarm_stays_on_the_workers_that_decoded_it(self):
        with threaded_software_pool(4) as pool:
            self.restore(pool, self.yuv444, 2)
            later = self.restore(pool, self.yuv444, 4)
            stats = pool.stats()
        self.assertEqual(sorted(set(later)), [0, 1])  # the cost of no background warm-ups: prewarm what restores meet
        self.assertEqual((stats["auto_warms"], stats["cold_jobs"]), (0, 2))
        for value in (True, 1, None, "no"):
            with self.assertRaisesRegex(ConfigurationError, "background decoder warm-ups were removed"):
                DecoderPool(gpu_id=0, gpu_uuid=UUID, workers=1, warm_all_workers=value)
        with self.assertRaises(TypeError):
            DecoderPool(gpu_id=0, gpu_uuid=UUID, workers=1, warm_idle_seconds=0.25)

    def test_prewarm_creates_every_geometry_on_every_worker(self):
        samples = [(*self.geometry(packet), packet.stream_units[0]) for packet in (self.yuv444, self.yuv420)]
        with threaded_software_pool(3) as pool:
            self.assertGreaterEqual(pool.prewarm(samples, timeout=120), 0.0)
            self.assertEqual(pool.prewarm(samples, timeout=120) >= 0.0, True)  # already warm: nothing queued
            slots = self.restore(pool, self.yuv444, 3) + self.restore(pool, self.yuv420, 3)
            stats = pool.stats()
        self.assertEqual(sorted(slots), [0, 0, 1, 1, 2, 2])
        for worker in stats["workers"]:
            self.assertEqual(sorted(worker["geometries"]), sorted(sample[:3] for sample in samples))
            self.assertEqual((worker["sessions_created"], worker["cold_jobs"], worker["auto_warms"]), (2, 0, 0))
        self.assertEqual(stats["cold_jobs"], 0)

    def test_a_job_creates_its_session_only_on_its_own_worker(self):
        a, b = (*self.geometry(self.yuv444), self.yuv444.stream_units[0]), self.geometry(self.yuv420)
        with threaded_software_pool(2, max_sessions_per_worker=1) as pool:
            pool.prewarm([a], timeout=120)
            slots = self.restore(pool, self.yuv420, 1)
            stats = pool.stats()
        target, other = stats["workers"][slots[0]], stats["workers"][1 - slots[0]]
        self.assertEqual((target["geometries"], target["evictions"]), ([b], 1))  # the job itself needs its session
        self.assertEqual((other["geometries"], other["evictions"], other["sessions_created"]), ([a[:3]], 0, 1))


class WarmupTests(unittest.TestCase):
    """One per-process decode-side warm-up (CR-12p2-11: the first restore of a fresh worker must not pay session
    creation): every canonical node canvas gets a decoder session on every worker."""

    def test_canonical_shapes_cover_every_node_canvas(self):
        from reelcache import node_packets

        for frames, height, width, channels in ((3, 30, 52, 1536), (8, 22, 40, 3072), (4, 44, 82, 4096)):
            shapes = canonical_canvas_shapes(frames, height, width, "yuv444p", channels=channels)
            ladder = node_packets.canvas_ladder(frames, height, width, "yuv444p", max_rank=channels)
            self.assertEqual(shapes, [(w, h, "yuv444p") for _, w, h in ladder])
            for rank in range(1, channels + 1, 7):
                canvas = node_packets.canonical_canvas(frames, height, width, "yuv444p", rank, channels=channels)
                self.assertIn((canvas["width"], canvas["height"], "yuv444p"), shapes, rank)
                bounded = canonical_canvas_shapes(frames, height, width, "yuv444p", channels=channels, max_rank=rank)
                self.assertEqual(bounded[-1], (canvas["width"], canvas["height"], "yuv444p"))
        self.assertEqual(len(canonical_canvas_shapes(3, 30, 52, "yuv444p", channels=1536)), 15)  # CausVid
        self.assertEqual(len(canonical_canvas_shapes(3, 30, 52, "yuv444p", channels=1536, max_rank=384)), 9)
        for bad in (0, 1537, 2.0):
            with self.assertRaises(ConfigurationError):
                canonical_canvas_shapes(3, 30, 52, "yuv444p", channels=1536, max_rank=bad)
        with self.assertRaises(UnsupportedError):
            canonical_canvas_shapes(3, 30, 52, "rgb24", channels=1536)

    def test_synthetic_units_decode_to_blank_canvases(self):
        for width, height, pix_fmt in ((224, 160, "yuv444p"), (288, 160, "yuv420p")):
            stream, offset, length, count = synthetic_unit(width, height, pix_fmt)
            self.assertEqual((offset, count), (0, 1))
            images, _ = _hevc.decode_pictures(stream[offset:offset + length], pictures=1, height=height, width=width,
                                              pix_fmt=pix_fmt)
            self.assertEqual(images.shape, (1, 3 if pix_fmt == "yuv444p" else 1, height, width))
            self.assertFalse(images.any())
        self.assertIs(synthetic_unit(224, 160, "yuv444p"), synthetic_unit(224, 160, "yuv444p"))  # once per process
        with self.assertRaises(ConfigurationError):
            synthetic_unit(8, 160, "yuv444p")

    def test_warmup_creates_every_canonical_session_on_every_worker_once(self):
        geometry = (3, 12, 20, "yuv444p", 256)  # two small canonical canvases
        shapes = canonical_canvas_shapes(*geometry[:4], channels=geometry[4])
        self.assertGreater(len(shapes), 1)
        with threaded_software_pool(3) as pool:
            restorer = PackedRestorer(pool, arena_bytes=1 << 20)
            report = restorer.warmup([geometry, geometry])
            again = restorer.warmup([geometry], timeout=60)
            stats = pool.stats()
            with self.assertRaises(CapacityError):
                pool.prewarm_shapes([(32 * (8 + n), 160, "yuv444p") for n in range(17)])
            with self.assertRaises(ConfigurationError):
                pool.prewarm_shapes([(224, 160)])
            with self.assertRaises(ConfigurationError):
                restorer.warmup([(3, 6, 10, "yuv444p")])
        self.assertEqual(report["canvas_shapes"], shapes)
        self.assertEqual((report["shapes"], report["warmed_shapes"], report["sessions_created"]),
                         (len(shapes), len(shapes), 3 * len(shapes)))
        self.assertEqual((again["warmed_shapes"], again["sessions_created"]), (0, 0))
        for worker in stats["workers"]:
            self.assertEqual(sorted(worker["geometries"]), sorted(shapes))
            self.assertEqual((worker["sessions_created"], worker["cold_jobs"], worker["jobs"]), (len(shapes), 0, 0))


class DecodeDeadlineTests(unittest.TestCase):
    """FFmpeg software-decode deadlines scale with the decoded picture samples like the encode side (Step 7)."""

    LONGCAT = 384 * 3 * 800 * 1568  # one LongCat node canvas, 4:4:4 (Step 7's encode-deadline example)
    RATE = "REELCACHE_FFMPEG_DECODE_SECONDS_PER_MEGASAMPLE"

    def test_deadline_is_size_scaled_and_configurable(self):
        with mock.patch.dict(os.environ):
            os.environ.pop(self.RATE, None)
            self.assertEqual(_hevc.decode_timeout(0), 300.0)
            self.assertEqual(_hevc.decode_timeout(180 * 3 * 288 * 544), 300.0)  # CausVid node: the floor
            self.assertAlmostEqual(_hevc.decode_timeout(self.LONGCAT), self.LONGCAT * 0.25 / 1e6)  # 361 s
            self.assertEqual(_hevc.decode_timeout(self.LONGCAT, 12.5), 12.5)
            os.environ[self.RATE] = "1.0"
            self.assertAlmostEqual(_hevc.decode_timeout(self.LONGCAT), self.LONGCAT / 1e6)
            for bad in ("0", "-1", "nan", "inf", "fast", ""):
                os.environ[self.RATE] = bad
                with self.assertRaises(ConfigurationError, msg=bad):
                    _hevc.decode_timeout(self.LONGCAT)
        for bad in (0, -1.0, float("nan"), float("inf"), True, "5"):
            with self.assertRaises(ConfigurationError, msg=repr(bad)):
                _hevc.decode_timeout(1, bad)
        with self.assertRaises(ConfigurationError):
            _hevc.decode_timeout(-1)

    def test_software_decodes_pass_the_deadline_to_ffmpeg(self):
        from reelcache.codecs import _validate_pack

        matrices, contexts, specs = node(seed=17)
        pack = _validate_pack(encode_node(matrices, contexts, specs, packing="node", node_id="d").packets[0],
                              verify=False)
        canvas, stream = pack["canvas"], pack["stream"]
        geometry = {"pictures": canvas["pictures"], "height": canvas["height"], "width": canvas["width"],
                    "pix_fmt": canvas["pix_fmt"]}
        samples = canvas["pictures"] * (3 if canvas["pix_fmt"] == "yuv444p" else 1) * canvas["height"] * canvas["width"]
        legacy = legacy_packets()[0]
        video = legacy.metadata["video"]
        streams = [legacy.payloads[f"coefficients.{index:03d}.hevc"] for index in range(video["segments"])]
        segment = video["frames"] * (3 if video["pix_fmt"] == "yuv444p" else 1) * video["image_height"] * video[
            "image_width"]
        seen, original = [], _hevc._run

        def spy(command, **kwargs):
            seen.append(kwargs.get("timeout"))
            return original(command, **kwargs)

        with mock.patch.object(_hevc, "_run", spy), mock.patch.dict(os.environ):
            # A rate that lifts both decodes above the 300 s floor proves the scaled value reaches FFmpeg.
            os.environ[self.RATE] = repr(400e6 / samples)
            images, _ = _hevc.decode_pictures(stream, **geometry)
            self.assertAlmostEqual(seen[-1], 400.0)
            os.environ[self.RATE] = repr(500e6 / segment)
            pixels, _ = _hevc.decode_pixels(streams, video)
            self.assertEqual(len(seen), 1 + video["segments"])
            for value in seen[1:]:
                self.assertAlmostEqual(value, 500.0)
            del os.environ[self.RATE]
            self.assertTrue(np.array_equal(_hevc.decode_pictures(stream, **geometry, timeout=42.0)[0], images))
            self.assertEqual(seen[-1], 42.0)
            self.assertTrue(np.array_equal(_hevc.decode_pixels(streams, video, timeout=43.0)[0], pixels))
            self.assertEqual(seen[-video["segments"]:], [43.0] * video["segments"])
            with self.assertRaises(ConfigurationError):
                _hevc.decode_pictures(stream, **geometry, timeout=0)
        with self.assertRaisesRegex(IntegrityError, "timed out"):
            _hevc.decode_pictures(stream, **geometry, timeout=1e-6)  # an expired deadline fails explicitly


# A process that exits normally with an open DecoderPool (stand-in sessions logging creation and destruction to
# argv[2]); argv[1] is the tree it must import, argv[3] the mode: "open" (exits without closing the pool),
# "unhooked" (the same with the exit hook removed: the pool before the fix), "stuck" (every decode blocks forever and
# the exit hook waits EXIT_CLOSE_SECONDS = 0.5 s).
EXIT_CHILD = r'''
import atexit, os, sys, threading
from types import SimpleNamespace
from unittest import mock

import torch

import reelcache
from reelcache import decoder_pool as dp

tree, log, mode = sys.argv[1:4]
if os.path.dirname(os.path.dirname(os.path.realpath(reelcache.__file__))) != os.path.realpath(tree):
    sys.exit(f"reelcache imported from {reelcache.__file__}, not {tree}")
gate = threading.Event()
if mode != "stuck":
    gate.set()


def note(line):
    with open(log, "a") as handle:
        handle.write(line + "\n")


class Stream:
    def wait_event(self, event):
        pass


class Session:
    def __init__(self, *, gpu_id, width, height, pix_fmt, api=None):
        self.geometry = (width, height, pix_fmt)
        self.stream = Stream()
        self.setup_seconds = {"caps": 0.0, "stream": 0.0, "create_decoder": 0.0}
        self.stats = {"first_picture_seconds": None}
        self.owner = threading.current_thread().name
        note(f"created {self.owner} {width}")

    def decode(self, units, canvas, first=0):
        gate.wait()
        self.stats["first_picture_seconds"] = 0.0

    def close(self):
        note(f"closed {threading.current_thread().name} {self.owner} {self.geometry[0]}")


class Event:
    def query(self):
        return True


mock.patch("torch.cuda.set_device").start()
mock.patch("reelcache.decoder_pool.CanvasSession", Session).start()
mock.patch.object(torch.Tensor, "record_stream", lambda tensor, stream: None).start()
uuid = "GPU-bf621a31-0ece-332e-d43d-d20e0d4a42be"
with mock.patch("torch.cuda.device_count", return_value=1), \
        mock.patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(uuid=uuid[4:])):
    pool = dp.DecoderPool(gpu_id=0, gpu_uuid=uuid, workers=2, api=object())
pool.device = torch.device("cpu")
if mode == "unhooked":
    atexit.unregister(dp.close_all_pools)
if mode == "stuck":
    dp.EXIT_CLOSE_SECONDS = 0.5
tickets = [pool.submit([(b"\0" * 8, 0, 8, 1)], torch.zeros((1, 3, 32, width), dtype=torch.uint8), worker=worker,
                       after=[Event()]) for width in (64, 96) for worker in (0, 1)]
if mode != "stuck":
    for ticket in tickets:
        ticket.synchronize(60)
print("EXITING", len(dp.open_pools()), flush=True)
'''


class ExitCloseTests(unittest.TestCase):
    """Pools still open at interpreter exit are closed by the exit hook, sessions destroyed on their own workers
    (never left to libnvcuvid's exit handler, which deadlocked on RTX: Step 11 job 4419034)."""

    def setUp(self):
        self.closed = []

        def close(session):
            self.closed.append((threading.current_thread().name, session.geometry))

        patches = [mock.patch.object(SoftwareSession, "close", close),
                   mock.patch.object(SoftwareSession, "decode", self.decode)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    @staticmethod
    def decode(session, units, canvas, first=0):
        session.stats["first_picture_seconds"] = 0.0

    @staticmethod
    def fill(pool, widths=(64, 96)):
        """One job per worker and canvas width; returns their tickets (the sessions exist once they end)."""
        return [pool.submit([(b"\0" * 8, 0, 8, 1)], torch.zeros((1, 3, 32, width), dtype=torch.uint8), worker=worker,
                            after=[FakeEvent()]) for width in widths for worker in range(pool.workers)]

    def test_close_all_pools_destroys_every_session_on_its_worker(self):
        with threaded_software_pool(workers=2) as pool:
            for ticket in self.fill(pool):
                ticket.synchronize(30)
            self.assertIn(pool, dp.open_pools())
            result = dp.close_all_pools(timeout=30)
            self.assertGreaterEqual(result["closed"], 1)
            self.assertNotIn(pool, dp.open_pools())
            self.assertTrue(pool.stats()["closed"])
            self.assertFalse(any(worker.thread.is_alive() for worker in pool._workers))
            self.assertEqual(pool.stats()["sessions_created"], 4)
            self.assertEqual(sorted(self.closed), sorted(
                (f"reelcache-nvdec-{index}", (width, 32, "yuv444p")) for index in range(2) for width in (64, 96)))
            self.assertTrue(pool.close())  # idempotent: nothing left to wait for
        self.assertEqual(len(self.closed), 4)

    def test_the_first_pool_registers_the_exit_hook_once(self):
        with mock.patch.object(dp, "_EXIT_HOOK", []), mock.patch.object(dp.atexit, "register") as register:
            with threaded_software_pool(workers=1), threaded_software_pool(workers=1):
                pass
        register.assert_called_once_with(dp.close_all_pools)

    def test_pools_of_another_process_are_skipped(self):
        with threaded_software_pool(workers=1) as pool:
            pool._owner_pid = -1  # a pool inherited across fork: its workers live in the parent
            try:
                self.assertNotIn(pool, dp.open_pools())
                dp.close_all_pools(timeout=5)
                self.assertFalse(pool.stats()["closed"])
            finally:
                pool._owner_pid = os.getpid()

    def test_close_timeout_bounds_the_wait_for_a_stuck_decode(self):
        release = threading.Event()

        def blocked(session, units, canvas, first=0):
            if not release.wait(60):
                raise TimeoutError("the test never released the decode")
            session.stats["first_picture_seconds"] = 0.0

        with mock.patch.object(SoftwareSession, "decode", blocked), threaded_software_pool(workers=1) as pool:
            ticket, = self.fill(pool, widths=(64,))
            started = time.monotonic()
            self.assertFalse(pool.close(timeout=0.2))
            self.assertLess(time.monotonic() - started, 10.0)
            self.assertIn(pool, dp.open_pools())  # still listed: the exit hook waits for it again
            self.assertEqual(self.closed, [])
            release.set()
            self.assertTrue(pool.close(timeout=30))
            ticket.synchronize(30)
            self.assertNotIn(pool, dp.open_pools())
            self.assertEqual(self.closed, [("reelcache-nvdec-0", (64, 32, "yuv444p"))])

    def test_context_exit_propagates_and_closes(self):
        with threaded_software_pool(workers=1) as pool:
            with self.assertRaises(ValueError):
                with pool:
                    raise ValueError("propagates")
            self.assertTrue(pool.stats()["closed"])
            self.assertNotIn(pool, dp.open_pools())

    def run_child(self, mode):
        tree = Path(reelcache.__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as scratch:  # neutral working directory: imports come from PYTHONPATH
            log = Path(scratch) / "sessions.log"
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(
                part for part in (str(tree), os.environ.get("PYTHONPATH", "")) if part))
            started = time.monotonic()
            result = subprocess.run([sys.executable, "-c", EXIT_CHILD, str(tree), str(log), mode], cwd=scratch,
                                    env=env, capture_output=True, text=True, timeout=180)
            seconds = time.monotonic() - started
            lines = [line.split() for line in log.read_text().splitlines()] if log.exists() else []
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        self.assertIn("EXITING 1", result.stdout)
        return lines, seconds

    def test_interpreter_exit_closes_open_pools_in_a_child_process(self):
        lines, _ = self.run_child("open")
        created = sorted((owner, width) for kind, owner, width in (line for line in lines if line[0] == "created"))
        closed = [line for line in lines if line[0] == "closed"]
        self.assertEqual(created, [(f"reelcache-nvdec-{index}", width) for index in range(2) for width in ("64", "96")])
        self.assertEqual(sorted((owner, width) for _, thread, owner, width in closed), created)
        self.assertTrue(all(thread == owner for _, thread, owner, _ in closed))  # each on its own worker thread
        # The pool before the fix: nothing destroyed the sessions before the native exit handlers ran.
        lines, _ = self.run_child("unhooked")
        self.assertEqual(sum(line[0] == "created" for line in lines), 4)
        self.assertFalse([line for line in lines if line[0] == "closed"])
        # A decode that never ends cannot hang the exit: the hook gives up after EXIT_CLOSE_SECONDS.
        lines, seconds = self.run_child("stuck")
        self.assertEqual(sum(line[0] == "created" for line in lines), 2)
        self.assertFalse([line for line in lines if line[0] == "closed"])
        self.assertLess(seconds, 120.0)


class CanvasTrimTests(unittest.TestCase):
    """Device-memory pressure trim of the decoded-canvas arena (``trim_device_cache(reason) -> bytes_freed``).

    Only free canvases are freed: released, every consumer event complete (queried, never waited on). Leased
    canvases and canvases a reconstruction may still read are never touched. The next restore re-allocates."""

    @classmethod
    def setUpClass(cls):
        RestorerTests.setUpClass.__func__(cls)

    restorer = RestorerTests.restorer
    nodes = RestorerTests.nodes

    @staticmethod
    def sources(batch):
        return {entry: rk.unpack_pictures(batch.source(entry)).numpy().copy() for entry in batch.entries()}

    def test_trim_frees_only_canvases_no_restore_uses(self):
        arena = CanvasArena("cpu", 10_000)
        leased = arena.acquire((1, 1, 10, 100))  # a restore decodes into it
        consumed = arena.acquire((1, 1, 10, 100))  # released, a reconstruction still reads it
        done = [arena.acquire((1, 1, 10, 100)), arena.acquire((1, 1, 20, 50))]  # released, consumers done
        reading = FakeEvent(False)
        arena.release(consumed, after=[FakeEvent(True), reading])
        arena.release(done[0], after=FakeEvent(True))
        arena.release(done[1])
        references = [weakref.ref(canvas) for canvas in done]
        del done
        self.assertEqual(arena.report()["free_bytes"], 2000)
        self.assertEqual(arena.trim("request start: free 0.9 GiB < reserve 2 GiB"), 2000)
        self.assertTrue(all(reference() is None for reference in references))  # the arena held the last references
        report = arena.report()
        self.assertEqual((report["used_bytes"], report["free"], report["free_bytes"], report["retiring"]),
                         (2000, 0, 0, 1))
        self.assertEqual((report["trims"], report["trimmed_canvases"], report["trimmed_bytes"]), (1, 2, 2000))
        self.assertEqual({key: report["last_trim"][key] for key in ("reason", "canvases", "bytes", "kept_bytes")},
                         {"reason": "request start: free 0.9 GiB < reserve 2 GiB", "canvases": 2, "bytes": 2000,
                          "kept_bytes": 2000})
        self.assertEqual(arena.trim("again"), 0)  # nothing idle: one canvas leased, one still read
        reading.done = True
        self.assertEqual(arena.trim("decode"), 1000)  # its reader finished: now trimmed
        arena.release(leased, after=FakeEvent(True))  # the lease survived every trim
        self.assertIs(arena.acquire((1, 1, 10, 100)), leased)
        self.assertEqual(arena.report()["used_bytes"], 1000)

    def test_trim_never_waits_on_a_consumer_event(self):
        class Strict(FakeEvent):
            def synchronize(self):
                raise AssertionError("trim waited on a consumer event")

        arena = CanvasArena("cpu", 4000)
        canvas = arena.acquire((1, 1, 10, 100))
        arena.release(canvas, after=Strict(False))
        started = time.monotonic()
        self.assertEqual(arena.trim("pressure"), 0)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(arena.report()["retiring"], 1)

    def test_a_trim_during_a_restore_leaves_its_canvases_alone(self):
        restorer = self.restorer()
        earlier = restorer.submit(self.nodes())
        earlier.close()
        idle = restorer.arena.report()["free_bytes"]
        self.assertGreater(idle, 0)
        batch = restorer.submit(self.nodes())  # leases every free canvas again
        self.assertEqual(restorer.trim_device_cache("decode hook"), 0)
        for (node_id, key), pixels in self.sources(batch).items():
            self.assertTrue(np.array_equal(pixels, self.coefficients[(node_id, key)][2]), (node_id, key))
        batch.close()
        self.assertEqual(restorer.trim_device_cache("idle"), idle)
        self.assertEqual(restorer.arena.report()["used_bytes"], 0)

    def test_restores_after_a_trim_reallocate_and_decode_identically(self):
        restorer = self.restorer()
        batch = restorer.submit(self.nodes())
        canvases, first = len(batch.jobs), self.sources(batch)
        batch.close()
        report = restorer.arena.report()
        self.assertEqual((report["allocations"], report["free"]), (canvases, canvases))
        self.assertEqual(restorer.trim_device_cache("request start"), report["free_bytes"])
        batch = restorer.submit(self.nodes())
        second = self.sources(batch)
        batch.close()
        report = restorer.arena.report()
        self.assertEqual((report["allocations"], report["reuses"]), (2 * canvases, 0))  # re-allocated after the trim
        self.assertEqual(first.keys(), second.keys())
        self.assertTrue(all(np.array_equal(first[entry], second[entry]) for entry in first))
        restorer.submit(self.nodes()).close()  # no trim: every canvas is reused
        self.assertEqual(restorer.arena.report()["reuses"], canvases)

    def test_the_pool_registers_its_trim_and_unregisters_on_close(self):
        from reelcache import device_pressure

        registry = device_pressure.DeviceTrimRegistry()
        with mock.patch.object(dp, "register_device_trim", registry.register):
            with threaded_software_pool(workers=1):
                self.assertEqual(registry.registered(), [{"name": "decoder_pool:0", "order": 10, "device": 0}])
            self.assertEqual(registry.registered(), [])
        with threaded_software_pool(workers=1) as pool:  # the process-wide registry by default
            self.assertIn(pool._trim_handle.key, device_pressure.REGISTRY._entries)
            pool.close()
            self.assertNotIn(pool._trim_handle.key, device_pressure.REGISTRY._entries)

    def test_registry_trims_idle_canvases_of_every_arena_the_pool_decodes_into(self):
        from reelcache import device_pressure

        registry = device_pressure.DeviceTrimRegistry()
        unit = [(b"\0" * 8, 0, 8, 1)]
        with mock.patch.object(dp, "register_device_trim", registry.register), \
                mock.patch.object(SoftwareSession, "decode", ExitCloseTests.decode), \
                threaded_software_pool(workers=2) as pool:
            arenas = [CanvasArena("cpu", 1 << 20), CanvasArena("cpu", 1 << 20)]  # e.g. a restorer's and a CP shard's
            leased = arenas[0].acquire((1, 3, 32, 64))  # a restore still decodes into it
            pending = pool.submit(unit, leased, after=None)
            for arena in arenas:  # an idle canvas per arena: decoded, released, readers done
                canvas = arena.acquire((1, 3, 32, 96))
                pool.submit(unit, canvas, after=None).synchronize(30)
                arena.release(canvas, after=FakeEvent(True))
            pending.synchronize(30)
            result = registry.trim("request_start", device=0, empty_cache=lambda: None)
            self.assertEqual([record["name"] for record in result["trims"]], ["decoder_pool:0"])
            self.assertNotIn("error", result["trims"][0])
            self.assertEqual(result["bytes"], 2 * 3 * 32 * 96)
            self.assertEqual((pool.stats()["trims"], pool.stats()["trimmed_bytes"]), (1, 2 * 3 * 32 * 96))
            self.assertEqual(arenas[0].report()["used_bytes"], 3 * 32 * 64)  # the leased canvas stays
            self.assertEqual(arenas[1].report()["used_bytes"], 0)
            self.assertEqual(registry.trim("decode", device=1, empty_cache=lambda: None)["trims"], [])  # other GPU
            arenas[0].release(leased, after=FakeEvent(True))
            self.assertEqual(registry.trim("oom", device=0, empty_cache=lambda: None)["bytes"], 3 * 32 * 64)

    def test_concurrent_restores_and_trims_keep_the_accounting(self):
        arena = CanvasArena("cpu", 64 * 1000, wait_seconds=5)
        errors, stop = [], threading.Event()

        def restores():
            try:
                for _ in range(300):
                    held = [arena.acquire((1, 1, 10, 100)) for _ in range(4)]
                    for canvas in held:
                        arena.release(canvas, after=FakeEvent(True))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                stop.set()

        worker = threading.Thread(target=restores)
        worker.start()
        trimmed = 0
        while not stop.is_set():
            trimmed += arena.trim("pressure")
        worker.join()
        self.assertEqual(errors, [])
        report = arena.report()
        self.assertEqual(report["used_bytes"], report["free_bytes"])  # nothing leased or retiring is left
        self.assertEqual(report["trimmed_bytes"], trimmed)
        self.assertEqual(report["allocations"] * 1000, trimmed + report["used_bytes"])
        self.assertEqual(arena.trim("final") + trimmed, report["allocations"] * 1000)


if __name__ == "__main__":
    unittest.main()
