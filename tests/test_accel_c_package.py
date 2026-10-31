"""Accel track C (C1 bytes-identical archival package + config): each opt-in lever stores exactly the flag-off
bytes (identical node refs and object ids), and defaults leave the base path untouched."""

from __future__ import annotations

import hashlib
import threading
import unittest
from unittest import mock

import numpy as np

from reelcache import accel_c, node_packets as npk
from reelcache.archive_pipeline import ArchivePipeline
from reelcache.codecs import _json_bytes
from reelcache.errors import ConfigurationError, IntegrityError
from reelcache.integrations.sglang.config import ReelCacheRuntimeConfig
from reelcache.lineage_commit import ArchiveOptions
from reelcache.run_telemetry import split_run_telemetry
from reelcache.types import EncodedTensor
from tests.step6_support import RuntimeFactory, chunk, method, registry
from tests.test_step6_node_packets import NodePackEncodeTests

ALL_CPU = accel_c.AccelOptions(validation_decoder="nvdec", hash_once=True, seal_prepare_after_publish=True,
                               gil_fastpaths=True)


def fake_nvdec(calls):
    """A CPU stand-in for the NVDEC validation decode: FFmpeg's pictures (normative decoding), counted."""

    def decode(stream, units, *, pictures, height, width, pix_fmt, pool=None, gpu_id=None):
        calls.append((len(stream), pictures, len(units), pool))
        return npk._validation_decode(stream, pictures=pictures, height=height, width=width, pix_fmt=pix_fmt,
                                      threads=1)
    return decode


class ConfigTests(unittest.TestCase):
    def test_defaults_are_the_base(self):
        config = ReelCacheRuntimeConfig(store_root="/tmp/accel-c-config")
        self.assertEqual(accel_c.AccelOptions.from_config(config), accel_c.OFF)
        self.assertFalse(accel_c.AccelOptions.from_config(config).active)

    def test_defaults_stay_out_of_the_records(self):
        config = ReelCacheRuntimeConfig(store_root="/tmp/accel-c-config")
        data = config.to_dict()
        for name in accel_c.FIELDS + ("pipelined_restore", "verify_bases"):
            self.assertNotIn(name, data)  # the W1.4.1 / W1.2 record freezes compare describe.config field for field
        self.assertEqual(ReelCacheRuntimeConfig.from_dict(data), config)
        on = ReelCacheRuntimeConfig(store_root="/tmp/accel-c-config", hash_once=True, validation_decoder="nvdec")
        self.assertEqual((on.to_dict()["hash_once"], on.to_dict()["validation_decoder"]), (True, "nvdec"))
        self.assertEqual(ReelCacheRuntimeConfig.from_dict(on.to_dict()), on)

    def test_fields_validate(self):
        config = ReelCacheRuntimeConfig.from_dict({"store_root": "/tmp/x", "validation_decoder": "nvdec",
                                                "hash_once": True, "seal_prepare_after_publish": True,
                                                "nvenc_device_input": True, "gil_fastpaths": True,
                                                "native_ext": True})
        self.assertEqual(accel_c.AccelOptions.from_config(config).to_dict(), {
            "validation_decoder": "nvdec", "hash_once": True, "seal_prepare_after_publish": True,
            "nvenc_device_input": True, "gil_fastpaths": True, "native_ext": True})
        for bad in ({"validation_decoder": "cpu"}, {"hash_once": 1}, {"gil_fastpaths": "yes"},
                    {"validation_decoder": "nvdec", "validate_decode": False}, {"native_ext": None}):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                ReelCacheRuntimeConfig.from_dict({"store_root": "/tmp/x", **bad})


class EncodeIdentityTests(unittest.TestCase):
    def setUp(self):
        helper = NodePackEncodeTests()
        helper.setUp()
        self.finalized = helper.finalized

    def pack(self, options, **kwargs):
        with accel_c.scope(options):
            return npk.encode_node_pack(self.finalized(**kwargs), threads=1, validate_decode=True)

    def assertSamePack(self, left, right):
        self.assertEqual(_json_bytes(left.metadata), _json_bytes(right.metadata))
        self.assertEqual(dict(left.payloads), dict(right.payloads))
        self.assertEqual(split_run_telemetry(dict(left.metrics))[0], split_run_telemetry(dict(right.metrics))[0])

    def test_each_lever_encodes_the_base_pack(self):
        base = self.pack(accel_c.OFF)
        calls = []
        with mock.patch.object(accel_c, "nvdec_validation_decode", fake_nvdec(calls)):
            for options in (accel_c.AccelOptions(hash_once=True), accel_c.AccelOptions(gil_fastpaths=True),
                            accel_c.AccelOptions(validation_decoder="nvdec"), ALL_CPU):
                with self.subTest(options=options):
                    self.assertSamePack(self.pack(options), base)
        self.assertEqual(len(calls), 2)  # the two nvdec arms dispatched to the (fake) NVDEC decoder
        self.assertTrue(all(tile["decoded_coefficients_sha256"] for tile in base.metadata["tiles"]))

    def test_post_codec_error_unchanged(self):
        base = self.pack(accel_c.OFF, keep=True)
        with mock.patch.object(accel_c, "nvdec_validation_decode", fake_nvdec([])):
            fast = self.pack(ALL_CPU, keep=True)
        self.assertEqual(base.metrics["post_codec_relative_error"], fast.metrics["post_codec_relative_error"])

    def test_hash_once_reuses_only_the_hashed_objects(self):
        packet = self.pack(accel_c.AccelOptions(hash_once=True))
        names = [tile["basis"]["payload"] for tile in packet.metadata["tiles"]]
        with accel_c.scope(accel_c.AccelOptions(hash_once=True)):
            before = accel_c.stats()["hash_once_hits"]
            metadata, payloads, bases = npk.detach_pack_bases(packet)
            self.assertEqual(accel_c.stats()["hash_once_hits"] - before, len(names))
            name = names[0]
            twin = bytes(bytearray(packet.payloads[name]))  # equal bytes, another object: hashed again
            self.assertIsNone(accel_c.known_digest(packet, name, twin))
        with accel_c.scope(accel_c.OFF):  # the record is ignored with the lever off
            self.assertIsNone(accel_c.known_digest(packet, name, packet.payloads[name]))
            self.assertEqual(npk.detach_pack_bases(packet)[2], bases)
        # A pack rebuilt with a substituted basis object of the same length is still caught.
        forged = dict(packet.payloads)
        forged[names[1]] = bytes(len(forged[names[1]]))
        rebuilt = EncodedTensor(packet.metadata, forged, None, packet.metrics)
        with accel_c.scope(accel_c.AccelOptions(hash_once=True)), self.assertRaises(IntegrityError):
            npk.detach_pack_bases(rebuilt)

    def test_store_reuses_the_digest_of_the_same_object_only(self):
        runtime = RuntimeFactory(self)()
        packet = self.pack(accel_c.AccelOptions(hash_once=True))
        stream = packet.payloads["stream.hevc"]
        with accel_c.scope(accel_c.AccelOptions(hash_once=True)), accel_c.known_payload_digests(packet):
            before = accel_c.stats()["hash_once_hits"]
            ref = runtime.store.put(stream)
            self.assertEqual(accel_c.stats()["hash_once_hits"] - before, 1)
            other = runtime.store.put(bytes(bytearray(stream[:100])))
            self.assertEqual(accel_c.stats()["hash_once_hits"] - before, 1)
        self.assertEqual(ref, "blob:" + hashlib.sha256(stream).hexdigest())
        self.assertEqual(other, "blob:" + hashlib.sha256(stream[:100]).hexdigest())
        self.assertIsNone(accel_c.store_digest(stream))  # outside the block


class SealRecorder:
    """A seal-time restorer stand-in: records each adoption and how many nodes were published at that time."""

    def __init__(self):
        self.calls = []
        self.pipeline = None
        self.lock = threading.Lock()

    def adopt(self, packet_ref, encoded):
        published = self.pipeline.stats()["published"] if self.pipeline is not None else None
        with self.lock:
            self.calls.append((packet_ref, published, sorted(encoded.payloads)))


def wait_for(predicate, timeout=60.0):
    """Seal-time preparation after publication runs on the publishing worker after drain() returned: poll."""
    import time

    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.01)


class PipelineIdentityTests(unittest.TestCase):
    def setUp(self):
        self.runtimes = RuntimeFactory(self)

    def archive(self, accel, count=3):
        runtime = self.runtimes("accel-c")
        runtime.create_lineage("main")
        recorder = SealRecorder()
        pipeline = ArchivePipeline(runtime, method=method(1), registry=registry(),
                                   options=ArchiveOptions(encode_threads=1), encode_workers=2, max_source_chunks=2,
                                   max_unpublished=3, restorer=recorder, package_accel=accel)
        recorder.pipeline = pipeline
        try:
            tickets = []
            for index in range(count):
                matrices, contexts = chunk(index)
                tickets.append(pipeline.submit("main", f"chunk-{index:03d}", matrices, contexts))
            pipeline.drain("main", timeout=300)
            refs = [ticket.result(timeout=60) for ticket in tickets]
            wait_for(lambda: len(recorder.calls) == count)
            stats = pipeline.stats()
        finally:
            pipeline.close(drain=False, timeout=60)
        objects = sorted(path.name for kind in ("blob", "basis", "manifest")
                         for path in (runtime.store.root / "objects" / kind).iterdir())
        return refs, objects, recorder.calls, stats

    def test_all_cpu_levers_archive_the_same_objects(self):
        base_refs, base_objects, base_seals, base_stats = self.archive(None)
        calls = []
        with mock.patch.object(accel_c, "nvdec_validation_decode", fake_nvdec(calls)):
            refs, objects, seals, stats = self.archive(ALL_CPU)
        self.assertEqual(refs, base_refs)
        self.assertEqual(objects, base_objects)
        self.assertEqual(len(calls), 3)
        self.assertNotIn("accel_c", base_stats)
        self.assertEqual(stats["accel_c"]["options"], ALL_CPU.to_dict())
        self.assertGreater(stats["accel_c"]["hash_once_hits"], 0)
        # The same packs are adopted; after their publication instead of before it.
        self.assertEqual([(ref, payloads) for ref, _, payloads in seals],
                         [(ref, payloads) for ref, _, payloads in base_seals])
        self.assertTrue(all(published <= index for index, (_, published, _) in enumerate(base_seals)))
        self.assertTrue(all(published >= index + 1 for index, (_, published, _) in enumerate(seals)))
        self.assertEqual(stats["adopt"]["calls"], 3)
        self.assertEqual(stats["adopt"]["failures"], 0)

    def test_seal_failures_after_publication_are_recorded_not_raised(self):
        runtime = self.runtimes("accel-c-fail")
        runtime.create_lineage("main")

        class Failing:
            def adopt(self, packet_ref, encoded):
                raise IntegrityError("injected seal failure")

        pipeline = ArchivePipeline(runtime, method=method(0), registry=registry(),
                                   options=ArchiveOptions(encode_threads=1), restorer=Failing(),
                                   package_accel=accel_c.AccelOptions(seal_prepare_after_publish=True))
        try:
            matrices, contexts = chunk(0)
            ref = pipeline.submit("main", "chunk-000", matrices, contexts).result(timeout=300)
            pipeline.drain("main", timeout=60)
            wait_for(lambda: pipeline.stats()["adopt"]["failures"] == 1)
            stats = pipeline.stats()
        finally:
            pipeline.close(drain=False, timeout=60)
        self.assertEqual(runtime.lineage_head("main"), ref)
        self.assertEqual(stats["adopt"]["failures"], 1)
        self.assertIn("injected seal failure", stats["adopt"]["last_failure"])

    def test_pipeline_rejects_a_non_options_accel(self):
        runtime = self.runtimes("accel-c-bad")
        with self.assertRaises(ConfigurationError):
            ArchivePipeline(runtime, method=method(0), registry=registry(), package_accel={"hash_once": True})
        pipeline = ArchivePipeline(runtime, method=method(0), registry=registry(), package_accel=accel_c.OFF)
        self.addCleanup(pipeline.close, drain=False, timeout=60)
        self.assertIsNone(pipeline.package_accel)


class DeviceInputTests(unittest.TestCase):
    """C1d plumbing on CPU (the NVENC device path itself is checked on the GPU: identical stream bytes)."""

    def canvas(self, device="cuda", pix_fmt="yuv444p", preset="nvenc-p1-hq"):
        from types import SimpleNamespace
        pictures = SimpleNamespace(device=SimpleNamespace(type=device), shape=(6, 3, 32, 32))
        return SimpleNamespace(device_pictures=pictures, codec={"pix_fmt": pix_fmt, "preset": preset})

    def test_predicate(self):
        on = accel_c.AccelOptions(nvenc_device_input=True)
        with accel_c.scope(on):
            self.assertTrue(accel_c.device_input(self.canvas()))
            self.assertFalse(accel_c.device_input(self.canvas(device="cpu")))
            self.assertFalse(accel_c.device_input(self.canvas(pix_fmt="yuv420p")))
            self.assertFalse(accel_c.device_input(self.canvas(preset="ultrafast")))
        self.assertFalse(accel_c.device_input(self.canvas()))  # flag off

    def test_wait_keeps_a_device_canvas(self):
        import torch

        class Event:
            synced = 0

            def synchronize(self):
                Event.synced += 1

        pictures = torch.zeros((2, 3, 16, 16), dtype=torch.uint8)
        node = npk.NodeCanvas("n", {}, {"pix_fmt": "yuv444p"}, {"pictures": 2}, [], pictures)
        node.event, node.keep_device = Event(), True
        node.wait()
        self.assertIs(node.device_pictures, pictures)
        self.assertEqual(Event.synced, 1)
        node.keep_device = False
        node.wait()
        self.assertIsNone(node.device_pictures)
        codec = {"kind": "svd_hevc", "pix_fmt": "yuv444p", "preset": "ultrafast", "qp": 8}
        unset = npk.NodeCanvas("n", {}, codec, {"pictures": 2}, [], pictures)
        with self.assertRaises(ConfigurationError):  # neither host pictures nor a kept device canvas (no encoder)
            unset.wait()
            npk.encode_node_pack(unset, validate_decode=False)


if __name__ == "__main__":
    unittest.main()
