import gc
import tempfile
import threading
import unittest
import weakref
from unittest import mock

import torch

from reelcache.codecs import decode_tensor, encode_tensor
from reelcache.distributed import DistributedOwner, PeerFailure
from reelcache.errors import ConfigurationError, IntegrityError
from reelcache.execution import BoundedExecutor
from reelcache.manager import KVRuntime
from reelcache.store import ObjectStore
from reelcache.telemetry import RuntimeTelemetry, STAGES, current_telemetry, span
from reelcache.types import CodecSpec, EncodedTensor, MatrixKey, ModelIdentity, TensorContext
from tests.test_codecs import context


class TelemetryTests(unittest.TestCase):
    def test_disabled_scope_never_reads_clock_and_nested_capture_restores(self):
        self.assertIsNone(current_telemetry())
        with mock.patch("reelcache.telemetry.time.perf_counter", side_effect=AssertionError("disabled")):
            with span("decode.validate"):
                pass
        first, second = RuntimeTelemetry(), RuntimeTelemetry()
        with first.capture():
            with second.capture(), span("decode.validate"):
                pass
            self.assertIs(current_telemetry(), first)
            with span("decode.basis"):
                pass
        self.assertIsNone(current_telemetry())
        self.assertEqual(first.snapshot()["stages"]["decode.validate"]["calls"], 0)
        self.assertEqual(second.snapshot()["stages"]["decode.validate"]["calls"], 1)

    def test_failures_bytes_inflight_and_detached_bounded_snapshot(self):
        metrics = RuntimeTelemetry()
        with metrics.capture(), mock.patch("reelcache.telemetry.time.perf_counter", side_effect=[10., 12.]):
            with self.assertRaisesRegex(ValueError, "original failure"):
                with span("store.hash", nbytes=7):
                    live = metrics.snapshot()["stages"]["store.hash"]
                    self.assertEqual(live["inflight"], 1)
                    raise ValueError("original failure")
        row = metrics.snapshot()["stages"]["store.hash"]
        self.assertEqual(row, {
            "calls": 1, "completed": 0, "failed": 1, "inflight": 0,
            "wall_seconds": 2., "attempted_bytes": 7, "completed_bytes": 0,
        })
        row["calls"] = 999
        self.assertEqual(metrics.snapshot()["stages"]["store.hash"]["calls"], 1)
        with metrics.capture():
            for args in (("unbounded-new-label", 0), ("store.read", -1), ("store.read", True)):
                with self.assertRaises(ConfigurationError), span(args[0], nbytes=args[1]):
                    pass
        self.assertEqual(len(metrics.snapshot()["stages"]), len(STAGES))

    def test_executor_carries_capture_only_for_its_own_job(self):
        metrics = RuntimeTelemetry()

        def work():
            with span("decode.validate"):
                return current_telemetry() is metrics

        with BoundedExecutor(max_workers=1) as executor:
            with metrics.capture():
                job = executor.submit(work)
            self.assertTrue(job.result(timeout=5))
            self.assertFalse(executor.submit(work).result(timeout=5))
        self.assertEqual(metrics.snapshot()["stages"]["decode.validate"]["calls"], 1)

    def test_concurrent_spans_are_thread_safe_and_failures_do_not_leak_capture(self):
        metrics = RuntimeTelemetry()
        barrier = threading.Barrier(2)

        def work(fail):
            with span("decode.reconstruct", nbytes=8):
                barrier.wait(timeout=5)
                if fail:
                    raise ValueError("worker")

        with BoundedExecutor(max_workers=2, max_pending=2) as executor:
            with metrics.capture():
                good = executor.submit(work, False)
                bad = executor.submit(work, True)
            good.result(timeout=5)
            with self.assertRaisesRegex(ValueError, "worker"):
                bad.result(timeout=5)
            self.assertIsNone(executor.submit(current_telemetry).result(timeout=5))
        row = metrics.snapshot()["stages"]["decode.reconstruct"]
        self.assertEqual((row["calls"], row["completed"], row["failed"], row["inflight"]), (2, 1, 1, 0))
        self.assertEqual((row["attempted_bytes"], row["completed_bytes"]), (16, 8))

    def test_real_raw_svd_hevc_decode_is_unchanged_and_counts_actual_work(self):
        generator = torch.Generator().manual_seed(619)
        source = torch.randn(24, 4, generator=generator)
        rng = torch.random.get_rng_state()
        for kind in ("raw", "svd", "svd_hevc"):
            with self.subTest(kind=kind):
                packet = encode_tensor(source, context(), CodecSpec(kind=kind, rank=None if kind == "raw" else 3))
                reference = decode_tensor(packet, columns=slice(0, 2))
                metrics = RuntimeTelemetry()
                with metrics.capture():
                    actual = decode_tensor(packet, columns=slice(0, 2))
                self.assertTrue(torch.equal(reference, actual))
                rows = metrics.snapshot()["stages"]
                self.assertEqual(rows["decode.validate"]["completed"], 1)
                self.assertEqual(rows["decode.reconstruct"]["completed"], 1)
                self.assertEqual(rows["decode.basis"]["completed"], int(kind != "raw"))
                self.assertEqual(rows["decode.hevc"]["completed"], int(kind == "svd_hevc"))
                if kind == "svd_hevc":
                    size = sum(len(v) for k, v in packet.payloads.items() if k.endswith(".hevc"))
                    self.assertEqual(rows["decode.hevc"]["completed_bytes"], size)
                self.assertFalse(metrics.snapshot()["cuda_synchronized"])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_invalid_packet_and_failed_codec_are_not_recorded_as_success(self):
        source = torch.ones(24, 4)
        packet = encode_tensor(source, context(), CodecSpec())
        corrupt = EncodedTensor(packet.metadata, {"tensor.bin": b"bad"})
        metrics = RuntimeTelemetry()
        with metrics.capture(), self.assertRaises(IntegrityError):
            decode_tensor(corrupt, verify_packet=True)
        self.assertEqual(metrics.snapshot()["stages"]["decode.validate"]["failed"], 1)
        packet = encode_tensor(source, context(), CodecSpec(kind="svd_hevc", rank=1))
        metrics = RuntimeTelemetry()
        with metrics.capture(), mock.patch("reelcache._hevc.decode_pixels", side_effect=RuntimeError("decode")):
            with self.assertRaisesRegex(RuntimeError, "decode"):
                decode_tensor(packet)
        self.assertEqual(metrics.snapshot()["stages"]["decode.hevc"]["failed"], 1)
        self.assertEqual(metrics.snapshot()["stages"]["decode.reconstruct"]["calls"], 0)

    def test_store_read_and_hash_counts_include_corruption(self):
        with tempfile.TemporaryDirectory() as directory, ObjectStore(directory, verify_reads=True) as store:
            oid = store.put(b"abc")
            metrics = RuntimeTelemetry()
            with metrics.capture():
                self.assertEqual(store.get(oid), b"abc")
            rows = metrics.snapshot()["stages"]
            self.assertEqual(rows["store.read"]["completed_bytes"], 3)
            self.assertEqual(rows["store.hash"]["completed_bytes"], 3)
            store.path_for(oid).write_bytes(b"xyz")
            with metrics.capture(), self.assertRaises(IntegrityError):
                store.get(oid)
            self.assertEqual(metrics.snapshot()["stages"]["store.hash"]["failed"], 1)

    def test_runtime_sync_and_executor_seal_restore_and_release(self):
        identity = ModelIdentity("telemetry", "test-v1", 1, 4, "canonical", "float32")
        tensors = {MatrixKey(0, t): torch.ones(24, 4) for t in ("K", "V")}
        contexts = {
            k: TensorContext(identity, k, 0, 2, 3, 4, "pre_rope" if k.target == "K" else "none")
            for k in tensors
        }
        for use_executor in (False, True):
            with self.subTest(executor=use_executor), tempfile.TemporaryDirectory() as directory:
                with BoundedExecutor() as executor, KVRuntime(
                    directory, identity, executor=executor if use_executor else None,
                ) as runtime:
                    runtime.create_branch("main")
                    metrics = RuntimeTelemetry()
                    with metrics.capture():
                        runtime.seal_chunk("main", "c0", tensors, contexts)
                    rows = metrics.snapshot()["stages"]
                    self.assertEqual(rows["seal.encode"]["completed"], 2)
                    self.assertEqual(rows["seal.persist"]["completed"], 2)
                    self.assertEqual(rows["seal.publish"]["completed"], 1)
                    self.assertEqual(rows["seal.wait"]["completed"], 2 if use_executor else 0)
                    with metrics.capture(), runtime.acquire_history("main", MatrixKey(0, "V")) as lease:
                        reference = weakref.ref(lease.tensor)
                        self.assertTrue(torch.equal(lease.tensor, tensors[MatrixKey(0, "V")]))
                    gc.collect()
                    self.assertIsNone(reference())
                    self.assertEqual(runtime.resident_bytes, 0)
                    rows = metrics.snapshot()["stages"]
                    self.assertEqual(rows["history.restore"]["completed_bytes"], 24 * 4 * 4)
                    self.assertEqual(rows["history.copy"]["completed_bytes"], 24 * 4 * 4)

    def test_collective_failure_is_counted_and_propagated(self):
        owner = DistributedOwner()
        metrics = RuntimeTelemetry()
        with metrics.capture(), self.assertRaisesRegex(PeerFailure, "transport failed"):
            owner._collective(mock.Mock(side_effect=ValueError("transport")))
        row = metrics.snapshot()["stages"]["transport.collective"]
        self.assertEqual((row["calls"], row["failed"], row["inflight"]), (1, 1, 0))


if __name__ == "__main__":
    unittest.main()
