from contextlib import ExitStack, contextmanager
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import weakref

import torch
import torch.distributed as dist

from reelcache import _hevc
from reelcache.codecs import decode_column_views, decode_tensor, encode_tensor
from reelcache.distributed import DistributedOwner, KVLayout, PeerFailure
from reelcache.errors import CapacityError, ConfigurationError, UnsupportedError
from reelcache.manager import KVRuntime
from reelcache.telemetry import RuntimeTelemetry
from reelcache.types import CodecSpec, MatrixKey, ModelIdentity, TensorContext
from tests.test_codecs import context
from tests import test_magi_adapter as fixtures


MODEL = ModelIdentity("grouped", "cpu-fixture-v1", 1, 8, "canonical", "float32")
GROUPS = [slice(0, 4), slice(4, 8)]


def seed(root, kind, *, chunks=1):
    runtime = KVRuntime(root, MODEL, max_resident_bytes=1 << 20)
    runtime.create_branch("main")
    rng = torch.Generator().manual_seed(531)
    for index in range(chunks):
        matrices = {MatrixKey(0, target): torch.randn(16, 8, generator=rng) for target in ("K", "V")}
        contexts = {
            key: TensorContext(MODEL, key, index * 16, 2, 2, 4, "pre_rope" if key.target == "K" else "none")
            for key in matrices
        }
        runtime.seal_chunk(
            "main", str(index), matrices, contexts,
            CodecSpec(kind=kind, rank=None if kind == "raw" else 3),
        )
    return runtime


def gloo_timeout_seconds() -> float:
    """Process-group timeout of the two-process test: still fails a deadlock, but not a slow start.

    Both spawned workers import torch and meet at a file rendezvous; on a machine loaded by other suites (concurrent
    agents) that alone exceeded the former fixed 30 s. The timeout scales with the 1-minute load per CPU (at least
    120 s); ``REELCACHE_TEST_GLOO_TIMEOUT`` overrides it.
    """
    override = os.environ.get("REELCACHE_TEST_GLOO_TIMEOUT")
    if override:
        return float(override)
    load = os.getloadavg()[0] / max(1, len(os.sched_getaffinity(0)))
    return max(120.0, 60.0 * (1.0 + load))


def grouped_gloo_worker(rank, root, timeout_seconds=120.0):
    root = Path(root)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=(root / "rendezvous").as_uri(),
        timeout=timedelta(seconds=timeout_seconds),
    )
    try:
        owner = DistributedOwner(expected_world_size=2, max_wire_bytes=64)
        layout = KVLayout(tokens=16, kv_heads=4, head_dim=2, world_size=2)
        runtime = owner.owner_call("seed", lambda: seed(root / "store", "svd_hevc"))
        key = MatrixKey(0, "V")

        @contextmanager
        def cold(columns):
            with runtime.acquire_history("main", key, columns=columns) as lease:
                yield lease.tensor

        with owner.acquire_columns(layout, cold, dtype=torch.float32) as value:
            expected = value.clone()
        metrics = RuntimeTelemetry()
        with metrics.capture(), ExitStack() as stack:
            grouped = owner.owner_call(
                "prepare", lambda: stack.enter_context(runtime.acquire_history("main", key, column_groups=GROUPS)),
            )

            @contextmanager
            def grouped_loader(columns):
                yield grouped.tensor[GROUPS.index(columns)]

            with owner.acquire_columns(layout, grouped_loader, dtype=torch.float32) as value:
                assert torch.equal(value, expected)
                assert value.is_contiguous()
                if rank == 0:
                    assert runtime.resident_bytes == 16 * 8 * 4
        if rank == 0:
            assert runtime.resident_bytes == runtime.residency.reserved_bytes == 0
            runtime.residency.max_bytes = 1
        try:
            owner.owner_call("capacity", lambda: runtime.acquire_history("main", key, column_groups=GROUPS))
        except PeerFailure:
            rejected = True
        else:
            rejected = False
        assert rejected
        owner.checkpoint("recovered")
        if rank == 0:
            runtime.close()
        (root / f"{rank}.json").write_text(json.dumps({
            "stages": metrics.snapshot()["stages"],
            "cuda_initialized": torch.cuda.is_initialized(),
            "capacity_rejected": rejected,
        }))
    finally:
        dist.destroy_process_group()


class GroupedHistoryTests(unittest.TestCase):
    def test_gloo_timeout_scales_with_load_and_honors_the_override(self):
        with mock.patch("os.getloadavg", return_value=(0.0, 0.0, 0.0)):
            self.assertEqual(gloo_timeout_seconds(), 120.0)
        cpus = len(os.sched_getaffinity(0))
        with mock.patch("os.getloadavg", return_value=(4.0 * cpus, 0.0, 0.0)):
            self.assertEqual(gloo_timeout_seconds(), 300.0)
        with mock.patch.dict(os.environ, {"REELCACHE_TEST_GLOO_TIMEOUT": "45"}):
            self.assertEqual(gloo_timeout_seconds(), 45.0)

    def test_real_providers_preserve_each_column_product_and_decode_once(self):
        source = torch.randn(24, 8, generator=torch.Generator().manual_seed(701))
        for kind in ("raw", "svd", "svd_hevc"):
            with self.subTest(kind=kind):
                packet = encode_tensor(
                    source, context(channels=8), CodecSpec(kind=kind, rank=None if kind == "raw" else 3),
                )
                expected = torch.stack([decode_tensor(packet, columns=columns) for columns in GROUPS])
                with mock.patch("reelcache._hevc.decode_pixels", wraps=_hevc.decode_pixels) as decoder:
                    actual = decode_column_views(packet, column_groups=GROUPS)
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(decoder.call_count, int(kind == "svd_hevc"))
                self.assertTrue(actual.is_contiguous())
                self.assertEqual(actual.untyped_storage().nbytes(), actual.numel() * actual.element_size())

    def test_ordered_sparse_groups_and_invalid_partitions(self):
        packet = encode_tensor(torch.arange(192).reshape(24, 8).float(), context(channels=8), CodecSpec())
        groups = [[3, 1], [6, 5]]
        actual = decode_column_views(packet, column_groups=groups)
        self.assertTrue(torch.equal(actual, torch.stack([decode_tensor(packet, columns=g) for g in groups])))
        for groups in ([], None, [slice(0, 4), slice(3, 7)], [[0, 1], [2]], [None, None]):
            with self.subTest(groups=groups), self.assertRaises(ConfigurationError):
                decode_column_views(packet, column_groups=groups)

    def test_partial_multichunk_restore_reads_each_packet_once_and_releases(self):
        with tempfile.TemporaryDirectory() as root, seed(root, "svd_hevc", chunks=2) as runtime:
            key = MatrixKey(0, "K")
            expected = []
            for columns in GROUPS:
                with runtime.acquire_history("main", key, token_start=7, token_end=27, columns=columns) as lease:
                    expected.append(lease.tensor.clone())
            with mock.patch.object(runtime, "_encoded", wraps=runtime._encoded) as reads:
                with runtime.acquire_history(
                    "main", key, token_start=7, token_end=27, column_groups=GROUPS,
                ) as lease:
                    reference = weakref.ref(lease.tensor)
                    self.assertTrue(torch.equal(lease.tensor, torch.stack(expected)))
                    self.assertEqual(runtime.resident_bytes, 2 * 20 * 4 * 4)
                    self.assertEqual(reads.call_count, 2)
            gc.collect()
            self.assertIsNone(reference())
            self.assertEqual(runtime.residency.reserved_bytes, 0)

    def test_capacity_precedes_payload_read_and_custom_decoder_is_not_bypassed(self):
        with tempfile.TemporaryDirectory() as root, seed(root, "raw") as runtime:
            key = MatrixKey(0, "V")
            runtime.residency.max_bytes = 1
            with mock.patch.object(runtime, "_encoded", side_effect=AssertionError("read before admission")):
                with self.assertRaises(CapacityError):
                    runtime.acquire_history("main", key, column_groups=GROUPS)
            self.assertEqual(runtime.residency.reserved_bytes, 0)
            runtime.decoder = mock.Mock()
            with self.assertRaises(UnsupportedError):
                runtime.acquire_history("main", key, column_groups=GROUPS)
            runtime.decoder.assert_not_called()

    def test_empty_range_and_invalid_groups_do_not_decode(self):
        with tempfile.TemporaryDirectory() as root, seed(root, "raw") as runtime:
            key = MatrixKey(0, "V")
            with mock.patch.object(runtime, "_encoded", side_effect=AssertionError("unused payload")):
                with runtime.acquire_history("main", key, token_end=0, column_groups=GROUPS) as lease:
                    self.assertEqual(tuple(lease.tensor.shape), (2, 0, 4))
                for options in (
                    {"column_groups": []}, {"column_groups": GROUPS, "columns": slice(0, 4)},
                    {"column_groups": [slice(0, 4), slice(0, 4)]},
                    {"column_groups": [slice(0, 4), slice(4, 7)]},
                ):
                    with self.assertRaises(ConfigurationError):
                        runtime.acquire_history("main", key, **options)

    def test_failure_releases_grouped_workspace_and_pins(self):
        with tempfile.TemporaryDirectory() as root, seed(root, "svd_hevc") as runtime:
            with mock.patch("reelcache._hevc.decode_pixels", side_effect=RuntimeError("decode failed")):
                with self.assertRaisesRegex(RuntimeError, "decode failed"):
                    runtime.acquire_history("main", MatrixKey(0, "K"), column_groups=GROUPS)
            self.assertEqual(runtime.active_leases, 0)
            self.assertEqual(runtime.residency.reserved_bytes, 0)
            with runtime.acquire_history("main", MatrixKey(0, "K"), column_groups=GROUPS):
                pass

    def test_native_adapter_opt_in_matches_default_with_real_packets(self):
        fixture = fixtures.MagiAdapterTests(methodName="runTest")
        fixture.setUp()
        try:
            with KVRuntime(fixture.root / "store", fixture.identity) as runtime:
                fixture.runtime = runtime
                fixture.adapter = fixture._adapter(specs=CodecSpec(kind="svd_hevc", rank=2))
                fixture.adapter.attach()
                fixture._forward(0, clean=True)
                fixture.adapter.seal_chunk()
                snapshot = fixture.adapter.snapshot(**fixture._state())
                expected, _ = fixture._forward(1, clean=False)
                fixture.adapter.shared_history_decode = True
                fixture.adapter.restore(snapshot, "grouped")
                actual, _ = fixture._forward(1, clean=False)
                for left, right in zip(expected, actual):
                    for a, b in zip(left, right):
                        self.assertTrue(torch.equal(a, b))
                self.assertEqual(runtime.resident_bytes, 0)
                fixture.adapter.detach()
        finally:
            fixture.tearDown()

    def test_two_process_gloo_matches_cold_path_and_propagates_admission_failure(self):
        with tempfile.TemporaryDirectory() as root:
            torch.multiprocessing.start_processes(
                grouped_gloo_worker, args=(root, gloo_timeout_seconds()), nprocs=2, join=True, start_method="spawn",
            )
            rows = [json.loads((Path(root) / f"{rank}.json").read_text()) for rank in range(2)]
        self.assertEqual(rows[0]["stages"]["decode.hevc"]["completed"], 1)
        self.assertEqual(rows[0]["stages"]["decode.reconstruct"]["completed"], 2)
        self.assertEqual(rows[1]["stages"]["decode.hevc"]["calls"], 0)
        self.assertTrue(all(row["capacity_rejected"] for row in rows))
        self.assertFalse(any(row["cuda_initialized"] for row in rows))

    def test_grouped_history_hot_owners_do_not_retain_temporary_group_storage(self):
        fixture = fixtures.MagiAdapterTests(methodName="runTest")
        fixture.setUp()
        try:
            with KVRuntime(fixture.root / "store", fixture.identity) as runtime:
                fixture.runtime = runtime
                fixture.adapter = fixture._adapter(
                    specs=CodecSpec(kind="svd_hevc", rank=2), shared_history_decode=True,
                    hot_window_bytes=384, hot_window_require_full=True,
                )
                fixture.adapter.attach()
                fixture._forward(0, clean=True)
                fixture.adapter.seal_chunk()
                telemetry = RuntimeTelemetry()
                with telemetry.capture(), fixture.adapter.execution(chunk_index=1, clean=False):
                    first, _ = fixture._forward(1, clean=False, guarded=False)
                    references = [weakref.ref(value) for value in fixture.adapter._hot.values()]
                    self.assertEqual(runtime.resident_bytes, 0)
                    self.assertEqual(runtime.residency.reserved_bytes, 0)
                    self.assertEqual(fixture.adapter._hot_bytes, 384)
                    self.assertTrue(all(value.untyped_storage().nbytes() == 96
                                        for value in fixture.adapter._hot.values()))
                    second, _ = fixture._forward(1, clean=False, guarded=False)
                    for left, right in zip(first, second):
                        for a, b in zip(left, right):
                            self.assertTrue(torch.equal(a, b))
                    self.assertEqual(telemetry.snapshot()["stages"]["decode.hevc"]["completed"], 4)
                    self.assertEqual(fixture.adapter._hot_stats["hits"], 4)
                gc.collect()
                self.assertTrue(all(ref() is None for ref in references))
                self.assertEqual(fixture.adapter._hot_bytes, 0)
                self.assertEqual(fixture.adapter.memory_report()["owner_pending_sources"]["pending_source_bytes"], 0)
                fixture.adapter.detach()
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
