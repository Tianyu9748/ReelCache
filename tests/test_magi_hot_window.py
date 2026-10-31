"""CPU contracts for opt-in attention-ready MAGI history reuse."""

import gc
import unittest
import weakref

import torch

from reelcache.distributed import PeerFailure
from reelcache.errors import ConfigurationError
from reelcache.types import CodecSpec
from tests import test_magi_adapter as fixtures


class MagiHotWindowTests(unittest.TestCase):
    setUp = fixtures.MagiAdapterTests.setUp
    tearDown = fixtures.MagiAdapterTests.tearDown
    _adapter = fixtures.MagiAdapterTests._adapter
    _forward = fixtures.MagiAdapterTests._forward
    _state = fixtures.MagiAdapterTests._state

    def _capture(self, **options):
        self.adapter = self._adapter(**options)
        self.adapter.attach()
        captured, _ = self._forward(0, clean=True)
        self.adapter.seal_chunk()
        return captured

    def test_native_ready_hits_are_exact_owned_and_released_after_epoch(self):
        captured = self._capture(hot_window_bytes=384, hot_window_require_full=True)
        references = []
        with self.adapter.execution(chunk_index=1, clean=False):
            first, _ = self._forward(1, clean=False, guarded=False)
            references = [weakref.ref(value) for value in self.adapter._hot.values()]
            self.assertEqual(self.adapter._hot_bytes, 384)
            self.assertEqual(self.runtime.pins, 1)
            second, _ = self._forward(1, clean=False, near=True, guarded=False)
            for expected, actual in zip(captured, first):
                self.assertTrue(torch.equal(expected[0], actual[0][:6]))
                self.assertTrue(torch.equal(expected[1], actual[1][:6]))
            for a, b in zip(first, second):
                self.assertTrue(torch.equal(a[0][:6], b[0][:6]))
                self.assertTrue(torch.equal(a[1][:6], b[1][:6]))
            self.assertEqual(len(self.runtime.requested_columns), 4)
            self.assertEqual(self.adapter._hot_stats["hits"], 4)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        self.assertEqual(self.runtime.pins, 0)
        self.assertEqual(self.runtime.resident_bytes, 0)
        self.assertEqual(self.adapter._hot_bytes, 0)
        self.assertFalse(self.adapter._hot)
        self.assertFalse(self.cache.key_value_memory_dict)

    def test_budget_admits_fixed_subset_without_replacing_missed_history(self):
        self._capture(hot_window_bytes=96)
        with self.adapter.execution(chunk_index=1, clean=False):
            first, _ = self._forward(1, clean=False, guarded=False)
            second, _ = self._forward(1, clean=False, guarded=False)
        for a, b in zip(first, second):
            for x, y in zip(a, b):
                self.assertTrue(torch.equal(x, y))
        self.assertEqual(self.adapter._hot_stats["hits"], 1)
        self.assertEqual(self.adapter._hot_stats["misses"], 7)
        self.assertEqual(self.adapter._hot_stats["bypasses"], 6)
        self.assertEqual(self.adapter._hot_stats["peak_bytes"], 96)
        self.assertEqual(len(self.runtime.requested_columns), 7)

    def test_full_budget_rejection_leaves_boundary_and_rng_unchanged(self):
        self._capture(hot_window_bytes=383, hot_window_require_full=True)
        before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(PeerFailure, "exceeds the hot-window"):
            with self.adapter.execution(chunk_index=1, clean=False):
                self.fail("Insufficient full-history budget entered native execution")
        self.assertEqual(self.adapter.phase, "ready")
        self.assertEqual(self.adapter.cursor, 1)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(self.runtime.pins, 0)
        self.assertEqual(self.runtime.requested_columns, [])

    def test_epoch_profile_mutation_fails_and_releases_allocations(self):
        self._capture(hot_window_bytes=384)
        with self.assertRaisesRegex(PeerFailure, "profile/positions changed"):
            with self.adapter.execution(chunk_index=1, clean=False):
                self._forward(1, clean=False, guarded=False)
                self.adapter.specs = {key: CodecSpec(qp=9) for key in self.adapter.keys}
                self._forward(1, clean=False, guarded=False)
        self.assertEqual(self.adapter.phase, "failed")
        self.assertEqual(self.runtime.pins, 0)
        self.assertEqual(self.adapter._hot_bytes, 0)

    def test_epoch_head_change_and_interruption_cannot_reuse_stale_history(self):
        self._capture(hot_window_bytes=384)
        with self.assertRaisesRegex(PeerFailure, "head changed"):
            with self.adapter.execution(chunk_index=1, clean=False):
                self._forward(1, clean=False, guarded=False)
                self.runtime.branches["parent"].append(self.runtime.branches["parent"][0])
                self._forward(1, clean=False, guarded=False)
        self.assertFalse(self.adapter._hot)
        self.assertEqual(self.runtime.pins, 0)

    def test_snapshot_fork_and_append_start_new_windows(self):
        self._capture(hot_window_bytes=2048)
        snapshot = self.adapter.snapshot(**self._state())
        with self.adapter.execution(chunk_index=1, clean=False):
            self._forward(1, clean=False, guarded=False)
            self._forward(1, clean=False, guarded=False)
        self.assertFalse(self.adapter._hot)
        self.adapter.fork(snapshot, "child")
        self.assertFalse(self.adapter._hot)
        with self.adapter.execution(chunk_index=1, clean=False):
            self._forward(1, clean=False, guarded=False)
        self.assertEqual(self.adapter._hot_stats["last_epoch_hits"], 0)
        self.assertEqual(self.adapter._hot_stats["last_epoch_misses"], 4)
        self._forward(1, clean=True)
        self.adapter.seal_chunk()
        self.assertFalse(self.adapter._hot)
        self.assertEqual(len(self.runtime.branches["parent"]), 1)
        self.assertEqual(len(self.runtime.branches["child"]), 2)
        self.adapter.detach()
        self.assertEqual(self.runtime.pins, 0)

    def test_inplace_rotary_mutation_and_cancellation_release_hot_history(self):
        self._capture(specs=CodecSpec(kind="svd", rank=2), hot_window_bytes=384)
        with self.assertRaisesRegex(PeerFailure, "profile/positions changed"):
            with torch.inference_mode(), self.adapter.execution(chunk_index=1, clean=False):
                self._forward(1, clean=False, guarded=False)
                self.adapter._ropes["0"][0].add_(0.5)
                self._forward(1, clean=False, guarded=False)
        self.assertFalse(self.adapter._hot)
        self.assertEqual(self.runtime.pins, 0)

    def test_cancellation_with_resident_hot_data_is_not_success(self):
        self._capture(hot_window_bytes=384)
        with self.assertRaisesRegex(PeerFailure, "cancel"):
            with self.adapter.execution(chunk_index=1, clean=False):
                self._forward(1, clean=False, guarded=False)
                self.adapter.transport.checkpoint("cancel_hot_epoch", cancelled=True)
        self.assertFalse(self.adapter._hot)
        self.assertEqual(self.adapter._hot_bytes, 0)
        self.assertEqual(self.runtime.pins, 0)
        self.assertEqual(self.adapter.phase, "failed")

    def test_real_hevc_hot_hits_match_decoded_representation_and_verify_bridge(self):
        from reelcache.manager import KVRuntime

        self.runtime = KVRuntime(self.root / "hevc_store", self.identity, max_resident_bytes=1024**2)
        try:
            self._capture(
                specs=CodecSpec(kind="svd_hevc", rank=2), hot_window_bytes=384,
                hot_window_require_full=True, hot_window_verify_hits=1,
            )
            with self.adapter.execution(chunk_index=1, clean=False):
                first, _ = self._forward(1, clean=False, guarded=False)
                second, _ = self._forward(1, clean=False, guarded=False)
                for value in self.adapter._hot.values():
                    self.assertEqual(value.dtype, torch.bfloat16)
                    self.assertEqual(value.shape, (6, 2, 4))
            for a, b in zip(first, second):
                for x, y in zip(a, b):
                    self.assertTrue(torch.equal(x, y))
            self.assertEqual(self.adapter._hot_stats["verified_hits"], 1)
            self.assertEqual(self.adapter._hot_stats["hits"], 4)
            self.assertEqual(self.runtime.active_leases, 0)
            self.assertEqual(self.adapter.memory_report()["decoded_hot_window"]["resident_bytes"], 0)
            self.assertFalse(torch.cuda.is_initialized())
        finally:
            self.adapter.detach()
            self.runtime.close()

    def test_default_disabled_and_bad_options(self):
        self._capture()
        with self.adapter.execution(chunk_index=1, clean=False):
            self._forward(1, clean=False, guarded=False)
            self._forward(1, clean=False, guarded=False)
        self.assertEqual(len(self.runtime.requested_columns), 8)
        self.assertEqual(self.adapter._hot_stats["epochs"], 0)
        for value in (-1, True, 0.1):
            with self.assertRaises(ConfigurationError):
                self._adapter(hot_window_bytes=value)
