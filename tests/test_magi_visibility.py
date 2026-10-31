from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from reelcache.distributed import PeerFailure
from reelcache.errors import ConfigurationError, UnsupportedError
from tests import test_magi_adapter as fixtures


class MagiVisibilityTests(unittest.TestCase):
    setUp = fixtures.MagiAdapterTests.setUp
    tearDown = fixtures.MagiAdapterTests.tearDown
    _adapter = fixtures.MagiAdapterTests._adapter
    _forward = fixtures.MagiAdapterTests._forward

    def forward_ranges(self, chunk, ranges, *, clean=False, near=False):
        validate = self.adapter._validate_meta

        def scoped(value, meta):
            meta.core_attn_params = SimpleNamespace(np_k_range=ranges)
            validate(value, meta)

        with mock.patch.object(self.adapter, "_validate_meta", side_effect=scoped):
            return self._forward(chunk, clean=clean, near=near)[0]

    def capture_two(self):
        self.adapter = self._adapter(visible_history_only=True)
        self.adapter.attach()
        first = self.forward_ranges(0, [[0, 6]], clean=True)
        self.adapter.seal_chunk()
        second = self.forward_ranges(1, [[6, 12]], clean=True)
        self.adapter.seal_chunk()
        self.runtime.requested_columns.clear()
        return first, second

    def test_clean_recapture_skips_history_and_preserves_absolute_current_slot(self):
        first, second = self.capture_two()
        self.assertEqual(self.adapter._stats["skipped_archive_chunk_visits"], 2)
        self.assertEqual(self.runtime.requested_columns, [])
        for a, b in zip(first, second):
            for x, y in zip(a, b):
                self.assertEqual(tuple(y.shape), (12, *x.shape[1:]))
                self.assertEqual(torch.count_nonzero(y[:6]).item(), 0)
                self.assertGreater(torch.count_nonzero(y[6:]).item(), 0)
        self.assertEqual(self.runtime.resident_bytes, 0)

    def test_visibility_skips_only_disjoint_chunks_without_rebasing(self):
        _, second = self.capture_two()
        actual = self.forward_ranges(2, np.array([[7, 18]], dtype=np.int32))
        self.assertEqual(len(self.runtime.requested_columns), 4)
        for old, new in zip(second, actual):
            for expected, value in zip(old, new):
                self.assertTrue(torch.equal(value[6:12], expected[6:12]))
                self.assertEqual(torch.count_nonzero(value[:6]).item(), 0)
                self.assertEqual(value.shape[0], 18)
        self.assertFalse(self.cache.key_value_memory_dict)
        self.assertEqual(self.runtime.resident_bytes, 0)

    def test_near_clean_duplicate_uses_union_of_all_query_ranges(self):
        first, _ = self.capture_two()
        actual = self.forward_ranges(2, [[0, 6], [12, 24]], near=True)
        self.assertEqual(len(self.runtime.requested_columns), 4)
        for old, new in zip(first, actual):
            for expected, value in zip(old, new):
                self.assertTrue(torch.equal(value[:6], expected))
                self.assertEqual(torch.count_nonzero(value[6:12]).item(), 0)
                self.assertEqual(value.shape[0], 24)

    def test_overlapping_ranges_restore_each_chunk_only_once(self):
        self.capture_two()
        self.forward_ranges(2, [[0, 18], [3, 24]], near=True)
        self.assertEqual(len(self.runtime.requested_columns), 8)

    def test_invalid_or_missing_ranges_fail_before_history_reads(self):
        self.capture_two()
        for ranges in ([], [[0, 6], [6, 12]], [[-1, 6]], [[6, 6]], [[0, 19]],
                       [[0., 6]], [[True, 6]], [0], None):
            with self.subTest(ranges=ranges):
                meta = SimpleNamespace(denoising_range_num=1, core_attn_params=SimpleNamespace(np_k_range=ranges))
                with self.assertRaises((ConfigurationError, UnsupportedError)):
                    self.adapter._visible_history(meta)
        self.assertEqual(self.runtime.requested_columns, [])
        with self.assertRaisesRegex(PeerFailure, "np_k_range"):
            self._forward(2, clean=False)
        self.assertEqual(self.runtime.requested_columns, [])
        self.assertFalse(self.cache.key_value_memory_dict)

    def test_default_off_and_options_are_strict(self):
        self.assertFalse(self.adapter.visible_history_only)
        self.assertFalse(self.adapter.shared_history_decode)
        for option in ("visible_history_only", "shared_history_decode"):
            for value in (1, None, "yes"):
                with self.assertRaises(ConfigurationError):
                    self._adapter(**{option: value})


if __name__ == "__main__":
    unittest.main()
