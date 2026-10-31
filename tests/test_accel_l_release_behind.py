"""accel L (2026-10-30): ``LineageRestorer.restore(release_behind=d)`` (CPU, software decoder pool).

LingBot's layer-overlapped restore (track L, plan.md section 12 A4) runs beside the next chunk's first forward, so the
reconstruction temporaries the restore holds until it ends (~3.45 GiB per 480P chunk on the native-RoPE path) would
add to the forward's working set. ``release_behind=d`` releases each layer's reconstruction handle once the layer
``d + 1`` positions later was issued. Covered here:
- the restored KV is bitwise the default's (plain and overlapped restores, prepared and staged histories);
- the default (None) holds every handle until the caller and reports nothing new;
- a handle is released only after later layers were issued, never before;
- invalid depths are refused.
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from reelcache import restore_kernels as rk
from reelcache.archive_restore import LineageRestorer, _retire_behind
from reelcache.errors import ConfigurationError
from reelcache.overlapped_restore import start_overlapped_restore
from reelcache.types import MatrixKey
from tests import step6_support as s6
from tests.test_packed_nvdec import SoftwarePool
from tests.test_step7_cold_hooks import _commit_chain

LAYERS = s6.MODEL.layers


def destination_factory(chunks=3, log=None):
    caches = {}

    def destination(node_id, key, tile):
        key = key if isinstance(key, MatrixKey) else key[0]
        if log is not None:
            log.append(("bind", key.layer))
        if key not in caches:
            caches[key] = torch.zeros((1, chunks * s6.TOKENS, 1, 128), dtype=torch.float32)
        return rk.NativeView(caches[key], "BTHD", token_offset=tile.token_start)

    return caches, destination


class Handle:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class RetireBehindTests(unittest.TestCase):
    def test_each_append_releases_the_handle_depth_plus_one_behind(self):
        for depth in (0, 1, 2, 5):
            handles = []
            for position in range(8):
                handles.append(Handle())
                _retire_behind(handles, depth)
                self.assertEqual([h.released for h in handles],
                                 [index <= position - depth - 1 for index in range(position + 1)], (depth, position))

    def test_none_releases_nothing_and_released_or_plain_entries_are_skipped(self):
        handles = [Handle() for _ in range(5)]
        self.assertEqual(_retire_behind(handles, None), 0)
        self.assertFalse(any(h.released for h in handles))
        handles[2].released = True
        self.assertEqual(_retire_behind(handles, 1), 0)  # index 2: already released
        self.assertEqual(_retire_behind([object(), object(), object()], 0), 0)  # no release(): left alone


class ReleaseBehindRestoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):  # one committed chain for the class (read-only below; every test builds its own restorer)
        cls.runtime, cls.heads = _commit_chain(SimpleNamespace(addCleanup=cls.addClassCleanup), "accel-l", count=3)
        cls.head = cls.heads[-1]
        cls.reference, destination = destination_factory()
        LineageRestorer(cls.runtime, SoftwarePool(), arena_bytes=1 << 30).restore(cls.head, destination)
        assert len(cls.reference) == 2 * LAYERS

    def restorer(self, prepared=True):
        restorer = LineageRestorer(self.runtime, SoftwarePool(), arena_bytes=1 << 30)
        self.addCleanup(restorer.close)
        if prepared:
            restorer.prewarm(restorer.history(self.head))
        return restorer

    def assert_equal_to_reference(self, caches):
        self.assertEqual(caches.keys(), self.reference.keys())
        for key, value in self.reference.items():
            self.assertTrue(torch.equal(caches[key], value), key)

    def test_plain_restores_equal_the_default_and_retire_handles_progressively(self):
        restorer = self.restorer()
        for depth in (None, 0, 1, 7):
            with self.subTest(depth=depth):
                caches, destination = destination_factory()
                report = restorer.restore(self.head, destination, release_behind=depth)
                self.assert_equal_to_reference(caches)
                handles = report["handles"]
                self.assertEqual(len(handles), LAYERS)  # one reconstruction per layer
                early = 0 if depth is None else max(0, LAYERS - depth - 1)
                self.assertEqual([h.released for h in handles], [i < early for i in range(LAYERS)])
                if depth is None:
                    self.assertNotIn("release_behind", report)
                else:
                    self.assertEqual(report["release_behind"], {"depth": depth, "released_early": early})
                for handle in handles:
                    handle.release()

    def test_a_handle_is_released_only_after_the_later_layer_was_issued(self):
        log = []
        original = rk.RestoreHandle.release

        def release(handle):
            if not handle.released:
                log.append(("release", id(handle)))
            return original(handle)

        restorer = self.restorer()
        caches, destination = destination_factory(log=log)
        with mock.patch.object(rk.RestoreHandle, "release", release):
            report = restorer.restore(self.head, destination, release_behind=0)
        self.assert_equal_to_reference(caches)
        index = {id(handle): position for position, handle in enumerate(report["handles"])}
        released = [index[item] for kind, item in log if kind == "release"]
        self.assertEqual(released, list(range(LAYERS - 1)))
        for kind, item in log:  # every release follows the binding of a layer two positions on
            if kind == "release":
                position = log.index((kind, item))
                bound = {layer for k, layer in log[:position] if k == "bind"}
                self.assertIn(index[item] + 1, bound)

    def test_a_staged_history_restores_identically(self):
        restorer = self.restorer(prepared=False)
        caches, destination = destination_factory()
        report = restorer.restore(self.head, destination, release_behind=0)
        self.assert_equal_to_reference(caches)
        self.assertEqual(report["release_behind"]["released_early"], LAYERS - 1)

    def test_the_overlapped_restore_passes_the_option_through(self):
        for prepared in (True, False):
            with self.subTest(prepared=prepared):
                caches, destination = destination_factory()
                readiness = start_overlapped_restore(self.restorer(prepared), self.head, destination, layers=LAYERS,
                                                     require_prepared=prepared, release_behind=0)
                for layer in range(LAYERS):
                    readiness.wait(layer)
                info = readiness.join()
                self.assert_equal_to_reference(caches)
                self.assertEqual(info["release_behind"], {"depth": 0, "released_early": LAYERS - 1})
                self.assertTrue(info["progressive"])

    def test_invalid_depths_are_refused(self):
        restorer = self.restorer()
        for bad in (-1, 1.0, True, "1"):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                restorer.restore(self.head, destination_factory()[1], release_behind=bad)


if __name__ == "__main__":
    unittest.main()
