"""decomp_opt (2026-10-25): pipelined cold restores and unhashed basis staging (CPU, software decoder pool).

``LineageRestorer(pipelined_staging=True)`` submits each staged pack's decode jobs as soon as staging prepared it; the
restored KV must equal the batched path's bitwise, also when the arena budget forces lazy submission.
``StagingEngine(verify_bases=False)`` skips basis-row hashing only for stores that do not verify reads.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from reelcache.archive_restore import LineageRestorer
from reelcache.errors import ConfigurationError, IntegrityError
from reelcache.manager import KVRuntime
from reelcache.restore_staging import StagingEngine
from tests import step6_support as s6
from tests.step6_support import MODEL, RuntimeFactory
from tests.test_packed_nvdec import SoftwarePool
from tests import test_restore_staging as staging_tests
from tests.test_step7_store_faults import SERVING, StoreFaultCase, _destination


def _engine(case, **options):
    engine = StagingEngine("cpu", threads=2, prepare_threads=2, pinned_bytes=1 << 22, **options)
    case.addCleanup(engine.close)
    return engine


class PipelinedRestoreTests(StoreFaultCase):
    def test_pipelined_cold_restore_equals_the_reference(self):
        for flags in ({}, SERVING):
            with self.subTest(flags=flags):
                runtime = self.open(**flags)
                restorer = LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 30, staging_engine=_engine(self),
                                           pipelined_staging=True)
                for name in ("a", "b"):
                    caches, destination = _destination(2 * s6.TOKENS)
                    report = restorer.restore(self.heads[name], destination)
                    self.assertSameCaches(caches, self.reference[name])
                    piped = report["pipelined"]
                    self.assertEqual((piped["batches"], piped["early_batches"], piped["lazy_batches"]),
                                     (report["nodes"], report["nodes"], 0))
                    self.assertIsNotNone(piped["first_submit_seconds"])
                    self.assertGreater(restorer.last_staging["on_prepared_seconds"], 0.0)
                # adopted into the prepared cache: the batched path serves the same history now
                caches, destination = _destination(2 * s6.TOKENS)
                report = restorer.restore(self.heads["a"], destination)
                self.assertNotIn("pipelined", report)
                self.assertSameCaches(caches, self.reference["a"])
                runtime.close()

    def test_packs_beyond_the_arena_budget_are_submitted_lazily(self):
        runtime = self.open()
        probe = LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 30, staging_engine=_engine(self),
                                pipelined_staging=True)
        caches, destination = _destination(2 * s6.TOKENS)
        total = probe.restore(self.heads["a"], destination)["pipelined"]["early_canvas_bytes"]
        tight = LineageRestorer(runtime, SoftwarePool(), arena_bytes=total - 1, staging_engine=_engine(self),
                                pipelined_staging=True)
        caches, destination = _destination(2 * s6.TOKENS)
        piped = tight.restore(self.heads["a"], destination)["pipelined"]
        self.assertSameCaches(caches, self.reference["a"])
        self.assertEqual((piped["early_batches"], piped["lazy_batches"]), (1, 1))

    def test_pipelined_staging_must_be_a_bool(self):
        runtime = self.open()
        with self.assertRaises(ConfigurationError):
            LineageRestorer(runtime, SoftwarePool(), pipelined_staging=1)


class StagingHooksTests(unittest.TestCase):
    lineage = staging_tests.StagingEngineTests.lineage
    history = staticmethod(staging_tests.StagingEngineTests.history)
    corrupt = staging_tests.StagingEngineTests.corrupt

    def setUp(self):
        self.runtimes = RuntimeFactory(self)

    def test_on_prepared_receives_each_pack_in_history_order(self):
        runtime = self.runtimes()
        commits = self.lineage(runtime, count=3)
        history = self.history(runtime, commits[-1].node_ref)
        engine, seen = _engine(self), []
        staged = engine.stage(runtime, history, on_prepared=seen.append)
        self.assertEqual(len(seen), 3)
        for got, ref in zip(seen, staged.refs, strict=True):
            self.assertIs(got, ref)
        staged.wait()
        staged.release()
        with self.assertRaises(ConfigurationError):
            engine.stage(runtime, history, on_prepared=1)

        def boom(_ref):
            raise RuntimeError("boom")

        with self.assertRaisesRegex(RuntimeError, "boom"):
            engine.stage(runtime, history, on_prepared=boom)
        staged = engine.stage(runtime, history)  # the engine stays usable
        staged.wait()
        staged.release()

    def test_verify_bases_off_skips_basis_hashing_only_for_unverified_reads(self):
        runtime = self.runtimes()  # verify_reads off
        commits = self.lineage(runtime, count=3)
        history = self.history(runtime, commits[-1].node_ref)
        hashing, skipping = _engine(self), _engine(self, verify_bases=False)
        expected, got = hashing.stage(runtime, history), skipping.stage(runtime, history)
        expected.wait()
        got.wait()
        self.assertEqual([item[:2] for item in got.versions], [item[:2] for item in expected.versions])
        for (_, _, mine, _), (_, _, theirs, _) in zip(got.versions, expected.versions, strict=True):
            self.assertTrue(torch.equal(mine, theirs))
        for ref, theirs in expected.prepared.items():
            for mine, tile in zip(got.prepared[ref].tiles, theirs.tiles, strict=True):
                self.assertTrue(torch.equal(mine.basis, tile.basis))
        self.assertGreater(got.stats["tips"], 0)
        self.assertEqual(skipping.stats()["unhashed_tips"], got.stats["tips"])
        self.assertEqual(hashing.stats()["unhashed_tips"], 0)
        expected.release()
        got.release()
        # a store that verifies reads keeps hashing bases whatever verify_bases says
        verifying = KVRuntime(runtime.store.root, MODEL, verify_reads=True)
        self.addCleanup(verifying.close)
        staged = skipping.stage(verifying, self.history(verifying, commits[-1].node_ref))
        staged.wait()
        staged.release()
        self.assertEqual(skipping.stats()["unhashed_tips"], got.stats["tips"])
        # the trade-off: without hashing, a damaged basis block is not detected
        packet = runtime.load_packet(history[-1]["packet"])
        block = runtime.basis_versions.blocks(packet["tile_bases"][0]["basis_version"])[-1]
        self.corrupt(runtime, block)
        staged = skipping.stage(runtime, history)
        staged.wait()
        staged.release()
        damaged = hashing.stage(runtime, history)
        with self.assertRaisesRegex(IntegrityError, "content address"):
            damaged.wait()
        with self.assertRaises(IntegrityError):
            damaged.release()

    def test_verify_bases_must_be_a_bool(self):
        with self.assertRaises(ConfigurationError):
            StagingEngine("cpu", verify_bases=1)


class ConfigTests(unittest.TestCase):
    def test_new_fields(self):
        from reelcache.integrations.sglang.config import ReelCacheRuntimeConfig

        with tempfile.TemporaryDirectory() as root:
            store = str(Path(root) / "store")
            default = ReelCacheRuntimeConfig(store_root=store)
            self.assertEqual((default.pipelined_restore, default.verify_bases), (False, True))
            ReelCacheRuntimeConfig(store_root=store, pipelined_restore=True, verify_reads=False, verify_bases=False)
            for bad in ({"verify_bases": False}, {"pipelined_restore": 1}, {"verify_bases": 0, "verify_reads": False}):
                with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                    ReelCacheRuntimeConfig(store_root=store, **bad)


if __name__ == "__main__":
    unittest.main()
