"""CPU tests of the LongCat-Video restorer's cold-session staging (CR-17-4; no SGLang, no GPU).

The LongCat worker's packed restorer is built with a Step 5c ``StagingEngine`` by default
(``reelcache_backend.lineage_restorer``; provider config ``staging_engine``, ``staging_threads``,
``staging_prepare_threads``, ``staging_pinned_bytes``): a cold history (packs not prepared, bases not resident) is
staged by the engine. These tests restore archived LongCat segment epochs through the backend with the staged and the
store path (a CPU decoder pool standing in for NVDEC) and require bitwise identical caches, check that staging
happened once per cold history and that verified reads still fail a corrupt packet typed, and check the defaults,
bounds and the attach report.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from reelcache_sglang_models.longcat_video import conditioning_cache as cc
from reelcache_sglang_models.longcat_video import reelcache_backend as mb
from reelcache_sglang_models.longcat_video import session_provider as sp
from tests.test_longcat_video_integrity import blobs, corrupt
from tests.test_longcat_video_reelcache_backend import ARCH, FINGERPRINT, HEADS, synthetic_cache
from tests.test_packed_nvdec import SoftwarePool


class LongCatStagingTests(unittest.TestCase):
    def setUp(self):
        from reelcache.archive_pipeline import ArchivePipeline
        from reelcache.archive_restore import SoftwareLineageRestorer
        from reelcache.lineage_commit import ArchiveOptions
        from reelcache.predictor_registry import RankPredictorRegistry

        self.root = Path(tempfile.mkdtemp(prefix="longcat-staging-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.runtime = mb.open_runtime(self.root / "store", mb.model_identity(FINGERPRINT, ARCH))  # serving defaults
        self.addCleanup(self.runtime.close)
        pipeline = ArchivePipeline(self.runtime, method=mb.default_method(),
                                   registry=RankPredictorRegistry("longcat-staging-test"),
                                   options=ArchiveOptions(encode_threads=1), device="cpu", encode_workers=1)
        archiver = mb.ReelCacheConditioningCacheBackend(self.runtime, pipeline=pipeline,
                                                     restorer=SoftwareLineageRestorer(self.runtime),
                                                     archive=("svd_hevc",), heads=HEADS, pin_memory=False,
                                                     capture_slots=3)
        self.caches = [synthetic_cache(11, 0), synthetic_cache(12, 1), synthetic_cache(13, 2)]
        for cache in self.caches:
            archiver.on_cache_built(cache)
        archiver.flush()
        pipeline.close()
        self.reference = archiver  # the software restorer's restores (store path, reference reconstruction)

    def restorer(self, staged: bool):
        restorer, engine = mb.lineage_restorer(self.runtime, SoftwarePool(), "cpu", arena_bytes=1 << 30,
                                               staging_engine=staged, staging_threads=2, staging_prepare_threads=2,
                                               staging_pinned_bytes=1 << 22)
        if engine is not None:
            self.addCleanup(engine.close)
        self.addCleanup(restorer.close)
        return restorer, engine

    def restore_all(self, restorer, caches=None) -> list:
        backend = mb.ReelCacheConditioningCacheBackend(self.runtime, restorer=restorer, archive=(), heads=HEADS,
                                                    pin_memory=False)
        backend.reload_index()
        backend.arm_restore("svd_hevc")
        return [backend.restore(cache.epoch) for cache in (self.caches if caches is None else caches)]

    @staticmethod
    def same(a: cc.LongCatConditioningCache, b: cc.LongCatConditioningCache) -> bool:
        return all(torch.equal(x, y) for x, y in zip(a.keys_pre_rope + a.values, b.keys_pre_rope + b.values))

    def test_staged_cold_restores_equal_the_store_path_bitwise(self):
        store_restorer, none = self.restorer(staged=False)
        staged_restorer, engine = self.restorer(staged=True)
        self.assertIsNone(none)
        self.assertIs(staged_restorer.staging_engine, engine)
        store = self.restore_all(store_restorer)
        self.assertIsNone(store_restorer.last_staging)
        staged = self.restore_all(staged_restorer)
        histories = engine.stats()["histories"]
        self.assertGreaterEqual(histories, 1)  # cold histories went through the engine
        self.assertIsNotNone(staged_restorer.last_staging)
        for a, b in zip(store, staged):
            self.assertTrue(self.same(a, b))
            self.assertEqual(a.source, cc.SOURCE_RESTORED)
        # warm now: the staged packs and bases were adopted, the engine is not used again; still bitwise
        again = self.restore_all(staged_restorer)
        self.assertEqual(engine.stats()["histories"], histories)
        for a, b in zip(store, again):
            self.assertTrue(self.same(a, b))
        # and the restored caches meet the archived epochs' identity (the software restorer's reconstruction)
        self.reference.arm_restore("svd_hevc")
        for cache, restored in zip(self.caches, store):
            reference = self.reference.restore(cache.epoch)
            for x, y in zip(reference.keys_pre_rope + reference.values, restored.keys_pre_rope + restored.values):
                self.assertLess((x.float() - y.float()).abs().max().item(), 0.05)

    def test_staged_restore_of_a_corrupt_packet_fails_typed(self):
        from reelcache.errors import IntegrityError

        restorer, engine = self.restorer(staged=True)
        backend = mb.ReelCacheConditioningCacheBackend(self.runtime, restorer=restorer, archive=(), heads=HEADS,
                                                    pin_memory=False)
        backend.reload_index()
        damaged, healthy = self.caches[2], self.caches[0]

        def packet_blobs(cache):
            packets = [item["packet"] for item in restorer.history(backend.node_ref(cache.epoch))]
            return {blob for packet in packets for blob in blobs(self.runtime.store, packet)}

        own = packet_blobs(damaged) - packet_blobs(self.caches[1]) - packet_blobs(healthy)
        corrupt(self.runtime.store, max(own, key=self.runtime.store.object_size))
        backend.arm_restore("svd_hevc")
        with self.assertRaisesRegex(IntegrityError, "is damaged and was not restored: IntegrityError"):
            backend.restore(damaged.epoch)
        self.assertEqual(backend.archive_state(damaged.epoch, "svd_hevc"), "quarantined")
        store_restorer, _ = self.restorer(staged=False)
        self.assertTrue(self.same(backend.restore(healthy.epoch), self.restore_all(store_restorer, [healthy])[0]))
        self.assertGreaterEqual(engine.stats()["histories"], 1)

    def test_defaults_bounds_and_attach_report(self):
        self.assertEqual(mb.STAGING_DEFAULTS, {"staging_engine": True, "staging_threads": None,
                                               "staging_prepare_threads": None, "staging_pinned_bytes": 512 << 20})
        threads, prepare = mb.staging_thread_counts()
        self.assertTrue(2 <= threads <= 8)
        self.assertEqual(prepare, max(1, threads // 2))
        self.assertEqual(mb.staging_thread_counts(3, 1), (3, 1))
        for bad in ((0, None), (None, 65), ("2", None)):
            with self.assertRaises(cc.ConditioningCacheError):
                mb.staging_thread_counts(*bad)
        with self.assertRaises(cc.ConditioningCacheError):
            mb.lineage_restorer(self.runtime, SoftwarePool(), "cpu", staging_pinned_bytes=1024)
        with self.assertRaises(cc.ConditioningCacheError):
            mb.lineage_restorer(self.runtime, SoftwarePool(), "cpu", staging_engine="yes")
        _, engine = self.restorer(staged=True)
        self.assertEqual(mb.staging_description(engine), {"enabled": True, "device": "cpu", "threads": 2,
                                                          "prepare_threads": 2, "pinned_bytes": 1 << 22})
        self.assertEqual(mb.staging_description(None)["enabled"], False)
        from reelcache.errors import ConfigurationError
        config = sp.LongCatSessionConfig.from_dict({"store_root": "x"})
        self.assertEqual(config.staging(), mb.STAGING_DEFAULTS)
        for bad in ({"staging_engine": 1}, {"staging_threads": 0}, {"staging_prepare_threads": 65},
                    {"staging_pinned_bytes": 4096}):
            with self.assertRaises(ConfigurationError):
                sp.LongCatSessionConfig.from_dict({"store_root": "x", **bad})
        # a worker without CUDA restores with the software decoder: nothing to stage, reported as such
        provider = sp.create_provider(config={"store_root": str(self.root / "provider")})
        self.addCleanup(provider.close)
        staging = provider.control("capabilities", {})["staging_engine"]
        self.assertEqual(staging["enabled"], False)
        self.assertIn("software restorer", staging["reason"])
        self.assertEqual(provider.startup["staging_engine"], staging)


if __name__ == "__main__":
    unittest.main()
