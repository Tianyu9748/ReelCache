"""CR-17-4 (SGLang Step 17): the bridge stages cold histories with Step 5c's staging engine by default.

The bridge built its packed restorer without ``LineageRestorer(staging_engine=...)``, so production cold restores and
session-open prewarms used the store path (``node_pack``: per-tile assembly and checks), which lost to recompute at
h <= 2 on every GPU at freeze A, while the staged path beat it in all of Step 5c's 108 rows (268/268 bitwise). With
``decoder="nvdec_packed"`` the bridge now creates one ``StagingEngine`` per worker on its GPU (``staging_engine``,
default on; ``staging_threads`` / ``staging_prepare_threads`` / ``staging_pinned_bytes``), passes it to every
restorer of the worker, warms it at attach (pinned ring, threads), reports it in ``describe()`` / ``stats()`` and
closes it after the restorers. The engine runs no PyTorch CPU compute op (CR-17-2: reads, hashlib and memmem on its
bounded threads; NumPy affines; CUDA copies on its own stream).

CPU tests run the bridge with the packed restorer on the software decoder pool (the ``nvdec_packed`` wiring with a CPU
pool): staged cold restores and staged session-open prewarms restore bitwise the store path's KV, through requests
that then continue generating. The CUDA test (GPU jobs only) checks the production wiring.
"""

from __future__ import annotations

import os
import shutil
import unittest
from unittest import mock

import torch

from reelcache.archive_restore import LineageRestorer
from reelcache.errors import ConfigurationError
from reelcache.integrations.sglang import CausalWanKVBridge, ReelCacheRuntimeConfig
from reelcache.integrations.sglang.lineage_archive import staging_threads
from reelcache.restore_staging import StagingEngine
import tests.test_sglang_bridge as t
from tests.test_packed_nvdec import SoftwarePool

STREAMS = ("cond", "uncond")
SESSIONS = (("s1", "A", 3), ("s2", "B", 2))


class StagingConfigTests(unittest.TestCase):
    def test_defaults_validation_and_automatic_threads(self):
        config = ReelCacheRuntimeConfig.from_dict({"store_root": "archive"})
        self.assertEqual((config.staging_engine, config.staging_threads, config.staging_prepare_threads,
                          config.staging_pinned_bytes), (True, None, None, 512 << 20))
        for cpus, expected in ((1, (2, 1)), (3, (2, 1)), (8, (4, 2)), (16, (8, 4)), (128, (8, 4))):
            with self.subTest(cpus=cpus), mock.patch("os.sched_getaffinity", return_value=set(range(cpus))):
                self.assertEqual(staging_threads(config), expected)
        explicit = ReelCacheRuntimeConfig.from_dict({"store_root": "archive", "staging_threads": 3,
                                                  "staging_prepare_threads": 5})
        self.assertEqual(staging_threads(explicit), (3, 5))
        for bad in ({"staging_engine": 1}, {"staging_threads": 0}, {"staging_prepare_threads": 65},
                    {"staging_threads": 2.0}, {"staging_pinned_bytes": 4096}, {"staging_pinned_bytes": None}):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                ReelCacheRuntimeConfig.from_dict({"store_root": "archive", **bad})


class BridgeStagingTests(t.BridgeTestCase):
    representation = "svd_hevc"

    def packed(self, bridge, staged):
        """The bridge's packed restorer on the software decoder pool, staged (as ``nvdec_packed`` with the default
        ``staging_engine``) or on the store path; the archive closes the engine after the restorer."""
        archive = bridge.archive
        engine = StagingEngine("cpu", threads=2, prepare_threads=1, pinned_bytes=1 << 20) if staged else None
        pool = SoftwarePool()
        self.addCleanup(pool.close)
        archive.restorer = LineageRestorer(archive.runtime, pool, arena_bytes=1 << 30, staging_engine=engine)
        archive.staging_engine = engine
        return engine

    def test_staged_cold_restores_and_prewarms_are_bitwise_the_store_path(self):
        source = self.bridge(store="source")
        self.assertEqual(source.startup["archive"]["staging_engine"],
                         {"enabled": False, "reason": "software decoder: no packed device restore"})
        for session_id, prompt, chunks in SESSIONS:
            t.drive(source, t.request(t.handle(session_id, pause_after_chunks=chunks), prompt=prompt, streams=STREAMS))
        source.close()
        results = {}
        for prewarm in (False, True):
            for staged in (False, True):
                name = f"copy-{prewarm}-{staged}"
                shutil.copytree(self.workspace / "source", self.workspace / name)  # the same persisted state
                bridge = self.bridge(store=name, prewarm=prewarm)
                engine = self.packed(bridge, staged)
                for session_id, prompt, chunks in SESSIONS:
                    summary, latents, caches, _ = t.drive(
                        bridge, t.request(t.handle(session_id, "resume"), prompt=prompt, streams=STREAMS))
                    self.assertEqual((summary["resumed_from_chunks"], summary["generated_chunks"]), (chunks, 4))
                    results[(prewarm, staged, session_id)] = (latents, caches)
                if staged:  # each session's cold history was staged once (by the prewarm, or by the restore)
                    self.assertEqual(engine.stats()["histories"], 2)
                    self.assertEqual(bridge.archive.staging_description(),
                                     {"enabled": True, "device": "cpu", "threads": 2, "prepare_threads": 1,
                                      "pinned_bytes": 1 << 20})
                    self.assertEqual(bridge.describe()["archive_stats"]["staging_engine"]["histories"], 2)
                self.assertLifetimesRetired(bridge)
                bridge.close()
                if staged:
                    self.assertTrue(engine._closed)
        for (prewarm, staged, session_id), (latents, caches) in results.items():
            reference_latents, reference_caches = results[(False, False, session_id)]
            with self.subTest(prewarm=prewarm, staged=staged, session=session_id):
                self.assertTrue(torch.equal(latents, reference_latents))
                for stream in reference_caches:
                    for got, want in zip(caches[stream], reference_caches[stream]):
                        self.assertEqual(got.end, want.end)
                        self.assertTrue(torch.equal(got.k[:, :got.end], want.k[:, :want.end]))
                        self.assertTrue(torch.equal(got.v[:, :got.end], want.v[:, :want.end]))


if os.environ.get("REELCACHE_STEP7_GPU_TESTS") == "1":
    class CUDABridgeStagingTests(t.BridgeTestCase):
        """The production wiring with ``decoder="nvdec_packed"`` on CUDA (inside GPU jobs only)."""

        def cuda_bridge(self, store, **overrides):
            values = {"store_root": str(self.workspace / store), "representation": "svd_hevc", "fit_device": "cpu",
                      "energy_key": 0.98, "energy_value": 0.95, "x265_threads": 2, "encode_workers": 2,
                      "staging_pinned_bytes": 8 << 20, "nvdec_workers": 2}
            values.update(overrides)
            bridge = CausalWanKVBridge(ReelCacheRuntimeConfig.from_dict(values), t.PROFILE, device="cuda",
                                       model_path="toy/causal-wan")
            self.addCleanup(bridge._close_at_exit)
            return bridge

        def test_the_nvdec_bridge_builds_warms_shares_and_closes_one_engine(self):
            if not torch.cuda.is_available():
                raise AssertionError("REELCACHE_STEP7_GPU_TESTS=1 requires a CUDA device")
            bridge = self.cuda_bridge("on")
            archive = bridge.archive
            engine = archive.staging_engine
            threads, prepare_threads = staging_threads(bridge.config)
            self.assertIsInstance(engine, StagingEngine)
            self.assertIs(archive.restorer.staging_engine, engine)
            self.assertEqual(bridge.startup["archive"]["staging_engine"],
                             {"enabled": True, "device": str(archive.device), "threads": threads,
                              "prepare_threads": prepare_threads, "pinned_bytes": 8 << 20})
            component = bridge.startup["warmup"]["components"]["staging_engine"]
            self.assertNotIn("error", component)
            self.assertEqual(component["detail"]["pinned_bytes"], 8 << 20)
            self.assertIsNotNone(engine.ring.buffer)  # the pinned ring was allocated at attach
            self.assertEqual(bridge.describe()["archive_stats"]["staging_engine"]["histories"], 0)
            bridge.close()
            self.assertTrue(engine._closed)
            off = self.cuda_bridge("off", staging_engine=False)
            self.assertIsNone(off.archive.staging_engine)
            self.assertIsNone(off.archive.restorer.staging_engine)
            self.assertEqual(off.startup["archive"]["staging_engine"],
                             {"enabled": False, "reason": "staging_engine=False: cold restores use the store path"})
            self.assertNotIn("staging_engine", off.startup["warmup"]["components"])


if __name__ == "__main__":
    unittest.main()
