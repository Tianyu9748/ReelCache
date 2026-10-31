"""Freeze F5 restore cuts (Step 11; coordinator 17:40; P4 breakdown).

* Cut 3: restore plans are cached by head node, stream, next chunk, tokens per frame and cache geometry. A fork at a
  cached head reuses its plan with bitwise the same restore. A durable pause plans the session's next resume ahead.
* Cut 4: a restore the archive says allocates nothing (``restore_allocates``: every attended range a GPU raw copy
  read in place, Step 5c's O4 fast path) runs without the pre-restore relief and the restore section, when the
  device's headroom covers the engine RoPE's scratch the in-place install still takes from the caching allocator
  (coordinator 19:4x: bounded by the synchronous install's 2 x one layer's K; the overlapped install needs one span's
  K). Without the in-place RAW install (a staged install), with a host or compressed range, or with the headroom short
  of the scratch, both stay.
* Cut 6: restored latents go to a CUDA request from pinned memory without a host wait.
* Cut 7: the bridge records its part of the pre-denoise time. A fork does no separate store lookup of its target; an
  archived target is still refused, typed.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import tests.test_sglang_bridge as t
from reelcache.errors import StateError
import reelcache.integrations.sglang.bridge as bridge_module
from reelcache.integrations.sglang.bridge import PlanCache, _BridgeRun, cache_geometry, rope_scratch_bytes


class RestoreCutsTestCase(t.BridgeTestCase):
    representation = "svd_hevc"

    def bridge(self, store="archive", **overrides):
        overrides.setdefault("raw_restore", "off")
        return super().bridge(store, **overrides)

    def session(self, bridge, chunks=3):
        summary, _, _, _ = t.drive(bridge, t.request(t.handle("L", pause_after_chunks=chunks)))
        bridge.drain_all()
        return summary

    def fork(self, bridge, name, checkpoint):
        summary, latents, caches, _ = t.drive(bridge, t.request(t.handle(
            name, "fork", from_session="L", checkpoint=checkpoint, pause_after_chunks=checkpoint + 1)))
        return summary, latents.clone(), {stream: [SimpleNamespace(end=c.end, k=c.k.clone(), v=c.v.clone())
                                                   for c in items] for stream, items in caches.items()}


class PlanCacheTests(RestoreCutsTestCase):
    def test_a_fork_at_a_cached_head_reuses_its_plan_bitwise(self):
        bridge = self.bridge()
        self.session(bridge)
        first, latents_first, caches_first = self.fork(bridge, "a", 2)
        second, latents_second, caches_second = self.fork(bridge, "b", 2)
        self.assertEqual(first["restore"]["plan_cache_hits"], {"cond": False})
        self.assertEqual(second["restore"]["plan_cache_hits"], {"cond": True})
        self.assertEqual(first["restore"]["plans"], second["restore"]["plans"])
        self.assertTrue(torch.equal(latents_first, latents_second))
        self.assertCachesEqual(caches_first, caches_second)
        self.assertEqual(first["chunk_digests"], second["chunk_digests"])
        other, _, _ = self.fork(bridge, "c", 1)  # another head node: another key
        self.assertEqual(other["restore"]["plan_cache_hits"], {"cond": False})
        stats = bridge.control("capabilities")["plan_cache"]  # the pauses plan the forks' next resumes too
        self.assertTrue(stats["hits"] >= 1 and stats["entries"] >= 2 and stats["prepared"] >= 1, stats)
        self.assertLifetimesRetired(bridge, other)

    def test_a_durable_pause_plans_the_next_resume(self):
        bridge = self.bridge()
        summary = self.session(bridge, chunks=2)
        prepared = summary["plan_prepared"]
        self.assertEqual((prepared["chunks"], prepared["already_cached"]), (2, {"cond": False}))
        resumed, _, _, _ = t.drive(bridge, t.request(t.handle("L", "resume", pause_after_chunks=3)))
        self.assertEqual(resumed["restore"]["mode"], "archive")  # new caches: never resident
        self.assertEqual(resumed["restore"]["plan_cache_hits"], {"cond": True})
        self.assertEqual(bridge.plans.describe()["prepared"], 2)  # this pause and the resume's own pause

    def test_the_key_separates_heads_streams_chunks_and_geometries(self):
        small, large = t.FullHistoryCache(4 * t.FRAMES * t.TPF), t.FullHistoryCache(8 * t.FRAMES * t.TPF)
        self.assertNotEqual(cache_geometry(small), cache_geometry(large))
        base = PlanCache.key("n1", "cond", small, (0, 8), 4)
        for other in (PlanCache.key("n2", "cond", small, (0, 8), 4), PlanCache.key("n1", "uncond", small, (0, 8), 4),
                      PlanCache.key("n1", "cond", small, (8, 8), 4), PlanCache.key("n1", "cond", large, (0, 8), 4),
                      PlanCache.key("n1", "cond", small, (0, 8), 2)):
            self.assertNotEqual(base, other)
        cache = PlanCache(capacity=2)
        for index in range(3):
            cache.put(("k", index), index)
        self.assertEqual((cache.get(("k", 0)), cache.get(("k", 2))), (None, 2))
        self.assertEqual(cache.describe()["evicted"], 1)


class GpuRawSectionTests(RestoreCutsTestCase):
    def test_an_all_gpu_raw_restore_read_in_place_skips_the_relief_and_the_section(self):
        bridge = self.bridge("raw", raw_restore="prefer", gpu_raw_budget_bytes=1 << 30)
        bridge.archive.raw_install_in_place = True  # as with Step 5c's O4 fast path
        reference = self.session(bridge, chunks=3)
        summary, _, _ = self.fork(bridge, "g", 2)
        restore = summary["restore"]
        self.assertEqual((restore["lease"]["tiers"]["chunks"], restore["allocates"]), ({"gpu_raw": 2}, False))
        self.assertIn("allocates nothing", restore["pressure"]["skipped"])
        self.assertIsNone(restore["pressure"]["section"])
        self.assertEqual(summary["chunk_digests"]["2"], reference["chunk_digests"]["2"])  # RAW: bitwise native
        self.assertLifetimesRetired(bridge, summary)

    def test_the_rope_scratch_is_checked_against_the_headroom(self):
        bridge = self.bridge("raw", raw_restore="prefer", gpu_raw_budget_bytes=1 << 30)
        bridge.archive.raw_install_in_place = True
        self.session(bridge, chunks=3)
        summary, _, _ = self.fork(bridge, "g", 2)
        scratch = summary["restore"]["scratch"]
        span = t.FRAMES * t.TPF * t.HEADS * t.DIM * 2  # one chunk's K (BF16): the overlapped install's scratch
        self.assertEqual((scratch["span_bytes"], scratch["bytes"]), (span, 2 * 2 * span))  # 2 x the layer's 2 spans
        self.assertEqual((scratch["checked"], scratch["covered"]), (False, True))  # CPU: no device memory to protect
        self.assertEqual(summary["restore"]["archive_allocates"], False)
        seen = []

        def short(device, need):
            seen.append(need)
            return {"checked": True, "free": need - 1, "headroom": need - 1, "covered": False, "seconds": 0.0}

        with mock.patch.object(bridge_module, "scratch_headroom", short):
            kept, _, _ = self.fork(bridge, "s", 2)  # the headroom does not cover the scratch: relief and section stay
        restore = kept["restore"]
        self.assertEqual((restore["archive_allocates"], restore["allocates"], seen), (False, True, [2 * 2 * span]))
        self.assertEqual(restore["scratch"]["headroom"], 2 * 2 * span - 1)
        self.assertNotIn("skipped", restore["pressure"])
        self.assertIsNotNone(restore["pressure"]["section"])
        self.assertEqual(kept["chunk_digests"]["2"], summary["chunk_digests"]["2"])

        def failing(device, need):
            raise RuntimeError("no memory info")

        with mock.patch.object(bridge_module, "scratch_headroom", failing):
            failed, _, _ = self.fork(bridge, "f", 2)  # a failing check keeps them too
        self.assertTrue(failed["restore"]["allocates"])
        self.assertIn("RuntimeError", failed["restore"]["scratch"]["error"])
        plain, _, _, _ = t.drive(bridge, t.request(t.handle("p", "fork", from_session="L", checkpoint=2,
                                                            pause_after_chunks=3, overlap_restore=False)))
        self.assertEqual((plain["restore"]["allocates"], plain["restore"]["overlap"]), (False, None))  # same bound
        self.assertLifetimesRetired(bridge, plain)

    def test_the_scratch_bound_covers_both_installs(self):
        plans = {"cond": SimpleNamespace(spans=[SimpleNamespace(length=4680)] * 8),
                 "uncond": SimpleNamespace(spans=[SimpleNamespace(length=4680)] * 8)}
        scratch = rope_scratch_bytes(plans, num_heads=12, head_dim=128, key_dtype=torch.bfloat16)
        self.assertEqual(scratch["span"], 4680 * 12 * 128 * 2)  # CausVid: 14.4 MB, one chunk's K
        self.assertEqual(scratch["bound"], 2 * 16 * scratch["span"])
        fp32 = rope_scratch_bytes({"cond": plans["cond"]}, num_heads=12, head_dim=128, key_dtype=torch.float32)
        self.assertEqual((fp32["span"], fp32["bound"]), (2 * scratch["span"], 2 * 8 * 2 * scratch["span"]))
        self.assertEqual(rope_scratch_bytes({}, num_heads=12, head_dim=128, key_dtype=torch.bfloat16),
                         {"span": 0, "bound": 0})
        cpu = bridge_module.scratch_headroom(torch.device("cpu"), 1 << 40)
        self.assertEqual((cpu["checked"], cpu["covered"]), (False, True))

    def test_without_the_in_place_raw_install_an_all_gpu_raw_restore_keeps_the_section(self):
        bridge = self.bridge("staged", raw_restore="prefer", gpu_raw_budget_bytes=1 << 30)
        bridge.archive.raw_install_in_place = False  # a staged RAW install (F4's restore_dense)
        self.session(bridge, chunks=3)
        summary, _, _ = self.fork(bridge, "g", 2)
        restore = summary["restore"]
        self.assertEqual((restore["lease"]["tiers"]["chunks"], restore["allocates"]), ({"gpu_raw": 2}, True))
        self.assertNotIn("skipped", restore["pressure"])
        self.assertIsNotNone(restore["pressure"]["section"])

    def test_a_restore_with_a_host_raw_range_keeps_the_relief_and_the_section(self):
        bridge = self.bridge("mixed", raw_restore="prefer")  # GPU budget: two entries, the oldest spills to host
        bridge.archive.raw_install_in_place = True
        self.session(bridge, chunks=3)
        summary, _, _ = self.fork(bridge, "m", 3)
        restore = summary["restore"]
        self.assertIn("host_raw", restore["lease"]["tiers"]["chunks"])
        self.assertTrue(restore["allocates"])
        self.assertNotIn("skipped", restore["pressure"])
        self.assertIsNotNone(restore["pressure"]["section"])

    def test_a_compressed_restore_keeps_the_relief_and_the_section(self):
        bridge = self.bridge("compressed")  # raw_restore="off"
        bridge.archive.raw_install_in_place = True
        self.session(bridge, chunks=2)
        summary, _, _ = self.fork(bridge, "c", 2)
        self.assertTrue(summary["restore"]["allocates"])
        self.assertIsNotNone(summary["restore"]["pressure"]["section"])

    def test_an_archive_without_the_query_keeps_the_relief(self):
        bridge = self.bridge("raw_ref", representation="raw", raw_restore="prefer")  # the RAW reference archive
        self.assertFalse(hasattr(bridge.archive, "restore_allocates"))
        self.session(bridge, chunks=2)
        summary, _, _ = self.fork(bridge, "r", 2)
        self.assertTrue(summary["restore"]["allocates"])
        self.assertNotIn("skipped", summary["restore"]["pressure"])


class LatentsCopyTests(unittest.TestCase):
    def test_restored_latents_reach_a_cuda_request_from_pinned_memory_without_a_host_wait(self):
        region = mock.MagicMock(is_cuda=True, shape=(1, 4, 3, 2, 2), dtype=torch.float32)
        latents = mock.MagicMock()
        latents.__getitem__.return_value = region
        run = SimpleNamespace(_block_starts=[0], target=SimpleNamespace(latents=latents))
        stored = mock.MagicMock(shape=(1, 4, 3, 2, 2), dtype=torch.float32, device=torch.device("cpu"))
        stored.pin_memory.return_value = "pinned"
        _BridgeRun._install_latents(run, 0, stored, 3)
        region.copy_.assert_called_once_with("pinned", non_blocking=True)

    def test_cpu_latents_copy_as_before(self):
        latents = torch.zeros(1, 4, 6, 2, 2)
        run = SimpleNamespace(_block_starts=[0, 3], target=SimpleNamespace(latents=latents))
        stored = torch.ones(1, 4, 3, 2, 2)
        _BridgeRun._install_latents(run, 1, stored, 3)
        self.assertTrue(torch.equal(latents[:, :, 3:], stored) and not latents[:, :, :3].any())


class ForkPathTests(RestoreCutsTestCase):
    def test_a_fork_records_the_bridges_pre_denoise_steps(self):
        bridge = self.bridge()
        self.session(bridge)
        summary, _, _ = self.fork(bridge, "a", 2)
        begin = summary["begin"]
        self.assertEqual(begin["action"], "fork")
        for key in ("checks_seconds", "controller_seconds", "release_idle_seconds", "before_run_seconds"):
            self.assertGreaterEqual(begin[key], 0.0)
        self.assertGreaterEqual(begin["before_run_seconds"], begin["controller_seconds"])

    def test_a_fork_does_no_target_lookup_and_still_refuses_an_archived_target(self):
        bridge = self.bridge()
        self.session(bridge)
        with mock.patch.object(bridge.archive, "head", wraps=bridge.archive.head) as head:
            bridge.controller.fork("L", "a", checkpoint=2)
        head.assert_not_called()
        bridge.control("close", {"session_id": "a"})  # unloaded, still archived
        archived = bridge.archive.head("a")
        self.assertIsNotNone(archived)
        with self.assertRaisesRegex(StateError, "fork target 'a' already exists"):
            bridge.controller.fork("L", "a", checkpoint=1)
        self.assertEqual(bridge.archive.head("a"), archived)  # a refused fork moves no head
        with self.assertRaisesRegex(StateError, "fork target 'L' already exists"):
            bridge.controller.fork("L", "L", checkpoint=2)  # loaded: the in-memory check


if __name__ == "__main__":
    unittest.main()
