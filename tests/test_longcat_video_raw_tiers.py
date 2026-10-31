"""CPU tests of the freeze-D raw tiers of the LongCat-Video port (Step 13; Step 7 ``DESIGN_D_tiers`` §6, §9, §11).

Per-epoch tiers (``raw_tiers.py``): budgets from fake cgroup v1/v2 trees (review amendment 2), the host-slot policy
of the provider (no minimum above the safe headroom), the GPU raw budget, NVENC selection with a fake probe (no
silent fallback, amendment 5), ``schedule: auto`` under the tier, the tier index (identity token -> node_ref,
eviction only after publication, LRU by access, invariant I1, prepare-on-evict), the backend's bitwise
``gpu_raw`` / ``host_raw`` restores and records, the provider's segment records and controls, ``create_provider``'s
CPU path, and the context-parallel tier (``cp_tiers.py``: one epoch-wide tier, preflight, collective commit and
rollback; 2-rank Gloo). No SGLang, no GPU; artifacts stay in the checkout.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import tempfile
import unittest
import uuid
from datetime import timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist

from reelcache_sglang_models.longcat_video import conditioning_cache as cc
from reelcache_sglang_models.longcat_video import reelcache_backend as mb
from reelcache_sglang_models.longcat_video import raw_tiers as rt
from reelcache_sglang_models.longcat_video import session_provider as sp
from tests.test_longcat_video_reelcache_backend import synthetic_cache
from tests.test_longcat_video_session_provider import (
    ARCH, FINGERPRINT, GEOMETRY, GRID, HEADS, conditioning_cache, frames_digest, initial_frames, join_restore,
    next_frames,
)
from reelcache_sglang_models.longcat_video.segment_resume import generator_state_hex, set_generator_state_hex

GIB = 1 << 30
REPO = Path(__file__).resolve().parents[1]
EPOCH_720P = 11_349_786_624


def artifacts(name: str) -> Path:
    """A fresh directory in the checkout's ``.test-artifacts``; in a read-only pinned tree (GPU jobs run the CUDA
    tests from the frozen tree) the job's ``TMPDIR`` (node-local scratch) instead."""
    try:
        base = REPO / ".test-artifacts"
        base.mkdir(parents=True, exist_ok=True)
        root = base / f"{name}-{uuid.uuid4().hex}"
        root.mkdir()
        return root
    except OSError:
        return Path(tempfile.mkdtemp(prefix=f"{name}-"))


def memory(headroom_gib: float) -> dict:
    return {"version": 2, "path": "/job", "limit_bytes": int(200 * GIB), "current_bytes": 0,
            "headroom_bytes": int(headroom_gib * GIB), "source": "cgroup v2 memory.max"}


def fake_cgroup(root: Path, *, version: int, limit: int, current: int, cache: int = 0) -> tuple[Path, Path]:
    """A Slurm-like cgroup tree: the job sets the limit, the leaf (step/task) reads ``max`` (v1: a huge limit); the
    job's ``memory.stat`` holds ``cache`` bytes of page cache, a quarter of it dirty or shared."""
    proc, sysfs = root / "proc", root / "sys"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text("MemTotal:  4096 kB\nMemAvailable:  2048 kB\n")
    if version == 2:
        (proc / "self" / "cgroup").write_text("0::/system.slice/slurmstepd.scope/job_7/step_0/user/task_0\n")
        job = sysfs / "system.slice/slurmstepd.scope/job_7"
        leaf = job / "step_0/user/task_0"
        leaf.mkdir(parents=True)
        (job / "memory.max").write_text(f"{limit}\n")
        (job / "memory.current").write_text(f"{current}\n")
        (job / "memory.stat").write_text(f"anon 1\nfile {cache}\nshmem {cache // 8}\nfile_dirty {cache // 16}\n"
                                         f"file_writeback {cache // 16}\n")
        (leaf / "memory.max").write_text("max\n")
        (leaf / "memory.current").write_text(f"{current // 2}\n")
    else:
        (proc / "self" / "cgroup").write_text("12:memory:/slurm/uid_1/job_7/step_0\n5:cpu,cpuacct:/slurm\n")
        job = sysfs / "memory" / "slurm/uid_1/job_7"
        step = job / "step_0"
        step.mkdir(parents=True)
        (job / "memory.limit_in_bytes").write_text(f"{limit}\n")
        (job / "memory.usage_in_bytes").write_text(f"{current}\n")
        (job / "memory.stat").write_text(f"cache 7\ntotal_cache {cache}\ntotal_shmem {cache // 8}\n"
                                         f"total_dirty {cache // 16}\ntotal_writeback {cache // 16}\n")
        (step / "memory.limit_in_bytes").write_text(f"{1 << 62}\n")
        (step / "memory.usage_in_bytes").write_text(f"{current // 2}\n")
    return proc, sysfs


class FakeEvent:
    """A D2H completion event that is still running until synchronized."""

    def __init__(self):
        self.done = False
        self.synchronized = 0

    def query(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.synchronized += 1
        self.done = True


class EnsureRecorder:
    """A restorer proxy with Step 7's ``ensure_prepared`` (dev22a), recording its calls."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def ensure_prepared(self, packet_refs):
        refs = list(packet_refs)
        self.calls.append(refs)
        return {"touched": 0, "prepared": len(refs), "failed": 0}


def same(a: cc.LongCatConditioningCache, b: cc.LongCatConditioningCache) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a.keys_pre_rope + a.values, b.keys_pre_rope + b.values))


def tiny_epoch_bytes() -> int:
    return synthetic_cache(0, 0).nbytes


# ---------------------------------------------------------------------------------------------------- budgets
class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.root = artifacts("lc-tiers-budget")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_cgroup_memory_v2_v1_and_meminfo_fallback(self):
        proc, sysfs = fake_cgroup(self.root / "v2", version=2, limit=110 * GIB, current=40 * GIB)
        record = rt.cgroup_memory(proc_root=proc, sys_root=sysfs)
        self.assertEqual((record["version"], record["limit_bytes"], record["current_bytes"]), ("v2", 110 * GIB, 40 * GIB))
        self.assertEqual(record["headroom_bytes"], 70 * GIB)
        self.assertEqual(record["path"], "/system.slice/slurmstepd.scope/job_7")  # the leaf reads "max"
        # Step 7 R5: the weights' clean page cache fills memory.current to the limit, but it is reclaimed first
        proc, sysfs = fake_cgroup(self.root / "v2c", version=2, limit=110 * GIB, current=108 * GIB, cache=64 * GIB)
        record = rt.cgroup_memory(proc_root=proc, sys_root=sysfs)
        self.assertEqual(record["reclaimable_cache_bytes"], 64 * GIB - 8 * GIB - 4 * GIB - 4 * GIB)
        self.assertEqual(record["headroom_bytes"], 2 * GIB + 48 * GIB)
        proc, sysfs = fake_cgroup(self.root / "v1", version=1, limit=64 * GIB, current=10 * GIB, cache=16 * GIB)
        record = rt.cgroup_memory(proc_root=proc, sys_root=sysfs)
        self.assertEqual((record["version"], record["headroom_bytes"]), ("v1", 54 * GIB + 12 * GIB))  # step: no limit
        (proc / "self" / "cgroup").write_text("0::/\n")
        record = rt.cgroup_memory(proc_root=proc, sys_root=self.root / "nothing")
        self.assertEqual((record["version"], record["headroom_bytes"]), (None, 2048 * 1024))
        self.assertIn("MemAvailable", record["source"])

    def test_host_budget_formula_workers_and_explicit_budgets(self):
        epoch, fixed = EPOCH_720P, 8 * GIB
        auto = rt.host_raw_budget(epoch_bytes=epoch, fixed_bytes=fixed, memory=memory(70))
        self.assertEqual(auto["budget_bytes"], int(0.5 * (70 * GIB - fixed)))  # Step 7 R5: fraction of the rest
        self.assertEqual((auto["slots"], auto["source"], auto["fits"]), (2, "auto", True))
        two = rt.host_raw_budget(epoch_bytes=epoch, workers=2, fixed_bytes=fixed, memory=memory(70))
        self.assertEqual(two["physical_bytes"], 35 * GIB - fixed)  # the worker's share, then its own pools
        self.assertEqual(two["slots"], 1)
        capped = rt.host_raw_budget(epoch_bytes=epoch, memory=memory(1000))
        self.assertEqual((capped["budget_bytes"], capped["slots"]), (rt.HOST_RAW_CAP_BYTES, 6))
        none = rt.host_raw_budget(epoch_bytes=epoch, fixed_bytes=fixed, memory=memory(20))
        self.assertEqual(none["slots"], 0)  # no minimum above the safe headroom
        explicit = rt.host_raw_budget(epoch_bytes=epoch, fixed_bytes=fixed, memory=memory(20), explicit_slots=2)
        self.assertEqual((explicit["source"], explicit["fits"]), ("capture_slots", False))
        explicit = rt.host_raw_budget(epoch_bytes=epoch, fixed_bytes=fixed, memory=memory(40), explicit_bytes=epoch)
        self.assertEqual((explicit["slots"], explicit["fits"]), (1, True))
        for bad in ({"epoch_bytes": 0}, {"epoch_bytes": epoch, "workers": 0}):
            with self.assertRaises(ValueError):
                rt.host_raw_budget(memory=memory(10), **bad)
        with mock.patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "2"}):
            self.assertEqual((rt.local_workers(), rt.local_workers(3)), (2, 3))
        with mock.patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "x"}):
            self.assertEqual(rt.local_workers(), 1)

    def test_gpu_raw_budget_a100_rtx_and_the_nvenc_reserve(self):
        model = mb.model_identity()
        self.assertEqual(mb.epoch_nbytes(model), EPOCH_720P)
        resident = int(37.29 * GIB)
        a100 = rt.gpu_raw_budget(total_bytes=int(79.25 * GIB), resident_bytes=resident, epoch_bytes=EPOCH_720P)
        self.assertEqual((a100["epochs"], a100["budget_bytes"]), (0, 0))
        rtx = rt.gpu_raw_budget(total_bytes=int(94.97 * GIB), resident_bytes=resident, epoch_bytes=EPOCH_720P)
        self.assertEqual((rtx["epochs"], rtx["budget_bytes"]), (1, EPOCH_720P))
        reserved = rt.gpu_raw_budget(total_bytes=int(94.97 * GIB), resident_bytes=resident, epoch_bytes=EPOCH_720P,
                                     reserved_bytes=7 * GIB)
        self.assertEqual(reserved["epochs"], 0)
        explicit = rt.gpu_raw_budget(total_bytes=int(79.25 * GIB), resident_bytes=resident, epoch_bytes=EPOCH_720P,
                                     explicit_bytes=EPOCH_720P)
        self.assertEqual((explicit["source"], explicit["epochs"], explicit["fits"]), ("gpu_raw_budget_bytes", 1, False))

    def test_provider_host_slot_policy(self):
        from reelcache.errors import ConfigurationError

        def plan(headroom, **config):
            cfg = sp.LongCatSessionConfig.from_dict({"store_root": "x", **config})
            fixed = sp.host_fixed_bytes(cfg, cuda=True)["bytes"]
            return sp.plan_host_slots(cfg, epoch_bytes=EPOCH_720P, fixed_bytes=fixed, memory=memory(headroom))

        cfg = sp.LongCatSessionConfig.from_dict({"store_root": "x"})
        fixed = sp.host_fixed_bytes(cfg, cuda=True)
        self.assertEqual(fixed["parts"], {"staging_ring": 512 << 20, "canvas_pool": 2 * rt.CANVAS_STAGING_BYTES,
                                          "finalize_workspace": rt.FINALIZE_WORKSPACE_BYTES,
                                          "prepared_cache": 4 * GIB, "nvenc_probe": 0})
        self.assertEqual(sp.host_fixed_bytes(cfg, cuda=False)["parts"]["prepared_cache"], 0)
        raw = sp.LongCatSessionConfig.from_dict({"store_root": "x", "representation": "raw"})
        self.assertEqual(sp.host_fixed_bytes(raw, cuda=True)["parts"]["finalize_workspace"], 0)  # nothing finalized
        self.assertLess(741_826_560, rt.FINALIZE_WORKSPACE_BYTES)  # E 4421998's measured peak fits
        c3 = plan(5, raw_tier=False)
        self.assertEqual((c3["raw_tier"], c3["capture_slots"], c3["archival"]), (False, 4, True))  # C3: unchecked
        self.assertEqual(plan(5, raw_tier=False, capture_slots=2)["capture_slots"], 2)
        on = plan(70)
        self.assertEqual((on["raw_tier"], on["capture_slots"], on["source"]), (True, 2, "auto"))
        one = plan(30)
        self.assertEqual((one["raw_tier"], one["capture_slots"]), (True, 1))
        degraded = plan(25)  # half of the share holds no epoch, the share itself holds one: C3 back-pressure
        self.assertEqual((degraded["raw_tier"], degraded["capture_slots"], degraded["archival"]), (False, 1, True))
        self.assertIn("raw tier disabled", degraded["reason"])
        off = plan(15)
        self.assertEqual((off["raw_tier"], off["capture_slots"], off["archival"]), (False, 0, False))
        self.assertIn("archival is off", off["reason"])
        zero = plan(70, host_raw_budget_bytes=0)  # Step 7: 0 = no host tier
        self.assertEqual((zero["raw_tier"], zero["capture_slots"]), (False, 4))
        self.assertEqual(plan(40, capture_slots=1)["capture_slots"], 1)
        with self.assertRaisesRegex(ConfigurationError, "capture_slots asks for"):
            plan(20, capture_slots=2)
        with self.assertRaisesRegex(ConfigurationError, "host_raw_budget_bytes asks for"):
            plan(20, host_raw_budget_bytes=3 * EPOCH_720P)
        self.assertEqual(plan(70, raw_tier_workers=2)["capture_slots"], 1)
        for bad in ({"raw_tier": 1}, {"raw_restore": "always"}, {"encoder": "qsv"}, {"gpu_raw_budget_bytes": -1},
                    {"host_raw_budget_bytes": 1.5}, {"raw_tier_workers": 0}, {"capture_slots": 0},
                    {"capture_slots": 1, "host_raw_budget_bytes": EPOCH_720P}, {"host_raw_pinned": "yes"}):
            with self.assertRaises(ConfigurationError, msg=str(bad)):
                sp.LongCatSessionConfig.from_dict({"store_root": "x", **bad})
        gpu = sp.plan_gpu_raw(cfg, enabled=True, total_bytes=int(94.97 * GIB), resident_bytes=int(37.29 * GIB),
                              epoch_bytes=EPOCH_720P)
        self.assertEqual(gpu["epochs"], 1)
        # d5: the overlapped restore's step-0 working set joins the generation need while overlap_restore is on
        self.assertEqual(gpu["need_bytes"], rt.SECTION_NEED_BYTES + rt.OVERLAP_RESTORE_NEED_BYTES)
        plain_cfg = sp.LongCatSessionConfig.from_dict({"store_root": "x", "overlap_restore": False})
        for total, epochs in ((79.25, 0), (94.97, 1)):  # A100-80GB, RTX PRO 6000: the same epochs either way
            for config in (cfg, plain_cfg):
                record = sp.plan_gpu_raw(config, enabled=True, total_bytes=int(total * GIB),
                                         resident_bytes=int(37.29 * GIB), epoch_bytes=EPOCH_720P)
                self.assertEqual(record["epochs"], epochs, (total, config.overlap_restore))
        self.assertEqual(sp.plan_gpu_raw(plain_cfg, enabled=True, total_bytes=int(94.97 * GIB),
                                         resident_bytes=int(37.29 * GIB), epoch_bytes=EPOCH_720P)["need_bytes"],
                         rt.SECTION_NEED_BYTES)
        self.assertEqual(sp.plan_gpu_raw(cfg, enabled=False, total_bytes=int(94.97 * GIB), resident_bytes=0,
                                         epoch_bytes=EPOCH_720P)["budget_bytes"], 0)

    def test_schedule_auto_is_eager_under_the_raw_tier(self):
        resident = int(37.4 * GIB)

        def decide(total, *, raw_tier, device="cuda", representation="svd_hevc", requested="auto"):
            return sp.resolve_schedule(requested, defer_planning=False, plan_device=device, plan_memory_bytes=8 * GIB,
                                       resident_bytes=resident, total_bytes=total, representation=representation,
                                       raw_tier=raw_tier)

        a100 = decide(int(79.25 * GIB), raw_tier=True)
        self.assertEqual((a100["chosen"], a100["defer_planning"], a100["release_planner_memory"]), ("eager", True, True))
        self.assertIn("gated out of every request", a100["reason"])
        rtx = decide(int(94.97 * GIB), raw_tier=True)
        self.assertEqual((rtx["chosen"], rtx["defer_planning"]), ("eager", False))
        self.assertEqual(decide(0, raw_tier=True, device="cpu")["chosen"], "eager")
        self.assertEqual(decide(int(79.25 * GIB), raw_tier=True, representation="raw")["chosen"], "deferred")
        self.assertEqual(decide(int(79.25 * GIB), raw_tier=False)["chosen"], "deferred")  # C3/g7 unchanged
        self.assertEqual(decide(int(79.25 * GIB), raw_tier=True, requested="deferred")["chosen"], "deferred")


# ---------------------------------------------------------------------------------------------------- encoder
class EncoderTests(unittest.TestCase):
    class Encoder:
        implementation = {"gpu": "NVIDIA RTX PRO 6000", "driver_version": "580.00"}
        version = "NVENC HEVC via PyNvVideoCodec (driver 580.00)"
        qualification = {"passed": True}

        def __init__(self, gpu_id):
            self.gpu_id = gpu_id

        def encoder_record(self, pix_fmt):
            return {"encoder": "nvenc", "pix_fmt": pix_fmt, "implementation": dict(self.implementation)}

    def test_select_encoder_probes_once_without_silent_fallback(self):
        from reelcache.errors import UnsupportedError

        calls = []

        def missing(gpu_id):
            calls.append(gpu_id)
            raise UnsupportedError("no NVENC on this GPU (A100)")

        encoder, record = mb.select_encoder("x265", 0, factory=missing)
        self.assertIsNone(encoder)
        self.assertEqual((record["selected"], calls), ("x265", []))  # no probe
        encoder, record = mb.select_encoder("auto", 0, factory=missing)
        self.assertIsNone(encoder)
        self.assertEqual(record["selected"], "x265")
        self.assertIn("NVENC unavailable", record["reason"])
        with self.assertRaisesRegex(cc.ConditioningCacheError, "nvenc' requested but NVENC is unavailable"):
            mb.select_encoder("nvenc", 0, factory=missing)
        encoder, record = mb.select_encoder("auto", 0, factory=self.Encoder)
        self.assertIsInstance(encoder, self.Encoder)
        self.assertEqual((record["selected"], record["device"], record["driver"]),
                         ("nvenc", "NVIDIA RTX PRO 6000", "580.00"))
        self.assertEqual(record["encoder_record"]["pix_fmt"], "yuv444p")
        if not torch.cuda.is_available():
            self.assertEqual(mb.select_encoder("auto", None)[1]["reason"], "auto: no CUDA device")
            with self.assertRaisesRegex(cc.ConditioningCacheError, "needs a CUDA device"):
                mb.select_encoder("nvenc", None)
        with self.assertRaises(cc.ConditioningCacheError):
            mb.select_encoder("vaapi", 0)


# ---------------------------------------------------------------------------------------------------- index
class TierIndexTests(unittest.TestCase):
    SHAPES = {"k": (4, 8), "v": (4, 8)}

    def entry(self, index, token, key, *, nbytes=100, host=True, event=None):
        slot, buffers = index.acquire(token, self.SHAPES, torch.float32) if host else (None, None)
        return index.add(rt.RawEntry(token, key, None, nbytes, slot=slot, host=buffers, host_event=event))

    def test_identity_publication_eviction_and_lru_by_access(self):
        evicted = []
        index = rt.EpochTierIndex(2, pin=False, retain=True, on_evict=evicted.append)
        a = self.entry(index, rt.new_token(), ("k", "a"))
        b = self.entry(index, rt.new_token(), ("k", "b"))
        self.assertNotEqual(a.token, b.token)
        with self.assertRaises(rt.HostFull):  # unpublished copies are never evicted (I1)
            index.acquire(rt.new_token(), self.SHAPES, torch.float32)
        self.assertEqual(index.free_slots(), 0)
        index.published(a.token, "node:a")
        index.published(b.token, "node:b")
        self.assertIs(index.by_ref("node:a"), a)
        self.assertEqual(index.free_slots(), 2)  # published entries are evictable
        found, tier = index.lookup(("k", "a"))  # a restore hit touches a: b is now least recently used
        self.assertEqual((found, tier), (a, "host_raw"))
        index.unpin(found)
        c = self.entry(index, rt.new_token(), ("k", "c"))
        self.assertEqual(evicted, [b])
        self.assertIsNone(index.by_ref("node:b"))
        self.assertIsNone(index.by_key(("k", "b")))
        self.assertEqual(index.lookup(("k", "b")), (None, None))
        stats = index.stats()
        self.assertEqual((stats["evictions"]["host"], stats["retained_after_publication"]), (1, 2))
        # a regenerated epoch with the same content key takes a new token; retiring the old one keeps the new
        again = self.entry(index, rt.new_token(), ("k", "c"))  # evicts a (published)
        index.retire(c.token)
        self.assertIs(index.by_key(("k", "c")), again)
        # a pinned published entry is not evicted
        index.published(again.token, "node:c2")
        pinned, _ = index.lookup(("k", "c"))
        with self.assertRaises(rt.HostFull):
            index.acquire(rt.new_token(), self.SHAPES, torch.float32)
            index.acquire(rt.new_token(), self.SHAPES, torch.float32)
        index.unpin(pinned)

    def test_without_retain_a_slot_is_freed_at_publication(self):
        index = rt.EpochTierIndex(1, pin=False, retain=False)
        a = self.entry(index, rt.new_token(), ("k", "a"))
        self.assertEqual(index.free_slots(), 0)
        index.published(a.token, "node:a")
        self.assertEqual(index.free_slots(), 1)
        self.assertIsNone(index.by_key(("k", "a")))
        self.assertFalse(index.retain_gpu(a.token, (), ()))  # no GPU budget: nothing retained

    def test_gpu_budget_lru_and_invariant_i1(self):
        evicted = []
        index = rt.EpochTierIndex(3, pin=False, retain=True, gpu_budget_bytes=100, on_evict=evicted.append)
        first_event = FakeEvent()
        a = self.entry(index, rt.new_token(), ("k", "a"), event=first_event)
        tensors = (torch.zeros(2), torch.ones(2))
        self.assertTrue(index.retain_gpu(a.token, (tensors[0],), (tensors[1],)))
        self.assertEqual(index.stats()["entries"]["resident"], 1)
        self.assertEqual(index.lookup(("k", "a"))[1], "host_raw")  # a resident copy is not served as gpu_raw
        index.unpin(a)
        index.released(("k", "a"))  # the segment ended: a's copy is gpu_raw within the budget
        found, tier = index.lookup(("k", "a"))
        self.assertEqual(tier, "gpu_raw")
        index.unpin(found, resident=True)  # restored zero-copy: resident again
        index.release_all()
        b = self.entry(index, rt.new_token(), ("k", "b"))
        index.retain_gpu(b.token, (torch.zeros(2),), (torch.ones(2),))
        index.released(("k", "b"))  # two retained copies, budget one: a (LRU) loses its GPU copy
        self.assertIsNone(a.gpu)
        self.assertEqual(first_event.synchronized, 1)  # unpublished: its host D2H completed before the drop (I1)
        self.assertEqual(evicted, [])  # a still has its host copy
        self.assertIsNotNone(b.gpu)
        stats = index.stats()  # GPU raw bytes = retained copies beyond the resident one (never above the budget)
        self.assertEqual((stats["evictions"]["gpu"], stats["gpu_bytes"], stats["gpu_peak_bytes"]), (1, 100, 100))
        index.reset_peaks()
        self.assertEqual(index.stats()["gpu_peak_bytes"], 100)
        self.assertEqual(index.enforce_gpu_budget(0), [b])  # A100: budget 0 drops every retained copy
        index.published(a.token, "node:a")
        index.published(b.token, "node:b")
        dropped = index.drop()
        self.assertEqual(dropped["entries"], 2)
        self.assertEqual(sorted(e.token for e in evicted), sorted([a.token, b.token]))  # prepare-on-evict
        self.assertEqual(index.stats()["entries"]["total"], 0)
        c = self.entry(index, rt.new_token(), ("k", "c"))
        index.retain_gpu(c.token, (torch.zeros(2),), (torch.ones(2),))
        index.released(("k", "c"))
        self.assertEqual(index.demote()["entries"], 1)  # GPU raw -> host raw
        self.assertIsNone(c.gpu)
        self.assertIsNotNone(c.host)
        index.note_backpressure(0.5, "host_full")
        stats = index.stats()
        self.assertEqual((stats["last_resort_backpressure_waits"], stats["last_resort_backpressure_causes"]),
                         (1, {"host_full": 1}))
        index.close()
        self.assertEqual(index.stats()["host_bytes"], 0)


# ---------------------------------------------------------------------------------------------------- backend
class BackendTierTests(unittest.TestCase):
    def setUp(self):
        from reelcache.archive_pipeline import ArchivePipeline
        from reelcache.archive_restore import SoftwareLineageRestorer
        from reelcache.lineage_commit import ArchiveOptions
        from reelcache.manager import KVRuntime
        from reelcache.predictor_registry import RankPredictorRegistry

        self.root = artifacts("lc-tiers-backend")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.model = mb.model_identity("test-longcat-fingerprint", ARCH)
        self.runtime = KVRuntime(self.root / "store", self.model)
        self.pipeline = ArchivePipeline(self.runtime, method=mb.default_method(),
                                        registry=RankPredictorRegistry("longcat-tier-test"),
                                        options=ArchiveOptions(encode_threads=1), device="cpu", encode_workers=1)
        self.restorer = EnsureRecorder(SoftwareLineageRestorer(self.runtime))
        self.addCleanup(self._close)

    def _close(self):
        try:
            if not self.pipeline._closed:
                self.pipeline.close(drain=False)
        finally:
            if not self.runtime._closed:
                self.runtime.close()

    def backend(self, **options):
        defaults = {"archive": ("svd_hevc",), "heads": HEADS, "pin_memory": False, "capture_slots": 2,
                    "schedule": "eager", "raw_tier": True, "gpu_raw_budget_bytes": tiny_epoch_bytes()}
        return mb.ReelCacheConditioningCacheBackend(self.runtime, pipeline=self.pipeline, restorer=self.restorer,
                                                 **{**defaults, **options})

    def test_raw_restores_are_bitwise_and_follow_the_tiers(self):
        backend = self.backend()
        cache0 = synthetic_cache(1, 0, session="s")
        backend.on_cache_built(cache0)
        restored = backend.restore_raw_tier(cache0.epoch)  # its own segment still holds it: host_raw
        self.assertTrue(same(restored, cache0))
        self.assertEqual(backend.last_restore_tier["tier"], "host_raw")
        self.assertEqual(restored.source, cc.SOURCE_RESTORED)
        self.assertNotEqual(restored.keys_pre_rope[0].data_ptr(), cache0.keys_pre_rope[0].data_ptr())
        backend.cache_released(cache0.epoch)
        restored = backend.restore_raw_tier(cache0.epoch)  # retained device copy, zero-copy
        self.assertEqual(backend.last_restore_tier["tier"], "gpu_raw")
        self.assertEqual(restored.keys_pre_rope[0].data_ptr(), cache0.keys_pre_rope[0].data_ptr())
        self.assertTrue(same(restored, cache0))
        backend.cache_released(cache0.epoch)
        self.assertIn(backend.raw_entries()[0]["state"], ("unpublished", "published"))  # eager: may be published
        backend.flush(wait=True)
        entry = backend.raw_entries()[0]
        self.assertEqual(entry["state"], "published")
        self.assertEqual(entry["node_ref"], backend.node_ref(cache0.epoch))  # token until publication, then node_ref
        stats = backend.raw_tier_stats()
        self.assertEqual(stats["raw_tier"]["retained_after_publication"], 1)
        self.assertEqual(stats["commit_to_publish"]["count"], 1)
        self.assertEqual(set(stats["raw_tier"]["restores"]), {"host_raw", "gpu_raw"})
        backend.raw_restore = "off"
        self.assertIsNone(backend.restore_raw_tier(cache0.epoch))
        backend.raw_restore = "prefer"
        # two more epochs: the third capture evicts epoch 0's host copy (LRU, published); the GPU budget then drops
        # its device copy, so its last raw copy is gone: prepare-on-evict at the next idle point
        caches = [synthetic_cache(2, 1, session="s"), synthetic_cache(3, 2, session="s")]
        for cache in caches:
            backend.on_cache_built(cache)
            backend.cache_released(cache.epoch)
        self.assertIsNone(backend.raw_tier_state(cache0.epoch))
        report = backend.prepare_evicted()
        self.assertEqual((report["epochs"], report["mode"]), (1, "ensure_prepared"))
        self.assertTrue(self.restorer.calls and self.restorer.calls[-1])
        self.assertEqual(backend.raw_tier_stats()["raw_tier"]["prepare_on_evict"]["prepared"], report["packets"])
        backend.arm_restore("svd_hevc")
        lossy = backend.restore(cache0.epoch)
        backend.arm_restore(None)
        self.assertEqual(backend.last_restore_tier["tier"], "compressed_cold")  # software restorer: no prepared cache
        self.assertFalse(same(lossy, cache0))
        backend.flush(wait=True)

    def test_the_planner_never_receives_a_device_copy_and_the_last_resort_is_recorded(self):
        import threading

        backend = self.backend(capture_slots=1)
        submitted = []
        real = self.pipeline.submit

        def recording(*args, **kwargs):
            submitted.append({value.data_ptr() for value in args[2].values()})
            return real(*args, **kwargs)

        held = threading.Event()
        self.pipeline.hooks = lambda stage, node: held.wait(60) if stage == "store" else None
        with mock.patch.object(self.pipeline, "submit", side_effect=recording):
            cache = synthetic_cache(5, 0, session="p")
            backend.on_cache_built(cache)
        device_ptrs = {t.data_ptr() for t in cache.keys_pre_rope + cache.values}
        self.assertEqual(len(submitted), 1)
        self.assertFalse(submitted[0] & device_ptrs)  # the planner gets the host copy, never the retained cache
        self.assertEqual(backend.free_capture_slots(), 0)  # the only slot holds the unpublished epoch
        timer = threading.Timer(0.3, held.set)
        timer.start()
        self.addCleanup(timer.cancel)
        result = backend.make_room()  # last resort: wait for the oldest submitted epoch to publish
        self.assertEqual((result["cause"], result["published"]), ("host_full", 1))
        self.assertGreater(result["seconds"], 0.1)
        stats = backend.raw_tier_stats()["raw_tier"]
        self.assertEqual((stats["last_resort_backpressure_waits"], stats["last_resort_backpressure_causes"]),
                         (1, {"host_full": 1}))
        self.assertEqual(backend.make_room()["cause"], None)  # a published entry is evictable: no wait
        self.pipeline.hooks = None

    def test_without_the_raw_tier_the_backend_is_c3(self):
        backend = self.backend(raw_tier=False, gpu_raw_budget_bytes=0, capture_slots=1)
        cache = synthetic_cache(7, 0, session="c3")
        backend.on_cache_built(cache)
        self.assertIsNone(backend.restore_raw_tier(cache.epoch))
        backend.flush(wait=True)
        self.assertEqual(backend.raw_entries(), [])  # the slot was freed at publication
        self.assertEqual(backend.free_capture_slots(), 1)
        self.assertEqual(backend.raw_tier_stats()["raw_tier"]["gpu_peak_bytes"], 0)


# ---------------------------------------------------------------------------------------------------- provider
def make_tier_provider(store: Path, *, gpu_budget: int = 0, slots: int = 2, **config) -> sp.LongCatSessionProvider:
    """A CPU provider with the raw tier on (as create_provider builds it, with explicit budgets)."""
    from reelcache.archive_pipeline import ArchivePipeline
    from reelcache.archive_restore import SoftwareLineageRestorer
    from reelcache.lineage_commit import ArchiveOptions
    from reelcache.predictor_registry import RankPredictorRegistry

    cfg = sp.LongCatSessionConfig.from_dict({"store_root": str(store), **config})
    runtime = mb.open_runtime(store, mb.model_identity(FINGERPRINT, ARCH), **cfg.integrity())
    pipeline = ArchivePipeline(runtime, method=mb.default_method(), registry=RankPredictorRegistry("tier-provider"),
                               options=ArchiveOptions(encode_threads=1, plan_memory_bytes=cfg.plan_memory_bytes),
                               device="cpu", encode_workers=1)
    decision = sp.resolve_schedule(cfg.schedule, representation=cfg.representation, defer_planning=cfg.defer_planning,
                                   plan_device="cpu", plan_memory_bytes=cfg.plan_memory_bytes, resident_bytes=0,
                                   total_bytes=0, raw_tier=True)
    backend = mb.ReelCacheConditioningCacheBackend(runtime, pipeline=pipeline,
                                                restorer=EnsureRecorder(SoftwareLineageRestorer(runtime)),
                                                archive=(cfg.representation,), heads=HEADS, pin_memory=False,
                                                capture_slots=slots, schedule=decision["chosen"], raw_tier=True,
                                                gpu_raw_budget_bytes=gpu_budget, raw_restore=cfg.raw_restore)
    return sp.LongCatSessionProvider(cfg, backend=backend, owned=(sp._Closer(pipeline, runtime),),
                                     startup={"schedule": decision})


def run_tier_request(provider, handle: dict, *, frames: np.ndarray | None = None, num_segments: int = 1,
                     seed: int = 0, prompt: str = "a cat") -> dict:
    """One session request as the port's stage runs it, including the cache release after denoising; every restored
    cache is compared with its native build (raw tiers must be bitwise)."""
    fields = {"request_id": None, "prompt": prompt, "seed": seed, "num_segments": num_segments, "num_frames": 93,
              "session_id": None, "start_segment_index": 0, "generator_state": None,
              "conditioning_cache_backend": None, "has_client_frames": frames is not None, "injected": []}
    run = provider.begin_request(fields, handle)
    generator = torch.Generator().manual_seed(seed)
    if run.plan.generator_state is not None:
        set_generator_state_hex(generator, run.plan.generator_state)
    cond = frames if run.plan.frames is None else run.plan.frames
    run.bind_request(frames=cond, generator_state=generator_state_hex(generator), geometry=GEOMETRY)
    bitwise, error = [], None
    try:
        for seg in range(run.plan.start_segment, run.plan.start_segment + run.plan.num_segments):
            epoch = cc.ConditioningCacheEpoch(segment_index=seg, token_grid=GRID, cond_latents_sha256=frames_digest(cond),
                                              model_fingerprint=FINGERPRINT, session_id=run.session.session_id,
                                              parent_segment_index=seg - 1 if seg else None)
            cache = provider.cache_backend.restore(epoch)
            native = conditioning_cache(cond, seg, run.session.session_id)
            overlap = {}
            if cache is None:
                cache = native
                provider.cache_backend.on_cache_built(cache)
                source = cc.SOURCE_NATIVE_BUILD
            else:
                source = cc.SOURCE_RESTORED
                overlap = join_restore(cache)  # as the stage: wait per layer, join (d5 default for svd_hevc)
                bitwise.append(same(cache, native))
            del cache
            provider.cache_backend.cache_released(epoch)
            cond = next_frames(cond, prompt, generator)
            record = {"epoch": epoch.as_dict(), "cache_source": source, "generator_state": "start-state", **overlap}
            if run.end_segment(segment_index=seg, record=record, next_frames=cond,
                               generator_state=generator_state_hex(generator), prompt=prompt):
                break
    except BaseException as exc:  # noqa: BLE001
        error = exc
        raise
    finally:
        summary = run.finish(error=error)
    return {"summary": summary, "bitwise": bitwise}


class ProviderTierTests(unittest.TestCase):
    def setUp(self):
        self.root = artifacts("lc-tiers-provider")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def provider(self, **config):
        provider = make_tier_provider(self.root / "store", **config)
        self.addCleanup(self._close, provider)
        return provider

    @staticmethod
    def _close(provider):
        try:
            provider.close()
        except Exception:  # noqa: BLE001
            pass

    def test_resume_fork_and_controls_with_tier_records(self):
        provider = self.provider(gpu_budget=tiny_epoch_bytes(), slots=2, durability="always")
        first = run_tier_request(provider, {"session_id": "a", "action": "generate"}, frames=initial_frames(1),
                                 num_segments=2)
        self.assertEqual(first["summary"]["archival"], "published")
        self.assertEqual(first["summary"]["gpu_raw_released"]["dropped"], 0)
        # resume from boundary 2: the epoch of segment 2 was never built; fork from boundary 1 restores segment 1's
        forked = run_tier_request(provider, {"session_id": "f", "action": "fork", "from_session": "a",
                                             "checkpoint": 1})
        segment = forked["summary"]["segments"][0]
        self.assertEqual((segment["provider_cache"], segment["tier"]), ("restore", "gpu_raw"))
        self.assertEqual(segment["provider_cache_detail"], "gpu_raw")
        self.assertGreater(segment["tier_bytes"], 0)
        self.assertEqual(forked["bitwise"], [True])
        # demote: GPU raw -> host raw; the next fork restores host_raw, bitwise
        demoted = provider.control("demote_raw", {"session_id": "a"})["demote_raw"]
        self.assertGreaterEqual(demoted["entries"], 0)
        provider.control("demote_raw", {"session_id": "f"})
        again = run_tier_request(provider, {"session_id": "g", "action": "fork", "from_session": "a",
                                            "checkpoint": 1})
        self.assertEqual(again["summary"]["segments"][0]["tier"], "host_raw")
        self.assertEqual(again["bitwise"], [True])
        # drop: compressed only (the software restorer has no prepared cache: compressed_cold)
        dropped = provider.control("drop_raw", {"session_id": "a"})["drop_raw"]
        self.assertGreaterEqual(dropped["entries"], 1)
        lossy = run_tier_request(provider, {"session_id": "h", "action": "fork", "from_session": "a",
                                            "checkpoint": 1})
        self.assertEqual(lossy["summary"]["segments"][0]["tier"], "compressed_cold")
        self.assertEqual(lossy["bitwise"], [False])
        stats = provider.control("archive_stats", {})
        pipeline = stats["pipeline"]
        for name in ("gpu_peak_bytes", "host_peak_bytes", "last_resort_backpressure_seconds",
                     "last_resort_backpressure_waits", "last_resort_backpressure_causes"):
            self.assertIn(name, pipeline["raw_tier"])
        for name in ("count", "mean", "p50", "p95", "max", "last", "recent"):
            self.assertIn(name, pipeline["commit_to_publish"])
        self.assertIn("commits", pipeline["encoder"])
        self.assertEqual(set(pipeline["raw_tier"]["restores"]), {"gpu_raw", "host_raw", "compressed_cold"})
        self.assertTrue(all(isinstance(row, dict) for row in stats["entries"]))
        self.assertTrue(provider.control("reset_raw_peaks", {})["reset_raw_peaks"])
        from reelcache.integrations.sglang.controls import SessionControlError

        for op, args in (("drop_raw", {}), ("drop_raw", {"session_id": "a", "drain": "yes"}),
                         ("demote_raw", {"session_id": 3}), ("archive_stats", {"session_id": 1})):
            with self.assertRaises(SessionControlError):
                provider.control(op, args)

    def test_raw_restore_off_and_resident_release(self):
        provider = self.provider(gpu_budget=0, slots=2, durability="always", raw_restore="off")
        run_tier_request(provider, {"session_id": "a", "action": "generate"}, frames=initial_frames(2),
                         num_segments=2)
        forked = run_tier_request(provider, {"session_id": "f", "action": "fork", "from_session": "a",
                                             "checkpoint": 1})
        self.assertEqual(forked["summary"]["segments"][0]["tier"], "compressed_cold")
        stats = provider.control("archive_stats", {})["pipeline"]["raw_tier"]
        self.assertEqual((stats["gpu_bytes"], stats["entries"]["resident"]), (0, 0))  # budget 0: nothing retained


class CreateProviderTierTests(unittest.TestCase):
    def setUp(self):
        self.root = artifacts("lc-tiers-create")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def create(self, headroom_gib: float, **config):
        with mock.patch.object(rt, "cgroup_memory", return_value=memory(headroom_gib)):
            provider = sp.create_provider(config={"store_root": str(self.root / f"store-{uuid.uuid4().hex}"),
                                                  **config})
        self.addCleanup(provider.close)
        return provider

    @unittest.skipIf(torch.cuda.is_available(), "the CPU (software restorer) path of create_provider")
    def test_cpu_attach_budgets_records_and_degradation(self):
        provider = self.create(200)
        tier = provider.startup["raw_tier"]
        canvas = 2 * rt.CANVAS_STAGING_BYTES
        self.assertEqual(tier["fixed_host"]["bytes"], canvas + rt.FINALIZE_WORKSPACE_BYTES)  # no ring / prepared cache
        self.assertEqual(provider.startup["overlap_restore"]["effective"], True)
        self.assertEqual((tier["enabled"], tier["capture_slots"]), (True, rt.HOST_RAW_CAP_BYTES // EPOCH_720P))
        self.assertEqual(tier["gpu"]["budget_bytes"], 0)
        self.assertEqual(provider.backend.raw_tier, True)
        self.assertEqual(provider.startup["schedule"]["chosen"], "eager")
        self.assertEqual(provider.startup["encoder"]["selected"], "x265")
        archival = provider.control("capabilities", {})["archival"]
        self.assertEqual(archival["raw_tier"]["capture_slots"], tier["capture_slots"])
        self.assertEqual(archival["encoder"]["reason"], "auto: no CUDA device")
        small = self.create(10)  # the share holds no epoch: archival off, recorded; sessions still work
        self.assertEqual((small.startup["raw_tier"]["archival"], small.backend.archive), (False, ()))
        c3 = self.create(10, raw_tier=False)
        self.assertEqual((c3.backend.raw_tier, c3.startup["raw_tier"]["capture_slots"]), (False, 4))
        self.assertEqual(c3.startup["schedule"]["chosen"], "deferred")
        with self.assertRaisesRegex(cc.ConditioningCacheError, "needs a CUDA device"):
            self.create(200, encoder="nvenc")


# ---------------------------------------------------------------------------------------------------- CP=2
def _cp_worker(rank, world, root):
    root = Path(root)
    torch.set_num_threads(1)
    output = {"rank": rank}
    try:
        from reelcache.distributed import DistributedOwner
        from reelcache.factor_sharding import FactorShardDomain, OwnerFactorStore
        from reelcache.integrations.longcat_factor_shard import RowShardedCacheArchive
        from reelcache.store import ObjectStore
        from reelcache.types import MatrixKey
        from reelcache_sglang_models.longcat_video.cp_tiers import TieredRowArchive
        from tests.test_longcat_factor_shard import ENERGY, HEAD_DIM, HEADS as CP_HEADS, LAYERS, epoch_for, \
            full_matrix, identity

        dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(seconds=120),
                                init_method=f"file://{root / 'rendezvous'}")
        data_group = dist.new_group(backend="gloo")
        store = OwnerFactorStore(ObjectStore(root / f"owner{rank}"), owner=rank, world_size=world)
        domain = FactorShardDomain(DistributedOwner(expected_world_size=world), data_group=data_group, store=store,
                                   tile_tokens=7, checkpoint_every=1)
        archive = RowShardedCacheArchive(domain, identity=identity(), layers=LAYERS, heads=CP_HEADS,
                                         head_dim=HEAD_DIM, energy=ENERGY, threads=1, pack_groups=2,
                                         representations=("svd_hevc",))
        epochs = {name: epoch_for(rank, world, session=f"tier-{name}") for name in ("a", "b")}
        rows = epochs["a"].row_index()
        source = {MatrixKey(layer, target): full_matrix(layer, target).index_select(0, rows)
                  for layer in range(LAYERS) for target in ("K", "V")}
        keys = [source[MatrixKey(i, "K")] for i in range(LAYERS)]
        values = [source[MatrixKey(i, "V")] for i in range(LAYERS)]
        entry_bytes = sum(t.numel() * t.element_size() for t in keys + values)
        tiered = TieredRowArchive(archive, host_raw_budget_bytes=entry_bytes)  # one published entry per rank
        seam = cc.CacheRowSeam(heads=CP_HEADS, capture=tiered.capture, install=tiered.install)

        def bitwise(cache, reference_keys, reference_values):
            return all(torch.equal(cache.layer_matrices(i)[0], reference_keys[i])
                       and torch.equal(cache.layer_matrices(i)[1], reference_values[i]) for i in range(LAYERS))

        def build(epoch):
            return cc.LongCatConditioningCache.from_matrices(epoch, keys, values, heads=CP_HEADS,
                                                             source=cc.SOURCE_NATIVE_BUILD)

        tiered.arm("svd_hevc")
        seam.on_cache_built(build(epochs["a"]))
        restored = seam.restore(epochs["a"])  # before publication: every rank holds its rows (the flush source)
        output["pre_publish"] = [tiered.last_restore_tier["tier"], tiered.last_restore_tier["per_rank"],
                                 bitwise(restored, keys, values)]
        tiered.flush()
        output["published"] = tiered.stats()["entries"]
        restored = seam.restore(epochs["a"])
        output["post_publish"] = [tiered.last_restore_tier["tier"], bitwise(restored, keys, values)]
        reference = archive.install(epochs["a"])  # the archive's own collective lossy install
        if rank == 1:
            output["dropped"] = tiered.drop_raw()["entries"]
        restored = seam.restore(epochs["a"])  # rank 1 has no raw rows: every rank takes the compressed tier
        output["mixed"] = [tiered.last_restore_tier["tier"], tiered.last_restore_tier["per_rank"],
                           bitwise(restored, *reference)]
        seam.on_cache_built(build(epochs["b"]))
        tiered.flush()  # budget one entry: rank 0 evicts a (least recently used)
        output["after_b"] = tiered.stats()["entries"]
        output["evictions"] = tiered.stats()["evictions"]
        if rank == 0:  # a preflight failure on one rank: every rank rolls back, nothing is installed partially
            original = tiered._stage

            def failing(entry, epoch):
                raise ValueError("injected preflight failure")

            tiered._stage = failing
        restored = seam.restore(epochs["b"])
        reference_b = archive.install(epochs["b"])
        output["rollback"] = [tiered.last_restore_tier["tier"], tiered.last_restore_tier["per_rank"],
                              tiered.last_restore_tier["rollback"] or "", bitwise(restored, *reference_b)]
        if rank == 0:
            tiered._stage = original
        restored = seam.restore(epochs["b"])
        output["after_rollback"] = [tiered.last_restore_tier["tier"], bitwise(restored, keys, values)]
        tiered.raw_restore = "off"
        seam.restore(epochs["b"])
        output["raw_off"] = tiered.last_restore_tier["tier"]
        tiered.raw_restore = "prefer"
        tiered.arm(None)
        output["unarmed_no_raw"] = [seam.restore(epochs["a"]) is None, tiered.last_restore_tier["tier"]]
        stats = tiered.stats()
        output["stats"] = {"rollbacks": stats["rollbacks"], "restores": stats["restores"],
                           "host_peak_bytes": stats["host_peak_bytes"], "entry_bytes": entry_bytes,
                           "gpu_raw_budget_bytes": stats["budgets"]["gpu_raw_budget_bytes"]}
        output["domain_not_broken"] = not domain._broken
        store.store.close()
        dist.destroy_process_group()
    except BaseException as exc:  # noqa: BLE001 - surfaced by the parent test
        import traceback

        output["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        (root / f"rank{rank}.json").write_text(json.dumps(output, default=str))
        os._exit(0)


class ContextParallelTierTests(unittest.TestCase):
    def setUp(self):
        self.root = artifacts("lc-tiers-cp")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_two_ranks_agree_on_one_epoch_wide_tier(self):
        world = 2
        context = multiprocessing.get_context("spawn")
        processes = [context.Process(target=_cp_worker, args=(rank, world, str(self.root))) for rank in range(world)]
        for process in processes:
            process.start()
        try:
            for process in processes:
                process.join(timeout=600)
                self.assertFalse(process.is_alive(), "Gloo worker timed out")
            rows = [json.loads((self.root / f"rank{rank}.json").read_text()) for rank in range(world)]
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
        for row in rows:
            self.assertNotIn("error", row, row.get("error"))
        for row in rows:
            rank = row["rank"]
            self.assertEqual(row["pre_publish"], ["host_raw", ["host_raw", "host_raw"], True], rank)
            self.assertEqual(row["published"]["published"], 1, rank)
            self.assertEqual(row["post_publish"], ["host_raw", True], rank)
            self.assertEqual(row["mixed"], ["compressed_cold", ["host_raw", None], True], rank)
            tier, per_rank, rollback, equal = row["rollback"]
            self.assertEqual((tier, per_rank, equal), ("compressed_cold", ["host_raw", "host_raw"], True), rank)
            self.assertIn("injected preflight failure", rollback, rank)  # agreed on every rank
            self.assertEqual(row["after_rollback"], ["host_raw", True], rank)
            self.assertEqual(row["raw_off"], "compressed_cold", rank)
            self.assertEqual(row["unarmed_no_raw"], [True, "build"], rank)
            self.assertEqual(row["stats"]["rollbacks"], 1, rank)
            self.assertEqual(row["stats"]["gpu_raw_budget_bytes"], 0, rank)
            self.assertTrue(row["domain_not_broken"], rank)
        self.assertEqual(rows[0]["evictions"], 1)  # rank 0 kept a, so b's publication evicted it
        self.assertEqual(rows[1]["dropped"], 1)


if __name__ == "__main__":
    unittest.main()


if os.environ.get("REELCACHE_LONGCAT_G7_GPU_TESTS") == "1":
    class RawTierCUDATests(unittest.TestCase):
        """Inside GPU jobs only (``REELCACHE_LONGCAT_G7_GPU_TESTS=1``, the LongCat GPU-test switch of ``job_sD.sbatch``):
        real pinned slots, the side-stream D2H and its event (I1), zero-copy ``gpu_raw`` and the direct H2D of
        ``host_raw``, on an epoch of the served geometry (2 layers, 4096 channels, 14,432 tokens); NVENC auto."""

        def setUp(self):
            if not torch.cuda.is_available():
                raise AssertionError("REELCACHE_LONGCAT_G7_GPU_TESTS=1 requires a CUDA device")
            self.root = artifacts("lc-tiers-cuda")
            self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

        def test_cuda_raw_tiers_are_bitwise_and_keep_invariant_i1(self):
            from reelcache.archive_pipeline import ArchivePipeline
            from reelcache.archive_restore import SoftwareLineageRestorer
            from reelcache.lineage_commit import ArchiveOptions
            from reelcache.manager import KVRuntime
            from reelcache.predictor_registry import RankPredictorRegistry
            from reelcache_sglang_models.longcat_video.spec import LongCatVideoArch

            arch = LongCatVideoArch(hidden_size=4096, depth=2, num_heads=32)
            model = mb.model_identity("freezeD-cuda-tiers", arch)
            runtime = KVRuntime(self.root / "store", model)
            self.addCleanup(runtime.close)
            pipeline = ArchivePipeline(runtime, method=mb.default_method(), registry=RankPredictorRegistry("cuda-tier"),
                                       options=ArchiveOptions(encode_threads=1), device="cpu", encode_workers=1)
            self.addCleanup(pipeline.close, drain=False)
            epoch_bytes = mb.epoch_nbytes(model)
            backend = mb.ReelCacheConditioningCacheBackend(
                runtime, pipeline=pipeline, restorer=SoftwareLineageRestorer(runtime), archive=("svd_hevc",),
                heads=32, pin_memory=True, capture_slots=2, schedule="deferred", raw_tier=True,
                gpu_raw_budget_bytes=epoch_bytes, restore_device="cuda")
            tokens = 4 * 44 * 82
            generator = torch.Generator(device="cuda").manual_seed(5)

            def cache(segment):
                keys = [torch.randn(1, tokens, 32, 128, generator=generator, device="cuda").to(torch.bfloat16)
                        for _ in range(2)]
                values = [torch.randn(1, tokens, 32, 128, generator=generator, device="cuda").to(torch.bfloat16)
                          for _ in range(2)]
                epoch = cc.ConditioningCacheEpoch(segment_index=segment, token_grid=(4, 44, 82),
                                                  cond_latents_sha256=f"{segment + 1:064x}",
                                                  model_fingerprint=model.fingerprint, session_id="cuda",
                                                  parent_segment_index=segment - 1 if segment else None)
                return cc.LongCatConditioningCache(epoch=epoch, keys_pre_rope=keys, values=values,
                                                   source=cc.SOURCE_NATIVE_BUILD, metadata={"build_seconds": 0.0})

            first = cache(0)
            reference = [t.clone() for t in first.keys_pre_rope + first.values]
            backend.on_cache_built(first)  # async D2H into a pinned slot on the copy stream (event recorded)
            entry = backend._tiers.by_key(mb.epoch_key(first.epoch))
            self.assertTrue(entry.host[next(iter(entry.host))].is_pinned())
            backend.cache_released(first.epoch)
            # I1: dropping the device copy of an unpublished epoch waits for its D2H first
            dropped = backend.enforce_gpu_raw_budget(0)
            self.assertEqual(dropped["dropped"], 1)
            self.assertIsNone(entry.gpu)
            restored = backend.restore_raw_tier(first.epoch)
            self.assertEqual((backend.last_restore_tier["tier"], restored.keys_pre_rope[0].device.type),
                             ("host_raw", "cuda"))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(restored.keys_pre_rope + restored.values,
                                                                   reference)))
            seconds = backend.last_restore_tier["seconds"]
            del restored
            second = cache(1)
            backend.on_cache_built(second)
            backend.cache_released(second.epoch)
            again = backend.restore_raw_tier(second.epoch)  # retained within the one-epoch budget: zero-copy
            self.assertEqual(backend.last_restore_tier["tier"], "gpu_raw")
            self.assertEqual(again.keys_pre_rope[0].data_ptr(), second.keys_pre_rope[0].data_ptr())
            backend.cache_released(second.epoch)
            stats = backend.raw_tier_stats()["raw_tier"]
            self.assertLessEqual(stats["gpu_peak_bytes"], epoch_bytes)
            print(f"\nhost_raw H2D of {epoch_bytes / GIB:.2f} GiB: {seconds:.3f} s "
                  f"({epoch_bytes / seconds / 1e9:.1f} GB/s); gpu_peak {stats['gpu_peak_bytes'] / GIB:.2f} GiB")
            backend.discard_pending("cuda")

        def test_cuda_encoder_auto_selects_by_gpu(self):
            name = torch.cuda.get_device_name(0)
            encoder, record = mb.select_encoder("auto", torch.cuda.current_device())
            try:
                if "A100" in name or "H100" in name or "H200" in name:
                    self.assertEqual(record["selected"], "x265", record)
                    self.assertIn("NVENC unavailable", record["reason"])
                    with self.assertRaises(cc.ConditioningCacheError):
                        mb.select_encoder("nvenc", torch.cuda.current_device())
                else:
                    self.assertEqual(record["selected"], "nvenc", record)
                    self.assertTrue(record["driver"])
                print(f"\n{name}: {json.dumps({k: record.get(k) for k in ('selected', 'reason', 'device', 'driver', 'probe_device_bytes', 'probe_host_bytes')}, default=str)}")
            finally:
                if encoder is not None:
                    encoder.close()
