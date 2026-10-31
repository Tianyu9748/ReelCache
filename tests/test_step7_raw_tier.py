"""Freeze D raw-chunk tier (Step 7 dev22a; ``DESIGN_D_tiers.md`` §11): tier invariants and budgets, never-block
admission with last-resort backpressure, placement-independent content, holders, failure handling, controls, the
C3 path with ``raw_tier=False``, restores by tier through the bridge, and the encoder choice (CPU). D2 (dev22g,
§12): the pooled host blocks (no unpinning on the hot path), flush wake-ups, resolved budgets in the stats and the
decode section's evacuation without placement."""

from __future__ import annotations

import contextlib
import os
import random
import shutil
import threading
import time
import unittest
import uuid
import weakref
from types import SimpleNamespace
from unittest import mock

import torch

from reelcache import raw_tier as rt
from reelcache.archive_pipeline import ArchiveFailure, ArchivePipeline
from reelcache.errors import CapacityError, ConfigurationError, StateError, UnsupportedError
from reelcache.integrations.sglang import lineage_archive as la
from reelcache.lineage_commit import ArchiveOptions
from reelcache.raw_tier import RawChunkTier
from reelcache.types import MatrixKey
from tests import step6_support as s6
from tests import test_sglang_bridge as tb

KEY = MatrixKey(0, "K")


def _matrix(seed: int, rows: int = 64, channels: int = 128, dtype=torch.bfloat16) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, channels, generator=generator).to(dtype)


def _adopt(tier: RawChunkTier, name: str, tensor: torch.Tensor, *, token_start: int = 0, release=None):
    with tier._cond:
        reservation = tier.reserve(tensor.numel() * tensor.element_size())
        return tier.adopt(reservation, branch="s", node_id=name, groups=[(name, token_start, {KEY: tensor})],
                          release=release)


def _check_invariants(test: unittest.TestCase, tier: RawChunkTier) -> None:
    with tier._cond:
        entries = [entry for entry in tier._entries.values() if not entry.dropped]
        device = sum(entry.nbytes for entry in entries if entry.on_device)
        host = sum(entry.host_nbytes for entry in entries if entry.host_state == "complete")
        unpublished = sum(entry.nbytes for entry in entries if entry.unpublished)
        test.assertEqual(tier._device_bytes, device)
        test.assertEqual(tier._host_bytes, host)
        test.assertEqual(tier._unpublished_bytes, unpublished)
        for entry in entries:
            if entry.unpublished:  # I1: an unpublished entry always has a complete copy
                test.assertTrue(entry.on_device or entry.host_state == "complete", entry.describe())


class TierUnitTests(unittest.TestCase):
    """The tier alone, with CPU tensors standing in for device copies."""

    def tier(self, **kwargs) -> RawChunkTier:
        kwargs.setdefault("host_budget_bytes", 1 << 30)
        tier = RawChunkTier(**kwargs)
        self.addCleanup(tier.close)
        return tier

    def test_spilled_entries_keep_one_complete_copy_and_restage_bitwise(self):
        sources = [_matrix(index) for index in range(3)]
        nbytes = sources[0].numel() * sources[0].element_size()
        tier = self.tier(gpu_budget_bytes=nbytes)
        released = []
        entries = [_adopt(tier, f"c{index}", source, release=released.append) for index, source in enumerate(sources)]
        tier.flush(30)
        _check_invariants(self, tier)
        self.assertLessEqual(tier._device_bytes, nbytes)
        # Queued (unfitted) entries spill newest first: the planner reads the oldest next.
        self.assertTrue(entries[0].on_device)
        self.assertEqual([entry.host_state for entry in entries[1:]], ["complete", "complete"])
        self.assertEqual(len(released), 2)  # producers' tensors given up only after a complete host copy
        for entry, source in zip(entries, sources):
            staged = tier.stage_for_plan(entry)
            self.assertTrue(torch.equal(staged[0][KEY], source))
            self.assertEqual((staged[0][KEY].dtype, staged[0][KEY].stride()), (source.dtype, source.stride()))
            with tier._cond:
                tier.fitted(entry)
        _check_invariants(self, tier)
        stats = tier.stats()
        self.assertGreaterEqual(stats["spills"], 2)  # a fitted entry may also be demoted once uploads refill the GPU
        self.assertEqual(stats["uploads"], 2)
        self.assertGreater(stats["host_peak_bytes"], 0)

    def test_published_entries_evict_by_access_and_prepare_on_drop(self):
        dropped = []
        sources = [_matrix(10 + index) for index in range(3)]
        host_bytes = rt._align(sources[0].numel() * sources[0].element_size())
        tier = self.tier(gpu_budget_bytes=0, host_budget_bytes=2 * host_bytes, on_host_drop=dropped.append)
        entries = []
        for index, source in enumerate(sources[:2]):
            entry = _adopt(tier, f"c{index}", source)
            tier.flush(30)
            with tier._cond:
                tier.fitted(entry)
                tier.published(entry, f"ref-{index}")
            entries.append(entry)
        lease = tier.acquire("ref-0")  # a restore hit: ref-0 becomes the most recently used
        self.assertEqual(lease.tier, "host_raw")
        self.assertTrue(torch.equal(lease.matrix(0, KEY), sources[0]))
        lease.release()
        third = _adopt(tier, "c2", sources[2])
        tier.flush(30)
        _check_invariants(self, tier)
        self.assertEqual(third.host_state, "complete")
        self.assertIsNone(tier.acquire("ref-1"))  # least recently used published copy went first
        self.assertIsNotNone(tier.acquire("ref-0"))
        self.assertEqual(dropped, ["ref-1"])  # its compressed node is prepared in host memory (R6)
        self.assertEqual(tier.stats()["prepare_on_drop"]["done"], 1)

    def test_failed_and_cancelled_entries_drop_at_once(self):
        tier = self.tier()
        failed, cancelled = _adopt(tier, "f", _matrix(1)), _adopt(tier, "c", _matrix(2))
        with tier._cond:
            tier.retire(failed, "failed")
            tier.retire(cancelled, "cancelled")
        tier.flush(30)
        _check_invariants(self, tier)
        stats = tier.stats()
        self.assertEqual((stats["drops_failed"], stats["drops_cancelled"]), (1, 1))
        self.assertEqual((stats["gpu_bytes"], stats["host_bytes"], stats["unpublished_bytes"]), (0, 0, 0))

    def test_holders_attach_and_detach_atomically_without_ratchet(self):
        tier = self.tier(gpu_budget_bytes=0)
        entry = _adopt(tier, "c0", _matrix(3))
        pending = _adopt(tier, "c1", _matrix(4))
        tier.flush(30)
        with tier._cond:
            tier.fitted(entry)
            tier.published(entry, "ref-a")
        tier.hold("fork", ["ref-a"])
        self.assertEqual(entry.holders, {"s", "fork"})
        tier.release_holder("s")
        self.assertFalse(entry.dropped)  # the fork still holds it
        self.assertFalse(pending.dropped)  # unpublished entries never drop on release (I1)
        tier.hold("fork", [])  # a rewind past it
        tier.flush(30)
        self.assertTrue(entry.dropped)
        with tier._cond:
            tier.retire(pending, "cancelled")
        tier.flush(30)
        tier.release_holder("fork")
        stats = tier.stats()
        self.assertEqual((stats["gpu_bytes"], stats["host_bytes"], stats["entries"]["total"]), (0, 0, 0))
        self.assertEqual(stats["host_buffers"]["live_bytes"], 0)

    def test_admission_last_resort_lag_bound_reservations_and_degradation(self):
        source = _matrix(5)
        nbytes = source.numel() * source.element_size()
        tier = self.tier(gpu_budget_bytes=nbytes, host_budget_bytes=rt._align(nbytes))
        with tier._cond:
            self.assertTrue(tier.usable(nbytes))
            self.assertIsNone(tier.admission(nbytes))
        first, second = _adopt(tier, "a", source), _adopt(tier, "b", _matrix(6))
        tier.flush(30)
        with tier._cond:
            self.assertEqual(tier.admission(nbytes), "host_full")
            tier.fitted(first)
            tier.published(first, "ref-a")
            self.assertIsNone(tier.admission(nbytes))
            reservation = tier.reserve(nbytes)
            self.assertEqual(tier.admission(nbytes), "host_full")  # reserved bytes count
            tier.unreserve(reservation)
            self.assertEqual(tier._reserved, 0)
            tier.record_wait("host_full", 0.25, branch="s", node_id="c")
        stats = tier.stats()
        self.assertEqual(stats["last_resort_causes"]["host_full"], 1)
        self.assertEqual(stats["last_resort_events"][-1]["node_id"], "c")
        lagged = self.tier(max_lag_seconds=0.5)
        _adopt(lagged, "x", _matrix(7))
        with lagged._cond:
            lagged._publish_times.extend([0.0, 1.0, 2.0])
            self.assertEqual(lagged.admission(nbytes), "lag_bound")
        small = self.tier(host_budget_bytes=nbytes // 2)
        with small._cond:
            self.assertFalse(small.usable(nbytes))
        self.assertIn("below one chunk", small.stats()["disabled_reason"])
        self.assertEqual(second.state, "submitted")

    def test_gpu_spill_admission_holds_device_copies_to_the_budget_plus_the_new_chunk(self):
        """While over-budget device copies are being demoted, a new entry waits for those copies (cause gpu_spill):
        with it, the device holds at most the budget plus the new chunk (C3's held sources plus the newest capture)."""
        gate = threading.Event()
        original = RawChunkTier._spill

        def gated(tier_self, entry, sources):
            gate.wait(30)
            return original(tier_self, entry, sources)

        source = _matrix(20)
        nbytes = source.numel() * source.element_size()
        with mock.patch.object(RawChunkTier, "_spill", gated):
            tier = self.tier(gpu_budget_bytes=nbytes)
            _adopt(tier, "a", source)
            with tier._cond:
                self.assertIsNone(tier.admission(nbytes))  # within the budget: no wait
            _adopt(tier, "b", _matrix(21))  # one entry over the budget: its demotion is in flight
            with tier._cond:
                self.assertEqual(tier.admission(nbytes), "gpu_spill")
            gate.set()
            tier.flush(30)
            with tier._cond:
                self.assertIsNone(tier.admission(nbytes))
                self.assertLessEqual(tier._device_bytes, nbytes)
            self.assertEqual(tier.stats()["gpu_peak_bytes"], 2 * nbytes)

    def test_controls_demote_promote_and_drop(self):
        source = _matrix(8)
        nbytes = source.numel() * source.element_size()
        tier = self.tier(gpu_budget_bytes=4 * nbytes)
        entry = _adopt(tier, "c0", source)
        with tier._cond:
            tier.fitted(entry)
        with self.assertRaises(StateError):
            tier.drop([entry])  # unpublished: never dropped (I1)
        with tier._cond:
            tier.published(entry, "ref-0")
        report = tier.demote([entry], timeout=30)
        self.assertEqual(report["entries"], 1)
        self.assertEqual((entry.on_device, entry.host_state), (False, "complete"))
        tier.promote([entry])
        self.assertTrue(entry.on_device)
        self.assertEqual(tier.acquire("ref-0").tier, "gpu_raw")
        tiny = self.tier(gpu_budget_bytes=0)
        other = _adopt(tiny, "c", _matrix(9))
        tiny.flush(30)
        with tiny._cond:
            tiny.fitted(other)
            tiny.published(other, "ref-x")
        with self.assertRaises(CapacityError):
            tiny.promote([other])
        tier.drop([entry])
        tier.flush(30)
        self.assertIsNone(tier.acquire("ref-0"))

    def test_cgroup_headroom_and_auto_budget(self):
        root = s6.ROOT / f"cgroup-{uuid.uuid4().hex}"
        self.addCleanup(shutil.rmtree, root, True)
        job = root / "fs" / "job_1"
        leaf = job / "step" / "task"
        leaf.mkdir(parents=True)
        (leaf / "memory.max").write_text("max\n")
        (leaf / "memory.current").write_text("1000\n")
        (job / "memory.max").write_text(str(128 << 30))
        (job / "memory.current").write_text(str(40 << 30))
        # 10 GiB of page cache, 2 GiB of it shared memory or dirty: 8 GiB is reclaimable headroom.
        (job / "memory.stat").write_text(f"anon {30 << 30}\nfile {10 << 30}\nshmem {1 << 30}\n"
                                         f"file_dirty {1 << 30}\nfile_writeback 0\n")
        proc = root / "cgroup"
        proc.write_text("0::/job_1/step/task\n")
        memory = rt.cgroup_memory(str(proc), str(root / "fs"))
        self.assertEqual((memory["limit"], memory["headroom"], memory["version"]), (128 << 30, 96 << 30, 2))
        self.assertEqual(memory["reclaimable_cache"], 8 << 30)
        budget, derivation = rt.auto_host_budget(reserved_bytes=4 << 30, workers=2, memory=memory)
        self.assertEqual(budget, int(0.5 * (92 << 30) / 2))
        self.assertEqual((derivation["source"], derivation["workers"]), ("cgroup", 2))
        tight, _ = rt.auto_host_budget(reserved_bytes=0, workers=1, memory={**memory, "headroom": 1 << 20})
        self.assertEqual(tight, 1 << 19)  # no minimum above the safe headroom
        capped, _ = rt.auto_host_budget(memory={**memory, "headroom": 1 << 40})
        self.assertEqual(capped, rt.AUTO_HOST_CAP_BYTES)
        v1 = root / "v1"
        (v1 / "memory" / "slurm").mkdir(parents=True)
        (v1 / "memory" / "slurm" / "memory.limit_in_bytes").write_text(str(8 << 30))
        (v1 / "memory" / "slurm" / "memory.usage_in_bytes").write_text(str(2 << 30))
        (root / "cgroup1").write_text("5:memory:/slurm\n")
        self.assertEqual(rt.cgroup_memory(str(root / "cgroup1"), str(v1))["headroom"], 6 << 30)
        with mock.patch.dict("os.environ", {"LOCAL_WORLD_SIZE": "2"}):
            self.assertEqual(rt.local_workers(), 2)

    def test_randomized_operations_keep_the_invariants(self):
        generator = random.Random(1234)
        nbytes = _matrix(0).numel() * 2
        tier = self.tier(gpu_budget_bytes=2 * nbytes, host_budget_bytes=4 * rt._align(nbytes))
        live: list = []
        published: list = []
        for step in range(160):
            op = generator.choice(["submit", "submit", "fit", "publish", "fail", "restore", "demote", "release"])
            with tier._cond:
                room = tier.admission(nbytes) is None
            if op == "submit" and room:
                live.append(_adopt(tier, f"n{step}", _matrix(step)))
            elif op == "fit" and live:
                entry = generator.choice(live)
                if entry.state == "submitted":
                    staged = tier.stage_for_plan(entry)
                    self.assertTrue(torch.equal(staged[0][KEY], _matrix(int(entry.label[1][1:]))))
                    with tier._cond:
                        tier.fitted(entry)
            elif op == "publish" and live:
                entry = live[0]
                if entry.state == "fitted":
                    with tier._cond:
                        tier.published(entry, f"ref-{entry.label[1]}")
                    published.append(live.pop(0))
            elif op == "fail" and live:
                entry = live.pop()
                with tier._cond:
                    tier.retire(entry, "failed")
            elif op == "restore" and published:
                lease = tier.acquire(f"ref-{generator.choice(published).label[1]}")
                if lease is not None:
                    lease.release()
            elif op == "demote":
                tier.demote(tier.entries(), wait=False)
            elif op == "release" and published:
                tier.hold("s", [f"ref-{entry.label[1]}" for entry in published[-2:]])
            if step % 8 == 0:
                tier.flush(30)
                _check_invariants(self, tier)
        tier.flush(30)
        _check_invariants(self, tier)
        with tier._cond:
            self.assertLessEqual(tier._host_bytes, tier.host_budget())


class _GateEvent:
    """A CUDA-event stand-in whose ``synchronize`` waits for a gate (a pending upload from a dropped host copy)."""

    def __init__(self, gate: threading.Event | None = None):
        self.gate = gate

    def synchronize(self) -> None:
        if self.gate is not None and not self.gate.wait(30):
            raise AssertionError("gate never opened")

    def query(self) -> bool:
        return self.gate is None or self.gate.is_set()


class HostPoolTests(unittest.TestCase):
    """Freeze D2 (coordinator 2026-10-17 03:50): dropped host copies return their blocks to a size-class pool and
    are never unpinned or freed on the hot path; the pool shrinks only at close, at a session release that leaves the
    tier idle above the keep bytes, or on an explicit idle shrink (recorded). flush() wakes when a release job ends;
    the stats carry the resolved budgets."""

    def tier(self, **kwargs) -> RawChunkTier:
        kwargs.setdefault("host_budget_bytes", 1 << 30)
        kwargs.setdefault("gpu_budget_bytes", 0)
        tier = RawChunkTier(**kwargs)
        self.addCleanup(tier.close)
        return tier

    def published(self, tier: RawChunkTier, count: int, *, seed: int = 0, prefix: str = "c") -> list:
        entries = []
        for index in range(count):
            entries.append(_adopt(tier, f"{prefix}{index}", _matrix(seed + index)))
        tier.flush(30)
        with tier._cond:
            for index, entry in enumerate(entries):
                tier.fitted(entry)
                tier.published(entry, f"ref-{prefix}{index}")
        return entries

    def test_size_classes_bound_the_rounding_and_keep_the_small_alignment(self):
        for nbytes in (1, 511, 16384, 1 << 20):
            self.assertEqual(rt._size_class(nbytes), rt._align(nbytes))
        # CausVid / LongLive 1.0, RAVEN K, LongLive 2.0 matrices (DESIGN_D_tiers.md; Step 19 CHUNK_RAW_BYTES / 60)
        for nbytes in ((1 << 20) + 1, 14_376_960, 28_753_920, 43_253_760, 118_226_944, (1 << 31) - 1):
            size = rt._size_class(nbytes)
            self.assertGreaterEqual(size, nbytes)
            self.assertLessEqual(size - nbytes, nbytes / 32)
            self.assertEqual(rt._size_class(size), size)  # a class is its own class
        self.assertEqual(rt._size_class(14_376_960), 14_417_920)  # +0.28 %

    def test_dropped_copies_return_blocks_to_the_pool_without_unpinning(self):
        closes = []
        original = rt._HostBlock.close

        def close(block):
            closes.append(block.nbytes)
            original(block)

        with mock.patch.object(rt._HostBlock, "close", close):
            tier = self.tier(gpu_budget_bytes=0)
            entries = self.published(tier, 3)
            per_entry = entries[0].host_nbytes
            self.assertEqual(per_entry, rt._size_class(_matrix(0).numel() * 2))
            pool = tier.stats()["host_buffers"]
            self.assertEqual((pool["allocations"], pool["live_bytes"], pool["cached_bytes"]), (3, 3 * per_entry, 0))
            tier.drop(entries[:2])  # explicit drop, then an eviction, a retirement and a release
            with tier._cond:
                tier.retire(_adopt(tier, "failed", _matrix(9)), "failed")
            tier.flush(30)
            tier.release_holder("s")  # the last entry; idle afterwards, but the pool is under its keep bytes
            tier.flush(30)
            pool = tier.stats()["host_buffers"]
            self.assertEqual(closes, [])  # nothing unpinned or freed on these paths
            self.assertEqual((pool["freed"], pool["shrinks"], pool["live_bytes"]), (0, 0, 0))
            self.assertEqual(pool["cached_bytes"], pool["capacity_bytes"])
            allocations = pool["allocations"]
            again = self.published(tier, 3, seed=20, prefix="d")  # the same class: every block reused
            pool = tier.stats()["host_buffers"]
            self.assertEqual(pool["allocations"], allocations)
            self.assertGreaterEqual(pool["reuses"], 3)
            for index, entry in enumerate(again):
                lease = tier.acquire(f"ref-d{index}")
                self.assertTrue(torch.equal(lease.matrix(0, KEY), _matrix(20 + index)))  # reused blocks hold the new copy
                lease.release()
            tier.drop(again)
            tier.flush(30)
            self.assertEqual(closes, [])
            cached = tier.stats()["host_buffers"]["cached_bytes"]
            event = tier.shrink_buffers(per_entry)  # explicit idle shrink, keeping one block
            self.assertEqual((event["reason"], event["bytes"]), ("explicit", cached - per_entry))
            self.assertEqual(sum(closes), cached - per_entry)
            pool = tier.stats()["host_buffers"]
            self.assertEqual((pool["shrinks"], pool["cached_bytes"]), (1, per_entry))
            self.assertEqual(pool["shrink_events"][-1]["reason"], "explicit")
            tier.close()
            self.assertEqual(sum(closes), cached)
            self.assertEqual(tier.stats()["host_buffers"]["shrink_events"][-1]["reason"], "close")

    def test_an_idle_session_release_shrinks_only_above_the_keep_bytes(self):
        tier = self.tier(keep_free_bytes=0)
        entries = self.published(tier, 2)
        pending = _adopt(tier, "pending", _matrix(7))  # another commit in flight: not idle
        tier.flush(30)
        report = tier.release_holder("s")
        tier.flush(30)
        self.assertEqual(report["freed_host_bytes"], 0)
        self.assertTrue(all(entry.dropped for entry in entries))
        self.assertEqual(tier.stats()["host_buffers"]["shrinks"], 0)
        with tier._cond:
            tier.retire(pending, "cancelled")
        tier.flush(30)
        cached = tier.stats()["host_buffers"]["cached_bytes"]
        self.assertGreater(cached, 0)
        report = tier.release_holder("s")  # idle now: the pool shrinks to keep_free_bytes (0)
        self.assertEqual(report["freed_host_bytes"], cached)
        events = tier.stats()["host_buffers"]["shrink_events"]
        self.assertEqual([(item["reason"], item["bytes"]) for item in events], [("session_release", cached)])
        with self.assertRaises(ConfigurationError):
            tier.shrink_buffers(-1)

    def test_the_pool_stays_within_the_host_budget_when_shapes_change(self):
        """R5: cached idle blocks count against the host budget (the pool's limit). Copies of another shape free
        cached blocks of the first shape before pinning new ones (reason ``budget``); one shape only reuses."""
        a, b = rt._size_class(_matrix(0).numel() * 2), rt._size_class(_matrix(0, rows=96).numel() * 2)
        self.assertNotEqual(a, b)
        budget = 3 * a
        tier = self.tier(host_budget_bytes=budget)
        tier.drop(self.published(tier, 3))
        tier.flush(30)
        pool = tier.stats()["host_buffers"]
        self.assertEqual((pool["cached_bytes"], pool["capacity_bytes"], pool["limit_bytes"]), (budget, budget, budget))
        entry = _adopt(tier, "other-shape", _matrix(5, rows=96))
        tier.flush(30)
        self.assertEqual(entry.host_state, "complete")
        stats = tier.stats()
        pool = stats["host_buffers"]
        self.assertLessEqual(pool["capacity_bytes"], budget)
        self.assertEqual(stats["host_pool_bytes"], pool["capacity_bytes"])
        self.assertLessEqual(stats["host_pool_peak_bytes"], budget)
        self.assertEqual([(item["reason"], item["bytes"]) for item in pool["shrink_events"]], [("budget", 2 * a)])
        self.assertEqual(pool["over_limit"], 0)
        reuses = pool["reuses"]
        again = _adopt(tier, "first-shape", _matrix(6))  # the remaining cached block of the first shape
        tier.flush(30)
        pool = tier.stats()["host_buffers"]
        self.assertEqual(again.host_state, "complete")
        self.assertEqual((pool["reuses"], pool["shrinks"]), (reuses + 1, 1))
        self.assertLessEqual(pool["capacity_bytes"], budget)

    def test_flush_wakes_when_a_release_job_ends(self):
        """Two release jobs (host copies dropped while uploads from them were pending): the first waits for its
        upload, flush(20) waits meanwhile and must return as soon as both end (before D2 it slept 20 s)."""
        tier = self.tier()
        entries = self.published(tier, 2)
        gate = threading.Event()
        with tier._cond:
            entries[0].host_read_event, entries[1].host_read_event = _GateEvent(gate), _GateEvent()
        tier.drop(entries)
        flushed = {}

        def flush():
            started = time.perf_counter()
            tier.flush(20)
            flushed["seconds"] = time.perf_counter() - started
            flushed["pool"] = tier.stats()["host_buffers"]

        thread = threading.Thread(target=flush)
        thread.start()
        time.sleep(0.2)
        self.assertNotIn("seconds", flushed)  # the first release still waits for its upload
        gate.set()
        thread.join(30)
        self.assertLess(flushed["seconds"], 5.0)
        self.assertEqual(flushed["pool"]["live_bytes"], 0)  # both releases ended before flush returned
        self.assertEqual(flushed["pool"]["returned"], 2)
        self.assertEqual((tier._active, len(tier._jobs)), (0, 0))

    def test_stats_record_the_resolved_budgets(self):
        tier = self.tier(gpu_budget_bytes=None, host_budget_bytes=123 << 20, gpu_budget_chunks=3)
        stats = tier.stats()
        self.assertEqual((stats["gpu_raw_budget_bytes"], stats["host_raw_budget_bytes"]), (0, 123 << 20))
        self.assertFalse(stats["budgets"]["gpu_resolved"])
        entry = _adopt(tier, "c0", _matrix(1))
        stats = tier.stats()
        self.assertEqual(stats["gpu_raw_budget_bytes"], 3 * entry.nbytes)
        self.assertEqual(stats["budgets"]["gpu_raw_budget_bytes"], 3 * entry.nbytes)
        self.assertEqual(stats["budgets"]["host_raw_budget_bytes"], 123 << 20)
        self.assertTrue(stats["budgets"]["gpu_resolved"])
        auto = RawChunkTier(gpu_budget_bytes=7 << 20, host_budget_bytes=None)
        self.addCleanup(auto.close)
        stats = auto.stats()
        self.assertIsInstance(stats["host_raw_budget_bytes"], int)
        self.assertEqual(stats["host_raw_budget_bytes"], stats["budgets"]["host_source"]["budget_bytes"])
        self.assertEqual(stats["gpu_raw_budget_bytes"], 7 << 20)

    def test_a_memory_critical_evacuation_keeps_host_copies_evictable(self):
        """``demote(place=False)`` (the decode section's evacuation): later host copies evict the oldest by access.
        An explicit placement exempts them, so the third demotion is blocked by a full host tier."""
        nbytes = rt._size_class(_matrix(0).numel() * 2)
        for place, blocked in ((False, 0), (True, 1)):
            tier = self.tier(gpu_budget_bytes=1 << 30, host_budget_bytes=2 * nbytes)
            entries = []
            for index in range(3):
                entry = _adopt(tier, f"c{index}", _matrix(30 + index))
                with tier._cond:
                    tier.fitted(entry)
                    tier.published(entry, f"ref-{index}")
                entries.append(entry)
                report = tier.demote([entry], timeout=30, place=place)
                self.assertEqual(report["failed"], blocked if index == 2 else 0, (place, index))
            tier.flush(30)
            _check_invariants(self, tier)
            evicted = tier.stats()["evictions_host"]
            self.assertEqual(evicted, 0 if place else 1, place)
            self.assertEqual([entry.placed for entry in entries[:2]], ["host_raw" if place else None] * 2)

    def test_the_archive_evacuates_gpu_raw_without_placing_it(self):
        calls = []
        tier = SimpleNamespace(enabled=True, entries=lambda: ["e"],
                               demote=lambda entries, **kwargs: calls.append((entries, kwargs)) or {"entries": 1},
                               pressure_check=lambda *args, **kwargs: None, section_begin=lambda: None,
                               section_end=lambda token, **kwargs: None, pressure_enabled=False)
        archive = _fake_archive(tier, defer=True)
        with la.LineageSessionArchive.memory_critical(archive, "decode") as report:
            pass
        self.assertEqual(calls, [(["e"], {"wait": True, "timeout": 30.0, "place": False})])
        self.assertEqual(report["raw_demoted"], {"entries": 1})



class _FakeDevice:
    """A device-memory provider for CPU tests (the tier's ``device_memory``): adjustable free, total and allocator
    bytes; counts its queries."""

    def __init__(self, *, total: int = 1 << 30, free: int = 1 << 30, reserved: int = 0, allocated: int = 0,
                 inactive_split: int = 0, peak: int = 0):
        self.memory = {"total": total, "free": free, "reserved": reserved, "allocated": allocated,
                       "inactive_split": inactive_split, "peak": peak}
        self.calls = 0

    def __call__(self, device):
        self.calls += 1
        return dict(self.memory)


class _FakeSegments:
    """An allocator-segment provider for CPU tests (the tier's ``device_segments``, dev22j8v4); counts its calls."""

    def __init__(self, segments):
        self.segments, self.calls = list(segments), 0

    def __call__(self, device):
        self.calls += 1
        return [dict(segment) for segment in self.segments]


def _segment(*, expandable: bool, blocks, kind: str = "large", address: int = 0, pool=(0, 0)) -> dict:
    """A ``torch.cuda.memory_snapshot`` segment: consecutive blocks ``(state, size)`` from ``address``."""
    out, at = [], address
    for state, size in blocks:
        out.append({"address": at, "size": size, "state": state})
        at += size
    return {"device": 0, "address": address, "total_size": at - address, "segment_type": kind,
            "allocated_size": sum(size for state, size in blocks if state == "active_allocated"),
            "is_expandable": expandable, "segment_pool_id": pool, "blocks": out}


class StrictHeadroomTests(unittest.TestCase):
    """dev22j8v4 (F5.2's RAVEN decode OOM; probes 4445094 / 4445098): what the caching allocator can actually give
    back -- wholly free segments and the whole pages of free blocks in expandable segments (20 MiB large, 2 MiB small),
    default pool only. Fragments, private pools and pending frees are not headroom."""

    def test_only_whole_free_segments_and_whole_expandable_pages_are_releasable(self):
        mib = 1 << 20
        segments = [
            _segment(expandable=False, blocks=[("inactive", 20 * mib)]),  # wholly free: released
            _segment(expandable=False, blocks=[("active_allocated", 2 * mib), ("inactive", 18 * mib)]),  # fragment
            # 20 MiB pages: a free 50 MiB block at offset 10 MiB holds the pages [20, 40) and [40, 60)
            _segment(expandable=True, blocks=[("active_allocated", 10 * mib), ("inactive", 50 * mib),
                                              ("active_allocated", 4 * mib)]),
            # 2 MiB pages (small pool): a free 3 MiB block at offset 1 MiB holds the page [2, 4)
            _segment(expandable=True, kind="small", address=1 << 40,
                     blocks=[("active_allocated", 1 * mib), ("inactive", 3 * mib)]),
            _segment(expandable=False, pool=(0, 7), blocks=[("inactive", 64 * mib)]),  # a private pool
            _segment(expandable=False, blocks=[("active_pending_free", 30 * mib)]),  # awaiting its stream's event
        ]
        self.assertEqual(rt.releasable_device_bytes(segments),
                         {"releasable": 62 * mib, "fragments": 29 * mib, "private": 64 * mib, "pending": 30 * mib,
                          "segments": 6})
        strict = rt.strict_device_headroom({"free": 5 * mib}, segments)
        self.assertEqual((strict["headroom"], strict["free"], strict["releasable"], strict["unreleasable"]),
                         (67 * mib, 5 * mib, 62 * mib, 123 * mib))
        # the probes' pattern: 3.3 MiB holes between live blocks -- nothing releasable, all of it fragments
        holes = _segment(expandable=True, blocks=[item for _ in range(10) for item in
                                                  (("active_allocated", int(1.7 * mib)), ("inactive", int(3.3 * mib)))])
        self.assertEqual(rt.releasable_device_bytes([holes])["releasable"], 0)
        self.assertEqual(rt.EXPANDABLE_PAGE_BYTES, {"small": 2 << 20, "large": 20 << 20})

    def test_the_tier_snapshots_only_cuda_or_an_injected_provider(self):
        provider = _FakeSegments([_segment(expandable=False, blocks=[("inactive", 4096)])])
        injected = RawChunkTier(device_memory=_FakeDevice(), device_segments=provider)
        self.addCleanup(injected.close)
        self.assertEqual(len(injected._segments(torch.device("cuda", 0))), 1)
        fake_only = RawChunkTier(device_memory=_FakeDevice())  # a fake device memory: no snapshot to take
        self.addCleanup(fake_only.close)
        self.assertIsNone(fake_only._segments(torch.device("cuda", 0)))
        plain = RawChunkTier()
        self.addCleanup(plain.close)
        self.assertIsNone(plain._segments(torch.device("cpu")))
        with self.assertRaises(ConfigurationError):
            RawChunkTier(device_segments=5)


class PressureTests(unittest.TestCase):
    """Freeze F pressure response (coordinator 05:20; DESIGN_D_tiers.md §13): below the device reserve, GPU raw copies
    of fitted and published chunks demote to host -- in the background at admission, synchronously at the decode seam
    -- least recently used first, never unpublished copies or explicit gpu_raw placements; on an OOM in the decode
    section everything demotes and the error is re-raised. CPU tensors stand in for device copies."""

    def tier(self, device: _FakeDevice, **kwargs) -> RawChunkTier:
        kwargs.setdefault("gpu_budget_bytes", 1 << 30)
        kwargs.setdefault("host_budget_bytes", 1 << 30)
        tier = RawChunkTier(device_memory=device, **kwargs)
        self.addCleanup(tier.close)
        return tier

    @staticmethod
    def publish(tier, entry, ref):
        with tier._cond:
            tier.fitted(entry)
            tier.published(entry, ref)

    def test_no_pressure_no_demotion_and_no_query_without_device_copies(self):
        device = _FakeDevice()
        tier = self.tier(device)
        entries = [_adopt(tier, f"c{index}", _matrix(index)) for index in range(2)]
        for index, entry in enumerate(entries):
            self.publish(tier, entry, f"ref-{index}")
        tier.flush(30)
        self.assertTrue(all(entry.on_device for entry in entries))
        event = tier.pressure_check("decode")
        self.assertFalse(event["pressure"])
        stats = tier.stats()["pressure"]
        self.assertGreaterEqual(stats["checks"], 2)  # admission (background) and decode
        self.assertEqual((stats["demotions"], stats["demoted_bytes"]), (0, 0))
        empty = self.tier(_FakeDevice())
        self.assertIsNone(empty.pressure_check("decode"))
        self.assertEqual(empty._device_memory.calls, 0)  # no device copies: no device query at all

    def test_decode_pressure_demotes_lru_published_copies_and_waits(self):
        device = _FakeDevice()
        nbytes = _matrix(0).numel() * 2
        tier = self.tier(device, gpu_reserve_bytes=4 * nbytes)
        old, recent, placed = (_adopt(tier, name, _matrix(index)) for index, name in enumerate(("old", "recent",
                                                                                                "placed")))
        for entry in (old, recent, placed):
            self.publish(tier, entry, f"ref-{entry.label[1]}")
        pending = _adopt(tier, "pending", _matrix(9))  # unpublished: keeps C3's device copy
        with tier._cond:
            placed.placed = "gpu_raw"  # an explicit placement (place_raw): never demoted by pressure
            tier._touch(recent)
        tier.flush(30)
        device.memory["free"] = int(2.5 * nbytes)  # deficit 1.5 entries: the two least recently used published go
        event = tier.pressure_check("decode", wait=True, timeout=30)
        self.assertTrue(event["pressure"])
        self.assertEqual((event["selected_entries"], event["freed_bytes"]), (2, 2 * nbytes))
        self.assertEqual(event["placed_skipped_bytes"], nbytes)
        self.assertTrue(event["waited"] and not event["timed_out"])
        self.assertEqual([entry.on_device for entry in (old, recent, placed, pending)], [False, False, True, True])
        self.assertEqual([entry.host_state for entry in (old, recent)], ["complete", "complete"])
        self.assertEqual([entry.placed for entry in (old, recent)], [None, None])  # not a placement: evictable
        _check_invariants(self, tier)
        stats = tier.stats()["pressure"]
        self.assertEqual((stats["demotions"], stats["demoted_entries"], stats["demoted_bytes"]), (1, 2, 2 * nbytes))
        self.assertEqual(stats["events"][-1]["where"], "decode")
        self.assertGreaterEqual(stats["max_wait_seconds"], 0.0)
        lease = tier.acquire("ref-old")  # still served raw, from host
        self.assertEqual(lease.tier, "host_raw")
        self.assertTrue(torch.equal(lease.matrix(0, KEY), _matrix(0)))
        lease.release()

    def test_a_synchronous_demotion_returns_after_the_device_copies_are_freed(self):
        """F gpucheck 4422935: the demotion's waiter woke at the spill's completion notify while the spill job's
        locals still held the chunk's device copies, so the decode check measured the headroom and emptied the CUDA
        cache before they were freed. Here the worker lingers after that notify; the copies must already be gone."""
        device = _FakeDevice()
        nbytes = _matrix(0).numel() * 2
        tier = self.tier(device, gpu_reserve_bytes=4 * nbytes)
        tensors = [_matrix(index) for index in range(2)]
        alive = [weakref.ref(tensor) for tensor in tensors]
        entries = [_adopt(tier, f"c{index}", tensors[index]) for index in range(2)]
        del tensors
        for index, entry in enumerate(entries):
            self.publish(tier, entry, f"ref-{index}")
        tier.flush(30)
        spill = tier._spill

        def lingering_spill(entry, sources):  # e.g. a worker descheduled right after the completion notify
            spill(entry, sources)
            time.sleep(0.3)

        tier._spill = lingering_spill
        device.memory["free"] = 0
        event = tier.pressure_check("decode", wait=True, timeout=30)
        self.assertEqual((event["selected_entries"], event["freed_bytes"]), (2, 2 * nbytes))
        self.assertEqual([ref() is None for ref in alive], [True, True])
        self.assertEqual([entry.host_state for entry in entries], ["complete", "complete"])
        tier.flush(30)
        _check_invariants(self, tier)

    def test_admission_checks_demote_in_the_background(self):
        device = _FakeDevice(free=0)
        nbytes = _matrix(0).numel() * 2
        tier = self.tier(device, gpu_reserve_bytes=nbytes)
        first = _adopt(tier, "c0", _matrix(0))
        tier.flush(30)
        self.assertTrue(first.on_device)  # unpublished: no pressure demotion
        self.publish(tier, first, "ref-0")
        second = _adopt(tier, "c1", _matrix(1))  # the admission check runs on the tier's worker
        tier.flush(30)
        self.assertEqual((first.on_device, first.host_state, second.on_device), (False, "complete", True))
        stats = tier.stats()["pressure"]
        self.assertGreaterEqual(stats["by_where"]["admission"], 2)
        self.assertEqual(stats["wait_seconds"], 0.0)  # background: nobody waited
        _check_invariants(self, tier)

    def test_a_fragmented_cache_demotes_only_at_the_synchronous_checks(self):
        """Admission (background) counts the allocator's split fragments as room: a process whose cache is merely
        fragmented keeps its GPU raw copies (restore tiers unchanged); the decode check does not count them."""
        nbytes = _matrix(0).numel() * 2
        device = _FakeDevice(free=0, reserved=10 * nbytes, allocated=4 * nbytes, inactive_split=6 * nbytes)
        tier = self.tier(device, gpu_reserve_bytes=2 * nbytes)
        old = _adopt(tier, "c0", _matrix(0))
        self.publish(tier, old, "ref-0")
        _adopt(tier, "c1", _matrix(1))  # admission check: optimistic headroom 6 entries >= 2
        tier.flush(30)
        self.assertTrue(old.on_device)
        self.assertEqual(tier.stats()["pressure"]["demotions"], 0)
        event = tier.pressure_check("decode", wait=True, timeout=30)  # conservative headroom 0 < 2 entries
        self.assertTrue(event["pressure"])
        self.assertEqual(event["inactive_split"], 6 * nbytes)
        self.assertFalse(old.on_device)

    def test_auto_reserve_and_working_set_learning(self):
        device = _FakeDevice(total=1000 * (1 << 20), allocated=100 << 20, peak=300 << 20)
        tier = self.tier(device)
        self.assertEqual(tier.reserve_bytes(), 0)
        entry = _adopt(tier, "c0", _matrix(0))
        tier.flush(30)
        floor = max(2 * rt._align(entry.nbytes), int(0.05 * device.memory["total"]))
        self.assertEqual(tier.reserve_bytes(), floor)  # the admission check recorded the device total
        token = tier.section_begin()
        device.memory.update(peak=900 << 20)  # the decode set a new peak: its working set is 800 MiB
        self.assertEqual(tier.section_end(token), 800 << 20)
        self.assertEqual(tier.reserve_bytes(), int(1.1 * (800 << 20)))
        token = tier.section_begin()
        self.assertIsNone(tier.section_end(token))  # no new peak: nothing learned
        stats = tier.stats()["pressure"]
        self.assertEqual((stats["working_set_bytes"], stats["reserve_source"]), (800 << 20, "auto"))

    def test_a_section_adds_back_the_raw_copies_the_tier_released_during_it_and_only_clean_ones_teach_the_floor(self):
        """dev22j8v5 (F5.4 finding 1: RAVEN's paused decodes read 5.60 GiB of a 6.81 GiB need -- the tier dropped a
        1.29 GB raw copy meanwhile): the GPU raw bytes the tier releases during a section before its peak lowered that
        peak, not the decode's need, and are added back; a release after the peak is not. Only a clean section (no
        node ran unpaused through it) raises the clean working set, the decode floor's learned value; every measured
        one raises the auto reserve."""
        device = _FakeDevice(total=1000 * (1 << 20), allocated=100 << 20, peak=300 << 20)
        tier = self.tier(device)
        entry, late = _adopt(tier, "c0", _matrix(0)), _adopt(tier, "c1", _matrix(1))
        self.publish(tier, entry, "ref-0")
        self.publish(tier, late, "ref-1")
        tier.flush(30)
        tier.demote([late], wait=True, timeout=30, place=False)  # before the section: not its release
        token = tier.section_begin()
        tier.demote([entry], wait=True, timeout=30, place=False)  # the tier releases a copy inside the section
        self.assertFalse(entry.on_device)
        device.memory.update(peak=700 << 20)  # the section's peak comes after the release
        self.assertEqual(tier.section_end(token), (600 << 20) + entry.nbytes)  # not clean: the auto reserve only
        stats = tier.stats()["pressure"]
        self.assertEqual((stats["working_set_bytes"], stats["clean_working_set_bytes"], stats["device_released_bytes"]),
                         ((600 << 20) + entry.nbytes, 0, entry.nbytes + late.nbytes))
        self.assertEqual(tier.learned_clean_working_set_bytes(), 0)
        again = _adopt(tier, "c2", _matrix(2))
        self.publish(tier, again, "ref-2")
        tier.flush(30)
        token = tier.section_begin()
        device.memory.update(peak=720 << 20)  # the peak first, then the release: it did not lower the peak
        tier.demote([again], wait=True, timeout=30, place=False)
        self.assertEqual(tier.section_end(token), 620 << 20)
        token = tier.section_begin()  # nothing released inside this one
        device.memory.update(peak=800 << 20)
        self.assertEqual(tier.section_end(token, clean=True), 700 << 20)
        self.assertEqual(tier.learned_clean_working_set_bytes(), 700 << 20)
        self.assertEqual((tier._open_sections, len(tier._section_drops)), (0, 0))
        token = tier.section_begin()
        device.memory.update(peak=850 << 20)
        self.assertEqual(tier.section_end(token, clean=True), 750 << 20)
        self.assertEqual((tier.learned_clean_working_set_bytes(), tier.stats()["pressure"]["working_set_bytes"]),
                         (750 << 20, 750 << 20))

    def test_reserve_zero_turns_the_response_off(self):
        device = _FakeDevice(free=0)
        tier = self.tier(device, gpu_reserve_bytes=0)
        entry = _adopt(tier, "c0", _matrix(0))
        self.publish(tier, entry, "ref-0")
        tier.flush(30)
        self.assertIsNone(tier.pressure_check("decode"))
        self.assertTrue(entry.on_device)
        self.assertEqual((device.calls, tier.stats()["pressure"]["checks"]), (0, 0))

    def test_oom_relief_demotes_everything_and_the_section_reraises(self):
        device = _FakeDevice()
        tier = self.tier(device)
        published, placed = _adopt(tier, "p", _matrix(0)), _adopt(tier, "g", _matrix(1))
        self.publish(tier, published, "ref-p")
        self.publish(tier, placed, "ref-g")
        pending = _adopt(tier, "u", _matrix(2))
        with tier._cond:
            placed.placed = "gpu_raw"
        tier.flush(30)
        archive = _fake_archive(tier, defer=False)
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")
        relief = report["oom_relief"]
        self.assertEqual((relief["where"], relief["selected_entries"]), ("oom", 3))
        self.assertEqual([entry.on_device for entry in (published, placed, pending)], [False, False, False])
        self.assertTrue(all(entry.host_state == "complete" for entry in (published, placed, pending)))  # I1
        self.assertEqual(tier.stats()["pressure"]["oom_reliefs"], 1)
        _check_invariants(self, tier)

    def test_the_decode_seam_checks_without_deferral_and_learns_the_working_set(self):
        device = _FakeDevice(allocated=10 << 20, peak=20 << 20)
        nbytes = _matrix(0).numel() * 2
        tier = self.tier(device, gpu_reserve_bytes=nbytes)
        entry = _adopt(tier, "c0", _matrix(0))
        self.publish(tier, entry, "ref-0")
        tier.flush(30)
        device.memory["free"] = 0  # the decode needs room: the seam demotes before the section
        archive = _fake_archive(tier, defer=False)
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            self.assertFalse(entry.on_device)  # demoted before the section body runs
            device.memory["peak"] = 50 << 20  # the decode's peak
        self.assertTrue(report["raw_pressure"]["pressure"])
        self.assertTrue(report["deferred_archival"])  # entry pressure defers archival for the section (dev22j2)
        self.assertEqual(report["raw_pressure"]["where"], "vae_decode")
        self.assertEqual(report["working_set_bytes"], 40 << 20)
        self.assertEqual(tier.stats()["pressure"]["working_set_bytes"], 40 << 20)

    def test_config_validation_and_cuda_provider_shape(self):
        from reelcache.integrations.sglang import ReelCacheRuntimeConfig
        with self.assertRaises(ConfigurationError):
            ReelCacheRuntimeConfig.from_dict({"store_root": str(s6.ROOT / "reserve-config"), "gpu_raw_reserve_bytes": -1})
        config = ReelCacheRuntimeConfig.from_dict({"store_root": str(s6.ROOT / "reserve-config"),
                                                "gpu_raw_reserve_bytes": 123})
        self.assertEqual(config.gpu_raw_reserve_bytes, 123)
        self.assertNotIn("gpu_raw_reserve_bytes", config.content_dict())  # execution, not content
        with self.assertRaises(ConfigurationError):
            RawChunkTier(host_budget_bytes=1 << 20, gpu_reserve_bytes=-5)
        self.assertEqual(rt.device_headroom({"free": 100, "reserved": 50, "allocated": 20, "inactive_split": 10}),
                         120)
        self.assertEqual(rt.device_headroom({"free": 100, "reserved": 50, "allocated": 45, "inactive_split": 10}),
                         100)



class DeviceTrimRegistryTests(unittest.TestCase):
    """``reelcache.device_pressure`` (freeze F): trims run in the fixed order for their device, stop once the headroom
    covers the need, record bytes / seconds / errors per component, then empty the cache once; bound methods are
    held weakly."""

    def test_order_device_filter_early_stop_errors_and_weak_methods(self):
        from reelcache import device_pressure as dp
        registry = dp.DeviceTrimRegistry()
        calls, state = [], {"free": 0}

        class Component:
            def __init__(self, name, freed):
                self.name, self.freed = name, freed

            def trim_device_cache(self, reason):
                calls.append((self.name, reason))
                state["free"] += self.freed
                return self.freed

        restorer, pool = Component("restorer", 30), Component("decoder_pool", 50)
        resident = Component("bridge_resident", 100)
        other_device = Component("decoder_pool:1", 7)

        def broken(reason):
            raise RuntimeError("boom")

        registry.register("restorer", restorer.trim_device_cache, device=0)
        registry.register("bridge_resident", resident.trim_device_cache, device=0)
        registry.register("decoder_pool", pool.trim_device_cache, device=0)
        registry.register("decoder_pool:1", other_device.trim_device_cache, device=1)
        handle = registry.register("broken", broken, order=5)
        self.assertEqual([item["name"] for item in registry.registered()],
                         ["broken", "decoder_pool", "decoder_pool:1", "restorer", "bridge_resident"])
        emptied = []
        report = registry.trim("decode", device=0, needed_bytes=70,
                               headroom=lambda: state["free"], empty_cache=lambda: emptied.append(1))
        self.assertEqual(calls, [("decoder_pool", "decode"), ("restorer", "decode")])  # 80 >= 70: stop
        self.assertEqual([(item["name"], item["bytes"]) for item in report["trims"]],
                         [("broken", 0), ("decoder_pool", 50), ("restorer", 30)])
        self.assertIn("RuntimeError: boom", report["trims"][0]["error"])
        self.assertEqual((report["bytes"], emptied), (80, [1]))
        handle.unregister()
        del pool
        import gc
        gc.collect()
        self.assertEqual([item["name"] for item in registry.registered()],
                         ["decoder_pool:1", "restorer", "bridge_resident"])  # the collected pool dropped out
        calls.clear()
        report = registry.trim("oom", device=0, empty_cache=lambda: None)
        self.assertEqual(calls, [("restorer", "oom"), ("bridge_resident", "oom")])

    def test_pressure_events_trim_after_demotion_and_at_request_start(self):
        from reelcache import device_pressure as dp
        device = _FakeDevice()
        nbytes = _matrix(0).numel() * 2
        tier = RawChunkTier(device_memory=device, gpu_budget_bytes=1 << 30, host_budget_bytes=1 << 30,
                            gpu_reserve_bytes=10 * nbytes)
        self.addCleanup(tier.close)
        entry = _adopt(tier, "c0", _matrix(0))
        with tier._cond:
            tier.fitted(entry)
            tier.published(entry, "ref-0")
        tier.flush(30)
        freed = []

        class Canvases:
            def trim_device_cache(self, reason):
                freed.append(reason)
                device.memory["free"] += 6 * nbytes
                return 6 * nbytes

        canvases = Canvases()
        handle = dp.register_device_trim("decoder_pool", canvases.trim_device_cache)
        self.addCleanup(handle.unregister)
        device.memory["free"] = 2 * nbytes  # the demotion frees 1 entry (fake: not reflected); trims cover the rest
        with mock.patch.object(dp.REGISTRY, "trim", wraps=dp.REGISTRY.trim) as trim:
            event = tier.pressure_check("decode", wait=True, timeout=30)
        self.assertTrue(event["pressure"] and not entry.on_device)
        self.assertEqual(trim.call_args.args[0], "decode")
        decoder = [item for item in event["trims"]["trims"] if item["name"] == "decoder_pool"]
        self.assertEqual(decoder[0]["bytes"], 6 * nbytes)
        self.assertEqual(event["headroom_after"], 8 * nbytes)
        stats = tier.stats()["pressure"]
        self.assertEqual(stats["trimmed_by_component"]["decoder_pool"], 6 * nbytes)
        # request start without GPU raw copies: the trims alone, for the KV cache about to be allocated
        archive = _fake_archive(tier, defer=False)
        tier._device_hint = torch.device("cpu")
        device.memory["free"] = 0
        event = archive.relieve_pressure("request_start", needed_bytes=3 * nbytes)
        self.assertTrue(event["pressure"])
        self.assertEqual(event["trims"]["bytes"], 6 * nbytes)
        self.assertEqual(freed, ["decode", "request_start"])
        self.assertEqual(tier.stats()["pressure"]["by_where"]["request_start"], 1)

    def test_the_restorer_trims_its_unpinned_device_bases(self):
        from reelcache.archive_restore import DeviceBasisCache, LineageRestorer
        restorer = LineageRestorer.__new__(LineageRestorer)
        restorer.stats_ = {"device_trims": 0, "device_trimmed_bytes": 0}
        restorer.bases = DeviceBasisCache.__new__(DeviceBasisCache)
        forgotten = []
        restorer.bases.refs = lambda: ["basis:a", "basis:b"]
        restorer.bases.forget = lambda refs: forgotten.append(list(refs)) or {"versions": 2, "bytes": 123, "held": 0}
        self.assertEqual(restorer.trim_device_cache("decode"), 123)
        self.assertEqual(forgotten, [["basis:a", "basis:b"]])
        self.assertEqual((restorer.stats_["device_trims"], restorer.stats_["device_trimmed_bytes"]), (1, 123))

    def test_the_spill_worker_drops_the_spilled_device_copies(self):
        """5b's carry-over census (0.8 GiB of chunk-sized BF16 buffers): the worker's last spill kept its sources
        alive after the device copy was dropped. They must die with the drop."""
        import gc
        import weakref
        tier = RawChunkTier(gpu_budget_bytes=0, host_budget_bytes=1 << 30)
        self.addCleanup(tier.close)
        source = _matrix(3)
        reference = weakref.ref(source)
        entry = _adopt(tier, "c0", source)
        del source
        tier.flush(30)
        gc.collect()
        self.assertEqual((entry.on_device, entry.host_state), (False, "complete"))
        self.assertIsNone(reference())

class _FakePipeline:
    @staticmethod
    def stats():
        return {"deferral_released_bytes": 0}

    @staticmethod
    def defer_gpu_work(**kwargs):
        return mock.MagicMock()


def _fake_archive(tier, *, defer: bool):
    """The attributes ``LineageSessionArchive.memory_critical`` reads, with its helper methods bound."""
    import types
    archive = SimpleNamespace(config=SimpleNamespace(defer_archival_during_decode=defer), fitter=object(),
                              raw_tier=tier, pipeline=_FakePipeline(), device=torch.device("cpu"))
    archive._deferred_section = types.MethodType(la.LineageSessionArchive._deferred_section, archive)
    archive._pressure_deferred_section = types.MethodType(la.LineageSessionArchive._pressure_deferred_section,
                                                          archive)
    archive.relieve_pressure = types.MethodType(la.LineageSessionArchive.relieve_pressure, archive)
    archive._oom_relief = la.LineageSessionArchive._oom_relief
    return archive


class _Pipelines:
    def pipeline(self, *, tier=None, hooks=None, name="main", lineages=("main",), **kwargs):
        runtime = s6.RuntimeFactory(self)(name)
        for lineage in lineages:
            runtime.create_lineage(lineage)
        pipeline = ArchivePipeline(runtime, method=s6.method(), registry=s6.registry(),
                                   options=ArchiveOptions(encode_threads=1), raw_tier=tier, hooks=hooks, **kwargs)
        self.addCleanup(lambda: pipeline.close(drain=False, timeout=60))
        return runtime, pipeline

    @staticmethod
    def content(runtime, ref):
        rows = s6.node_content(runtime, ref)["rows"]
        return {key: (row[0], row[1], row[4], row[5]) for key, row in rows.items()}  # decision, rank, energy, payloads


class PipelineTierTests(_Pipelines, unittest.TestCase):
    """The pipeline with a raw tier (and without: C3)."""

    def serial(self, tier, *, demote_at=None):
        runtime, pipeline = self.pipeline(tier=tier)
        refs = []
        for index in range(3):
            matrices, contexts = s6.chunk(index)
            pipeline.submit("main", f"chunk-{index:06d}", matrices, contexts)
            refs.append(pipeline.drain("main", timeout=120))
            if demote_at == index:
                tier.demote(tier.entries(), timeout=30)
        return runtime, pipeline, refs

    def test_content_and_ids_are_independent_of_tier_placement(self):
        """R3: serial runs (identical registry visibility) give identical node ids and contents with the tier off
        (C3), on, with every source spilled and re-staged (GPU budget 0), and with a demotion mid-run."""
        reference_runtime, _, reference = self.serial(None)
        expected = [s6.node_content(reference_runtime, ref) for ref in reference]
        for kwargs in ({}, {"gpu_budget_bytes": 0}, {"demote": 1}):
            demote = kwargs.pop("demote", None)
            tier = RawChunkTier(host_budget_bytes=1 << 30, **kwargs)
            runtime, pipeline, refs = self.serial(tier, demote_at=demote)
            self.assertEqual(refs, reference, kwargs)
            self.assertEqual([s6.node_content(runtime, ref) for ref in refs], expected)
            stats = pipeline.stats()["raw_tier"]
            if kwargs.get("gpu_budget_bytes") == 0:
                self.assertGreaterEqual(stats["spills"], 3)
            if demote is not None:
                self.assertGreaterEqual(stats["demotions"], 1)

    def test_concurrent_sessions_with_random_delays_store_equal_payloads(self):
        """R3: two sessions archived concurrently, random fit/encode/store/publish delays and spills: payloads,
        decisions and ranks equal a serial C3 reference (ids may differ through registry provenance)."""
        reference_runtime, _, reference = self.serial(None)
        expected = [self.content(reference_runtime, ref) for ref in reference]
        generator = random.Random(99)
        lock = threading.Lock()

        def hooks(stage, node_id):
            with lock:
                delay = generator.uniform(0.0, 0.03)
            time.sleep(delay)

        tier = RawChunkTier(gpu_budget_bytes=0, host_budget_bytes=1 << 30)
        runtime, pipeline = self.pipeline(tier=tier, hooks=hooks, lineages=("a", "b"))
        errors = []

        def produce(branch):
            try:
                for index in range(3):
                    matrices, contexts = s6.chunk(index)
                    pipeline.submit(branch, f"chunk-{index:06d}", matrices, contexts)
                    time.sleep(generator.uniform(0.0, 0.02))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=produce, args=(branch,)) for branch in ("a", "b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(120)
        self.assertEqual(errors, [])
        for branch in ("a", "b"):
            pipeline.drain(branch, timeout=120)
            path = runtime.node_path(runtime.lineage_head(branch))[1:]
            self.assertEqual([self.content(runtime, item[0]) for item in path], expected)
        tier.flush(30)
        self.assertGreaterEqual(pipeline.stats()["raw_tier"]["spills"], 1)

    def test_submit_never_blocks_until_the_budgets_fill_then_records_the_last_resort(self):
        def slow(stage, node_id):
            if stage == "plan":
                time.sleep(0.4)

        tier = RawChunkTier(host_budget_bytes=1 << 30)
        _, pipeline = self.pipeline(tier=tier, hooks=slow)
        started = time.perf_counter()
        tickets = []
        for index in range(4):
            matrices, contexts = s6.chunk(index)
            tickets.append(pipeline.submit("main", f"chunk-{index:06d}", matrices, contexts))
        self.assertLess(time.perf_counter() - started, 1.0)  # the slow fitter never held generation
        # Back-to-back submits may wait for a demotion copy (gpu_spill, milliseconds), never for archival.
        self.assertTrue(all(ticket.admission["raw_tier"] and ticket.admission["cause"] in (None, "gpu_spill")
                            for ticket in tickets))
        pipeline.drain("main", timeout=120)
        nbytes = tickets[0].admission["bytes"]
        tight = RawChunkTier(gpu_budget_bytes=nbytes, host_budget_bytes=rt._align(nbytes) + 4096)
        _, small = self.pipeline(tier=tight, hooks=slow, name="tight")
        waits = []
        for index in range(3):
            matrices, contexts = s6.chunk(index)
            waits.append(small.submit("main", f"chunk-{index:06d}", matrices, contexts).admission)
        self.assertFalse(waits[1]["waited"])
        self.assertEqual((waits[2]["waited"], waits[2]["cause"]), (True, "host_full"))
        stats = small.stats()
        self.assertEqual(stats["raw_tier"]["last_resort_causes"]["host_full"], 1)
        self.assertGreater(stats["raw_tier"]["last_resort_backpressure_seconds"], 0.0)
        self.assertEqual(stats["raw_tier"]["last_resort_events"][-1]["node_id"], "chunk-000002")
        self.assertEqual(stats["backpressure_waits"], 1)

    def test_failed_commits_drop_their_raw_entries(self):
        def fail(stage, node_id):
            if stage == "encode" and node_id == "chunk-000001":
                raise RuntimeError("injected encode failure")

        tier = RawChunkTier(host_budget_bytes=1 << 30)
        _, pipeline = self.pipeline(tier=tier, hooks=fail)
        for index in range(3):
            matrices, contexts = s6.chunk(index)
            pipeline.submit("main", f"chunk-{index:06d}", matrices, contexts)
        with self.assertRaises(ArchiveFailure):
            pipeline.drain("main", timeout=120)
        pipeline.reset("main", timeout=120)
        tier.flush(30)
        stats = tier.stats()
        self.assertEqual(stats["unpublished_bytes"], 0)
        self.assertEqual(stats["drops_failed"], 2)  # the failed chunk and its descendant (R7)
        self.assertEqual(stats["entries"]["total"], 1)  # the published chunk 0

    def test_bookkeeping_error_after_publication_disables_the_tier_not_the_commit(self):
        tier = RawChunkTier(host_budget_bytes=1 << 30)
        runtime, pipeline = self.pipeline(tier=tier)
        with mock.patch.object(tier, "published", side_effect=RuntimeError("bookkeeping bug")):
            matrices, contexts = s6.chunk(0)
            ticket = pipeline.submit("main", "chunk-000000", matrices, contexts)
            head = pipeline.drain("main", timeout=120)
        self.assertEqual(ticket.result(0), head)
        self.assertEqual(runtime.lineage_head("main"), head)
        self.assertTrue(tier.stats()["disabled_reason"].startswith("bookkeeping error"))
        matrices, contexts = s6.chunk(1)
        later = pipeline.submit("main", "chunk-000001", matrices, contexts)  # C3 bounds from now on
        self.assertFalse(later.admission["raw_tier"])
        pipeline.drain("main", timeout=120)

    def test_raw_tier_off_is_the_c3_pipeline(self):
        """``raw_tier=False``: no tier object, C3 bounds (max_unpublished / source leases) and releases at fit."""
        released = []

        def slow(stage, node_id):
            if stage == "plan":
                time.sleep(0.3)

        _, pipeline = self.pipeline(hooks=slow, max_source_chunks=1, max_unpublished=1)
        tickets = []
        for index in range(2):
            matrices, contexts = s6.chunk(index)
            tickets.append(pipeline.submit("main", f"chunk-{index:06d}", matrices, contexts,
                                           release=released.append))
        pipeline.drain("main", timeout=120)
        self.assertIsNone(pipeline.raw_tier)
        self.assertIsNone(pipeline.stats()["raw_tier"])
        self.assertEqual([ticket.admission["raw_tier"] for ticket in tickets], [False, False])
        self.assertEqual((tickets[1].admission["waited"], tickets[1].admission["cause"]), (True, "c3_bounds"))
        self.assertEqual(len(released), 2)  # sources released when fitted, as in C3
        self.assertIsNone(tickets[0].admission["bytes"])

    def test_publication_records_lag_fits_ranks_planner_and_encoder(self):
        tier = RawChunkTier(host_budget_bytes=1 << 30)
        _, pipeline = self.pipeline(tier=tier)
        tickets = []
        for index in range(3):
            matrices, contexts = s6.chunk(index)
            tickets.append(pipeline.submit("main", f"chunk-{index:06d}", matrices, contexts))
        pipeline.drain("main", timeout=120)
        stats = pipeline.stats()
        self.assertEqual(stats["commit_to_publish"]["count"], 3)
        self.assertEqual(len(stats["commit_to_publish"]["recent"]), 3)
        self.assertGreater(stats["commit_to_publish"]["max"], 0.0)
        self.assertEqual(stats["fits"]["nodes"], 3)
        self.assertEqual(stats["fits"]["total"], stats["fits"]["fresh"] + stats["fits"]["refit"]
                         + stats["fits"]["diagnostic_full_width"])
        self.assertEqual(stats["ranks"]["matrices"], 3 * len(s6.KEYS))
        self.assertGreater(stats["ranks"]["mean"], 0)
        self.assertGreater(stats["planner_seconds"], 0.0)
        self.assertEqual(stats["encoder"]["encoder"], "libx265")
        self.assertEqual(stats["encoder"]["commits"], {"libx265": 3})
        self.assertEqual([ticket.encoder for ticket in tickets], ["libx265"] * 3)


class BridgeTierTests(tb.BridgeTestCase):
    """Restores by tier through the SGLang bridge (CPU toy engine)."""

    representation = "svd_hevc"

    def test_restores_take_raw_copies_bitwise_then_host_then_compressed(self):
        reference = self.bridge("reference", representation="raw")
        _, _, raw_caches, _ = tb.drive(reference, tb.request(tb.handle("ref"), blocks=3))
        bridge = self.bridge()
        first, _, _, _ = tb.drive(bridge, tb.request(tb.handle("s1", pause_after_chunks=2), blocks=3))
        tokens = 2 * tb.FRAMES * tb.TPF
        forks = iter(range(10))

        def fork():
            name = f"f{next(forks)}"
            summary, _, caches, _ = tb.drive(bridge, tb.request(
                tb.handle(name, "fork", from_session="s1", checkpoint=2), blocks=3))
            tiers = [item["tier"] for item in summary["restore"]["lease"]["tiers"]["per_chunk"]]
            exact = all(torch.equal(getattr(caches["cond"][layer], key)[:, :tokens],
                                    getattr(raw_caches["cond"][layer], key)[:, :tokens])
                        for layer in range(tb.LAYERS) for key in ("k", "v"))
            return summary, tiers, exact

        summary, tiers, exact = fork()
        self.assertEqual(tiers, ["gpu_raw", "gpu_raw"])  # the CPU stand-in for the device copies
        self.assertTrue(exact)  # RAW install with the engine's RoPE: bitwise the native cache
        self.assertEqual(summary["restore"]["lease"]["verification"],
                         {"mode": "sync", "verified_payloads": 0, "payload_wait_seconds": 0.0})  # all raw: none read
        self.assertLifetimesRetired(bridge, summary)
        report = bridge.archive.demote_raw("s1", timeout=30)
        self.assertEqual(report["failed"], 0)
        summary, tiers, exact = fork()
        self.assertEqual((tiers, exact), (["host_raw", "host_raw"], True))
        bridge.archive.drop_raw("s1", timeout=60)
        summary, tiers, exact = fork()
        self.assertEqual(len(tiers), 2)
        self.assertTrue(all(tier in ("compressed_prepared", "compressed_cold") for tier in tiers), tiers)
        self.assertFalse(exact)  # the lossy restore
        stats = bridge.archive.stats()["pipeline"]["raw_tier"]
        self.assertGreaterEqual(stats["restores"]["gpu_raw"], 2)
        self.assertGreaterEqual(stats["restores"]["host_raw"], 2)
        self.assertLifetimesRetired(bridge, summary)
        self.assertEqual(first["archive"]["backend"], "lineage")

    def test_compressed_restores_report_verified_payloads(self):
        """``restore.lease.verification`` of compressed restores with the store verifying reads: the configured mode
        and the payloads the restore verified (> 0), on the CPU path (``SoftwareLineageRestorer``) and the store path
        (the packed restorer without a staging engine; ``SoftwarePool`` stands in for NVDEC); without verification
        the mode is off and nothing is counted."""
        from reelcache.archive_restore import LineageRestorer
        from tests.test_packed_nvdec import SoftwarePool

        def fork(bridge, name):
            summary, _, _, _ = tb.drive(bridge, tb.request(tb.handle(name, "fork", from_session="s1", checkpoint=2),
                                                           blocks=3))
            tiers = {item["tier"] for item in summary["restore"]["lease"]["tiers"]["per_chunk"]}
            self.assertTrue(tiers <= {"compressed_prepared", "compressed_cold"}, tiers)
            self.assertLifetimesRetired(bridge, summary)
            return summary["restore"]["lease"]["verification"]

        bridge = self.bridge(raw_restore="off")
        tb.drive(bridge, tb.request(tb.handle("s1", pause_after_chunks=2), blocks=3))
        cpu = fork(bridge, "cpu")
        self.assertEqual(cpu["mode"], "sync")
        self.assertGreater(cpu["verified_payloads"], 0)
        bridge.archive.restorer = LineageRestorer(bridge.archive.runtime, SoftwarePool(), arena_bytes=1 << 30)
        store_path = fork(bridge, "store-path")
        self.assertEqual(store_path["mode"], "sync")
        self.assertGreater(store_path["verified_payloads"], 0)
        unverified = self.bridge("unverified", raw_restore="off", verify_reads=False)
        tb.drive(unverified, tb.request(tb.handle("s1", pause_after_chunks=2), blocks=3))
        self.assertEqual(fork(unverified, "plain"), {"mode": "off", "verified_payloads": 0,
                                                     "payload_wait_seconds": 0.0})

    def test_refit_rule_reaches_the_rank_method(self):
        self.assertEqual(self.bridge("default").archive.method.refit_rule, "paper")  # user decision 2026-10-17
        self.assertEqual(self.bridge("cumulative", refit_rule="cumulative").archive.method.refit_rule, "cumulative")

    def test_raw_restore_off_and_raw_tier_off_restore_compressed(self):
        for overrides in ({"raw_restore": "off"}, {"raw_tier": False}):
            bridge = self.bridge(f"store-{len(overrides)}-{next(iter(overrides))}", **overrides)
            tb.drive(bridge, tb.request(tb.handle("s1", pause_after_chunks=1), blocks=2))
            summary, _, _, _ = tb.drive(bridge, tb.request(tb.handle("s1", "resume"), blocks=2))
            tiers = [item["tier"] for item in summary["restore"]["lease"]["tiers"]["per_chunk"]]
            self.assertTrue(all(tier.startswith("compressed") for tier in tiers), (overrides, tiers))
            if overrides.get("raw_tier") is False:
                self.assertIsNone(bridge.archive.raw_tier)
                self.assertIsNone(bridge.archive.pipeline.raw_tier)
                self.assertFalse(bridge.archive.describe()["raw_tier"]["enabled"])
            self.assertLifetimesRetired(bridge, summary)

    def test_fork_rewind_delete_hold_and_release_raw_entries(self):
        bridge = self.bridge()
        tb.drive(bridge, tb.request(tb.handle("s1", pause_after_chunks=2), blocks=3))
        archive = bridge.archive
        archive.drain("s1", timeout=60)
        archive.release("s1", timeout=60)  # pipeline state retired: the lineage head may move
        nodes = archive.path(archive.head("s1"))[1:]
        self.assertEqual(len(nodes), 2)
        archive.fork(nodes[0].ref, "f1")
        entries = {entry.node_ref: entry for entry in archive.raw_tier.entries()}
        self.assertIn("f1", entries[nodes[0].ref].holders)
        self.assertNotIn("f1", entries[nodes[1].ref].holders)
        archive.delete("s1")
        archive.raw_tier.flush(30)
        remaining = {entry.node_ref for entry in archive.raw_tier.entries()}
        self.assertEqual(remaining, {nodes[0].ref})  # the fork keeps its history's raw copy
        archive.raw_release("f1")
        archive.raw_tier.flush(30)
        self.assertEqual(archive.raw_tier.entries(), [])


class RefitRuleConfigTests(unittest.TestCase):
    """``ReelCacheRuntimeConfig.refit_rule`` (freeze D, D-A): the paper rule by default again (user decision 2026-10-17;
    freezes D to F defaulted to the cumulative rule, which stays selectable); it enters a session's content identity
    only away from the paper rule (C3 ids kept)."""

    def config(self, **values):
        from reelcache.integrations.sglang import ReelCacheRuntimeConfig
        return ReelCacheRuntimeConfig.from_dict({"store_root": str(s6.ROOT / "refit-config"), **values})

    def test_default_validation_and_content_identity(self):
        self.assertEqual(self.config().refit_rule, "paper")
        self.assertEqual(self.config(refit_rule="cumulative").refit_rule, "cumulative")
        with self.assertRaises(ConfigurationError):
            self.config(refit_rule="greedy")
        paper = self.config().content_dict()
        self.assertNotIn("refit_rule", paper)  # the C3 content of a paper-rule session
        cumulative = self.config(refit_rule="cumulative").content_dict()
        self.assertEqual(cumulative["refit_rule"], "cumulative")
        self.assertEqual(set(paper), set(cumulative) - {"refit_rule"})


class RestorerVerificationWiringTests(unittest.TestCase):
    """Step 5c's D-E/D-G wiring in the lineage restorer: the decoder pool defers queued warm-ups to the restorer's
    activity, the staging engine's verification record reaches the restore report, and the archive hoists it into
    ``restore.lease.verification`` with the configured mode (count 0 when no compressed payload was read)."""

    def restorer(self, *, engine=None, calls=None):
        from reelcache.archive_restore import LineageRestorer
        runtime = s6.RuntimeFactory(self)("wiring")
        pool = SimpleNamespace(device=torch.device("cpu"))
        if calls is not None:
            pool.defer_warm_ups_to = calls.append
        with mock.patch("reelcache.packed_decode.PackedRestorer"):
            return LineageRestorer(runtime, pool, staging_engine=engine)

    def test_pool_defers_warm_ups_to_the_restore_activity(self):
        calls = []
        restorer = self.restorer(calls=calls)
        self.assertEqual(calls, [restorer.activity])
        self.restorer()  # a pool without the hook (C3's) is accepted unchanged

    def test_staging_engine_verification_record_reaches_the_restorer(self):
        class Engine:
            def __init__(self):
                self.calls = []

            def stage(self, runtime, history):
                return SimpleNamespace(refs=["ref"], stats={"verification": {"mode": "async", "verified_payloads": 7,
                                                                             "payload_wait_seconds": 0.5}})

            def verification_report(self, store, staged=None, *, since=None):
                self.calls.append(staged)
                if staged is not None:
                    return dict(staged.stats["verification"])
                return {"mode": "async", "verified_payloads": 0, "payload_wait_seconds": 0.0}

        engine = Engine()
        restorer = self.restorer(engine=engine)
        refs, pins = restorer.node_refs([{"packet": "manifest:x", "node_id": "n"}])  # not prepared: staged
        self.assertEqual(refs, ["ref"])
        self.assertEqual(restorer.last_verification, {"mode": "async", "verified_payloads": 7,
                                                      "payload_wait_seconds": 0.5})
        restorer._staged.clear()
        restorer.node_refs([])  # nothing to stage: the configured mode, 0 payloads
        self.assertEqual(restorer.last_verification, {"mode": "async", "verified_payloads": 0,
                                                      "payload_wait_seconds": 0.0})
        self.assertIsNone(self.restorer().last_verification)  # engines without the record (C3)

    def test_archive_hoists_the_configured_mode_and_payload_counts(self):
        fake = SimpleNamespace(verification_mode=lambda: "async")
        record = la.LineageSessionArchive._verification_record
        self.assertEqual(record(fake, []), {"mode": "async", "verified_payloads": 0, "payload_wait_seconds": 0.0})
        reports = [{"verification": {"mode": "async", "verified_payloads": 3, "payload_wait_seconds": 0.25}},
                   {"verification": {"mode": "sync", "verified_payloads": 2, "payload_wait_seconds": 0.5}},
                   {"seconds": 1.0}]
        self.assertEqual(record(fake, reports), {"mode": "async", "verified_payloads": 5, "payload_wait_seconds": 0.75,
                                                 "reported_mode": "sync"})
        archive = SimpleNamespace(staging_engine=None, store=None, config=SimpleNamespace(verify_reads=True))
        self.assertEqual(la.LineageSessionArchive.verification_mode(archive), "sync")
        archive.config.verify_reads = False
        self.assertEqual(la.LineageSessionArchive.verification_mode(archive), "off")
        archive.staging_engine = SimpleNamespace(verification_mode=lambda store: "async")
        self.assertEqual(la.LineageSessionArchive.verification_mode(archive), "async")


class SealPreparationFailureTests(_Pipelines, unittest.TestCase):
    """Seal-time preparation (restorer.adopt) only speeds up later restores: a failed decoder warm-up or preparation
    is recorded and never fails the commit (freeze D, R2; Step 13's finding); integrity errors still fail it."""

    def restorer(self, runtime, pool=None):
        from reelcache.archive_restore import LineageRestorer
        from tests.test_packed_nvdec import SoftwarePool
        return LineageRestorer(runtime, pool if pool is not None else SoftwarePool(), arena_bytes=1 << 30)

    def run_chain(self, name, *, restorer_for=None, chunks=2):
        runtime = s6.RuntimeFactory(self)(name)
        runtime.create_lineage("main")
        restorer = None if restorer_for is None else restorer_for(runtime)
        pipeline = ArchivePipeline(runtime, method=s6.method(), registry=s6.registry(),
                                   options=ArchiveOptions(encode_threads=1), restorer=restorer)
        self.addCleanup(lambda: pipeline.close(drain=False, timeout=60))
        tickets = [pipeline.submit("main", f"chunk-{index:06d}", *s6.chunk(index)) for index in range(chunks)]
        return runtime, pipeline, restorer, tickets

    def test_a_failed_decoder_warm_up_is_recorded_and_the_commit_publishes(self):
        from tests.test_packed_nvdec import SoftwarePool

        class OutOfMemoryPool(SoftwarePool):
            def prewarm(self, samples, timeout=None):
                raise RuntimeError("CUDA out of memory: creating a decoder session for a new canvas shape")

        reference_runtime, reference, _, expected = self.run_chain("reference")
        reference.drain("main", timeout=120)
        runtime, pipeline, restorer, tickets = self.run_chain(
            "oom", restorer_for=lambda runtime: self.restorer(runtime, OutOfMemoryPool()))
        head = pipeline.drain("main", timeout=120)
        self.assertEqual([ticket.result(0) for ticket in tickets][-1], head)
        stats = pipeline.stats()["adopt"]
        self.assertEqual((stats["calls"], stats["failures"]), (2, 0))  # recorded inside the restorer, not raised
        self.assertGreaterEqual(stats["prewarm_failures"], 1)
        self.assertEqual(stats["prepare_failures"], 0)
        self.assertIn("out of memory", stats["last_seal_failure"])
        self.assertEqual(stats["adopted"], 2)  # the packs are prepared; only the decoder sessions stay cold
        self.assertEqual([s6.node_content(runtime, ticket.result(0)) for ticket in tickets],
                         [s6.node_content(reference_runtime, ticket.result(0)) for ticket in expected])

    def test_a_failed_preparation_leaves_the_pack_unprepared_and_the_commit_publishes(self):
        runtime, pipeline, restorer, tickets = self.run_chain("prepare", restorer_for=self.restorer)
        with mock.patch.object(type(restorer), "_adopt", side_effect=MemoryError("pinned host memory exhausted")):
            pipeline.drain("main", timeout=120)
        self.assertTrue(all(ticket.done() and ticket.error is None for ticket in tickets))
        stats = pipeline.stats()["adopt"]
        self.assertEqual((stats["prepare_failures"], stats["failures"]), (2, 0))
        self.assertIn("MemoryError", stats["last_seal_failure"])
        self.assertEqual(restorer.prepared.refs(), [])  # nothing half-prepared: a later restore prepares it

    def test_a_failure_escaping_the_restorer_is_caught_by_the_pipeline(self):
        class Restorer:
            activity = None

            def adopt(self, packet_ref, encoded):
                raise RuntimeError("decoder pool closed")

        _, pipeline, _, tickets = self.run_chain("escape", restorer_for=lambda runtime: Restorer())
        pipeline.drain("main", timeout=120)
        stats = pipeline.stats()["adopt"]
        self.assertEqual((stats["calls"], stats["failures"]), (2, 2))
        self.assertIn("decoder pool closed", stats["last_failure"])

    def test_an_integrity_error_in_seal_preparation_still_fails_the_commit(self):
        from reelcache.errors import IntegrityError
        _, pipeline, restorer, _ = self.run_chain("integrity", restorer_for=self.restorer, chunks=1)
        with mock.patch.object(type(restorer), "_adopt", side_effect=IntegrityError("sealed bytes differ")):
            with self.assertRaises(ArchiveFailure):
                pipeline.drain("main", timeout=120)


class EncoderSelectionTests(unittest.TestCase):
    """``encoder="auto"``: NVENC where the device has it, libx265 elsewhere, chosen once and recorded (R8)."""

    @staticmethod
    def config(encoder="auto"):
        return SimpleNamespace(encoder=encoder, representation="svd_hevc", hevc_qp=8, hevc_pix_fmt="yuv444p")

    def test_cpu_and_explicit_choices(self):
        with mock.patch("reelcache.nvenc.NVENCEncoder", side_effect=AssertionError("must not probe")):
            encoder, record = la.select_encoder(self.config(), torch.device("cpu"))
            self.assertIsNone(encoder)
            self.assertEqual((record["selected"], record["reason"]), ("libx265", "the bridge device is not CUDA"))
            encoder, record = la.select_encoder(self.config("x265"), torch.device("cuda", 0))
            self.assertEqual((encoder, record["reason"]), (None, "encoder='x265'"))
            with self.assertRaises(ConfigurationError):
                la.select_encoder(self.config("nvenc"), torch.device("cpu"))
        raw = SimpleNamespace(encoder="auto", representation="raw")
        self.assertEqual(la.select_encoder(raw, torch.device("cpu"))[1]["selected"], None)

    def test_auto_on_cuda_selects_nvenc_or_records_why_not(self):
        class FakeEncoder:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.settings = SimpleNamespace(label="nvenc-p1-hq", qp=kwargs["qp"])
                self.implementation = {"gpu": "RTX PRO 6000", "driver_version": "580.0", "library_version": "2.0"}

        with mock.patch("torch.cuda.mem_get_info", return_value=(10 << 30, 96 << 30)), \
                mock.patch("reelcache.nvenc.NVENCEncoder", FakeEncoder):
            encoder, record = la.select_encoder(self.config(), torch.device("cuda", 0))
        self.assertIsInstance(encoder, FakeEncoder)
        self.assertEqual(encoder.kwargs, {"qp": 8, "preset": "p1", "tuning": "hq", "gpu_id": 0,
                                          "pix_fmts": ("yuv444p",)})
        self.assertEqual((record["selected"], record["preset"], record["driver"]), ("nvenc", "nvenc-p1-hq", "580.0"))
        self.assertEqual(record["nvenc_device_bytes"], 0)
        failing = mock.patch("reelcache.nvenc.NVENCEncoder", side_effect=UnsupportedError("no NVENC on A100"))
        with mock.patch("torch.cuda.mem_get_info", return_value=(10 << 30, 40 << 30)), failing:
            encoder, record = la.select_encoder(self.config(), torch.device("cuda", 0))
            self.assertIsNone(encoder)
            self.assertIn("NVENC unavailable", record["reason"])
            with self.assertRaises(UnsupportedError):
                la.select_encoder(self.config("nvenc"), torch.device("cuda", 0))


class PressureDeferralTests(_Pipelines, unittest.TestCase):
    """Freeze F dev22j2 (coordinator 07:40): a ``memory_critical(reason)`` section whose entry check finds the headroom
    below the target defers archival GPU work until it exits (as ``defer_archival_during_decode`` would, for that
    section only; the entry wait is capped); no pressure, no deferral; the pressure event carries the caller's
    reason. CPU pipeline with a stand-in fitter, CPU tensors standing in for device copies."""

    def setup_archive(self, *, free: int, pipeline=None, fitter=object()):
        nbytes = _matrix(0).numel() * 2
        device = _FakeDevice(free=free)
        tier = RawChunkTier(device_memory=device, gpu_budget_bytes=1 << 30, host_budget_bytes=1 << 30,
                            gpu_reserve_bytes=2 * nbytes)
        self.addCleanup(tier.close)
        entry = _adopt(tier, "c0", _matrix(0))
        with tier._cond:
            tier.fitted(entry)
            tier.published(entry, "ref-0")
        tier.flush(30)
        archive = _fake_archive(tier, defer=False)
        if pipeline is None:
            _, pipeline = self.pipeline(name=f"defer-{uuid.uuid4().hex[:8]}")
        archive.pipeline, archive.fitter = pipeline, fitter
        return archive, tier, device, entry

    def test_entry_pressure_defers_archival_for_the_section(self):
        archive, tier, device, entry = self.setup_archive(free=0)
        pipeline = archive.pipeline
        with la.LineageSessionArchive.memory_critical(archive, "restore") as report:
            self.assertTrue(pipeline.stats()["deferred"])  # no GPU fitting starts inside the section
            self.assertFalse(entry.on_device)  # the GPU raw copy left before the section body
        self.assertFalse(pipeline.stats()["deferred"])
        self.assertEqual(pipeline.stats()["deferrals"], 1)
        self.assertTrue(report["deferred_archival"])
        self.assertGreaterEqual(report["archival_deferred_seconds"], 0.0)
        self.assertGreaterEqual(report["deferral_wait_seconds"], 0.0)
        event = report["raw_pressure"]
        self.assertEqual((event["where"], event["deferred_archival"]), ("restore", True))  # the caller's label
        stats = tier.stats()["pressure"]
        self.assertEqual(stats["by_where"].get("restore"), 1)
        self.assertNotIn("decode", stats["by_where"])
        self.assertEqual(stats["deferred_sections"], 1)
        self.assertTrue(stats["events"][-1]["deferred_archival"])

    def test_no_entry_pressure_no_deferral(self):
        archive, tier, device, entry = self.setup_archive(free=1 << 30)
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            self.assertFalse(archive.pipeline.stats()["deferred"])
        self.assertEqual(archive.pipeline.stats()["deferrals"], 0)
        self.assertNotIn("deferred_archival", report)
        self.assertFalse(report["raw_pressure"]["pressure"])
        self.assertEqual(report["raw_pressure"]["where"], "vae_decode")
        self.assertTrue(entry.on_device)

    def test_a_capped_entry_wait_leaves_the_section_undeferred(self):
        calls = []

        class Stuck(_FakePipeline):
            @staticmethod
            def defer_gpu_work(**kwargs):
                calls.append(kwargs)
                raise CapacityError("the archive fitter did not leave its GPU work within the deferral timeout")

        archive, tier, device, entry = self.setup_archive(free=0, pipeline=Stuck())
        ran = []
        with la.LineageSessionArchive.memory_critical(archive, "restore") as report:
            ran.append(True)
        self.assertEqual(ran, [True])  # the section still runs
        self.assertEqual(calls, [{"wait": True, "release_cached_memory": True,
                                  "timeout": la.PRESSURE_DEFER_TIMEOUT}])
        self.assertFalse(report["deferred_archival"])
        self.assertIn("CapacityError", report["deferral_error"])
        self.assertFalse(tier.stats()["pressure"]["events"][-1]["deferred_archival"])

    def test_cpu_fitting_and_the_configured_deferral_keep_their_paths(self):
        archive, tier, device, entry = self.setup_archive(free=0, fitter=None)
        with la.LineageSessionArchive.memory_critical(archive, "restore") as report:
            pass
        self.assertNotIn("deferred_archival", report)  # nothing to defer without a GPU fitter
        archive, tier, device, entry = self.setup_archive(free=0)
        archive.config.defer_archival_during_decode = True
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            self.assertTrue(archive.pipeline.stats()["deferred"])
        self.assertTrue(report["deferred"])
        self.assertNotIn("deferred_archival", report)  # the configured path, as before

    def test_the_policy_never_defers_a_restore_section(self):
        """F4 (coordinator 15:35, 5b 4427631): ``defer_archival_during_decode`` applies to decode sections only. A
        restore section without pressure demotes, defers and releases nothing; with pressure it takes the pressure
        path, as without the policy; a decode section keeps the policy path."""
        archive, tier, device, entry = self.setup_archive(free=1 << 30)
        archive.config.defer_archival_during_decode = True
        released = []
        real = archive.pipeline.defer_gpu_work

        def defer_gpu_work(**kwargs):
            released.append(kwargs)
            return real(**kwargs)

        archive.pipeline.defer_gpu_work = defer_gpu_work
        with la.LineageSessionArchive.memory_critical(archive, "restore") as report:
            self.assertFalse(archive.pipeline.stats()["deferred"])
            self.assertTrue(entry.on_device)  # nothing evacuated
        self.assertEqual(released, [])  # no deferral, no cache release
        self.assertFalse(report["deferred"])
        self.assertNotIn("raw_demoted", report)
        self.assertNotIn("deferred_archival", report)
        self.assertFalse(report["raw_pressure"]["pressure"])
        self.assertEqual(tier.stats()["demotions"], 0)
        # Under pressure the restore section takes the pressure-triggered deferral, as without the policy.
        archive, tier, device, entry = self.setup_archive(free=0)
        archive.config.defer_archival_during_decode = True
        with la.LineageSessionArchive.memory_critical(archive, "restore") as report:
            self.assertTrue(archive.pipeline.stats()["deferred"])
        self.assertTrue(report["deferred_archival"])  # the pressure path's record...
        self.assertNotIn("raw_demoted", report)  # ...not the policy's evacuation
        self.assertEqual(report["raw_pressure"]["where"], "restore")
        # The decode seam keeps the policy path without pressure.
        archive, tier, device, entry = self.setup_archive(free=1 << 30)
        archive.config.defer_archival_during_decode = True
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            self.assertTrue(archive.pipeline.stats()["deferred"])
            self.assertFalse(entry.on_device)  # evacuated by the policy
        self.assertTrue(report["deferred"])
        self.assertEqual([la.policy_defers(name) for name in ("vae_decode", "decode", "latent_decode", "restore",
                                                              "request_start", None)],
                         [True, True, True, False, False, False])


#: dev22j8v4: the strict-headroom probe, run in a subprocess with expandable segments (read at CUDA init).
_STRICT_PROBE = r"""
import json, os, statistics, time
import torch
from reelcache import raw_tier as rt
MiB = 1 << 20
dev = torch.device("cuda", 0)
torch.cuda.init()
out = {"alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "history": bool(torch._C._cuda_isHistoryEnabled())}
a, b, c = (torch.cuda.Stream(dev) for _ in range(3))
with torch.cuda.stream(a):  # live blocks with freed ones between them (a decode's caches)
    tensors = [torch.empty(int((3.3 if i % 2 else 1.7) * MiB) + 4096 * (i % 7), dtype=torch.uint8, device=dev)
               for i in range(600)]
    keep = tensors[::2]
    del tensors
with torch.cuda.stream(b):  # a working set freed on another stream: wholly free, releasable
    work = [torch.empty(256 * MiB, dtype=torch.uint8, device=dev) for _ in range(8)]
    del work
torch.cuda.synchronize()
memory = rt.cuda_device_memory(dev)
strict = rt.strict_device_headroom(memory, rt.cuda_device_segments(dev))
formula = rt.device_headroom(memory)
out.update(formula=formula, strict=strict, inactive_split=memory["inactive_split"])
for name, promise in (("strict_trial", strict["headroom"]), ("formula_trial", formula)):
    trial = {"bytes": promise - 64 * MiB}
    try:
        with torch.cuda.stream(c):
            block = torch.empty(trial["bytes"], dtype=torch.uint8, device=dev)
            del block
        trial["fits"] = True
    except torch.OutOfMemoryError as exc:
        trial["fits"], trial["error"] = False, str(exc).split("If reserved")[0][-200:]
    torch.cuda.synchronize()
    out[name] = trial
timings = {}
params = []
for count in (2000, 8000):  # a model's parameters and buffers: small and medium live blocks
    while len(params) < count:
        i = len(params)
        params.append(torch.empty(4096 * (1 + i % 50) if i % 4 else int(MiB * (1 + i % 3)), dtype=torch.uint8,
                                  device=dev))
    blocks = sum(len(segment.get("blocks") or ()) for segment in rt.cuda_device_segments(dev))
    seconds = []
    for _ in range(25):
        started = time.perf_counter()
        rt.cuda_device_segments(dev)
        seconds.append(time.perf_counter() - started)
    timings[str(count)] = {"blocks": blocks, "p50": statistics.median(seconds), "max": max(seconds)}
out["snapshot"] = timings
print("STRICT_PROBE " + json.dumps(out))
"""


if os.environ.get("REELCACHE_STEP7_GPU_TESTS") == "1":
    class CUDARawTierTests(_Pipelines, unittest.TestCase):
        """CUDA (inside GPU jobs only; collected only with REELCACHE_STEP7_GPU_TESTS=1): pinned spills and uploads are
        bitwise, a CUDA pipeline's nodes are identical with every source spilled and re-staged, and the encoder
        choice (NVENC on RTX PRO 6000, libx265 elsewhere) with NVENC's repeated-encode byte identity (R8)."""

        def setUp(self):
            if not torch.cuda.is_available():
                raise AssertionError("REELCACHE_STEP7_GPU_TESTS=1 requires a CUDA device")

        def test_cuda_spill_and_upload_are_bitwise_through_pinned_memory(self):
            source = torch.randn(4680, 1536, device="cuda").to(torch.bfloat16)
            nbytes = source.numel() * source.element_size()
            tier = RawChunkTier(gpu_budget_bytes=0, host_budget_bytes=4 * nbytes)
            self.addCleanup(tier.close)
            event = torch.cuda.Event()
            event.record()
            with tier._cond:
                reservation = tier.reserve(nbytes)
                entry = tier.adopt(reservation, branch="s", node_id="c0", groups=[("c0", 0, {KEY: source})],
                                   ready_event=event)
            tier.flush(60)
            self.assertEqual((entry.on_device, entry.host_state), (False, "complete"))
            self.assertTrue(entry.host_buffer.pinned)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                staged = tier.stage_for_plan(entry, stream)
            stream.synchronize()
            self.assertEqual(staged[0][KEY].device, source.device)
            self.assertTrue(torch.equal(staged[0][KEY], source))
            stats = tier.stats()
            print(f"raw tier CUDA: spill {stats['spill_bytes'] / max(stats['spill_seconds'], 1e-9) / 1e9:.1f} GB/s, "
                  f"upload {stats['upload_bytes'] / max(stats['upload_seconds'], 1e-9) / 1e9:.1f} GB/s, "
                  f"pin {stats['host_buffers']['pin_seconds']:.3f} s for {nbytes / 1e6:.0f} MB")

        def test_cuda_dropped_copies_keep_their_pinned_blocks_for_reuse(self):
            """D2: dropping a pinned host copy never unregisters it (the driver lock); the next copy reuses it."""
            closes = []
            original = rt._HostBlock.close

            def close(block):
                closes.append(block.nbytes)
                original(block)

            with mock.patch.object(rt._HostBlock, "close", close):
                tier = RawChunkTier(gpu_budget_bytes=0, host_budget_bytes=1 << 30)
                self.addCleanup(tier.close)
                for round_ in range(2):
                    source = torch.randn(4680, 1536, device="cuda").to(torch.bfloat16)
                    event = torch.cuda.Event()
                    event.record()
                    with tier._cond:
                        entry = tier.adopt(tier.reserve(source.numel() * 2), branch="s", node_id=f"c{round_}",
                                           groups=[(f"c{round_}", 0, {KEY: source})], ready_event=event)
                    tier.flush(60)
                    self.assertTrue(entry.host_pinned)
                    with tier._cond:
                        tier.fitted(entry)
                        tier.published(entry, f"ref-{round_}")
                    lease = tier.acquire(f"ref-{round_}")
                    self.assertTrue(torch.equal(lease.matrix(0, KEY).cuda(), source))
                    lease.release()
                    tier.drop([entry])
                    tier.flush(60)
                pool = tier.stats()["host_buffers"]
                self.assertEqual(closes, [])
                self.assertEqual((pool["allocations"], pool["pinned"], pool["reuses"]), (1, 1, 1))
                event = tier.shrink_buffers()
                self.assertEqual((event["blocks"], event["pinned_blocks"]), (1, 1))
                print(f"raw tier CUDA pool: pin {pool['pin_seconds']:.3f} s once, unpin {event['seconds']:.3f} s "
                      f"at the explicit shrink ({event['bytes'] / 1e6:.0f} MB)")

        def run_cuda_chain(self, name, tier):
            from reelcache.gpu_fitting import GPUFitter
            runtime, pipeline = self.pipeline(
                tier=tier, name=name, device="cuda", encode_workers=2,
                fitter=GPUFitter(gpu_id=0, gpu_uuid="GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix(
                    "GPU-"), method="gram_batched", eigh_solver="auto"))
            refs = []
            for index in range(3):
                matrices, contexts = s6.chunk(index)
                matrices = {key: value.cuda() for key, value in matrices.items()}
                event = torch.cuda.Event()
                event.record()
                pipeline.submit("main", f"chunk-{index:06d}", matrices, contexts, ready_event=event)
                refs.append(pipeline.drain("main", timeout=300))
            return [s6.node_content(runtime, ref) for ref in refs], refs, pipeline.stats()

        def test_cuda_nodes_are_identical_with_every_source_spilled_and_restaged(self):
            expected, expected_refs, _ = self.run_cuda_chain("cuda-c3", None)
            contents, refs, stats = self.run_cuda_chain("cuda-spilled", RawChunkTier(gpu_budget_bytes=0,
                                                                                     host_budget_bytes=1 << 30))
            self.assertEqual(refs, expected_refs)
            self.assertEqual(contents, expected)
            self.assertGreaterEqual(stats["raw_tier"]["spills"], 1)

        def test_cuda_warm_up_measures_the_fitter_peak_and_a_pause_holds_a_node_at_a_step_boundary(self):
            """dev22j7 on CUDA: the warm-up learns the fitter's peak working set (dev22j8: live bytes); a node paused
            by ``pause_gpu_work`` stops at a step boundary, its device allocation does not grow while paused, and it
            completes with the same content once the pause is lifted. dev22j8v3 (F5.2 gates 4444374/4444375): the
            bound's basis may be the online measurement (the larger peak) and is never below the warm-up's; the plan
            hook is released before the pipeline closes, also when an assertion fails."""
            from reelcache import archive_pipeline as ap
            from reelcache.gpu_fitting import GPUFitter
            release, planning = threading.Event(), threading.Event()

            def hooks(stage, node_id):
                if stage == "plan" and node_id == "chunk-000001":
                    planning.set()
                    self.assertTrue(release.wait(120))

            runtime, pipeline = self.pipeline(
                name="cuda-pause", device="cuda", encode_workers=2, hooks=hooks,
                fitter=GPUFitter(gpu_id=0, gpu_uuid="GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix(
                    "GPU-"), method="gram_batched", eigh_solver="auto"))
            self.addCleanup(release.set)  # runs before the pipeline's close (cleanups run last-in first-out)
            matrices, contexts = s6.chunk(0)
            pipeline.prewarm(contexts, encode=False)
            peak = pipeline.stats()["fitter_pause"]
            print(f"fitter warm-up peak {peak['peak_bytes'] / 2**20:.1f} MiB for {peak['peak_source_bytes'] / 2**20:.1f}"
                  f" MiB of sources")
            self.assertGreater(peak["peak_bytes"], 0)
            self.assertGreater(peak["peak_source_bytes"], 0)
            for index in range(2):
                matrices, contexts = s6.chunk(index)
                matrices = {key: value.cuda() for key, value in matrices.items()}
                event = torch.cuda.Event()
                event.record()
                pipeline.submit("main", f"chunk-{index:06d}", matrices, contexts, ready_event=event)
            self.assertTrue(planning.wait(300))
            bound = pipeline.inflight_growth_bound()
            self.assertTrue(bound["inflight"])
            self.assertIn(bound["basis"], ("warmup", "online"))  # a learned peak, never the working-set fallback
            self.assertGreaterEqual(bound["bytes"], peak["warmup_peak_bytes"])
            _release_when(release, lambda: pipeline.stats()["fitter_pause"]["open"] >= 1)
            with mock.patch.object(ap, "PAUSE_ACK_TIMEOUT", 60.0), pipeline.pause_gpu_work() as info:
                self.assertEqual((info["acked"], info["inflight"]), (True, True))
                torch.cuda.synchronize()
                held = torch.cuda.memory_allocated()
                time.sleep(0.5)
                torch.cuda.synchronize()
                self.assertLessEqual(torch.cuda.memory_allocated(), held)  # paused: the fitter allocates nothing
                self.assertTrue(pipeline.stats()["fitter_pause"]["paused"])
            ref = pipeline.drain("main", timeout=300)
            expected, expected_refs, _ = self.run_cuda_chain("cuda-unpaused", None)
            self.assertEqual(ref, expected_refs[1])

        def test_cuda_the_warm_up_peak_counts_what_the_fitter_stream_already_holds(self):
            """dev22j8v3 (the F5.2 gpuchecks 4444374 / 4444375, both GPUs and environments): ``torch.cuda.Stream()``
            hands out pooled streams, so a pipeline's fitter stream may already carry an earlier user's live blocks
            (there: 8 / 32 MiB, a cuBLAS workspace kept per handle and stream). The online samples and the guard's
            ``held`` count them; the warm-up's peak now starts from them too (absolute), so the online sample of a
            smaller node no longer exceeds it."""
            from reelcache import archive_pipeline as ap
            from reelcache.gpu_fitting import GPUFitter
            runtime, pipeline = self.pipeline(
                name="cuda-baseline", device="cuda", encode_workers=2,
                fitter=GPUFitter(gpu_id=0, gpu_uuid="GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix(
                    "GPU-"), method="gram_batched", eigh_solver="auto"))
            deadline = time.monotonic() + 60
            while pipeline._fitter_stream is None and time.monotonic() < deadline:
                time.sleep(0.01)
            kept = 16 << 20
            with torch.cuda.stream(pipeline._fitter_stream):  # an earlier user's live block on the same stream
                earlier = torch.empty(kept, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            matrices, contexts = s6.chunk(0)
            report = pipeline.prewarm(contexts, encode=False)
            warm = pipeline.stats()["fitter_pause"]
            self.assertGreaterEqual(warm["warmup_baseline_bytes"], kept)
            self.assertEqual(warm["warmup_peak_bytes"], warm["warmup_baseline_bytes"]
                             + report["fitter_live_peak"]["bytes"] - report["source_bytes"])
            with mock.patch.object(ap, "PEAK_SAMPLE_INTERVAL", 0.0):
                matrices = {key: value.cuda() for key, value in matrices.items()}
                event = torch.cuda.Event()
                event.record()
                pipeline.submit("main", "chunk-000000", matrices, contexts, ready_event=event)
                pipeline.drain("main", timeout=300)
            pause = pipeline.stats()["fitter_pause"]
            print(f"fitter peaks with {kept >> 20} MiB already on the stream: warm-up {warm['warmup_peak_bytes'] / 2**20:.2f}"
                  f" MiB (baseline {warm['warmup_baseline_bytes'] / 2**20:.2f}), online {pause['online_peak_bytes'] / 2**20:.2f} MiB")
            self.assertGreaterEqual(pause["online_peak_bytes"], kept)  # the samples count it as well
            self.assertEqual(pause["peak_bytes"], max(warm["warmup_peak_bytes"], pause["online_peak_bytes"]))
            self.assertGreaterEqual(pipeline.fitter_memory()["held"], kept)
            del earlier

        def test_cuda_the_strict_headroom_is_allocatable_where_the_formula_is_not(self):
            """dev22j8v4 (F5.2's RAVEN decode OOM; probes 4445094 / 4445098): under expandable segments the allocator
            hides free fragments inside used segments from ``inactive_split``; the formula counts them, an allocation
            of its promise runs out of memory. The strict headroom (driver free + releasable: wholly free segments,
            whole pages) is allocatable on another stream; the snapshot is taken without history recording."""
            import json
            import subprocess
            import sys
            env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            done = subprocess.run([sys.executable, "-c", _STRICT_PROBE], env=env, capture_output=True, text=True,
                                  timeout=600)
            lines = [line for line in done.stdout.splitlines() if line.startswith("STRICT_PROBE ")]
            self.assertTrue(lines, done.stdout[-2000:] + done.stderr[-2000:])
            out = json.loads(lines[-1][len("STRICT_PROBE "):])
            strict = out["strict"]
            mib = 1 << 20
            print(f"strict headroom {strict['headroom'] / mib:.0f} MiB (free {strict['free'] / mib:.0f} + releasable "
                  f"{strict['releasable'] / mib:.0f}; unreleasable {strict['unreleasable'] / mib:.0f}) vs formula "
                  f"{out['formula'] / mib:.0f} MiB (inactive_split {out['inactive_split'] / mib:.0f}); trials: strict "
                  f"{out['strict_trial']['fits']}, formula {out['formula_trial']['fits']}; snapshot seconds "
                  + ", ".join(f"{item['blocks']} blocks p50 {item['p50']:.4f} max {item['max']:.4f}"
                              for item in out["snapshot"].values()))
            self.assertFalse(out["history"])
            self.assertEqual(out["alloc_conf"], "expandable_segments:True")
            self.assertGreaterEqual(strict["releasable"], 2000 * mib)  # the other stream's freed working set
            self.assertGreater(out["formula"] - strict["headroom"], 512 * mib)  # the fragments it counted
            self.assertTrue(out["strict_trial"]["fits"], out["strict_trial"])
            self.assertFalse(out["formula_trial"]["fits"], out["formula_trial"])

        def test_cuda_a_decode_short_of_its_floor_waits_for_the_node_rather_than_running_paused(self):
            """dev22j8v5 (F5.4 finding 1) on CUDA: next to a node, the real allocator's strict headroom lies above the
            prior-based target (dev22j8v4: a paused fit) but below the decode floor; a decode needing more than that
            target and less than the floor is no paused fit: after the pause (taken before the snapshot) and the relief
            it waits for the node."""
            import types
            from reelcache import archive_pipeline as ap
            from reelcache.gpu_fitting import GPUFitter
            release, planning = threading.Event(), threading.Event()

            def hooks(stage, node_id):
                if stage == "plan" and node_id == "chunk-000000":
                    planning.set()
                    self.assertTrue(release.wait(120))

            runtime, pipeline = self.pipeline(
                name="cuda-floor", device="cuda", encode_workers=2, hooks=hooks,
                fitter=GPUFitter(gpu_id=0, gpu_uuid="GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix(
                    "GPU-"), method="gram_batched", eigh_solver="auto"))
            self.addCleanup(release.set)  # before the pipeline's close (cleanups run last-in first-out)
            mib = 1 << 20
            tier = RawChunkTier(gpu_budget_bytes=1 << 30, host_budget_bytes=1 << 30, gpu_reserve_bytes=64 * mib)
            self.addCleanup(tier.close)
            entry = _adopt(tier, "c0", _matrix(0).cuda())
            with tier._cond:
                tier.fitted(entry)
                tier.published(entry, "ref-0")
            tier.flush(30)
            archive = _fake_archive(tier, defer=False)
            archive.config.decode_reserve_prior_bytes = 64 * mib
            for name in ("_decode_entry_guard", "_release_cached_blocks"):
                setattr(archive, name, types.MethodType(getattr(la.LineageSessionArchive, name), archive))
            archive.decode_guard_ = la._decode_guard_state(True)
            archive.pipeline, archive.device = pipeline, torch.device("cuda", torch.cuda.current_device())
            matrices, contexts = s6.chunk(0)
            matrices = {key: value.cuda() for key, value in matrices.items()}
            event = torch.cuda.Event()
            event.record()
            ticket = pipeline.submit("main", "chunk-000000", matrices, contexts, ready_event=event)
            self.assertTrue(planning.wait(300))
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            ballast = [torch.empty(max(0, torch.cuda.mem_get_info()[0] - 1024 * mib), dtype=torch.uint8, device="cuda")]

            def give_back():  # the ballast's segment back to the driver: later tests and their subprocesses need it
                ballast.clear()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            self.addCleanup(give_back)  # first of the cleanups, also when an assertion fails
            free = torch.cuda.mem_get_info()[0]
            archive.config.decode_working_set_floor_bytes = free + 2048 * mib  # beyond the strict headroom
            _release_when(release, lambda: pipeline.stats()["fitter_pause"]["open"] >= 1)
            with mock.patch.object(ap, "PAUSE_ACK_TIMEOUT", 60.0):
                with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
                    self.assertTrue(pipeline.stats()["deferred"])
                    self.assertEqual(pipeline.stats()["fitter_pause"]["open"], 0)  # waited for the node, not paused
                    decode = torch.empty(512 * mib, dtype=torch.uint8, device="cuda")  # above the old target
                    del decode
            guard = report["decode_guard"]
            print(f"decode floor on CUDA: strict headroom {guard['strict_headroom'] / mib:.0f} MiB, target before the "
                  f"floor {guard['target_before_floor'] / mib:.0f} MiB, floor {guard['floor_bytes'] / mib:.0f} MiB; "
                  f"{report['deferral_reason']}, waited {report['deferral_wait_seconds']:.3f} s, snapshots "
                  f"{guard['snapshot_count']} ({guard['snapshot_seconds'] * 1000:.2f} ms first)")
            self.assertEqual((guard["decided_by"], guard["floor_source"], guard["inflight"]), ("strict", "config", True))
            self.assertLessEqual(guard["target_before_floor"], guard["strict_headroom"])  # dev22j8v4: it fit
            self.assertLess(guard["strict_headroom"], guard["floor_bytes"])
            self.assertTrue(guard["pause"]["acked"])
            self.assertEqual((guard["short"], report["deferral_reason"]), (True, "short_after_relief"))
            self.assertNotIn("fits", guard)
            state = archive.decode_guard_
            self.assertEqual((state["pauses_first"], state["floor_raised"], state["fits_paused"]), (1, 1, 0))
            self.assertTrue(ticket.result(timeout=300).startswith("manifest:"))
            give_back()
            self.assertGreater(torch.cuda.mem_get_info()[0], free + 4096 * mib)

        def test_cuda_the_fitter_stream_is_its_own_and_no_pooled_stream_counts_in_held(self):
            """dev22j8v6 (F6 b): the fitter's stream is created with the driver API, so no pooled stream (a restore's
            ``torch.cuda.Stream()``) is it and a tensor on any of them never counts in the decode guard's ``held``.
            With ``fitter_stream="pooled"`` (F4-F5.5) the fitter's stream is one of the pool's: a restore that
            draws it inflates ``held``."""
            from reelcache.gpu_fitting import GPUFitter

            def fitter():
                return GPUFitter(gpu_id=0, gpu_uuid="GPU-" + str(torch.cuda.get_device_properties(0).uuid)
                                 .removeprefix("GPU-"), method="gram_batched", eigh_solver="auto")

            _, dedicated = self.pipeline(name="cuda-dedicated", device="cuda", fitter=fitter(), fitter_stream="dedicated")
            _, pooled = self.pipeline(name="cuda-pooled", device="cuda", fitter=fitter(), fitter_stream="pooled")
            deadline = time.monotonic() + 60
            while (dedicated._fitter_stream is None or pooled._fitter_stream is None) and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsInstance(dedicated._fitter_stream, torch.cuda.ExternalStream)
            self.assertEqual(dedicated.stats()["fitter_stream"], {"mode": "dedicated", "note": None, "dedicated": True})
            self.assertEqual(pooled.stats()["fitter_stream"], {"mode": "pooled", "note": None, "dedicated": False})
            before = {name: pipeline.fitter_memory()["held"] for name, pipeline in (("dedicated", dedicated),
                                                                                    ("pooled", pooled))}
            tensors, handles = [], set()
            for _ in range(64):  # a restore's pooled streams: the whole pool, twice
                stream = torch.cuda.Stream()
                handles.add(stream.cuda_stream)
                with torch.cuda.stream(stream):
                    tensors.append(torch.ones(1 << 20, dtype=torch.float32, device="cuda"))  # 4 MiB each
            torch.cuda.synchronize()
            self.assertNotIn(dedicated._fitter_stream.cuda_stream, handles)
            self.assertIn(pooled._fitter_stream.cuda_stream, handles)
            after = {name: pipeline.fitter_memory()["held"] for name, pipeline in (("dedicated", dedicated),
                                                                                   ("pooled", pooled))}
            print(f"fitter held with 64 pooled-stream tensors: dedicated {before['dedicated']} -> {after['dedicated']}"
                  f" B, pooled {before['pooled']} -> {after['pooled']} B")
            self.assertEqual(after["dedicated"], before["dedicated"])
            self.assertGreaterEqual(after["pooled"] - before["pooled"], 2 * (4 << 20))  # its pool slot, drawn twice
            tensors.clear()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        def test_cuda_live_peaks_and_holdings_are_measured_on_the_fitter_streams(self):
            """dev22j8 (coordinator 13:1x): the warm-up's live high-water (allocator history; the synthetic sources
            excluded), the measured nodes' sampled live high-water and ``fitter_memory`` (live bytes on the fitter's
            streams, the raw tier's copies excluded) on CUDA; reserved bytes are diagnostics."""
            from reelcache import archive_pipeline as ap
            from reelcache.gpu_fitting import GPUFitter
            tier = RawChunkTier(gpu_budget_bytes=0, host_budget_bytes=1 << 30)  # every node re-staged on the fitter
            runtime, pipeline = self.pipeline(
                name="cuda-live", device="cuda", encode_workers=2, tier=tier,
                fitter=GPUFitter(gpu_id=0, gpu_uuid="GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix(
                    "GPU-"), method="gram_batched", eigh_solver="auto"))
            matrices, contexts = s6.chunk(0)
            report = pipeline.prewarm(contexts, encode=False)
            warm = pipeline.stats()["fitter_pause"]
            self.assertEqual(report["fitter_live_peak"]["method"], "history")
            self.assertGreater(warm["warmup_peak_bytes"], 0)
            self.assertEqual(warm["warmup_peak_bytes"], report["fitter_live_peak"]["bytes"] - report["source_bytes"])
            self.assertGreaterEqual(warm["warmup_reserved_bytes"], warm["warmup_peak_bytes"])
            self.assertFalse(torch._C._cuda_isHistoryEnabled())  # recorded only over the warm-up
            with mock.patch.object(ap, "PEAK_SAMPLE_INTERVAL", 0.0):
                for index in range(ap.PEAK_MEASURE_FIRST):
                    matrices, contexts = s6.chunk(index)
                    matrices = {key: value.cuda() for key, value in matrices.items()}
                    event = torch.cuda.Event()
                    event.record()
                    pipeline.submit("main", f"chunk-{index:06d}", matrices, contexts, ready_event=event)
                    pipeline.drain("main", timeout=300)
            pause = pipeline.stats()["fitter_pause"]
            memory = pipeline.fitter_memory()
            print(f"fitter live peaks (tiny model): warm-up {pause['warmup_peak_bytes'] / 2**20:.2f} MiB "
                  f"(reserved {pause['warmup_reserved_bytes'] / 2**20:.2f}), online {pause['online_peak_bytes'] / 2**20:.2f}"
                  f" MiB over {pause['online_samples']} samples in {pause['online_sample_seconds']:.3f} s; now {memory}")
            self.assertEqual(pause["online_measurements"], ap.PEAK_MEASURE_FIRST)
            self.assertGreater(pause["online_peak_bytes"], 0)
            self.assertGreaterEqual(memory["reserved"], memory["allocated"])
            self.assertEqual(memory["held"], memory["allocated"] - memory["tier"])
            self.assertGreaterEqual(memory["tier"], 0)

        def test_cuda_a_failed_encode_releases_its_device_canvas(self):
            """dev22j8 (item 5; R-D2b on F5.1): a node that fails before its canvas was waited for is retired with
            its device canvas released (it lived on through the failed job's ranges)."""
            import gc
            from reelcache import lineage_commit as lc
            from reelcache.gpu_fitting import GPUFitter
            pictures = []
            assemble = lc.assemble_canvas

            def recording(*args, **kwargs):
                canvas = assemble(*args, **kwargs)
                if canvas.device_pictures is not None and canvas.device_pictures.is_cuda:
                    pictures.append(weakref.ref(canvas.device_pictures))
                return canvas

            def fail(stage, node_id):
                if stage == "encode" and node_id == "chunk-000001":
                    raise RuntimeError("injected encode failure")

            with mock.patch.object(lc, "assemble_canvas", recording):
                runtime, pipeline = self.pipeline(
                    name="cuda-canvas", device="cuda", encode_workers=2, hooks=fail,
                    fitter=GPUFitter(gpu_id=0, gpu_uuid="GPU-" + str(torch.cuda.get_device_properties(0).uuid)
                                     .removeprefix("GPU-"), method="gram_batched", eigh_solver="auto"))
                tickets = []
                for index in range(2):
                    matrices, contexts = s6.chunk(index)
                    matrices = {key: value.cuda() for key, value in matrices.items()}
                    event = torch.cuda.Event()
                    event.record()
                    tickets.append(pipeline.submit("main", f"chunk-{index:06d}", matrices, contexts,
                                                   ready_event=event))
                self.assertTrue(tickets[0].result(timeout=300).startswith("manifest:"))
                with self.assertRaises(ArchiveFailure):
                    tickets[1].result(timeout=300)
                with self.assertRaises(ArchiveFailure):
                    pipeline.drain("main", timeout=300)
            job = pipeline._lineages["main"].last_job
            self.assertEqual(job.node_id, "chunk-000001")
            gc.collect()
            self.assertEqual(len(pictures), 2)
            self.assertEqual([ref() for ref in pictures], [None, None])  # F5.1 kept chunk 1's on the GPU
            pipeline.reset("main", timeout=300)

        def test_encoder_auto_selection_and_nvenc_repeat_identity(self):
            config = SimpleNamespace(encoder="auto", representation="svd_hevc", hevc_qp=8, hevc_pix_fmt="yuv444p")
            device = torch.device("cuda", 0)
            encoder, record = la.select_encoder(config, device)
            print(f"encoder auto on {torch.cuda.get_device_name(0)}: {record}")
            if encoder is None:
                self.assertEqual(record["selected"], "libx265")
                self.assertIn("NVENC unavailable", record["reason"])
                return
            self.assertEqual(record["selected"], "nvenc")
            import numpy as np
            pictures = np.random.default_rng(3).integers(0, 256, (3, 3, 256, 512), dtype=np.uint8)
            first = encoder.encode_pictures(pictures, "yuv444p")[0]
            again = encoder.encode_pictures(pictures, "yuv444p")[0]
            other, _ = la.select_encoder(config, device)
            fresh = other.encode_pictures(pictures, "yuv444p")[0]
            self.assertEqual(first, again)  # same session, repeated
            self.assertEqual(first, fresh)  # a new encoder (new sessions): byte-identical



class StoredFailureTests(_Pipelines, unittest.TestCase):
    """R-D2b (F2 census 4423272): a failed commit's kept error pinned its staged sources (60 x [4680, 1536] BF16 on
    CausVid) through the traceback of ``ArchivePipeline._fit``. Errors are kept frame-free
    (``lineage_commit.frame_free``: lineage, tickets, jobs, warm-ups), raised to callers as copies
    (``raisable``), and a retired job drops its sources."""

    def test_frame_free_releases_every_chained_traceback_and_raisable_copies(self):
        from reelcache.lineage_commit import frame_free, raisable

        refs = []

        def failing():
            tensor = torch.ones(16)
            refs.append(weakref.ref(tensor))
            try:
                raise ValueError("inner failure")
            except ValueError as inner:
                raise StateError("outer failure") from inner

        try:
            failing()
        except StateError as exc:
            error = exc
        failure = ArchiveFailure("wrapped", branch_id="b", node_id="n", cause=error)
        self.assertIs(frame_free(failure), failure)
        import gc
        gc.collect()
        self.assertIsNone(refs[0]())  # the frame's local went with the traceback
        self.assertEqual([item.__traceback__ for item in (failure, error, error.__cause__)], [None] * 3)
        self.assertTrue(any("traceback when stored" in note and "failing" in note for note in error.__notes__))
        clone = raisable(failure)
        self.assertIsNot(clone, failure)
        self.assertIsInstance(clone, ArchiveFailure)
        self.assertEqual((str(clone), clone.branch_id, clone.node_id), ("wrapped", "b", "n"))
        self.assertIs(clone.cause, error)
        try:
            raise clone
        except ArchiveFailure:
            pass
        self.assertIsNone(failure.__traceback__)  # raising the copy never touches the kept error

    def test_a_failed_plan_keeps_no_source_alive(self):
        def fail(stage, node_id):
            if stage == "plan" and node_id == "chunk-000001":
                raise RuntimeError("injected planning failure")

        tier = RawChunkTier(host_budget_bytes=1 << 30)
        _, pipeline = self.pipeline(tier=tier, hooks=fail)
        tickets, alive = [], []
        for index in range(2):
            matrices, contexts = s6.chunk(index)
            if index == 1:
                alive = [weakref.ref(tensor) for tensor in matrices.values()]
            tickets.append(pipeline.submit("main", f"chunk-{index:06d}", matrices, contexts))
            del matrices
        with self.assertRaises(ArchiveFailure) as caught:
            tickets[1].result(timeout=120)
        self.assertIn("injected planning failure", str(caught.exception))
        with self.assertRaises(ArchiveFailure):
            pipeline.drain("main", timeout=120)
        tier.flush(30)
        import gc
        gc.collect()
        self.assertEqual([ref() is None for ref in alive], [True] * len(alive))  # pinned on dev22j3
        stored = tickets[1].error
        self.assertIsNot(caught.exception, stored)
        self.assertEqual([stored.__traceback__, stored.cause.__traceback__], [None, None])
        self.assertTrue(any("_fit" in note for note in stored.cause.__notes__))
        lineage = pipeline._lineages["main"]
        self.assertIsNone(lineage.error.__traceback__)
        self.assertTrue(all(not item.sources for item in lineage.last_job.ranges))  # retired: sources dropped
        self.assertEqual(tier.stats()["drops_failed"], 1)
        pipeline.reset("main", timeout=120)

    def test_a_failed_warm_up_keeps_no_synthetic_source_alive(self):
        from reelcache import archive_pipeline as ap

        refs = []

        def failing(*args, **kwargs):
            sources = {"K": torch.ones(64, 32)}
            refs.append(weakref.ref(sources["K"]))
            raise RuntimeError("injected warm-up failure")

        _, pipeline = self.pipeline()
        matrices, contexts = s6.chunk(0)
        raised = None
        with mock.patch.object(ap, "warm_commit_path", failing):
            try:  # kept as a caller would keep it (assertRaises clears the frames itself and would hide the pin)
                pipeline.prewarm(contexts, encode=False, timeout=60)
            except RuntimeError as exc:
                raised = exc
        self.assertIn("injected warm-up failure", str(raised))
        import gc
        gc.collect()
        self.assertIsNone(refs[0]())  # pinned on dev22j3 by the raised error's fitter-thread frames

    def test_drop_sources_waits_for_the_canvas_copy_then_drops_canvas_and_encoded(self):
        """dev22j8 (item 5; Step 11's census 4432533, R-D2b +232 MiB on F5.1): a cancelled ``CommitJob`` kept its
        116 MiB encode canvas on the GPU (``NodeCanvas.device_pictures`` <- ``_Range.canvas`` <- ``CommitJob.ranges``).
        ``drop_sources`` now waits for the canvas's D2H event, then drops the canvas and the encoded pack -- also
        when the wait fails."""
        import gc
        from reelcache.lineage_commit import CommitJob, _Range
        from reelcache.node_packets import NodeCanvas

        for broken in (False, True):
            with self.subTest(broken_wait=broken):
                waits = []

                class Event:  # the D2H copy's event
                    def synchronize(self):
                        waits.append(True)
                        if broken:
                            raise RuntimeError("CUDA error: an illegal memory access was encountered")

                pictures = torch.zeros(4, 3, 8, 8, dtype=torch.uint8)
                canvas = NodeCanvas("n", {}, {"pix_fmt": "yuv444p"}, {"pictures": 4, "width": 8, "height": 8}, [],
                                    pictures, event=Event())
                alive = (weakref.ref(pictures), weakref.ref(canvas))
                item = _Range("c0", {}, {KEY: torch.ones(2, 2)}, canvas=canvas, encoded=object())
                del pictures, canvas
                CommitJob.drop_sources(SimpleNamespace(ranges=[item]))
                gc.collect()
                self.assertEqual(waits, [True])
                self.assertEqual((item.sources, item.canvas, item.encoded), ({}, None, None))
                self.assertEqual([ref() for ref in alive], [None, None])

    def test_a_failed_encode_drops_the_job_s_canvases(self):
        """dev22j8 (item 5): a node that fails at its encode (finalized: its canvas exists, never waited for) is
        retired with its canvases dropped, although its job lives on as the lineage's last job."""
        import gc
        from reelcache import lineage_commit as lc

        canvases = []
        assemble = lc.assemble_canvas

        def recording(*args, **kwargs):
            canvas = assemble(*args, **kwargs)
            canvases.append(weakref.ref(canvas))
            return canvas

        def fail(stage, node_id):
            if stage == "encode" and node_id == "chunk-000001":
                raise RuntimeError("injected encode failure")

        with mock.patch.object(lc, "assemble_canvas", recording):
            _, pipeline = self.pipeline(hooks=fail, name=f"canvas-{uuid.uuid4().hex[:8]}")
            tickets = []
            for index in range(2):
                matrices, contexts = s6.chunk(index)
                tickets.append(pipeline.submit("main", f"chunk-{index:06d}", matrices, contexts))
            self.assertTrue(tickets[0].result(timeout=120).startswith("manifest:"))
            with self.assertRaises(ArchiveFailure) as caught:
                tickets[1].result(timeout=120)
            self.assertIn("injected encode failure", str(caught.exception))
            with self.assertRaises(ArchiveFailure):
                pipeline.drain("main", timeout=120)
        job = pipeline._lineages["main"].last_job
        self.assertEqual(job.node_id, "chunk-000001")  # the failed job lives on
        self.assertEqual([(item.canvas, item.encoded) for item in job.ranges], [(None, None)] * len(job.ranges))
        gc.collect()
        self.assertEqual(len(canvases), 2)
        self.assertEqual([ref() for ref in canvases], [None, None])  # chunk 0's after its store, chunk 1's retired
        pipeline.reset("main", timeout=120)


class FitterRecheckTests(_Pipelines, unittest.TestCase):
    """dev22j4 (5b's F2 never-block OOM, RAVEN 6->12 on A100-40GB): GPU work the fitter starts inside a watched
    memory-critical section is re-checked against the reserve plus its own bytes; short, it waits for the section
    (capped; lifted while a caller waits for archival). Recorded as ``deferred_archival`` events with ``reason:
    "fitter_recheck"``. CPU pipeline, CPU tensors, a fake device-memory provider."""

    def planned_pipeline(self, **kwargs):
        planned = threading.Event()

        def hooks(stage, node_id):
            if stage == "plan":
                planned.set()

        _, pipeline = self.pipeline(hooks=hooks, name=f"recheck-{uuid.uuid4().hex[:8]}", **kwargs)
        return pipeline, planned

    @staticmethod
    def chunk_bytes(matrices):
        return sum(tensor.numel() * tensor.element_size() for tensor in matrices.values())

    @staticmethod
    def wait_for_deferral(pipeline, count=1, timeout=60.0):
        """Until the fitter has picked the work up and deferred it: its recorded wait starts then, not at submit."""
        deadline = time.monotonic() + timeout
        while pipeline.stats()["fitter_recheck"]["deferrals"] < count:
            if time.monotonic() > deadline:
                raise AssertionError("the fitter never deferred the work")
            time.sleep(0.005)

    def test_short_work_waits_for_the_section_then_runs(self):
        pipeline, planned = self.planned_pipeline()
        needs = []

        def recheck(need):
            needs.append(need)
            return {"short": True, "headroom": 0, "reserve": 100, "target": 100 + need}

        matrices, contexts = s6.chunk(0)
        nbytes = self.chunk_bytes(matrices)
        with pipeline.watch_section("vae_decode", recheck) as events:
            opened = time.perf_counter()
            ticket = pipeline.submit("main", "chunk-000000", matrices, contexts)
            self.wait_for_deferral(pipeline)  # the recorded wait starts at the fitter's pickup, not at submit
            self.assertFalse(planned.wait(0.5))  # no GPU work starts inside the section
            last_inside = time.perf_counter()
        closed = time.perf_counter()
        self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))  # it ran once the section closed
        self.assertEqual(needs[0], nbytes)  # no measured working set, no plan_memory_bytes: the node's source bytes
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual((event["event"], event["deferred_archival"], event["reason"], event["section"],
                          event["node_id"], event["need_bytes"]),
                         ("deferred_archival", True, "fitter_recheck", "vae_decode", "chunk-000000", nbytes))
        # The wait began inside the section, before the 0.5 s hold, and ended when the section closed.
        self.assertTrue(opened <= event["_started"] <= last_inside - 0.5)
        self.assertTrue(last_inside <= event["_started"] + event["waited_seconds"] <= closed)
        self.assertGreaterEqual(event["waited_seconds"], 0.5)
        stats = pipeline.stats()["fitter_recheck"]
        self.assertEqual((stats["deferrals"], stats["timeouts"], stats["bypassed"], stats["watched_sections"]),
                         (1, 0, 0, 0))
        self.assertEqual(stats["events"][-1]["outcome"], "sections_closed")

    def test_work_that_fits_runs_inside_the_section(self):
        pipeline, planned = self.planned_pipeline()
        matrices, contexts = s6.chunk(0)
        with pipeline.watch_section("vae_decode", lambda need: {"short": False}) as events:
            ticket = pipeline.submit("main", "chunk-000000", matrices, contexts)
            self.assertTrue(planned.wait(60))
        ticket.result(timeout=120)
        self.assertEqual(events, [])
        self.assertEqual(pipeline.stats()["fitter_recheck"]["deferrals"], 0)

    def test_the_wait_is_capped_and_a_waiting_caller_lifts_it(self):
        from reelcache import archive_pipeline as ap

        pipeline, planned = self.planned_pipeline()
        short = lambda need: {"short": True}  # noqa: E731 - test stub
        matrices, contexts = s6.chunk(0)
        with mock.patch.object(ap, "RECHECK_DEFER_TIMEOUT", 0.3):
            with pipeline.watch_section("vae_decode", short):
                pipeline.submit("main", "chunk-000000", matrices, contexts)
                self.assertTrue(planned.wait(60))  # the cap: archival never deadlocks behind a section
                pipeline.drain("main", timeout=120)
        self.assertEqual(pipeline.stats()["fitter_recheck"]["timeouts"], 1)
        pipeline2, planned2 = self.planned_pipeline()
        matrices, contexts = s6.chunk(0)
        with pipeline2.watch_section("restore", short):
            pipeline2.submit("main", "chunk-000000", matrices, contexts)
            started = time.perf_counter()
            pipeline2.drain("main", timeout=120)  # a caller waits for archival: no hold (no 30 s cap reached)
            self.assertLess(time.perf_counter() - started, 20.0)
        stats = pipeline2.stats()["fitter_recheck"]
        self.assertEqual(stats["timeouts"], 0)
        self.assertGreaterEqual(stats["checks"], 0)

    def test_memory_critical_rechecks_against_the_reserve_plus_the_work(self):
        pipeline, planned = self.planned_pipeline()
        # Ample room at setup: the background admission check keeps the published GPU raw copy.
        archive, tier, device, entry = PressureDeferralTests.setup_archive(self, free=1 << 30, pipeline=pipeline)
        matrices, contexts = s6.chunk(0)
        need = self.chunk_bytes(matrices)
        reserve = tier.reserve_bytes()
        device.memory["free"] = reserve + need // 2  # above the reserve (no entry pressure), short of reserve + work
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            ticket = pipeline.submit("main", "chunk-000000", matrices, contexts)
            self.wait_for_deferral(pipeline)  # the recorded wait starts at the fitter's pickup, not at submit
            self.assertFalse(planned.wait(0.5))
        ticket.result(timeout=120)
        self.assertFalse(report["raw_pressure"]["pressure"])
        self.assertTrue(entry.on_device)  # nothing demoted: the reserve itself was covered
        self.assertTrue(report["deferred_archival"])
        self.assertEqual(report["deferral_reason"], "fitter_recheck")
        (event,) = report["fitter_rechecks"]
        self.assertEqual((event["need_bytes"], event["reserve"], event["target"], event["headroom"]),
                         (need, reserve, reserve + need, reserve + need // 2))
        self.assertGreaterEqual(report["archival_deferred_seconds"], 0.5)
        stats = tier.stats()["pressure"]
        self.assertEqual((stats["fitter_recheck_deferrals"], stats["deferred_sections"]), (1, 1))
        self.assertGreaterEqual(stats["rechecks"], 1)
        recorded = stats["events"][-1]
        self.assertEqual((recorded["where"], recorded["reason"], recorded["deferred_archival"]),
                         ("vae_decode", "fitter_recheck", True))
        device.memory["free"] = reserve + 2 * need  # room for the reserve and the work: it runs inside the section
        planned.clear()
        matrices, contexts = s6.chunk(1)
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            ticket = pipeline.submit("main", "chunk-000001", matrices, contexts)
            self.assertTrue(planned.wait(60))
        ticket.result(timeout=120)
        self.assertNotIn("fitter_rechecks", report)
        self.assertNotIn("deferred_archival", report)


class InPlacePlacementTests(unittest.TestCase):
    """dev22j6 (Step 11, F5): placing a GPU raw copy that is already on the device pins it like a new placement; it was
    left evictable. CPU tensors stand in for device copies."""

    def test_placing_a_copy_already_on_the_device_pins_it(self):
        nbytes = _matrix(0).numel() * 2
        device = _FakeDevice(free=1 << 30)
        tier = RawChunkTier(device_memory=device, gpu_budget_bytes=4 * nbytes, host_budget_bytes=1 << 30,
                            gpu_reserve_bytes=2 * nbytes)
        self.addCleanup(tier.close)
        entries = []
        for index in range(2):
            entry = _adopt(tier, f"c{index}", _matrix(index))
            with tier._cond:
                tier.fitted(entry)
                tier.published(entry, f"ref-{index}")
            entries.append(entry)
        tier.flush(30)
        kept, other = entries
        self.assertTrue(kept.on_device and other.on_device)
        self.assertIsNone(kept.placed)
        report = tier.promote([kept])  # already on the device: no copy, a placement
        self.assertEqual((report["entries"], report["bytes"], report["placed_in_place"]), (0, 0, 1))
        self.assertEqual(kept.placed, "gpu_raw")
        device.memory["free"] = 0  # the pressure response demotes what it may
        event = tier.pressure_check("vae_decode", wait=True, timeout=30)
        self.assertTrue(event["pressure"])
        self.assertTrue(kept.on_device)  # placed: kept
        self.assertFalse(other.on_device)  # not placed: demoted
        with tier._cond:
            tier.gpu_budget_bytes = 0  # the budget's eviction skips placed copies too
            tier._balance()
        self.assertTrue(kept.on_device)
        _check_invariants(self, tier)


def _release_when(release: threading.Event, predicate, timeout: float = 60.0) -> threading.Thread:
    """Set ``release`` once ``predicate()`` holds (polled every 10 ms): an ordering by state, not by a timer."""
    def run():
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.01)
        release.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


class DecodeEntryGuardTests(_Pipelines, unittest.TestCase):
    """dev22j6 (5b's nb_defer A/B on F4: the first never-block decode of a fresh process ran out of memory with the
    policy off; and the RAVEN l3w4x3 OOM), reworked in dev22j7 (the F5 gate: latency) and dev22j8 (the F5.1 gate): a
    decode section needs max(reserve, prior) plus the growth bound of a node on the GPU, checked against the device
    HEADROOM. When it fits nothing is done; short with a node on the GPU, its measured remaining growth (bound minus
    what the fitter's streams hold) may fit (nothing done either); else the fitter pauses at its next step boundary
    (growth 0); still short, cached blocks go back to the driver when the fragments could cover the shortfall and the
    pressure response runs; still short, the decode waits for the node (``short_after_relief``, or
    ``unlearned_reserve`` while the reserve is not learned: no unconditional first-decode deferral since dev22j8). CPU
    pipeline, a stand-in fitter, a fake device-memory provider (1 GiB: prior 16 %), CPU tensors standing in for device
    copies."""

    def archive(self, *, free, reserve=None, pipeline=None, guard=True, prior=None, segments=None, floor=1):
        """``floor``: the configured decode floor (dev22j8v5); the default, 1 byte, is below every target here (the
        dev22j8v4 arithmetic); None leaves the floor to the profile, the learned clean working set or a wait."""
        import types
        nbytes = _matrix(0).numel() * 2
        device = _FakeDevice(free=free)
        tier = RawChunkTier(device_memory=device, gpu_budget_bytes=1 << 30, host_budget_bytes=1 << 30,
                            gpu_reserve_bytes=reserve, device_segments=segments)
        self.addCleanup(tier.close)
        entry = _adopt(tier, "c0", _matrix(0))
        with tier._cond:
            tier.fitted(entry)
            tier.published(entry, "ref-0")
        tier.flush(30)
        archive = _fake_archive(tier, defer=False)
        archive.config.decode_reserve_prior_bytes = prior
        archive.config.decode_working_set_floor_bytes = floor
        for name in ("_decode_entry_guard", "_release_cached_blocks"):
            setattr(archive, name, types.MethodType(getattr(la.LineageSessionArchive, name), archive))
        archive.decode_guard_ = la._decode_guard_state(guard)
        if pipeline is None:
            _, pipeline = self.pipeline(name=f"guard-{uuid.uuid4().hex[:8]}")
        archive.pipeline, archive.fitter = pipeline, object()
        return archive, tier, device, entry

    @staticmethod
    def learn(tier, nbytes=1 << 20):
        with tier._cond:
            tier._working_set = nbytes  # as if a decode had been measured

    def blocked_pipeline(self):
        """A pipeline whose first node stops inside its GPU work (the plan hook) until ``release`` is set."""
        release, planning = threading.Event(), threading.Event()

        def hooks(stage, node_id):
            if stage == "plan":
                planning.set()
                self.assertTrue(release.wait(60))

        _, pipeline = self.pipeline(hooks=hooks, name=f"inflight-{uuid.uuid4().hex[:8]}")
        self.addCleanup(release.set)
        matrices, contexts = s6.chunk(0)
        ticket = pipeline.submit("main", "chunk-000000", matrices, contexts)
        self.assertTrue(planning.wait(60))
        return pipeline, ticket, release, sum(t.numel() * t.element_size() for t in matrices.values())

    def test_with_a_node_on_the_gpu_unreleasable_fragments_are_no_headroom_and_the_decode_waits(self):
        """dev22j8v4 (F5.2's RAVEN decode OOM, 4444415 / 4444409): the formula counted 2.0-2.4 GB of cached bytes
        no one could reclaim (free fragments in used expandable segments: ``inactive_split`` 0), the paused decode
        ran and ran out of memory. With a node on the GPU the strict headroom decides: short on it after the pause
        and the relief, the decode waits for the node (F5.1's wait); every record carries both numbers."""
        pipeline, ticket, release, source_bytes = self.blocked_pipeline()
        provider = _FakeSegments([_segment(expandable=True, blocks=[("active_allocated", 4 << 20),
                                                                     ("inactive", 12 << 20),
                                                                     ("active_allocated", 4 << 20)])])
        archive, tier, device, _ = self.archive(free=1 << 30, pipeline=pipeline, segments=provider)
        self.learn(tier)
        floor = max(tier.reserve_bytes(), la.decode_reserve_prior_bytes(archive.config, {"total": 1 << 30}))
        # the formula: free + cached bytes >= floor + the bound; strict: the free bytes only, short of the floor
        device.memory.update(free=floor // 2, reserved=floor + source_bytes, allocated=0, inactive_split=0)
        _release_when(release, lambda: pipeline.stats()["fitter_pause"]["open"] >= 1)
        from reelcache import archive_pipeline as ap
        with mock.patch.object(ap, "PAUSE_ACK_TIMEOUT", 30.0):
            with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
                self.assertTrue(pipeline.stats()["deferred"])  # it waits for the node, not next to a paused one
        guard = report["decode_guard"]
        self.assertGreaterEqual(guard["formula_headroom"], floor + source_bytes)  # the old verdict: as is
        self.assertEqual((guard["decided_by"], guard["strict_override"], guard["strict_free"],
                          guard["strict_releasable"], guard["strict_unreleasable"], guard["strict_headroom"]),
                         ("strict", True, floor // 2, 0, 12 << 20, floor // 2))
        self.assertEqual((guard["headroom"], guard["headroom_after"], guard["strict_headroom_after"]),
                         (floor // 2, floor // 2, floor // 2))
        self.assertGreaterEqual(guard["formula_headroom_after"], guard["target"])
        self.assertGreaterEqual(guard["snapshot_seconds"], 0.0)
        self.assertTrue(guard["pause"]["acked"])
        self.assertEqual([action["action"] for action in guard["actions"]], ["relief"])  # no cache release
        self.assertEqual((guard["short"], report["deferral_reason"]), (True, "short_after_relief"))
        self.assertEqual(provider.calls, 2)  # at the check and after the relief
        state = archive.decode_guard_
        self.assertEqual((state["strict_checks"], state["strict_overrides"], state["strict_release_skips"],
                          state["fits"], state["fits_paused"], state["fits_after_relief"], state["short_deferrals"]),
                         (1, 1, 1, 0, 0, 0, 1))
        self.assertGreaterEqual(state["snapshot_seconds_max"], 0.0)
        self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))

    def test_with_a_node_on_the_gpu_releasable_cached_bytes_count_as_headroom(self):
        """Wholly free segments (on any stream) are released by the allocator itself: strict headroom."""
        pipeline, ticket, release, source_bytes = self.blocked_pipeline()
        archive, tier, device, _ = self.archive(free=1 << 30, pipeline=pipeline)
        self.learn(tier)
        floor = max(tier.reserve_bytes(), la.decode_reserve_prior_bytes(archive.config, {"total": 1 << 30}))
        cached = floor + source_bytes  # one wholly free segment
        tier._device_segments = provider = _FakeSegments([_segment(expandable=False, blocks=[("inactive", cached)])])
        device.memory.update(free=floor // 2, reserved=cached, allocated=0, inactive_split=0)
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            self.assertFalse(pipeline.stats()["deferred"])
        release.set()
        guard = report["decode_guard"]
        self.assertEqual((guard["decided_by"], guard["fits"], guard["actions"], guard["strict_releasable"],
                          guard["strict_headroom"]), ("strict", "as_is", [], cached, floor // 2 + cached))
        self.assertNotIn("strict_override", guard)
        self.assertEqual(provider.calls, 1)
        self.assertNotIn("deferral_reason", report)
        self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))

    def test_without_a_node_on_the_gpu_the_formula_decides_and_no_snapshot_is_taken(self):
        provider = _FakeSegments([])
        archive, tier, device, _ = self.archive(free=1 << 30, segments=provider)
        self.learn(tier)
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            pass
        guard = report["decode_guard"]
        self.assertEqual((guard["decided_by"], guard["fits"], guard["inflight"]), ("formula", "as_is", False))
        self.assertNotIn("strict_headroom", guard)
        self.assertEqual(provider.calls, 0)
        self.assertEqual(archive.decode_guard_["strict_checks"], 0)

    def test_the_prior_is_a_share_of_the_device_or_configured(self):
        memory = {"total": 40 << 30}
        self.assertEqual(la.decode_reserve_prior_bytes(SimpleNamespace(), memory), int(0.16 * (40 << 30)))
        self.assertEqual(la.decode_reserve_prior_bytes(SimpleNamespace(decode_reserve_prior_bytes=0), memory), 0)
        self.assertEqual(la.decode_reserve_prior_bytes(SimpleNamespace(decode_reserve_prior_bytes=7), memory), 7)
        from reelcache.integrations.sglang import ReelCacheRuntimeConfig
        config = ReelCacheRuntimeConfig.from_dict({"store_root": "guard"})
        self.assertEqual((config.decode_entry_guard, config.decode_reserve_prior_bytes), (True, None))
        self.assertEqual(ReelCacheRuntimeConfig.from_dict({"store_root": "guard", "decode_reserve_prior_bytes": 0})
                         .decode_reserve_prior_bytes, 0)
        for bad in ({"decode_entry_guard": 1}, {"decode_reserve_prior_bytes": -1},
                    {"decode_reserve_prior_bytes": 1.5}):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                ReelCacheRuntimeConfig.from_dict({"store_root": "guard", **bad})

    def test_a_first_decode_with_room_is_neither_deferred_nor_waits(self):
        """dev22j8 (item 4; 5b: the one-time ``unlearned_reserve`` wait was most of CausVid's remaining gap): an
        unlearned decode takes the same checks with the prior as its floor; with room nothing is done, and its working
        set is still learned at the section's end."""
        archive, tier, device, entry = self.archive(free=1 << 30)
        self.assertEqual((tier.learned_working_set_bytes(), tier.reserve_configured()), (0, False))
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            self.assertFalse(archive.pipeline.stats()["deferred"])  # archival keeps running next to the decode
            device.memory["peak"] = 50 << 20  # the decode's own peak: learned at the section's end
        guard = report["decode_guard"]
        self.assertEqual((guard["unlearned"], guard["prior"], guard["actions"], guard["short"], guard["fits"]),
                         (True, int(0.16 * (1 << 30)), [], False, "as_is"))
        self.assertNotIn("deferral_reason", report)
        self.assertFalse(report["deferred"])
        self.assertTrue(entry.on_device)
        self.assertEqual(tier.learned_working_set_bytes(), 50 << 20)
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as again:
            self.assertFalse(archive.pipeline.stats()["deferred"])
        self.assertNotIn("deferral_reason", again)
        self.assertEqual(again["decode_guard"]["unlearned"], False)
        state = archive.decode_guard_
        self.assertEqual((state["checks"], state["fits"], state["unlearned_deferrals"], state["reliefs"]), (2, 2, 0, 0))

    def test_a_first_decode_short_of_the_prior_demotes_gpu_raw_copies_first(self):
        archive, tier, device, entry = self.archive(free=0)
        prior = la.decode_reserve_prior_bytes(archive.config, {"total": 1 << 30})
        device.memory["free"] = tier.reserve_bytes() + 1  # the reserve check passes; the prior does not
        self.assertLess(device.memory["free"], prior)
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            self.assertTrue(archive.pipeline.stats()["deferred"])
        self.assertFalse(report["raw_pressure"]["pressure"])
        guard = report["decode_guard"]
        (relief,) = [action for action in guard["actions"] if action["action"] == "relief"]
        self.assertEqual((relief["event"]["where"], relief["event"]["target"]), ("vae_decode:decode_guard", prior))
        self.assertFalse(entry.on_device)  # the GPU raw copy went to host before the decode
        self.assertEqual(report["deferral_reason"], "unlearned_reserve")
        self.assertEqual(tier.stats()["pressure"]["by_where"].get("vae_decode:decode_guard"), 1)

    def test_unlearned_decodes_with_room_never_defer_and_the_guard_skips_restores_configured_reserves_and_off(self):
        archive, tier, device, entry = self.archive(free=1 << 30)
        reasons = []
        for _ in range(3):  # no new allocator peak: nothing is learned, and nothing is deferred either (dev22j8)
            with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
                self.assertFalse(archive.pipeline.stats()["deferred"])
            reasons.append(report.get("deferral_reason"))
        self.assertEqual(reasons, [None, None, None])
        self.assertEqual((archive.decode_guard_["unlearned_deferrals"], archive.decode_guard_["fits"]), (0, 3))
        fresh, _, _, _ = self.archive(free=1 << 30)
        with la.LineageSessionArchive.memory_critical(fresh, "restore") as report:  # never a restore
            self.assertFalse(fresh.pipeline.stats()["deferred"])
        self.assertNotIn("decode_guard", report)
        configured, tier2, _, _ = self.archive(free=1 << 30, reserve=2 * _matrix(0).numel() * 2)
        self.assertTrue(tier2.reserve_configured())
        with la.LineageSessionArchive.memory_critical(configured, "vae_decode") as report:  # nothing to learn
            self.assertFalse(configured.pipeline.stats()["deferred"])
        self.assertNotIn("deferral_reason", report)
        off, _, _, _ = self.archive(free=1 << 30, guard=False)
        with la.LineageSessionArchive.memory_critical(off, "vae_decode") as report:  # decode_entry_guard=False
            self.assertFalse(off.pipeline.stats()["deferred"])
        self.assertNotIn("decode_guard", report)

    def test_still_short_after_relief_defers_the_decode(self):
        archive, tier, device, entry = self.archive(free=0)
        self.learn(tier)
        prior = la.decode_reserve_prior_bytes(archive.config, {"total": 1 << 30})
        device.memory["free"] = max(tier.reserve_bytes(), prior // 2) + 1  # above the reserve, short of the prior
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            self.assertTrue(archive.pipeline.stats()["deferred"])
        self.assertTrue(report["decode_guard"]["short"])
        self.assertEqual(report["deferral_reason"], "short_after_relief")
        self.assertEqual(archive.decode_guard_["short_deferrals"], 1)

    def test_room_in_the_cached_segments_needs_no_release_relief_or_pause(self):
        """The F5 gate's LongLive 2.0 records: the driver's free memory below the target, the headroom (free plus
        wholly free cached segments, which the allocator releases itself) well above it. dev22j6 released the cache
        at every such decode (0.36-0.45 s of TTFNC); dev22j7 does nothing."""
        archive, tier, device, entry = self.archive(free=1 << 30)  # room at admission: the copy stays on the GPU
        self.learn(tier)
        floor = max(tier.reserve_bytes(), la.decode_reserve_prior_bytes(archive.config, {"total": 1 << 30}))
        device.memory.update(free=floor // 2, reserved=floor, allocated=0, inactive_split=0)
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            self.assertFalse(archive.pipeline.stats()["deferred"])
        guard = report["decode_guard"]
        self.assertEqual((guard["actions"], guard["short"], guard["fits"], guard["inflight"], guard["target"]),
                         ([], False, "as_is", False, floor))
        self.assertGreater(guard["headroom"], guard["target"])
        self.assertLess(guard["free"], guard["target"])
        self.assertNotIn("deferral_reason", report)
        self.assertTrue(entry.on_device)
        state = archive.decode_guard_
        self.assertEqual((state["fits"], state["cache_releases"], state["reliefs"], state["pauses"]), (1, 0, 0, 0))

    def test_an_inflight_node_short_of_its_bound_pauses_at_its_next_step_boundary_instead_of_waiting(self):
        from reelcache import archive_pipeline as ap
        pipeline, ticket, release, source_bytes = self.blocked_pipeline()
        self.assertEqual(pipeline.inflight_growth_bound(), {"inflight": True, "bytes": source_bytes,
                                                            "basis": "working_set"})
        archive, tier, device, entry = self.archive(free=1 << 30, pipeline=pipeline)
        self.learn(tier)
        floor = max(tier.reserve_bytes(), la.decode_reserve_prior_bytes(archive.config, {"total": 1 << 30}))
        device.memory["free"] = floor + source_bytes // 2  # above the floor, short of floor + the node's bound
        # once the guard asked for the pause, the node leaves the plan hook and stops at its next step boundary
        _release_when(release, lambda: pipeline.stats()["fitter_pause"]["open"] >= 1)
        with mock.patch.object(ap, "PAUSE_ACK_TIMEOUT", 30.0):
            with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
                stats = pipeline.stats()
                self.assertTrue(stats["fitter_pause"]["paused"])  # the node waits at its boundary, still on the GPU
                self.assertEqual(stats["fitter_pause"]["open"], 1)
                self.assertTrue(pipeline.inflight_growth_bound()["inflight"])
                self.assertFalse(ticket.done())
                time.sleep(0.2)
                self.assertFalse(ticket.done())  # it stays paused while the decode runs
        guard = report["decode_guard"]
        self.assertEqual((guard["inflight_bytes"], guard["growth"], guard["target"], guard["short"], guard["fits"]),
                         (source_bytes, 0, floor, False, "paused"))
        self.assertTrue(guard["pause"]["acked"])
        self.assertGreater(guard["pause"]["section_seconds"], 0.0)
        self.assertEqual(guard["actions"], [])  # no cache release, no relief
        self.assertNotIn("deferral_reason", report)  # no wait for the node
        self.assertTrue(entry.on_device)
        state = archive.decode_guard_
        self.assertEqual((state["pauses"], state["pause_acks"], state["fits_paused"], state["short_deferrals"],
                          state["inflight_deferrals"]), (1, 1, 1, 0, 0))
        self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))  # resumed when the section closed
        pause = pipeline.stats()["fitter_pause"]
        self.assertEqual((pause["requests"], pause["acked"], pause["pauses"], pause["open"], pause["paused"]),
                         (1, 1, 1, 0, False))

    def test_an_unacknowledged_pause_counts_the_bound_and_the_decode_waits_only_when_still_short(self):
        """The node stays inside a step (here the plan hook) past ``PAUSE_ACK_TIMEOUT``: its growth bound counts;
        short after the release and the relief, the decode waits for it (``short_after_relief``)."""
        pipeline, ticket, release, source_bytes = self.blocked_pipeline()
        archive, tier, device, _ = self.archive(free=0, pipeline=pipeline)
        self.learn(tier)
        floor = max(tier.reserve_bytes(), la.decode_reserve_prior_bytes(archive.config, {"total": 1 << 30}))
        device.memory["free"] = floor + source_bytes // 2
        # the node stays in its step until the guard gave up on the acknowledgement, then finishes while the
        # decode waits for it
        _release_when(release, lambda: pipeline.stats()["fitter_pause"]["ack_timeouts"] >= 1)
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            self.assertTrue(pipeline.stats()["deferred"])
            self.assertFalse(pipeline.inflight_growth_bound()["inflight"])  # it left the GPU before the decode ran
            self.assertEqual(pipeline.stats()["fitter_pause"]["open"], 0)  # the pause was lifted for the wait
        self.assertFalse(report["raw_pressure"]["pressure"])
        guard = report["decode_guard"]
        self.assertFalse(guard["pause"]["acked"])
        self.assertEqual((guard["inflight_bytes"], guard["growth"], guard["target"], guard["short"]),
                         (source_bytes, source_bytes, floor + source_bytes, True))
        self.assertEqual([action["action"] for action in guard["actions"]], ["relief"])  # CPU: no cache to release
        self.assertEqual((report["deferral_reason"], report["deferred_archival"]), ("short_after_relief", True))
        self.assertGreater(report["deferral_wait_seconds"], 0.0)
        state = archive.decode_guard_
        self.assertEqual((state["inflight_checks"], state["pause_ack_timeouts"], state["short_deferrals"],
                          state["inflight_deferrals"]), (1, 1, 1, 1))
        self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))

    def test_an_inflight_node_that_fits_keeps_running(self):
        pipeline, ticket, release, source_bytes = self.blocked_pipeline()
        archive, tier, device, _ = self.archive(free=0, pipeline=pipeline)
        self.learn(tier)
        floor = max(tier.reserve_bytes(), la.decode_reserve_prior_bytes(archive.config, {"total": 1 << 30}))
        device.memory["free"] = floor + 2 * source_bytes  # room for the floor and the node
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            self.assertFalse(pipeline.stats()["deferred"])
            self.assertEqual(pipeline.inflight_gpu_bytes(), source_bytes)  # still on the GPU, not waited for
        release.set()
        self.assertEqual(report["decode_guard"]["actions"], [])
        self.assertNotIn("deferral_reason", report)
        self.assertEqual((archive.decode_guard_["inflight_checks"], archive.decode_guard_["inflight_deferrals"]), (1, 0))
        self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))

    def floor(self, archive, tier):
        return max(tier.reserve_bytes(), la.decode_reserve_prior_bytes(archive.config, {"total": 1 << 30}))

    @staticmethod
    def fitter_memory(held, fragments=0):
        return lambda: {"held": held, "allocated": held, "tier": 0, "reserved": held + fragments,
                        "fragments": fragments, "segments": 1}

    def test_a_short_decode_whose_measured_remaining_growth_fits_needs_no_pause(self):
        """dev22j8 (item 2; 5b's F5.1 warm-up rerun: LongLive 2.0 was short of base + its full 16 GiB bound at every
        in-flight decode): on the slow path one snapshot of the fitter's streams gives what the node holds; the bound
        minus that fits, so nothing is done -- no pause, no acknowledgement, no release. dev22j8v5: the slow path
        with the formula fitting the full bound (short on it, the fitter pauses before the snapshot): here an
        unreleasable fragment the formula counts."""
        pipeline, ticket, release, source_bytes = self.blocked_pipeline()
        with pipeline._cond:
            pipeline._keep_peak(8 * source_bytes, source_bytes, "warmup")
        fragment = 5 * source_bytes
        provider = _FakeSegments([_segment(expandable=True, blocks=[("active_allocated", 4 << 20),
                                                                     ("inactive", fragment),
                                                                     ("active_allocated", 4 << 20)])])
        archive, tier, device, entry = self.archive(free=1 << 30, pipeline=pipeline, segments=provider)
        self.learn(tier)
        floor = self.floor(archive, tier)
        # short of floor + 8x on the free and the strict headroom; room for floor + the remaining 2x; the formula
        # (free + the fragment) fits the full bound
        device.memory.update(free=floor + 3 * source_bytes, reserved=fragment, allocated=0, inactive_split=0)
        with mock.patch.object(pipeline, "fitter_memory", self.fitter_memory(6 * source_bytes)):
            with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
                self.assertFalse(pipeline.stats()["deferred"])
                self.assertEqual(pipeline.stats()["fitter_pause"]["open"], 0)
        release.set()
        guard = report["decode_guard"]
        self.assertEqual((guard["inflight_bytes"], guard["bound_basis"], guard["held"], guard["remaining"],
                          guard["growth"], guard["target"], guard["fits"], guard["actions"]),
                         (8 * source_bytes, "warmup", 6 * source_bytes, 2 * source_bytes, 2 * source_bytes,
                          floor + 2 * source_bytes, "measured", []))
        self.assertNotIn("pause", guard)
        self.assertNotIn("deferral_reason", report)
        self.assertEqual((guard["decided_by"], guard["strict_headroom"], guard["strict_override"]),
                         ("strict", floor + 3 * source_bytes, True))
        self.assertTrue(entry.on_device)
        state = archive.decode_guard_
        self.assertEqual((state["fits_measured"], state["pauses"], state["cache_releases"], state["reliefs"],
                          state["pauses_first"]), (1, 0, 0, 0, 0))
        self.assertEqual(pipeline.stats()["fitter_pause"]["requests"], 0)
        self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))

    def test_an_unacknowledged_pause_counts_the_measured_remaining_growth_and_records_the_phase(self):
        """dev22j8 (item 2): the node stays inside a step past ``PAUSE_ACK_TIMEOUT`` (here the plan hook: neither a step
        boundary nor a wait without allocation): its measured remaining growth counts, not the bound, and the timeout
        records the fitter's phase. Its own free blocks are no reason for a cache release (releasing them only moves
        its growth into the headroom)."""
        pipeline, ticket, release, source_bytes = self.blocked_pipeline()
        with pipeline._cond:
            pipeline._keep_peak(8 * source_bytes, source_bytes, "warmup")
        archive, tier, device, _ = self.archive(free=0, pipeline=pipeline)
        self.learn(tier)
        floor = self.floor(archive, tier)
        device.memory.update(free=floor + 2 * source_bytes, inactive_split=source_bytes)
        releases = []
        archive._release_cached_blocks = lambda: releases.append(True) or 0
        _release_when(release, lambda: pipeline.stats()["fitter_pause"]["ack_timeouts"] >= 1)
        with mock.patch.object(pipeline, "fitter_memory", self.fitter_memory(5 * source_bytes, source_bytes)):
            with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
                self.assertTrue(pipeline.stats()["deferred"])
        guard = report["decode_guard"]
        self.assertFalse(guard["pause"]["acked"])
        self.assertEqual((guard["pause"]["landed"], guard["pause"]["phase"]), (None, "plan"))
        self.assertGreater(guard["pause"]["boundary_seconds"], 0.0)
        self.assertEqual((guard["remaining"], guard["growth"], guard["target"], guard["short"]),
                         (3 * source_bytes, 3 * source_bytes, floor + 3 * source_bytes, True))
        self.assertEqual((guard["shortfall"], guard["fragments"], releases), (source_bytes, 0, []))
        self.assertEqual([action["action"] for action in guard["actions"]], ["relief"])
        self.assertEqual(guard["remaining_after"], 3 * source_bytes)  # measured again after the relief
        self.assertEqual(report["deferral_reason"], "short_after_relief")
        pause = pipeline.stats()["fitter_pause"]
        self.assertEqual(pause["ack_timeout_phases"], {"plan": 1})
        self.assertEqual(pause["ack_timeout_events"][-1]["phase"], "plan")
        state = archive.decode_guard_
        self.assertEqual((state["pause_ack_timeouts"], state["cache_release_skips"], state["short_deferrals"]),
                         (1, 1, 1))
        self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))

    def test_the_cache_release_runs_only_when_fragmented_cached_bytes_could_cover_the_shortfall(self):
        """dev22j8 (item 3; 5b: the 15-16.5 GiB released at each short LongLive 2.0 decode moved its headroom by
        0.17 GiB): wholly free cached segments are already in the headroom (the allocator frees them on demand), so
        the release runs only when the fragmented cached bytes (inactive split) could cover the shortfall."""
        for fragments, released in ((1 << 20, False), (8 << 20, True)):
            with self.subTest(fragments=fragments):
                archive, tier, device, entry = self.archive(free=1 << 30)
                self.learn(tier)
                floor = self.floor(archive, tier)
                device.memory.update(free=floor - (4 << 20), inactive_split=fragments)

                def release_fragments():  # the release returns the fragments to the driver
                    freed = device.memory["inactive_split"]
                    device.memory.update(free=device.memory["free"] + freed, inactive_split=0)
                    return freed

                archive._release_cached_blocks = release_fragments
                with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
                    pass
                guard = report["decode_guard"]
                self.assertEqual((guard["shortfall"], guard["fragments"]), (4 << 20, fragments))
                actions = [action["action"] for action in guard["actions"]]
                state = archive.decode_guard_
                if released:
                    self.assertEqual(actions, ["release_cached"])
                    self.assertEqual((guard["short"], guard["fits"]), (False, "after_relief"))
                    self.assertNotIn("deferral_reason", report)
                    self.assertTrue(entry.on_device)  # no relief: no demotion
                    self.assertEqual((state["cache_releases"], state["cache_release_skips"]), (1, 0))
                else:
                    self.assertEqual(actions, ["relief"])
                    self.assertEqual(report["deferral_reason"], "short_after_relief")
                    self.assertEqual((state["cache_releases"], state["cache_release_skips"]), (0, 1))

    def test_an_unlearned_decode_next_to_a_node_waits_only_when_still_short_after_the_pause_and_relief(self):
        """dev22j8 (item 4): no unconditional first-decode deferral, but an unlearned decode still short after the
        pause (here not acknowledged) and the relief waits for the node. dev22j8v6 (the F5.5 gate's RAVEN d32
        4446697): next to a node the decode floor is known by then, so the wait is ``short_after_relief`` even while
        the reserve has learned nothing; the record names the step that held the fitter past the acknowledgement
        timeout and whether an acknowledged pause would have let the decode run."""
        pipeline, ticket, release, source_bytes = self.blocked_pipeline()
        archive, tier, device, _ = self.archive(free=0, pipeline=pipeline)
        floor = self.floor(archive, tier)
        device.memory["free"] = floor + source_bytes // 2  # short of floor + the node's bound (its source bytes)
        _release_when(release, lambda: pipeline.stats()["fitter_pause"]["ack_timeouts"] >= 1)
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            self.assertTrue(pipeline.stats()["deferred"])
        guard = report["decode_guard"]
        self.assertEqual((guard["unlearned"], guard["pause"]["acked"], guard["short"]), (True, False, True))
        self.assertEqual((report["deferral_reason"], report["deferred_archival"]), ("short_after_relief", True))
        self.assertEqual((guard["ack_timeout_phase"], guard["pause"]["phase"], guard["fits_if_acked"]),
                         ("plan", "plan", True))  # the plan hook held the fitter; paused, it would have fit
        self.assertGreaterEqual(guard["ack_timeout_boundary_seconds"], guard["pause"]["ack_seconds"])
        state = archive.decode_guard_
        self.assertEqual((state["unlearned_deferrals"], state["short_deferrals"], state["inflight_deferrals"]),
                         (0, 1, 1))
        self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))

    def test_the_decode_floor_is_configured_the_profile_s_at_its_default_geometry_or_learned(self):
        """dev22j8v5 (F5.4 finding 1): the floor is 1.1 x the decode's clean working set -- configured, the bundled
        value of the profile while archival runs at its default geometry (never scaled to another), else the largest
        clean one this process measured; none: the decode waits for the node."""
        from reelcache.integrations.sglang import ReelCacheRuntimeConfig
        mib = 1 << 20
        self.assertEqual({name: value[0] for name, value in la.DECODE_WORKING_SETS.items()},
                         {"causvid": 7390852608, "raven": 7308185088, "longlive1": 7300996608,
                          "longlive2": 15486119936})  # dev22j8v6 (F6 c): LongLive 1.0's 6.80 GiB (4446362)
        self.assertEqual((la.DECODE_WORKING_SETS["raven"][1], la.DECODE_WORKING_SETS["longlive1"][1],
                          la.DECODE_WORKING_SETS["longlive2"][1]), ((30, 52), (30, 52), (22, 40)))
        learned = [0]
        tier = SimpleNamespace(learned_clean_working_set_bytes=lambda: learned[0])
        archive = SimpleNamespace(config=SimpleNamespace(decode_working_set_floor_bytes=None),
                                  _decode_profile=("raven", (30, 52)), _capture_grid=(30, 52))
        self.assertEqual(la._decode_floor(archive, tier), (int(1.1 * 7308185088), "profile"))
        archive._decode_profile, archive._capture_grid = ("longlive2", (22, 40)), (22, 40)
        self.assertEqual(la._decode_floor(archive, tier), (int(1.1 * 15486119936), "profile"))
        archive._decode_profile, archive._capture_grid = ("longlive1", (30, 52)), (30, 52)
        self.assertEqual(la._decode_floor(archive, tier), (int(1.1 * 7300996608), "profile"))
        for profile, grid in ((("longlive2", (22, 40)), (30, 52)),  # another geometry: not scaled
                              (("raven", (30, 52)), None),  # archival's geometry not known yet
                              (("longlive", (30, 52)), (30, 52)),  # no bundled value (not a bundled profile name)
                              (None, (30, 52))):  # no warm-up profile
            archive._decode_profile, archive._capture_grid = profile, grid
            learned[0] = 0
            self.assertEqual(la._decode_floor(archive, tier), (None, "unlearned-wait"), (profile, grid))
            learned[0] = 600 * mib
            self.assertEqual(la._decode_floor(archive, tier), (int(1.1 * 600 * mib), "learned"), (profile, grid))
        archive.config.decode_working_set_floor_bytes = 5 << 30  # configured: always
        self.assertEqual(la._decode_floor(archive, tier), (5 << 30, "config"))
        self.assertIsNone(ReelCacheRuntimeConfig.from_dict({"store_root": "floor"}).decode_working_set_floor_bytes)
        self.assertEqual(ReelCacheRuntimeConfig.from_dict({"store_root": "floor", "decode_working_set_floor_bytes": 7})
                         .decode_working_set_floor_bytes, 7)
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ConfigurationError):
                ReelCacheRuntimeConfig.from_dict({"store_root": "floor", "decode_working_set_floor_bytes": bad})

    def test_without_a_floor_a_decode_next_to_a_node_waits_for_it_and_measures_its_need_after_the_wait(self):
        """dev22j8v5 (F5.4 finding 1, rule b): no floor known (another geometry, no clean measurement yet), the first
        decode next to a node waits for it at once (F5.1; no pause, no snapshot, ``unlearned_reserve``). Its
        working-set mark is taken after the wait -- the node's memory freed during the wait no longer hides the
        decode's own need -- and, clean, it teaches the floor of the decodes after it."""
        pipeline, ticket, release, source_bytes = self.blocked_pipeline()
        provider = _FakeSegments([])
        archive, tier, device, _ = self.archive(free=1 << 30, pipeline=pipeline, segments=provider, floor=None)
        self.learn(tier)
        mib = 1 << 20
        device.memory.update(allocated=300 * mib, peak=300 * mib)  # 200 MiB of them the node's

        def node_finishes():  # the node's memory goes when it finishes, inside the decode's wait
            deadline = time.monotonic() + 60
            while not pipeline.stats()["deferred"] and time.monotonic() < deadline:
                time.sleep(0.01)
            device.memory.update(allocated=100 * mib)
            release.set()

        threading.Thread(target=node_finishes, daemon=True).start()
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            self.assertTrue(pipeline.stats()["deferred"])
            device.memory.update(peak=500 * mib)  # the decode's need: 400 MiB above what is left after the wait
        guard = report["decode_guard"]
        self.assertEqual((guard["floor_bytes"], guard["floor_source"], guard["short"]), (None, "unlearned-wait", True))
        self.assertEqual(guard["target_before_floor"], guard["base"] + guard["inflight_bytes"])
        self.assertNotIn("pause", guard)
        self.assertNotIn("strict_headroom", guard)
        self.assertEqual(provider.calls, 0)
        self.assertEqual((report["deferral_reason"], report["deferred_archival"]), ("unlearned_reserve", True))
        self.assertEqual((report["working_set_bytes"], report["working_set_clean"]), (400 * mib, True))
        state = archive.decode_guard_
        self.assertEqual((state["floor_waits"], state["pauses"], state["strict_checks"], state["unlearned_deferrals"]),
                         (1, 0, 0, 0))
        self.assertEqual(tier.learned_clean_working_set_bytes(), 400 * mib)
        self.assertEqual(la._decode_floor(archive, tier), (int(1.1 * 400 * mib), "learned"))
        self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))

    def test_a_decode_short_of_its_floor_is_no_paused_fit_and_the_pause_comes_before_the_snapshot(self):
        """dev22j8v5 (F5.4 finding 1: a RAVEN decode needing 6.81 GiB ran paused next to a 6.32 GiB target with
        7.03 GiB of strict headroom): the floor raises the target; strict headroom above the old target but below the
        floor is no paused fit -- after the relief the decode waits for the node. Short on the formula, the fitter is
        paused before the allocator snapshot (finding 2: a snapshot next to an allocating node took 32.5 ms)."""
        pipeline, ticket, release, source_bytes = self.blocked_pipeline()
        open_at_snapshot = []

        class Provider(_FakeSegments):
            def __call__(self, device):
                open_at_snapshot.append(pipeline.stats()["fitter_pause"]["open"])
                return super().__call__(device)

        provider = Provider([])
        archive, tier, device, _ = self.archive(free=1 << 30, pipeline=pipeline, segments=provider)
        self.learn(tier)
        base = self.floor(archive, tier)
        floor = base + 8 * source_bytes
        archive.config.decode_working_set_floor_bytes = floor
        device.memory.update(free=base + 4 * source_bytes)  # above the old target (base + the bound), below the floor
        _release_when(release, lambda: pipeline.stats()["fitter_pause"]["open"] >= 1)
        from reelcache import archive_pipeline as ap
        with mock.patch.object(ap, "PAUSE_ACK_TIMEOUT", 30.0):
            with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
                self.assertTrue(pipeline.stats()["deferred"])
                self.assertEqual(pipeline.stats()["fitter_pause"]["open"], 0)  # waited for the node, not paused
        guard = report["decode_guard"]
        self.assertEqual((guard["floor_source"], guard["floor_bytes"], guard["target_before_floor"], guard["base"]),
                         ("config", floor, base + guard["inflight_bytes"], floor))
        self.assertLessEqual(guard["target_before_floor"], guard["strict_headroom"])  # dev22j8v4: a paused fit
        self.assertLess(guard["strict_headroom"], floor)
        self.assertTrue(guard["pause"]["acked"])
        self.assertEqual(open_at_snapshot[0], 1)  # the pause was open when the snapshot was taken
        self.assertEqual((guard["target"], guard["short"], report["deferral_reason"]),
                         (floor, True, "short_after_relief"))
        self.assertEqual([action["action"] for action in guard["actions"]], ["relief"])
        state = archive.decode_guard_
        self.assertEqual((state["floor_raised"], state["pauses_first"], state["fits_paused"], state["short_deferrals"],
                          state["free_fits"]), (1, 1, 0, 1, 0))
        self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))

    def test_when_the_free_memory_covers_the_target_next_to_a_node_no_snapshot_is_taken(self):
        """dev22j8v5 (F5.4 finding 2): the driver's free memory is a lower bound of the strict headroom; when it alone
        covers the target (floor included) and the node's bound, the decode runs as is without the snapshot."""
        pipeline, ticket, release, source_bytes = self.blocked_pipeline()
        provider = _FakeSegments([])
        archive, tier, device, _ = self.archive(free=1 << 30, pipeline=pipeline, segments=provider)
        self.learn(tier)
        base = self.floor(archive, tier)
        archive.config.decode_working_set_floor_bytes = base + source_bytes
        device.memory.update(free=base + 2 * source_bytes)  # the floor plus the node's bound
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            self.assertFalse(pipeline.stats()["deferred"])
        release.set()
        guard = report["decode_guard"]
        self.assertEqual((guard["decided_by"], guard["fits"], guard["headroom"], guard["target"], guard["floor_source"]),
                         ("free", "as_is", base + 2 * source_bytes, base + 2 * source_bytes, "config"))
        self.assertNotIn("strict_headroom", guard)
        self.assertNotIn("pause", guard)
        self.assertEqual(provider.calls, 0)
        state = archive.decode_guard_
        self.assertEqual((state["free_fits"], state["fits"], state["strict_checks"], state["floor_raised"]),
                         (1, 1, 0, 1))
        # a node running unpaused through the section: its working set is no clean measurement
        self.assertFalse(report.get("working_set_clean", False))
        self.assertEqual(tier.learned_clean_working_set_bytes(), 0)
        self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))

    def test_a_section_next_to_an_unpaused_node_is_not_clean(self):
        """dev22j8v5: the working set of a section with a node running unpaused through it (here: as is next to it)
        raises the auto reserve, not the clean working set; nor does a restore section's."""
        pipeline, ticket, release, source_bytes = self.blocked_pipeline()
        archive, tier, device, _ = self.archive(free=1 << 30, pipeline=pipeline)
        self.learn(tier)
        mib = 1 << 20
        device.memory.update(allocated=100 * mib, peak=100 * mib)
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            device.memory.update(peak=900 * mib)
        release.set()
        self.assertEqual(report["decode_guard"]["fits"], "as_is")
        self.assertEqual((report["working_set_bytes"], report["working_set_clean"]), (800 * mib, False))
        self.assertEqual((tier.learned_working_set_bytes(), tier.learned_clean_working_set_bytes()), (800 * mib, 0))
        self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))
        archive.fitter = None  # no archival GPU work: every decode section is clean
        device.memory.update(peak=950 * mib)
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            device.memory.update(peak=1000 * mib)
        self.assertEqual((report["working_set_bytes"], report["working_set_clean"]), (900 * mib, True))
        self.assertEqual(tier.learned_clean_working_set_bytes(), 900 * mib)
        with la.LineageSessionArchive.memory_critical(archive, "restore") as report:  # not a decode: no floor
            device.memory.update(peak=1100 * mib)
        self.assertEqual((report["working_set_bytes"], report["working_set_clean"]), (1000 * mib, False))
        self.assertEqual(tier.learned_clean_working_set_bytes(), 900 * mib)

    def test_a_garbage_collection_during_the_snapshot_is_recorded(self):
        """dev22j8v5 (F5.4 finding 2): each guard record counts its snapshots and the garbage collections that ran
        during them (any thread's), the guard's stats their sum."""
        import gc
        pipeline, ticket, release, source_bytes = self.blocked_pipeline()
        archive, tier, device, _ = self.archive(free=1 << 30, pipeline=pipeline)
        self.learn(tier)
        base = self.floor(archive, tier)
        cached = base + source_bytes  # one wholly free segment: the formula and the strict headroom fit

        class Provider(_FakeSegments):
            def __call__(self, device):
                gc.collect()
                return super().__call__(device)

        tier._device_segments = provider = Provider([_segment(expandable=False, blocks=[("inactive", cached)])])
        device.memory.update(free=base // 2, reserved=cached, allocated=0, inactive_split=0)
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            pass
        release.set()
        guard = report["decode_guard"]
        self.assertEqual((guard["decided_by"], guard["fits"], guard["snapshot_count"]), ("strict", "as_is", 1))
        self.assertGreaterEqual(guard["snapshot_gc"], 1)
        self.assertEqual(archive.decode_guard_["snapshot_gcs"], guard["snapshot_gc"])
        self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))

class FitterPauseTests(_Pipelines, unittest.TestCase):
    """dev22j7: ``ArchivePipeline.pause_gpu_work`` pauses the node on the GPU at its next step boundary (the fitter's
    step hook) without waiting for it, holds queued nodes, and never holds archival someone waits for; and
    ``inflight_growth_bound`` (the fitter's warm-up peak, scaled by the node's size)."""

    def test_the_fitter_s_unpausable_spans_and_an_ack_timeout_s_step_are_measured(self):
        """dev22j8v6 (the F5.5 gate's RAVEN d32 4446697: three pauses timed out in steps 0.44-0.74 s without a
        boundary): the fitter's spans without a step boundary are counted per phase, and an acknowledgement timeout
        learns how long its step ran on after it -- what a longer acknowledgement window would have had to wait."""
        from reelcache import archive_pipeline as ap
        planning = threading.Event()

        def hooks(stage, node_id):
            if stage == "plan":
                planning.set()
                time.sleep(0.4)  # one step without a boundary

        _, pipeline = self.pipeline(hooks=hooks, name=f"spans-{uuid.uuid4().hex[:8]}")
        matrices, contexts = s6.chunk(0)
        ticket = pipeline.submit("main", "chunk-000000", matrices, contexts)
        self.assertTrue(planning.wait(60))
        with pipeline.pause_gpu_work() as info:  # PAUSE_ACK_TIMEOUT (0.1 s) runs out inside the step
            pass
        self.assertFalse(info["acked"])
        self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))
        stats = pipeline.stats()["fitter_pause"]
        [event] = stats["ack_timeout_events"]
        self.assertEqual(event["phase"], "plan")
        self.assertIn(event["step_end"], ("boundary", "blocked", "idle"))
        self.assertGreater(event["boundary_after_seconds"], 0.15)  # the step ran on ~0.3 s after the timeout
        self.assertLess(event["boundary_after_seconds"], 5.0)
        self.assertGreaterEqual(event["step_seconds"], 0.4)
        self.assertEqual(stats["step_span_buckets"], list(ap.SPAN_BUCKETS))
        spans = stats["step_spans"]
        longest = max(spans.values(), key=lambda record: record["max"])
        self.assertGreaterEqual(longest["max"], 0.4)
        self.assertGreaterEqual(sum(longest["buckets"][4:]), 1)  # in a bucket above 0.25 s
        self.assertEqual(sum(sum(record["buckets"]) for record in spans.values()),
                         sum(record["n"] for record in spans.values()))

    def blocked(self):
        release, planning = threading.Event(), threading.Event()

        def hooks(stage, node_id):
            if stage == "plan":
                planning.set()
                self.assertTrue(release.wait(60))

        _, pipeline = self.pipeline(hooks=hooks, name=f"pause-{uuid.uuid4().hex[:8]}")
        self.addCleanup(release.set)
        matrices, contexts = s6.chunk(0)
        ticket = pipeline.submit("main", "chunk-000000", matrices, contexts)
        self.assertTrue(planning.wait(60))
        return pipeline, ticket, release

    def test_an_idle_fitter_acknowledges_at_once_and_queued_nodes_wait_for_the_section(self):
        _, pipeline = self.pipeline(name=f"idle-{uuid.uuid4().hex[:8]}")
        with pipeline.pause_gpu_work(ack_timeout=5.0) as info:
            self.assertEqual((info["acked"], info["inflight"]), (True, False))
            self.assertLess(info["ack_seconds"], 1.0)
            matrices, contexts = s6.chunk(0)
            ticket = pipeline.submit("main", "chunk-000000", matrices, contexts)
            time.sleep(0.3)
            self.assertTrue(pipeline.stats()["deferred"])
            self.assertFalse(ticket.done())  # held like a defer_gpu_work(wait=False) section
        self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))
        self.assertEqual(pipeline.stats()["fitter_pause"]["requests"], 1)

    def test_a_caller_waiting_for_archival_lifts_the_pause(self):
        from reelcache import archive_pipeline as ap
        pipeline, ticket, release = self.blocked()
        _release_when(release, lambda: pipeline.stats()["fitter_pause"]["open"] >= 1)
        with mock.patch.object(ap, "PAUSE_ACK_TIMEOUT", 30.0), pipeline.pause_gpu_work() as info:
            self.assertEqual((info["acked"], info["inflight"]), (True, True))
            self.assertTrue(pipeline.stats()["fitter_pause"]["paused"])
            done = {}
            waiter = threading.Thread(target=lambda: done.setdefault("ref", ticket.result(timeout=120)))
            waiter.start()  # a ticket's result waits for archival: the paused node continues
            waiter.join(120)
            self.assertTrue(done["ref"].startswith("manifest:"))
        pause = pipeline.stats()["fitter_pause"]
        self.assertGreaterEqual(pause["bypassed"], 1)
        self.assertEqual((pause["open"], pause["paused"]), (0, False))

    def test_a_pause_never_outlasts_its_cap(self):
        from reelcache import archive_pipeline as ap
        pipeline, ticket, release = self.blocked()
        _release_when(release, lambda: pipeline.stats()["fitter_pause"]["open"] >= 1)
        with mock.patch.object(ap, "PAUSE_ACK_TIMEOUT", 30.0), mock.patch.object(ap, "PAUSE_MAX_SECONDS", 0.3):
            with pipeline.pause_gpu_work() as info:
                self.assertTrue(info["acked"])
                deadline = time.monotonic() + 120
                while not ticket.done() and time.monotonic() < deadline:  # polled: a ticket wait would lift the pause
                    time.sleep(0.05)
                self.assertTrue(ticket.done())  # expired after the cap: it does not pause again for this request
                self.assertEqual(pipeline.stats()["fitter_pause"]["open"], 1)
        pause = pipeline.stats()["fitter_pause"]
        self.assertEqual((pause["timeouts"], pause["pauses"]), (1, 1))
        self.assertTrue(ticket.result(timeout=10).startswith("manifest:"))

    def test_the_growth_bound_is_the_warm_up_peak_scaled_by_the_node(self):
        pipeline, ticket, release = self.blocked()
        source_bytes = pipeline.inflight_gpu_bytes()
        self.assertEqual(pipeline.inflight_growth_bound(), {"inflight": True, "bytes": source_bytes,
                                                            "basis": "working_set"})
        with pipeline._cond:
            pipeline._fitter_peak_bytes, pipeline._fitter_peak_source_bytes = 1000, source_bytes // 4
        self.assertEqual(pipeline.inflight_growth_bound(), {"inflight": True, "bytes": 4000, "basis": "warmup",
                                                            "scale": 4.0, "upload": 0})
        with pipeline._cond:
            pipeline._fitter_peak_source_bytes = 2 * source_bytes  # a node smaller than the warm-up: no scaling down
        self.assertEqual(pipeline.inflight_growth_bound()["bytes"], 1000)
        release.set()
        self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))
        self.assertEqual(pipeline.inflight_growth_bound(), {"inflight": False, "bytes": 0, "basis": None})

    def test_the_growth_bound_takes_the_larger_measured_peak_with_its_source_bytes(self):
        """dev22j8 (item 1): the warm-up's and the online live peaks are kept apart; the bound takes the larger peak,
        scaled by the node's source bytes over that measurement's (a raw tier entry's bytes while the tier holds the
        sources), plus the re-upload of a raw copy that is not on the device."""
        pipeline, ticket, release = self.blocked()
        source_bytes = pipeline.inflight_gpu_bytes()
        # 4000 B of live growth above the warm-up's synthetic sources (half the node's bytes)
        pipeline._learn_fitter_peak(None, {"source_bytes": source_bytes // 2},
                                    SimpleNamespace(bytes=4000 + source_bytes // 2, method="history"))
        self.assertEqual({key: pipeline.inflight_growth_bound()[key] for key in ("bytes", "basis", "scale")},
                         {"bytes": 8000, "basis": "warmup", "scale": 2.0})
        pipeline._node_sample = {"peak": 6000, "reserved": 9000, "samples": 2, "seconds": 0.0, "next": 0.0}
        pipeline._end_node(source_bytes, True)  # a measured node: 6000 B live over the whole node, the larger peak
        self.assertEqual({key: pipeline.inflight_growth_bound()[key] for key in ("bytes", "basis", "scale")},
                         {"bytes": 6000, "basis": "online", "scale": 1.0})
        pause = pipeline.stats()["fitter_pause"]
        self.assertEqual((pause["warmup_peak_bytes"], pause["warmup_source_bytes"], pause["warmup_method"],
                          pause["online_peak_bytes"], pause["online_source_bytes"], pause["online_reserved_bytes"],
                          pause["peak_bytes"], pause["peak_basis"]),
                         (4000, source_bytes // 2, "history", 6000, source_bytes, 9000, 6000, "online"))
        from reelcache import archive_pipeline as ap
        demoted = SimpleNamespace(raw=SimpleNamespace(nbytes=3 * source_bytes, on_device=False, dropped=False),
                                  groups=[])
        self.assertEqual(ap._source_bytes(demoted), 3 * source_bytes)  # the tier's entry, not the empty groups
        with pipeline._cond:
            running = pipeline._gpu_item
            pipeline._gpu_item = demoted
            try:
                bound = pipeline.inflight_growth_bound()
            finally:
                pipeline._gpu_item = running
        self.assertEqual((bound["scale"], bound["upload"], bound["bytes"]), (3.0, 3 * source_bytes,
                                                                             18000 + 3 * source_bytes))
        release.set()
        self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))

    def test_the_online_peak_is_sampled_on_the_first_nodes_then_every_16th(self):
        """dev22j8 (item 1, live bytes): the first ``PEAK_MEASURE_FIRST`` nodes, then every ``PEAK_MEASURE_EVERY``-th,
        sample what the fitter's streams hold live (``fitter_memory``) at their step boundaries and blocked waits; the
        largest sample is kept with its node's source bytes."""
        from reelcache import archive_pipeline as ap
        _, pipeline = self.pipeline(name=f"online-{uuid.uuid4().hex[:8]}")
        readings = []

        def measure():
            readings.append((len(readings) + 1) << 10)
            return {"held": readings[-1], "allocated": readings[-1], "tier": 0, "reserved": 2 * readings[-1],
                    "fragments": 0, "segments": 1}

        def run(index):
            matrices, contexts = s6.chunk(index)
            pipeline.submit("main", f"chunk-{index:06d}", matrices, contexts)
            pipeline.drain("main", timeout=120)
            return sum(tensor.numel() * tensor.element_size() for tensor in matrices.values())

        with mock.patch.object(ap, "PEAK_SAMPLE_INTERVAL", 0.0), mock.patch.object(pipeline, "fitter_memory", measure):
            for index in range(ap.PEAK_MEASURE_FIRST):
                node_bytes = run(index)
            sampled = len(readings)
            self.assertGreater(sampled, ap.PEAK_MEASURE_FIRST)  # several samples per measured node
            run(ap.PEAK_MEASURE_FIRST)
            self.assertEqual(len(readings), sampled)  # the 4th node is not measured
            with pipeline._cond:
                pipeline._fitter_nodes = ap.PEAK_MEASURE_EVERY - 1
            run(ap.PEAK_MEASURE_FIRST + 1)  # the 16th node
            self.assertGreater(len(readings), sampled)
        pause = pipeline.stats()["fitter_pause"]
        self.assertEqual((pause["online_measurements"], pause["online_samples"], pause["online_peak_bytes"],
                          pause["online_source_bytes"], pause["online_reserved_bytes"], pause["nodes"]),
                         (4, len(readings), readings[-1], node_bytes, 2 * readings[-1], ap.PEAK_MEASURE_EVERY))
        self.assertEqual((pause["peak_bytes"], pause["peak_basis"], pause["measuring"]), (readings[-1], "online", False))
        self.assertIsNone(pipeline.fitter_memory())  # off CUDA: nothing to measure

    def test_a_pause_lands_while_the_fitter_waits_without_allocating_and_holds_at_the_wait_s_end(self):
        """dev22j8 (item 2): inside ``deferred_reads.blocked_wait`` the fitter allocates nothing, so a pause requested
        meanwhile is acknowledged at once (``landed`` = the wait's label); the wait's end is a step boundary, where the
        landed pause holds the node."""
        from reelcache.deferred_reads import blocked_wait
        release, waiting = threading.Event(), threading.Event()

        def hooks(stage, node_id):
            if stage == "plan":
                with blocked_wait("test_wait"):  # on the fitter thread, inside the node's GPU work
                    waiting.set()
                    self.assertTrue(release.wait(60))

        _, pipeline = self.pipeline(hooks=hooks, name=f"landed-{uuid.uuid4().hex[:8]}")
        self.addCleanup(release.set)
        matrices, contexts = s6.chunk(0)
        ticket = pipeline.submit("main", "chunk-000000", matrices, contexts)
        self.assertTrue(waiting.wait(60))
        self.assertEqual((pipeline.stats()["fitter_pause"]["phase"], pipeline.stats()["fitter_pause"]["blocked"]),
                         ("test_wait", True))
        with pipeline.pause_gpu_work(ack_timeout=30.0) as info:
            self.assertEqual((info["acked"], info["landed"], info["inflight"]), (True, "test_wait", True))
            self.assertLess(info["ack_seconds"], 5.0)
            release.set()  # the wait ends: the fitter stops at its end
            deadline = time.monotonic() + 60
            while not pipeline.stats()["fitter_pause"]["paused"] and time.monotonic() < deadline:
                time.sleep(0.01)
            pause = pipeline.stats()["fitter_pause"]
            self.assertEqual((pause["paused"], pause["blocked"], pause["phase"]), (True, False, "plan"))
            time.sleep(0.2)
            self.assertFalse(ticket.done())
        self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))
        pause = pipeline.stats()["fitter_pause"]
        self.assertEqual((pause["acked"], pause["landed"], pause["ack_timeouts"], pause["pauses"]), (1, 1, 0, 1))

    def test_the_rate_lag_dependency_wait_is_a_landed_state(self):
        """dev22j8 (item 2; 5b: every LongLive 2.0 pause timed out while the fitter held nothing): a node planning with
        an ancestor's measured rates (``rate_lag``; here 0: its parent's) waits for that ancestor's store without
        allocating -- a pause lands at once and holds the node when the rates arrive."""
        release, encoding = threading.Event(), threading.Event()

        def hooks(stage, node_id):
            if stage == "encode" and node_id == "chunk-000000":
                encoding.set()
                self.assertTrue(release.wait(60))

        _, pipeline = self.pipeline(hooks=hooks, name=f"rates-{uuid.uuid4().hex[:8]}")
        self.addCleanup(release.set)
        tickets = []
        for index in range(2):
            matrices, contexts = s6.chunk(index)
            tickets.append(pipeline.submit("main", f"chunk-{index:06d}", matrices, contexts))
        self.assertTrue(encoding.wait(60))
        self.wait_for_phase(pipeline, "rate_wait")
        with pipeline.pause_gpu_work(ack_timeout=30.0) as info:
            self.assertEqual((info["acked"], info["landed"]), (True, "rate_wait"))
            self.assertLess(info["ack_seconds"], 5.0)
            release.set()  # chunk 0 encodes and stores: its rates end chunk 1's wait, which then holds at the pause
            deadline = time.monotonic() + 60
            while not pipeline.stats()["fitter_pause"]["paused"] and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(pipeline.stats()["fitter_pause"]["paused"])
            self.assertFalse(tickets[1].done())
        for ticket in tickets:
            self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))

    def test_the_staging_slot_wait_is_a_landed_state(self):
        """dev22j8 (item 2): finalization waiting for a host staging slot (held by another node's encode) allocates
        no device memory: a pause lands at once."""
        release, encoding = threading.Event(), threading.Event()

        def hooks(stage, node_id):
            if stage == "encode" and node_id == "chunk-000000":
                encoding.set()
                self.assertTrue(release.wait(60))

        runtime = s6.RuntimeFactory(self)(f"slots-{uuid.uuid4().hex[:8]}")
        runtime.create_lineage("main")
        pipeline = ArchivePipeline(runtime, method=s6.method(rate_lag=1), registry=s6.registry(),
                                   options=ArchiveOptions(encode_threads=1), hooks=hooks, max_staged_canvases=1)
        self.addCleanup(lambda: pipeline.close(drain=False, timeout=60))
        self.addCleanup(release.set)
        tickets = []
        for index in range(2):
            matrices, contexts = s6.chunk(index)
            tickets.append(pipeline.submit("main", f"chunk-{index:06d}", matrices, contexts))
        self.assertTrue(encoding.wait(60))
        self.wait_for_phase(pipeline, "staging_wait")
        with pipeline.pause_gpu_work(ack_timeout=30.0) as info:
            self.assertEqual((info["acked"], info["landed"]), (True, "staging_wait"))
            release.set()
            deadline = time.monotonic() + 60
            while not pipeline.stats()["fitter_pause"]["paused"] and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(pipeline.stats()["fitter_pause"]["paused"])
        for ticket in tickets:
            self.assertTrue(ticket.result(timeout=120).startswith("manifest:"))

    @staticmethod
    def wait_for_phase(pipeline, phase, timeout=60.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pause = pipeline.stats()["fitter_pause"]
            if pause["phase"] == phase and pause["blocked"]:
                return
            time.sleep(0.01)
        raise AssertionError(f"the fitter never waited in {phase}: {pipeline.stats()['fitter_pause']['phase']}")

    def test_host_reads_are_blocked_waits_and_plan_has_its_step_boundaries_and_phases(self):
        """dev22j8 (item 2): ``read_host`` (the lockstep's batched reads) waits inside ``blocked_wait("host_read")``;
        ``CommitJob.plan`` waits for an ancestor's rates inside ``blocked_wait("rate_wait")`` and has step boundaries
        before and after the source fingerprints and after the version store, with its phases named. Recorded on a
        synchronous commit (the caller's thread runs the stages)."""
        from reelcache import deferred_reads as dr
        events = []

        @contextlib.contextmanager
        def blocked(label):
            events.append(("blocked", label))
            yield

        with dr.step_hook(lambda: events.append(("boundary",)), blocked=blocked,
                          phase=lambda label: events.append(("phase", label))):
            self.assertEqual(dr.read_host([(torch.tensor(2.0), torch.arange(3))])[0][0], 2.0)
            self.assertEqual(events, [("blocked", "host_read")])
            runtime = s6.RuntimeFactory(self)(f"boundaries-{uuid.uuid4().hex[:8]}")
            runtime.create_lineage("main")
            matrices, contexts = s6.chunk(0)
            runtime.commit_node("main", "chunk-000000", matrices, contexts, method=s6.method(),
                                registry=s6.registry(), options=ArchiveOptions(encode_threads=1))
        self.assertIn(("blocked", "rate_wait"), events)
        self.assertIn(("blocked", "host_read"), events[1:])
        phases = [event[1] for event in events if event[0] == "phase"]
        self.assertEqual([phase for phase in phases if not phase.startswith("planner:")][:3],
                         ["fingerprints", "planner", "versions"])
        self.assertTrue(any(phase.startswith("planner:") for phase in phases))  # the planner's stages, named
        start = events.index(("phase", "fingerprints"))
        self.assertEqual(events[start + 1], ("boundary",))  # before the fingerprints
        planner = events.index(("phase", "planner"))
        self.assertEqual(events[planner - 1], ("boundary",))  # after them
        versions = events.index(("phase", "versions"))
        self.assertEqual(events[versions + 1], ("boundary",))  # after the version store
        with dr.step_hook(None):  # no hooks: no-ops
            with dr.blocked_wait("nothing"):
                dr.step_phase("nothing")

    def test_the_warm_up_peak_is_the_live_high_water_without_the_synthetic_sources(self):
        """dev22j8 (coordinator 13:1x): the warm-up peak is the live (allocated) high-water of the fitter's streams
        minus the warm-up's own synthetic sources (in production the sources are the raw tier's); the reserved growth
        (dev22j7's measure: fragmentation and the sources included) is a diagnostic."""
        _, pipeline = self.pipeline(name=f"peak-{uuid.uuid4().hex[:8]}")
        readings = iter([9 << 20])
        with mock.patch.object(pipeline, "_reserved_bytes", lambda: next(readings)):
            pipeline._learn_fitter_peak(1 << 20, {"source_bytes": 3 << 20},
                                        SimpleNamespace(bytes=7 << 20, method="history"))
        stats = pipeline.stats()["fitter_pause"]
        self.assertEqual((stats["peak_bytes"], stats["peak_source_bytes"], stats["warmup_peak_bytes"],
                          stats["warmup_reserved_bytes"], stats["warmup_method"], stats["peak_basis"]),
                         (4 << 20, 3 << 20, 4 << 20, 8 << 20, "history", "warmup"))
        pipeline._learn_fitter_peak(None, {"source_bytes": 1 << 20}, SimpleNamespace(bytes=2 << 20,
                                                                                     method="global_peak"))
        self.assertEqual(pipeline.stats()["fitter_pause"]["peak_bytes"], 4 << 20)  # smaller: the larger peak stays
        # dev22j8v3: what the fitter's streams already held counts (absolute, like the online samples and ``held``)
        pipeline._learn_fitter_peak(None, {"source_bytes": 1 << 20}, SimpleNamespace(bytes=3 << 20, method="history"),
                                    baseline=5 << 20)
        stats = pipeline.stats()["fitter_pause"]
        self.assertEqual((stats["peak_bytes"], stats["warmup_baseline_bytes"], stats["warmup_peak_bytes"]),
                         (7 << 20, 5 << 20, 7 << 20))
        pipeline._learn_fitter_peak(None, {}, None)  # off CUDA: nothing learned
        pipeline._learn_fitter_peak(None, {"source_bytes": 1}, SimpleNamespace(bytes=None, method=None))
        self.assertEqual(pipeline.stats()["fitter_pause"]["peak_bytes"], 7 << 20)

    def test_the_warm_up_plans_and_finalizes_in_a_node_s_groups(self):
        """dev22j8 (bench 4444057: LongLive 2.0's warm-up planned all 60 matrices at once -- 13.5 GiB live -- while its
        nodes, under ``plan_memory_bytes`` = 4 GiB, peaked at 0.9 GiB): ``warm_commit_path`` plans and finalizes in
        the groups a node's ``CommitJob`` forms, so the learned peak is a node's."""
        from reelcache import lineage_commit as lc
        sizes, budgets = [], []
        finalize, planner = lc.finalize_tiles, lc.LatestMethodPlanner

        def recording_finalize(batch, **kwargs):
            sizes.append(len(batch))
            return finalize(batch, **kwargs)

        def recording_planner(*args, **kwargs):
            budgets.append(kwargs.get("max_working_bytes"))
            return planner(*args, **kwargs)

        matrices, contexts = s6.chunk(0)
        with mock.patch.object(lc, "finalize_tiles", recording_finalize), \
                mock.patch.object(lc, "LatestMethodPlanner", recording_planner):
            for budget in (None, 1 << 20):
                sizes.clear()
                budgets.clear()
                report = lc.warm_commit_path(s6.method(), s6.registry(), contexts,
                                             options=ArchiveOptions(encode_threads=1, plan_memory_bytes=budget))
                self.assertEqual(budgets, [budget, budget])  # both warm-up chunks plan with the node's bound
                expected = [len(contexts)] * 2 if budget is None else [1] * (2 * len(contexts))
                self.assertEqual(sizes, expected)  # a tiny budget: one matrix per finalization group
                self.assertEqual(report["keys"], len(contexts))


class AttachWarmupPeakTests(unittest.TestCase):
    """dev22j8 (item 1; 5b's F5.1 gate: ``fitter_pause.peak_bytes`` was 0 in every process, so every bound fell back
    to the contaminated working set): the attach warm-up (``LineageSessionArchive.warmup(profile)``, the bridge's
    call) runs the archive prewarm without the encode when the decode guard is active, so the fitter's peak is learned
    by default; ``archive=True`` keeps the opt-in with the encode; ``archive=False`` or an inactive guard skip it."""

    def archive(self, *, guard=True, defer=False, fitter=True, tier=True, pressure=True):
        import types
        calls = []
        pipeline = SimpleNamespace(prewarm=lambda contexts, encode=True: calls.append(encode) or {"source_bytes": 1})
        raw = SimpleNamespace(enabled=True, pressure_enabled=pressure) if tier else None
        archive = SimpleNamespace(staging_engine=None, restorer=None, pool=None, device=torch.device("cuda", 0),
                                  model=None, config=SimpleNamespace(defer_archival_during_decode=defer),
                                  pipeline=pipeline, fitter=object() if fitter else None, raw_tier=raw,
                                  decode_guard_=la._decode_guard_state(guard))
        archive._decode_guard_active = types.MethodType(la.LineageSessionArchive._decode_guard_active, archive)
        return archive, calls

    def warmup(self, stand_in, **kwargs):
        profile = SimpleNamespace(default_grid=lambda: (4, 6), chunk_frames=2)
        with mock.patch.object(la, "matrix_contexts", lambda *args, **kw: {"cond": "contexts"}), \
                mock.patch.object(la, "release_cached_blocks", lambda report, device: None):
            return la.LineageSessionArchive.warmup(stand_in, profile, **kwargs)

    def test_the_default_attach_warm_up_measures_the_fitter_when_the_guard_is_active(self):
        archive, calls = self.archive()
        report = self.warmup(archive)  # the bridge's call: archive=None
        self.assertEqual(calls, [False])  # the prewarm, without the encoder warm-up
        self.assertIn("archive", report["components"])
        for kwargs, expected in (({"archive": False}, []), ({"archive": True}, [True])):
            archive, calls = self.archive()
            self.warmup(archive, **kwargs)
            self.assertEqual(calls, expected, kwargs)
        for options in ({"guard": False}, {"defer": True}, {"fitter": False}, {"tier": False}, {"pressure": False}):
            archive, calls = self.archive(**options)
            report = self.warmup(archive)
            self.assertEqual(calls, [], options)  # the guard is not active: nothing to measure for it
            self.assertNotIn("archive", report["components"])


if __name__ == "__main__":
    unittest.main()
