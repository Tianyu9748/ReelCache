"""Step 7 genfirst: generation-first archival (``defer_archival_during_generation``, default off; user 2026-10-20).

A request's generation -- from its restore's return to the end of its VAE decode seam -- is a **generation hold** of
the archive pipeline: no new GPU archival work starts inside it; a node already on the GPU finishes first (the
request's ``archival_defer_wait``); fitting resumes when it ends. The starvation guard lets a node through inside a
hold when the unpublished raw backlog passes its fraction of the raw budget, when the oldest unpublished chunk passes
the age bound, or while a caller waits on archival (never-block); each such node is an override by reason.

* pipeline (in process, fakes: the Step 6 runtime and the pipeline's stage hooks as a fake clock of the fitter): off by
  default nothing exists; no fit starts inside a hold and publication equals an unheld run's; the node on the GPU
  finishes first and that wait is reported; the guard by backlog (a raw tier), by age and by a waiting caller (a full
  queue and a drain never deadlock); publication completes in the gaps between holds; holds nest; configuration;
* bridge (fresh processes, W1.3's F7 scenario and its SGLang-like variant with a VAE decode seam): off, the records
  equal freeze W1.4.1's, run live beside this tree in the same environment (as F7 runs W1.2); on, the hold spans
  [restore return, decode seam exit] and each request's summary carries ``archival_defer``; the wait for a node on the
  GPU is an exposed stall of source ``archival_defer_wait``, counted once.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import reelcache.archive_pipeline as archive_pipeline_module
from reelcache.archive_pipeline import ArchivePipeline
from reelcache.errors import ConfigurationError, StateError
from reelcache.lineage_commit import ArchiveOptions
from reelcache.integrations.sglang import generation_defer as gd
from reelcache.raw_tier import RawChunkTier
from tests import step6_support as s6
from tests.assets import asset
from tests.test_step7_defer_gpu_work import Recorder

ROOT = Path(__file__).resolve().parents[1]
#: The base freeze W1.4.1, run live beside this tree (the same machine and environment: its CPU set and the CPU
#: thread count, on which the CPU-fitted svd_hevc content depends, are the same on both sides): the freeze's copy under
#: the release's package names (freeze_w141 TREE_ID ddf3e6c8c6c2eeac, renamed), a test asset.
BASE_TREE_ID = "1271426c1cc86c55"


def base_tree() -> Path:
    return asset("frozen", "freeze_w141", "tree", override="REELCACHE_GENFIRST_BASE_TREE")


class _HoldCase(unittest.TestCase):
    """A real ArchivePipeline on the Step 6 runtime (fakes: the stage hooks as the fitter's clock)."""

    def setUp(self):
        self.runtimes = s6.RuntimeFactory(self)

    def pipeline(self, name, *, holds=True, backlog_fraction=0.5, max_age_seconds=30.0, inflight="wait", **kwargs):
        runtime = self.runtimes(name)
        runtime.create_lineage("main")
        options = dict(encode_workers=2, max_source_chunks=3, max_unpublished=4)
        options.update(kwargs)
        pipeline = ArchivePipeline(runtime, method=s6.method(1), registry=s6.registry(),
                                   options=ArchiveOptions(encode_threads=1), **options)
        self.addCleanup(self._close, pipeline)
        if holds:
            extra = {} if inflight == "wait" else {"inflight": inflight}  # genfirst3's option only when used
            pipeline.enable_generation_holds(backlog_fraction=backlog_fraction, max_age_seconds=max_age_seconds,
                                             **extra)
        return runtime, pipeline

    @staticmethod
    def _close(pipeline):
        try:
            pipeline.close(drain=False, timeout=60)
        except BaseException:  # noqa: BLE001 - tests assert failures themselves
            pass

    def reference(self, count):
        runtime, pipeline = self.pipeline("reference", holds=False)
        tickets = [pipeline.submit("main", f"chunk-{index:03d}", *s6.chunk(index)) for index in range(count)]
        pipeline.drain("main", timeout=300)
        return [s6.node_content(runtime, ticket.result()) for ticket in tickets]



class GenerationHoldTests(_HoldCase):
    def test_off_by_default_nothing_exists(self):
        runtime, pipeline = self.pipeline("off", holds=False)
        self.assertIsNone(pipeline.open_generation())
        self.assertIsNone(pipeline.generation_overrides())
        self.assertFalse(pipeline.close_generation(1))
        ticket = pipeline.submit("main", "chunk-000", *s6.chunk(0))
        pipeline.drain("main", timeout=300)
        self.assertTrue(ticket.done())
        self.assertNotIn("generation_defer", pipeline.stats())

    def test_no_fit_starts_inside_a_hold_and_publication_is_unchanged(self):
        expected = self.reference(3)
        recorder = Recorder()
        runtime, pipeline = self.pipeline("held", hooks=recorder)
        opened = pipeline.open_generation("generation")
        self.assertEqual((opened["inflight"], pipeline.stats()["generation_defer"]["open"]), (False, 1))
        tickets = [pipeline.submit("main", f"chunk-{index:03d}", *s6.chunk(index)) for index in range(3)]
        time.sleep(0.5)
        self.assertEqual(recorder.stage("plan"), [])  # nothing started inside the hold
        self.assertFalse(any(ticket.done() for ticket in tickets))
        closed_at = time.perf_counter()
        self.assertTrue(pipeline.close_generation(opened["token"]))
        deadline = time.monotonic() + 5
        while not recorder.stage("plan") and time.monotonic() < deadline:  # no waiter here: only the close wakes it
            time.sleep(0.005)
        pipeline.drain("main", timeout=300)
        plans = recorder.stage("plan")
        self.assertEqual([node for node, _ in plans], ["chunk-000", "chunk-001", "chunk-002"])
        self.assertTrue(all(at >= closed_at for _, at in plans))  # fitting resumed with the hold's end ...
        self.assertLess(plans[0][1] - closed_at, 0.3)  # ... at once (the close wakes the fitter)
        self.assertEqual([s6.node_content(runtime, ticket.result()) for ticket in tickets], expected)
        stats = pipeline.stats()["generation_defer"]
        self.assertEqual((stats["holds"], stats["open"], stats["overrides"]), (1, 0, {"backlog": 0, "age": 0,
                                                                                    "waiter": 0}))
        self.assertGreater(stats["held_seconds"], 0.4)
        self.assertEqual(stats["publications"], 3)
        self.assertIsNotNone(stats["publication_lag_p90"])

    def test_the_node_on_the_gpu_finishes_first(self):
        expected = self.reference(2)
        recorder = Recorder()
        entered, release = threading.Event(), threading.Event()
        recorder.blocks[("finalize", "chunk-000")] = (entered, release)
        runtime, pipeline = self.pipeline("in-flight", hooks=recorder)
        tickets = [pipeline.submit("main", f"chunk-{index:03d}", *s6.chunk(index)) for index in range(2)]
        self.assertTrue(entered.wait(60))  # chunk 0 is on the fitter
        result, opened = {}, threading.Event()

        def generation():
            result["started"] = time.perf_counter()
            result["hold"] = pipeline.open_generation("generation")
            result["opened"] = time.perf_counter()
            opened.set()
        worker = threading.Thread(target=generation)
        worker.start()
        time.sleep(0.4)
        self.assertFalse(opened.is_set())  # waits for the in-flight node
        released = time.perf_counter()
        release.set()
        self.assertTrue(opened.wait(60))
        worker.join(60)
        hold = result["hold"]
        self.assertTrue(hold["inflight"])
        self.assertGreaterEqual(result["opened"], released)
        self.assertGreaterEqual(hold["wait_seconds"], released - result["started"] - 0.05)
        time.sleep(0.4)
        self.assertEqual([node for node, _ in recorder.stage("plan")], ["chunk-000"])  # chunk 1 is held
        closed_at = time.perf_counter()
        pipeline.close_generation(hold["token"])
        pipeline.drain("main", timeout=300)
        plans = recorder.stage("plan")
        self.assertEqual([node for node, _ in plans], ["chunk-000", "chunk-001"])
        self.assertGreaterEqual(plans[1][1], closed_at)
        self.assertEqual([s6.node_content(runtime, ticket.result()) for ticket in tickets], expected)
        stats = pipeline.stats()["generation_defer"]
        self.assertEqual((stats["waits"], stats["inflight_waits"]), (1, 1))
        self.assertAlmostEqual(stats["wait_seconds"], hold["wait_seconds"], places=6)

    def test_the_guard_lets_a_node_through_by_backlog(self):
        """A raw tier: the unpublished raw bytes above the fraction of (device + host) budget start a node inside the
        hold (one ``backlog`` override); below it, nothing starts."""
        expected = self.reference(2)
        nbytes = sum(tensor.numel() * tensor.element_size() for tensor in s6.chunk(0)[0].values())
        budget = 64 * nbytes
        recorder = Recorder()
        tier = RawChunkTier(gpu_budget_bytes=budget // 2, host_budget_bytes=budget // 2, pinned=False)
        # one chunk's bytes are below the fraction, two are above it
        runtime, pipeline = self.pipeline("backlog", hooks=recorder, raw_tier=tier,
                                          backlog_fraction=1.5 * nbytes / budget)
        hold = pipeline.open_generation("generation")
        tickets = [pipeline.submit("main", "chunk-000", *s6.chunk(0))]
        time.sleep(0.4)
        self.assertEqual(recorder.stage("plan"), [])
        tickets.append(pipeline.submit("main", "chunk-001", *s6.chunk(1)))
        deadline = time.monotonic() + 60
        while not recorder.stage("plan") and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual([node for node, _ in recorder.stage("plan")][:1], ["chunk-000"])  # inside the hold
        stats = pipeline.stats()["generation_defer"]
        self.assertGreaterEqual(stats["overrides"]["backlog"], 1)
        self.assertGreaterEqual(stats["backlog_peak_bytes"], 2 * nbytes)
        self.assertEqual(stats["open"], 1)
        pipeline.close_generation(hold["token"])
        pipeline.drain("main", timeout=300)
        self.assertEqual([s6.node_content(runtime, ticket.result()) for ticket in tickets], expected)

    def test_the_guard_lets_a_node_through_by_age(self):
        recorder = Recorder()
        runtime, pipeline = self.pipeline("age", hooks=recorder, max_age_seconds=0.6)
        hold = pipeline.open_generation("generation")
        submitted = time.perf_counter()
        ticket = pipeline.submit("main", "chunk-000", *s6.chunk(0))
        time.sleep(0.3)
        self.assertEqual(recorder.stage("plan"), [])  # younger than the bound
        deadline = time.monotonic() + 60
        while not recorder.stage("plan") and time.monotonic() < deadline:
            time.sleep(0.02)
        plans = recorder.stage("plan")
        self.assertEqual([node for node, _ in plans], ["chunk-000"])
        self.assertGreaterEqual(plans[0][1] - submitted, 0.6)
        self.assertLess(plans[0][1] - submitted, 3.0)  # the fitter re-checks the age while it waits
        self.assertEqual(pipeline.stats()["generation_defer"]["overrides"]["age"], 1)
        pipeline.close_generation(hold["token"])
        pipeline.drain("main", timeout=300)
        self.assertTrue(ticket.done())

    def test_a_waiting_caller_never_deadlocks(self):
        """Never-block: a generation thread holding the pipeline submits into a full queue (backpressure) and drains;
        both wait on archival, so the guard lets the fitter through (``waiter``) instead of deadlocking."""
        expected = self.reference(3)
        runtime, pipeline = self.pipeline("waiter", max_source_chunks=1, max_unpublished=1)
        done, outcome = threading.Event(), {}

        def generation():
            try:
                hold = pipeline.open_generation("generation")
                tickets = [pipeline.submit("main", f"chunk-{index:03d}", *s6.chunk(index)) for index in range(3)]
                pipeline.drain("main", timeout=120)
                pipeline.close_generation(hold["token"])
                outcome["tickets"] = tickets
            except BaseException as exc:  # noqa: BLE001 - asserted below
                outcome["error"] = exc
            finally:
                done.set()
        worker = threading.Thread(target=generation, daemon=True)
        worker.start()
        self.assertTrue(done.wait(180), "a submit or drain inside a hold blocked (deadlock)")
        self.assertNotIn("error", outcome)
        self.assertEqual([s6.node_content(runtime, ticket.result()) for ticket in outcome["tickets"]], expected)
        stats = pipeline.stats()["generation_defer"]
        self.assertGreaterEqual(stats["overrides"]["waiter"], 1)
        self.assertEqual(stats["overrides"]["backlog"], 0)  # no raw tier: no backlog rule

    def test_publication_completes_in_the_gaps(self):
        """Closed-loop turns: each turn's chunk is submitted inside its hold and fitted and published in the gap that
        follows (gaps longer than one node's archival); nothing starts inside a hold."""
        recorder = Recorder()
        runtime, pipeline = self.pipeline("gaps", hooks=recorder)
        turns = []
        for index in range(3):
            hold = pipeline.open_generation("generation")
            opened = time.perf_counter()
            ticket = pipeline.submit("main", f"chunk-{index:03d}", *s6.chunk(index))
            time.sleep(0.2)  # the request's denoise and decode
            closed = time.perf_counter()
            pipeline.close_generation(hold["token"])
            ticket.result(timeout=120)  # published within the gap
            turns.append((opened, closed, time.perf_counter(), hold))
        plans = dict(recorder.stage("plan"))
        for index, (opened, closed, published, hold) in enumerate(turns):
            node = f"chunk-{index:03d}"
            self.assertGreaterEqual(plans[node], closed)  # fitted after its turn's generation
            self.assertFalse(hold["inflight"])  # the previous turn's node had finished in the gap
        stats = pipeline.stats()["generation_defer"]
        self.assertEqual((stats["holds"], stats["inflight_waits"], stats["publications"]), (3, 0, 3))
        self.assertEqual(stats["overrides"], {"backlog": 0, "age": 0, "waiter": 0})
        self.assertIsNotNone(stats["publication_lag_p50"])

    def test_holds_nest_and_configuration(self):
        runtime, pipeline = self.pipeline("nest")
        first, second = pipeline.open_generation("a"), pipeline.open_generation("b")
        self.assertEqual(pipeline.stats()["generation_defer"]["open"], 2)
        self.assertTrue(pipeline.close_generation(first["token"]))
        self.assertFalse(pipeline.close_generation(first["token"]))  # idempotent
        self.assertEqual(pipeline.stats()["generation_defer"]["open"], 1)
        self.assertTrue(pipeline.close_generation(second["token"]))
        for arguments in ({"backlog_fraction": 0}, {"backlog_fraction": True}, {"max_age_seconds": -1},
                          {"max_age_seconds": float("nan")}):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ConfigurationError):
                    pipeline.enable_generation_holds(**arguments)
        errors = []
        pipeline.hooks = lambda stage, node: errors.append(self._fitter_open(pipeline)) if stage == "plan" else None
        pipeline.submit("main", "chunk-000", *s6.chunk(0))
        pipeline.drain("main", timeout=300)
        self.assertEqual(errors, ["StateError"])

    @staticmethod
    def _fitter_open(pipeline):
        try:
            pipeline.open_generation("fitter")
            return None
        except StateError:
            return "StateError"


class ReviewFixTests(_HoldCase):
    """genfirst3: the W1.4.2 review's findings (rubber-duck f81ae08b), one test each (each fails on genfirst2), and
    the ``overlap`` mode of the node in flight when a hold opens (coordinator 2026-10-20 02:12)."""

    def test_the_opener_waits_only_for_the_node_on_the_gpu(self):
        """#4: the opener waits for the node on the GPU when its hold opened, not for a successor the guard started
        before the opener re-checked (white-box: the fitter's in-flight item is swapped under the condition, the
        interleaving that otherwise depends on the lock race; the fitter itself idles on an empty queue)."""
        runtime, pipeline = self.pipeline("successor")
        first, second = object(), object()
        with pipeline._cond:
            pipeline._gpu_busy, pipeline._gpu_item = True, first
        opened = {}
        thread = threading.Thread(target=lambda: opened.update(hold=pipeline.open_generation("generation")),
                                  daemon=True)
        thread.start()
        time.sleep(0.2)
        self.assertNotIn("hold", opened)  # it waits for the node in flight
        with pipeline._cond:  # that node finished and the guard started a successor before the opener re-checked
            pipeline._gpu_item = second
            pipeline._dispatch_id = getattr(pipeline, "_dispatch_id", 0) + 1  # a dispatch, as the fitter counts it
            pipeline._cond.notify_all()
        thread.join(2)
        try:
            self.assertIn("hold", opened, "the opener also waited for the successor")
            self.assertTrue(opened["hold"]["inflight"])
            self.assertEqual(pipeline.stats()["generation_defer"]["inflight_waits"], 1)
        finally:
            with pipeline._cond:
                pipeline._gpu_busy, pipeline._gpu_item = False, None
                pipeline._cond.notify_all()
            thread.join(10)
        pipeline.close_generation(opened["hold"]["token"])

    def test_a_timed_out_recheck_wait_rechecks_the_holds(self):
        """#3: a watched-section re-check wait that times out goes back through the hold gate: a hold opened
        meanwhile keeps the node queued."""
        recorder = Recorder()
        runtime, pipeline = self.pipeline("recheck", hooks=recorder)
        original = archive_pipeline_module.RECHECK_DEFER_TIMEOUT
        archive_pipeline_module.RECHECK_DEFER_TIMEOUT = 0.4
        self.addCleanup(setattr, archive_pipeline_module, "RECHECK_DEFER_TIMEOUT", original)
        calls = []

        def verdict(item):
            calls.append(getattr(item, "node_id", None))
            return {"event": "deferred_archival", "node_id": calls[-1]} if len(calls) == 1 else None
        pipeline._recheck_verdict = verdict
        with pipeline._cond:
            pipeline._watches.append(SimpleNamespace(events=[], reason="test"))
        ticket = pipeline.submit("main", "chunk-000", *s6.chunk(0))
        deadline = time.monotonic() + 30
        while not calls and time.monotonic() < deadline:
            time.sleep(0.01)
        hold = pipeline.open_generation("generation")  # opens during the re-check wait
        time.sleep(0.9)  # the re-check wait timed out at 0.4 s
        self.assertEqual(recorder.stage("plan"), [])
        with pipeline._cond:
            pipeline._watches.clear()
        pipeline.close_generation(hold["token"])
        ticket.result(timeout=120)
        self.assertEqual([node for node, _ in recorder.stage("plan")], ["chunk-000"])

    def test_a_failed_chunk_never_ages_the_guard(self):
        """#8: a failed chunk (its lineage failed) does not keep the age rule on: a young chunk stays held."""
        recorder = Recorder()

        def hooks(stage, node):
            recorder(stage, node)
            if stage == "plan" and node == "bad-000":
                raise RuntimeError("injected fit failure")
        runtime, pipeline = self.pipeline("aging", hooks=hooks, max_age_seconds=0.5)
        runtime.create_lineage("bad")
        bad = pipeline.submit("bad", "bad-000", *s6.chunk(0))
        with self.assertRaises(BaseException):
            bad.result(timeout=120)
        time.sleep(0.8)  # the failed chunk is older than the age bound now
        hold = pipeline.open_generation("generation")
        ticket = pipeline.submit("main", "chunk-000", *s6.chunk(0))
        time.sleep(0.3)  # the new chunk is younger than the bound
        self.assertEqual([node for node, _ in recorder.stage("plan") if node == "chunk-000"], [])
        self.assertEqual(pipeline.stats()["generation_defer"]["overrides"]["age"], 0)
        pipeline.close_generation(hold["token"])
        ticket.result(timeout=120)

    def test_the_age_clock_starts_at_capture(self):
        """#9: the age rule measures a chunk from its ``submit`` call (its capture), not from its admission after
        a backpressure wait."""
        recorder = Recorder()
        gate = (threading.Event(), threading.Event())
        recorder.blocks[("publish", "chunk-000")] = gate
        runtime, pipeline = self.pipeline("capture", hooks=recorder, max_source_chunks=1, max_unpublished=1)
        pipeline.submit("main", "chunk-000", *s6.chunk(0))
        self.assertTrue(gate[0].wait(60))  # chunk-000 is publishing: chunk-001's admission waits
        hold = pipeline.open_generation("generation")
        captured = {}

        def second():
            captured["called"] = time.perf_counter()
            captured["ticket"] = pipeline.submit("main", "chunk-001", *s6.chunk(1))
            captured["admitted"] = time.perf_counter()
        thread = threading.Thread(target=second, daemon=True)
        thread.start()
        time.sleep(0.5)
        gate[1].set()
        thread.join(60)
        self.assertGreaterEqual(captured["admitted"] - captured["called"], 0.4)
        with pipeline._cond:
            oldest = pipeline._generation.oldest(pipeline)
        self.assertIsNotNone(oldest)
        self.assertLess(abs(oldest - captured["called"]), 0.1)
        pipeline.close_generation(hold["token"])
        captured["ticket"].result(timeout=120)

    def test_a_new_waiter_wakes_a_held_fitter_at_once(self):
        """#10: a caller that starts waiting on archival wakes a fitter held only by a hold at once (the waiter
        rule), not at its next age poll (up to 1 s later)."""
        recorder = Recorder()
        runtime, pipeline = self.pipeline("wake", hooks=recorder)
        hold = pipeline.open_generation("generation")
        ticket = pipeline.submit("main", "chunk-000", *s6.chunk(0))
        time.sleep(0.15)  # the fitter sleeps in its held wait (age poll 1 s)
        started = time.perf_counter()
        waiter = threading.Thread(target=lambda: ticket.result(timeout=120), daemon=True)
        waiter.start()
        deadline = time.monotonic() + 30
        while not recorder.stage("plan") and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertLess(recorder.stage("plan")[0][1] - started, 0.4)
        waiter.join(120)
        stats = pipeline.stats()["generation_defer"]
        self.assertEqual(stats["overrides"]["waiter"], 1)
        pipeline.close_generation(hold["token"])

    def test_overrides_are_counted_per_hold(self):
        """#7: ``close_generation(token, report)`` reports the overrides counted while that hold was open."""
        runtime, pipeline = self.pipeline("pertoken")
        first = pipeline.open_generation("a")
        ticket = pipeline.submit("main", "chunk-000", *s6.chunk(0))
        ticket.result(timeout=120)  # a waiter: the guard lets the node through inside the hold
        report = {}
        self.assertTrue(pipeline.close_generation(first["token"], report))
        second = pipeline.open_generation("b")
        later = {}
        self.assertTrue(pipeline.close_generation(second["token"], later))
        self.assertEqual(report["overrides"], {"backlog": 0, "age": 0, "waiter": 1})
        self.assertEqual(later["overrides"], {"backlog": 0, "age": 0, "waiter": 0})
        self.assertFalse(report["inflight_overlap"])

    def test_one_hold_per_request_key_and_nested_decode_seams(self):
        """#7: concurrent openers of one request key reserve it before waiting (one hold, none leaked); a decode
        seam nested in another ends the hold only at the outermost exit."""
        recorder = Recorder()
        gate = (threading.Event(), threading.Event())
        recorder.blocks[("finalize", "chunk-000")] = gate
        runtime, pipeline = self.pipeline("reserve", hooks=recorder)
        config = SimpleNamespace(defer_archival_during_generation=True, archival_defer_backlog_fraction=0.5,
                                 archival_defer_max_age_seconds=30.0, archival_defer_inflight="wait")
        bridge = SimpleNamespace(config=config, archive=SimpleNamespace(pipeline=pipeline), _summaries={})
        pipeline.submit("main", "chunk-000", *s6.chunk(0))
        self.assertTrue(gate[0].wait(60))  # a node on the GPU: opening waits for it
        threads = [threading.Thread(target=gd.open_hold, args=(bridge, "req", "restore"), daemon=True)
                   for _ in range(2)]
        for thread in threads:
            thread.start()
        time.sleep(0.3)
        gate[1].set()
        for thread in threads:
            thread.join(60)
        self.assertEqual(pipeline.stats()["generation_defer"]["open"], 1)
        gd.close_hold(bridge, "req", "decode")
        self.assertEqual(pipeline.stats()["generation_defer"]["open"], 0)
        with gd._DecodeHold(bridge, "req2", contextlib.nullcontext()):
            with gd._DecodeHold(bridge, "req2", contextlib.nullcontext()):
                pass
            self.assertEqual(pipeline.stats()["generation_defer"]["open"], 1)  # the outer seam still holds
        self.assertEqual(pipeline.stats()["generation_defer"]["open"], 0)

    def test_the_guard_reports_which_bound_applies(self):
        """The ``guard`` record: no raw tier, an enabled raw tier, and a zero host budget (svd_hevc: the tier
        disables itself at the first submit) -- the backlog rule is inert then and never overrides."""
        nbytes = sum(tensor.numel() * tensor.element_size() for tensor in s6.chunk(0)[0].values())
        rules = []
        runtime, pipeline = self.pipeline("guard-none")
        rules.append(pipeline.stats()["generation_defer"]["guard"]["backlog_rule"])
        tier = RawChunkTier(gpu_budget_bytes=32 * nbytes, host_budget_bytes=32 * nbytes, pinned=False)
        runtime, pipeline = self.pipeline("guard-tier", raw_tier=tier)
        rules.append(pipeline.stats()["generation_defer"]["guard"]["backlog_rule"])
        self.assertEqual(rules, ["inert: no raw tier", "raw_tier"])
        zero = RawChunkTier(gpu_budget_bytes=2 * nbytes, host_budget_bytes=0, pinned=False)
        runtime, pipeline = self.pipeline("guard-zero", raw_tier=zero, backlog_fraction=0.01)
        hold = pipeline.open_generation("generation")
        tickets = [pipeline.submit("main", f"chunk-{index:03d}", *s6.chunk(index)) for index in range(2)]
        time.sleep(0.3)
        stats = pipeline.stats()["generation_defer"]
        self.assertEqual(stats["guard"]["backlog_rule"], "inert: raw tier disabled")
        self.assertIn("below one chunk", stats["guard"]["raw_tier_disabled_reason"])
        self.assertEqual((stats["guard"]["sources_held"], stats["overrides"]["backlog"]), (2, 0))
        pipeline.close_generation(hold["token"])
        pipeline.drain("main", timeout=300)
        self.assertTrue(all(ticket.done() for ticket in tickets))

    def test_the_inflight_option_is_validated_and_off_by_default(self):
        """``archival_defer_inflight``: ``"wait"`` (default; dropped from the record like the other genfirst
        defaults) or ``"overlap"``; anything else is refused."""
        from reelcache.integrations.sglang.config import ReelCacheRuntimeConfig
        base = {"store_root": tempfile.mkdtemp(prefix="genfirst_cfg_"), "representation": "svd_hevc"}
        self.assertEqual(ReelCacheRuntimeConfig.from_dict(base).archival_defer_inflight, "wait")
        self.assertNotIn("archival_defer_inflight", ReelCacheRuntimeConfig.from_dict(base).to_dict())
        overlap = ReelCacheRuntimeConfig.from_dict({**base, "archival_defer_inflight": "overlap"})
        self.assertEqual(overlap.to_dict()["archival_defer_inflight"], "overlap")
        for value in ("yield", None, True):
            with self.subTest(value=value):
                with self.assertRaises(ConfigurationError):
                    ReelCacheRuntimeConfig.from_dict({**base, "archival_defer_inflight": value})

    def test_overlap_mode_opens_at_once_and_reports_the_overlap(self):
        """``inflight="overlap"``: opening a hold does not wait for the node on the GPU (no wait, no inflight wait),
        its remainder runs beside the request and is reported when the hold closes; new nodes stay held."""
        recorder = Recorder()
        gate = (threading.Event(), threading.Event())
        recorder.blocks[("finalize", "chunk-000")] = gate
        runtime, pipeline = self.pipeline("overlap", hooks=recorder, inflight="overlap")
        tickets = [pipeline.submit("main", "chunk-000", *s6.chunk(0))]
        self.assertTrue(gate[0].wait(60))
        tickets.append(pipeline.submit("main", "chunk-001", *s6.chunk(1)))
        clock = time.perf_counter()
        hold = pipeline.open_generation("generation")
        self.assertLess(time.perf_counter() - clock, 0.2)
        self.assertEqual((hold["inflight"], hold["waited"]), (True, False))
        time.sleep(0.3)
        gate[1].set()  # the node in flight finishes 0.3 s into the hold
        time.sleep(0.5)
        self.assertEqual([node for node, _ in recorder.stage("plan")], ["chunk-000"])  # chunk-001 stays held
        report = {}
        pipeline.close_generation(hold["token"], report)
        self.assertTrue(report["inflight_overlap"])
        self.assertGreater(report["inflight_overlap_seconds"], 0.25)
        self.assertLess(report["inflight_overlap_seconds"], 0.75)
        stats = pipeline.stats()["generation_defer"]
        self.assertEqual((stats["inflight"], stats["inflight_waits"], stats["inflight_overlaps"]), ("overlap", 0, 1))
        pipeline.drain("main", timeout=300)
        self.assertTrue(all(ticket.done() for ticket in tickets))
        with self.assertRaises(ConfigurationError):
            pipeline.enable_generation_holds(inflight="yield")


class _Flicker:
    """A ``_waiters`` value whose truth is read once as True, then False: a waiter that leaves right after the gate
    saw it (white-box; the interleaving of review I3)."""

    def __init__(self):
        self.reads = 0

    def __bool__(self):
        self.reads += 1
        return self.reads == 1


class DeltaReviewFixTests(_HoldCase):
    """genfirst4: the W1.4.2.1 delta review's findings (f81ae08b round 2), one test each (each fails on genfirst3)."""

    def test_one_admission_decision_permits_and_accounts(self):
        """I3: the waiter that lets a node through a hold is the one counted -- a waiter seen by the gate but gone
        at the decision holds the node back (no fit inside a hold without an override)."""
        recorder = Recorder()
        runtime, pipeline = self.pipeline("atomic", hooks=recorder)
        hold = pipeline.open_generation("generation")
        ticket = pipeline.submit("main", "chunk-000", *s6.chunk(0))
        time.sleep(0.2)  # held: the fitter waits
        flicker = _Flicker()
        with pipeline._cond:
            pipeline._waiters = flicker
            pipeline._cond.notify_all()
        time.sleep(0.5)
        with pipeline._cond:
            pipeline._waiters = 0
        plans = len(recorder.stage("plan"))
        overrides = sum(pipeline.stats()["generation_defer"]["overrides"].values())
        self.assertGreaterEqual(flicker.reads, 2)  # the fitter re-read the predicate after the gate let it through
        self.assertLessEqual(plans, overrides)  # every node started inside the hold is an override
        pipeline.close_generation(hold["token"])
        ticket.result(timeout=120)

    def test_a_held_fitter_waits_instead_of_spinning(self):
        """I3's decision is the last word, the gate's wait keeps the fitter asleep while a hold blocks its head: few
        admission decisions per held second (a gate without the hold would spin on the decision)."""
        recorder = Recorder()
        runtime, pipeline = self.pipeline("spin", hooks=recorder)
        holds = pipeline._generation
        calls = []
        original = holds.admission
        holds.admission = lambda owner: calls.append(1) or original(owner)
        hold = pipeline.open_generation("generation")
        ticket = pipeline.submit("main", "chunk-000", *s6.chunk(0))
        time.sleep(0.6)
        self.assertEqual(recorder.stage("plan"), [])
        self.assertLess(len(calls), 20)
        pipeline.close_generation(hold["token"])
        ticket.result(timeout=120)

    def test_a_restarted_node_is_a_new_dispatch(self):
        """I1: the same node handed back and restarted is a new dispatch -- the opener waits for the dispatch on the
        GPU when its hold opened only (white-box: the restart is made under the condition)."""
        runtime, pipeline = self.pipeline("restart")
        node = object()
        with pipeline._cond:
            pipeline._gpu_busy, pipeline._gpu_item = True, node
        opened = {}
        thread = threading.Thread(target=lambda: opened.update(hold=pipeline.open_generation("generation")),
                                  daemon=True)
        thread.start()
        time.sleep(0.2)
        self.assertNotIn("hold", opened)
        with pipeline._cond:  # handed back and restarted -- the same object -- before the opener re-checked
            pipeline._gpu_busy, pipeline._gpu_item = True, node
            pipeline._dispatch_id = getattr(pipeline, "_dispatch_id", 0) + 1
            pipeline._cond.notify_all()
        thread.join(2)
        try:
            self.assertIn("hold", opened, "the opener also waited for the restarted node")
        finally:
            with pipeline._cond:
                pipeline._gpu_busy, pipeline._gpu_item = False, None
                pipeline._cond.notify_all()
            thread.join(10)
        pipeline.close_generation(opened["hold"]["token"])
        before = pipeline._dispatch_id
        tickets = [pipeline.submit("main", f"chunk-{index:03d}", *s6.chunk(index)) for index in range(2)]
        pipeline.drain("main", timeout=300)
        self.assertTrue(all(ticket.done() for ticket in tickets))
        self.assertEqual(pipeline._dispatch_id - before, 2)  # the fitter counts every dispatch (holds on)

    def test_the_overlap_excludes_the_lock_entry(self):
        """M5: a hold's overlap starts once the opener holds the condition, not when it asked for it."""
        recorder = Recorder()
        gate = (threading.Event(), threading.Event())
        recorder.blocks[("finalize", "chunk-000")] = gate
        runtime, pipeline = self.pipeline("lockentry", hooks=recorder, inflight="overlap")
        ticket = pipeline.submit("main", "chunk-000", *s6.chunk(0))
        self.assertTrue(gate[0].wait(60))
        ready = threading.Event()

        def holder():
            with pipeline._cond:
                ready.set()
                time.sleep(0.4)
        thread = threading.Thread(target=holder, daemon=True)
        thread.start()
        self.assertTrue(ready.wait(10))
        hold = pipeline.open_generation("generation")  # waits ~0.4 s for the condition
        gate[1].set()  # the node in flight finishes right after the hold opened
        time.sleep(0.3)
        report = {}
        pipeline.close_generation(hold["token"], report)
        self.assertTrue(report["inflight_overlap"])
        self.assertLess(report["inflight_overlap_seconds"], 0.25)
        ticket.result(timeout=120)

    def test_the_config_documents_the_hold_as_t2_to_vae_end(self):
        """M7: the configuration's documentation of the hold says [t2, vae_end]."""
        from reelcache.integrations.sglang.config import ReelCacheRuntimeConfig
        doc = " ".join((ReelCacheRuntimeConfig.__doc__ or "").split())
        section = doc[doc.index("``defer_archival_during_generation``"):][:400]
        self.assertIn("[t2, vae_end]", section)
        self.assertNotIn("[t2, t3]", section)

    def test_c3_backpressure_at_host_raw_zero_is_a_waiter_override(self):
        """M6: at a zero host raw budget (svd_hevc: the raw tier disables itself) with max_source_chunks 2, a third
        chunk's submit waits on the C3 bounds; that waiter lets a fit through the hold (no backlog rule)."""
        nbytes = sum(tensor.numel() * tensor.element_size() for tensor in s6.chunk(0)[0].values())
        tier = RawChunkTier(gpu_budget_bytes=2 * nbytes, host_budget_bytes=0, pinned=False)
        recorder = Recorder()
        runtime, pipeline = self.pipeline("c3", hooks=recorder, raw_tier=tier, max_source_chunks=2,
                                          max_unpublished=4)
        hold = pipeline.open_generation("generation")
        tickets = [pipeline.submit("main", f"chunk-{index:03d}", *s6.chunk(index)) for index in range(2)]
        time.sleep(0.2)
        self.assertEqual(recorder.stage("plan"), [])  # both held
        third = {}
        thread = threading.Thread(target=lambda: third.update(ticket=pipeline.submit("main", "chunk-002",
                                                                                     *s6.chunk(2))), daemon=True)
        thread.start()
        thread.join(120)
        self.assertIn("ticket", third)  # never blocked for good: the waiter rule let a fit through
        admission = third["ticket"].admission
        self.assertEqual((admission["waited"], admission["cause"], admission["raw_tier"]), (True, "c3_bounds", False))
        stats = pipeline.stats()
        defer = stats["generation_defer"]
        self.assertEqual(stats["peak_sources_held"], 2)
        self.assertEqual(defer["overrides"]["backlog"], 0)
        self.assertGreaterEqual(defer["overrides"]["waiter"], 1)
        self.assertEqual(defer["guard"]["backlog_rule"], "inert: raw tier disabled")
        pipeline.close_generation(hold["token"])
        pipeline.drain("main", timeout=300)
        self.assertTrue(all(ticket.done() for ticket in tickets + [third["ticket"]]))

    def test_a_same_key_opener_joins_the_wait(self):
        """I2: a second opener of the same request key (wait mode) joins the first one's wait for the node on the
        GPU instead of returning ahead of it."""
        recorder = Recorder()
        gate = (threading.Event(), threading.Event())
        recorder.blocks[("finalize", "chunk-000")] = gate
        runtime, pipeline = self.pipeline("join", hooks=recorder)
        config = SimpleNamespace(defer_archival_during_generation=True, archival_defer_backlog_fraction=0.5,
                                 archival_defer_max_age_seconds=30.0, archival_defer_inflight="wait")
        bridge = SimpleNamespace(config=config, archive=SimpleNamespace(pipeline=pipeline), _summaries={})
        pipeline.submit("main", "chunk-000", *s6.chunk(0))
        self.assertTrue(gate[0].wait(60))
        returned = []

        def opener():
            gd.open_hold(bridge, "req", "restore")
            returned.append(time.perf_counter())
        threads = [threading.Thread(target=opener, daemon=True) for _ in range(2)]
        for thread in threads:
            thread.start()
        time.sleep(0.3)
        self.assertEqual(returned, [])  # neither returns ahead of the node on the GPU
        released = time.perf_counter()
        gate[1].set()
        for thread in threads:
            thread.join(60)
        self.assertEqual(len(returned), 2)
        self.assertTrue(all(at >= released for at in returned))
        self.assertEqual(pipeline.stats()["generation_defer"]["open"], 1)
        gd.close_hold(bridge, "req", "decode")
        self.assertEqual(pipeline.stats()["generation_defer"]["open"], 0)


# -- the bridge: W1.3's F7 scenario, and its SGLang-like variant with a VAE decode seam --------------------------------
#: The F7 scenario's requests, each followed by the worker's VAE decode seam (as SGLang's decoding stage calls it).
SEAM_ANCHOR = "    summary = run_tree(bridge, req)\n"
SEAM = '    with bridge.memory_critical_section("vae_decode", request_id=req.request_id):\n        pass\n'


def scenario(seam: bool) -> str:
    from tests.test_wl_detach import F7_SCENARIO
    if not seam:
        return F7_SCENARIO
    if F7_SCENARIO.count(SEAM_ANCHOR) != 1:
        raise AssertionError("W1.4.1's F7 scenario changed: no single request line to follow with a decode seam")
    return F7_SCENARIO.replace(SEAM_ANCHOR, SEAM_ANCHOR + SEAM)


def run_scenario(seam: bool, extra: dict, *, tree: Path = ROOT, workload: bool = False) -> dict:
    """The scenario in a fresh process (svd_hevc: the archive pipeline encodes); its output record."""
    work = Path(tempfile.mkdtemp(prefix="genfirst_", dir=os.environ.get("TMPDIR")))
    spec = {"root": str(work / "store"), "representation": "svd_hevc",
            "extra": {**extra, **({"workload_telemetry": True} if workload else {})}, "out": str(work / "out.json")}
    code = scenario(seam)
    env = {key: value for key, value in os.environ.items() if not key.startswith("REELCACHE_PERF")}
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    if "REELCACHE_EXPECT_TREE" in env:
        env["REELCACHE_EXPECT_TREE"] = str(tree)
    result = subprocess.run([sys.executable, "-W", "ignore", "-c", code, json.dumps(spec)], cwd=tree, env=env,
                            capture_output=True, text=True, timeout=1800)
    if result.returncode:
        raise AssertionError(result.stderr[-3000:])
    record = json.load(open(spec["out"]))
    record["_root"] = spec["root"]
    return record


def normalized(record: dict) -> dict:
    """F7's normalization (floats and the store root masked; the test's TIMING_DEPENDENT step-span buckets; each
    restore window's garbage-collection count, allocation telemetry)."""
    from tests.test_wl_detach import TIMING_DEPENDENT, _flatten

    gc_count = re.compile(r"\.restore\.gc\.collections\[\d+\]$")
    root = record.pop("_root")
    flat = {}
    _flatten(record, "", flat)
    return {path: ("<gc>" if gc_count.search(path) else "<timing>" if TIMING_DEPENDENT.search(path)
                   else "<float>" if isinstance(value, float)
                   else value.replace(root, "<root>") if isinstance(value, str) else value)
            for path, value in flat.items()}


class BridgeGenerationFirstTests(unittest.TestCase):
    def test_default_off_records_equal_freeze_w141(self):
        """Off (the default, and explicitly False): W1.3's F7 scenario and its decode-seam variant give the records
        freeze W1.4.1 gives, run live beside this tree in fresh processes (as F7 runs W1.2), field for field."""
        base = base_tree()
        self.assertTrue((base / "reelcache").is_dir(), f"the W1.4.1 tree is missing: {base}")
        if "REELCACHE_GENFIRST_BASE_TREE" not in os.environ:
            self.assertEqual((base.parent / "TREE_ID").read_text().strip(), BASE_TREE_ID)
        for seam, variants in ((False, ({},)), (True, ({}, {"defer_archival_during_generation": False}))):
            expected = normalized(run_scenario(seam, {}, tree=base))
            for extra in variants:
                with self.subTest(seam=seam, extra=extra):
                    actual = normalized(run_scenario(seam, extra))
                    self.assertEqual(sorted(set(actual) ^ set(expected)), [])
                    self.assertEqual({path: [expected[path], actual[path]] for path in expected
                                      if expected[path] != actual[path]}, {})

    def test_on_the_hold_spans_restore_return_to_the_decode_seam(self):
        """On: each request holds from its restore's return to its decode seam's exit (the worker's first request
        too: genfirst3); the records' content equals the default run's; the per-request ``archival_defer`` and the
        pipeline's ``generation_defer`` are recorded."""
        on = run_scenario(True, {"defer_archival_during_generation": True}, workload=True)
        default = normalized(run_scenario(True, {}, workload=True))
        stats = on["archive_stats"]["pipeline"]["generation_defer"]
        letters = sorted(on["requests"])
        self.assertEqual(stats["open"], 0)
        self.assertGreaterEqual(stats["holds"], len(letters))
        records = {letter: on["requests"][letter]["summary"]["archival_defer"] for letter in letters}
        for letter in letters:
            self.assertEqual((records[letter]["opened_at"], records[letter]["closed_by"]), (["restore"], "decode"))
        content = {path: value for path, value in normalized(on).items()
                   if not re.search(r"generation_defer|archival_defer|\.workload\b|\.workload\.|"
                                    r"defer_archival_during_generation", path)}
        differing = sorted(path for path in set(content) & set(default) if content[path] != default[path])
        # timing only: the hold moves fits into the gaps (fitter step spans, publication order is unchanged)
        self.assertEqual([path for path in differing if not re.search(r"fitter_pause|backpressure|waits?\b", path)],
                         [])
        refs = [path for path in default if path.startswith(".refs") or ".commits[" in path and ".node" in path]
        self.assertTrue(refs)
        self.assertEqual({path: content.get(path) for path in refs}, {path: default[path] for path in refs})

    def test_the_restore_is_never_held(self):
        """Every restore of the F7 scenario (resident and archive restores, a rewind's) runs with no hold open; the
        hold opens after it returns."""
        from tests.test_wl_detach import F7_SCENARIO
        probe = textwrap.dedent("""
            import reelcache.integrations.sglang.bridge as _B
            _OPEN = []
            for _name in ("_restore_archive", "_restore_resident"):
                def _wrap(original):
                    def inside(self, *args, **kwargs):
                        stats = self.bridge.archive.pipeline.stats().get("generation_defer") or {}
                        _OPEN.append([self.key, stats.get("open", 0)])
                        return original(self, *args, **kwargs)
                    return inside
                setattr(_B._BridgeRun, _name, _wrap(getattr(_B._BridgeRun, _name)))
        """)
        code = probe + scenario(True) + "\nprint('OPEN ' + json.dumps(_OPEN))\n"
        work = Path(tempfile.mkdtemp(prefix="genfirst_restore_", dir=os.environ.get("TMPDIR")))
        spec = {"root": str(work / "store"), "representation": "svd_hevc",
                "extra": {"defer_archival_during_generation": True}, "out": str(work / "out.json")}
        env = {key: value for key, value in os.environ.items() if not key.startswith("REELCACHE_PERF")}
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-W", "ignore", "-c", code, json.dumps(spec)], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=1800)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        opened = json.loads(next(line for line in result.stdout.splitlines() if line.startswith("OPEN "))[5:])
        self.assertGreaterEqual(len(opened), 4)  # B..F restore history (resident or archive)
        self.assertEqual([count for _, count in opened], [0] * len(opened))
        self.assertTrue(F7_SCENARIO)

    def test_the_wait_for_a_node_on_the_gpu_is_an_exposed_stall(self):
        """A request whose restore returns while a node is on the fitter waits for it: one exposed stall of source
        ``archival_defer_wait`` (kind host_wait), counted once, and no span (the CONTRACT's closed span list is
        unchanged); the summary's wait matches."""
        code = REPORT_STALL
        result = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=ROOT,
                                env={**{key: value for key, value in os.environ.items()
                                        if not key.startswith("REELCACHE_PERF")},
                                     "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                                capture_output=True, text=True, timeout=900)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        data = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(data["sources"], ["archival_defer_wait"])
        self.assertEqual(data["kinds"], ["host_wait"])
        self.assertEqual(data["spans"], 0)
        self.assertGreaterEqual(data["record"]["wait_seconds"], 0.3)
        self.assertEqual(data["record"]["inflight_waits"], 1)

    def test_overlap_mode_records_the_overlap_not_a_stall(self):
        """``archival_defer_inflight="overlap"``: the same run opens request B's hold at once beside A's node on the
        fitter: no ``archival_defer_wait`` stall, B's record counts the overlap (``inflight_overlap_seconds``)."""
        result = subprocess.run([sys.executable, "-W", "ignore", "-c", REPORT_STALL, "overlap"], cwd=ROOT,
                                env={**{key: value for key, value in os.environ.items()
                                        if not key.startswith("REELCACHE_PERF")},
                                     "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                                capture_output=True, text=True, timeout=900)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        data = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(data["sources"], [])
        record = data["record"]
        self.assertEqual((record["inflight_waits"], record["wait_seconds"], record["inflight_overlaps"]), (0, 0.0, 1))
        self.assertGreater(record["inflight_overlap_seconds"], 0.0)

    def test_lifecycle_refusals_keep_the_running_hold_and_a_raising_finish_closes_its_own(self):
        """#6: ``begin_request`` / ``close`` refused while a request runs leave its hold open; #5: a ``finish`` that
        raises still closes its request's hold."""
        result = subprocess.run([sys.executable, "-W", "ignore", "-c", LIFECYCLE], cwd=ROOT,
                                env={**{key: value for key, value in os.environ.items()
                                        if not key.startswith("REELCACHE_PERF")},
                                     "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                                capture_output=True, text=True, timeout=900)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        data = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(data["refused"], ["PendingError", "PendingError"])
        self.assertEqual(data["open_after_refusals"], 1)
        self.assertEqual(data["open_before_finish"], 1)
        self.assertIsNotNone(data["finish_error"])
        self.assertEqual(data["open_after_failed_finish"], 0)
        record = data["failed_record"]  # genfirst4 (M4): on the run's own summary
        self.assertIsNotNone(record)
        self.assertEqual((record["holds"], record["opened_at"], record["closed_by"]), (1, ["restore"], "failure"))

    def test_a_workers_first_request_holds_until_its_decode_seam(self):
        """genfirst3: a worker's first request keeps its hold past its finish until its decode seam, so its own
        chunk's fit starts only after the seam (no in-flight wait at the seam; genfirst2 closed that hold at finish).
        A worker without decode seams (a CPU harness): that first hold closes when the next request begins and every
        later hold closes at its finish; a caller waiting on archival inside the open first hold is let through
        (never-block)."""
        def run(mode):
            result = subprocess.run([sys.executable, "-W", "ignore", "-c", FIRST_REQUEST, mode], cwd=ROOT,
                                    env={**{key: value for key, value in os.environ.items()
                                            if not key.startswith("REELCACHE_PERF")},
                                         "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                                    capture_output=True, text=True, timeout=900)
            self.assertEqual(result.returncode, 0, result.stderr[-3000:])
            return json.loads(result.stdout.strip().splitlines()[-1])

        with self.subTest(mode="seam"):
            data = run("seam")
            first, second = data["records"]["gf-A"], data["records"]["gf-B"]
            self.assertEqual((first["opened_at"], first["closed_by"], first["inflight_waits"]),
                             (["restore"], "decode", 0))
            self.assertEqual((second["opened_at"], second["closed_by"]), (["restore"], "decode"))
            self.assertTrue(data["plans"])
            self.assertGreaterEqual(min(at for _, at in data["plans"]), data["marks"]["A_seam_exit"])
            self.assertEqual(data["stats"]["open"], 0)
            self.assertEqual(data["stats"]["inflight_waits"], 0)
        with self.subTest(mode="seamless"):
            data = run("seamless")
            first, second = data["records"]["gf-A"], data["records"]["gf-B"]
            self.assertEqual((first["closed_by"], second["closed_by"]), ("next_request", "finish"))
            self.assertGreaterEqual(min(at for _, at in data["plans"]), data["marks"]["B_begin"])
            self.assertEqual(data["stats"]["open"], 0)
        with self.subTest(mode="seamless_wait"):
            data = run("seamless_wait")
            first = data["records"]["gf-A"]
            self.assertEqual(first["closed_by"], "next_request")
            self.assertGreaterEqual(first["overrides"]["waiter"], 1)
            self.assertTrue(data["published_A"])


#: A bridge run whose fitter is held on a node (a pipeline hook) while the second request's restore returns: the worker
#: has seen a decode seam (so a request's hold lasts until its seam); request A's node is released 0.4 s into the hold
#: of request B (a new session, whose restore depends on nothing).
REPORT_STALL = textwrap.dedent("""
    import json, sys, tempfile, threading, time
    import tests.test_wl_tree as t
    from reelcache.integrations.sglang import CausalWanKVBridge, ReelCacheRuntimeConfig
    mode = sys.argv[1] if len(sys.argv) > 1 else "wait"
    root = tempfile.mkdtemp(prefix="genfirst_stall_")
    values = {"store_root": root, "representation": "svd_hevc", "fit_device": "cpu", "decoder": "software",
              "energy_key": 0.98, "energy_value": 0.95, "x265_threads": 2, "encode_workers": 2,
              "tree_retention": "full", "host_raw_budget_bytes": 1 << 30, "workload_telemetry": True,
              "defer_archival_during_generation": True,
              **({} if mode == "wait" else {"archival_defer_inflight": mode})}
    bridge = CausalWanKVBridge(ReelCacheRuntimeConfig.from_dict(values), t.PROFILE, device="cpu",
                               model_path="toy/causal-wan")
    pipeline = bridge.archive.pipeline
    gate = {"block": False, "entered": threading.Event(), "release": threading.Event()}

    def hooks(stage, node):
        if stage == "finalize" and gate["block"]:
            gate["block"] = False
            gate["entered"].set()
            gate["release"].wait(60)
    pipeline.hooks = hooks
    with bridge.memory_critical_section("vae_decode", request_id="warm-up"):  # the worker's seams are in use
        pass
    session = t.handle("s", "generate", pause_after_chunks=1, durable=False)
    req = t.tree_request(session, [t.PROMPTS["A"]] * t.SESSION_TOTAL, generation_index=1)
    req.request_id = "gf-A"
    gate["block"] = True  # A's node, which starts when A's hold closes
    t.run_tree(bridge, req)
    with bridge.memory_critical_section("vae_decode", request_id="gf-A"):
        pass
    assert gate["entered"].wait(60), "the first chunk never reached the fitter"
    threading.Timer(0.4, gate["release"].set).start()
    session = t.handle("s2", "generate", pause_after_chunks=1, durable=False)  # a new session: restores nothing
    req = t.tree_request(session, [t.PROMPTS["B"]] * t.SESSION_TOTAL, generation_index=1)
    req.request_id = "gf-B"
    t.run_tree(bridge, req)
    with bridge.memory_critical_section("vae_decode", request_id="gf-B"):
        pass
    summary = bridge.request_summary("gf-B")
    workload = summary["workload"]
    stalls = [item for item in workload.get("stalls", []) if item.get("source") == "archival_defer_wait"]
    spans = [item for item in workload["spans"] if item["name"] == "archival_defer_wait"]
    print(json.dumps({"sources": [item["source"] for item in stalls], "kinds": [item["kind"] for item in stalls],
                      "spans": len(spans), "record": summary["archival_defer"],
                      "restore": (summary.get("restore") or {}).get("mode")}, default=str))
    bridge.close()
""")


#: genfirst3 #5 / #6: a hold of a running request (its key opened directly; ``_active`` set as begin_request would) and
#: the lifecycle calls a running request refuses; then a request whose finish raises (``_remember`` fails).
LIFECYCLE = textwrap.dedent("""
    import json, tempfile
    from types import SimpleNamespace
    import torch
    import tests.test_wl_tree as t
    from reelcache.integrations.sglang import CausalWanKVBridge, ReelCacheRuntimeConfig
    from reelcache.integrations.sglang import generation_defer as gd
    root = tempfile.mkdtemp(prefix="genfirst_life_")
    values = {"store_root": root, "representation": "svd_hevc", "fit_device": "cpu", "decoder": "software",
              "energy_key": 0.98, "energy_value": 0.95, "x265_threads": 2, "encode_workers": 2,
              "tree_retention": "full", "host_raw_budget_bytes": 1 << 30, "defer_archival_during_generation": True}
    bridge = CausalWanKVBridge(ReelCacheRuntimeConfig.from_dict(values), t.PROFILE, device="cpu",
                               model_path="toy/causal-wan")
    pipeline = bridge.archive.pipeline
    out = {"refused": []}
    gd.open_hold(bridge, "gf-running", "restore")
    bridge._active = object()  # a request runs on this worker
    session = t.handle("s", "generate", pause_after_chunks=1, durable=False)
    req = t.tree_request(session, [t.PROMPTS["A"]] * t.SESSION_TOTAL, generation_index=1)
    req.request_id = "gf-A"
    for call in (lambda: bridge.begin_request(req), bridge.close):
        try:
            call()
            out["refused"].append(None)
        except Exception as exc:  # noqa: BLE001
            out["refused"].append(type(exc).__name__)
    out["open_after_refusals"] = pipeline.stats()["generation_defer"]["open"]
    bridge._active = None
    gd.close_hold(bridge, "gf-running", "test")

    def failing(*args, **kwargs):
        raise RuntimeError("injected _remember failure")
    # run_tree's own retry (finish(error=...)) would close the hold through the normal path: one finish call here
    run = bridge.begin_request(req)
    target = SimpleNamespace(caches={stream: [t.GuardedCache() for _ in range(t.LAYERS)] for stream in req.streams},
                             latents=torch.zeros(1, 4, sum(req.block_sizes), *t.GRID),
                             generator=torch.Generator().manual_seed(req.conditioning["seed"]),
                             rope_cos_sin=t.rope_cos_sin, rotate_key=t.rotate_key, device=torch.device("cpu"))
    run.restore(target)  # its hold opens
    out["open_before_finish"] = pipeline.stats()["generation_defer"]["open"]
    bridge._remember = failing
    try:
        run.finish(error=None)
        out["finish_error"] = None
    except Exception as exc:  # noqa: BLE001
        out["finish_error"] = type(exc).__name__
    out["open_after_failed_finish"] = pipeline.stats()["generation_defer"]["open"]
    summary = getattr(run, "summary", None)
    out["failed_record"] = summary.get("archival_defer") if isinstance(summary, dict) else None
    del bridge._remember
    bridge._active = None  # the failed finish did not release the worker
    try:
        bridge.close()
    except Exception as exc:  # noqa: BLE001
        out["close_error"] = repr(exc)
    print(json.dumps(out))
""")


#: genfirst3: requests A and B (new sessions) on a fresh worker, with (``seam``) or without (``seamless``,
#: ``seamless_wait``: A's commit reports are awaited inside its open first hold) a VAE decode seam after each; the
#: pipeline's ``plan`` hook times each chunk's fit start; prints the marks, the fit starts and both requests' records.
FIRST_REQUEST = textwrap.dedent("""
    import json, sys, tempfile, time
    import tests.test_wl_tree as t
    from reelcache.integrations.sglang import CausalWanKVBridge, ReelCacheRuntimeConfig
    mode = sys.argv[1]
    root = tempfile.mkdtemp(prefix="genfirst_first_")
    values = {"store_root": root, "representation": "svd_hevc", "fit_device": "cpu", "decoder": "software",
              "energy_key": 0.98, "energy_value": 0.95, "x265_threads": 2, "encode_workers": 2,
              "tree_retention": "full", "host_raw_budget_bytes": 1 << 30, "defer_archival_during_generation": True}
    bridge = CausalWanKVBridge(ReelCacheRuntimeConfig.from_dict(values), t.PROFILE, device="cpu",
                               model_path="toy/causal-wan")
    pipeline = bridge.archive.pipeline
    plans = []
    pipeline.hooks = lambda stage, node: plans.append((node, time.perf_counter())) if stage == "plan" else None
    marks, records, published = {}, {}, None
    for sid, letter in (("s", "A"), ("s2", "B")):
        session = t.handle(sid, "generate", pause_after_chunks=1, durable=False)
        req = t.tree_request(session, [t.PROMPTS[letter]] * t.SESSION_TOTAL, generation_index=1)
        req.request_id = "gf-" + letter
        marks[letter + "_begin"] = time.perf_counter()
        summary = t.run_tree(bridge, req)[0]
        records[req.request_id] = summary["archival_defer"]  # the live record (its seam keeps updating it)
        if mode == "seamless_wait" and letter == "A":  # inside A's open first hold: never-block
            published = all(item.get("published") for item in bridge.commit_reports("gf-A", wait=True)
                            if item.get("kind") == "chunk")
        time.sleep(0.3)  # the stage's VAE decode: the first chunk must not start fitting here
        if mode == "seam":
            with bridge.memory_critical_section("vae_decode", request_id=req.request_id):
                time.sleep(0.1)
        marks[letter + "_seam_exit"] = time.perf_counter()
        time.sleep(0.5)  # the gap between requests: fitting runs here
    stats = pipeline.stats()["generation_defer"]
    bridge.close()
    print(json.dumps({"marks": marks, "plans": plans, "records": records, "stats": stats, "published_A": published},
                     default=str))
""")


if __name__ == "__main__":
    unittest.main()
