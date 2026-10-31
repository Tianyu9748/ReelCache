"""SGLang Step 7: archival yields to restores in flight (Step 5b's restore-stall diagnosis: GIL contention).

Lineage restorers bracket every restore in their ``RestoreActivity``. An ``ArchivePipeline`` with ``yield_to``
(default: its seal-time restorer's activity) checks it before every node, every routine step of planning and
finalization (``deferred_reads.step_hook``) and its other yield points, and every encode, store and publication, and
waits at most ``yield_max_seconds`` per restore busy period, so archival always progresses. While a caller waits on
archival (a drain, a ticket, backpressure, a deferral) nobody waits (Step 19 follow-up: a restore waiting for its own
dependencies inside its bracket is never held by the pause). Yielding changes timing only: node contents are
identical.
"""

from __future__ import annotations

import threading
import time
import unittest

import torch

from reelcache import deferred_reads
from reelcache.archive_pipeline import ArchivePipeline, _RestoreYield
from reelcache.archive_restore import LineageRestorer, RawLineageRestorer, RestoreActivity, SoftwareLineageRestorer
from reelcache.errors import ConfigurationError
from reelcache.lineage_commit import ArchiveOptions
from reelcache.types import MatrixKey
from tests import step6_support as s6
from tests.test_packed_nvdec import SoftwarePool


_SLICE = _RestoreYield.SLICE


def _routine(rounds):
    for index in range(rounds):
        yield (torch.tensor([float(index)]),)
    return rounds


class StepHookTests(unittest.TestCase):
    def test_hook_runs_before_every_routine_step_and_yield_point_of_its_thread_only(self):
        calls = []
        with deferred_reads.step_hook(lambda: calls.append(threading.get_ident())):
            self.assertEqual(deferred_reads.run_lockstep([_routine(3), _routine(1)]), [3, 1])
            self.assertEqual(len(calls), 6)  # every resumption: 3 reads + completion, 1 read + completion
            other = threading.Thread(target=deferred_reads.run_lockstep, args=([_routine(2)],))
            other.start()
            other.join(60)
            self.assertEqual(len(calls), 6)
            with deferred_reads.step_hook(None):
                deferred_reads.run_lockstep([_routine(2)])
                deferred_reads.yield_point()
            self.assertEqual(len(calls), 6)
            deferred_reads.run_lockstep([_routine(1)])
            self.assertEqual(len(calls), 8)
            deferred_reads.yield_point()
            self.assertEqual(len(calls), 9)
        deferred_reads.run_lockstep([_routine(2)])
        deferred_reads.yield_point()
        self.assertEqual(len(calls), 9)
        self.assertEqual(set(calls), {threading.get_ident()})


class RestoreActivityTests(unittest.TestCase):
    def test_busy_periods_and_bounded_idle_waits(self):
        activity = RestoreActivity()
        self.assertFalse(activity.active())
        self.assertTrue(activity.wait_idle(0))
        with activity.running():
            self.assertEqual(activity.epoch, 1)
            with activity.running():  # overlapping restores share one busy period
                self.assertEqual(activity.epoch, 1)
            self.assertTrue(activity.active())
            self.assertFalse(activity.wait_idle(0.01))
        self.assertFalse(activity.active())
        with activity.running():
            self.assertEqual(activity.epoch, 2)
        self.assertEqual(activity.stats(), {"restores": 3, "epoch": 2, "active": 0})
        entered, finished = threading.Event(), threading.Event()

        def restore():
            with activity.running():
                entered.set()
                time.sleep(0.2)
            finished.set()
        worker = threading.Thread(target=restore)
        worker.start()
        self.assertTrue(entered.wait(30))
        self.assertTrue(activity.wait_idle(60))  # woken when the restore ends
        self.assertTrue(finished.wait(30))
        worker.join(30)
        with self.assertRaises(RuntimeError):
            with activity.running():
                raise RuntimeError("restore failed")
        self.assertFalse(activity.active())

    def test_restorers_bracket_every_restore(self):
        runtime = s6.RuntimeFactory(self)("bracket")
        runtime.create_lineage("main")
        registry = s6.registry()
        head = None
        for index in range(2):
            head = runtime.commit_node("main", f"chunk-{index:03d}", *s6.chunk(index), method=s6.method(),
                                       registry=registry, options=ArchiveOptions(encode_threads=1)).node_ref
        for restorer in (LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 30),
                         SoftwareLineageRestorer(runtime)):
            with self.subTest(restorer=type(restorer).__name__):
                seen, caches = [], {}

                def destination(node_id, key, tile, restorer=restorer, seen=seen, caches=caches):
                    from reelcache import restore_kernels as rk
                    seen.append(restorer.activity.active())
                    key = key if isinstance(key, MatrixKey) else key[0]
                    if key not in caches:
                        caches[key] = torch.zeros((1, 2 * s6.TOKENS, 1, 128), dtype=torch.float32)
                    return rk.NativeView(caches[key], "BTHD", token_offset=tile.token_start)
                restorer.restore(head, destination)
                self.assertTrue(seen and all(seen))
                self.assertFalse(restorer.activity.active())
                self.assertEqual(restorer.activity.stats()["restores"], 1)
        raw = RawLineageRestorer(runtime)
        with self.assertRaises(ConfigurationError):  # a failing restore leaves its bracket
            raw.restore(head, lambda *args: None, rope=lambda *args: None)
        self.assertEqual(raw.activity.stats(), {"restores": 1, "epoch": 1, "active": 0})


class PipelineYieldTests(unittest.TestCase):
    def setUp(self):
        self.runtimes = s6.RuntimeFactory(self)

    def pipeline(self, name, **kwargs):
        runtime = self.runtimes(name)
        runtime.create_lineage("main")
        pipeline = ArchivePipeline(runtime, method=s6.method(), registry=s6.registry(),
                                   options=ArchiveOptions(encode_threads=1), **kwargs)
        self.addCleanup(self._close, pipeline)
        return runtime, pipeline

    @staticmethod
    def _close(pipeline):
        try:
            pipeline.close(drain=False, timeout=60)
        except BaseException:  # noqa: BLE001 - tests assert failures themselves
            pass

    def reference(self, count):
        runtime, pipeline = self.pipeline("reference", yield_to=None)
        tickets = [pipeline.submit("main", f"chunk-{index:03d}", *s6.chunk(index)) for index in range(count)]
        pipeline.drain("main", timeout=300)
        self.assertIsNone(pipeline.stats()["restore_yield"])
        return [s6.node_content(runtime, ticket.result()) for ticket in tickets]

    def test_archival_waits_while_a_restore_runs_and_stores_the_same_nodes(self):
        expected = self.reference(2)
        activity = RestoreActivity()
        events = []
        runtime, pipeline = self.pipeline(
            "waits", yield_to=activity, yield_max_seconds=120,
            hooks=lambda stage, node: events.append((stage, node, time.perf_counter())))
        with activity.running():
            tickets = [pipeline.submit("main", f"chunk-{index:03d}", *s6.chunk(index)) for index in range(2)]
            time.sleep(0.3)
            self.assertEqual(events, [])  # the fitter waits before planning the first node
            self.assertFalse(tickets[0].done())
            last_inside = time.perf_counter()
        pipeline.drain("main", timeout=300)
        # Deterministic under load: no stage of either node ran before the restore ended (the fitter's recorded
        # wait starts at its pickup, so its length is not compared with the 0.3 s hold).
        self.assertTrue(events and all(at >= last_inside for _, _, at in events), events)
        self.assertEqual([s6.node_content(runtime, ticket.result()) for ticket in tickets], expected)
        stats = pipeline.stats()["restore_yield"]
        self.assertGreaterEqual(stats["yields"], 1)
        self.assertGreater(stats["yield_seconds"], 0.0)
        self.assertEqual(stats["yield_timeouts"], 0)

    def test_planning_steps_yield_to_a_restore_that_starts_mid_plan(self):
        activity = RestoreActivity()
        entered, times = threading.Event(), {}

        def restore():
            with activity.running():
                entered.set()
                time.sleep(0.4)
                times["restore_last_inside"] = time.perf_counter()

        def hooks(stage, node):
            times[stage] = time.perf_counter()
            if stage == "plan":  # the fitter thread, just before the node's first lockstep step
                threading.Thread(target=restore).start()
                self.assertTrue(entered.wait(30))
        runtime, pipeline = self.pipeline("mid-plan", yield_to=activity, yield_max_seconds=120, hooks=hooks)
        ticket = pipeline.submit("main", "chunk-000", *s6.chunk(0))
        deadline = time.monotonic() + 300
        while not ticket.done() and time.monotonic() < deadline:  # no caller waits on archival meanwhile
            time.sleep(0.01)
        pipeline.drain("main", timeout=300)
        self.assertGreater(times["finalize"] - times["plan"], 0.3)
        # Deterministic under load: the node finalized only after the restore ended. The fitter's recorded yield
        # starts at its next step, after the hook, so its length is not compared with the 0.4 s restore.
        self.assertGreaterEqual(times["finalize"], times["restore_last_inside"])
        stats = pipeline.stats()["restore_yield"]
        self.assertGreaterEqual(stats["yields"], 1)
        self.assertGreater(stats["yield_seconds"], 0.0)

    def test_waits_are_bounded_per_busy_period(self):
        expected = self.reference(2)
        activity = RestoreActivity()
        runtime, pipeline = self.pipeline("bounded", yield_to=activity, yield_max_seconds=0.05, encode_workers=2)
        with activity.running():
            tickets = [pipeline.submit("main", f"chunk-{index:03d}", *s6.chunk(index)) for index in range(2)]
            deadline = time.monotonic() + 300
            while not all(ticket.done() for ticket in tickets) and time.monotonic() < deadline:
                time.sleep(0.01)  # nobody waits on archival: each worker waits once per busy period, then works
        self.assertEqual([s6.node_content(runtime, ticket.result()) for ticket in tickets], expected)
        stats = pipeline.stats()["restore_yield"]
        self.assertGreaterEqual(stats["yield_timeouts"], 1)
        self.assertEqual(stats["yields"], stats["yield_timeouts"])
        self.assertLessEqual(stats["yields"], 1 + 2)  # the fitter thread and two encode workers

    def test_a_drain_inside_a_restore_releases_the_pause(self):
        expected = self.reference(2)
        activity = RestoreActivity()
        runtime, pipeline = self.pipeline("released", yield_to=activity, yield_max_seconds=60, encode_workers=2)
        with activity.running():
            tickets = [pipeline.submit("main", f"chunk-{index:03d}", *s6.chunk(index)) for index in range(2)]
            started = time.perf_counter()
            pipeline.drain("main", timeout=300)  # the restore waits for its own dependencies: released at once
            drained = time.perf_counter() - started
        self.assertLess(drained, 30)  # far below one 60 s bound per worker
        self.assertEqual([s6.node_content(runtime, ticket.result()) for ticket in tickets], expected)
        stats = pipeline.stats()["restore_yield"]
        self.assertEqual(stats["yield_timeouts"], 0)
        self.assertLessEqual(stats["yield_seconds"], stats["yields"] * (_SLICE + 0.5))

    def test_a_waiting_ticket_releases_the_pause(self):
        activity = RestoreActivity()
        runtime, pipeline = self.pipeline("ticket", yield_to=activity, yield_max_seconds=60)
        with activity.running():
            ticket = pipeline.submit("main", "chunk-000", *s6.chunk(0))
            started = time.perf_counter()
            ticket.result(timeout=300)
            self.assertLess(time.perf_counter() - started, 30)
        self.assertEqual(pipeline.stats()["restore_yield"]["yield_timeouts"], 0)

    def test_yield_configuration(self):
        restorer = LineageRestorer(self.runtimes("config-runtime"), SoftwarePool(), arena_bytes=1 << 30)
        _, auto = self.pipeline("auto", restorer=restorer)
        self.assertIs(auto._yield.activity, restorer.activity)
        self.assertIsNotNone(auto.stats()["restore_yield"])
        _, plain = self.pipeline("plain")
        self.assertIsNone(plain.stats()["restore_yield"])
        _, off = self.pipeline("off", restorer=restorer, yield_to=None)
        self.assertIsNone(off.stats()["restore_yield"])
        _, explicit = self.pipeline("explicit", yield_to=restorer)
        self.assertIs(explicit._yield.activity, restorer.activity)
        for bad in ({"yield_to": "always"}, {"yield_to": object()}, {"yield_max_seconds": 0},
                    {"yield_max_seconds": True}, {"yield_max_seconds": -1.0}, {"yield_max_seconds": "1"}):
            with self.subTest(**{key: repr(value) for key, value in bad.items()}):
                runtime = self.runtimes(f"bad-{len(bad)}-{id(bad)}")
                with self.assertRaises(ConfigurationError):
                    ArchivePipeline(runtime, method=s6.method(), registry=s6.registry(), **bad)


if __name__ == "__main__":
    unittest.main()
