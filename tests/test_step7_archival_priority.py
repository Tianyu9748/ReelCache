"""Archival FFmpeg next to restores (Step 19 follow-up): pipe I/O, the pump pause and the opt-in CPU priority.

Step 19 (A100-40GB CausVid, nsys) found prepared restores 1.12-1.13x slower while another session's archival ran,
on the decode/CPU side, with the restore-yield counter at 0. Archival yields only at stage boundaries, and an x265
encode or validation decode (an FFmpeg process, seconds per node) already running was never paused; its pipes were
pumped by ``Popen.communicate`` (4 KiB writes: ~35k GIL hand-offs per node). Now:

* FFmpeg pipes move in large blocking reads/writes on helper threads (results, errors and timeouts unchanged);
* ``ArchivePipeline(pause_ffmpeg=True)`` (bridge ``archival_pump_pause``, default on) pauses the pump chunk by chunk
  while a restore runs, bounded per busy period and released while a caller waits on archival (no self-deadlock
  when a restore waits for its own dependencies);
* ``ArchivePipeline(cpu_priority=...)`` (bridge ``archival_cpu_priority``, opt-in) starts the FFmpeg processes at
  background priority (SCHED_IDLE or a nice increment), inherited by their threads. Output never changes.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
import unittest
from unittest import mock

import numpy as np

from reelcache import _hevc
from reelcache.archive_pipeline import ArchivePipeline
from reelcache.errors import ConfigurationError, UnsupportedError
from reelcache.lineage_commit import ArchiveOptions
from tests.step6_support import RuntimeFactory, chunk, method, node_content, registry

# The child reports its own and a new thread's scheduling policy and nice value.
_PROBE = ("import os, threading; seen = []; "
          "t = threading.Thread(target=lambda: seen.append((os.sched_getscheduler(0), os.getpriority(os.PRIO_PROCESS, "
          "threading.get_native_id())))); t.start(); t.join(); "
          "print(os.sched_getscheduler(0), os.getpriority(os.PRIO_PROCESS, 0), *seen[0])")


def _policy_report(priority):
    with _hevc.ffmpeg_priority(priority):
        result = _hevc._run([sys.executable, "-c", _PROBE])
    return tuple(int(value) for value in result.stdout.split())


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux scheduling policies")
class FFmpegPriorityTests(unittest.TestCase):
    def test_children_and_their_threads_run_at_the_requested_priority(self):
        nice = os.getpriority(os.PRIO_PROCESS, 0)
        self.assertEqual(_policy_report(None), (os.SCHED_OTHER, nice, os.SCHED_OTHER, nice))
        self.assertEqual(_policy_report("idle"), (os.SCHED_IDLE, nice, os.SCHED_IDLE, nice))
        lowered = min(19, nice + 10)
        self.assertEqual(_policy_report(10), (os.SCHED_OTHER, lowered, os.SCHED_OTHER, lowered))
        # The calling thread keeps its own priority, and the setting ends with its context.
        self.assertEqual((os.sched_getscheduler(0), os.getpriority(os.PRIO_PROCESS, 0)), (os.SCHED_OTHER, nice))
        self.assertIsNone(_hevc._PRIORITY.get())

    def test_values_timeouts_and_failures_keep_their_semantics(self):
        for value in ("high", 0, 20, -1, 1.5, True):
            with self.assertRaises(ConfigurationError):
                _hevc.check_cpu_priority(value)
            with self.assertRaises(ConfigurationError):
                ArchivePipeline(None, method=method(), registry=registry(), cpu_priority=value)
        with _hevc.ffmpeg_priority("idle"):
            with self.assertRaisesRegex(UnsupportedError, "FFmpeg execution failed"):
                _hevc._run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)
            with self.assertRaisesRegex(UnsupportedError, r"FFmpeg failed \(3\)"):
                _hevc._run([sys.executable, "-c", "import sys; sys.exit(3)"])
            echoed = _hevc._run([sys.executable, "-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"],
                                data=b"x" * 300000)
        self.assertEqual(echoed.stdout, b"x" * 300000)

    def test_encodes_are_bitwise_identical(self):
        pixels = np.random.default_rng(5).integers(0, 255, size=(3, 3, 64, 64), dtype=np.uint8)
        binary = _hevc.discover_ffmpeg()
        streams = {}
        for priority in (None, "idle", 19):
            with _hevc.ffmpeg_priority(priority):
                streams[priority] = _hevc._encode_segment(pixels, "yuv444p", binary, threads=4)
        self.assertEqual(streams["idle"], streams[None])
        self.assertEqual(streams[19], streams[None])


class StreamingPipeTests(unittest.TestCase):
    """FFmpeg pipes exchanged in large blocking reads/writes on helper threads instead of ``Popen.communicate``'s
    select loop (4 KiB writes, 32 KiB reads: two syscalls and GIL hand-offs per step, contending with the process's
    other Python threads, e.g. a restore's decoder workers)."""

    ECHO = [sys.executable, "-c", "import sys; data = sys.stdin.buffer.read(); sys.stderr.write('n=%d' % len(data)); "
            "sys.stdout.buffer.write(data[::-1])"]

    @staticmethod
    def communicate(process, data, timeout, pause=None):
        """The former exchange (``Popen.communicate``), as the reference."""
        return process.communicate(data, timeout=timeout)

    def exchange(self, streaming, data):
        writes = []
        original = os.write

        def counting(fd, payload):
            writes.append(len(payload))
            return original(fd, payload)
        reference = contextlib.nullcontext() if streaming else mock.patch.object(_hevc, "_exchange", self.communicate)
        with reference, mock.patch.object(os, "write", counting):
            result = _hevc._run(self.ECHO, data=data)
        return result, writes

    def test_results_match_communicate_with_few_large_writes(self):
        data = np.random.default_rng(3).integers(0, 255, size=6 << 20, dtype=np.uint8).tobytes()
        streamed, few = self.exchange(True, data)
        selected, many = self.exchange(False, data)
        self.assertEqual((streamed.stdout, streamed.stderr, streamed.returncode),
                         (selected.stdout, selected.stderr, selected.returncode))
        self.assertEqual(streamed.stdout, data[::-1])
        self.assertEqual(sum(few), len(data))
        self.assertLessEqual(len(few), 8)
        self.assertGreater(len(many), 1000)  # 4 KiB per step

    def test_encode_and_validation_decode_are_bitwise_identical(self):
        from reelcache.node_packets import _validation_decode
        pixels = np.random.default_rng(7).integers(0, 255, size=(4, 3, 64, 96), dtype=np.uint8)
        binary = _hevc.discover_ffmpeg()
        results = {}
        for streaming in (True, False):
            with (contextlib.nullcontext() if streaming else mock.patch.object(_hevc, "_exchange", self.communicate)):
                stream = _hevc._encode_segment(pixels, "yuv444p", binary, threads=4)
                images, _ = _validation_decode(stream, pictures=4, height=64, width=96, pix_fmt="yuv444p",
                                               threads=4)
                results[streaming] = (stream, images.tobytes())
        self.assertEqual(results[True], results[False])

    def test_a_child_that_exits_early_reports_its_status(self):
        with self.assertRaisesRegex(UnsupportedError, r"FFmpeg failed \(5\)"):
            _hevc._run([sys.executable, "-c", "import sys; sys.exit(5)"], data=b"y" * (8 << 20))


class PumpPauseTests(unittest.TestCase):
    """The pipe pump pauses before every chunk while its hook blocks (an archival pipeline: a restore in flight)."""

    def test_the_pump_calls_its_hook_per_chunk_and_results_are_unchanged(self):
        data = bytes(range(256)) * ((5 << 20) // 256)
        calls = []
        with _hevc.ffmpeg_pump_pause(lambda: calls.append(1)):
            paused = _hevc._run(StreamingPipeTests.ECHO, data=data)
        plain = _hevc._run(StreamingPipeTests.ECHO, data=data)
        self.assertEqual((paused.stdout, paused.stderr), (plain.stdout, plain.stderr))
        # stdin in 1 MiB pieces plus stdout reads: at least one call per MiB each way.
        self.assertGreaterEqual(len(calls), 2 * (len(data) >> 20))
        with self.assertRaises(ConfigurationError):
            with _hevc.ffmpeg_pump_pause(object()):
                pass

    def test_a_blocking_hook_stalls_the_child_until_released(self):
        release, entered = threading.Event(), threading.Event()

        def hook():
            entered.set()
            release.wait(30)
        result, failure = {}, []

        def run():
            try:
                with _hevc.ffmpeg_pump_pause(hook):
                    result["value"] = _hevc._run(StreamingPipeTests.ECHO, data=b"z" * (3 << 20))
            except BaseException as exc:  # noqa: BLE001
                failure.append(exc)
        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(entered.wait(30))
        time.sleep(0.3)
        self.assertNotIn("value", result)  # nothing moved while the hook blocked
        release.set()
        worker.join(60)
        self.assertEqual(failure, [])
        self.assertEqual(result["value"].stdout, b"z" * (3 << 20))


class PipelinePriorityTests(unittest.TestCase):
    def setUp(self):
        self.runtimes = RuntimeFactory(self)

    def archived(self, priority, count=3):
        runtime = self.runtimes(f"priority-{priority}")
        runtime.create_lineage("main")
        pipeline = ArchivePipeline(runtime, method=method(), registry=registry(),
                                   options=ArchiveOptions(encode_threads=2), cpu_priority=priority)
        self.addCleanup(lambda: pipeline.close(drain=False, timeout=60))
        lowered = []
        original = _hevc._lower_priority

        def record(pid, value):
            lowered.append(value)
            original(pid, value)
        with mock.patch.object(_hevc, "_lower_priority", record):
            tickets = []
            for index in range(count):
                matrices, contexts = chunk(index)
                tickets.append(pipeline.submit("main", f"chunk-{index:03d}", matrices, contexts))
            pipeline.drain("main", timeout=300)
        self.assertEqual(pipeline.stats()["cpu_priority"], priority)
        return [node_content(runtime, ticket.result()) for ticket in tickets], lowered

    def test_pipeline_ffmpeg_runs_at_idle_priority_with_identical_archives(self):
        plain, none_lowered = self.archived(None)
        idle, idle_lowered = self.archived("idle")
        self.assertEqual(none_lowered, [])
        # One x265 encode and one validation decode per node, all lowered.
        self.assertGreaterEqual(len(idle_lowered), 6)
        self.assertEqual(set(idle_lowered), {"idle"})
        self.assertEqual(idle, plain)

    def paused_pipeline(self, name, activity, *, max_seconds=0.2):
        runtime = self.runtimes(name)
        runtime.create_lineage("main")
        pipeline = ArchivePipeline(runtime, method=method(), registry=registry(),
                                   options=ArchiveOptions(encode_threads=2), yield_to=activity,
                                   yield_max_seconds=max_seconds, pause_ffmpeg=True)
        self.addCleanup(lambda: pipeline.close(drain=False, timeout=60))
        return runtime, pipeline

    def test_paused_pumps_wait_for_restores_and_change_nothing(self):
        from reelcache.archive_restore import RestoreActivity
        plain, _ = self.archived(None)
        activity = RestoreActivity()
        runtime, pipeline = self.paused_pipeline("paused", activity)
        hold, stop = threading.Event(), threading.Event()
        pumped = []
        original = _hevc._exchange

        def recording(process, data, timeout, pause=None):
            pumped.append(pause is not None)
            return original(process, data, timeout, pause)

        def restore_loop():  # back-to-back "restores": every worker check finds one running
            while not stop.is_set():
                with activity.running():
                    hold.set()
                    stop.wait(0.05)
        holder = threading.Thread(target=restore_loop)
        with mock.patch.object(_hevc, "_exchange", recording):
            holder.start()
            try:
                self.assertTrue(hold.wait(10))
                tickets = []
                for index in range(3):
                    matrices, contexts = chunk(index)
                    tickets.append(pipeline.submit("main", f"chunk-{index:03d}", matrices, contexts))
                pipeline.drain("main", timeout=600)
            finally:
                stop.set()
                holder.join()
        stats = pipeline.stats()["restore_yield"]
        self.assertGreater(stats["yields"], 0)
        self.assertTrue(pumped and all(pumped))  # every FFmpeg exchange of the pipeline got the pause hook
        self.assertEqual([node_content(runtime, ticket.result()) for ticket in tickets], plain)
        with self.assertRaises(ConfigurationError):
            ArchivePipeline(runtime, method=method(), registry=registry(), pause_ffmpeg=1)

    def drain_inside_restore(self, release):
        """Submit 3 chunks, then drain inside a restore bracket held by this thread (a restore waiting for its own
        dependencies). Returns the drain seconds, the timeline and the yield statistics."""
        from reelcache.archive_restore import RestoreActivity
        activity = RestoreActivity()
        runtime, pipeline = self.paused_pipeline(f"bracket-{release}", activity, max_seconds=3.0)
        if not release:
            pipeline._yield.bypass = None  # the behavior without the release, for the timeline's comparison
        timeline, origin = [], time.perf_counter()
        pipeline.hooks = lambda stage, node: timeline.append((round(time.perf_counter() - origin, 3), stage, node))
        with activity.running():
            timeline.append((round(time.perf_counter() - origin, 3), "restore_begin", "-"))
            for index in range(3):
                matrices, contexts = chunk(index)
                pipeline.submit("main", f"chunk-{index:03d}", matrices, contexts)
            started = time.perf_counter()
            timeline.append((round(started - origin, 3), "drain_begin", "-"))
            pipeline.drain("main", timeout=600)
            seconds = time.perf_counter() - started
            timeline.append((round(time.perf_counter() - origin, 3), "drain_end", "-"))
        return seconds, timeline, pipeline.stats()["restore_yield"]

    def test_a_restore_waiting_for_its_own_dependencies_is_not_held_by_the_pause(self):
        released, timeline, stats = self.drain_inside_restore(True)
        held, held_timeline, held_stats = self.drain_inside_restore(False)
        # Without the release every worker waits out its 3 s bound per busy period before the drain completes.
        self.assertGreater(held, 3.0, held_timeline)
        self.assertGreater(held_stats["yield_timeouts"], 0)
        # With it the drain runs at archival speed: no timeout, the waits end within a slice of the drain's start.
        self.assertEqual(stats["yield_timeouts"], 0, timeline)
        self.assertLess(released, held - 2.0, (timeline, held_timeline))
        self.assertLess(stats["yield_seconds"], held_stats["yield_seconds"])

    def test_shared_hooks_expire_once_per_busy_period(self):
        from reelcache.archive_pipeline import _RestoreYield
        from reelcache.archive_restore import RestoreActivity
        activity = RestoreActivity()
        gate = _RestoreYield(activity, 0.2)
        hook = gate.shared()
        with activity.running():
            started = time.perf_counter()
            threads = [threading.Thread(target=hook) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            first = time.perf_counter() - started
            clock = time.perf_counter()
            hook()  # expired for this busy period: no further wait
            again = time.perf_counter() - clock
        self.assertLess(first, 1.0)
        self.assertLess(again, 0.05)
        self.assertEqual(gate.stats()["yield_timeouts"], 3)
        with activity.running():  # a new busy period waits again
            clock = time.perf_counter()
            hook()
            self.assertGreaterEqual(time.perf_counter() - clock, 0.15)

    def test_bridge_config_reaches_the_pipeline(self):
        import tests.test_sglang_bridge as t
        from reelcache.integrations.sglang.config import ReelCacheRuntimeConfig
        base = {"store_root": "x", "representation": "svd_hevc"}
        self.assertIsNone(ReelCacheRuntimeConfig.from_dict(base).archival_cpu_priority)
        self.assertIs(ReelCacheRuntimeConfig.from_dict(base).archival_pump_pause, True)
        with self.assertRaises(ConfigurationError):
            ReelCacheRuntimeConfig.from_dict({**base, "archival_pump_pause": 1})
        for value in ("idle", 5):
            self.assertEqual(ReelCacheRuntimeConfig.from_dict({**base, "archival_cpu_priority": value})
                             .archival_cpu_priority, value)
        for value in ("low", 0, 25, 2.0):
            with self.assertRaises(ConfigurationError):
                ReelCacheRuntimeConfig.from_dict({**base, "archival_cpu_priority": value})

        class Case(t.BridgeTestCase):
            representation = "svd_hevc"

            def runTest(self):
                pass
        case = Case()
        case.setUp()
        try:
            bridge = case.bridge(archival_cpu_priority="idle")
            self.assertEqual((bridge.archive.pipeline.cpu_priority, bridge.archive.pipeline.pause_ffmpeg), ("idle", True))
            bridge.close()
            bridge = case.bridge(store="no-pause", archival_pump_pause=False)
            self.assertEqual((bridge.archive.pipeline.cpu_priority, bridge.archive.pipeline.pause_ffmpeg), (None, False))
            bridge.close()
        finally:
            case.doCleanups()


if __name__ == "__main__":
    unittest.main()
