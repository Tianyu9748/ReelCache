"""SGLang Step 7: ``ArchivePipeline.defer_gpu_work`` -- a caller's memory-critical section (Step 5b memory study:
LongLive 2.0's new-chunk archival overlapping the VAE decode OOMs 7-chunk requests on an 80 GB card).

Inside a section no node (or prewarm) starts GPU fitting; a node already on the GPU finishes before the section is
entered; queued nodes keep their order and bounds, publication stays ordered and the stored nodes are those of an
undeferred run. A thread inside a section never waits on work only the deferred fitter could do (its ``submit``
raises CapacityError, its ``drain`` / ``prewarm`` / ``close(drain=True)`` raise StateError), so a full queue cannot
deadlock; other threads wait as usual. CUDA tests (``REELCACHE_STEP7_GPU_TESTS=1``, GPU jobs only): the section holds no
fitter allocations and ``release_cached_memory`` returns the fitter's cached pool.
"""

from __future__ import annotations

import os
import threading
import time
import unittest

import torch

from reelcache.archive_pipeline import ArchivePipeline
from reelcache.errors import CapacityError, ConfigurationError, StateError
from reelcache.lineage_commit import ArchiveOptions
from tests import step6_support as s6


class Recorder:
    """Pipeline hook: records (stage, node, time); optional blocking gates per (stage, node)."""

    def __init__(self):
        self.events, self.blocks, self.lock = [], {}, threading.Lock()

    def __call__(self, stage, node):
        with self.lock:
            self.events.append((stage, node, time.perf_counter()))
        gate = self.blocks.get((stage, node))
        if gate is not None:
            entered, release = gate
            entered.set()
            if not release.wait(60):
                raise RuntimeError("test gate was not released")

    def stage(self, name):
        with self.lock:
            return [(node, at) for stage, node, at in self.events if stage == name]


class DeferGPUWorkTests(unittest.TestCase):
    def setUp(self):
        self.runtimes = s6.RuntimeFactory(self)

    def pipeline(self, name, **kwargs):
        runtime = self.runtimes(name)
        runtime.create_lineage("main")
        options = dict(encode_workers=2, max_source_chunks=3, max_unpublished=4)
        options.update(kwargs)
        pipeline = ArchivePipeline(runtime, method=s6.method(1), registry=s6.registry(),
                                   options=ArchiveOptions(encode_threads=1), **options)
        self.addCleanup(self._close, pipeline)
        return runtime, pipeline

    @staticmethod
    def _close(pipeline):
        try:
            pipeline.close(drain=False, timeout=60)
        except BaseException:  # noqa: BLE001 - tests assert failures themselves
            pass

    def reference(self, count):
        runtime, pipeline = self.pipeline("reference")
        tickets = [pipeline.submit("main", f"chunk-{index:03d}", *s6.chunk(index)) for index in range(count)]
        pipeline.drain("main", timeout=300)
        return [s6.node_content(runtime, ticket.result()) for ticket in tickets]

    def test_no_fit_starts_inside_a_section_and_publication_is_unchanged(self):
        expected = self.reference(3)
        recorder = Recorder()
        runtime, pipeline = self.pipeline("deferred", hooks=recorder)
        with pipeline.defer_gpu_work():
            self.assertTrue(pipeline.stats()["deferred"])
            tickets = [pipeline.submit("main", f"chunk-{index:03d}", *s6.chunk(index)) for index in range(3)]
            time.sleep(0.4)
            self.assertEqual(recorder.stage("plan"), [])
            self.assertFalse(any(ticket.done() for ticket in tickets))
            frontiers = pipeline.frontiers("main")
            self.assertEqual(frontiers["generated_frontier"], 3 * s6.TOKENS)
            self.assertEqual(frontiers["published_frontier"], 0)
            exited = time.perf_counter()
        pipeline.drain("main", timeout=300)
        self.assertFalse(pipeline.stats()["deferred"])
        self.assertEqual([node for node, _ in recorder.stage("plan")], ["chunk-000", "chunk-001", "chunk-002"])
        self.assertTrue(all(at >= exited for _, at in recorder.stage("plan")))
        self.assertEqual([s6.node_content(runtime, ticket.result()) for ticket in tickets], expected)
        stats = pipeline.stats()
        self.assertEqual(stats["deferrals"], 1)
        self.assertGreater(stats["deferred_seconds"], 0.3)
        self.assertEqual(stats["deferral_released_bytes"], 0)  # CPU pipeline: nothing to release

    def test_entry_waits_for_the_node_on_the_gpu_then_holds_the_next(self):
        expected = self.reference(2)
        recorder = Recorder()
        entered_finalize, release = threading.Event(), threading.Event()
        recorder.blocks[("finalize", "chunk-000")] = (entered_finalize, release)
        runtime, pipeline = self.pipeline("in-flight", hooks=recorder)
        tickets = [pipeline.submit("main", f"chunk-{index:03d}", *s6.chunk(index)) for index in range(2)]
        self.assertTrue(entered_finalize.wait(60))  # chunk 0 is on the fitter
        inside, leave, times = threading.Event(), threading.Event(), {}

        def section():
            with pipeline.defer_gpu_work():
                times["entered"] = time.perf_counter()
                inside.set()
                leave.wait(60)
                times["left"] = time.perf_counter()
        worker = threading.Thread(target=section)
        worker.start()
        time.sleep(0.3)
        self.assertFalse(inside.is_set())  # waits for the in-flight node
        with self.assertRaises(CapacityError):  # a bounded entry wait gives up and leaves no section open
            with pipeline.defer_gpu_work(timeout=0.05):
                pass
        released = time.perf_counter()
        release.set()
        self.assertTrue(inside.wait(60))
        self.assertGreaterEqual(times["entered"], released)
        time.sleep(0.3)
        self.assertEqual([node for node, _ in recorder.stage("plan")], ["chunk-000"])  # chunk 1 is held back
        leave.set()
        worker.join(60)
        pipeline.drain("main", timeout=300)
        plans = recorder.stage("plan")
        self.assertEqual([node for node, _ in plans], ["chunk-000", "chunk-001"])
        self.assertGreaterEqual(plans[1][1], times["left"])
        self.assertEqual([s6.node_content(runtime, ticket.result()) for ticket in tickets], expected)

    def test_a_full_queue_never_deadlocks(self):
        expected = self.reference(3)
        runtime, pipeline = self.pipeline("full", max_source_chunks=1, max_unpublished=1)
        tickets, other, outcomes = [], {}, {}
        holding, leave = threading.Event(), threading.Event()

        def section():  # the deferring thread, e.g. a request around its VAE decode
            with pipeline.defer_gpu_work():
                tickets.append(pipeline.submit("main", "chunk-000", *s6.chunk(0)))
                # Only the deferred fitter could make room or progress: immediate, clear errors.
                for name, call in (("submit", lambda: pipeline.submit("main", "chunk-001", *s6.chunk(1))),
                                   ("drain", lambda: pipeline.drain("main")),
                                   ("prewarm", lambda: pipeline.prewarm(s6.chunk(0)[1])),
                                   ("close", lambda: pipeline.close(drain=True))):
                    try:
                        call()
                        outcomes[name] = None
                    except BaseException as exc:  # noqa: BLE001 - asserted below
                        outcomes[name] = exc
                holding.set()
                leave.wait(60)
        deferring = threading.Thread(target=section, daemon=True)
        deferring.start()
        self.assertTrue(holding.wait(120), "a call inside the section blocked (deadlock)")
        self.assertIsInstance(outcomes["submit"], CapacityError)
        for name in ("drain", "prewarm", "close"):
            self.assertIsInstance(outcomes[name], StateError, name)
        # Another thread waits with ordinary backpressure and proceeds once the section exits.
        started = threading.Event()

        def submit_more():
            started.set()
            other["tickets"] = [pipeline.submit("main", f"chunk-{index:03d}", *s6.chunk(index)) for index in (1, 2)]
        worker = threading.Thread(target=submit_more, daemon=True)
        worker.start()
        self.assertTrue(started.wait(60))
        time.sleep(0.3)
        self.assertNotIn("tickets", other)
        leave.set()
        deferring.join(60)
        worker.join(120)
        self.assertFalse(worker.is_alive(), "backpressured submit did not resume after the section")
        tickets += other["tickets"]
        pipeline.drain("main", timeout=300)
        self.assertEqual([s6.node_content(runtime, ticket.result()) for ticket in tickets], expected)
        stats = pipeline.stats()
        self.assertGreaterEqual(stats["deferral_rejections"], 1)
        self.assertGreaterEqual(stats["backpressure_waits"], 1)

    def test_sections_nest_and_overlap_across_threads(self):
        recorder = Recorder()
        runtime, pipeline = self.pipeline("nested", hooks=recorder)
        holding, leave = threading.Event(), threading.Event()

        def other_section():
            with pipeline.defer_gpu_work():
                holding.set()
                leave.wait(60)
        worker = threading.Thread(target=other_section)
        with pipeline.defer_gpu_work():
            with pipeline.defer_gpu_work():
                ticket = pipeline.submit("main", "chunk-000", *s6.chunk(0))
            worker.start()
            self.assertTrue(holding.wait(60))
        time.sleep(0.3)  # this thread's sections are closed; the other thread's is still open
        self.assertEqual(recorder.stage("plan"), [])
        self.assertTrue(pipeline.stats()["deferred"])
        leave.set()
        worker.join(60)
        pipeline.drain("main", timeout=300)
        self.assertTrue(ticket.done())
        self.assertEqual(pipeline.stats()["deferrals"], 3)

    def test_configuration_and_the_fitter_thread(self):
        caught = []
        runtime, pipeline = None, None

        def hooks(stage, node):
            if stage == "plan":
                try:
                    with pipeline.defer_gpu_work():
                        pass
                except StateError as exc:
                    caught.append(exc)
        runtime, pipeline = self.pipeline("config", hooks=hooks)
        pipeline.submit("main", "chunk-000", *s6.chunk(0))
        pipeline.drain("main", timeout=300)
        self.assertEqual(len(caught), 1)  # the fitter thread cannot defer its own work
        for bad in ({"wait": 1}, {"release_cached_memory": "yes"}, {"timeout": -1}, {"timeout": True}):
            with self.subTest(**{key: repr(value) for key, value in bad.items()}):
                with self.assertRaises(ConfigurationError):
                    with pipeline.defer_gpu_work(**bad):
                        pass
        with pipeline.defer_gpu_work(wait=False, release_cached_memory=True):
            pass
        self.assertFalse(pipeline.stats()["deferred"])
        pipeline.close()
        with self.assertRaises(StateError):
            with pipeline.defer_gpu_work():
                pass


if os.environ.get("REELCACHE_STEP7_GPU_TESTS") == "1":
    class CUDADeferGPUWorkTests(unittest.TestCase):
        """CUDA (inside GPU jobs only; collected only with REELCACHE_STEP7_GPU_TESTS=1)."""

        def setUp(self):
            if not torch.cuda.is_available():
                raise AssertionError("REELCACHE_STEP7_GPU_TESTS=1 requires a CUDA device")
            self.runtimes = s6.RuntimeFactory(self)

        def run_chain(self, name, defer):
            from reelcache.gpu_fitting import GPUFitter
            runtime = self.runtimes(name)
            runtime.create_lineage("main")
            pipeline = ArchivePipeline(runtime, method=s6.method(1), registry=s6.registry(),
                                       options=ArchiveOptions(encode_threads=1), device="cuda",
                                       fitter=GPUFitter(gpu_id=0, gpu_uuid="GPU-" + str(
                                           torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-"),
                                           method="gram_batched", eigh_solver="auto"),
                                       encode_workers=2, max_source_chunks=3, max_unpublished=4)
            try:
                sources = []
                for index in range(4):
                    matrices, contexts = s6.chunk(index)
                    sources.append(({key: value.cuda() for key, value in matrices.items()}, contexts))
                torch.cuda.synchronize()

                def submit(index):
                    event = torch.cuda.Event()
                    event.record()
                    return pipeline.submit("main", f"chunk-{index:03d}", *sources[index], ready_event=event)
                tickets = [submit(0)]
                pipeline.drain("main", timeout=300)  # the fitter's pool now holds cached blocks
                report = {}
                if defer:
                    with pipeline.defer_gpu_work(release_cached_memory=True):
                        tickets += [submit(index) for index in (1, 2, 3)]
                        torch.cuda.synchronize()
                        allocated = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        time.sleep(0.5)  # queued nodes wait; nothing of the fitter may allocate meanwhile
                        report["peak_inside"] = torch.cuda.max_memory_allocated()
                        report["allocated_inside"] = allocated
                        report["plans_inside"] = sum(ticket.done() for ticket in tickets[1:])
                else:
                    tickets += [submit(index) for index in (1, 2, 3)]
                pipeline.drain("main", timeout=300)
                report["stats"] = pipeline.stats()
                return [s6.node_content(runtime, ticket.result()) for ticket in tickets], report
            finally:
                pipeline.close(drain=False, timeout=60)

        def test_section_holds_no_fitter_allocations_and_releases_the_pool(self):
            expected, _ = self.run_chain("cuda-plain", False)
            torch.cuda.empty_cache()
            contents, report = self.run_chain("cuda-deferred", True)
            self.assertEqual(contents, expected)
            self.assertEqual(report["plans_inside"], 0)
            self.assertEqual(report["peak_inside"], report["allocated_inside"])  # no fitter allocation in the section
            self.assertGreater(report["stats"]["deferral_released_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
