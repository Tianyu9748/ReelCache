"""Step 6: ordered asynchronous archival (ArchivePipeline) contracts on CPU."""

from __future__ import annotations

import threading
import time
import unittest

import torch

from reelcache.archive_pipeline import ArchiveFailure, ArchivePipeline
from reelcache.errors import CapacityError, ConfigurationError, PendingError, StateError, UnsupportedError
from reelcache.lineage_commit import ArchiveOptions
from reelcache.manager import KVRuntime
from tests.step6_support import KEYS, MODEL, RuntimeFactory, chunk, method, node_content, registry


class Gate:
    """Test hook: optional per-(stage, node) delays, blocking gates and injected failures."""

    def __init__(self):
        self.delays, self.failures, self.blocks = {}, {}, {}
        self.events = []
        self.lock = threading.Lock()

    def __call__(self, stage, node_id):
        with self.lock:
            self.events.append((stage, node_id, time.perf_counter()))
        gate = self.blocks.get((stage, node_id))
        if gate is not None:
            entered, release = gate
            entered.set()
            if not release.wait(30):
                raise RuntimeError("test gate was not released")
        delay = self.delays.get((stage, node_id))
        if delay:
            time.sleep(delay)
        failure = self.failures.get((stage, node_id))
        if failure is not None:
            raise failure

    def order(self, stage):
        with self.lock:
            return [node for name, node, _ in self.events if name == stage]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.runtimes = RuntimeFactory(self)

    def pipeline(self, runtime, *, lag=0, hooks=None, reg=None, **kwargs):
        pipeline = ArchivePipeline(runtime, method=method(lag), registry=reg or registry(),
                                   options=kwargs.pop("options", ArchiveOptions(encode_threads=1)), hooks=hooks,
                                   **kwargs)
        self.addCleanup(self._close, pipeline)
        return pipeline

    @staticmethod
    def _close(pipeline):
        try:
            pipeline.close(drain=False, timeout=60)
        except BaseException:  # noqa: BLE001 - tests assert failures themselves
            pass

    def sync_reference(self, count, *, lag=0, streams=()):
        runtime = self.runtimes("sync")
        runtime.create_lineage("main", streams=streams)
        reg = registry()
        refs = []
        for index in range(count):
            matrices, contexts = chunk(index, streams=streams)
            refs.append(runtime.commit_node("main", f"chunk-{index:03d}", matrices, contexts, method=method(lag),
                                            registry=reg, options=ArchiveOptions(encode_threads=1)).node_ref)
        return [node_content(runtime, ref) for ref in refs], reg.to_dict()

    def submit_chain(self, pipeline, count, *, streams=(), release=None, start=0):
        tickets = []
        for index in range(start, start + count):
            matrices, contexts = chunk(index, streams=streams)
            tickets.append(pipeline.submit("main", f"chunk-{index:03d}", matrices, contexts,
                                           release=None if release is None else (
                                               lambda event, matrices=matrices, i=index: release(i, matrices, event))))
        return tickets

    def test_pipelined_archive_equals_the_synchronous_reference(self):
        for lag in (0, 1):
            with self.subTest(rate_lag=lag):
                expected, expected_registry = self.sync_reference(3, lag=lag)
                runtime = self.runtimes("async")
                runtime.create_lineage("main")
                reg = registry()
                pipeline = self.pipeline(runtime, lag=lag, reg=reg, encode_workers=2, max_source_chunks=2,
                                         max_unpublished=3)
                tickets = self.submit_chain(pipeline, 3)
                head = pipeline.drain("main")
                self.assertEqual(head, tickets[-1].result())
                self.assertEqual([node_content(runtime, ticket.result()) for ticket in tickets], expected)
                self.assertEqual(reg.to_dict(), expected_registry)
                frontiers = pipeline.frontiers("main")
                self.assertEqual(frontiers["generated_frontier"], frontiers["published_frontier"])

    def test_publication_is_ordered_under_injected_delays(self):
        gate = Gate()
        gate.delays[("encode", "chunk-000")] = 0.8
        runtime = self.runtimes()
        runtime.create_lineage("main")
        pipeline = self.pipeline(runtime, hooks=gate, lag=1, encode_workers=2, max_source_chunks=2,
                                 max_unpublished=3)
        tickets = self.submit_chain(pipeline, 3)
        pipeline.drain("main", timeout=120)
        stored = [stage for stage in gate.order("store")]
        published = gate.order("publish")
        self.assertEqual(published, ["chunk-000", "chunk-001", "chunk-002"])
        self.assertNotEqual(stored[0], "chunk-000")  # chunk 1 finished encoding first...
        times = {node: t for stage, node, t in gate.events if stage == "publish"}
        self.assertLess(times["chunk-000"], times["chunk-001"])  # ...but published only after chunk 0
        path = runtime.node_path(runtime.lineage_head("main"))
        self.assertEqual([node.chunk_id for _, node, _ in path[1:]], ["chunk-000", "chunk-001", "chunk-002"])

    def test_failed_ancestor_blocks_descendants_and_retires_their_work(self):
        gate = Gate()
        gate.failures[("encode", "chunk-001")] = RuntimeError("injected encoder failure")
        runtime = self.runtimes()
        runtime.create_lineage("main")
        released = []
        pipeline = self.pipeline(runtime, hooks=gate, lag=1, encode_workers=2, max_source_chunks=3,
                                 max_unpublished=3)
        tickets = self.submit_chain(pipeline, 3, release=lambda index, matrices, event: released.append(index))
        first = tickets[0].result(timeout=120)
        with self.assertRaises(ArchiveFailure) as failed:
            tickets[1].result(timeout=120)
        self.assertIn("injected encoder failure", str(failed.exception))
        with self.assertRaises(ArchiveFailure) as descendant:
            tickets[2].result(timeout=120)
        self.assertIn("ancestor chunk-001", str(descendant.exception))
        with self.assertRaises(ArchiveFailure):
            pipeline.drain("main", timeout=60)
        with self.assertRaises(ArchiveFailure):
            self.submit_chain(pipeline, 1, start=3)
        head = pipeline.reset("main", timeout=60)
        self.assertEqual(head, first)
        self.assertEqual(runtime.lineage_head("main"), first)
        self.assertEqual(sorted(released), [0, 1, 2])
        # Nothing of the failed chunks stays reachable or leased; the lineage continues from its published head.
        self.assertEqual(runtime.store.collect_garbage()["objects_removed"], 0)
        self.assertEqual(runtime.store.stats()["leases"], 0)
        gate.failures.clear()
        tickets = self.submit_chain(pipeline, 1, start=1)
        self.assertEqual(runtime.load_node(tickets[0].result(timeout=120)).parent_ref, first)

    def test_cancellation_retires_leases_and_publishes_no_partial_state(self):
        gate = Gate()
        entered, release_gate = threading.Event(), threading.Event()
        gate.blocks[("plan", "chunk-001")] = (entered, release_gate)
        runtime = self.runtimes()
        runtime.create_lineage("main")
        released = []
        pipeline = self.pipeline(runtime, hooks=gate, lag=1, encode_workers=1, max_source_chunks=2,
                                 max_unpublished=3)
        tickets = self.submit_chain(pipeline, 3, release=lambda index, matrices, event: released.append(index))
        first = tickets[0].result(timeout=120)
        self.assertTrue(entered.wait(60))
        canceller = threading.Thread(target=lambda: setattr(self, "cancelled_head", pipeline.cancel("main",
                                                                                                    timeout=60)))
        canceller.start()
        time.sleep(0.2)
        release_gate.set()
        canceller.join(60)
        self.assertEqual(self.cancelled_head, first)
        for ticket in tickets[1:]:
            with self.assertRaises(ArchiveFailure):
                ticket.result(timeout=10)
        self.assertEqual(sorted(released), [0, 1, 2])
        self.assertEqual(released.count(1), 1)
        self.assertEqual(runtime.lineage_head("main"), first)
        self.assertNotIn("main", runtime._lineage_pending)
        self.assertEqual(len(runtime.node_path(first)), 2)

    def test_drain_waits_for_the_published_frontier(self):
        gate = Gate()
        gate.delays[("store", "chunk-001")] = 0.5
        runtime = self.runtimes()
        runtime.create_lineage("main")
        pipeline = self.pipeline(runtime, hooks=gate, encode_workers=2, max_source_chunks=2, max_unpublished=3)
        tickets = self.submit_chain(pipeline, 2)
        with self.assertRaises(PendingError):
            pipeline.drain("main", timeout=0.01)
        through = pipeline.drain("main", through="chunk-000", timeout=120)
        self.assertEqual(through, tickets[0].result())
        head = pipeline.drain("main", timeout=120)
        self.assertEqual(head, tickets[1].result())
        self.assertEqual(runtime.load_node(head).frontier, pipeline.frontiers("main")["generated_frontier"])
        with self.assertRaises(ConfigurationError):
            pipeline.drain("main", through="chunk-999")

    def test_bounded_queues_apply_explicit_backpressure(self):
        gate = Gate()
        entered, release_gate = threading.Event(), threading.Event()
        gate.blocks[("encode", "chunk-000")] = (entered, release_gate)
        runtime = self.runtimes()
        runtime.create_lineage("main")
        pipeline = self.pipeline(runtime, hooks=gate, encode_workers=1, max_source_chunks=1, max_unpublished=1)
        self.submit_chain(pipeline, 1)
        self.assertTrue(entered.wait(60))
        matrices, contexts = chunk(1)
        with self.assertRaises(CapacityError):
            pipeline.submit("main", "chunk-001", matrices, contexts, timeout=0)
        with self.assertRaises(CapacityError):
            pipeline.submit("main", "chunk-001", matrices, contexts, timeout=0.05)
        threading.Timer(0.3, release_gate.set).start()
        started = time.perf_counter()
        ticket = pipeline.submit("main", "chunk-001", matrices, contexts, timeout=60)
        self.assertGreater(time.perf_counter() - started, 0.2)
        ticket.result(timeout=120)
        self.assertGreaterEqual(pipeline.stats()["backpressure_waits"], 1)
        with self.assertRaises(ConfigurationError):
            ArchivePipeline(runtime, method=method(), registry=registry(), max_source_chunks=3, max_unpublished=2)
        with self.assertRaises(UnsupportedError):
            ArchivePipeline(runtime, method=method(), registry=registry(),
                            options=ArchiveOptions(packing="per_matrix"))

    def test_cfg_streams_publish_atomically(self):
        streams = ("cond", "uncond")
        expected, _ = self.sync_reference(2, streams=streams)
        runtime = self.runtimes()
        runtime.create_lineage("main", streams=streams)
        pipeline = self.pipeline(runtime, encode_workers=2, max_source_chunks=2, max_unpublished=3)
        tickets = self.submit_chain(pipeline, 2, streams=streams)
        pipeline.drain("main", timeout=120)
        contents = [node_content(runtime, ticket.result()) for ticket in tickets]
        self.assertEqual(contents, expected)
        self.assertTrue(all(content["streams"] == streams for content in contents))
        matrices, contexts = chunk(2, streams=streams)
        partial = {entry: value for entry, value in matrices.items() if entry[1] == "cond"}
        with self.assertRaises(ConfigurationError):
            pipeline.submit("main", "chunk-002", partial, {entry: contexts[entry] for entry in partial})
        gate = Gate()
        gate.failures[("finalize", "chunk-002")] = RuntimeError("one stream failed")
        pipeline.hooks = gate
        ticket = pipeline.submit("main", "chunk-002", matrices, contexts)
        with self.assertRaises(ArchiveFailure):
            ticket.result(timeout=120)
        self.assertEqual(pipeline.reset("main", timeout=60), tickets[-1].result())
        self.assertEqual(runtime.lineage_head("main"), tickets[-1].result())

    def test_sources_are_never_read_after_their_release(self):
        expected, _ = self.sync_reference(3, lag=1)
        runtime = self.runtimes()
        runtime.create_lineage("main")
        pipeline = self.pipeline(runtime, lag=1, encode_workers=2, max_source_chunks=1, max_unpublished=3)

        def poison(index, matrices, event):
            self.assertIsNone(event)  # CPU sources: no CUDA event to wait on
            for value in matrices.values():
                value.fill_(float("nan"))  # any read after release would corrupt or fail the archive

        tickets = self.submit_chain(pipeline, 3, release=poison)
        pipeline.drain("main", timeout=120)
        self.assertEqual([node_content(runtime, ticket.result()) for ticket in tickets], expected)

    def test_restart_recovers_from_the_published_head(self):
        expected, _ = self.sync_reference(4, lag=1)
        runtime = self.runtimes()
        runtime.create_lineage("main")
        gate = Gate()
        entered, release_gate = threading.Event(), threading.Event()
        gate.blocks[("store", "chunk-002")] = (entered, release_gate)
        reg = registry()
        pipeline = ArchivePipeline(runtime, method=method(1), registry=reg, options=ArchiveOptions(encode_threads=1),
                                   hooks=gate, encode_workers=1, max_source_chunks=2, max_unpublished=4)
        tickets = self.submit_chain(pipeline, 4)
        tickets[1].result(timeout=120)
        self.assertTrue(entered.wait(60))
        threading.Timer(0.3, release_gate.set).start()
        pipeline.cancel("main", timeout=60)
        pipeline.close(drain=False, timeout=60)
        root = runtime.store.root
        published = runtime.lineage_head("main")
        runtime.close()
        reopened = KVRuntime(root, MODEL)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.lineage_head("main"), published)
        reopened.store.collect_garbage()
        path = reopened.node_path(published)
        done = len(path) - 1
        self.assertEqual([node_content(reopened, ref) for ref, _, _ in path[1:]], expected[:done])
        resumed = ArchivePipeline(reopened, method=method(1), registry=reg, options=ArchiveOptions(encode_threads=1),
                                  encode_workers=2, max_source_chunks=2, max_unpublished=4)
        self.addCleanup(self._close, resumed)
        more = self.submit_chain(resumed, 4 - done, start=done)
        resumed.drain("main", timeout=120)
        final = reopened.node_path(more[-1].result())
        self.assertEqual([node_content(reopened, ref) for ref, _, _ in final[1:]], expected)


if __name__ == "__main__":
    unittest.main()
