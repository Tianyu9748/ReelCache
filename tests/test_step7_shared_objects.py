"""CR-14-7 (SGLang Step 14 M3): an abort never collects objects another pending commit or a published node uses.

Two sessions committing identical content share content-addressed objects in one store. A commit pinned only the
basis versions it created, so a version that another session had written first stayed unpinned between planning and
storing the node pack; that session's abort collected it and the commit then failed with "missing immutable object".
Commits now pin every object they reference in the transaction that writes or finds it, and record every object they
touched as collection candidates (collection removes only objects nothing published or pinned reaches). The store's
``hooks(event, object_id)`` (after each committed transaction, outside every store lock: "created" / "existing" /
"deleted") drives the interleavings deterministically; pipeline hooks cannot, because both sessions use the same node
ids.
"""

from __future__ import annotations

import threading
import time
import unittest

from reelcache.archive_pipeline import ArchivePipeline
from reelcache.lineage_commit import ArchiveOptions
from tests import step6_support as s6


def _objects(runtime) -> set[str]:
    with runtime.store.transaction():
        return {row[0] for row in runtime.store._db.execute("SELECT id FROM objects")}


def _cancel(test, pipeline, branch, release, cancelled_before):
    """Cancel ``branch`` while its chunk is held; release the chunk once the cancellation is recorded."""
    canceller = threading.Thread(target=lambda: pipeline.cancel(branch, timeout=120))
    canceller.start()
    deadline = time.monotonic() + 60
    while pipeline.stats()["cancelled"] <= cancelled_before and time.monotonic() < deadline:
        time.sleep(0.01)
    test.assertGreater(pipeline.stats()["cancelled"], cancelled_before)
    release.set()
    canceller.join(120)
    test.assertFalse(canceller.is_alive())


class SharedObjectTests(unittest.TestCase):
    def setUp(self):
        self.runtimes = s6.RuntimeFactory(self)

    def pipeline(self, runtime, hooks=None):
        pipeline = ArchivePipeline(runtime, method=s6.method(), registry=s6.registry(),
                                   options=ArchiveOptions(encode_threads=1), hooks=hooks, encode_workers=2,
                                   max_source_chunks=2, max_unpublished=4)
        self.addCleanup(self._close, pipeline)
        return pipeline

    @staticmethod
    def _close(pipeline):
        try:
            pipeline.close(drain=False, timeout=60)
        except BaseException:  # noqa: BLE001 - tests assert failures themselves
            pass

    def reference(self):
        runtime = self.runtimes("reference")
        runtime.create_lineage("main")
        pipeline = self.pipeline(runtime)
        pipeline.submit("main", "chunk-000", *s6.chunk(0))
        return s6.node_content(runtime, pipeline.drain("main", timeout=300))

    def test_an_abort_never_collects_a_version_another_pending_commit_shares(self):
        expected = self.reference()
        runtime = self.runtimes("interleaving")
        runtime.create_lineage("a")
        runtime.create_lineage("b")
        a_at_encode, release_a, encodes = threading.Event(), threading.Event(), []

        def pipeline_hooks(stage, node):
            if stage == "encode" and not encodes:  # session a (b is held in the fitter below)
                encodes.append(node)
                a_at_encode.set()
                if not release_a.wait(60):
                    raise RuntimeError("test gate a was not released")
        pipeline = self.pipeline(runtime, pipeline_hooks)
        watching, b_holding, release_b, shared, deleted = (threading.Event(), threading.Event(), threading.Event(),
                                                           [], [])

        def store_hooks(event, object_id):
            if event == "deleted":
                deleted.append(object_id)
            elif (event == "existing" and object_id.startswith("manifest:") and watching.is_set()
                  and not b_holding.is_set()):
                shared.append(object_id)  # b's planning found a basis version session a wrote first
                b_holding.set()
                if not release_b.wait(60):
                    raise RuntimeError("test gate b was not released")
        runtime.store.hooks = store_hooks
        pipeline.submit("a", "chunk-000", *s6.chunk(0))
        self.assertTrue(a_at_encode.wait(120))  # a planned: its versions are written; a waits before encoding
        watching.set()
        pipeline.submit("b", "chunk-000", *s6.chunk(0))  # identical content and node id
        self.assertTrue(b_holding.wait(120))  # b holds a shared version between planning and storing its pack
        _cancel(self, pipeline, "a", release_a, 0)  # a retires; the cancellation collects a's candidates
        release_b.set()
        head = pipeline.drain("b", timeout=300)  # before the fix: ArchiveFailure ... "missing immutable object"
        self.assertNotIn(shared[0], deleted)
        self.assertIn(shared[0], _objects(runtime))
        runtime.store.verify(head, recursive=True)
        self.assertEqual(s6.node_content(runtime, head), expected)

    def test_a_full_collection_between_a_commits_transactions_keeps_its_objects(self):
        for packing in ("node", "per_matrix"):
            with self.subTest(packing=packing):
                runtime = self.runtimes(f"gc-{packing}")
                runtime.create_lineage("main")
                collected = []

                def store_hooks(event, object_id, runtime=runtime, collected=collected):
                    # After the first transaction that writes a basis version: collect everything unreachable.
                    if event == "created" and object_id.startswith("manifest:") and not collected:
                        collected.append(runtime.store.collect_garbage())
                runtime.store.hooks = store_hooks
                commit = runtime.commit_node("main", "chunk-000", *s6.chunk(0), method=s6.method(),
                                             registry=s6.registry(),
                                             options=ArchiveOptions(encode_threads=1, packing=packing))
                runtime.store.hooks = None
                self.assertTrue(collected)
                runtime.store.verify(commit.node_ref, recursive=True)

    def test_an_unshared_abort_frees_its_objects_and_shared_ones_go_with_the_last_holder(self):
        runtime = self.runtimes("abort-only")
        runtime.create_lineage("a")
        gates, order = {}, []

        def hooks(stage, node):
            if stage == "encode" and len(order) < 2:
                name = "ab"[len(order)]
                order.append(name)
                entered, release = gates[name]
                entered.set()
                if not release.wait(60):
                    raise RuntimeError("test gate was not released")
        pipeline = self.pipeline(runtime, hooks)
        gates.update({name: (threading.Event(), threading.Event()) for name in ("a", "b")})
        baseline = _objects(runtime)
        pipeline.submit("a", "chunk-000", *s6.chunk(0))
        self.assertTrue(gates["a"][0].wait(120))
        self.assertTrue(_objects(runtime) - baseline)
        _cancel(self, pipeline, "a", gates["a"][1], 0)
        self.assertEqual(_objects(runtime) - baseline, set())  # unshared: every object of the abort is freed
        # Shared: b pins what a wrote, so a's abort keeps it; b's abort, the last holder, frees it.
        runtime.create_lineage("b")
        order.clear()
        gates.update({name: (threading.Event(), threading.Event()) for name in ("a", "b")})
        baseline = _objects(runtime)
        pipeline.submit("a", "chunk-000", *s6.chunk(0))
        self.assertTrue(gates["a"][0].wait(120))
        written = _objects(runtime) - baseline
        pipeline.submit("b", "chunk-000", *s6.chunk(0))
        self.assertTrue(gates["b"][0].wait(120))
        cancelled = pipeline.stats()["cancelled"]
        _cancel(self, pipeline, "a", gates["a"][1], cancelled)
        self.assertTrue(written & _objects(runtime))  # b's pending commit still pins the shared objects
        _cancel(self, pipeline, "b", gates["b"][1], cancelled + 1)
        self.assertEqual(_objects(runtime) - baseline, set())

    def test_collection_still_reclaims_unreachable_objects(self):
        runtime = self.runtimes("reclaim")
        heads = {}
        for name, contents in (("a", (0, 1)), ("b", (0, 2))):  # identical first chunk, different second
            runtime.create_lineage(name)
            pipeline = self.pipeline(runtime)
            for position, index in enumerate(contents):
                matrices, contexts = s6.chunk(index, token_start=position * s6.TOKENS)
                pipeline.submit(name, f"chunk-{position:03d}", matrices, contexts)
            heads[name] = pipeline.drain(name, timeout=300)
            pipeline.close()
        everything = _objects(runtime)
        runtime.delete_lineage("a")
        report = runtime.collect_garbage()
        remaining = _objects(runtime)
        self.assertLess(len(remaining), len(everything))  # a's own objects are reclaimed
        self.assertEqual(report["live_objects"], len(remaining))
        runtime.store.verify(heads["b"], recursive=True)  # objects b shares with a stay


if __name__ == "__main__":
    unittest.main()
