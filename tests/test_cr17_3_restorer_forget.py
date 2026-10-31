"""CR-17-3: after the store's garbage collection, the restorer forgets the prepared packs and device-resident basis
versions of reclaimed objects (deleted / aborted sessions); live sessions' versions stay resident and correct."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from reelcache.archive_restore import DeviceBasisCache, LineageRestorer, _PreparedCache
from tests.step6_support import RuntimeFactory, chunk, method, registry, TOKENS


class _Prepared:
    def __init__(self, nbytes: int):
        self.stream_bytes = nbytes


class RestorerForgetTests(unittest.TestCase):
    def setUp(self):
        self.runtimes = RuntimeFactory(self)

    def commit(self, runtime, branch, indexes, *, reg):
        runtime.create_lineage(branch)
        commits = []
        for position, index in enumerate(indexes):
            matrices, contexts = chunk(index, token_start=position * TOKENS)
            commits.append(runtime.commit_node(branch, f"chunk-{position:03d}", matrices, contexts,
                                               method=method(), registry=reg))
        return commits

    def versions(self, runtime, commits):
        refs = {}
        for commit in commits:
            for _, record_ref in commit.node.matrix_records:
                record = runtime.load_record(record_ref)
                refs[record.basis_version_ref] = record.packet_ref
        return refs

    def restorer(self, runtime):
        """A LineageRestorer's cache state without its NVDEC pool (CPU)."""
        restorer = object.__new__(LineageRestorer)
        restorer.runtime = runtime
        restorer.bases = DeviceBasisCache(runtime, "cpu", 1 << 30)
        restorer.prepared = _PreparedCache(1 << 30)
        return restorer

    def adopt_all(self, runtime, restorer, refs):
        with runtime.store.transaction():
            chains = {ref: [item for item, _ in runtime.basis_versions.chain(ref)] for ref in refs}
            full = {ref: runtime.basis_versions.assemble(ref) for ref in refs}
        for ref in sorted(refs, key=lambda item: len(chains[item])):  # commit order: descendants absorb ancestors
            restorer.bases.adopt([(ref, chains[ref], full[ref])])
        return chains, full

    def test_gc_then_forget_drops_only_reclaimed_versions_and_packets(self):
        runtime = self.runtimes()
        reg = registry()
        keep = self.commit(runtime, "keep", [0, 1, 2], reg=reg)
        drop = self.commit(runtime, "drop", [7, 8, 9], reg=reg)
        keep_refs, drop_refs = self.versions(runtime, keep), self.versions(runtime, drop)
        self.assertFalse(set(keep_refs) & set(drop_refs), "precondition: the sessions share no version")
        restorer = self.restorer(runtime)
        self.adopt_all(runtime, restorer, list(keep_refs) + list(drop_refs))
        for packet in set(keep_refs.values()) | set(drop_refs.values()):
            restorer.prepared.put(packet, _Prepared(1000), [])
        before = restorer.bases.stats()
        resident_drop = [ref for ref in restorer.bases.refs() if ref in drop_refs]
        self.assertTrue(resident_drop)
        # Without the store GC nothing is forgotten (every object is still stored).
        self.assertEqual(restorer.forget_unstored()["versions"], 0)
        runtime.delete_lineage("drop")
        runtime.store.collect_garbage()
        report = restorer.forget_unstored()
        after = restorer.bases.stats()
        self.assertEqual(report["versions"], len(resident_drop))
        self.assertEqual(report["held"], 0)
        self.assertEqual(report["packets"], len(set(drop_refs.values())))
        self.assertFalse(set(restorer.bases.refs()) & set(drop_refs))
        self.assertFalse(set(restorer.prepared.refs()) & set(drop_refs.values()))
        self.assertEqual(after["resident_bytes"], before["resident_bytes"] - report["bytes"])
        self.assertEqual(after["resident_bytes"], sum(entry["bytes"] for entry in restorer.bases._entries.values()))
        # The live session is still served from residency, bitwise.
        requests = []
        for commit in keep:
            for _, record_ref in commit.node.matrix_records:
                record = runtime.load_record(record_ref)
                requests.append((record.basis_version_ref, record.rank_view))
        with runtime.store.transaction():
            expected = [runtime.basis_versions.assemble(ref, rank) for ref, rank in requests]
        uploads = restorer.bases.stats()["uploads"]
        views, tokens = restorer.bases.acquire_many(requests)
        self.assertEqual(restorer.bases.stats()["uploads"], uploads)
        for view, want in zip(views, expected):
            self.assertTrue(torch.equal(view, want))
        for token in tokens:
            restorer.bases.release(token)
        self.assertEqual(restorer.forget_unstored(), {"versions": 0, "bytes": 0, "held": 0, "packets": 0})

    def test_pinned_reclaimed_version_stays_until_released(self):
        runtime = self.runtimes()
        reg = registry()
        self.commit(runtime, "keep", [0], reg=reg)
        drop = self.commit(runtime, "drop", [7, 8], reg=reg)
        drop_refs = self.versions(runtime, drop)
        restorer = self.restorer(runtime)
        self.adopt_all(runtime, restorer, list(drop_refs))
        pinned_ref = restorer.bases.refs()[0]
        view, token = restorer.bases.acquire(pinned_ref, 1)
        runtime.delete_lineage("drop")
        runtime.store.collect_garbage()
        report = restorer.forget_unstored()
        self.assertEqual(report["held"], 1)
        self.assertIn(pinned_ref, restorer.bases.refs())
        restorer.bases.release(token)
        report = restorer.forget_unstored()
        self.assertEqual(report["versions"], 1)
        self.assertEqual(restorer.bases.refs(), [])
        self.assertEqual(restorer.bases.stats()["resident_bytes"], 0)

    def test_shared_ancestor_of_a_deleted_fork_is_still_served(self):
        runtime = self.runtimes()
        reg = registry()
        base = self.commit(runtime, "base", [0, 1, 2], reg=reg)
        runtime.fork_lineage(base[0].node_ref, "branch")
        matrices, contexts = chunk(11, token_start=TOKENS)
        branch = [runtime.commit_node("branch", "chunk-b1", matrices, contexts, method=method(), registry=reg)]
        base_refs, branch_refs = self.versions(runtime, base), self.versions(runtime, branch)
        restorer = self.restorer(runtime)
        self.adopt_all(runtime, restorer, list(dict.fromkeys(list(base_refs) + list(branch_refs))))
        runtime.delete_lineage("branch")
        runtime.store.collect_garbage()
        restorer.forget_unstored()
        present = runtime.store.indexed(restorer.bases.refs())
        self.assertEqual(present, set(restorer.bases.refs()))
        self.assertTrue(set(restorer.bases._chain_index.values()) <= set(restorer.bases._entries))
        requests = []
        for commit in base:
            for _, record_ref in commit.node.matrix_records:
                record = runtime.load_record(record_ref)
                requests.append((record.basis_version_ref, record.rank_view))
        with runtime.store.transaction():
            expected = [runtime.basis_versions.assemble(ref, rank) for ref, rank in requests]
        views, tokens = restorer.bases.acquire_many(requests)
        for view, want in zip(views, expected):
            self.assertTrue(torch.equal(view, want))
        for token in tokens:
            restorer.bases.release(token)

    def test_store_indexed_reports_present_ids_without_raising(self):
        runtime = self.runtimes()
        commits = self.commit(runtime, "keep", [0], reg=registry())
        ref = commits[0].node_ref
        absent = "manifest:" + "0" * 64
        self.assertEqual(runtime.store.indexed([ref, absent, ref]), {ref})
        self.assertEqual(runtime.store.indexed([]), set())

    def test_bridge_collect_garbage_forgets_restorer_caches(self):
        from reelcache.integrations.sglang.lineage_archive import LineageSessionArchive
        archive = object.__new__(LineageSessionArchive)
        archive.runtime = mock.Mock()
        archive.runtime.collect_garbage.return_value = {"objects_removed": 3, "bytes_reclaimed": 10, "live_objects": 1}
        archive._records, archive._states = {"a": 1}, {"b": 2}
        archive.restorer = mock.Mock()
        archive.restorer.forget_unstored.return_value = {"versions": 2, "bytes": 64, "held": 0, "packets": 1}
        report = archive.collect_garbage()
        archive.restorer.forget_unstored.assert_called_once_with()
        self.assertEqual(report["restorer_forgotten"]["versions"], 2)
        self.assertEqual((archive._records, archive._states), ({}, {}))
        archive.restorer = object()  # SoftwareLineageRestorer: no caches, nothing to forget
        self.assertNotIn("restorer_forgotten", archive.collect_garbage())


if __name__ == "__main__":
    unittest.main()
