"""CR-17-3 (SGLang Step 17): garbage collection leaves no cached state of reclaimed objects behind.

Step 5b found a deleted session's basis versions resident on the GPU after session GC (``DeviceBasisCache`` evicts
only above its 4 GiB budget; LongLive 2.0 ratcheted ~200 MiB per restore cycle). Their fix
(``LineageRestorer.forget_unstored``, ``DeviceBasisCache.forget``, ``ObjectStore.indexed``, called by the bridge's
``collect_garbage``) has its own tests (``tests/test_cr17_3_restorer_forget.py``). These cover the host side
the coordinator added: ``KVRuntime.collect_garbage`` also drops the parsed manifests (lineage nodes, matrix
records, node packs, epochs, basis versions) and basis memos of reclaimed objects
(``forget_unstored_manifests``), so the caches hold only live objects, and the bridge's GC releases both, with
the live sessions restored bitwise as before.
"""

from __future__ import annotations

import os
import shutil
import unittest
from unittest import mock

import torch

from reelcache.archive_restore import LineageRestorer
from reelcache.lineage_commit import ArchiveOptions
from reelcache.store import ObjectStore
import tests.test_sglang_bridge as t
from tests import step6_support as s6
from tests.test_packed_nvdec import SoftwarePool
from tests.test_step7_store_faults import _destination

STREAMS = ("cond", "uncond")


def cached_ids(runtime):
    return ([*runtime._lineage_nodes, *runtime._lineage_records, *runtime._lineage_packets,
             *runtime._lineage_epochs] + runtime.basis_versions.cached_ids())


class ManifestCacheTests(unittest.TestCase):
    def setUp(self):
        self.runtime = s6.RuntimeFactory(self)("cr173-host")
        registry = s6.registry()
        self.heads = {}
        for lineage, first in (("keep", 0), ("drop", 5)):
            self.runtime.create_lineage(lineage)
            for position in range(2):
                self.heads[lineage] = self.runtime.commit_node(
                    lineage, f"chunk-{position:03d}", *s6.chunk(first + position, token_start=position * s6.TOKENS),
                    method=s6.method(), registry=registry, options=ArchiveOptions(encode_threads=1)).node_ref

    def restore(self, head):
        caches, destination = _destination(2 * s6.TOKENS)
        LineageRestorer(self.runtime, SoftwarePool(), arena_bytes=1 << 30).restore(head, destination)
        return caches

    def test_gc_drops_the_parsed_manifests_of_reclaimed_objects_only(self):
        reference = self.restore(self.heads["keep"])
        self.restore(self.heads["drop"])  # parses every manifest of both lineages; view_bytes memoizes blocks
        for head in self.heads.values():  # restores (R1) skip matrix records: parse them too
            for _, node, _ in self.runtime.node_path(head):
                for _, record in node.matrix_records:
                    self.runtime.load_record(record)
        before = set(cached_ids(self.runtime))
        self.assertEqual(self.runtime.store.indexed(before), before)
        self.runtime.delete_lineage("drop")
        result = self.runtime.collect_garbage()
        forgotten = result["manifests_forgotten"]
        self.assertGreater(result["objects_removed"], 0)
        for name in ("nodes", "records", "packets", "basis_versions", "finite_blocks"):
            self.assertGreater(forgotten[name], 0, name)
        after = set(cached_ids(self.runtime))
        self.assertEqual(self.runtime.store.indexed(after), after)  # only live objects stay cached
        self.assertLess(len(after), len(before))
        # The live lineage's manifests stayed parsed (no store read) and it restores bitwise.
        with mock.patch.object(ObjectStore, "get_manifest_with_references",
                               side_effect=AssertionError("a cached live manifest was re-read")):
            self.runtime.node_path(self.heads["keep"])
        restored = self.restore(self.heads["keep"])
        for key in reference:
            self.assertTrue(torch.equal(restored[key], reference[key]), key)
        again = self.runtime.collect_garbage()["manifests_forgotten"]
        self.assertTrue(all(value == 0 for value in again.values()), again)

    def test_empty_caches_and_unchanged_store(self):
        self.runtime.drop_manifest_caches()
        report = self.runtime.forget_unstored_manifests()
        self.assertTrue(all(value == 0 for value in report.values()), report)
        self.assertEqual(set(report), {"nodes", "records", "packets", "epochs", "basis_versions", "finite_blocks",
                                       "view_digests"})
        self.runtime.node_path(self.heads["drop"])
        self.assertEqual(self.runtime.collect_garbage()["manifests_forgotten"]["nodes"], 0)  # nothing reclaimed


class BridgeReclaimTests(t.BridgeTestCase):
    """The bridge's GC (``delete`` control with gc) releases the deleted session's device bases, prepared packs and
    parsed manifests; the other session's are kept and it resumes bitwise as before."""

    representation = "svd_hevc"

    def packed(self, bridge):
        archive = bridge.archive
        pool = SoftwarePool()
        self.addCleanup(pool.close)
        archive.restorer = LineageRestorer(archive.runtime, pool, arena_bytes=1 << 30)
        return archive.restorer

    def test_deleting_a_session_releases_its_cached_restore_state(self):
        source = self.bridge(store="source")
        for session_id, prompt in (("s1", "A"), ("s2", "B")):
            t.drive(source, t.request(t.handle(session_id, pause_after_chunks=2), prompt=prompt, streams=STREAMS))
        source.close()
        results = {}
        for delete in (False, True):
            name = f"copy-{delete}"
            shutil.copytree(self.workspace / "source", self.workspace / name)
            bridge = self.bridge(store=name)
            restorer = self.packed(bridge)
            runtime = bridge.archive.runtime
            for session_id, prompt in (("s1", "A"), ("s2", "B")):  # prepared packs and resident bases of both
                bridge.control("open", {"session_id": session_id})
            self.assertTrue(restorer.bases.refs() and restorer.prepared.refs())
            if delete:
                report = bridge.control("delete", {"session_id": "s2"})["gc"]
                self.assertGreater(report["restorer_forgotten"]["versions"], 0)
                self.assertGreater(report["restorer_forgotten"]["packets"], 0)
                self.assertGreater(report["manifests_forgotten"]["packets"], 0)
                kept = restorer.bases.refs() + restorer.prepared.refs() + cached_ids(runtime)
                self.assertEqual(runtime.store.indexed(kept), set(kept))  # nothing of s2 stays cached
                self.assertTrue(restorer.bases.refs() and restorer.prepared.refs())  # s1's stay
            summary, latents, caches, _ = t.drive(bridge, t.request(t.handle("s1", "resume"), prompt="A",
                                                                     streams=STREAMS))
            self.assertEqual(summary["resumed_from_chunks"], 2)
            results[delete] = (latents, caches)
            self.assertLifetimesRetired(bridge)
            bridge.close()
        (latents, caches), (reference_latents, reference_caches) = results[True], results[False]
        self.assertTrue(torch.equal(latents, reference_latents))
        for stream in reference_caches:
            for got, want in zip(caches[stream], reference_caches[stream]):
                self.assertEqual(got.end, want.end)
                self.assertTrue(torch.equal(got.k[:, :got.end], want.k[:, :want.end]))
                self.assertTrue(torch.equal(got.v[:, :got.end], want.v[:, :want.end]))


if os.environ.get("REELCACHE_STEP7_GPU_TESTS") == "1":
    class CUDAReclaimTests(unittest.TestCase):
        """Device memory: a deleted lineage's seal-time-adopted bases are freed by the GC-time forget (GPU jobs)."""

        def test_forgotten_bases_return_their_device_memory(self):
            from reelcache.archive_restore import DeviceBasisCache
            if not torch.cuda.is_available():
                raise AssertionError("REELCACHE_STEP7_GPU_TESTS=1 requires a CUDA device")
            runtime = s6.RuntimeFactory(self)("cr173-cuda")
            registry = s6.registry()
            refs = {}
            for lineage, first in (("keep", 0), ("drop", 5)):
                runtime.create_lineage(lineage)
                for position in range(2):
                    commit = runtime.commit_node(
                        lineage, f"chunk-{position:03d}", *s6.chunk(first + position, token_start=position * s6.TOKENS),
                        method=s6.method(), registry=registry, options=ArchiveOptions(encode_threads=1))
                    for _, record_ref in commit.node.matrix_records:
                        refs.setdefault(lineage, set()).add(runtime.load_record(record_ref).basis_version_ref)
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            restorer = object.__new__(LineageRestorer)  # the restorer's caches on CUDA, no decoder pool
            restorer.runtime = runtime
            restorer.bases = DeviceBasisCache(runtime, "cuda", 1 << 30)
            from reelcache.archive_restore import _PreparedCache
            restorer.prepared = _PreparedCache(1 << 30)
            with runtime.store.transaction():
                items = [(ref, [item for item, _ in runtime.basis_versions.chain(ref)],
                          runtime.basis_versions.assemble(ref)) for ref in sorted(refs["keep"] | refs["drop"])]
            for item in sorted(items, key=lambda value: len(value[1])):  # seal-time adoption, commit order
                restorer.bases.adopt([item])
            torch.cuda.synchronize()
            resident = torch.cuda.memory_allocated() - baseline
            self.assertEqual(resident, restorer.bases.stats()["resident_bytes"])
            runtime.delete_lineage("drop")
            runtime.collect_garbage()
            report = restorer.forget_unstored()
            torch.cuda.synchronize()
            self.assertGreater(report["bytes"], 0)
            self.assertEqual(torch.cuda.memory_allocated() - baseline, resident - report["bytes"])
            self.assertEqual(restorer.bases.stats()["resident_bytes"], resident - report["bytes"])
            self.assertEqual(runtime.store.indexed(restorer.bases.refs()), set(restorer.bases.refs()))
            restorer.bases.clear()
            torch.cuda.synchronize()
            self.assertEqual(torch.cuda.memory_allocated(), baseline)


if __name__ == "__main__":
    unittest.main()
