"""CR-17-1 (SGLang Step 17): restore reads are verified by default and a damaged store head never stops a server.

Step 17 (job 4408845) found two failure modes of a corrupted store. With the bridge's former default
``verify_reads=False`` a corrupted coefficient packet restored into wrong KV without an error. With
``verify_reads=True`` the open hash-audited every head (``KVRuntime.__init__``), one corrupt object raised
IntegrityError and the bridge, hence the server, failed to start. Now:

* ``KVRuntime(audit_on_open=None|bool)`` decouples the open-time audit from read verification (None keeps the old
  coupling); ``verify_references=False`` keeps publication, leases and GC at their index-check cost while every read
  for use is hash-checked (the old ``verify_reads=True`` also re-hashed every reachable object on each publication).
* ``KVRuntime(on_open_error="raise"|"quarantine")``: a damaged head fails the open (old default) or is quarantined:
  listed in ``runtime.quarantined``, ``QuarantinedError`` (an IntegrityError) whenever it is loaded, deletable, and
  GC keeps damaged graphs (store ``on_damage="keep"``) instead of failing for every session.
* The SGLang bridge defaults to ``verify_reads=True, audit_on_open=False, on_open_error="quarantine"``.
"""

from __future__ import annotations

import os
import shutil
import unittest
import uuid
from unittest import mock

import torch

from reelcache import restore_kernels as rk
from reelcache.archive_restore import LineageRestorer
from reelcache.errors import ConfigurationError, IntegrityError, QuarantinedError
from reelcache.integrations.sglang.controls import error_of
from reelcache.lineage_commit import ArchiveOptions
from reelcache.manager import KVRuntime
from reelcache.store import ObjectStore
from reelcache.types import MatrixKey
import tests.test_sglang_bridge as t
from tests import step6_support as s6
from tests.core_support import CoreWorkspaceTest, metadata as core_metadata, model as core_model
from tests.test_adaptive_manager import inputs as adaptive_inputs
from tests.test_packed_nvdec import SoftwarePool

SERVING = {"verify_reads": True, "verify_references": False, "audit_on_open": False, "on_open_error": "quarantine"}
drive = t.drive


def _commit(runtime, registry, lineage, first, count=2, start=0):
    head = None
    for position in range(start, start + count):
        head = runtime.commit_node(lineage, f"chunk-{position:03d}",
                                   *s6.chunk(first + position, token_start=position * s6.TOKENS),
                                   method=s6.method(), registry=registry,
                                   options=ArchiveOptions(encode_threads=1)).node_ref
    return head


def _reachable(store, head):
    with store.transaction():
        return store._walk([head], verify=False)


def _indexed(store):
    with store.transaction():
        return {row[0] for row in store._db.execute("SELECT id FROM objects")}


def _corrupt(store, object_id):
    """Change one byte in place (same length): a payload restores to wrong values unless reads are verified."""
    path = store.path_for(object_id)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x40
    path.write_bytes(bytes(data))


def _corrupt_structure(store, object_id):
    """Change the first double quote at or after a manifest's midpoint (same length): the manifest no longer parses.
    ``_corrupt``'s midpoint byte itself depends on the layout and may fall inside a value, leaving valid JSON."""
    path = store.path_for(object_id)
    data = bytearray(path.read_bytes())
    data[data.index(b'"', len(data) // 2)] ^= 0x40
    path.write_bytes(bytes(data))


def _corrupt_inside_a_string(store, object_id):
    """Change one letter inside a string of a manifest's data section (same length): the JSON still parses, so only
    a verified read (hash) finds it. Independent of the manifest's layout (the midpoint byte may be structural)."""
    path = store.path_for(object_id)
    data = bytearray(path.read_bytes())
    end = data.rindex(b'"references":')  # canonical JSON: sorted keys, so the last one is the top-level list
    inside = escaped = False
    for position in range(end):
        byte = data[position]
        if inside and not escaped and position >= end // 2 and 0x61 <= byte <= 0x7A and byte != 0x62:
            data[position] ^= 0x40  # a lowercase letter (not "b", which would become a quote) -> printable ASCII
            path.write_bytes(bytes(data))
            return position
        if escaped:
            escaped = False
        elif inside and byte == 0x5C:
            escaped = True
        elif byte == 0x22:
            inside = not inside
    raise AssertionError(f"no string letter to change in {object_id}")


def _destination(tokens):
    caches = {}

    def destination(node_id, key, tile):
        key = key if isinstance(key, MatrixKey) else key[0]
        if key not in caches:
            caches[key] = torch.zeros((1, tokens, 1, s6.CHANNELS), dtype=torch.float32)
        return rk.NativeView(caches[key], "BTHD", token_offset=tile.token_start)
    return caches, destination


class StoreFaultCase(unittest.TestCase):
    """Two lineages ``a`` and ``b`` (two chunks each) in a closed store; ``a``'s newest node pack is the target."""

    def setUp(self):
        self.root = s6.ROOT / f"cr17-{uuid.uuid4().hex}"
        self.addCleanup(shutil.rmtree, self.root, True)
        runtime = KVRuntime(self.root, s6.MODEL)
        try:
            self.registry = s6.registry()
            self.heads = {}
            for lineage, first in (("a", 0), ("b", 4)):
                runtime.create_lineage(lineage)
                self.heads[lineage] = _commit(runtime, self.registry, lineage, first)
            self.reference = {name: self.restore(runtime, head) for name, head in self.heads.items()}
            node = runtime.load_node(self.heads["a"])
            self.pack = node.packet_ref
            self.payloads = runtime.load_packet(self.pack)["payloads"]
            only_a = _reachable(runtime.store, self.heads["a"]) - _reachable(runtime.store, self.heads["b"])
            self.assertIn(self.pack, only_a)
            self.assertTrue(set(self.payloads.values()) <= only_a)
        finally:
            runtime.close()

    def open(self, **flags):
        runtime = KVRuntime(self.root, s6.MODEL, **flags)
        self.addCleanup(lambda: runtime._closed or runtime.close())
        return runtime

    def restore(self, runtime, head, restorer=None):
        caches, destination = _destination(2 * s6.TOKENS)
        (restorer or LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 30)).restore(head, destination)
        return caches

    def assertSameCaches(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]), key)

    def path(self, object_id):
        with ObjectStore(self.root) as store:
            return store.path_for(object_id)

    def corrupt(self, object_id, inside_a_string=False):
        with ObjectStore(self.root) as store:
            (_corrupt_inside_a_string if inside_a_string else _corrupt)(store, object_id)


class OpenAuditTests(StoreFaultCase):
    def test_audit_on_open_none_keeps_the_old_coupling(self):
        self.corrupt(self.payloads["affine.000.f32"])
        with self.assertRaisesRegex(IntegrityError, "hash or length mismatch"):
            KVRuntime(self.root, s6.MODEL, verify_reads=True)  # old: verify_reads audits every head at open
        with self.assertRaisesRegex(IntegrityError, "hash or length mismatch"):
            KVRuntime(self.root, s6.MODEL, audit_on_open=True)
        old = self.open()  # old default: the graph check (manifests parse, references exist) passes
        self.assertEqual((old.verify_reads, old.audit_on_open, old.on_open_error), (False, False, "raise"))
        self.assertEqual((old.store.verify_references, old.store.on_damage), (False, "raise"))
        old.close()
        decoupled = self.open(verify_reads=True, audit_on_open=False)
        self.assertEqual((decoupled.audit_on_open, decoupled.store.verify_references), (False, True))
        self.assertEqual(decoupled.quarantined, {})
        for bad in ({"audit_on_open": 1}, {"on_open_error": "ignore"}, {"verify_references": "no"}):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                KVRuntime(self.root, s6.MODEL, **bad)

    def test_a_damaged_head_raises_by_default_and_is_quarantined_on_request(self):
        self.corrupt(self.pack, inside_a_string=True)  # a manifest: the graph check reads (verify_reads hashes) it
        for flags in ({"verify_reads": True}, {"verify_reads": True, "audit_on_open": False}):
            with self.subTest(flags=flags), self.assertRaisesRegex(IntegrityError, "hash or length mismatch"):
                KVRuntime(self.root, s6.MODEL, **flags)
        self.open().close()  # unverified reads: the changed byte still parses (only structure is checked)
        runtime = self.open(**SERVING)
        self.assertEqual(set(runtime.quarantined), {"lineages/a"})
        self.assertIn("IntegrityError", runtime.quarantined["lineages/a"])
        self.assertEqual(runtime.list_lineages(), self.heads)  # refs are listed and kept (GC roots)
        self.assertEqual(runtime.audit(deep=True), runtime.quarantined)  # explicit audit: never raises for damage

    def test_missing_manifest_and_full_audit_quarantine(self):
        self.path(self.pack).unlink()
        runtime = self.open(**SERVING)
        self.assertIn("cannot read immutable object", runtime.quarantined["lineages/a"])
        self.assertSameCaches(self.restore(runtime, self.heads["b"]), self.reference["b"])
        runtime.close()
        self.setUp()  # a fresh store
        self.corrupt(self.payloads["stream.hevc"])  # payload damage: only a hash audit finds it at open
        plain = self.open(**SERVING)
        self.assertEqual(plain.quarantined, {})
        plain.close()
        audited = self.open(**{**SERVING, "audit_on_open": True})
        self.assertEqual(set(audited.quarantined), {"lineages/a"})
        self.assertIn("hash or length mismatch", audited.quarantined["lineages/a"])


class VerifiedReadTests(StoreFaultCase):
    def test_a_corrupt_packet_restores_wrong_without_verification(self):
        """The CR-17-1 failure mode: verification off restores a corrupted packet into wrong KV, no error."""
        self.corrupt(self.payloads["affine.001.f32"])
        runtime = self.open()
        wrong = self.restore(runtime, self.heads["a"])
        self.assertTrue(any(not torch.equal(wrong[key], self.reference["a"][key]) for key in wrong))

    def test_a_corrupt_packet_fails_the_restore_typed_and_changes_nothing(self):
        self.corrupt(self.payloads["affine.001.f32"])
        runtime = self.open(**SERVING)
        self.assertEqual(runtime.quarantined, {})  # payload damage is invisible to the open's graph check
        restorer = LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 30)
        before = (runtime.list_lineages(), runtime.store.stats()["leases"], runtime.active_leases,
                  runtime.resident_bytes, restorer.stats()["restorer"]["arena"]["used_bytes"])
        caches, destination = _destination(2 * s6.TOKENS)
        with self.assertRaisesRegex(IntegrityError, "hash or length mismatch"):
            restorer.restore(self.heads["a"], destination)
        self.assertTrue(all(not value.any() for value in caches.values()))  # nothing was written
        after = (runtime.list_lineages(), runtime.store.stats()["leases"], runtime.active_leases,
                 runtime.resident_bytes, restorer.stats()["restorer"]["arena"]["used_bytes"])
        self.assertEqual(after, before)
        self.assertEqual(restorer.stats()["bases"]["pinned"], 0)
        # The other lineage restores exactly, and archival continues on it.
        self.assertSameCaches(self.restore(runtime, self.heads["b"], restorer), self.reference["b"])
        head = _commit(runtime, self.registry, "b", 4, count=1, start=2)
        self.assertEqual(runtime.lineage_head("b"), head)
        with self.assertRaises(IntegrityError):  # the explicit audit stays a full forced hash audit
            runtime.store.verify(self.heads["a"])

    def test_staged_cold_restore_fails_typed(self):
        from reelcache.restore_staging import StagingEngine
        self.corrupt(self.payloads["stream.hevc"])
        runtime = self.open(**SERVING)
        engine = StagingEngine("cpu", threads=2, prepare_threads=2, pinned_bytes=1 << 22)
        self.addCleanup(engine.close)
        restorer = LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 30, staging_engine=engine)
        caches, destination = _destination(2 * s6.TOKENS)
        with self.assertRaisesRegex(IntegrityError, "hash or length mismatch"):
            restorer.restore(self.heads["a"], destination)
        self.assertTrue(all(not value.any() for value in caches.values()))
        self.assertSameCaches(self.restore(runtime, self.heads["b"], restorer), self.reference["b"])


class QuarantineTests(StoreFaultCase):
    def test_quarantined_lineage_is_typed_everywhere_and_the_others_work(self):
        self.corrupt(self.pack)
        runtime = self.open(**SERVING)
        head = self.heads["a"]
        for operation in (lambda: runtime.load_node(head), lambda: runtime.node_path(head),
                          lambda: runtime.resolve_history(head), lambda: runtime.fork_lineage(head, "c"),
                          lambda: runtime.rewind_lineage("a", head), lambda: self.restore(runtime, head),
                          lambda: _commit(runtime, self.registry, "a", 0, count=1, start=2)):
            with self.subTest(operation=operation), self.assertRaisesRegex(QuarantinedError, "lineages/a"):
                operation()
        self.assertIsInstance(QuarantinedError("x"), IntegrityError)
        self.assertNotIn("c", runtime.list_lineages())
        self.assertEqual(runtime.list_lineages()["a"], head)
        # Other lineages restore, fork, commit; the restorer's session prewarm skips the quarantined head.
        restorer = LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 30)
        self.assertEqual(restorer.prewarm_lineages()["quarantined"], ["a"])
        self.assertSameCaches(self.restore(runtime, self.heads["b"], restorer), self.reference["b"])
        runtime.fork_lineage(self.heads["b"], "c")
        _commit(runtime, self.registry, "c", 4, count=1, start=2)
        # GC keeps the damaged graph (no reads of it) and collects garbage elsewhere instead of failing.
        runtime.delete_lineage("c")
        before = _indexed(runtime.store)
        result = runtime.collect_garbage()
        self.assertGreater(result["objects_removed"], 0)
        self.assertGreaterEqual(result["damaged_objects"], 1)
        self.assertIn(self.pack, runtime.store.damaged)
        self.assertTrue(_reachable_by_edges(runtime.store, head) <= _indexed(runtime.store) <= before)
        # Deleting the quarantined head lifts its quarantine; its objects become garbage.
        runtime.delete_lineage("a")
        self.assertEqual(runtime.quarantined, {})
        runtime.collect_garbage()
        self.assertNotIn(self.pack, _indexed(runtime.store))
        self.assertEqual(runtime.store.damaged, {})  # the latest collection found no damage
        runtime.create_lineage("a")  # the name is reusable
        self.assertSameCaches(self.restore(runtime, self.heads["b"]), self.reference["b"])

    def test_gc_with_a_damaged_graph_raises_under_the_old_policy(self):
        runtime = self.open(verify_reads=True, verify_references=False)
        _corrupt(runtime.store, self.pack)
        with self.assertRaisesRegex(IntegrityError, "hash or length mismatch"):
            runtime.collect_garbage()

class BranchQuarantineTests(CoreWorkspaceTest):
    def test_branch_and_snapshot_heads_are_quarantined_too(self):
        root = self.workspace / "runtime"
        runtime = KVRuntime(root, core_model(layers=1))
        runtime.create_branch("main")
        runtime.create_branch("other")
        runtime.seal_chunk("main", "c0", *adaptive_inputs(runtime.model))
        runtime.seal_chunk("other", "c0", *adaptive_inputs(runtime.model, rank=2))
        tensors = {"rng": torch.Generator().manual_seed(19).get_state(), "latent": torch.arange(4.0)}
        snapshot = runtime.snapshot("main", metadata=core_metadata(), tensors=tensors)
        chunk = runtime.store.get_manifest(runtime.store.get_ref("branches/main"))["chunks"][0]
        self.assertNotIn(chunk, runtime.store.get_manifest(runtime.store.get_ref("branches/other"))["chunks"])
        _corrupt_structure(runtime.store, chunk)
        runtime.close()
        with self.assertRaises(IntegrityError):
            KVRuntime(root, core_model(layers=1))
        runtime = KVRuntime(root, core_model(layers=1), on_open_error="quarantine")
        self.addCleanup(runtime.close)
        self.assertEqual(set(runtime.quarantined), {"branches/main", "snapshots/" + snapshot.split(":")[1]})
        with self.assertRaises(QuarantinedError):
            runtime.branch_info("main")
        with self.assertRaises(QuarantinedError):
            runtime.load_snapshot(snapshot)
        with self.assertRaises(QuarantinedError):
            runtime.fork(snapshot, "copy")
        self.assertEqual(runtime.branch_info("other")["chunk_ids"], ["c0"])
        runtime.delete_snapshot(snapshot)
        runtime.delete_branch("main")
        self.assertEqual(runtime.quarantined, {})
        runtime.collect_garbage()
        self.assertNotIn(chunk, _indexed(runtime.store))
        self.assertEqual(runtime.branch_info("other")["chunk_ids"], ["c0"])


def _reachable_by_edges(store, head):
    with store.transaction():
        pending, seen = [head], set()
        while pending:
            item = pending.pop()
            if item not in seen:
                seen.add(item)
                pending.extend(row[0] for row in store._db.execute("SELECT child FROM edges WHERE parent=?", (item,)))
        return seen


class ReferenceVerificationTests(unittest.TestCase):
    """``verify_references=False`` keeps graph operations at index cost; strict mode (None) re-hashes the history."""

    def hashed_per_commit(self, count, **flags):
        """Bytes hash-checked by each of ``count`` successive commits of one lineage."""
        root = s6.ROOT / f"cr17-cost-{uuid.uuid4().hex}"
        self.addCleanup(shutil.rmtree, root, True)
        runtime = KVRuntime(root, s6.MODEL, **flags)
        self.addCleanup(lambda: runtime._closed or runtime.close())
        registry = s6.registry()
        runtime.create_lineage("main")
        original, hashed, per_commit = ObjectStore._get, [], []

        def counting(store, object_id, *, verify=None):
            data = original(store, object_id, verify=verify)
            if (store.verify_reads if verify is None else verify):
                hashed.append(len(data))
            return data
        with mock.patch.object(ObjectStore, "_get", counting):
            for position in range(count):
                hashed.clear()
                _commit(runtime, registry, "main", 0, count=1, start=position)
                per_commit.append(sum(hashed))
        return per_commit

    def test_publication_cost(self):
        strict = self.hashed_per_commit(6, verify_reads=True)
        serving = self.hashed_per_commit(6, verify_reads=True, verify_references=False)
        # Strict mode re-hashes everything reachable on every publication: it grows with the history.
        self.assertGreater(strict[5], 1.5 * strict[1])
        # Serving hashes what each commit reads for use (manifests, the parent's basis blocks): constant once the
        # version chains exist, whatever the history length.
        self.assertLessEqual(serving[5], 1.05 * serving[2])
        self.assertLess(serving[5], strict[5] / 5)

    def test_reads_only_mode_detects_on_read_not_on_graph_operations(self):
        root = s6.ROOT / f"cr17-store-{uuid.uuid4().hex}"
        self.addCleanup(shutil.rmtree, root, True)
        with ObjectStore(root, verify_reads=True, verify_references=False) as store:
            blob = store.put(b"correct")
            store.set_ref("branch", blob)
            store.path_for(blob).write_bytes(b"corrupt")
            self.assertEqual(store.put(b"correct"), blob)  # dedup, publication, leases and GC check the index
            store.set_ref("new", blob)
            store.lease(blob).release()
            store.gc()
            with self.assertRaisesRegex(IntegrityError, "hash or length mismatch"):
                store.get(blob)
            with self.assertRaisesRegex(IntegrityError, "hash or length mismatch"):
                store.get_many([blob])
            with self.assertRaises(IntegrityError):
                store.verify(blob)

    def test_gc_damage_policy_collects_healthy_stores_identically(self):
        results = []
        for policy in ("raise", "keep"):
            root = s6.ROOT / f"cr17-gc-{policy}-{uuid.uuid4().hex}"
            self.addCleanup(shutil.rmtree, root, True)
            runtime = KVRuntime(root, s6.MODEL, on_open_error="raise" if policy == "raise" else "quarantine")
            self.assertEqual(runtime.store.on_damage, policy)
            registry = s6.registry()
            for lineage in ("a", "b"):
                runtime.create_lineage(lineage)
                _commit(runtime, registry, lineage, 0 if lineage == "a" else 4)
            runtime.delete_lineage("a")
            result = runtime.collect_garbage()
            results.append((result, sorted(_indexed(runtime.store))))
            runtime.close()
        self.assertEqual(results[0], results[1])
        self.assertNotIn("damaged_objects", results[0][0])


class BridgeConfigTests(unittest.TestCase):
    def test_serving_defaults_and_validation(self):
        from reelcache.integrations.sglang import ReelCacheRuntimeConfig
        config = ReelCacheRuntimeConfig.from_dict({"store_root": "archive"})
        self.assertEqual((config.verify_reads, config.audit_on_open, config.on_open_error), (True, False, "quarantine"))
        self.assertEqual(ReelCacheRuntimeConfig.from_dict(config.to_dict()), config)
        for bad in ({"audit_on_open": 1}, {"on_open_error": "ignore"}, {"verify_reads": None}):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                ReelCacheRuntimeConfig.from_dict({"store_root": "archive", **bad})


class BridgeFaultChecks:
    """Bridge behavior on a damaged store (both representations): typed request errors, no server exit."""

    def archived(self, **overrides):
        """Sessions s1 (prompt A, two chunks) and s2 (prompt B, one chunk) published by a closed bridge."""
        bridge = self.bridge(**overrides)
        runtime = bridge.archive.runtime
        self.assertEqual((runtime.verify_reads, runtime.store.verify_references, runtime.audit_on_open,
                          runtime.on_open_error, runtime.store.on_damage),
                         (True, False, False, "quarantine", "keep"))
        drive(bridge, t.request(t.handle("s1", pause_after_chunks=2), prompt="A"))
        drive(bridge, t.request(t.handle("s2", pause_after_chunks=1), prompt="B"))
        heads = {session_id: bridge.archive.head(session_id) for session_id in ("s1", "s2")}
        root, self.identity = bridge.archive.root, bridge.model
        bridge.close()
        with ObjectStore(root) as store:
            only_s1 = _reachable(store, heads["s1"]) - _reachable(store, heads["s2"])
            blobs = sorted((store.object_size(item), item) for item in only_s1 if item.startswith("blob:"))
        self.assertIn(heads["s1"], only_s1)
        return root, heads, blobs[-1][1]

    def assertBaseline(self, bridge):
        self.assertFalse(bridge.active_request)
        self.assertEqual(rk.pending_deferred(), 0)
        self.assertEqual(bridge.archive.runtime.residency.stats()["reserved_bytes"], 0)
        self.assertEqual(bridge.archive.runtime.active_leases, 0)
        stats = getattr(getattr(bridge.archive, "restorer", None), "stats", dict)()
        if "restorer" in stats:  # the packed restorer: decoded canvases and resident bases are released
            self.assertEqual(stats["restorer"]["arena"]["used_bytes"], 0)
            self.assertEqual(stats["bases"]["pinned"], 0)

    def test_a_corrupt_object_fails_the_request_typed_and_changes_nothing(self):
        for prewarm in (True, False):
            with self.subTest(prewarm=prewarm):
                root, heads, blob = self.archived(store=f"read-{prewarm}")
                with ObjectStore(root) as store:
                    _corrupt(store, blob)
                    refs = store.list_refs()
                bridge = self.bridge(store=f"read-{prewarm}", prewarm=prewarm)  # starts: no audit at open
                self.assertEqual(bridge.archive.quarantined(), {})
                with self.assertRaisesRegex(IntegrityError, "hash or length mismatch") as caught:
                    drive(bridge, t.request(t.handle("s1", "resume"), prompt="A"))
                control = error_of(caught.exception)
                self.assertEqual((control.code, control.status), ("integrity", 500))
                self.assertEqual(bridge.archive.store.list_refs(), refs)  # published state unchanged
                self.assertBaseline(bridge)
                resumed, _, _, _ = drive(bridge, t.request(t.handle("s2", "resume"), prompt="B"))
                self.assertEqual((resumed["status"], resumed["generated_chunks"]), ("completed", 4))
                bridge.drain_all()
                self.assertBaseline(bridge)

    def test_a_damaged_head_is_quarantined_and_the_bridge_serves_the_others(self):
        for damage in ("corrupt", "missing"):
            with self.subTest(damage=damage):
                store_name = f"head-{damage}"
                root, heads, _ = self.archived(store=store_name)
                with ObjectStore(root) as store:
                    if damage == "corrupt":
                        _corrupt(store, heads["s1"])
                    else:
                        store.path_for(heads["s1"]).unlink()
                with self.assertRaises(IntegrityError):  # the former behavior: the server fails to start
                    self.bridge(store=store_name, on_open_error="raise")
                bridge = self.bridge(store=store_name)
                self.assertEqual(set(bridge.archive.quarantined()), {"s1"})
                self.assertEqual(set(bridge.startup["archive"]["quarantined"]), {"s1"})
                self.assertEqual(set(bridge.describe()["archive_stats"]["quarantined"]), {"s1"})
                listed = bridge.control("list", {})
                self.assertEqual([item["session_id"] for item in listed["sessions"]], ["s2"])
                self.assertEqual(bridge.archive.head("s1"), heads["s1"])  # kept: delete or replace it
                t.control_error(self, "integrity", "describe", {"session_id": "s1"}, bridge)
                with self.assertRaises(QuarantinedError):
                    drive(bridge, t.request(t.handle("s1", "resume"), prompt="A"))
                self.assertBaseline(bridge)
                resumed, _, _, _ = drive(bridge, t.request(t.handle("s2", "resume"), prompt="B"))
                self.assertEqual(resumed["status"], "completed")
                drive(bridge, t.request(t.handle("s3", pause_after_chunks=1), prompt="C"))
                self.assertEqual(bridge.control("delete", {"session_id": "s3"})["deleted"], True)
                self.assertGreaterEqual(bridge.control("gc", {})["gc"].get("damaged_objects", 0), 1)
                self.assertTrue(bridge.control("delete", {"session_id": "s1"})["deleted"])
                self.assertEqual(bridge.archive.quarantined(), {})
                self.assertNotIn(heads["s1"], _indexed(bridge.archive.store))
                created, _, _, _ = drive(bridge, t.request(t.handle("s1", pause_after_chunks=1), prompt="A"))
                self.assertEqual(created["status"], "paused")
                self.assertBaseline(bridge)


class RawBridgeFaultTests(BridgeFaultChecks, t.BridgeTestCase):
    representation = "raw"


class LineageBridgeFaultTests(BridgeFaultChecks, t.BridgeTestCase):
    representation = "svd_hevc"


if os.environ.get("REELCACHE_STEP7_GPU_TESTS") == "1":
    class CUDAStoreFaultTests(StoreFaultCase):
        """The serving read path on CUDA (NVDEC packed restore, plain and staged cold restores; inside GPU jobs)."""

        def setUp(self):
            if not torch.cuda.is_available():
                raise AssertionError("REELCACHE_STEP7_GPU_TESTS=1 requires a CUDA device")
            super().setUp()

        def restorer(self, runtime, staged):
            from reelcache.decoder_pool import DecoderPool
            from reelcache.restore_staging import StagingEngine
            uuid_ = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
            pool = DecoderPool(gpu_id=0, gpu_uuid=uuid_, workers=2, max_sessions_per_worker=16)
            engine = StagingEngine("cuda", threads=2, prepare_threads=2, pinned_bytes=1 << 24) if staged else None
            restorer = LineageRestorer(runtime, pool, arena_bytes=1 << 28, staging_engine=engine)
            self.addCleanup(pool.close)
            if engine is not None:
                self.addCleanup(engine.close)
            self.addCleanup(restorer.close)
            return restorer

        def cuda_restore(self, restorer, head):
            caches = {}

            def destination(node_id, key, tile):
                key = key if isinstance(key, MatrixKey) else key[0]
                if key not in caches:
                    caches[key] = torch.zeros((1, 2 * s6.TOKENS, 1, s6.CHANNELS), dtype=torch.float32,
                                              device="cuda")
                return rk.NativeView(caches[key], "BTHD", token_offset=tile.token_start)
            try:
                restorer.restore(head, destination)
            finally:
                torch.cuda.synchronize()
            return {key: value.cpu() for key, value in caches.items()}

        def test_nvdec_restores_of_a_corrupt_packet_fail_typed_without_leaks(self):
            for staged in (False, True):
                with self.subTest(staged=staged):
                    clean = self.open(**SERVING)
                    reference = self.cuda_restore(self.restorer(clean, staged), self.heads["b"])
                    clean.close()
                    self.corrupt(self.payloads["stream.hevc"])
                    try:
                        self.check_corrupt_restores(staged, reference)
                    finally:
                        self.corrupt(self.payloads["stream.hevc"])  # the same flip restores the byte

        def check_corrupt_restores(self, staged, reference):
            runtime = self.open(**SERVING)
            restorer = self.restorer(runtime, staged)
            self.assertSameCaches(self.cuda_restore(restorer, self.heads["b"]), reference)
            arena = restorer.stats()["restorer"]["arena"]["used_bytes"]  # canvases the arena keeps (free list)
            allocated = []
            for _ in range(3):
                with self.assertRaisesRegex(IntegrityError, "hash or length mismatch"):
                    self.cuda_restore(restorer, self.heads["a"])
                with self.assertRaisesRegex(IntegrityError, "hash or length mismatch"):
                    restorer.prewarm(restorer.history(self.heads["a"]))  # the bridge's session open
                stats = restorer.stats()
                self.assertLessEqual(stats["restorer"]["arena"]["used_bytes"], arena)
                self.assertEqual(stats["restorer"]["arena"]["retiring"], 0)
                self.assertEqual(stats["bases"]["pinned"], 0)
                self.assertEqual((runtime.active_leases, runtime.store.stats()["leases"]), (0, 0))
                allocated.append(torch.cuda.memory_allocated())
            self.assertEqual(allocated[2], allocated[1])  # no growth per failed request
            self.assertSameCaches(self.cuda_restore(restorer, self.heads["b"]), reference)
            runtime.close()


if __name__ == "__main__":
    unittest.main()
