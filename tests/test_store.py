import hashlib
import os
from pathlib import Path
import subprocess
import sys
import threading
from unittest.mock import patch

from reelcache.errors import CapacityError, IntegrityError, StateError
from reelcache.store import ObjectStore
from tests.core_support import CoreWorkspaceTest


class StoreTests(CoreWorkspaceTest):
    def open_store(self, **kwargs):
        store = ObjectStore(self.workspace / "store", **kwargs)
        self.addCleanup(store.close)
        return store

    def test_actual_bytes_content_addressing_and_distinct_basis_namespace(self):
        store = self.open_store()
        payload = b"coefficient-packet"
        blob = store.put(payload)
        self.assertEqual(blob, "blob:" + hashlib.sha256(payload).hexdigest())
        self.assertEqual(store.put(payload), blob)
        basis = store.put(payload, kind="basis")
        self.assertNotEqual(blob, basis)
        self.assertEqual(store.path_for(basis).read_bytes(), payload)
        self.assertEqual(store.stats()["object_count"], 2)
        self.assertEqual(store.stats()["object_bytes"], 2 * len(payload))
        self.assertEqual(store.stats()["basis_bytes"], len(payload))

    def test_canonical_manifests_and_graph_gc(self):
        store = self.open_store()
        blob = store.put(b"raw")
        left = store.put_manifest({"a": 1, "b": 2}, references=[blob])
        right = store.put_manifest({"b": 2, "a": 1}, references=[blob, blob])
        self.assertEqual(left, right)
        store.set_ref("branch", left, expected=None)
        store.put(b"unreachable")
        result = store.gc()
        self.assertEqual(result["objects_removed"], 1)
        self.assertEqual(store.get(blob), b"raw")
        self.assertEqual(store.get_manifest(left), {"a": 1, "b": 2})

    def test_targeted_staging_cleanup_preserves_preexisting_unrooted_objects(self):
        store = self.open_store()
        preexisting = store.put(b"preexisting")
        unrelated = store.put(b"unrelated-orphan")
        with store.transaction():
            fresh = store.put(b"staged")
            packet = store.put_manifest({"old": preexisting, "new": fresh},
                                        references=[preexisting, fresh])
            self.assertEqual(store.put(b"preexisting"), preexisting)
            created = store.created_objects
        self.assertEqual(set(created), {fresh, packet})
        store.gc(candidates=created)
        self.assertEqual(store.get(preexisting), b"preexisting")
        self.assertEqual(store.get(unrelated), b"unrelated-orphan")
        with self.assertRaises(IntegrityError):
            store.get(fresh)
        with self.assertRaises(StateError):
            _ = store.created_objects

    def test_targeted_gc_preserves_incoming_manifest_references(self):
        store = self.open_store()
        child = store.put(b"shared")
        parent = store.put_manifest({"child": child}, references=[child])
        self.assertEqual(store.gc(candidates=[child])["objects_removed"], 0)
        self.assertEqual(store.get_manifest(parent), {"child": child})
        self.assertEqual(store.get(child), b"shared")

    def test_detects_corruption_on_read_dedup_publication_and_gc(self):
        store = self.open_store(verify_reads=True)
        blob = store.put(b"correct")
        store.set_ref("branch", blob)
        store.path_for(blob).write_bytes(b"corrupt")
        for operation in (
            lambda: store.get(blob), lambda: store.put(b"correct"),
            lambda: store.set_ref("new", blob), store.gc,
        ):
            with self.subTest(operation=operation), self.assertRaises(IntegrityError):
                operation()
        self.assertIsNone(store.get_ref("new"))

    def test_default_reads_skip_hash_but_explicit_audit_detects_corruption(self):
        store = self.open_store()
        self.assertFalse(store.verify_reads)
        blob = store.put(b"correct")
        store.set_ref("branch", blob)
        store.path_for(blob).write_bytes(b"corrupt")
        self.assertEqual(store.get(blob), b"corrupt")
        store.check(blob)
        with self.assertRaises(IntegrityError):
            store.verify(blob)
        with self.assertRaises(IntegrityError):
            store._get(blob, verify=True)

    def test_missing_objects_and_invalid_ids_fail_explicitly(self):
        store = self.open_store()
        blob = store.put(b"payload")
        store.path_for(blob).unlink()
        with self.assertRaises(IntegrityError):
            store.get(blob)
        for object_id in ("../../secret", "blob:" + "a" * 64, "basis:bad"):
            with self.subTest(object_id=object_id), self.assertRaises(IntegrityError):
                store.get(object_id)

    def test_missing_index_is_not_silently_reinitialized_or_objects_deleted(self):
        store = self.open_store()
        blob = store.put(b"must-survive-index-failure")
        store.set_ref("root", blob)
        path = store.path_for(blob)
        store.close()
        (self.workspace / "store" / "index.sqlite3").unlink()
        with self.assertRaisesRegex(IntegrityError, "index is missing"):
            self.open_store()
        self.assertEqual(path.read_bytes(), b"must-survive-index-failure")

    def test_truncated_index_is_not_treated_as_a_new_store(self):
        store = self.open_store()
        blob = store.put(b"must-survive-truncation")
        store.set_ref("root", blob)
        path = store.path_for(blob)
        store.close()
        (self.workspace / "store" / "index.sqlite3").write_bytes(b"")
        with self.assertRaisesRegex(IntegrityError, "truncated"):
            self.open_store()
        self.assertEqual(path.read_bytes(), b"must-survive-truncation")

    def test_atomic_multi_reference_rollback_and_restart_recovery(self):
        store = self.open_store()
        old = store.put(b"old")
        store.set_ref("a", old)
        store.set_ref("b", old)
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with store.transaction():
                new = store.put(b"new")
                store.set_ref("a", new)
                store.set_ref("b", new)
                raise RuntimeError("interrupted")
        self.assertEqual(store.list_refs(), {"a": old, "b": old})
        orphan = store.path_for(new)
        self.assertTrue(orphan.exists())
        store.close()
        reopened = self.open_store()
        self.assertEqual(reopened.list_refs(), {"a": old, "b": old})
        self.assertFalse(orphan.exists())
        self.assertEqual(reopened.get(old), b"old")

    def test_nested_failure_cannot_be_ignored_into_partial_commit(self):
        store = self.open_store()
        original = store.put(b"original")
        store.set_ref("root", original)
        with self.assertRaises(StateError):
            with store.transaction():
                replacement = store.put(b"replacement")
                store.set_ref("root", replacement)
                try:
                    store.get("blob:" + "0" * 64)
                except IntegrityError:
                    pass
        self.assertEqual(store.get_ref("root"), original)

    def test_compare_and_swap_preserves_concurrent_root(self):
        store = self.open_store()
        first, second = store.put(b"first"), store.put(b"second")
        store.set_ref("root", first, expected=None)
        with self.assertRaises(StateError):
            store.set_ref("root", second, expected=None)
        self.assertEqual(store.get_ref("root"), first)

    def test_leases_pin_entire_graph_across_stores(self):
        writer = self.open_store()
        blob = writer.put(b"persistent coefficients")
        head = writer.put_manifest({"tensor": blob}, references=[blob])
        writer.set_ref("branch", head)
        reader = self.open_store()
        pin = reader.lease(head)
        writer.delete_ref("branch")
        self.assertEqual(writer.gc()["objects_removed"], 0)
        self.assertEqual(reader.get(blob), b"persistent coefficients")
        pin.release()
        self.assertEqual(writer.gc()["objects_removed"], 2)
        with self.assertRaises(IntegrityError):
            reader.get(blob)

    def test_capacity_charges_unique_full_basis_and_metadata(self):
        store = self.open_store(max_bytes=100)
        basis = store.put(b"x" * 80, kind="basis")
        self.assertEqual(store.put(b"x" * 80, kind="basis"), basis)
        self.assertEqual(store.stats()["object_bytes"], 80)
        with self.assertRaises(CapacityError):
            store.put(b"y" * 21)
        with self.assertRaises(CapacityError):
            store.put_manifest({"view_rank": 1}, references=[basis])
        self.assertEqual(store.stats()["object_count"], 1)

    def test_interrupted_atomic_file_write_never_publishes(self):
        store = self.open_store()
        with patch("reelcache.store.os.link", side_effect=OSError("injected disk failure")):
            with self.assertRaises(OSError):
                store.put(b"interrupted")
        self.assertEqual(store.stats()["object_count"], 0)
        self.assertEqual(list((store.root / "staging").iterdir()), [])

    def test_restart_removes_only_owned_partial_files(self):
        store = self.open_store()
        staging = store.root / "staging"
        owned = staging / ("pending-" + "a" * 32)
        other = staging / "user-file"
        owned.write_bytes(b"incomplete")
        other.write_bytes(b"preserve")
        store.close()
        self.open_store()
        self.assertFalse(owned.exists())
        self.assertEqual(other.read_bytes(), b"preserve")

    def test_dead_process_pins_are_recovered(self):
        root = self.workspace / "dead-owner"
        script = (
            "import os,sys\n"
            "from reelcache.store import ObjectStore\n"
            "s=ObjectStore(sys.argv[1])\n"
            "o=s.put(b'crash-retained')\n"
            "p=s.lease(o)\n"
            "os._exit(0)\n"
        )
        subprocess.run([sys.executable, "-c", script, str(root)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
        with ObjectStore(root) as store:
            self.assertEqual(store.stats()["leases"], 0)
            self.assertEqual(store.gc()["objects_removed"], 1)

    def test_process_death_mid_publication_recovers_old_root(self):
        root = self.workspace / "crashed-publication"
        with ObjectStore(root) as store:
            old = store.put(b"old-state")
            store.set_ref("a", old)
            store.set_ref("b", old)
        script = (
            "import os,sys\n"
            "from reelcache.store import ObjectStore\n"
            "s=ObjectStore(sys.argv[1])\n"
            "with s.transaction():\n"
            "    new=s.put(b'uncommitted-state')\n"
            "    s.set_ref('a',new)\n"
            "    os._exit(0)\n"
        )
        subprocess.run([sys.executable, "-c", script, str(root)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
        with ObjectStore(root) as store:
            self.assertEqual(store.list_refs(), {"a": old, "b": old})
            self.assertEqual(store.stats()["object_count"], 1)
            self.assertEqual(store.get(old), b"old-state")
            self.assertEqual(len(list((store.root / "objects" / "blob").iterdir())), 1)

    def test_readers_observe_atomic_root_pairs(self):
        writer = self.open_store()
        first, second = writer.put(b"first"), writer.put(b"second")
        with writer.transaction():
            writer.set_ref("a", first)
            writer.set_ref("b", first)
        reader = self.open_store()
        midway, continue_commit = threading.Event(), threading.Event()
        observed, failures = [], []

        def publish():
            try:
                with writer.transaction():
                    writer.set_ref("a", second)
                    midway.set()
                    if not continue_commit.wait(5):
                        raise RuntimeError("test synchronization timeout")
                    writer.set_ref("b", second)
            except Exception as error:
                failures.append(error)

        def observe():
            try:
                observed.append(reader.list_refs())
            except Exception as error:
                failures.append(error)

        publication = threading.Thread(target=publish)
        publication.start()
        self.assertTrue(midway.wait(5))
        observation = threading.Thread(target=observe)
        observation.start()
        continue_commit.set()
        publication.join(5)
        observation.join(5)
        self.assertFalse(publication.is_alive())
        self.assertFalse(observation.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(observed, [{"a": second, "b": second}])

    def test_closed_store_and_lease_are_explicit(self):
        store = self.open_store()
        lease = store.lease(store.put(b"x"))
        store.close()
        self.assertFalse(lease.active)
        lease.release()
        with self.assertRaises(StateError):
            store.get_ref("any")
