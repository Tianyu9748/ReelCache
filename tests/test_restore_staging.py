"""Cold-session staging (reelcache.restore_staging) and the restore-side fast paths it relies on (CPU)."""

from __future__ import annotations

import gc
import json
import math
import random
import threading
import time
import unittest
import weakref
from types import MappingProxyType
from unittest import mock

import numpy as np
import torch

from reelcache import _hevc, restore_staging
from reelcache.archive_restore import LineageRestorer, SoftwareLineageRestorer
from reelcache.codecs import _canon, _json_bytes, _validate_pack
from reelcache.errors import ConfigurationError, IntegrityError, QuarantinedError, StateError
from reelcache.manager import KVRuntime
from reelcache.packed_decode import prepare_packet
from reelcache.restore_staging import PinnedRing, StagingEngine, StagingTicket
from reelcache.types import EncodedTensor
from tests.step6_support import MODEL, RuntimeFactory, chunk, method, registry
from tests.test_packed_nvdec import SoftwarePool
from tests import step6_support as s6
from tests.test_step7_store_faults import SERVING, StoreFaultCase, _destination, _reachable


class StagingEngineTests(unittest.TestCase):
    def setUp(self):
        self.runtimes = RuntimeFactory(self)
        self.engine = StagingEngine("cpu", threads=3, prepare_threads=2, pinned_bytes=1 << 20)
        self.addCleanup(self.engine.close)

    def lineage(self, runtime, count=3):
        runtime.create_lineage("main")
        reg = registry()
        return [runtime.commit_node("main", f"chunk-{index:03d}", *chunk(index), method=method(), registry=reg)
                for index in range(count)]

    @staticmethod
    def history(runtime, node_ref):
        return SoftwareLineageRestorer(runtime).history(node_ref)

    def corrupt(self, runtime, object_id, *, truncate=False):
        path = runtime.store.path_for(object_id)
        data = bytearray(path.read_bytes())
        if truncate:
            del data[-1]
        else:
            data[len(data) // 2] ^= 0x10
        path.write_bytes(bytes(data))

    def test_staged_packs_equal_the_store_path_with_shared_version_chains(self):
        runtime = self.runtimes()
        commits = self.lineage(runtime)
        history = self.history(runtime, commits[-1].node_ref)
        self.assertEqual(len(history), 3)
        staged = self.engine.stage(runtime, history)
        staged.wait()
        tiles = 0
        for item, ref in zip(history, staged.refs):
            self.assertEqual(ref.node_id, item["node_id"])
            expected = prepare_packet(runtime.node_pack(item["packet"]))
            got = staged.prepared[item["packet"]]
            self.assertIs(ref.packets[0], got)
            self.assertEqual((got.width, got.height, got.pictures, got.stream_units, got.random_access),
                             (expected.width, expected.height, expected.pictures, expected.stream_units,
                              expected.random_access))
            for mine, theirs in zip(got.tiles, expected.tiles, strict=True):
                self.assertEqual((mine.key, mine.units, mine.rect, mine.rank, mine.pictures),
                                 (theirs.key, theirs.units, theirs.rect, theirs.rank, theirs.pictures))
                self.assertTrue(np.array_equal(mine.scale, theirs.scale) and np.array_equal(mine.offset, theirs.offset))
                self.assertTrue(torch.equal(mine.basis, theirs.basis))
                self.assertIsNone(mine.ready.event())
                tiles += 1
            packet = runtime.load_packet(item["packet"])
            self.assertEqual(staged.bindings[item["packet"]],
                             tuple((binding["basis_version"], binding["rank_view"]) for binding in packet["tile_bases"]))
        # Later chunks share or extend earlier versions: one staged tensor per chain tip serves several tiles.
        self.assertLess(staged.stats["tips"], tiles)
        for version_ref, chain, tensor, ticket in staged.versions:
            self.assertEqual(chain[-1], version_ref)
            self.assertTrue(torch.equal(tensor, runtime.basis_versions.assemble(version_ref)))
            self.assertTrue(ticket.done())
        staged.release()
        staged.release()
        self.assertEqual(staged.refs, [])
        stats = self.engine.stats()
        self.assertEqual((stats["histories"], stats["packs"]), (1, 3))
        self.assertEqual(stats["block_bytes"], stats["tip_bytes"] + 0 * stats["blocks"])

    def test_a_corrupted_basis_block_fails_its_ticket_and_is_quarantined(self):
        """Bases are verified before their copy (reconstruction never reads unverified rows): the tickets of the
        tiles on the corrupt block fail, the gate fails, and the block is quarantined."""
        runtime = self.runtimes()
        commits = self.lineage(runtime, count=2)
        history = self.history(runtime, commits[-1].node_ref)
        packet = runtime.load_packet(history[-1]["packet"])
        version = packet["tile_bases"][0]["basis_version"]
        block = runtime.basis_versions.blocks(version)[-1]
        self.corrupt(runtime, block)
        staged = self.engine.stage(runtime, history)
        failed = [tile for prepared in staged.prepared.values() for tile in prepared.tiles
                  if not _ticket_ok(tile.ready)]
        self.assertTrue(failed)
        with self.assertRaisesRegex(IntegrityError, "content address"):
            failed[0].ready.event()
        with self.assertRaisesRegex(IntegrityError, "content address"):
            staged.wait()
        self.assertIn(block, runtime.store.quarantined_objects)
        with self.assertRaisesRegex(IntegrityError, "content address"):
            staged.release()
        self.assertEqual(self.engine.stats()["verify_failures"], 1)
        # Later reads verify the block before any use and fail typed: staging (at planning) and the store path.
        with self.assertRaisesRegex(QuarantinedError, "quarantined"):
            self.engine.stage(runtime, history)
        with self.assertRaises(QuarantinedError):
            runtime.node_pack(history[-1]["packet"])
        with self.assertRaises(IntegrityError):  # the explicit audit too
            runtime.store.verify(block, recursive=False)
        self.assertEqual(self.engine.stats()["verify_failures"], 1)  # nothing reached the verifier again
        # Repaired, the next use verifies it and lifts the quarantine.
        self.corrupt(runtime, block)  # the same bit flipped back: the original bytes
        staged = self.engine.stage(runtime, history)
        self.assertNotIn(block, runtime.store.quarantined_objects)
        staged.wait()
        staged.release()
        self.corrupt(runtime, block)
        with self.assertRaises(IntegrityError):  # an audit quarantines too; repaired, an audit lifts
            runtime.store.verify(block, recursive=False)
        self.assertIn(block, runtime.store.quarantined_objects)
        self.corrupt(runtime, block)
        runtime.store.verify(block, recursive=False)
        self.assertNotIn(block, runtime.store.quarantined_objects)

    def test_synchronous_verification_fails_the_ticket_before_the_copy(self):
        engine = StagingEngine("cpu", threads=3, prepare_threads=2, pinned_bytes=1 << 20, verify_threads=0)
        self.addCleanup(engine.close)
        runtime = self.runtimes()
        commits = self.lineage(runtime, count=2)
        history = self.history(runtime, commits[-1].node_ref)
        packet = runtime.load_packet(history[-1]["packet"])
        version = packet["tile_bases"][0]["basis_version"]
        block = runtime.basis_versions.blocks(version)[-1]
        self.corrupt(runtime, block)
        staged = engine.stage(runtime, history)
        self.assertFalse(staged.stats["verify_async"])
        failed = [tile for prepared in staged.prepared.values() for tile in prepared.tiles
                  if not _ticket_ok(tile.ready)]
        self.assertTrue(failed)
        with self.assertRaisesRegex(IntegrityError, "content address"):
            failed[0].ready.event()
        with self.assertRaises(IntegrityError):
            staged.wait()
        with self.assertRaises(IntegrityError):
            staged.release()
        self.assertIn(block, runtime.store.quarantined_objects)
        self.assertEqual(engine.stats()["verify_objects"], 0)  # nothing ran on verifier threads
        # The store path detects the same corruption.
        with self.assertRaises(IntegrityError):
            runtime.node_pack(history[-1]["packet"])

    def test_large_payloads_are_read_as_parallel_ranges(self):
        runtime = self.runtimes()
        commits = self.lineage(runtime, count=2)
        history = self.history(runtime, commits[-1].node_ref)
        with mock.patch.object(restore_staging, "_PARALLEL_READ_BYTES", 2048), \
                mock.patch.object(restore_staging, "_READ_CHUNK_BYTES", 1000):
            staged = self.engine.stage(runtime, history)
            for item in history:
                expected = prepare_packet(runtime.node_pack(item["packet"]))
                self.assertEqual(staged.prepared[item["packet"]].stream_units, expected.stream_units)
            staged.release()
            packet = runtime.load_packet(history[-1]["packet"])
            self.assertGreater(packet["metadata"]["payloads"]["stream.hevc"]["bytes"], 2048)
            self.corrupt(runtime, packet["payloads"]["stream.hevc"], truncate=True)
            with self.assertRaisesRegex(IntegrityError, "descriptor declares"):
                self.engine.stage(runtime, history)

    def test_a_truncated_payload_fails_staging_and_leaves_the_engine_usable(self):
        runtime = self.runtimes()
        commits = self.lineage(runtime, count=2)
        last = self.history(runtime, commits[-1].node_ref)
        packet = runtime.load_packet(last[-1]["packet"])
        self.corrupt(runtime, packet["payloads"]["stream.hevc"], truncate=True)
        with self.assertRaisesRegex(IntegrityError, "descriptor declares"):
            self.engine.stage(runtime, last)
        first = self.history(runtime, commits[0].node_ref)
        staged = self.engine.stage(runtime, first)
        staged.release()
        self.assertEqual(self.engine.stats()["histories"], 1)

    def verifying(self, count=1):
        runtime = self.runtimes()
        commits = self.lineage(runtime, count=count)
        verifying = KVRuntime(runtime.store.root, MODEL, verify_reads=True)  # audits the graph at open
        self.addCleanup(verifying.close)
        return runtime, verifying, commits

    def test_verify_reads_hashes_payloads(self):
        runtime, verifying, commits = self.verifying()
        history = self.history(verifying, commits[0].node_ref)
        packet = verifying.load_packet(history[0]["packet"])
        affine = next(blob for name, blob in packet["payloads"].items() if name.startswith("affine."))
        self.corrupt(runtime, affine)
        sync = StagingEngine("cpu", threads=2, prepare_threads=2, pinned_bytes=1 << 20, verify_threads=0)
        self.addCleanup(sync.close)
        with self.assertRaisesRegex(IntegrityError, "hash or length mismatch"):  # before use
            sync.stage(verifying, history)
        self.assertIn(affine, verifying.store.quarantined_objects)
        verifying.store.lift_quarantine(affine)
        # Asynchronous (default): the pack is prepared from the unverified bytes; the gate fails.
        staged = self.engine.stage(verifying, history)
        self.assertTrue(staged.stats["verify_async"] and staged.stats["verify_payloads"])
        with self.assertRaisesRegex(IntegrityError, f"hash or length mismatch: {affine}"):
            staged.wait()
        with self.assertRaises(IntegrityError):
            staged.release()
        self.assertIn(affine, verifying.store.quarantined_objects)
        with self.assertRaisesRegex(QuarantinedError, affine):
            self.engine.stage(verifying, history)

    def test_async_staging_equals_synchronous_staging_and_verifies_every_object(self):
        runtime, verifying, commits = self.verifying(count=3)
        history = self.history(verifying, commits[-1].node_ref)
        sync = StagingEngine("cpu", threads=2, prepare_threads=2, pinned_bytes=1 << 20, verify_threads=0)
        self.addCleanup(sync.close)
        expected, got = sync.stage(verifying, history), self.engine.stage(verifying, history)
        expected.wait()
        got.wait()
        for ref, theirs in expected.prepared.items():
            mine = got.prepared[ref]
            self.assertEqual((mine.stream_units, mine.random_access), (theirs.stream_units, theirs.random_access))
            for a, b in zip(mine.tiles, theirs.tiles, strict=True):
                self.assertTrue(torch.equal(a.basis, b.basis) and np.array_equal(a.scale, b.scale)
                                and np.array_equal(a.offset, b.offset))
        self.assertEqual([item[:2] for item in got.versions], [item[:2] for item in expected.versions])
        for (_, _, tensor, _), (_, _, theirs, _) in zip(got.versions, expected.versions):
            self.assertTrue(torch.equal(tensor, theirs))
        payloads = {blob for item in history for blob in verifying.load_packet(item["packet"])["payloads"].values()}
        # The restore's verification record (restore.lease.verification): configured mode, payloads verified.
        for staged, mode in ((got, "async"), (expected, "sync")):
            report = self.engine.verification_report(verifying.store, staged) if mode == "async" else \
                sync.verification_report(verifying.store, staged)
            self.assertEqual((report["mode"], report["verified_payloads"]), (mode, len(payloads)))
            self.assertEqual(report["payload_wait_seconds"], staged.stats["payload_wait_seconds"])
        self.assertEqual(self.engine.verification_report(verifying.store),
                         {"mode": "async", "verified_payloads": 0, "payload_wait_seconds": 0.0})  # staged nothing
        self.assertEqual(self.engine.verification_report(runtime.store)["mode"], "off")  # store does not verify
        stats = self.engine.stats()
        self.assertEqual(stats["verify_failures"], 0)
        self.assertEqual((stats["verify_objects"], stats["verify_bytes"]), (len(payloads), stats["payload_bytes"]))
        self.assertGreater(stats["thread_hash_seconds"], 0.0)  # bases: on the staging threads, before their copy
        got.release()
        expected.release()

    def test_the_gate_waits_for_every_verification(self):
        runtime, verifying, commits = self.verifying(count=2)
        history = self.history(verifying, commits[-1].node_ref)
        opened, entered = threading.Event(), threading.Event()
        original = StagingEngine._check_object

        def held(engine, store, object_id, data):
            entered.set()
            self.assertTrue(opened.wait(30))
            return original(engine, store, object_id, data)

        with mock.patch.object(StagingEngine, "_check_object", held):
            staged = self.engine.stage(verifying, history)  # returns with payload verification pending
            self.assertTrue(entered.wait(30))
            self.assertTrue(all(_ticket_ok(tile.ready) for prepared in staged.prepared.values()
                                for tile in prepared.tiles))
            done = threading.Event()
            waiter = threading.Thread(target=lambda: (staged.wait(), done.set()))
            waiter.start()
            self.assertFalse(done.wait(0.3))  # the gate holds while a verification is pending
            opened.set()
            waiter.join(30)
        self.assertTrue(done.is_set())
        self.assertGreater(staged.stats["verify_gate_wait_seconds"], 0.0)
        self.assertEqual(self.engine.stats()["verify_gate_waits"], 1)
        staged.release()

    def staged_failure(self, runtime, history):
        """The IntegrityError of staging ``history``: raised by ``stage`` (preparing the unverified bytes failed) or
        by the gate; a staged history is released either way."""
        try:
            staged = self.engine.stage(runtime, history)
        except IntegrityError as exc:
            return exc
        with self.assertRaises(IntegrityError) as caught:
            staged.wait()
        with self.assertRaises(IntegrityError):
            staged.release()
        return caught.exception

    def test_large_objects_are_verified_once_staging_read_them(self):
        runtime, verifying, commits = self.verifying(count=2)
        history = self.history(verifying, commits[-1].node_ref)
        packet = verifying.load_packet(history[-1]["packet"])
        stream = packet["payloads"]["stream.hevc"]
        self.assertGreater(packet["metadata"]["payloads"]["stream.hevc"]["bytes"], 2048)
        submitted = []
        original = StagingEngine._submit_verifications

        def recording(engine, store, packs, staged):
            submitted.append(all(future.done() for pack in packs for future in pack["futures"]))
            return original(engine, store, packs, staged)

        with mock.patch.object(restore_staging, "_PARALLEL_READ_BYTES", 2048), \
                mock.patch.object(restore_staging, "_READ_CHUNK_BYTES", 1000), \
                mock.patch.object(StagingEngine, "_submit_verifications", recording):
            staged = self.engine.stage(verifying, history)
            staged.wait()
            staged.release()
            self.assertEqual(submitted, [True])  # after every read, never alongside them
            self.corrupt(runtime, stream)  # same length: only the hash can tell
            error = self.staged_failure(verifying, history)
        self.assertNotIsInstance(error, QuarantinedError)
        self.assertIn(f"hash or length mismatch: {stream}", str(error))
        self.assertIn(stream, verifying.store.quarantined_objects)

    def test_a_structural_failure_of_unverified_bytes_reports_the_hash_mismatch(self):
        runtime, verifying, commits = self.verifying(count=1)
        history = self.history(verifying, commits[0].node_ref)
        stream = verifying.load_packet(history[0]["packet"])["payloads"]["stream.hevc"]
        path = runtime.store.path_for(stream)
        data = bytearray(path.read_bytes())
        start = data.find(_hevc._VPS_START, data.find(_hevc._VPS_START) + 1)  # the second picture's VPS
        self.assertGreater(start, 0)
        data[start + len(_hevc._VPS_START) - 1] ^= 0x40  # one picture fewer than the descriptor declares
        path.write_bytes(bytes(data))
        with self.assertRaisesRegex(IntegrityError, f"hash or length mismatch: {stream}") as caught:
            self.engine.stage(verifying, history)
        self.assertIsInstance(caught.exception.__cause__, IntegrityError)  # the structural check that failed first
        self.assertIn(stream, verifying.store.quarantined_objects)

    def test_closed_engine_and_empty_history(self):
        runtime = self.runtimes()
        with self.assertRaises(ConfigurationError):
            self.engine.stage(runtime, [])
        engine = StagingEngine("cpu", threads=1, prepare_threads=1, pinned_bytes=4096)
        self.assertEqual(engine.warmup()["pinned_bytes"], 4096)
        engine.close()
        engine.close()
        with self.assertRaises(StateError):
            engine.stage(runtime, [{"packet": "x", "node_id": "n"}])
        for bad in (0, -1, 1.5):
            with self.assertRaises(ConfigurationError):
                StagingEngine("cpu", threads=bad)


class AsyncVerificationFaultTests(StoreFaultCase):
    """D-E fault injection through the production restorer (``LineageRestorer`` + StagingEngine, verified reads):
    a corrupt payload, basis block or manifest fails the restore with a typed IntegrityError, nothing is adopted,
    the object is quarantined (a retry fails unread), and the restorer keeps serving other histories."""

    def setUp(self):
        super().setUp()
        self.runtime = self.open(**SERVING)
        self.engine = StagingEngine("cpu", threads=2, prepare_threads=2, pinned_bytes=1 << 22)
        self.addCleanup(self.engine.close)
        self.pool = SoftwarePool()
        self.restorer = LineageRestorer(self.runtime, self.pool, arena_bytes=1 << 30, staging_engine=self.engine)

    def assertNothingAdopted(self, before):
        self.assertEqual((set(self.restorer.prepared.refs()), set(self.restorer.bases.refs())), before)
        self.assertEqual(self.restorer.stats()["bases"]["pinned"], 0)
        arena = self.restorer.restorer.arena  # every canvas is back (free for reuse), none leased or retiring
        report = arena.report()
        free = sum(arena._size(tuple(canvas.shape)) for items in arena._free.values() for canvas in items)
        self.assertEqual((report["retiring"], report["used_bytes"] - free), (0, 0))

    def check_fault(self, object_id, message, *, decodes):
        before = (set(self.restorer.prepared.refs()), set(self.restorer.bases.refs()))
        jobs = len(self.pool.jobs)
        with self.assertRaisesRegex(IntegrityError, message) as caught:
            self.restore(self.runtime, self.heads["a"], self.restorer)
        self.assertNotIsInstance(caught.exception, QuarantinedError)
        self.assertEqual(len(self.pool.jobs) > jobs, decodes)  # payloads and bases decode before they verify
        self.assertNothingAdopted(before)
        self.assertIn(object_id, self.runtime.store.quarantined_objects)
        jobs = len(self.pool.jobs)
        with self.assertRaisesRegex(QuarantinedError, object_id):  # a retry verifies it first: no decode
            self.restore(self.runtime, self.heads["a"], self.restorer)
        self.assertEqual(len(self.pool.jobs), jobs)
        self.assertNothingAdopted(before)
        # The server keeps serving: another history restores exactly through the same restorer, pool and engine.
        self.assertSameCaches(self.restore(self.runtime, self.heads["b"], self.restorer), self.reference["b"])
        # Repaired (the flipped byte flipped back), the next restore verifies the object, lifts it and is exact.
        self.corrupt(object_id)
        self.runtime.drop_manifest_caches()
        self.restorer.drop_caches()
        self.assertSameCaches(self.restore(self.runtime, self.heads["a"], self.restorer), self.reference["a"])
        self.assertNotIn(object_id, self.runtime.store.quarantined_objects)
        self.assertEqual(self.engine.stats()["verify_failures"], int(decodes))  # bulk bytes: the verifier

    def test_a_corrupt_affine_payload(self):
        # Decodes and reconstructs (wrong values) while it is verified: the gate fails the restore.
        self.corrupt(self.payloads["affine.001.f32"])
        self.check_fault(self.payloads["affine.001.f32"], "hash or length mismatch", decodes=True)

    def test_a_corrupt_bitstream_payload(self):
        # Whatever decoding the corrupt stream does, the restore fails with the hash mismatch.
        self.corrupt(self.payloads["stream.hevc"])
        self.check_fault(self.payloads["stream.hevc"], "hash or length mismatch", decodes=True)

    def test_a_corrupt_basis_block(self):
        with self.runtime.store.transaction():
            blocks = (_reachable(self.runtime.store, self.heads["a"])
                      - _reachable(self.runtime.store, self.heads["b"]))
        block = sorted(item for item in blocks if item.startswith("basis:"))[-1]
        self.corrupt(block)
        self.check_fault(block, "content address", decodes=True)

    def test_a_corrupt_manifest_fails_before_any_decode(self):
        # Corrupted after the open (whose graph check passed): metadata is verified before use, synchronously.
        self.runtime.drop_manifest_caches()
        self.corrupt(self.pack)
        self.check_fault(self.pack, "hash or length mismatch", decodes=False)


class VerificationRecordTests(StoreFaultCase):
    """``verified_payloads`` (freeze D convention): the payload objects a restore verified, on every path -- staged
    (the staging verifier), store path (packs not prepared, no staging engine) and CPU (``SoftwareLineageRestorer``),
    the latter two as the store's per-thread count of hash-checked payloads across the restore; 0 when nothing was
    read (a prepared restore); ``mode`` the configured one."""

    def payload_count(self, runtime, head):
        history = SoftwareLineageRestorer(runtime).history(head)
        return sum(len(runtime.load_packet(item["packet"])["payloads"]) for item in history)

    def test_staged_store_path_cpu_and_prepared_restores(self):
        from reelcache.restore_staging import store_verification_report

        runtime = self.open(**SERVING)
        store, expected = runtime.store, self.payload_count(runtime, self.heads["a"])
        self.assertGreater(expected, 0)
        engine = StagingEngine("cpu", threads=2, prepare_threads=2, pinned_bytes=1 << 22)
        self.addCleanup(engine.close)
        # staged (cold): the staging verifier's count
        staged = LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 30, staging_engine=engine)
        since = store.verified_payloads()
        self.assertSameCaches(self.restore(runtime, self.heads["a"], staged), self.reference["a"])
        record = staged.last_staging["verification"]
        self.assertEqual((record["mode"], record["verified_payloads"]), ("async", expected))
        self.assertEqual(store.verified_payloads(), since)  # the store hashed no payload itself
        # prepared now: nothing read, nothing verified
        since = store.verified_payloads()
        self.assertSameCaches(self.restore(runtime, self.heads["a"], staged), self.reference["a"])
        self.assertEqual(engine.verification_report(store, since=since),
                         {"mode": "async", "verified_payloads": 0, "payload_wait_seconds": 0.0})
        # store path (no staging engine): the store hash-checks every payload it loads, counted for this thread
        plain = LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 30)
        since = store.verified_payloads()
        self.assertSameCaches(self.restore(runtime, self.heads["b"], plain), self.reference["b"])
        expected_b = self.payload_count(runtime, self.heads["b"])
        self.assertEqual(engine.verification_report(store, since=since)["verified_payloads"], expected_b)
        self.assertEqual(store_verification_report(store, since),
                         {"mode": "sync", "verified_payloads": expected_b, "payload_wait_seconds": 0.0})
        # CPU (SoftwareLineageRestorer)
        since = store.verified_payloads()
        SoftwareLineageRestorer(runtime).restore(self.heads["a"], _destination(2 * s6.TOKENS)[1])
        self.assertEqual(store_verification_report(store, since)["verified_payloads"], expected)
        self.assertEqual(engine.verification_report(store), {"mode": "async", "verified_payloads": 0,
                                                             "payload_wait_seconds": 0.0})  # no since: 0
        runtime.close()
        # a store that does not verify reads: mode off, nothing counted
        plain_runtime = self.open()
        since = plain_runtime.store.verified_payloads()
        SoftwareLineageRestorer(plain_runtime).restore(self.heads["a"], _destination(2 * s6.TOKENS)[1])
        self.assertEqual(store_verification_report(plain_runtime.store, since),
                         {"mode": "off", "verified_payloads": 0, "payload_wait_seconds": 0.0})
        self.assertEqual(engine.verification_mode(plain_runtime.store), "off")

    def test_counts_belong_to_the_calling_thread(self):
        runtime = self.open(**SERVING)
        store = runtime.store
        blobs = sorted(set(self.payloads_of(runtime, self.heads["a"])))
        since = store.verified_payloads()
        store.get_many(blobs + blobs[:1])  # reader threads hash; the caller is credited, a repeated id once
        self.assertEqual(store.verified_payloads() - since, len(blobs))
        other = []
        thread = threading.Thread(target=lambda: other.append((store.get(blobs[0]), store.verified_payloads())))
        thread.start()
        thread.join(60)
        self.assertEqual(other[0][1], 1)  # the other thread's own count
        self.assertEqual(store.verified_payloads() - since, len(blobs))  # unchanged here
        window = store.read_ahead(blobs[:2], max_bytes=1 << 30)
        self.addCleanup(window.close)
        before = store.verified_payloads()
        self.assertIsNotNone(window.take(blobs[0]))  # read ahead on a pool thread, credited to the taker
        self.assertEqual(store.verified_payloads() - before, 1)
        manifests = [ref for ref in _reachable(store, self.heads["a"]) if ref.startswith("manifest:")]
        before = store.verified_payloads()
        store.get(manifests[0])
        self.assertEqual(store.verified_payloads(), before)  # manifests are metadata, not payloads

    @staticmethod
    def payloads_of(runtime, head):
        history = SoftwareLineageRestorer(runtime).history(head)
        return [blob for item in history for blob in runtime.load_packet(item["packet"])["payloads"].values()]


def _longcat_provider(store, *, overlap):
    """The LongCat session provider of Step 13's CPU harness (``make_provider``: serving integrity defaults), restoring
    through the production packed restorer instead of its software one: ``mb.lineage_restorer`` with its
    StagingEngine (asynchronous verification) on a CPU decoder pool. ``overlap``: the backend's opt-in layer-ahead
    restore (``set_overlap``; off by default)."""
    from reelcache_sglang_models.longcat_video import reelcache_backend as mb
    from tests.test_longcat_video_session_provider import make_provider

    provider = make_provider(store)
    restorer, engine = mb.lineage_restorer(provider.runtime, SoftwarePool(), "cpu", arena_bytes=1 << 30,
                                           staging_engine=True, staging_threads=2, staging_prepare_threads=2,
                                           staging_pinned_bytes=1 << 22)
    provider.backend.restorer = restorer  # the backend reads it at every restore
    provider.backend.set_overlap(overlap)
    return provider, restorer, engine


def _run_segments(provider, handle, *, outputs, step0=None, frames=None, prompt="a cat", seed=0, num_segments=1,
                  **request):
    """``tests.test_longcat_video_session_provider.run_request`` with the denoising stage's handling of an overlapped
    restore (``sglang_stages._denoise``): step 0 reads the cache layer by layer as it lands (``step0(cache)``, the
    DiT's ``kv_cache_ready.wait``), then ``ready.join()`` re-raises a restore failure -- before any later step, the
    VAE decode, any emitted frame or the segment's boundary commit (``end_segment``). ``outputs`` collects what the
    request emitted."""
    from reelcache_sglang_models.longcat_video import conditioning_cache as cc
    from reelcache_sglang_models.longcat_video.segment_resume import generator_state_hex, set_generator_state_hex
    from tests.test_longcat_video_session_provider import (FINGERPRINT, GEOMETRY, GRID, conditioning_cache,
                                                           frames_digest, next_frames)

    fields = {"request_id": None, "prompt": prompt, "seed": seed, "num_segments": num_segments, "num_frames": 93,
              "session_id": None, "start_segment_index": 0, "generator_state": None,
              "conditioning_cache_backend": None, "has_client_frames": frames is not None, "injected": [], **request}
    run = provider.begin_request(fields, handle)
    generator = torch.Generator().manual_seed(seed)
    if run.plan.generator_state is not None:
        set_generator_state_hex(generator, run.plan.generator_state)
    cond = frames if run.plan.frames is None else run.plan.frames
    run.bind_request(frames=cond, generator_state=generator_state_hex(generator), geometry=GEOMETRY)
    error = None
    try:
        for seg in range(run.plan.start_segment, run.plan.start_segment + run.plan.num_segments):
            epoch = cc.ConditioningCacheEpoch(segment_index=seg, token_grid=GRID, cond_latents_sha256=frames_digest(cond),
                                              model_fingerprint=FINGERPRINT, session_id=run.session.session_id,
                                              parent_segment_index=seg - 1 if seg else None)
            cache = provider.cache_backend.restore(epoch)
            if cache is None:
                cache = conditioning_cache(cond, seg, run.session.session_id)
                provider.cache_backend.on_cache_built(cache)
                source = cc.SOURCE_NATIVE_BUILD
            else:
                source = cc.SOURCE_RESTORED
                if cache.ready is not None:
                    (step0 or (lambda cache: None))(cache)
                    cache.ready.join()
                    cache.ready = None
            cond = next_frames(cond, prompt, generator)
            outputs.append(cond)
            record = {"epoch": epoch.as_dict(), "cache_source": source, "generator_state": "start-state"}
            del cache
            if run.end_segment(segment_index=seg, record=record, next_frames=cond,
                               generator_state=generator_state_hex(generator), prompt=prompt):
                break
    except BaseException as exc:  # noqa: BLE001 - the stage reports it, then re-raises
        error = exc
        raise
    finally:
        summary = run.finish(error=error)
    return summary


class LongCatGateTests(unittest.TestCase):
    """D-E through the LongCat session provider (coordinator 23:44): a corrupt payload of an archived epoch, detected
    by the asynchronous verification, fails the request typed; nothing emitted or committed, the session heads
    unchanged (a fork or rewind is rolled back), the restore's caches reset, the next requests work.

    Default (``overlap`` off, production): the backend returns the cache only after the restorer's integrity gate,
    so step 0 never reads unverified layers. Opt-in overlapped restore: step 0 reads layers 0..L-2 while the
    payloads are still being verified; the gate fails the last layer and ``ready.join()``. Its failure must surface
    as an IntegrityError (``is_damage``: rollback; ``integrity`` code): ``LayerReadiness.wait`` / ``join`` wrap it in a
    ConditioningCacheError today, so the overlapped test applies the change requested of Step 13 (``_typed``)."""

    def setUp(self):
        import shutil
        import tempfile
        from pathlib import Path

        self.root = Path(tempfile.mkdtemp(prefix="d-e-longcat-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def provider(self, name, *, overlap):
        provider, restorer, engine = _longcat_provider(self.root / name, overlap=overlap)
        self.addCleanup(engine.close)
        self.addCleanup(restorer.close)
        self.addCleanup(lambda: provider._run is None and provider.close())
        return provider, restorer, engine

    @staticmethod
    def affine_blob(provider, epoch):
        runtime = provider.runtime
        node = runtime.load_node(provider.backend.node_ref(epoch))
        records = list(node.matrix_records) or [pair for item in runtime.load_epoch(node.epoch_ref).ranges
                                                for pair in item.records]
        packet = sorted({runtime.load_record(ref).packet_ref for _, ref in records})[0]
        payloads = runtime.load_packet(packet)["payloads"]
        return next(blob for name, blob in sorted(payloads.items()) if name.startswith("affine."))

    def check(self, *, overlap, action):
        from reelcache.errors import IntegrityError
        from reelcache_sglang_models.longcat_video import reelcache_backend as mb
        from tests.test_longcat_video_integrity import code_of, segment_epochs
        from tests.test_longcat_video_session_provider import LAYERS, initial_frames

        provider, restorer, engine = self.provider(f"{action}-{overlap}", overlap=overlap)
        first = []
        _run_segments(provider, {"session_id": "a", "action": "generate", "durable": True}, outputs=first,
                      frames=initial_frames(2), num_segments=2)
        epochs = segment_epochs(provider, "a")
        affine = self.affine_blob(provider, epochs[1])
        restorer.drop_caches()  # a cold restore: the staging engine reads, the verifier hashes
        provider.runtime.drop_manifest_caches()
        path = provider.runtime.store.path_for(affine)
        original = path.read_bytes()
        damaged = bytearray(original)
        damaged[len(damaged) // 2] ^= 0x40  # same length: decodes and reconstructs; only the hash tells
        path.write_bytes(bytes(damaged))
        before = provider.control("describe", {"session_id": "a", "history": 0})
        lineages = provider.runtime.list_lineages()
        adopted = (set(restorer.prepared.refs()), set(restorer.bases.refs()))
        consumed, started = [], threading.Event()

        def step0(cache):  # the DiT's first step: layer by layer as the restore marks them
            for layer in range(len(cache.keys_pre_rope)):
                cache.ready.wait(layer)
                consumed.append(layer)
                started.set()  # step 0 runs on restored layers: let the payload verification proceed

        original_check = StagingEngine._check_object

        def after_step0(engine_, store, object_id, data):
            if overlap:
                self.assertTrue(started.wait(60))
            return original_check(engine_, store, object_id, data)

        allocated, real = [], provider.backend._allocate

        def tracking(epoch, device):
            keys, values = real(epoch, device)
            allocated.extend(weakref.ref(tensor) for tensor in keys + values)
            return keys, values

        handle = ({"session_id": "b", "action": "fork", "from_session": "a", "checkpoint": 1} if action == "fork"
                  else {"session_id": "a", "action": "rewind", "checkpoint": 1})
        outputs = []
        with mock.patch.object(StagingEngine, "_check_object", after_step0), \
                mock.patch.object(provider.backend, "_allocate", side_effect=tracking), \
                self.assertRaises(IntegrityError) as caught:
            _run_segments(provider, handle, outputs=outputs, step0=step0)
        error, summary = caught.exception, list(provider._requests.values())[-1]
        del caught
        if overlap:  # step 0 read every restored layer but the last before the gate failed the restore
            self.assertEqual(consumed, list(range(LAYERS - 1)))
        else:
            self.assertEqual(consumed, [])
        # typed; nothing emitted or committed; every session head as before (the fork / rewind undone)
        self.assertEqual(code_of(error), "integrity")
        self.assertTrue(mb.is_damage(error))
        self.assertIn(f"hash or length mismatch: {affine}", repr(error) + repr(error.__cause__))
        self.assertEqual((outputs, summary["error_code"], summary["segments"]), ([], "integrity", []))
        self.assertEqual(summary["rolled_back"]["action"], action)
        self.assertEqual(provider.runtime.list_lineages(), lineages)
        after = provider.control("describe", {"session_id": "a", "history": 0})
        self.assertEqual((after["head"], after["generated_segments"], after["status"]),
                         (before["head"], before["generated_segments"], "idle"))
        self.assertNotIn("b", [item["session_id"] for item in provider.control("list", {})["sessions"]])
        self.assertIsNone(provider._run)
        # caches reset as in the decode-failure path: nothing adopted, the restore's buffers released; the epoch is
        # quarantined by the backend and the object by the store
        self.assertEqual((set(restorer.prepared.refs()), set(restorer.bases.refs())), adopted)
        self.assertEqual(restorer.stats()["bases"]["pinned"], 0)
        del error
        gc.collect()
        self.assertEqual(len(allocated), 2 * LAYERS)
        self.assertTrue(all(ref() is None for ref in allocated))
        self.assertEqual(provider.backend.archive_state(epochs[1], "svd_hevc"), "quarantined")
        self.assertIn(affine, provider.runtime.store.quarantined_objects)
        self.assertEqual(engine.stats()["verify_failures"], 1)
        # the next requests work: the healthy epoch restores (cold, verified) and generation continues
        path.write_bytes(original)
        forked = []
        summary = _run_segments(provider, {"session_id": "d", "action": "fork", "from_session": "a", "checkpoint": 0},
                                outputs=forked, step0=step0)
        self.assertEqual(summary["segments"][0]["provider_cache"], "restore")
        self.assertTrue((forked[0] == first[0]).all())
        resumed = []
        summary = _run_segments(provider, {"session_id": "a", "action": "resume"}, outputs=resumed)
        self.assertEqual((summary["segments"][0]["provider_cache"], len(resumed)), ("build", 1))

    def test_default_restore_fork(self):
        self.check(overlap=False, action="fork")

    def test_default_restore_rewind(self):
        self.check(overlap=False, action="rewind")

    def overlapped(self, action):
        from reelcache_sglang_models.longcat_video import conditioning_cache as cc

        def typed(readiness):  # the change requested of Step 13 (LayerReadiness.wait / join)
            message = f"overlapped conditioning-cache restore failed: {readiness._error!r}"
            if isinstance(readiness._error, IntegrityError):
                return IntegrityError(message)
            return cc.ConditioningCacheError(message)

        real_wait, real_join = cc.LayerReadiness.wait, cc.LayerReadiness.join

        def wait(readiness, layer, stream=None):
            try:
                return real_wait(readiness, layer, stream)
            except cc.ConditioningCacheError:
                if readiness._error is None:
                    raise
                raise typed(readiness) from readiness._error

        def join(readiness):
            try:
                return real_join(readiness)
            except cc.ConditioningCacheError:
                if readiness._error is None:
                    raise
                raise typed(readiness) from readiness._error

        with mock.patch.object(cc.LayerReadiness, "wait", wait), mock.patch.object(cc.LayerReadiness, "join", join):
            self.check(overlap=True, action=action)

    def test_a_payload_found_corrupt_after_step_0_started_fork(self):
        self.overlapped("fork")

    def test_a_payload_found_corrupt_after_step_0_started_rewind(self):
        self.overlapped("rewind")


def _ticket_ok(ticket):
    try:
        ticket.event(5)
        return True
    except IntegrityError:
        return False


class PinnedRingTests(unittest.TestCase):
    def test_fifo_wrap_waits_for_the_released_copy_and_oversize_requests(self):
        ring = PinnedRing(1024)
        first = ring.reserve(600)
        second = ring.reserve(200)
        self.assertEqual((first.start, second.start), (0, 768))
        self.assertEqual(len(first.view), 600)
        first.view[:3] = b"abc"
        self.assertEqual(bytes(ring.buffer[:3].numpy()), b"abc")
        threading.Timer(0.1, PinnedRing.release, (first, None)).start()
        started = time.perf_counter()
        third = ring.reserve(500)  # wraps onto the first segment: waits until it is released
        self.assertGreaterEqual(time.perf_counter() - started, 0.05)
        self.assertEqual(third.start, 0)
        self.assertEqual(ring.stats["waits"], 1)
        PinnedRing.release(second, None)
        PinnedRing.release(third, None)
        big = ring.reserve(4096)
        self.assertEqual((big.start, len(big.view), ring.stats["oversize"]), (-1, 4096, 1))
        with self.assertRaises(ConfigurationError):
            ring.reserve(0)
        with self.assertRaises(ConfigurationError):
            PinnedRing(16)

    def test_ticket(self):
        ticket = StagingTicket("t")
        with self.assertRaises(StateError):
            ticket.event(0.01)
        ticket._set("event")
        self.assertEqual(ticket.event(), "event")
        failing = StagingTicket("f")
        failing._fail(IntegrityError("bad rows"))
        with self.assertRaisesRegex(IntegrityError, "bad rows"):
            failing.event()
        other = StagingTicket("o")
        other._fail(OSError("disk"))
        with self.assertRaises(StateError):
            other.event()


class RestoreFastPathTests(unittest.TestCase):
    def test_canon_is_exactly_json_bytes(self):
        values = [
            {"b": [1, 2.5, -0.0, True, None, "é"], "a": {"z": (1, 2), "y": 1e300}},
            [1, 1.0, "1", False, 0],
            {"nested": [{"k": [[]]}, {}], "x": float(2**60)},
            MappingProxyType({"proxy": [MappingProxyType({"inner": 1})]}),
            (),
            "text",
        ]
        rng = random.Random(3)
        for _ in range(200):
            values.append({str(rng.randrange(50)): [rng.random(), rng.randrange(-5, 5), str(rng.random())]
                           for _ in range(rng.randrange(1, 6))})
        for value in values:
            self.assertEqual(_canon(value), _json_bytes(value))
        for invalid in ({1: "int key"}, [math.nan], {"a": math.inf}, {"a": object()}):
            with self.assertRaises(Exception) as caught:
                _json_bytes(invalid)
            with self.assertRaises(type(caught.exception)):
                _canon(invalid)

    def test_vps_scan_matches_bytes_find(self):
        rng = random.Random(5)
        pattern = _hevc._VPS_START
        for _ in range(300):
            data = bytearray(rng.randbytes(rng.randrange(0, 400)))
            for _ in range(rng.randrange(0, 4)):
                position = rng.randrange(0, len(data) + 1)
                data[position:position] = pattern
            data = bytes(data)
            start = rng.randrange(0, len(data) + 1)
            end = rng.randrange(start, len(data) + 1)
            self.assertEqual(_hevc._find_vps(data, start, end), data.find(pattern, start, end))
        data = b"\x00" * 7 + pattern
        self.assertEqual(_hevc._find_vps(data, 0, len(data) - 1), -1)
        self.assertEqual(_hevc._find_vps(data, 0, len(data)), 7)

    def test_external_bases_validation(self):
        from tests.step6_support import KEYS

        factory = RuntimeFactory(self)
        runtime = factory()
        runtime.create_lineage("main")
        commit = runtime.commit_node("main", "chunk-000", *chunk(0), method=method(), registry=registry())
        history = SoftwareLineageRestorer(runtime).history(commit.node_ref)
        packet = runtime.load_packet(history[0]["packet"])
        payloads = {name: runtime.store.get(blob) for name, blob in packet["payloads"].items()}
        external = EncodedTensor(packet["metadata"], payloads)
        full = runtime.node_pack(history[0]["packet"])
        bases = [(prepare_packet(full).tiles[index].basis, None) for index in range(len(KEYS))]
        prepared = prepare_packet(external, bases=bases)
        self.assertTrue(all(torch.equal(a.basis, b.basis) for a, b in zip(prepared.tiles, prepare_packet(full).tiles)))
        with self.assertRaises(IntegrityError):  # embedded bases are not external
            _validate_pack(full, verify=False, external_bases=True)
        with self.assertRaises(IntegrityError):  # missing bases are not embedded
            _validate_pack(external, verify=False)
        with self.assertRaises(ConfigurationError):
            prepare_packet(external, bases=bases[:-1])
        wrong = [(basis[:-1], None) for basis, _ in bases]
        with self.assertRaises(IntegrityError):
            prepare_packet(external, bases=wrong)
        with self.assertRaises(IntegrityError):
            prepare_packet(external, bases=[(basis.float(), None) for basis, _ in bases])
        with self.assertRaises(ConfigurationError):
            _validate_pack(external, verify=False, external_bases=1)
        # Validation ahead of the payload reads (same-length placeholders) gives the same prepared pack.
        placeholders = {name: bytes(len(raw)) for name, raw in external.payloads.items()}
        shadow = object.__new__(EncodedTensor)
        object.__setattr__(shadow, "metadata", external.metadata)
        object.__setattr__(shadow, "payloads", placeholders)
        object.__setattr__(shadow, "basis", None)
        object.__setattr__(shadow, "metrics", None)
        ahead = _validate_pack(shadow, verify=False, external_bases=True)
        early = prepare_packet(external, bases=bases, validated=ahead)
        self.assertEqual([tile.units for tile in early.tiles], [tile.units for tile in prepared.tiles])
        self.assertIs(early.tiles[0].units[0][0], external.payloads["stream.hevc"])
        shorter = dict(external.payloads)
        name = next(name for name in shorter if name.startswith("affine."))
        shorter[name] = shorter[name][:-4]
        for kwargs in ({"validated": ahead, "verify": True}, {"validated": dict(ahead, metadata={})}):
            with self.assertRaisesRegex(IntegrityError, "validated ahead"):
                prepare_packet(external, bases=bases, **kwargs)
        with self.assertRaisesRegex(IntegrityError, "validated ahead"):
            prepare_packet(EncodedTensor(external.metadata, shorter), bases=bases, validated=ahead)
        with self.assertRaisesRegex(IntegrityError, "validated ahead"):
            prepare_packet(external, validated=ahead)  # validated for external bases, used without
        metadata = json.loads(json.dumps(packet["metadata"]))
        metadata["payloads"][metadata["tiles"][0]["basis"]["payload"]]["sha256"] = "Z" * 64
        with self.assertRaisesRegex(IntegrityError, "SHA256"):
            _validate_pack(EncodedTensor(metadata, payloads), verify=False, external_bases=True)


if __name__ == "__main__":
    unittest.main()
