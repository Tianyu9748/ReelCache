"""(c) SGLang freeze C: verified RAW history reads hash several objects at a time (history read-ahead).

A SHA-256 cannot be split, and every content address is the SHA-256 of a whole object, so a verified read of one
object is serial. LongCat's RAW restore (Step 13 RAWT) asks ``acquire_history`` for one layer's K, then V, layer
after layer: one object per call, so every byte was hashed on the calling thread (8.3-11.9 s verified vs 3.0-4.8 s
unverified). Now, once two calls of one head and token range ask for consecutive model keys, the runtime reads the
following keys' payloads ahead on the store's read-ahead threads (``store.ReadAhead``: the checks of ``get``, a
bound on bytes held), where their hashes run in parallel. A read-ahead object whose read failed is read again where
it is used, which raises exactly as before; unchecked bytes are never used.
"""

from __future__ import annotations

import threading
import time
from unittest import mock

import torch

from reelcache.errors import ConfigurationError, IntegrityError, StateError
from reelcache.integrations.sglang.archive import RawSessionArchive
from reelcache.manager import KVRuntime
from reelcache.store import ObjectStore, ReadAhead
import tests.test_sglang_bridge as t
from tests.core_support import CoreWorkspaceTest, chunk_inputs, model as core_model

LAYERS = 6
MODEL = core_model(layers=LAYERS, channels=16)
FRAMES, HEIGHT, WIDTH = 2, 4, 4
TOKENS = FRAMES * HEIGHT * WIDTH
SERVING = {"verify_reads": True, "verify_references": False, "audit_on_open": False}


def _corrupt(root, object_id):
    with ObjectStore(root) as store:
        path = store.path_for(object_id)
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 0x40
        path.write_bytes(bytes(data))


class ReadAheadCase(CoreWorkspaceTest):
    """A RAW branch of two chunks; every key's history is two payload objects."""

    def setUp(self):
        super().setUp()
        self.root = self.workspace / "store"
        runtime = KVRuntime(self.root, MODEL)
        try:
            runtime.create_branch("main")
            parts = {}
            for index in range(2):
                matrices, contexts = chunk_inputs(MODEL, start=index * TOKENS, frames=FRAMES, height=HEIGHT,
                                                  width=WIDTH, offset=1000 * index)
                runtime.seal_chunk("main", f"c{index}", matrices, contexts)
                for key, value in matrices.items():
                    parts.setdefault(key, []).append(value.clone())
            self.expected = {key: torch.cat(values) for key, values in parts.items()}
            self.keys = runtime._keys
            with runtime._lock, runtime.store.transaction():
                _, chunks = runtime._load_head(runtime._ready("main"))
            self.payloads = {key: [object_id for chunk in chunks
                                   for object_id in chunk["records"][key][3]["payloads"].values()]
                             for key in self.keys}
            self.payload_bytes = runtime.store.object_size(self.payloads[self.keys[0]][0])
        finally:
            runtime.close()

    def open(self, **flags):
        runtime = KVRuntime(self.root, MODEL, **flags)
        self.addCleanup(lambda: runtime._closed or runtime.close())
        return runtime

    def restore(self, runtime, keys=None, *, errors=None, **kwargs):
        results = {}
        for key in self.keys if keys is None else keys:
            try:
                with runtime.acquire_history("main", key, **kwargs) as lease:
                    results[key] = lease.tensor.clone()
            except IntegrityError as exc:
                if errors is None:
                    raise
                errors[key] = exc
        return results


class HistoryReadAheadTests(ReadAheadCase):
    def test_sequential_restores_read_ahead_and_restore_bitwise(self):
        runtime = self.open(**SERVING)
        restored = self.restore(runtime)
        for key in self.keys:
            self.assertTrue(torch.equal(restored[key], self.expected[key]), key)
        stats = runtime.read_ahead_stats()
        # The second call opens the window over every following key's payloads, each used exactly once.
        self.assertEqual(stats["windows"], 1)
        self.assertEqual((stats["taken"], stats["failed"], stats["plan_failures"]), (2 * (len(self.keys) - 2), 0, 0))
        self.assertEqual(stats["taken_bytes"], 2 * (len(self.keys) - 2) * self.payload_bytes)
        self.assertEqual(stats["open"]["held_bytes"], 0)
        disabled = self.open(**SERVING, history_read_ahead_bytes=0)
        restored = self.restore(disabled)
        self.assertTrue(all(torch.equal(restored[key], self.expected[key]) for key in self.keys))
        self.assertEqual(disabled.read_ahead_stats()["windows"], 0)
        unverified = self.restore(self.open())
        self.assertTrue(all(torch.equal(unverified[key], self.expected[key]) for key in self.keys))

    def test_the_window_is_bounded_and_reads_run_in_parallel(self):
        original, active, peak, lock = ObjectStore._read_checked, [0], [0], threading.Lock()

        def slow(store, *args):
            if not threading.current_thread().name.startswith("reelcache-store-read-ahead"):
                return original(store, *args)
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            try:
                time.sleep(0.05)
                return original(store, *args)
            finally:
                with lock:
                    active[0] -= 1
        runtime = self.open(**SERVING, history_read_ahead_bytes=4 * self.payload_bytes)
        with mock.patch.object(ObjectStore, "_read_checked", slow):
            restored = self.restore(runtime)
        self.assertTrue(all(torch.equal(restored[key], self.expected[key]) for key in self.keys))
        window = runtime.read_ahead_stats()["open"]
        self.assertLessEqual(window["peak_held_bytes"], 4 * self.payload_bytes)
        self.assertEqual(window["peak_held_bytes"], 4 * self.payload_bytes)
        self.assertGreaterEqual(peak[0], 2)  # verified reads of different objects ran concurrently
        self.assertLessEqual(peak[0], ObjectStore.READ_AHEAD_WORKERS)
        # A bound below one object still reads one object at a time.
        small = self.open(**SERVING, history_read_ahead_bytes=1)
        self.restore(small)
        self.assertEqual(small.read_ahead_stats()["open"]["peak_held_bytes"], self.payload_bytes)

    def test_a_corrupt_payload_fails_where_it_is_used_and_nowhere_else(self):
        target = self.keys[4]
        _corrupt(self.root, self.payloads[target][0])
        for read_ahead in (None, 0):
            runtime = self.open(**SERVING, history_read_ahead_bytes=read_ahead)
            errors = {}
            restored = self.restore(runtime, errors=errors)
            self.assertEqual(list(errors), [target])
            self.assertRegex(str(errors[target]), "hash or length mismatch")
            self.assertTrue(all(torch.equal(restored[key], self.expected[key]) for key in restored))
            if read_ahead is None:
                self.assertEqual(runtime.read_ahead_stats()["failed"], 1)  # then read (and failed) where used
        # Unverified reads: read-ahead returns exactly what the unverified read returns (the CR-17-1 failure mode
        # is unchanged, neither hidden nor introduced).
        plain = self.restore(self.open())
        plain_without = self.restore(self.open(history_read_ahead_bytes=0))
        self.assertFalse(torch.equal(plain[target], self.expected[target]))
        self.assertTrue(all(torch.equal(plain[key], plain_without[key]) for key in self.keys))

    def test_a_failed_plan_never_fails_or_invalidates_the_call(self):
        runtime = self.open(**SERVING)
        original = runtime._load_head

        def damaged(head_id, *, matrix_key=None, metadata_only=False):
            if matrix_key is None and not metadata_only:
                with runtime.store.transaction():
                    raise IntegrityError("damaged manifest of a later key")
            return original(head_id, matrix_key=matrix_key, metadata_only=metadata_only)
        with mock.patch.object(runtime, "_load_head", damaged):
            restored = self.restore(runtime, self.keys[:4])
        self.assertTrue(all(torch.equal(restored[key], self.expected[key]) for key in restored))
        stats = runtime.read_ahead_stats()
        self.assertEqual((stats["windows"], stats["plan_failures"]), (0, 3))
        # A plan inside a caller's store transaction is not attempted (its failure would invalidate the caller's).
        with runtime.store.transaction():
            self.restore(runtime, self.keys[:3])
        self.assertEqual(runtime.read_ahead_stats()["windows"], 0)
        self.restore(runtime, self.keys[:3])
        self.assertEqual(runtime.read_ahead_stats()["windows"], 1)

    def test_only_consecutive_keys_of_one_range_read_ahead(self):
        runtime = self.open(**SERVING)
        self.restore(runtime, self.keys[::-1])
        self.restore(runtime, self.keys[::2])
        self.assertEqual(runtime.read_ahead_stats()["windows"], 0)
        restored = self.restore(runtime, self.keys[:4], token_start=TOKENS, token_end=2 * TOKENS)
        self.assertTrue(all(torch.equal(restored[key], self.expected[key][TOKENS:]) for key in restored))
        stats = runtime.read_ahead_stats()
        self.assertEqual((stats["windows"], stats["taken"]), (1, 2))  # one payload per key in this range
        # A partial range is another sequence; it does not take the active window's bound.
        partial = self.restore(runtime, self.keys[:3], token_start=0, token_end=TOKENS // 2)
        self.assertTrue(all(torch.equal(partial[key], self.expected[key][:TOKENS // 2]) for key in partial))
        self.assertEqual(runtime.read_ahead_stats()["windows"], 1)
        with self.assertRaises(ConfigurationError):
            KVRuntime(self.root, MODEL, history_read_ahead_bytes=-1)

    def test_one_window_per_runtime_and_its_end(self):
        runtime = self.open(**SERVING)
        self.restore(runtime, self.keys[:3])
        owner = runtime.read_ahead_stats()["open"]
        self.assertGreater(owner["held_bytes"], 0)
        # Another range's sequence does not take the bound while the owner is active ...
        self.restore(runtime, self.keys[:3], token_end=TOKENS)
        self.assertEqual(runtime.read_ahead_stats()["windows"], 1)
        # ... and does once it is idle; the idle window's unused bytes are dropped.
        with mock.patch.object(KVRuntime, "HISTORY_READ_AHEAD_IDLE_SECONDS", 0.0):
            self.restore(runtime, self.keys[3:5], token_end=TOKENS)
        stats = runtime.read_ahead_stats()
        self.assertEqual(stats["windows"], 2)
        self.assertGreater(stats["dropped"], 0)
        # Garbage collection and close drop the open window.
        runtime.collect_garbage()
        self.assertIsNone(runtime.read_ahead_stats()["open"])
        self.restore(runtime, self.keys[:3])
        self.assertIsNotNone(runtime.read_ahead_stats()["open"])
        runtime.close()
        self.assertIsNone(runtime._read_ahead_window)


class ReadAheadWindowTests(ReadAheadCase):
    def test_take_returns_the_checked_bytes_once_in_any_order(self):
        runtime = self.open(**SERVING)
        store = runtime.store
        ids = [object_id for key in self.keys[:3] for object_id in self.payloads[key]]
        window = store.read_ahead(ids + ids[:2] + ["blob:" + "0" * 64], max_bytes=2 * self.payload_bytes)
        self.assertEqual(window.stats()["scheduled"], len(ids))  # repeats once; unindexed ids are skipped
        self.assertEqual(window.stats()["held"], 2)
        self.assertEqual(window.take(ids[1]), store.get(ids[1]))
        self.assertIsNone(window.take(ids[1]))  # once
        self.assertEqual(window.take(ids[0]), store.get(ids[0]))
        self.assertIsNone(window.take(ids[-1]))  # queued, not started: the reader reads it itself
        self.assertIsNone(window.take("blob:" + "0" * 64))
        stats = window.stats()
        self.assertEqual((stats["taken"], stats["missed"]), (2, 1))
        self.assertLessEqual(stats["peak_held_bytes"], 2 * self.payload_bytes)
        window.close()
        self.assertEqual((window.stats()["queued"], window.stats()["held"], window.held_bytes), (0, 0, 0))
        self.assertEqual(window.schedule(ids), 0)
        with self.assertRaises(ConfigurationError):
            ReadAhead(store, 0)

    def test_a_failed_read_is_none_and_the_readers_own_read_raises(self):
        _corrupt(self.root, self.payloads[self.keys[2]][0])
        runtime = self.open(**SERVING)
        window = runtime.store.read_ahead(self.payloads[self.keys[2]], max_bytes=1 << 20)
        self.assertIsNone(window.take(self.payloads[self.keys[2]][0]))
        self.assertEqual(window.stats()["failed"], 1)
        with self.assertRaisesRegex(IntegrityError, "hash or length mismatch"):
            runtime.store.get(self.payloads[self.keys[2]][0])
        self.assertEqual(window.take(self.payloads[self.keys[2]][1]),
                         runtime.store.get(self.payloads[self.keys[2]][1]))
        # Unverified stores read ahead without checks, as their get does.
        plain = ObjectStore(self.workspace / "plain")
        self.addCleanup(plain.close)
        blob = plain.put(b"x" * 64)
        self.assertEqual(plain.read_ahead([blob], max_bytes=64).take(blob), b"x" * 64)

    def test_a_closed_store_ends_read_ahead(self):
        store = ObjectStore(self.workspace / "closing")
        blobs = [store.put(bytes([index]) * 32) for index in range(4)]
        window = store.read_ahead(blobs, max_bytes=64)
        store.close()
        self.assertIn(window.take(blobs[0]), (None, bytes([0]) * 32))
        with self.assertRaises(StateError):
            window.schedule(blobs)


class RawBridgeReadAheadTests(t.BridgeTestCase):
    """The SGLang bridge's RAW session restore reads its dense payloads ahead in ``restore_dense`` order."""

    representation = "raw"

    def resumed(self, store, read_ahead):
        bytes_ = KVRuntime.HISTORY_READ_AHEAD_BYTES if read_ahead else 0
        with mock.patch.object(KVRuntime, "HISTORY_READ_AHEAD_BYTES", bytes_):
            bridge = self.bridge(store=store)
        self.assertTrue(bridge.archive.runtime.verify_reads)
        t.drive(bridge, t.request(t.handle("s1", pause_after_chunks=2), prompt="A"))
        summary, latents, caches, _ = t.drive(bridge, t.request(t.handle("s1", "resume"), prompt="A"))
        self.assertEqual(summary["status"], "completed")
        return latents, caches, bridge.archive.stats()["read_ahead"]

    def test_resume_reads_ahead_and_restores_bitwise(self):
        on_latents, on_caches, on = self.resumed("read-ahead", True)
        off_latents, off_caches, off = self.resumed("no-read-ahead", False)
        self.assertLatentsEqual(on_latents, off_latents)
        self.assertCachesEqual(on_caches, off_caches)
        self.assertEqual((on["windows"], on["failed"], on["plan_failures"]), (1, 0, 0))
        self.assertGreater(on["taken"], 0)
        self.assertEqual((off["windows"], off["max_bytes"]), (0, 0))
        with self.assertRaises(ConfigurationError):
            RawSessionArchive(str(self.workspace / "bad"), MODEL, device="cpu", read_ahead_bytes=-1)
