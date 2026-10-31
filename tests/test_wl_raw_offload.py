"""Workload v2 (CONTRACT §10, wl-w1b): ``representation="raw_offload"`` -- the RAW archive behind the raw tiers (GPU raw
-> host raw, LRU, the same budget fields) with an ordered asynchronous write-behind -- and ``store_io`` (the object
store's page-cache policy), on the CPU toy harness without SGLang.

``raw_offload`` must restore bitwise what F6's synchronous ``raw`` restores, pay no write inside generation (commits
return before their objects are durable; backpressure only when the tier's budgets are full of unwritten chunks,
recorded with its cause), never evict a tier entry before its node is durable, publish every session's nodes in
commit order, and after a crash keep exactly the written nodes (unwritten commits are lost, the head stays a written
node). ``store_io="nocache"`` drops every object file's pages after its durable write and after every read, on every
read path (store API, read-ahead, RAW restores, latents, the lineage cold path and the restore staging readers).
"""

import builtins
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import torch

from reelcache import restore_staging
from reelcache.errors import ConfigurationError, StateError
from reelcache.integrations.sglang import ReelCacheRuntimeConfig
from reelcache.restore_staging import StagingEngine
from reelcache.store import ObjectStore
from tests.step6_support import RuntimeFactory, chunk, method, registry
from tests.test_sglang_bridge import handle
from tests.test_wl_tree import TreeStage, TreeTestCase, assert_caches_bitwise, run_tree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ObjectIoAudit:
    """Object-store files opened for reading under ``root`` and the ``POSIX_FADV_DONTNEED`` advice given on them (every
    thread: ``open``, ``io.open``, ``os.open`` and ``os.posix_fadvise`` are patched while the audit runs)."""

    def __init__(self, root):
        self.root = os.path.realpath(str(root))
        self.opened, self.advised, self.lock = [], [], threading.Lock()

    def _object(self, path):
        path = os.path.realpath(os.fspath(path))
        parent, name = os.path.split(path)
        is_object = (os.path.basename(parent) in ("blob", "basis", "manifest") and len(name) == 64
                     and all(char in "0123456789abcdef" for char in name))
        return path if is_object and path.startswith(self.root) else None

    def __enter__(self):
        real_open, real_os_open, real_fadvise = builtins.open, os.open, os.posix_fadvise
        audit = self

        def tracked_open(file, mode="r", *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)) and "r" in mode and "+" not in mode:
                path = audit._object(file)
                if path is not None:
                    with audit.lock:
                        audit.opened.append(path)
            return real_open(file, mode, *args, **kwargs)

        def tracked_os_open(path, flags, *args, **kwargs):
            if not flags & (os.O_WRONLY | os.O_RDWR):
                item = audit._object(path)
                if item is not None:
                    with audit.lock:
                        audit.opened.append(item)
            return real_os_open(path, flags, *args, **kwargs)

        def tracked_fadvise(fd, offset, length, advice):
            if advice == os.POSIX_FADV_DONTNEED:
                with audit.lock:
                    audit.advised.append(os.path.realpath(os.readlink(f"/proc/self/fd/{fd}")))
            return real_fadvise(fd, offset, length, advice)

        self.patches = [mock.patch("builtins.open", tracked_open), mock.patch("io.open", tracked_open),
                        mock.patch("os.open", tracked_os_open), mock.patch("os.posix_fadvise", tracked_fadvise)]
        for patch in self.patches:
            patch.start()
        return self

    def __exit__(self, *exc):
        for patch in reversed(self.patches):
            patch.stop()
        return False

    def undropped(self):
        return sorted(set(self.opened) - set(self.advised))


class OffloadCase(TreeTestCase):
    representation = "raw_offload"
    tree = "head"

    def entry_bytes(self):
        """Raw-tier bytes of one toy chunk commit (one stream)."""
        probe = self.bridge("probe-entry")
        run_tree(probe, self.request_for(handle("p"), "A"))
        probe.checkpoint("p")
        return probe.archive.raw_tier.stats()["largest_entry_bytes"]

    def build(self, bridge, letters, *, session="s", flush=True):
        """One request per chunk (generate, then resumes on the resident KV of a kept stage), each written before the
        next: no restore touches the raw tier's LRU."""
        stage = TreeStage()
        for depth in range(1, len(letters) + 1):
            run_tree(bridge, self.request_for(handle(session, "generate" if depth == 1 else "resume"),
                                              letters[:depth]), stage=stage)
            bridge.checkpoint(session)
            if flush and bridge.archive.raw_tier is not None:
                bridge.archive.raw_tier.flush()

    def restore_child(self, bridge, letters, *, session="s"):
        """Drop the resident KV, restore the session head from the archive and generate one more chunk."""
        bridge.drop_resident()
        return run_tree(bridge, self.request_for(handle(session, "resume"), letters))

    def tiers(self, summary):
        return {item["chunk"]: item["tier"] for item in summary["restore"]["lease"]["tiers"]["per_chunk"]}

    def assert_same(self, left, right):
        self.assertTrue(torch.equal(left[1], right[1]))
        assert_caches_bitwise(self, left[2], right[2])


class RawOffloadTests(OffloadCase):
    def test_the_options_select_the_write_behind_raw_archive(self):
        bridge = self.bridge()
        archive = bridge.archive
        self.assertEqual((archive.name, archive.asynchronous, archive.write_behind, archive.kv_objects),
                         ("raw_offload", True, True, True))
        self.assertIsNotNone(archive.raw_tier)
        self.assertTrue(bridge.startup["archive"]["write_behind"])
        raw = self.bridge("raw", representation="raw")
        self.assertEqual((raw.archive.name, raw.archive.asynchronous, raw.archive.raw_tier),
                         ("raw_exact", False, None))
        self.assertEqual(raw.archive.store.io_policy, "buffered")

    def test_restores_take_gpu_raw_then_host_raw_then_the_store_bitwise_equal_to_raw(self):
        raw, offload = self.bridge("raw", representation="raw"), self.bridge("offload")
        self.build(raw, "ABCD")
        self.build(offload, "ABCD")
        placed = offload.control("raw_tier", {"session_id": "s", "action": "place",
                                              "placement": {"0": "compressed", "1": "host_raw", "2": "gpu_raw",
                                                            "3": "gpu_raw"}})
        self.assertEqual(placed["report"]["store"]["entries"], 1)
        expected = self.restore_child(raw, "ABCDE")
        got = self.restore_child(offload, "ABCDE")
        self.assertEqual(got[0]["restore"]["mode"], "archive")
        self.assertEqual(self.tiers(got[0]), {0: "store", 1: "host_raw", 2: "gpu_raw", 3: "gpu_raw"})
        record = got[0]["restore"]["lease"]["tiers"]
        self.assertEqual(record["chunks"], {"store": 1, "host_raw": 1, "gpu_raw": 2})
        self.assertEqual(sum(record["bytes"].values()), 4 * record["per_chunk"][0]["bytes"])
        self.assertEqual(record["store_io"], "buffered")
        self.assert_same(got, expected)  # restored caches and the new chunk, bitwise F6 raw
        self.assertEqual(offload.archive.stats()["write_behind"]["failed"], 0)

    def test_lru_budgets_keep_the_newest_copies_and_the_store_serves_the_rest(self):
        size = self.entry_bytes()
        raw = self.bridge("raw", representation="raw")
        offload = self.bridge("offload", gpu_raw_budget_bytes=size, host_raw_budget_bytes=2 * size)
        self.build(raw, "ABCDE")
        self.build(offload, "ABCDE")
        stats = offload.archive.raw_tier.stats()
        self.assertEqual(stats["entries"], {"total": 3, "device_only": 1, "host_only": 2, "both": 0})
        self.assertEqual((stats["evictions_host"], stats["unpublished_entries"]), (2, 0))
        expected = self.restore_child(raw, "ABCDEF")
        got = self.restore_child(offload, "ABCDEF")
        self.assertEqual(self.tiers(got[0]), {0: "store", 1: "store", 2: "host_raw", 3: "host_raw", 4: "gpu_raw"})
        self.assert_same(got, expected)

    def test_commits_return_before_their_writes_which_publish_in_order_and_only_then_evict(self):
        bridge = self.bridge()
        archive, gate, order = bridge.archive, threading.Event(), []

        def hook(stage, job):
            order.append(job.node_id)
            self.assertTrue(gate.wait(60))
        archive.write_hook = hook
        summary, latents, caches, _ = run_tree(bridge, self.request_for(handle("s"), "ABC"))
        # The request returned with all three commits unwritten: no write was paid inside generation.
        self.assertFalse(summary["published"])
        self.assertEqual(len(summary["pending"]), 3)
        self.assertEqual(summary["archive"]["frontiers"]["pending"], 3)
        state = bridge.controller.get("s")
        self.assertEqual(archive.head("s"), state.root_ref)  # nothing durable yet: the store head is the root
        tier = archive.raw_tier.stats()
        self.assertEqual((tier["unpublished_entries"], tier["entries"]["total"], tier["drops_published"],
                          tier["drops_released"]), (3, 3, 0, 0))  # unwritten entries are never evicted
        gate.set()
        head = bridge.checkpoint("s")["head"]
        nodes = archive.path(head)[1:]
        self.assertEqual([node.chunk_index for node in nodes], [0, 1, 2])
        self.assertEqual(order, summary["pending"])  # written in commit order
        self.assertEqual(archive.raw_tier.stats()["unpublished_entries"], 0)
        latents_ref, caches_ref, _ = self.fresh("ABC")
        self.assertTrue(torch.equal(latents, latents_ref))
        assert_caches_bitwise(self, caches, caches_ref)

    def test_backpressure_only_when_the_budgets_are_full_of_unwritten_chunks(self):
        size = self.entry_bytes()
        # W1.3 hardening: explicit budgets that hold every entry on the device (a derived device budget can demote
        # copies, and a commit then waits for an in-progress spill, ``gpu_spill``, depending on thread timing).
        roomy = self.bridge("roomy", gpu_raw_budget_bytes=8 * size, host_raw_budget_bytes=8 * size)
        gate = threading.Event()
        roomy.archive.write_hook = lambda stage, job: gate.wait(60)
        run_tree(roomy, self.request_for(handle("s"), "ABCD"))
        self.assertEqual(roomy.archive.backpressure(), (0, 0.0))  # budgets admit: never waits on writes
        gate.set()
        roomy.checkpoint("s")
        tight = self.bridge("tight", gpu_raw_budget_bytes=size, host_raw_budget_bytes=size)
        gate2, finished, waiting = threading.Event(), [], threading.Event()
        tight.archive.write_hook = lambda stage, job: gate2.wait(60)
        admission = tight.archive.raw_tier.admission

        def observed(nbytes):  # deterministic (W1.3 hardening): the test sees the commit start waiting for room
            cause = admission(nbytes)
            if cause is not None:
                waiting.set()
            return cause
        tight.archive.raw_tier.admission = observed
        request = self.request_for(handle("s"), "ABCD")
        worker = threading.Thread(target=lambda: finished.append(run_tree(tight, request)[0]), daemon=True)
        worker.start()
        self.assertTrue(waiting.wait(60))  # the third commit waits: two unwritten chunks fill G + H
        self.assertTrue(worker.is_alive())
        self.assertEqual(tight.archive.raw_tier.stats()["unpublished_entries"], 2)
        gate2.set()
        worker.join(60)
        self.assertFalse(worker.is_alive())
        waits, seconds = tight.archive.backpressure()
        self.assertGreaterEqual(waits, 1)
        self.assertGreater(seconds, 0.0)
        stats = tight.archive.stats()
        self.assertGreaterEqual(stats["write_behind"]["admission_causes"].get("host_full", 0), 1)
        self.assertGreaterEqual(stats["raw_tier"]["last_resort_causes"]["host_full"], 1)
        self.assertGreaterEqual(finished[0]["archive"]["backpressure"]["waits"], 1)
        commits = tight.commit_reports(request.request_id)
        self.assertTrue(any((item.get("admission") or {}).get("cause") == "host_full" for item in commits))
        head = tight.checkpoint("s")["head"]
        self.assertEqual([node.chunk_index for node in tight.archive.path(head)[1:]], [0, 1, 2, 3])

    def test_a_failed_write_retires_the_session_tail_and_the_session_continues(self):
        bridge = self.bridge()
        calls, release = [], threading.Event()

        def hook(stage, job):
            calls.append(job.node_id)
            if len(calls) == 2:  # B fails once the request returned (an asynchronous failure)
                release.wait(60)
                raise OSError("injected: disk full")
        bridge.archive.write_hook = hook
        summary, *_ = run_tree(bridge, self.request_for(handle("s"), "ABC"))
        self.assertIsNone(summary["archive_error"])
        release.set()
        bridge.archive._wb_wait_idle(60)
        with self.assertRaisesRegex(StateError, "archival failed"):
            bridge.checkpoint("s")
        state = bridge.controller.get("s")
        nodes = bridge.archive.path(state.head_ref)[1:]
        self.assertEqual([node.chunk_index for node in nodes], [0])  # A was written; B failed, C retired
        self.assertEqual(state.generated_chunks, 1)
        stats = bridge.archive.write_behind_stats()
        self.assertEqual((stats["failed"], stats["cancelled"], stats["written"]), (1, 1, 1))
        self.assertEqual(bridge.archive.raw_tier.stats()["unpublished_entries"], 0)
        bridge.archive.write_hook = None
        _, latents, caches, _ = run_tree(bridge, self.request_for(handle("s", "resume"), "AB"))
        bridge.checkpoint("s")
        latents_ref, caches_ref, _ = self.fresh("AB")
        self.assertTrue(torch.equal(latents, latents_ref))
        assert_caches_bitwise(self, caches, caches_ref)

    def test_replace_abandons_unwritten_commits(self):
        bridge = self.bridge()
        gate, archive = threading.Event(), bridge.archive
        archive.write_hook = lambda stage, job: gate.wait(60)
        run_tree(bridge, self.request_for(handle("s"), "ABC"))
        tickets = [item["ticket"] for item in bridge.controller.get("s").pending]
        done = []
        worker = threading.Thread(target=lambda: done.append(run_tree(
            bridge, self.request_for(handle("s", "generate", replace=True), "A"))[0]), daemon=True)
        worker.start()
        deadline = time.monotonic() + 60  # W1.3 hardening: wait for the cancellation itself, not a fixed delay
        while archive.write_behind_stats()["cancelled"] < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        gate.set()  # the job being written finishes, sees the cancellation and never publishes
        worker.join(60)
        self.assertFalse(worker.is_alive())
        self.assertTrue(all(ticket.done() and ticket.error is not None for ticket in tickets))
        self.assertEqual(archive.write_behind_stats()["cancelled"], 2)
        head = bridge.checkpoint("s")["head"]
        self.assertEqual([node.chunk_index for node in archive.path(head)[1:]], [0])
        self.assertEqual(archive.raw_tier.stats()["unpublished_entries"], 0)

    def test_drop_and_demote_controls_and_records(self):
        bridge = self.bridge()
        self.build(bridge, "ABC", flush=False)
        demoted = bridge.control("raw_tier", {"session_id": "s", "action": "demote"})
        self.assertEqual(demoted["raw_tier"]["entries"]["device_only"], 0)
        dropped = bridge.control("raw_tier", {"session_id": "s", "action": "drop"})
        self.assertEqual(dropped["report"]["entries"], 3)
        bridge.drop_resident()
        request = self.request_for(handle("s", "resume"), "ABCD")
        got = run_tree(bridge, request)
        self.assertEqual(set(self.tiers(got[0]).values()), {"store"})
        bridge.checkpoint("s")
        stats = bridge.describe()["archive_stats"]
        self.assertEqual(stats["write_behind"]["written"], 4)
        self.assertEqual(stats["store_io"]["policy"], "buffered")
        reports = bridge.commit_reports(request.request_id)
        self.assertEqual(reports[-1]["representation"], "raw_offload")
        self.assertGreaterEqual(reports[-1]["queue_seconds"], 0.0)

    def test_restart_loses_unwritten_commits_and_restores_the_written_ones(self):
        root, result = self.workspace / "crashed", self.workspace / "result.json"
        spec = {"root": str(root), "result": str(result)}
        log = self.workspace / "crash_child.log"
        with open(log, "w") as output:
            child = subprocess.Popen([sys.executable, "-c", RAW_OFFLOAD_CRASH_CHILD, json.dumps(spec)], cwd=ROOT,
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, stdout=output,
                                     stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 600
                while not result.exists():
                    if child.poll() is not None:
                        self.fail(f"crash child exited early: {log.read_text()[-3000:]}")
                    if time.monotonic() > deadline:
                        self.fail("crash child never queued its commits")
                    time.sleep(0.05)
                time.sleep(0.2)
                child.send_signal(signal.SIGKILL)
                child.wait(60)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(60)
        self.assertEqual(child.returncode, -signal.SIGKILL)
        before = json.loads(result.read_text())
        self.assertEqual(len(before["pending"]), 2)
        restarted = self.bridge("crashed")
        state = restarted.controller.get("s")
        self.assertEqual(state.head_ref, before["written"])  # C and D were queued, never written: lost
        self.assertEqual(state.generated_chunks, 2)
        restarted.archive.store.check(state.head_ref)
        _, latents, caches, _ = run_tree(restarted, self.request_for(handle("s", "resume"), "ABC"))
        latents_ref, caches_ref, _ = self.fresh("ABC")
        self.assertTrue(torch.equal(latents, latents_ref))
        assert_caches_bitwise(self, caches, caches_ref)
        nodes = restarted.archive.path(restarted.checkpoint("s")["head"])[1:]
        restarted.drop_resident()
        run_tree(restarted, self.request_for(handle("s", "rewind", checkpoint=nodes[0].ref), "AB"))
        self.assertEqual(restarted.checkpoint("s")["head"], before["written"])  # B regenerated: the written node
        self.assertGreaterEqual(restarted.collect_garbage()["objects_removed"], 0)


RAW_OFFLOAD_CRASH_CHILD = """
import json, sys, time
import tests.test_wl_tree as t
from reelcache.integrations.sglang import CausalWanKVBridge, ReelCacheRuntimeConfig

spec = json.loads(sys.argv[1])
values = {"store_root": spec["root"], "representation": "raw_offload", "fit_device": "cpu", "decoder": "software",
          "energy_key": 0.98, "energy_value": 0.95}
bridge = CausalWanKVBridge(ReelCacheRuntimeConfig.from_dict(values), t.PROFILE, device="cpu", model_path="toy/causal-wan")
t.run_tree(bridge, t.tree_request(t.handle("s"), ["pa"]))
t.run_tree(bridge, t.tree_request(t.handle("s", "resume"), ["pa", "pb"]))
written = bridge.checkpoint("s")["head"]
bridge.archive.write_hook = lambda stage, job: time.sleep(3600)  # the writer stalls before C's objects
t.run_tree(bridge, t.tree_request(t.handle("s", "resume"), ["pa", "pb", "pc", "pd"]))
pending = [item["node_id"] for item in bridge.controller.get("s").pending]
with open(spec["result"] + ".tmp", "w") as handle:
    json.dump({"written": written, "pending": pending}, handle)
import os
os.replace(spec["result"] + ".tmp", spec["result"])
time.sleep(3600)
"""


class RawOffloadTreeTests(OffloadCase):
    tree = "full"

    def test_a_to_f_on_raw_offload_is_bitwise_raw_and_owns_every_node(self):
        raw, offload = self.bridge("raw", representation="raw"), self.bridge("offload")
        raw_nodes, raw_results = self.build_a_to_f(raw)
        nodes, results = self.build_a_to_f(offload)
        for letter in "ABCDEF":
            self.assert_same(results[letter], raw_results[letter])
        self.assertEqual(sorted(offload.controller.tree("s")), [1, 2, 3, 4, 5, 6])
        self.assertEqual(offload.controller.tree("s")[6], nodes["F"])
        # Every owned node is a target; regenerating C from B deduplicates to C under a new generation index.
        offload.drop_resident()
        run_tree(offload, self.request_for(handle("s", "rewind", checkpoint=nodes["B"]), "ABC"))
        self.assertEqual(self.head(offload), nodes["C"])
        self.assertEqual(offload.controller.tree("s")[7], nodes["C"])
        raw_root = raw.archive.path(raw_nodes["A"])[0].ref
        self.assertNotEqual(raw_root, offload.archive.path(nodes["A"])[0].ref)  # the method is hashed content


class StoreIoTests(unittest.TestCase):
    def setUp(self):
        import shutil
        import tempfile
        self.root = tempfile.mkdtemp(prefix="w1b-store-io-")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.addCleanup(shutil.rmtree, self.root + "-other", True)

    def test_nocache_drops_pages_after_durable_writes_and_every_read(self):
        store = ObjectStore(self.root)
        self.addCleanup(store.close)
        self.assertEqual(store.io_stats()["policy"], "buffered")
        with ObjectIoAudit(self.root) as audit:
            first = store.put(b"x" * 4096)
            store.get(first)
        self.assertEqual(audit.advised, [])  # buffered (F6): no advice at all
        self.assertEqual(len(audit.opened), 1)
        store.set_io_policy("nocache")
        with ObjectIoAudit(self.root) as audit:
            blobs = [store.put(bytes([index]) * (3000 + index)) for index in range(6)]
            manifest = store.put_manifest({"kind": "test"}, references=blobs)
            self.assertEqual(store.get(blobs[0]), bytes([0]) * 3000)
            self.assertEqual(len(store.get_many(blobs, workers=3)), 6)
            window = store.read_ahead(blobs, max_bytes=1 << 20)
            self.assertEqual(window.take(blobs[2]), bytes([2]) * 3002)
            window.close()
            store.get_manifest(manifest)
            store.verify(manifest)
        self.assertEqual(audit.undropped(), [])
        self.assertGreater(len(audit.opened), 10)
        stats = store.io_stats()
        self.assertEqual((stats["policy"], stats["errors"]), ("nocache", 0))
        self.assertEqual(stats["write_calls"], 7)  # six blobs and the manifest: dropped once durable
        self.assertEqual(stats["write_bytes"], sum(3000 + index for index in range(6)) + store.object_size(manifest))
        self.assertGreaterEqual(stats["read_calls"], 6 + 6 + 1 + 1)
        # Pages left by an earlier process: dropped when the policy is set with drop_existing.
        reopened = ObjectStore(self.root + "-other")
        self.addCleanup(reopened.close)
        reopened.put(b"y" * 100)
        with ObjectIoAudit(self.root + "-other") as audit:
            report = reopened.set_io_policy("nocache", drop_existing=True)
        self.assertEqual((report["existing_calls"], report["existing_bytes"]), (1, 100))
        self.assertEqual(audit.undropped(), [])
        with self.assertRaises(ConfigurationError):
            store.set_io_policy("direct")

    def test_failed_advice_is_recorded_and_the_bytes_stay_correct(self):
        store = ObjectStore(self.root)
        self.addCleanup(store.close)
        store.set_io_policy("nocache")
        with mock.patch("os.posix_fadvise", side_effect=OSError(22, "unsupported")):
            ref = store.put(b"z" * 512)
            self.assertEqual(store.get(ref), b"z" * 512)
        stats = store.io_stats()
        self.assertEqual(stats["errors"], 2)
        self.assertIn("unsupported", stats["last_error"])

    def test_the_restore_staging_readers_drop_pages(self):
        runtimes = RuntimeFactory(self)
        engine = StagingEngine("cpu", threads=3, prepare_threads=2, pinned_bytes=1 << 20)
        self.addCleanup(engine.close)
        runtime = runtimes()
        runtime.create_lineage("main")
        reg = registry()
        commits = [runtime.commit_node("main", f"chunk-{index:03d}", *chunk(index), method=method(), registry=reg)
                   for index in range(3)]
        from reelcache.archive_restore import SoftwareLineageRestorer

        history = SoftwareLineageRestorer(runtime).history(commits[-1].node_ref)
        runtime.store.set_io_policy("nocache")
        # Small payloads (one read task), large ones as parallel ranges, basis tips: every staged file is dropped.
        for parallel in (1 << 30, 2048):
            with mock.patch.object(restore_staging, "_PARALLEL_READ_BYTES", parallel), \
                    mock.patch.object(restore_staging, "_READ_CHUNK_BYTES", 1000), \
                    ObjectIoAudit(runtime.store.root) as audit:
                staged = engine.stage(runtime, history)
                staged.wait()
                staged.release()
            self.assertEqual(audit.undropped(), [])
            self.assertTrue(any(os.sep + "basis" + os.sep in path for path in audit.opened))
            self.assertGreater(len(audit.opened), 3)


class StoreIoBridgeTests(OffloadCase):
    def test_raw_offload_nocache_cold_restores_drop_every_object_read(self):
        bridge = self.bridge(store_io="nocache")
        self.assertEqual(bridge.archive.store.io_policy, "nocache")
        self.build(bridge, "ABC", flush=False)
        bridge.control("raw_tier", {"session_id": "s", "action": "drop"})
        with ObjectIoAudit(bridge.archive.store.root) as audit:
            got = self.restore_child(bridge, "ABCD")
            bridge.checkpoint("s")
        self.assertEqual(set(self.tiers(got[0]).values()), {"nvme"})  # cold by construction
        self.assertEqual(got[0]["restore"]["lease"]["tiers"]["store_io"], "nocache")
        self.assertEqual(audit.undropped(), [])
        packets = [blob for node in bridge.archive.path(self.head(bridge))[1:4]
                   for refs in bridge.archive._manifest(node.ref)["packets"].values() for packet in refs
                   for blob in bridge.archive.store.get_manifest(packet)["payloads"].values()]
        read = set(audit.opened)
        self.assertTrue(all(str(bridge.archive.store.path_for(blob)) in read for blob in packets))
        stats = bridge.archive.stats()["store_io"]
        self.assertGreater(stats["write_calls"], 0)
        self.assertGreater(stats["read_calls"], 0)
        latents_ref, caches_ref, _ = self.fresh("ABCD")
        self.assertTrue(torch.equal(got[1], latents_ref))

    def test_raw_nocache_sync_writes_reads_and_latents_are_dropped(self):
        bridge = self.bridge(representation="raw", store_io="nocache")
        with ObjectIoAudit(bridge.archive.store.root) as audit:
            self.build(bridge, "AB")
            self.restore_child(bridge, "ABC")
        self.assertEqual(audit.undropped(), [])
        self.assertGreater(bridge.archive.store.io_stats()["write_calls"], 0)

    def test_store_io_buffered_is_f6_and_nocache_stores_the_same_nodes(self):
        buffered = self.bridge("buffered", representation="raw")
        nocache = self.bridge("nocache", representation="raw", store_io="nocache")
        with ObjectIoAudit(self.workspace) as audit:
            self.build(buffered, "AB")
        self.assertEqual(audit.advised, [])
        self.build(nocache, "AB")
        self.assertEqual(self.head(buffered), self.head(nocache))  # store_io is run configuration, not content
        self.assertNotIn("store_io", buffered.startup["archive"])
        self.assertNotIn("store_io", buffered.describe())  # F6 records unchanged
        self.assertEqual(nocache.describe()["store_io"]["policy"], "nocache")

    def test_lineage_compressed_cold_path_with_nocache(self):
        bridge = self.bridge("lineage", representation="svd_hevc", store_io="nocache")
        self.assertEqual(bridge.archive.store.io_policy, "nocache")
        with ObjectIoAudit(bridge.archive.store.root) as audit:
            self.build(bridge, "AB", flush=False)
            bridge.control("raw_tier", {"session_id": "s", "action": "drop"})
            got = self.restore_child(bridge, "ABC")
            bridge.checkpoint("s")
        self.assertEqual(audit.undropped(), [])
        self.assertTrue(any(os.sep + "blob" + os.sep in path for path in audit.opened))
        self.assertEqual(got[0]["restore"]["mode"], "archive")
        recorded = bridge.describe()["store_io"]  # the lineage arm's policy and bytes (CONTRACT §10)
        self.assertEqual(recorded["policy"], "nocache")
        self.assertGreater(recorded["read_bytes"], 0)
        self.assertGreater(recorded["write_bytes"], 0)


class ConfigAndSeamHelperTests(unittest.TestCase):
    def test_config_fields(self):
        config = ReelCacheRuntimeConfig.from_dict({"store_root": "x"})
        self.assertEqual(config.store_io, "buffered")
        self.assertNotIn("store_io", config.content_dict())
        offload = ReelCacheRuntimeConfig.from_dict({"store_root": "x", "representation": "raw_offload",
                                                 "store_io": "nocache"})
        self.assertEqual((offload.representation, offload.store_io), ("raw_offload", "nocache"))
        for bad in ({"store_io": "direct"}, {"representation": "raw_offload", "raw_tier": False},
                    {"representation": "raw_offload", "restore_mode": "recompute"}):
            with self.assertRaises(ConfigurationError):
                ReelCacheRuntimeConfig.from_dict({"store_root": "x", **bad})

    def test_lifecycle_accepts_raw_offload_without_canvas_staging(self):
        from reelcache_sglang_models.causal_wan.lifecycle import bridge_runtime_config
        from reelcache_sglang_models.causal_wan.profiles import get_profile

        config = bridge_runtime_config(get_profile("longlive1"), store_root="/tmp/x", representation="raw_offload")
        self.assertEqual(config["representation"], "raw_offload")
        self.assertNotIn("max_staged_canvases", config)
        self.assertEqual(ReelCacheRuntimeConfig.from_dict(config).representation, "raw_offload")

    def test_replay_seam_helpers(self):
        from reelcache_sglang_models.causal_wan.lifecycle import REPLAY_SEAM_MARKER, needs_replay_seam, pipeline_stages

        class Native:  # SGLang's own causal stage: the seam without the replay hook
            def _begin_causal_kv_run(self, batch, server_args, **kwargs):
                return None, 0

        class Profile(Native):  # ReelCache's profile stage: the hook on its class
            def _begin_causal_kv_run(self, batch, server_args, **kwargs):
                return None, 0
        setattr(Profile._begin_causal_kv_run, REPLAY_SEAM_MARKER, True)

        class TextEncoder:
            pass

        stages = [TextEncoder(), Native(), Profile()]
        pipeline = type("Pipeline", (), {"stages": property(lambda self: stages)})()
        worker = type("Worker", (), {})()
        worker.pipeline = pipeline
        self.assertEqual(pipeline_stages(pipeline), stages)
        self.assertEqual(pipeline_stages(worker), stages)
        self.assertEqual(pipeline_stages(stages), stages)
        self.assertEqual([needs_replay_seam(stage) for stage in stages], [False, True, False])
        wrapped = stages[1]._begin_causal_kv_run

        def begin(batch, server_args, **kwargs):
            return wrapped(batch, server_args, **kwargs)
        setattr(begin, REPLAY_SEAM_MARKER, True)
        stages[1]._begin_causal_kv_run = begin  # what install_replay_seam does: idempotent afterwards
        self.assertFalse(needs_replay_seam(stages[1]))


if __name__ == "__main__":
    unittest.main()
