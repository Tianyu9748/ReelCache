"""Workload v2 W1.3 (wl-w1b4): drain-free rewind (``tree_switch="detach"``) on the CPU toy harness, without SGLang.

A tree-mode rewind waits only for its target's own commit; the session's pending commits keep publishing in order to a
temporary per-branch ref, binding their tree refs and sidecars as attached publications do, while the session head
moves at once. Checked on raw_offload (write-behind) and the lineage archive (``svd_hevc``; ``svd`` as the fixed arm),
in the runner's geometry (requests sized for a fixed session total, ``pause_after_chunks = depth + 1``,
``durable=False``), with archival held by event gates -- no wall-clock thresholds:

* (a) drain mode equals W1.2: node for node (``tests.test_wl_tree``: ``test_options_off_equal_f6_node_for_node``) and
  record for record (``DrainRecordsTests``: request summaries, workload records, commit reports, bridge description
  and archive stats against the frozen W1.2 tree, F7);
* (b) A-F: a rewind to a published target waits for nothing while the left branch publishes afterwards; the final
  tree (refs, sidecars, generation indices) equals drain mode's -- node for node and bitwise on the exact arm;
* (c) a pending target waits only through its own commit (current branch; a detached branch at the archive level);
* (d) a failed detached commit fails its branch tail, is recorded, never fails the session; a rewind to it is a clear
  error; no raw-tier holder or binding leaks;
* (e) crash / reopen (SIGKILL): stale temporary refs are removed, published refs stay valid, ``sessions()`` hides them;
* (f) close and delete with a detached backlog;
* (g) detached commits count toward the same in-flight bounds;

and one test per design-review finding: F1 (raw_offload publish window), F2 (a published detached recache stays
rooted), F3 (delete / replace fail closed), F4 (all-or-nothing detach, the drain fallback), F5 (a detach waits for at
most one publication), F6 (bindings, failure records, garbage collection after ``drain_all``), F7 (above), F8 (commit
reports of detached commits), F9 (the owned raw copy does not depend on the sidecar).
"""

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock

import torch

from reelcache.errors import StateError
from reelcache.integrations.sglang import session as session_module
from reelcache.integrations.sglang.nodes import (DETACHED_KEEP_PREFIX, DETACHED_REF_PREFIX, TREE_HOLDER_SUFFIX,
                                              detached_keep_ref, is_detached_branch, tree_prefix)
from tests.assets import asset
from tests.test_sglang_bridge import handle, stored
from tests.test_wl_tree import PROMPTS, SESSION_TOTAL, TreeTestCase, assert_caches_bitwise, run_tree, tree_request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMEOUT = 300


def w12_tree() -> str:
    """The frozen W1.2 tree drain mode is compared with, record for record (F7): the freeze's copy under the release's
    package names (freeze_w12 TREE_ID 29daee2e9c90b3cf, renamed), a test asset."""
    return str(asset("frozen", "freeze_w12", "tree", override="REELCACHE_W12_TREE"))


class Gate:
    """Event gates on archival, per arrival: the raw_offload writer before a job's write (``write_hook``), the lineage
    pipeline before a node's ``stage`` (``hooks``). ``hold(ids)`` holds the next arrival of each node id (a later
    commit with the same id -- chunk ids repeat across branches -- passes); ``hold_all()`` holds every arrival until
    ``release()``; ``fail(id)`` fails the next arrival of the id once it passes. Waiting arrivals are recorded."""

    def __init__(self, bridge, stage="encode"):
        self._cond = threading.Condition()
        self._queued, self._events, self._fail = {}, {}, {}
        self._all = None
        self.arrived = []
        archive = bridge.archive
        if getattr(archive, "write_behind", False):
            archive.write_hook = lambda _stage, job: self._pass(job.node_id)
        else:
            archive.pipeline.hooks = lambda current, node_id: self._pass(node_id) if current == stage else None

    def hold(self, *node_ids):
        with self._cond:
            for node_id in node_ids:
                event = threading.Event()
                self._queued.setdefault(node_id, []).append(event)
                self._events.setdefault(node_id, []).append(event)

    def hold_all(self):
        with self._cond:
            self._all = threading.Event()

    def fail(self, node_id):
        with self._cond:
            self._fail[node_id] = self._fail.get(node_id, 0) + 1

    def release(self, *node_ids):
        with self._cond:
            if not node_ids and self._all is not None:
                self._all.set()
            for node_id in node_ids or list(self._events):
                for event in self._events.get(node_id, []):
                    event.set()

    def _pass(self, node_id):
        with self._cond:
            queued = self._queued.get(node_id)
            event = queued.pop(0) if queued else self._all
            failing = self._fail.get(node_id, 0) > 0
            if failing:
                self._fail[node_id] -= 1
            if event is not None and not event.is_set():
                self.arrived.append(node_id)
                self._cond.notify_all()
        if event is not None and not event.wait(TIMEOUT):
            raise TimeoutError(f"the gate never released {node_id}")
        if failing:
            raise OSError(f"injected: the archival of {node_id} failed")

    def wait_arrivals(self, count):
        with self._cond:
            return self._cond.wait_for(lambda: len(self.arrived) >= count, TIMEOUT)


class DetachCase(TreeTestCase):
    """Tree mode in the runner's geometry, with ``tree_switch="detach"`` (or the default drain)."""

    tree = "full"
    exact = True  # the archive restores exact history (node content does not depend on rank decisions)

    def detach_bridge(self, store="detach", **overrides):
        return self.bridge(store, tree_switch="detach", **overrides)

    def plan_request(self, path, action, *, index, checkpoint=None, durable=False):
        """A runner request: the path's prompts padded to the session total, pausing after its new chunk(s)."""
        extra = {} if checkpoint is None else {"checkpoint": checkpoint}
        session = handle("s", action, pause_after_chunks=len(path), durable=durable, **extra)
        prompts = [PROMPTS[letter] for letter in path + path[-1] * (SESSION_TOTAL - len(path))]
        return tree_request(session, prompts, recache=self.recache, streams=self.streams, generation_index=index)

    @staticmethod
    def refs_of(bridge, summary):
        """The published refs of a request's chunk commits (the runner's ``output_ref(wait=True)``)."""
        reports = bridge.commit_reports(summary["request_id"], wait=True, timeout=TIMEOUT)
        return [item["node"] for item in reports if item.get("kind") == "chunk" and item.get("node")]

    def published_ab(self, bridge):
        """A1, B2: generated and published (the runner waits for its source refs)."""
        refs = {}
        for letter, action, path, index in (("A", "generate", "A", 1), ("B", "resume", "AB", 2)):
            refs[letter] = self.refs_of(bridge, run_tree(bridge, self.plan_request(path, action, index=index))[0])[-1]
        return refs

    def run_plan(self, bridge, *, gate=None):
        """A1 B2 C3 D4, E5 from B, F6 from D. With ``gate``, D's commit is held across E's rewind (the switch must not
        wait for it) and released before F's rewind targets it."""
        refs, results = {}, {}

        def go(letter, action, path, index, target=None):
            results[letter] = run_tree(bridge, self.plan_request(
                path, action, index=index, checkpoint=None if target is None else refs[target]))
            return results[letter][0]

        for letter, action, path, index in (("A", "generate", "A", 1), ("B", "resume", "AB", 2),
                                            ("C", "resume", "ABC", 3)):
            refs[letter] = self.refs_of(bridge, go(letter, action, path, index))[-1]
        if gate is not None:
            gate.hold("c000003")  # D's commit
        summary_d = go("D", "resume", "ABCD", 4)
        if gate is not None:
            self.assertTrue(gate.wait_arrivals(1))
        summary_e = go("E", "rewind", "ABE", 5, target="B")
        if gate is not None:  # E's switch moved the head without waiting: D is detached, still held
            [branch] = bridge.controller.get("s").detached
            self.assertEqual([item["node_id"] for item in branch["items"]], ["c000003"])
            self.assertFalse(branch["items"][0]["ticket"].done())
            gate.release()
        refs["D"] = self.refs_of(bridge, summary_d)[-1]
        refs["E"] = self.refs_of(bridge, summary_e)[-1]
        refs["F"] = self.refs_of(bridge, go("F", "rewind", "ABCDF", 6, target="D"))[-1]
        bridge.checkpoint("s")  # the whole tree durable (detached branches too)
        return refs, results

    def sidecars(self, bridge):
        """Every owned node's tree sidecar without run identities (request ids differ between runs)."""
        out = {}
        for index, ref in bridge.controller.tree("s").items():
            sidecar = dict(bridge.archive.store.telemetry(ref)["bridge"]["tree"])
            sidecar.pop("request_id", None)
            sidecar["generations"] = [(item["session_id"], item["generation_index"])
                                      for item in sidecar["generations"]]
            out[index] = sidecar
        return out

    def structure(self, bridge):
        """The tree's shape independent of node ids: per generation index its depth, chunk, parent's index, prompts."""
        tree = bridge.controller.tree("s")
        index_of = {ref: index for index, ref in tree.items()}
        return {index: (sidecar["depth"], sidecar["chunk_index"], index_of.get(sidecar["parent"]),
                        [item["text"] for item in sidecar["prompt_schedule"]["blocks"]])
                for index, sidecar in self.sidecars(bridge).items()}

    @staticmethod
    def detached_refs(bridge):
        """Temporary refs of detached branches (raw_offload head refs; lineage arms: detached lineages)."""
        archive = bridge.archive
        if getattr(archive, "write_behind", False):
            return archive.store.list_refs(DETACHED_REF_PREFIX)
        return {name: ref for name, ref in archive.runtime.list_lineages().items() if is_detached_branch(name)}

    @staticmethod
    def bindings(bridge):
        """Publication bindings (refs set with a commit's publication) still registered."""
        archive = bridge.archive
        if getattr(archive, "write_behind", False):
            return dict(archive._publication_refs)
        return dict(archive.runtime._publication_refs)

    @staticmethod
    def detach_target(bridge):
        """The object whose ``detach`` and ``detach_hook`` hand over in-flight commits."""
        archive = bridge.archive
        return archive if getattr(archive, "write_behind", False) else archive.pipeline

    @staticmethod
    def raw_holders(bridge, ref):
        """The holders of ``ref``'s raw-tier entry (empty when it has none, or no entry: it was dropped)."""
        entries = bridge.archive.raw_tier.entries(node_refs=[ref])
        return set() if not entries else set(entries[0].holders)

    def assert_no_detached_leftovers(self, bridge):
        self.assertEqual(self.detached_refs(bridge), {})
        self.assertFalse([key for key in self.bindings(bridge) if is_detached_branch(key[0])])
        tier = bridge.archive.raw_tier
        if tier is not None:
            self.assertFalse([holder for entry in tier.entries() for holder in entry.holders
                              if is_detached_branch(holder)])

    def assert_new_chunks_bitwise(self, results, letters_paths):
        for letter, path in letters_paths:
            latents, caches, rng = self.fresh(path)
            _, got_latents, got_caches, got_rng = results[letter]
            self.assertTrue(torch.equal(got_latents[:, :, :latents.shape[2]], latents), letter)
            assert_caches_bitwise(self, got_caches, caches)
            self.assertTrue(torch.equal(got_rng.get_state(), rng.get_state()), letter)


class DetachChecks:
    """(b)-(g) and F3, F4, F6 on one asynchronous arm."""

    # -- (b) -------------------------------------------------------------------------------------------------------
    def test_b_a_to_f_switches_without_waiting_and_ends_with_the_drain_tree(self):
        drained = self.bridge("drain")
        drain_refs, drain_results = self.run_plan(drained)
        bridge = self.detach_bridge()
        refs, results = self.run_plan(bridge, gate=Gate(bridge))
        summary = results["E"][0]
        switch = summary["restore"]["switch"]
        # The rewind to B (published) waited for nothing: D was detached unpublished and published afterwards.
        self.assertEqual((switch["switch_mode"], switch["detached_at_switch"], switch["drain_seconds"],
                          switch["switch_fallback"]), ("detach", 1, 0.0, False))
        self.assertEqual((summary["switch_mode"], summary["detached_at_switch"], summary["switch_fallback"],
                          summary["switch_fallback_reason"]), ("detach", 1, False, None))
        self.assertNotIn(refs["D"], [node.ref for node in bridge.archive.path(refs["E"])])
        # Drain mode records none of the W1.3 fields (F7).
        self.assertFalse({"switch_mode", "detached_at_switch", "switch_fallback"} & set(drain_results["E"][0]))
        self.assertNotIn("switch_mode", drain_results["E"][0]["restore"]["switch"])
        self.assertNotIn("tree_switch", drained.describe())
        # The final tree equals drain mode's: indices, sidecars, structure; node for node and bitwise on the exact arm.
        self.assertEqual(sorted(bridge.controller.tree("s")), [1, 2, 3, 4, 5, 6])
        self.assertEqual(self.structure(bridge), self.structure(drained))
        if self.exact:
            self.assertEqual(refs, drain_refs)
            self.assertEqual(bridge.controller.tree("s"), drained.controller.tree("s"))
            self.assertEqual(bridge.archive.store.list_refs(tree_prefix("s")),
                             drained.archive.store.list_refs(tree_prefix("s")))
            self.assertEqual(self.sidecars(bridge), self.sidecars(drained))
            for letter in "ABCDEF":
                self.assertTrue(torch.equal(results[letter][1], drain_results[letter][1]), letter)
            self.assert_new_chunks_bitwise(results, (("D", "ABCD"), ("E", "ABE"), ("F", "ABCDF")))
        self.assertEqual(bridge.archive.head("s"), refs["F"])
        stats = bridge.describe()["tree_switch"]
        self.assertEqual((stats["mode"], stats["detached_branches"], stats["switch_fallbacks"]),
                         ("detach", {"opened": 1, "closed": 1}, 0))
        self.assertEqual(stats["detached_commits"], {"detached": 1, "published": 1, "failed": 0})
        self.assertEqual(stats["detached_backlog"], {"current": 0, "peak": 1})
        self.assertEqual(sorted(bridge.archive.sessions()), ["s"])
        self.assert_no_detached_leftovers(bridge)
        self.assertEqual(self.bindings(bridge), {})  # every binding was consumed or dropped (F6)
        self.assertEqual(bridge.archive.store.list_refs(DETACHED_KEEP_PREFIX), {})
        tier = bridge.archive.raw_tier
        held = {entry.node_ref: set(entry.holders) for entry in tier.entries()}
        self.assertIn("s" + TREE_HOLDER_SUFFIX, held[refs["D"]])  # the left branch stays in the LRU, owned

    # -- (c) -------------------------------------------------------------------------------------------------------
    def test_c_a_pending_target_waits_only_through_its_own_commit(self):
        bridge = self.detach_bridge()
        gate = Gate(bridge)
        self.published_ab(bridge)
        gate.hold("c000002", "c000003")  # C and D: generated by one request, both unpublished
        summary = run_tree(bridge, self.plan_request("ABCD", "resume", index=4))[0]
        self.assertTrue(gate.wait_arrivals(1))
        tickets = {item["node_id"]: item["ticket"] for item in bridge.controller.get("s").pending}
        drains, called, real = [], threading.Event(), bridge.archive.drain

        def watched(session_id, *, through=None, timeout=None):
            drains.append((session_id, through))
            called.set()
            return real(session_id, through=through, timeout=timeout)
        bridge.archive.drain = watched
        done = []
        rewind = self.plan_request("ABCG", "rewind", index=5, checkpoint=3)  # 3 chunks: C, still pending
        worker = threading.Thread(target=lambda: done.append(run_tree(bridge, rewind)[0]), daemon=True)
        worker.start()
        self.assertTrue(called.wait(TIMEOUT))
        self.assertEqual(drains[0], ("s", "c000002"))  # through C only, never D
        self.assertTrue(worker.is_alive())  # C is held: the switch waits for it
        gate.release("c000002")
        worker.join(TIMEOUT)
        self.assertFalse(worker.is_alive())
        bridge.archive.drain = real
        switch = done[0]["restore"]["switch"]
        self.assertEqual((switch["detached_at_switch"], switch["switch_fallback"]), (1, False))
        self.assertGreater(switch["drain_seconds"], 0.0)
        self.assertNotIn("c000003", [through for _, through in drains])
        self.assertTrue(tickets["c000002"].done())
        self.assertFalse(tickets["c000003"].done())  # D was not waited for: detached, still held
        gate.release()
        d_ref = self.refs_of(bridge, summary)[-1]
        g_ref = self.refs_of(bridge, done[0])[-1]
        bridge.checkpoint("s")
        tree = bridge.controller.tree("s")
        self.assertEqual((tree[4], tree[5], sorted(tree)), (d_ref, g_ref, [1, 2, 3, 4, 5]))
        self.assertEqual(bridge.archive.load(g_ref).parent_ref, tree[3])  # G continues C (the target)
        self.assert_no_detached_leftovers(bridge)

    def test_c_a_detached_branch_drains_through_one_commit_at_the_archive(self):
        bridge = self.detach_bridge()
        gate = Gate(bridge)
        refs = self.published_ab(bridge)
        gate.hold("c000002", "c000003", "c000004")
        run_tree(bridge, self.plan_request("ABCDE", "resume", index=5))  # C, D, E (generations 3-5) unpublished
        self.assertTrue(gate.wait_arrivals(1))
        summary_g = run_tree(bridge, self.plan_request("ABG", "rewind", index=6, checkpoint=refs["B"]))[0]
        [branch] = bridge.controller.get("s").detached
        self.assertEqual([item["node_id"] for item in branch["items"]], ["c000002", "c000003", "c000004"])
        tickets = {item["node_id"]: item["ticket"] for item in branch["items"]}
        result = []
        worker = threading.Thread(target=lambda: result.append(bridge.archive.drain(
            branch["branch"], through="c000003", timeout=TIMEOUT)), daemon=True)
        worker.start()
        gate.release("c000002", "c000003")
        worker.join(TIMEOUT)
        self.assertFalse(worker.is_alive())
        self.assertTrue(tickets["c000002"].done() and tickets["c000003"].done())
        self.assertFalse(tickets["c000004"].done())  # the drain went through D only
        self.assertEqual(result[0], tickets["c000003"].result(0))
        gate.release()
        g_ref = self.refs_of(bridge, summary_g)[-1]
        bridge.checkpoint("s")
        self.assertEqual(sorted(bridge.controller.tree("s")), [1, 2, 3, 4, 5, 6])
        self.assertEqual(bridge.archive.head("s"), g_ref)
        self.assert_no_detached_leftovers(bridge)

    # -- (d) -------------------------------------------------------------------------------------------------------
    def test_d_a_failed_detached_commit_fails_its_branch_tail_and_never_the_session(self):
        reference = self.bridge("reference")  # drain mode: the node the failed commit would have published
        self.published_ab(reference)
        would_be = self.refs_of(reference, run_tree(reference, self.plan_request("ABCD", "resume", index=4))[0])
        bridge = self.detach_bridge()
        gate = Gate(bridge)
        refs = self.published_ab(bridge)
        gate.hold("c000002", "c000003")
        gate.fail("c000002")  # C fails: D, its descendant on the detached branch, never publishes
        run_tree(bridge, self.plan_request("ABCD", "resume", index=4))
        self.assertTrue(gate.wait_arrivals(1))
        summary_g = run_tree(bridge, self.plan_request("ABG", "rewind", index=5, checkpoint=refs["B"]))[0]
        self.assertEqual(summary_g["detached_at_switch"], 2)
        gate.release()
        g_ref = self.refs_of(bridge, summary_g)[-1]  # the live branch publishes: the session never failed
        bridge.checkpoint("s")
        state = bridge.controller.get("s")
        self.assertEqual((state.archive_failures, state.detached), (0, []))
        failures = list(state.detach_failures.values())
        self.assertEqual(sorted(event["node_id"] for event in failures), ["c000002", "c000003"])
        self.assertEqual(sorted(event["generation_index"] for event in failures), [3, 4])
        self.assertIn("detached_commit_failed", [item["event"] for item in state.history])
        self.assertEqual(bridge.controller.tree("s"), {1: refs["A"], 2: refs["B"], 5: g_ref})
        stats = bridge.describe()["tree_switch"]
        self.assertEqual((stats["detached_commits"]["failed"], stats["detached_branches"]),
                         (2, {"opened": 1, "closed": 1}))
        self.assert_no_detached_leftovers(bridge)
        self.assertEqual(bridge.archive.store.list_refs(DETACHED_KEEP_PREFIX), {})  # the cut (B) has its tree ref
        self.assertEqual(bridge.archive.raw_tier.stats()["unpublished_entries"], 0)
        # A rewind to the node the failed commit would have published is a clear error (never retargeted).
        for target in would_be:
            with self.assertRaises(StateError) as caught:
                bridge.controller.rewind("s", target)
            self.assertIn("failed and were never published", str(caught.exception))
            self.assertIn("c000002", str(caught.exception))
        run_tree(bridge, self.plan_request("ABH", "rewind", index=6, checkpoint=refs["B"]))  # the session continues
        bridge.checkpoint("s")
        self.assertEqual(sorted(bridge.controller.tree("s")), [1, 2, 5, 6])

    # -- (e) -------------------------------------------------------------------------------------------------------
    def test_e_crash_with_detached_work_in_flight_removes_stale_refs_at_reopen(self):
        root, result = (self.workspace / "crashed").resolve(), (self.workspace / "result.json").resolve()
        log = self.workspace / "detach_crash_child.log"
        spec = {"root": str(root), "result": str(result), "representation": self.representation}
        with open(log, "w") as output:
            child = subprocess.Popen([sys.executable, "-c", DETACH_CRASH_CHILD, json.dumps(spec)], cwd=ROOT,
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, stdout=output,
                                     stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 600
                while not result.exists():
                    if child.poll() is not None:
                        self.fail(f"crash child exited early: {log.read_text()[-3000:]}")
                    if time.monotonic() > deadline:
                        self.fail("crash child never detached")
                    time.sleep(0.05)
                child.send_signal(signal.SIGKILL)
                child.wait(60)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(60)
        self.assertEqual(child.returncode, -signal.SIGKILL)
        before = json.loads(result.read_text())
        self.assertEqual(len(before["branches"]), 1)  # D's branch, in flight at the crash
        self.assertEqual(before["detached_refs"], before["branches"])  # with its temporary ref
        self.assertEqual(before["sessions"], ["s"])  # the temporary ref was never a session
        restarted = self.bridge(str(root), tree_switch="detach")
        self.assertEqual(self.detached_refs(restarted), {})
        self.assertEqual(restarted.archive.store.list_refs(DETACHED_KEEP_PREFIX), {})  # its head (C) is owned
        self.assertEqual(restarted.controller.switch_stats()["stale_refs_removed"], before["branches"])
        self.assertEqual(restarted.archive.raw_tier.stats()["entries"]["total"], 0)  # no orphan raw entries
        self.assertEqual(sorted(restarted.archive.sessions()), ["s"])
        state = restarted.controller.get("s")
        self.assertEqual(state.head_ref, before["refs"]["B"])  # the rewind's head move was durable
        self.assertEqual(restarted.controller.tree("s"), {index + 1: before["refs"][letter]
                                                          for index, letter in enumerate("ABC")})
        summary, latents, caches, _ = run_tree(restarted, self.plan_request("ABCH", "rewind", index=6,
                                                                            checkpoint=before["refs"]["C"]))
        self.assertEqual((summary["restore"]["switch"]["switch_mode"], summary["generated_chunks"]), ("detach", 4))
        if self.exact:  # the published refs stay valid: C restores exactly
            fresh_latents, fresh_caches, _ = self.fresh("ABCH")
            self.assertTrue(torch.equal(latents[:, :, :fresh_latents.shape[2]], fresh_latents))
            assert_caches_bitwise(self, caches, fresh_caches)

    # -- (f) -------------------------------------------------------------------------------------------------------
    def detached_backlog(self, store):
        """A1 B2 published; C3 held at its archival; a rewind to A detaches it. Returns (bridge, gate, refs, ticket)."""
        bridge = self.detach_bridge(store)
        gate = Gate(bridge)
        refs = self.published_ab(bridge)
        gate.hold("c000002")
        refs["request_c"] = run_tree(bridge, self.plan_request("ABC", "resume", index=3))[0]["request_id"]
        self.assertTrue(gate.wait_arrivals(1))
        bridge.controller.rewind("s", refs["A"])
        [branch] = bridge.controller.get("s").detached
        [item] = branch["items"]
        self.assertFalse(item["ticket"].done())
        return bridge, gate, refs, item["ticket"]

    def test_f_close_publishes_a_detached_backlog(self):
        bridge, gate, refs, ticket = self.detached_backlog("close")
        entered, real = threading.Event(), bridge.controller._drain_detached

        def watched(state, **kwargs):
            entered.set()
            return real(state, **kwargs)
        bridge.controller._drain_detached = watched
        report = []
        worker = threading.Thread(target=lambda: report.append(bridge.controller.close_session("s")), daemon=True)
        worker.start()
        self.assertTrue(entered.wait(TIMEOUT))
        self.assertTrue(worker.is_alive())  # the close waits for the held detached commit
        gate.release()
        worker.join(TIMEOUT)
        self.assertFalse(worker.is_alive())
        self.assertTrue(ticket.done())
        self.assertIn("detached_wait_seconds", report[0])
        self.assertIsNone(bridge.controller.state("s"))
        tree = bridge.controller.tree("s")
        self.assertEqual((sorted(tree), tree[3]), ([1, 2, 3], ticket.result(0)))  # C published at the close
        self.assertEqual(bridge.archive.head("s"), refs["A"])
        self.assert_no_detached_leftovers(bridge)

    def test_f_delete_aborts_a_detached_backlog(self):
        bridge, gate, refs, ticket = self.detached_backlog("delete")
        entered, real = threading.Event(), bridge.archive.abandon_detached

        def watched(branch):
            entered.set()
            return real(branch)
        bridge.archive.abandon_detached = watched
        report = []
        worker = threading.Thread(target=lambda: report.append(bridge.controller.close_session("s", delete=True)),
                                  daemon=True)
        worker.start()
        self.assertTrue(entered.wait(TIMEOUT))
        gate.release()  # the commit being archived sees the abort and never publishes
        worker.join(TIMEOUT)
        self.assertFalse(worker.is_alive())
        self.assertEqual(report[0]["detached_retired"], ["c000002"])
        self.assertTrue(ticket.done() and ticket.error is not None)
        self.assertIsNone(bridge.controller.state("s"))  # its records went with it (F6)
        self.assertEqual(bridge.archive.store.list_refs(tree_prefix("s")), {})
        self.assertEqual(bridge.archive.store.list_refs(DETACHED_KEEP_PREFIX), {})
        self.assert_no_detached_leftovers(bridge)
        bridge.collect_garbage()
        self.assertEqual(stored(bridge, {refs["A"], refs["B"]}), set())

    def test_f3_delete_fails_closed_when_the_abort_fails(self):
        bridge, gate, refs, ticket = self.detached_backlog("failclosed")
        real, calls = bridge.archive.abandon_detached, []

        def failing(branch):
            calls.append(branch)
            if len(calls) == 1:
                raise OSError("injected: the abort of a detached branch failed")
            return real(branch)
        bridge.archive.abandon_detached = failing
        store = bridge.archive.store
        tree_before = store.list_refs(tree_prefix("s"))
        with self.assertRaises(StateError) as caught:
            bridge.controller.close_session("s", delete=True)
        self.assertIn("not aborted", str(caught.exception))
        state = bridge.controller.state("s")  # nothing was deleted: the session and its branch stay tracked
        self.assertNotEqual(state.status, "closed")
        self.assertEqual(len(state.detached), 1)
        self.assertEqual(store.list_refs(tree_prefix("s")), tree_before)
        self.assertEqual(len(self.detached_refs(bridge)), 1)
        gate.release()  # C publishes to its detached branch while the tree is intact: its tree ref is consistent
        c_ref = ticket.result(TIMEOUT)
        report = bridge.controller.close_session("s", delete=True)  # the retry succeeds
        self.assertEqual((len(calls), report["detached_retired"], report["deleted"]), (2, [], True))
        if getattr(bridge.archive, "write_behind", False):
            bridge.archive._wb_wait_idle(TIMEOUT)
        self.assertEqual(store.list_refs(tree_prefix("s")), {})  # quiescent before the delete: nothing reappears
        self.assertEqual(store.list_refs(DETACHED_KEEP_PREFIX), {})
        self.assert_no_detached_leftovers(bridge)
        bridge.collect_garbage()
        self.assertEqual(stored(bridge, {refs["A"], refs["B"], c_ref}), set())

    def test_f3_replace_fails_closed_when_the_abort_fails(self):
        bridge, gate, refs, ticket = self.detached_backlog("replace")
        real = bridge.archive.abandon_detached
        bridge.archive.abandon_detached = mock.Mock(side_effect=OSError("injected: the abort failed"))
        replace = tree_request(handle("s", "generate", replace=True), [PROMPTS["H"]] * SESSION_TOTAL)
        with self.assertRaises(StateError):
            run_tree(bridge, replace)
        self.assertEqual(len(bridge.controller.state("s").detached), 1)  # the old tree and branch are untouched
        self.assertEqual(sorted(bridge.controller.tree("s")), [1, 2])
        self.assertEqual(len(self.detached_refs(bridge)), 1)
        bridge.archive.abandon_detached = real
        gate.release()
        c_ref = ticket.result(TIMEOUT)
        run_tree(bridge, tree_request(handle("s", "generate", replace=True), [PROMPTS["H"]] * SESSION_TOTAL))
        bridge.checkpoint("s")
        tree = bridge.controller.tree("s")  # the replaced tree is gone (C too, published before the replace)
        self.assertEqual(sorted(tree), list(range(1, SESSION_TOTAL + 1)))
        self.assertFalse({refs["A"], refs["B"], c_ref} & set(tree.values()))
        self.assert_no_detached_leftovers(bridge)

    # -- F4 --------------------------------------------------------------------------------------------------------
    def test_f4_a_detach_is_all_or_nothing(self):
        for step in ("validated", "created", "transfer"):
            with self.subTest(step=step):
                bridge = self.detach_bridge(f"f4-{step}")
                gate = Gate(bridge)
                refs = self.published_ab(bridge)
                gate.hold("c000002")
                run_tree(bridge, self.plan_request("ABC", "resume", index=3))
                self.assertTrue(gate.wait_arrivals(1))
                called, seen = threading.Event(), []

                def hook(current, step=step, bridge=bridge, seen=seen, called=called):
                    if current == step:
                        seen.append(dict(self.detached_refs(bridge)))
                        called.set()
                        raise OSError(f"injected at {current}")
                self.detach_target(bridge).detach_hook = hook
                done = []
                worker = threading.Thread(target=lambda bridge=bridge, done=done: done.append(run_tree(
                    bridge, self.plan_request("ABG", "rewind", index=4, checkpoint=refs["B"]))[0]), daemon=True)
                worker.start()
                self.assertTrue(called.wait(TIMEOUT))
                gate.release()  # before the commit point the switch drains instead: C must publish
                worker.join(TIMEOUT)
                self.assertFalse(worker.is_alive())
                summary = done[0]
                stats = bridge.describe()["tree_switch"]
                if step == "transfer":  # after the commit point: fully detached, the bookkeeping error recorded
                    self.assertEqual((summary["switch_fallback"], summary["detached_at_switch"]), (False, 1))
                    counters = (bridge.archive.wb_stats_ if getattr(bridge.archive, "write_behind", False)
                                else bridge.archive.pipeline.stats_)
                    self.assertEqual(counters["detach_bookkeeping_errors"], 1)
                    tier = bridge.archive.raw_tier.stats()
                    self.assertFalse(tier["enabled"])
                    self.assertIn("bookkeeping error at detach", tier["disabled_reason"])
                    self.assertEqual(stats["switch_fallbacks"], 0)
                else:  # before it: nothing moved (the created ref rolled back); the switch drained
                    self.assertEqual(len(seen[0]), int(step == "created"))
                    self.assertEqual((summary["switch_fallback"], summary["detached_at_switch"]), (True, 0))
                    self.assertIn(f"injected at {step}", summary["switch_fallback_reason"])
                    self.assertEqual(stats["switch_fallbacks"], 1)
                    self.assertEqual(stats["detached_branches"]["opened"], 0)
                    self.assertGreater(summary["restore"]["switch"]["drain_seconds"], 0.0)
                    self.assertEqual(self.detached_refs(bridge), {})
                    # R2: C, published by the fallback drain, keeps the tree holder after the head left its branch
                    self.assertIn("s" + TREE_HOLDER_SUFFIX, self.raw_holders(bridge, bridge.controller.tree("s")[3]))
                g_ref = self.refs_of(bridge, summary)[-1]
                bridge.checkpoint("s")
                tree = bridge.controller.tree("s")
                self.assertEqual((sorted(tree), tree[4]), ([1, 2, 3, 4], g_ref))
                self.assertEqual(bridge.archive.load(tree[3]).parent_ref, refs["B"])
                self.assert_no_detached_leftovers(bridge)

    # -- R1 (controller): only a rolled-back detach falls back to a drain --------------------------------------------
    def test_r1_an_error_after_the_commit_point_never_falls_back(self):
        bridge = self.detach_bridge("r1c")
        gate = Gate(bridge)
        refs = self.published_ab(bridge)
        gate.hold("c000002")
        run_tree(bridge, self.plan_request("ABC", "resume", index=3))
        self.assertTrue(gate.wait_arrivals(1))
        real, drains = bridge.archive.detach, []

        def detach_then_fail(session_id, branch):
            real(session_id, branch)
            raise RuntimeError("injected after the commit point")
        bridge.archive.detach = detach_then_fail
        real_drain = bridge.archive.drain

        def drain(*args, **kwargs):
            drains.append(args)
            return real_drain(*args, **kwargs)
        bridge.archive.drain = drain
        with self.assertRaisesRegex(RuntimeError, "injected after the commit point"):
            bridge.controller.rewind("s", refs["A"])
        self.assertEqual(drains, [])  # never drained as if nothing had moved
        self.assertEqual(bridge.controller.switch_stats()["switch_fallbacks"], 0)
        bridge.archive.detach, bridge.archive.drain = real, real_drain
        gate.release()

    # -- R3: an abort that raised fails the destructive operation even when its branch closed after all --------------
    def abort_then_raise(self, bridge, entered=None):
        real, calls = bridge.archive.abandon_detached, []

        def wrapper(branch):
            calls.append(branch)
            if entered is not None:
                entered.set()
            result = real(branch)
            if len(calls) == 1:
                raise OSError("injected: the abort raised after aborting")
            return result
        bridge.archive.abandon_detached = wrapper
        return calls

    def test_r3_delete_fails_when_the_abort_raised_after_quiescing(self):
        bridge, gate, refs, ticket = self.detached_backlog("r3delete")
        entered = threading.Event()
        calls = self.abort_then_raise(bridge, entered)
        store, tree_before = bridge.archive.store, bridge.archive.store.list_refs(tree_prefix("s"))
        failure = []

        def delete():
            try:
                bridge.controller.close_session("s", delete=True)
            except StateError as exc:
                failure.append(exc)
        worker = threading.Thread(target=delete, daemon=True)
        worker.start()
        self.assertTrue(entered.wait(TIMEOUT))
        gate.release()  # the real abort completes (C retires unpublished), then the wrapper raises
        worker.join(TIMEOUT)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(failure), 1)  # the delete failed although the branch closed
        self.assertIn("aborting its detached branches failed", str(failure[0]))
        state = bridge.controller.state("s")
        self.assertEqual((state.status != "closed", state.detached), (True, []))
        self.assertEqual(store.list_refs(tree_prefix("s")), tree_before)  # nothing was deleted
        self.assertTrue(ticket.done() and ticket.error is not None)
        report = bridge.controller.close_session("s", delete=True)  # the retry succeeds: nothing left to abort
        self.assertEqual((len(calls), report["deleted"]), (1, True))
        if getattr(bridge.archive, "write_behind", False):
            bridge.archive._wb_wait_idle(TIMEOUT)
        self.assertEqual(store.list_refs(tree_prefix("s")), {})
        self.assert_no_detached_leftovers(bridge)

    def test_r3_replace_fails_when_the_abort_raised_after_quiescing(self):
        bridge, gate, refs, ticket = self.detached_backlog("r3replace")
        gate.release()
        ticket.result(TIMEOUT)  # C published on its branch; the abort then finds nothing in flight
        self.abort_then_raise(bridge)
        replace = tree_request(handle("s", "generate", replace=True), [PROMPTS["H"]] * SESSION_TOTAL)
        with self.assertRaises(StateError):
            run_tree(bridge, replace)
        self.assertEqual(sorted(bridge.controller.tree("s")), [1, 2, 3])  # the old tree is untouched
        run_tree(bridge, tree_request(handle("s", "generate", replace=True), [PROMPTS["H"]] * SESSION_TOTAL))
        bridge.checkpoint("s")
        self.assertEqual(sorted(bridge.controller.tree("s")), list(range(1, SESSION_TOTAL + 1)))
        self.assertFalse({refs["A"], refs["B"]} & set(bridge.controller.tree("s").values()))
        self.assert_no_detached_leftovers(bridge)

    # -- F3: a failed close is retried (the branch stays tracked meanwhile) ------------------------------------------
    def test_f3_a_failed_close_is_retried_by_release_idle(self):
        bridge, gate, refs, ticket = self.detached_backlog("closeretry")
        real, calls, broken = bridge.archive.close_detached, [], threading.Event()
        broken.set()

        def failing(branch, *args, **kwargs):
            calls.append(branch)
            if broken.is_set():
                raise OSError("injected: the close failed")
            return real(branch, *args, **kwargs)
        bridge.archive.close_detached = failing
        gate.release()
        c_ref = ticket.result(TIMEOUT)
        bridge.checkpoint("s")  # whole-tree durable; the close failures are recorded, never raised here
        state = bridge.controller.get("s")
        self.assertEqual(len(state.detached), 1)
        self.assertIn("injected: the close failed", state.describe()["detached"][0]["close_error"])
        self.assertEqual(len(self.detached_refs(bridge)), 1)  # its temporary ref stays until a close succeeds
        failures = bridge.describe()["tree_switch"]["close_failures"]
        self.assertEqual(failures, len(calls))
        self.assertGreaterEqual(failures, 1)
        broken.clear()
        bridge.controller.release_idle()  # the retry
        self.assertEqual((len(calls), state.detached), (failures + 1, []))
        self.assertEqual(bridge.controller.tree("s")[3], c_ref)
        self.assert_no_detached_leftovers(bridge)

    # -- F6 --------------------------------------------------------------------------------------------------------
    def test_f6_gc_drains_and_closes_detached_branches_first(self):
        bridge, gate, refs, ticket = self.detached_backlog("gc")
        entered, real = threading.Event(), bridge.controller.drain_detached

        def watched(state, **kwargs):
            entered.set()
            return real(state, **kwargs)
        bridge.controller.drain_detached = watched
        result = []
        worker = threading.Thread(target=lambda: result.append(bridge.collect_garbage()), daemon=True)
        worker.start()
        self.assertTrue(entered.wait(TIMEOUT))
        self.assertTrue(worker.is_alive())  # garbage collection waits for the held detached commit
        gate.release()
        worker.join(TIMEOUT)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(result[0]["drain_failure"])
        tree = bridge.controller.tree("s")
        self.assertEqual((sorted(tree), tree[3]), ([1, 2, 3], ticket.result(0)))
        self.assertEqual(stored(bridge, set(tree.values())), set(tree.values()))  # nothing owned was collected
        self.assertEqual(bridge.controller.get("s").detached, [])
        self.assert_no_detached_leftovers(bridge)
        self.assertEqual(self.bindings(bridge), {})

    def test_f6_failure_records_are_bounded(self):
        with mock.patch.object(session_module, "DETACH_FAILURES_KEPT", 1):
            bridge = self.detach_bridge()
            gate = Gate(bridge)
            refs = self.published_ab(bridge)
            gate.hold("c000002")
            gate.fail("c000002")
            run_tree(bridge, self.plan_request("ABCD", "resume", index=4))  # C fails, D with it
            self.assertTrue(gate.wait_arrivals(1))
            bridge.controller.rewind("s", refs["B"])
            gate.release()
            bridge.checkpoint("s")
            state = bridge.controller.get("s")
            self.assertEqual([event["node_id"] for event in state.detach_failures.values()], ["c000003"])
            self.assertEqual(bridge.describe()["tree_switch"]["detached_commits"]["failed"], 2)
            self.assert_no_detached_leftovers(bridge)

    # -- F9 --------------------------------------------------------------------------------------------------------
    def test_f9_the_owned_raw_copy_never_depends_on_the_sidecar(self):
        bridge, gate, refs, ticket = self.detached_backlog("f9")
        real = bridge.archive.annotate
        bridge.archive.annotate = mock.Mock(side_effect=OSError("injected: the sidecar write failed"))
        gate.release()
        c_ref = ticket.result(TIMEOUT)
        bridge.checkpoint("s")  # reconciles the detached publication and closes its branch (holder released)
        bridge.archive.annotate = real
        events = [item["event"] for item in bridge.controller.get("s").history]
        self.assertIn("telemetry_failed", events)
        self.assertNotIn("tree_hold_failed", events)
        [entry] = bridge.archive.raw_tier.entries(node_refs=[c_ref])
        self.assertEqual(set(entry.holders), {"s" + TREE_HOLDER_SUFFIX})  # still owned, in the LRU
        self.assertEqual(bridge.controller.tree("s")[3], c_ref)
        self.assert_no_detached_leftovers(bridge)


class RawOffloadDetachTests(DetachChecks, DetachCase):
    """raw_offload: the write-behind's detached sessions publish to ``sglang/detached/<branch>``."""

    representation = "raw_offload"

    def test_r4_every_binding_access_holds_the_bindings_lock(self):
        bridge = self.detach_bridge()
        archive = bridge.archive
        checked = LockCheckedDict(archive._publication_refs, archive._bindings_lock)
        archive._publication_refs = checked
        gate = Gate(bridge)
        refs = self.published_ab(bridge)
        gate.hold("c000002")
        run_tree(bridge, self.plan_request("ABC", "resume", index=3))  # C held at the writer
        self.assertTrue(gate.wait_arrivals(1))
        bridge.controller.rewind("s", refs["A"])  # re-keys C's binding to the detached branch
        other = tree_request(handle("x"), [PROMPTS["A"], PROMPTS["B"]])  # another session binds and publishes
        run_tree(bridge, other)
        gate.release()
        bridge.checkpoint("s")  # C publishes (its binding popped), the branch closes (its bindings dropped)
        bridge.checkpoint("x")
        bridge.controller.close_session("x", delete=True)  # unbinds the session
        self.assertEqual(checked.unlocked, [])
        self.assertEqual(len(bridge.controller.tree("s")), 3)

    def test_r4_a_close_never_races_another_sessions_publication(self):
        bridge = self.detach_bridge()
        archive = bridge.archive
        gate = Gate(bridge)
        refs = self.published_ab(bridge)
        gate.hold("c000002")
        run_tree(bridge, self.plan_request("ABC", "resume", index=3))
        self.assertTrue(gate.wait_arrivals(1))
        bridge.controller.rewind("s", refs["A"])  # C detached, held at the writer
        [branch] = bridge.controller.get("s").detached
        racing = RacingDict(archive._publication_refs)
        archive._publication_refs = racing
        inside, go, real_publish = threading.Event(), threading.Event(), archive._publish_head

        def publish_head(session_id, node_id, node_ref, expected, *, name=None):  # x's writer, inside _publish_head
            if session_id == "x" and not go.is_set():
                inside.set()
                go.wait(TIMEOUT)
            return real_publish(session_id, node_id, node_ref, expected, name=name)
        archive._publish_head = publish_head
        run_tree(bridge, tree_request(handle("x"), [PROMPTS["A"], PROMPTS["B"]]))  # bound; queued behind C
        gate.release()
        branch["items"][0]["ticket"].result(TIMEOUT)  # C published on its branch, which closes at s's next reconcile
        self.assertTrue(inside.wait(TIMEOUT))  # x's writer is inside _publish_head, its binding not popped yet
        threading.Thread(target=lambda: (racing.iterating.wait(TIMEOUT), go.set()), daemon=True).start()
        racing.armed.set()
        bridge.checkpoint("s")  # closes s's branch: iterates the bindings while x's writer pops its own
        racing.armed.clear()
        go.set()
        state = bridge.controller.get("s")
        self.assertEqual((state.detached, state.describe().get("detached")), ([], None))
        self.assertEqual(bridge.describe()["tree_switch"]["close_failures"], 0)
        self.assertTrue(racing.iterating.is_set())
        bridge.checkpoint("x")
        self.assertEqual(sorted(bridge.controller.tree("x")), [1, 2])
        self.assert_no_detached_leftovers(bridge)

    def entry_bytes(self):
        probe = self.bridge("probe-entry", tree_retention="head")
        run_tree(probe, self.request_for(handle("p"), "A"))
        probe.checkpoint("p")
        return probe.archive.raw_tier.stats()["largest_entry_bytes"]

    def test_g_backpressure_counts_detached_unwritten_chunks(self):
        size = self.entry_bytes()
        bridge = self.detach_bridge(gpu_raw_budget_bytes=size, host_raw_budget_bytes=size)
        gate = Gate(bridge)
        refs = self.published_ab(bridge)
        gate.hold("c000002", "c000003")
        run_tree(bridge, self.plan_request("ABCD", "resume", index=4))  # C, D unwritten: G + H full
        self.assertTrue(gate.wait_arrivals(1))
        waiting, admission = threading.Event(), bridge.archive.raw_tier.admission

        def observed(nbytes):
            cause = admission(nbytes)
            if cause is not None:
                waiting.set()
            return cause
        bridge.archive.raw_tier.admission = observed
        done = []
        worker = threading.Thread(target=lambda: done.append(run_tree(bridge, self.plan_request(
            "ABG", "rewind", index=5, checkpoint=refs["B"]))[0]), daemon=True)
        worker.start()
        self.assertTrue(waiting.wait(TIMEOUT))  # G waits: the detached C and D still fill the budgets (one limit)
        self.assertTrue(worker.is_alive())
        self.assertEqual(len(bridge.controller.get("s").detached), 1)
        self.assertEqual(bridge.archive.raw_tier.stats()["unpublished_entries"], 2)
        gate.release()
        worker.join(TIMEOUT)
        self.assertFalse(worker.is_alive())
        self.assertEqual(done[0]["detached_at_switch"], 2)
        self.assertGreaterEqual(bridge.archive.backpressure()[0], 1)
        bridge.checkpoint("s")
        self.assertEqual(sorted(bridge.controller.tree("s")), [1, 2, 3, 4, 5])
        self.assert_no_detached_leftovers(bridge)


class LockCheckedDict(dict):
    """A dict recording every access made without ``lock`` held by the accessing thread (R4)."""

    def __init__(self, data, lock):
        super().__init__(data)
        self.lock, self.unlocked = lock, []

    def _check(self, op):
        if not self.lock._is_owned():
            self.unlocked.append((op, threading.current_thread().name))

    def __getitem__(self, key):
        self._check("getitem")
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        self._check("setitem")
        super().__setitem__(key, value)

    def __delitem__(self, key):
        self._check("delitem")
        super().__delitem__(key)

    def __contains__(self, key):
        self._check("contains")
        return super().__contains__(key)

    def __iter__(self):
        self._check("iter")
        return super().__iter__()

    def get(self, key, default=None):
        self._check("get")
        return super().get(key, default)

    def pop(self, key, *default):
        self._check("pop")
        return super().pop(key, *default)

    def keys(self):
        self._check("keys")
        return super().keys()

    def items(self):
        self._check("items")
        return super().items()


class RacingDict(dict):
    """R4: iterating yields one key, then (when ``armed``) lets another thread mutate before going on (``popped``
    set by a pop elsewhere; bounded wait: with the lock the pop cannot happen meanwhile)."""

    def __init__(self, data):
        super().__init__(data)
        self.armed, self.iterating, self.popped = threading.Event(), threading.Event(), threading.Event()

    def __iter__(self):
        inner = super().__iter__()
        first = True
        for key in inner:
            yield key
            if first and self.armed.is_set():
                first = False
                self.iterating.set()
                self.popped.wait(2.0)

    def pop(self, key, *default):
        value = super().pop(key, *default)
        if self.iterating.is_set():
            self.popped.set()
        return value


class FullyDetachedRecacheChecks:
    """R2b: a recache epoch detached before any of its nodes published (raw_offload: one node per range)."""

    def test_r2b_a_fully_detached_recache_keeps_the_tree_holder_on_every_node(self):
        for fail_chunk in (False, True):
            with self.subTest(fail_chunk=fail_chunk):
                bridge = self.detach_bridge(f"r2b-{fail_chunk}")
                gate = Gate(bridge)
                refs = self.published_ab(bridge)
                gate.hold_all()  # the epoch and C are held at their archival: the whole epoch moves to the branch
                if fail_chunk:
                    gate.fail("c000002")  # the epoch publishes, its chunk C fails: no tree ref covers the epoch
                run_tree(bridge, self.plan_request("ABC", "resume", index=3))
                self.assertTrue(gate.wait_arrivals(1))
                bridge.controller.rewind("s", refs["A"])
                state = bridge.controller.get("s")
                [branch] = state.detached
                self.assertEqual([item["kind"] for item in branch["items"]], ["recache", "chunk"])
                epoch_ticket, chunk_ticket = (item["ticket"] for item in branch["items"])
                gate.release()
                bridge.checkpoint("s")  # the branch publishes and closes (its holder released)
                self.assertEqual(state.detached, [])
                epoch, ref = [], epoch_ticket.result(0)
                while ref != refs["B"]:  # every node of the epoch (raw_offload: the first range's piece too)
                    epoch.append(ref)
                    ref = bridge.archive.load(ref).parent_ref
                self.assertEqual(len(epoch), 2 if getattr(bridge.archive, "write_behind", False) else 1)
                nodes = epoch + ([] if fail_chunk else [chunk_ticket.result(0)])
                holder = "s" + TREE_HOLDER_SUFFIX
                self.assertEqual({ref: holder in self.raw_holders(bridge, ref) for ref in nodes},
                                 {ref: True for ref in nodes})
                bridge.controller.rewind("s", refs["B"])  # a later switch (an exclusive tree hold) keeps them too
                self.assertEqual({ref: holder in self.raw_holders(bridge, ref) for ref in nodes},
                                 {ref: True for ref in nodes})
                self.assert_no_detached_leftovers(bridge)


class RawOffloadRecacheDetachTests(FullyDetachedRecacheChecks, DetachCase):
    """raw_offload with LongLive-1.0-like recache epochs (one node per re-materialized chunk range): F1, F2."""

    representation = "raw_offload"
    recache = True

    def test_f1_a_detach_never_sees_a_published_piece_before_the_head_advanced(self):
        bridge = self.detach_bridge()
        archive = bridge.archive
        refs = self.published_ab(bridge)
        names, real_publish = [], archive._publish_head

        def publish_head(binding, node_id, node_ref, parent, *, name=None):
            names.append((node_ref, name))
            return real_publish(binding, node_id, node_ref, parent, name=name)
        archive._publish_head = publish_head
        entered, proceed, pieces = threading.Event(), threading.Event(), []

        def hook(job, node_ref):  # between the compare-and-swap of the epoch's first piece and the state update
            if job.kind == "recache" and not pieces:
                pieces.append(node_ref)
                entered.set()
                if not proceed.wait(TIMEOUT):
                    raise TimeoutError("never released")
        archive.publish_hook = hook
        summary_c = run_tree(bridge, self.plan_request("ABC", "resume", index=3))[0]  # epoch (2 pieces), then C
        self.assertTrue(entered.wait(TIMEOUT))
        detaching, real_detach = threading.Event(), archive.detach

        def detach(session_id, branch):
            detaching.set()
            return real_detach(session_id, branch)
        archive.detach = detach
        worker = threading.Thread(target=lambda: bridge.controller.rewind("s", refs["A"]), daemon=True)
        worker.start()
        self.assertTrue(detaching.wait(TIMEOUT))
        proceed.set()
        worker.join(TIMEOUT)
        self.assertFalse(worker.is_alive())
        state = bridge.controller.get("s")
        [branch] = state.detached
        self.assertEqual(len(branch["items"]), 2)  # the epoch and C
        self.assertEqual(branch["cut"], pieces[0])  # the detach saw the published piece as the durable head
        # R2: the cut (published under the session, off the target's path) keeps the tree holder after the head moved
        self.assertIn("s" + TREE_HOLDER_SUFFIX, self.raw_holders(bridge, pieces[0]))
        c_ref = self.refs_of(bridge, summary_c)[-1]
        bridge.checkpoint("s")
        self.assertEqual(state.detach_failures, {})  # the epoch's next piece and C published on the branch
        temporary = DETACHED_REF_PREFIX + branch["branch"]
        published = dict(names)
        second = archive.load(c_ref).parent_ref
        self.assertEqual(archive.load(second).parent_ref, pieces[0])
        self.assertEqual((published[pieces[0]], published[second], published[c_ref]), (None, temporary, temporary))
        self.assertEqual((archive.head("s"), bridge.controller.tree("s")[3]), (refs["A"], c_ref))
        for ref in (pieces[0], second, c_ref):  # R2: every node of the closed branch stays owned in the LRU
            self.assertIn("s" + TREE_HOLDER_SUFFIX, self.raw_holders(bridge, ref))
        self.assert_no_detached_leftovers(bridge)

    def test_f2_a_published_detached_recache_stays_rooted_when_its_chunk_fails(self):
        bridge = self.detach_bridge()
        gate = Gate(bridge)
        refs = self.published_ab(bridge)
        gate.hold_all()
        gate.fail("c000002")  # the epoch publishes, its chunk C fails
        run_tree(bridge, self.plan_request("ABC", "resume", index=3))
        self.assertTrue(gate.wait_arrivals(1))
        bridge.controller.rewind("s", refs["A"])
        state = bridge.controller.get("s")
        [branch] = state.detached
        epoch_ticket = branch["items"][0]["ticket"]
        gate.release()
        bridge.checkpoint("s")
        last = epoch_ticket.result(0)
        first = bridge.archive.load(last).parent_ref
        self.assertEqual(bridge.archive.load(first).parent_ref, refs["B"])
        self.assertEqual(sorted(event["node_id"] for event in state.detach_failures.values()), ["c000002"])
        keep = detached_keep_ref("s", branch["branch"])
        self.assertEqual(bridge.archive.store.list_refs(DETACHED_KEEP_PREFIX), {keep: last})
        self.assertNotIn(keep, bridge.archive.sessions())
        bridge.collect_garbage()
        self.assertEqual(stored(bridge, {first, last}), {first, last})  # the clean epoch survives collection
        bridge.controller.close_session("s", delete=True)  # the whole tree: the kept root goes with it
        self.assertEqual(bridge.archive.store.list_refs(DETACHED_KEEP_PREFIX), {})
        bridge.collect_garbage()
        self.assertEqual(stored(bridge, {first, last, refs["A"], refs["B"]}), set())


class LineageDetachTests(DetachChecks, DetachCase):
    """The lineage archive (``svd_hevc``): detached pipeline lineages publish to ``lineages/<branch>``. Rank decisions
    may follow the predictor's publication order (lead decision 8): the tree is compared by structure."""

    representation = "svd_hevc"
    exact = False

    def test_g_c3_bounds_count_detached_unpublished_commits(self):
        bridge = self.detach_bridge(raw_tier=False, max_source_chunks=2, max_unpublished=2)
        gate = Gate(bridge)
        refs = self.published_ab(bridge)
        gate.hold("c000002", "c000003")
        run_tree(bridge, self.plan_request("ABCD", "resume", index=4))  # C, D unpublished: max_unpublished reached
        self.assertTrue(gate.wait_arrivals(1))  # C fitted (its sources released) and held before its encode
        pipeline, counted = bridge.archive.pipeline, threading.Event()
        real = pipeline._unpublished_of

        def observed(lineage):
            count = real(lineage)
            if lineage.branch_id == "s" and not lineage.entries and count >= pipeline.max_unpublished:
                counted.set()  # the live lineage has none of its own: the detached ones count
            return count
        pipeline._unpublished_of = observed
        done = []
        worker = threading.Thread(target=lambda: done.append(run_tree(bridge, self.plan_request(
            "ABG", "rewind", index=5, checkpoint=refs["B"]))[0]), daemon=True)
        worker.start()
        self.assertTrue(counted.wait(TIMEOUT))
        self.assertTrue(worker.is_alive())  # G waits: its session's detached commits count (C3, one bound)
        gate.release()
        worker.join(TIMEOUT)
        self.assertFalse(worker.is_alive())
        self.assertEqual(done[0]["detached_at_switch"], 2)
        self.assertGreaterEqual(pipeline.stats_["backpressure_waits"], 1)
        bridge.checkpoint("s")
        self.assertEqual(sorted(bridge.controller.tree("s")), [1, 2, 3, 4, 5])
        self.assert_no_detached_leftovers(bridge)

    def test_f5_a_detach_waits_for_at_most_one_publication(self):
        # Four encode workers: C's publication holds one (inside its transaction), D's waits for the publisher on
        # another, E is encoded and stored on a third.
        bridge = self.detach_bridge(encode_workers=4)
        gate = Gate(bridge, stage="publish")
        refs = self.published_ab(bridge)
        gate.hold("c000002")  # C publishes (held inside its transaction); D and E are stored behind it
        run_tree(bridge, self.plan_request("ABCDE", "resume", index=5))
        self.assertTrue(gate.wait_arrivals(1))
        pipeline = bridge.archive.pipeline
        lineage = pipeline._lineages["s"]
        self.assertTrue(poll(lambda: [entry.stage for entry in list(lineage.entries)[1:]] == ["stored", "stored"]))
        worker = threading.Thread(target=lambda: bridge.controller.rewind("s", refs["B"]), daemon=True)
        worker.start()
        self.assertTrue(poll(lambda: lineage.detaching))  # the detach waits for C's transaction
        gate.release()
        worker.join(TIMEOUT)
        self.assertFalse(worker.is_alive())
        state = bridge.controller.get("s")
        [branch] = state.detached
        self.assertEqual([item["node_id"] for item in branch["items"]], ["c000003", "c000004"])  # C only published
        self.assertEqual(state.last_switch["detached_at_switch"], 2)
        # R2: C, published while the detach waited, keeps the tree holder after the head left its branch
        self.assertIn("s" + TREE_HOLDER_SUFFIX, self.raw_holders(bridge, bridge.controller.tree("s")[3]))
        self.assertEqual(pipeline._detached_by_origin, {"s": {branch["branch"]}})  # R5 bookkeeping
        bridge.checkpoint("s")
        tree = bridge.controller.tree("s")
        self.assertEqual(sorted(tree), [1, 2, 3, 4, 5])
        self.assertEqual(bridge.archive.head("s"), refs["B"])  # D and E never moved the session head
        self.assertEqual(branch["cut"], tree[3])
        self.assertEqual(pipeline._detached_by_origin, {})
        self.assert_no_detached_leftovers(bridge)

    def test_r1_the_detach_starts_no_thread_after_its_commit_point(self):
        bridge = self.detach_bridge(encode_workers=4)
        gate = Gate(bridge, stage="publish")
        refs = self.published_ab(bridge)
        gate.hold("c000002")  # C publishes (held inside its transaction); D and E are stored behind it
        summary = run_tree(bridge, self.plan_request("ABCDE", "resume", index=5))[0]
        self.assertTrue(gate.wait_arrivals(1))
        pipeline = bridge.archive.pipeline
        lineage = pipeline._lineages["s"]
        self.assertTrue(poll(lambda: [entry.stage for entry in list(lineage.entries)[1:]] == ["stored", "stored"]))
        original_start, active, refused = threading.Thread.start, {"thread": None}, []

        def start(thread):  # after the commit point no thread can start on the detaching thread
            if active["thread"] == threading.get_ident():
                refused.append(thread.name)
                raise RuntimeError("can't start new thread (injected)")
            return original_start(thread)

        def hook(step):
            if step == "transfer":  # after the commit point
                active["thread"] = threading.get_ident()
        pipeline.detach_hook = hook
        real_detach = pipeline.detach

        def detach(*args):
            try:
                return real_detach(*args)
            finally:
                active["thread"] = None
        pipeline.detach = detach
        with mock.patch.object(threading.Thread, "start", start):
            worker = threading.Thread(target=lambda: bridge.controller.rewind("s", refs["B"]), daemon=True)
            original_start(worker)
            self.assertTrue(poll(lambda: lineage.detaching))
            gate.release()
            worker.join(TIMEOUT)
        self.assertFalse(worker.is_alive())
        state = bridge.controller.get("s")
        self.assertEqual((refused, state.last_switch["switch_fallback"], state.last_switch["detached_at_switch"]),
                         ([], False, 2))
        published = self.refs_of(bridge, summary)  # D and E publish on the detached branch (no stuck publisher)
        bridge.checkpoint("s")
        self.assertEqual(sorted(bridge.controller.tree("s")), [1, 2, 3, 4, 5])
        self.assertEqual(published[-1], bridge.controller.tree("s")[5])
        self.assert_no_detached_leftovers(bridge)

    def test_r5_drain_mode_admission_never_scans_other_lineages(self):
        bridge = self.bridge("r5", raw_tier=False)  # drain mode; without the raw tier every submit checks C3
        pipeline, inside = bridge.archive.pipeline, threading.local()

        class ScanSpy(dict):
            scans = 0

            def values(self):
                if getattr(inside, "flag", False):
                    ScanSpy.scans += 1
                return super().values()

            def items(self):
                if getattr(inside, "flag", False):
                    ScanSpy.scans += 1
                return super().items()

            def __iter__(self):
                if getattr(inside, "flag", False):
                    ScanSpy.scans += 1
                return super().__iter__()
        pipeline._lineages = ScanSpy(pipeline._lineages)
        real, calls = pipeline._unpublished_of, []

        def unpublished_of(lineage):
            inside.flag = True
            calls.append(lineage.branch_id)
            try:
                return real(lineage)
            finally:
                inside.flag = False
        pipeline._unpublished_of = unpublished_of
        self.run_plan(bridge)
        self.assertGreater(len(calls), 0)
        self.assertEqual((ScanSpy.scans, pipeline._detached_by_origin), (0, {}))
        self.assertEqual(sorted(bridge.controller.tree("s")), [1, 2, 3, 4, 5, 6])

    def test_f8_commit_reports_of_detached_commits_keep_their_timeline(self):
        bridge, gate, refs, ticket = self.detached_backlog("f8")
        branch = bridge.controller.get("s").detached[0]["branch"]
        self.assertEqual(ticket.publication_branch_id, branch)
        gate.release()
        [report] = [item for item in bridge.commit_reports(refs["request_c"], wait=True, timeout=TIMEOUT)
                    if item.get("kind") == "chunk"]
        self.assertEqual(report["node"], ticket.result(0))
        self.assertIsNotNone(report["timeline"])
        self.assertEqual((report["timeline"]["branch"], report["timeline"]["node_id"]), (branch, "c000002"))


class LineageRecacheDetachTests(FullyDetachedRecacheChecks, DetachCase):
    """The lineage archive (``svd_hevc``) with recache epochs: F2 (one epoch node per commit)."""

    representation = "svd_hevc"
    recache = True
    exact = False

    def test_f2_a_published_detached_recache_stays_rooted_when_its_chunk_fails(self):
        bridge = self.detach_bridge()
        gate = Gate(bridge)
        refs = self.published_ab(bridge)
        gate.hold_all()
        gate.fail("c000002")  # the epoch publishes, its chunk C fails
        run_tree(bridge, self.plan_request("ABC", "resume", index=3))
        self.assertTrue(gate.wait_arrivals(1))
        bridge.controller.rewind("s", refs["A"])
        state = bridge.controller.get("s")
        [branch] = state.detached
        self.assertEqual([item["kind"] for item in branch["items"]], ["recache", "chunk"])
        epoch_ticket = branch["items"][0]["ticket"]
        gate.release()
        bridge.checkpoint("s")
        epoch = epoch_ticket.result(0)
        self.assertEqual(bridge.archive.load(epoch).parent_ref, refs["B"])
        self.assertEqual(sorted(event["node_id"] for event in state.detach_failures.values()), ["c000002"])
        keep = detached_keep_ref("s", branch["branch"])
        self.assertEqual(bridge.archive.store.list_refs(DETACHED_KEEP_PREFIX), {keep: epoch})
        self.assertNotIn(keep, bridge.archive.sessions())
        bridge.collect_garbage()
        self.assertEqual(stored(bridge, {epoch}), {epoch})  # the clean epoch survives collection
        bridge.controller.close_session("s", delete=True)
        self.assertEqual(bridge.archive.store.list_refs(DETACHED_KEEP_PREFIX), {})
        bridge.collect_garbage()
        self.assertEqual(stored(bridge, {epoch, refs["A"], refs["B"]}), set())


class SvdDetachTests(DetachCase):
    """A fixed-rank arm (``svd``, ``reuse=False``): A-F with a held encoder ends with drain mode's tree structure."""

    representation = "svd"
    exact = False

    def bridge(self, store="archive", profile=None, **overrides):
        return super().bridge(store, profile=profile, **{"reuse": False, **overrides})

    def test_b_a_to_f_on_the_fixed_arm(self):
        drained = self.bridge("drain")
        self.run_plan(drained)
        bridge = self.detach_bridge()
        refs, results = self.run_plan(bridge, gate=Gate(bridge))
        self.assertEqual(results["E"][0]["restore"]["switch"]["detached_at_switch"], 1)
        self.assertEqual(self.structure(bridge), self.structure(drained))
        self.assert_no_detached_leftovers(bridge)


class DrainRecordsTests(DetachCase):
    """F7: drain mode (the default, and ``tree_switch`` spelled ``drain``) writes W1.2's records, field for field and
    value for value: the A-F runner plan (durable requests: no timing-dependent publication state) in a fresh process
    on this tree and on the frozen W1.2 tree. Compared after the same normalization on both sides -- floats (timings)
    and the store root are masked -- with declared exceptions: ``pressure.kv.new_chunks`` (P9: the planned new chunks,
    W1.2 counted every remaining block) and the values that vary between runs of the same tree with thread timing
    (``TIMING_DEPENDENT``: the fitter's timing histograms, whether an overlapped restore's layer was ready when the
    forward reached it); their fields are still compared. W1.4.3: a window's garbage collections are compared summed
    over generations (``GC_GENERATION``)."""

    def records(self, tree, representation, tag, extra=None):
        root, out = (self.workspace / tag).resolve(), (self.workspace / f"{tag}.json").resolve()
        spec = {"root": str(root), "representation": representation, "extra": extra or {}, "out": str(out)}
        proc = subprocess.run([sys.executable, "-W", "ignore", "-c", F7_SCENARIO, json.dumps(spec)], cwd=tree,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                              capture_output=True, text=True, timeout=1800)
        self.assertEqual(proc.returncode, 0, proc.stderr[-4000:])
        flat = {}
        _flatten(json.loads(out.read_text()), "", flat)
        return {path: ("<float>" if isinstance(value, float) else
                       value.replace(str(root), "<root>") if isinstance(value, str) else value)
                for path, value in flat.items()}

    def test_drain_records_equal_w12(self):
        tree = w12_tree()
        self.assertTrue(os.path.isdir(os.path.join(tree, "reelcache")), f"the W1.2 tree is missing: {tree}")
        for representation, variants in (("raw_offload", {"default": None, "drain": {"tree_switch": "drain"}}),
                                         ("svd_hevc", {"default": None})):
            w12 = self.records(tree, representation, f"w12-{representation}")
            for name, extra in variants.items():
                with self.subTest(representation=representation, variant=name):
                    w13 = self.records(ROOT, representation, f"w13-{name}-{representation}", extra)
                    self.assertEqual(sorted(set(w13) - set(w12)), [])  # no new field anywhere
                    self.assertEqual(sorted(set(w12) - set(w13)), [])  # none missing
                    new_chunks = {path for path in w13 if path.endswith(".pressure.kv.new_chunks")}
                    self.assertEqual(len(new_chunks), 12)  # finish and summary records of six requests
                    self.assertEqual({w13[path] for path in new_chunks}, {1})  # P9: one planned chunk each
                    timing = {path for path in w13 if TIMING_DEPENDENT.search(path)}
                    gc_phase = {path for path in w13 if GC_GENERATION.search(path)}
                    differing = sorted(path for path in w12 if w12[path] != w13[path]
                                       and path not in new_chunks | timing | gc_phase)
                    self.assertEqual(differing, [])
                    self.assertEqual(_gc_totals(w13), _gc_totals(w12))  # the collections in each window, summed
                    self.assertTrue(any(".restore.switch.drain_seconds" in path for path in w13))


#: F7: record values that vary between runs of one tree with thread timing (their fields are still compared).
TIMING_DEPENDENT = re.compile(r"\.step_spans\.[^.]+\.buckets\[\d+\]$|\.overlap\.waited_before_ready$"
                              r"|\.overlap\.waits\[\d+\]\.ready_at_wait$")
#: W1.4.3: a window's garbage collections per generation (``restore.gc.collections[g]``). How many collections a
#: window triggers is its own allocations'; which generation each one collects depends on the process's whole
#: allocation history (environment, code imported earlier): run from a /tmp copy on an interactive node, the unmodified
#: W1.4.1 tree records [0, 1, 0] in request B's restore window where W1.2 records [1, 0, 0] (its receipt run passed).
#: Compared summed per window (each threshold event is exactly one collection, whatever its generation).
GC_GENERATION = re.compile(r"\.gc\.collections\[\d+\]$")


def _gc_totals(flat: dict) -> dict:
    totals: dict = {}
    for path, value in flat.items():
        if GC_GENERATION.search(path):
            window = path.rsplit("[", 1)[0]
            totals[window] = totals.get(window, 0) + value
    return totals


def poll(predicate):
    """Wait for a state the code under test signals no event for (a condition, not a duration)."""
    deadline = time.monotonic() + TIMEOUT
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.001)
    return True


def _flatten(value, path, out):
    if isinstance(value, dict):
        if not value:
            out[path + "{}"] = "{}"
        for key in value:
            _flatten(value[key], f"{path}.{key}", out)
    elif isinstance(value, list):
        if not value:
            out[path + "[]"] = "[]"
        for index, item in enumerate(value):
            _flatten(item, f"{path}[{index}]", out)
    else:
        out[path] = value


#: F7: the A-F runner plan (durable) on the tree in the working directory; dumps every record a run reports. The toy
#: stage driver is defined here (``tests.test_wl_tree.run_tree`` with the N5 read fix), identical on both trees: only
#: the bridge under comparison differs (the W1.2 tree's own driver reads layer 0 before an overlapped restore lands).
F7_SCENARIO = r"""
import json, sys
from types import SimpleNamespace
import torch
import tests.test_wl_tree as t
from reelcache.integrations.sglang import CausalWanKVBridge, ReelCacheRuntimeConfig


def run_tree(bridge, req):
    caches = {stream: [t.GuardedCache() for _ in range(t.LAYERS)] for stream in req.streams}
    latents = torch.zeros(1, 4, sum(req.block_sizes), *t.GRID)
    generator = torch.Generator().manual_seed(req.conditioning["seed"])
    torch.randn(3, generator=generator)
    run = bridge.begin_request(req)
    target = SimpleNamespace(caches=caches, latents=latents, generator=generator, rope_cos_sin=t.rope_cos_sin,
                             rotate_key=t.rotate_key, device=torch.device("cpu"))
    try:
        resume = run.restore(target)
        run.block_prompts = list(req.block_prompts)
        if getattr(run, "replay_plan", None) is not None:
            t.replay(run, req, caches)
        start = sum(req.block_sizes[:resume])
        for index in range(resume, len(req.block_sizes)):
            frames, prompt = req.block_sizes[index], req.block_prompts[index]
            chunk = t.Chunk(index, start, frames, start * t.TPF, frames * t.TPF, float(start), False, 0)
            reference = caches["cond"][0]
            reference.await_restored_history()
            history = reference.k[:, :reference.end].float().mean() if reference.end else 0.0
            noise = torch.randn(1, 4, frames, *t.GRID, generator=generator)
            latents[:, :, start:start + frames] = noise + float(history)
            t.toy_forward(caches, req.streams, prompt, chunk, run, "clean_commit")
            start += frames
            if run.end_chunk(chunk, latents[:, :, start - frames:start].clone()):
                break
        return run.finish(error=None)
    except BaseException as exc:
        run.finish(error=exc)
        raise


spec = json.loads(sys.argv[1])
values = {"store_root": spec["root"], "representation": spec["representation"], "fit_device": "cpu",
          "decoder": "software", "energy_key": 0.98, "energy_value": 0.95, "x265_threads": 2, "encode_workers": 2,
          "tree_retention": "full", "host_raw_budget_bytes": 1 << 30, **spec["extra"]}
bridge = CausalWanKVBridge(ReelCacheRuntimeConfig.from_dict(values), t.PROFILE, device="cpu", model_path="toy/causal-wan")
out, refs = {"requests": {}}, {}
for letter, action, target, path, index in t.SESSION_MAX_STEPS:
    extra = {} if target is None else {"checkpoint": refs[target]}
    session = t.handle("s", action, pause_after_chunks=len(path), durable=True, **extra)
    prompts = [t.PROMPTS[item] for item in path + path[-1] * (t.SESSION_TOTAL - len(path))]
    req = t.tree_request(session, prompts, generation_index=index)
    req.request_id = "f7-" + letter
    summary = run_tree(bridge, req)
    refs[letter] = bridge.checkpoint("s")["head"]
    out["requests"][letter] = {"finish": summary, "summary": bridge.request_summary(req.request_id),
                               "workload": bridge.workload_record(req.request_id),
                               "commits": bridge.commit_reports(req.request_id, wait=True, timeout=300)}
out["describe"] = bridge.describe()
out["archive_stats"] = bridge.archive.stats()
out["sessions"] = bridge.controller.sessions()
out["tree"] = bridge.controller.tree("s")
out["refs"] = bridge.archive.store.list_refs("")
bridge.close()
with open(spec["out"], "w") as handle:
    json.dump(out, handle, default=repr, sort_keys=True)
"""


#: A tree worker SIGKILLed while a detached branch is in flight: D's archival never finishes; E (the new branch) is
#: queued behind it. The rewind's head move (to B) is durable; nothing of D or E is.
DETACH_CRASH_CHILD = r"""
import json, os, sys, time
import tests.test_wl_tree as t
from reelcache.integrations.sglang import CausalWanKVBridge, ReelCacheRuntimeConfig
from reelcache.integrations.sglang.nodes import DETACHED_REF_PREFIX, is_detached_branch

spec = json.loads(sys.argv[1])
values = {"store_root": spec["root"], "representation": spec["representation"], "fit_device": "cpu",
          "decoder": "software", "energy_key": 0.98, "energy_value": 0.95, "x265_threads": 2, "encode_workers": 2,
          "tree_retention": "full", "tree_switch": "detach"}
bridge = CausalWanKVBridge(ReelCacheRuntimeConfig.from_dict(values), t.PROFILE, device="cpu", model_path="toy/causal-wan")
archive = bridge.archive
offload = getattr(archive, "write_behind", False)


def request(path, action, index, checkpoint=None):
    extra = {} if checkpoint is None else {"checkpoint": checkpoint}
    session = t.handle("s", action, pause_after_chunks=len(path), durable=False, **extra)
    prompts = [t.PROMPTS[letter] for letter in path + path[-1] * (t.SESSION_TOTAL - len(path))]
    return t.tree_request(session, prompts, generation_index=index)


refs = {}
for letter, action, path, index in (("A", "generate", "A", 1), ("B", "resume", "AB", 2), ("C", "resume", "ABC", 3)):
    summary = t.run_tree(bridge, request(path, action, index))[0]
    reports = bridge.commit_reports(summary["request_id"], wait=True, timeout=300)
    refs[letter] = [item["node"] for item in reports if item.get("kind") == "chunk"][-1]
if offload:  # archival stalls from here: D (and E) never finish
    archive.write_hook = lambda stage, job: time.sleep(3600)
else:
    archive.pipeline.hooks = lambda stage, node_id: time.sleep(3600) if stage == "encode" else None
t.run_tree(bridge, request("ABCD", "resume", 4))
t.run_tree(bridge, request("ABE", "rewind", 5, checkpoint=refs["B"]))  # D detached; E behind it
state = bridge.controller.get("s")
detached = (sorted(name[len(DETACHED_REF_PREFIX):] for name in archive.store.list_refs(DETACHED_REF_PREFIX)) if offload
            else sorted(name for name in archive.runtime.list_lineages() if is_detached_branch(name)))
result = {"refs": refs, "detached_refs": detached, "sessions": sorted(archive.sessions()),
          "branches": [item["branch"] for item in state.detached]}
with open(spec["result"] + ".tmp", "w") as handle:
    json.dump(result, handle)
os.replace(spec["result"] + ".tmp", spec["result"])
time.sleep(3600)
"""


if __name__ == "__main__":
    unittest.main()
