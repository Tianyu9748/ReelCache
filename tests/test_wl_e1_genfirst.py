"""W1.4.3 (E1, archive eviction) x W1.4.2.2 (generation-first archival): the ablation arms run E1 with
``defer_archival_during_generation=True`` and ``archival_defer_inflight="overlap"``. CPU toy harness (svd_hevc / hevc,
``reuse=False``, the software decoder).

Where the two meet: a generation hold keeps the fitter from STARTING a node; encodes, publications, the raw tier's
copies and evictions are never held. An archive-budget reservation is taken when a commit WRITES -- its basis rows in
the fitter's version store, its packet payloads before the encode worker stores them -- so a commit the hold keeps
queued holds none, and a commit holding one always progresses: beside the hold (overlap) or through it (the starvation
guard's overrides).

* (a) a commit held back by a hold reserves nothing; let through (the age bound) it writes and keeps its reservation
  until it publishes; a write that must wait inside a hold for another session's commit to publish waits for it (that
  commit can still publish: no false "below the commits in flight"), and everything publishes once the hold closes;
* (b) an eviction inside a hold spares pinned nodes, leased nodes and pending commits -- their reservations and the
  shared identity basis a pending HEVC-only commit references (all or nothing per node);
* (c) a miss inside a request (``_restore_evicted``: the deepest available ancestor installed, the rest replayed from
  the stored latents, the replayed node promoted into host raw) opens the request's hold at the replay's end; the
  restore's availability pins end with the anchor's install, so another session's commit waiting for budget room
  writes before the replay starts -- no deadlock between the replay's promotions and pending commits' reservations;
* (d) overlap: the fit on the GPU (here the CPU fitter) when a hold opens keeps its reservation, completes beside the
  request and publishes inside the hold; the accounting (live bytes = the live nodes' payloads, nothing reserved at
  rest, the peak within the budget) stays consistent.

Each scenario orders its events with the pipeline's stage hooks (gates) and fails within bounded waits instead of
hanging. Sessions other than ``s`` generate other prompts: the toy KV depends on the prompt only, so the same prompt
would write the same content-addressed objects (deduplicated, review round 2) and even the same nodes. Negative
controls (one mutation each, every named test must fail): ``dev_w1b/scratch_w13/w143_negctl.py`` (the ``gf_*``
controls).
"""

from __future__ import annotations

import contextlib
import threading
import time
from types import SimpleNamespace
from unittest import mock

import torch

from reelcache.integrations.sglang import bridge as bridge_module
from reelcache.lineage_commit import identity_object
from tests.test_sglang_bridge import CHANNELS, handle
from tests.test_wl_archive_eviction import HUGE, EvictionCase
from tests.test_wl_tree import assert_caches_bitwise, run_tree

GENERATION_FIRST = {"defer_archival_during_generation": True, "archival_defer_inflight": "overlap"}
WAIT = 30.0  # a bounded wait for an event the scenario guarantees


def until(predicate, timeout: float = WAIT, step: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(step)
    return True


class StageGates:
    """The archival pipeline's stage hook, with the entry: which ``(stage, session, node)`` were reached, and a gated
    one held until the test opens it (stages: plan, finalize -- the basis rows are written --, encode, store -- the
    payloads are written next --, publish)."""

    def __init__(self, pipeline):
        self._lock = threading.Lock()
        self.reached: dict = {}
        self.gates: dict = {}
        original = pipeline._hook

        def hook(stage, entry):
            key = (stage, entry.lineage.branch_id, entry.node_id)
            with self._lock:
                event = self.reached.setdefault(key, threading.Event())
                gate = self.gates.get(key)
            event.set()
            if gate is not None and not gate.wait(120):
                raise TimeoutError(f"stage gate {key} was never opened")
            return original(stage, entry)
        pipeline._hook = hook

    def gate(self, stage: str, session: str, node: str = "c000000") -> threading.Event:
        with self._lock:
            return self.gates.setdefault((stage, session, node), threading.Event())

    def open_all(self) -> None:
        with self._lock:
            gates = list(self.gates.values())
        for gate in gates:
            gate.set()

    def was_reached(self, stage: str, session: str, node: str = "c000000") -> bool:
        with self._lock:
            event = self.reached.get((stage, session, node))
        return event is not None and event.is_set()

    def wait(self, stage: str, session: str, node: str = "c000000", timeout: float = WAIT) -> bool:
        return until(lambda: self.was_reached(stage, session, node), timeout)


class GenerationFirstEvictionCase(EvictionCase):
    def gf(self, store="gf", budget=HUGE, **overrides):
        bridge = self.evicting(store, budget, **GENERATION_FIRST, **overrides)
        bridge.archive.eviction.wait_seconds = WAIT  # a wait the scenario does not end fails within the test
        return bridge

    @contextlib.contextmanager
    def during(self, callback):
        """Run ``callback(run)`` inside the next request's generation -- its hold open -- right after its first
        chunk's commit was submitted (``end_chunk``)."""
        real = bridge_module._BridgeRun.end_chunk
        fired = []

        def end_chunk(run, *args, **kwargs):
            result = real(run, *args, **kwargs)
            if not fired:
                fired.append(True)
                callback(run)
            return result
        with mock.patch.object(bridge_module._BridgeRun, "end_chunk", end_chunk):
            yield fired
        self.assertTrue(fired, "the request never reached its generation")

    @staticmethod
    def holds_open(bridge) -> int:
        pipeline = bridge.archive.pipeline
        with pipeline._cond:
            return 0 if pipeline._generation is None else len(pipeline._generation.tokens)

    @staticmethod
    def set_budget(bridge, budget: int) -> None:
        """The budget for the next phase; the peak restarts with it (as the E1 smoke's calibration)."""
        eviction = bridge.archive.eviction
        with eviction._cond:
            eviction.budget = int(budget)
            eviction.peak_bytes = eviction.live_bytes + eviction.reserved_bytes
            eviction._cond.notify_all()

    def assert_accounting(self, bridge) -> None:
        """At rest: live bytes = the live nodes' payload objects (each once) plus orphans, nothing reserved, the peak
        within the budget, every live payload in the store, no bookkeeping error."""
        eviction, store = bridge.archive.eviction, bridge.archive.store
        with eviction._cond:
            objects = {object_id for payload in eviction._nodes.values() for object_id in payload.objects}
            orphans = sum(eviction._orphans.values())
            live, reserved, peak, budget = (eviction.live_bytes, eviction.reserved_bytes, eviction.peak_bytes,
                                            eviction.budget)
            held = dict(eviction._reserved)
        sizes = store.object_sizes(sorted(objects)) if objects else {}
        self.assertEqual(live, sum(sizes.values()) + orphans)
        self.assertEqual((reserved, held), (0, {}))
        self.assertLessEqual(peak, budget)
        self.assertEqual(store.evicted_ids(objects), set())
        self.assertEqual(eviction.stats()["errors"], 0)


class E1GenerationFirstTests(GenerationFirstEvictionCase):
    def test_a_held_commit_reserves_nothing_and_a_wait_inside_a_hold_never_fails_fast(self):
        """(a) t's chunk, submitted inside its request's hold (a worker's first request keeps it until the next one),
        is held back: not started, nothing reserved. The age bound lets it through inside the hold: it writes and
        keeps its whole node's reservation until it publishes (gated here). s's chunk, let through inside s's hold,
        must wait for room that only t's publication makes: t's commit holds a reservation, is not waiting and is of
        another session -- it can still publish, so s's write waits (no fail-fast), then evicts t's node (least
        recently used) and publishes inside the hold; once the holds close every commit publishes."""
        bridge = self.gf(archival_defer_max_age_seconds=1.0)
        eviction, pipeline = bridge.archive.eviction, bridge.archive.pipeline
        gates = StageGates(pipeline)
        publish_t = gates.gate("publish", "t")
        outcome: dict = {}
        try:
            run_tree(bridge, self.request_for(handle("t"), "E"))
            time.sleep(0.3)  # below the age bound (1 s)
            outcome["held"] = (self.holds_open(bridge), gates.was_reached("plan", "t"), eviction.reserved_bytes,
                               eviction.stats()["reservations"])
            self.assertEqual(outcome["held"], (1, False, 0, 0))
            self.assertTrue(gates.wait("publish", "t"))  # let through by age, written, its reservation held
            self.assertEqual(self.holds_open(bridge), 1)
            node_t = eviction.reserved_bytes
            self.assertGreater(node_t, 0)
            self.assertEqual(pipeline.generation_overrides()["age"], 1)
            self.set_budget(bridge, int(1.25 * node_t))  # another node's rows fit only once t's node is evicted

            def inside(run):  # s's request generates (its hold open); its chunk was just submitted
                outcome["holds"] = self.holds_open(bridge)
                ticket = run.tickets[-1]["ticket"]
                # let through by age; its basis rows wait for the room t's reserved node holds
                outcome["waited"] = until(lambda: eviction.stats()["reserve_waits"] >= 1 or ticket.done())
                time.sleep(0.3)
                outcome["failed_early"] = ticket.done()
                publish_t.set()  # t publishes; s's write evicts t's node and goes on
                outcome["published"] = until(ticket.done) and ticket.error is None
                outcome["still_inside"] = self.holds_open(bridge)
            with self.during(inside):
                summary = run_tree(bridge, self.request_for(handle("s"), "A"))[0]
        finally:
            gates.open_all()
        self.assertEqual({key: outcome.get(key) for key in ("holds", "waited", "failed_early", "published",
                                                             "still_inside")},
                         {"holds": 1, "waited": True, "failed_early": False, "published": True, "still_inside": 1})
        self.assertGreaterEqual(summary["archival_defer"]["overrides"]["age"], 1)
        stats = eviction.stats()
        self.assertEqual((stats["evict"]["nodes"], stats["live_nodes"], stats["evicted_nodes"]), (1, 1, 1))
        self.assertGreaterEqual(stats["reserve_waits"], 1)
        # the holds closed: nothing is left behind (the next commit publishes, no deadlock)
        run_tree(bridge, self.request_for(handle("s", "rewind", checkpoint=bridge.checkpoint("s")["head"]), "AB"))
        for session in ("s", "t"):
            bridge.checkpoint(session)
        self.assertEqual(self.holds_open(bridge), 0)
        self.assert_accounting(bridge)

    def test_b_an_eviction_inside_a_hold_spares_pinned_leased_and_pending_nodes(self):
        """(b) Inside a hold (u's request generating) a write needs one node's room: the least recently used node A is
        pinned (a restore reads it) and B has a leased payload object, so C goes -- whole -- while D stays; t's
        pending commit (written, its publication gated) keeps its reservation and publishes after it."""
        bridge = self.gf()
        eviction, store, pipeline = bridge.archive.eviction, bridge.archive.store, bridge.archive.pipeline
        nodes = self.chain(bridge, "ABCD")  # published in order: least recently used first A, B, C, D
        gates = StageGates(pipeline)
        gates.gate("publish", "t")
        outcome: dict = {}
        try:
            run_tree(bridge, self.request_for(handle("t"), "E"))
            self.assertTrue(gates.wait("publish", "t"))  # written after its request's hold, not published
            pending = eviction.reserved_bytes
            self.assertGreater(pending, 0)
            payload = {letter: eviction.node_payload(nodes[letter]).objects for letter in "ABCD"}
            size_c = sum(store.object_sizes(sorted(payload["C"])).values())

            def inside(run):
                outcome["holds"] = self.holds_open(bridge)
                token = eviction.pin_available([nodes["A"]], touch=False)
                lease = store.lease([sorted(payload["B"])[0]], verify=False)
                key = SimpleNamespace(budget_order=("writer", 1))
                try:
                    self.set_budget(bridge, eviction.live_bytes + eviction.reserved_bytes + 1)
                    eviction.reserve(key, size_c)
                    outcome["evicted"] = sorted(letter for letter in "ABCD" if eviction.is_evicted(nodes[letter]))
                    outcome["gone"] = {letter: bool(store.evicted_ids(payload[letter])) for letter in "ABCD"}
                    outcome["skipped_leased"] = eviction.stats()["evict"]["skipped_leased"]
                    outcome["pending_kept"] = eviction.reserved_bytes == pending + size_c
                finally:
                    eviction.release(key)
                    eviction.unpin(token)
                    lease.release()
                    self.set_budget(bridge, HUGE)
            with self.during(inside):
                run_tree(bridge, self.request_for(handle("u"), "G"))
        finally:
            gates.open_all()
        self.assertEqual(outcome, {"holds": 1, "evicted": ["C"],
                                   "gone": {"A": False, "B": False, "C": True, "D": False},
                                   "skipped_leased": 1, "pending_kept": True})
        node_t = bridge.checkpoint("t")["head"]
        bridge.checkpoint("u")
        self.assertFalse(eviction.is_evicted(node_t))
        self.assert_accounting(bridge)

    def test_b_a_pending_commits_shared_basis_survives_an_eviction_inside_a_hold(self):
        """(b, HEVC-only arm) t's pending commit references the identity basis A alone holds. Inside a hold a write
        needs A's room: A cannot go whole while t's staging leases reach the basis, so the write waits for t (it can
        still publish: no fail-fast); once t publishes and holds the basis too, A goes without it."""
        self.hevc_geometry()
        bridge = self.gf(representation="hevc")
        eviction, store, pipeline = bridge.archive.eviction, bridge.archive.store, bridge.archive.pipeline
        run_tree(bridge, self.request_for(handle("s"), "A"))
        node_a = bridge.checkpoint("s")["head"]
        identity = identity_object(CHANNELS)[0]
        self.assertIn(identity, eviction.node_payload(node_a).objects)
        own_a = sorted(eviction.node_payload(node_a).objects - {identity})
        need = sum(store.object_sizes(own_a).values())  # A's own objects: what A frees once t holds the basis too
        gates = StageGates(pipeline)
        publish_t = gates.gate("publish", "t")
        outcome: dict = {}
        try:
            run_tree(bridge, self.request_for(handle("t"), "E"))
            self.assertTrue(gates.wait("publish", "t"))

            def inside(run):
                outcome["holds"] = self.holds_open(bridge)
                key, done, errors = SimpleNamespace(budget_order=("writer", 1)), threading.Event(), []

                def write():
                    try:
                        eviction.reserve(key, need)
                    except BaseException as exc:  # noqa: BLE001 - recorded
                        errors.append(exc)
                    finally:
                        done.set()
                self.set_budget(bridge, eviction.live_bytes + eviction.reserved_bytes + 1)
                writer = threading.Thread(target=write, daemon=True)
                writer.start()
                try:
                    time.sleep(0.5)
                    outcome["waits_for_t"] = not done.is_set()
                    outcome["kept_while_pending"] = (not eviction.is_evicted(node_a)
                                                     and not store.evicted_ids([identity]))
                    outcome["skipped"] = eviction.stats()["evict"]["skipped_leased"] >= 1
                    publish_t.set()
                    outcome["written"] = done.wait(WAIT) and not errors
                    outcome["a_evicted"] = eviction.is_evicted(node_a)
                    outcome["basis_kept"] = not store.evicted_ids([identity])
                finally:
                    publish_t.set()
                    done.wait(WAIT)
                    eviction.release(key)
                    self.set_budget(bridge, HUGE)
            with self.during(inside):
                run_tree(bridge, self.request_for(handle("u"), "G"))
        finally:
            gates.open_all()
        self.assertEqual(outcome, {"holds": 1, "waits_for_t": True, "kept_while_pending": True, "skipped": True,
                                   "written": True, "a_evicted": True, "basis_kept": True})
        node_t = bridge.checkpoint("t")["head"]
        bridge.checkpoint("u")
        self.assertIn(identity, eviction.node_payload(node_t).objects)
        self.assertFalse(eviction.is_evicted(node_t))
        self.assert_accounting(bridge)

    def test_c_a_miss_opens_its_hold_at_the_replay_end_and_frees_pending_writers_first(self):
        """(c) s's request misses (D evicted, no raw copy): it pins its path's archived nodes A..C, installs the anchor
        C from the raw tiers, replays D from its stored latents (promoted into host raw) and only then opens its
        hold; F is generated inside it. t's commit plans while s holds the pins: its rows need a node's room, so it
        waits -- and writes once the anchor's install ends the pins, before s's replay begins (pins kept through the
        replay would hold it until s finishes). No deadlock between the replay's promotions and pending commits'
        reservations; the miss is bitwise a clean full replay (exact raw anchor)."""
        bridge = self.gf(raw_tier=True, gpu_raw_budget_bytes=1 << 30, host_raw_budget_bytes=1 << 30,
                         raw_tier_retain="lru", raw_tier_promote="host")
        eviction, pipeline, tier = bridge.archive.eviction, bridge.archive.pipeline, bridge.archive.raw_tier
        nodes = self.chain(bridge, "ABCD")
        tier.flush()
        tier.drop(tier.entries(node_refs=[nodes["D"]]))
        self.evict(bridge, nodes["D"])
        gates = StageGates(pipeline)
        plan_t = gates.gate("plan", "t")
        outcome: dict = {}
        try:
            run_tree(bridge, self.request_for(handle("t"), "E"))
            self.assertTrue(gates.wait("plan", "t"))  # t's commit on the fitter, before its writes
            smallest = min(self.node_bytes(bridge, nodes[letter]) for letter in "ABC")
            self.set_budget(bridge, eviction.live_bytes + smallest // 4)  # t's rows need a node of s's path to go
            real_hold = bridge.archive.hold_available

            def hold_available(refs):
                held = real_hold(refs)  # s's path pinned
                plan_t.set()
                outcome["t_waits_on_the_pins"] = until(lambda: eviction.stats()["reserve_waits"] >= 1, 10.0)
                return held
            real_begin_replay = bridge_module._BridgeRun.begin_replay

            def begin_replay(run, *args, **kwargs):
                outcome["holds_before_replay"] = self.holds_open(bridge)
                outcome["t_wrote_before_the_replay"] = gates.wait("finalize", "t", timeout=10.0)
                return real_begin_replay(run, *args, **kwargs)
            with mock.patch.object(bridge.archive, "hold_available", hold_available), \
                    mock.patch.object(bridge_module._BridgeRun, "begin_replay", begin_replay):
                summary, latents, caches, _ = run_tree(
                    bridge, self.request_for(handle("s", "rewind", checkpoint=nodes["D"]), "ABCDF"))
        finally:
            gates.open_all()
        self.assertEqual(outcome, {"t_waits_on_the_pins": True, "holds_before_replay": 0,
                                   "t_wrote_before_the_replay": True})
        restore = summary["restore"]
        self.assertEqual((restore["source"], restore["recompute"]["anchor"], restore["recompute"]["anchor_source"]),
                         ("replay", nodes["C"], "raw"))
        record = summary["archival_defer"]
        self.assertEqual((record["holds"], record["opened_at"]), (1, ["replay"]))
        fresh_latents, fresh_caches, _ = self.fresh("ABCDF")
        self.assertTrue(torch.equal(latents, fresh_latents))
        assert_caches_bitwise(self, caches, fresh_caches)
        node_t = bridge.checkpoint("t")["head"]
        bridge.checkpoint("s")
        tier.flush()
        self.assertEqual({entry.node_ref: entry.exact for entry in tier.entries() if entry.origin == "replay"},
                         {nodes["D"]: True})  # the replayed node, promoted
        self.assertTrue(eviction.is_evicted(nodes["A"]))  # t's room: s's least recently used node, after the pins
        self.assertFalse(eviction.is_evicted(node_t))
        self.assertEqual(self.holds_open(bridge), 0)
        self.assert_accounting(bridge)

    def test_d_an_inflight_fit_completes_beside_the_request_and_publishes_inside_its_hold(self):
        """(d) overlap: s's chunk (held by the first request's hold) starts when t's request begins and is on the
        fitter, its basis rows written and reserved, when t's hold opens -- at once, not waiting for it. It completes
        beside t's generation and publishes inside the hold: live bytes = its node's payload, nothing reserved; t's
        own chunk stays held until the hold closes (reported as one in-flight overlap), then evicts s's node for its
        room within the budget."""
        probe = self.evicting("probe")
        size = max(self.node_bytes(probe, ref) for ref in self.chain(probe, "AB").values())
        bridge = self.gf(budget=int(1.25 * size))
        eviction, pipeline = bridge.archive.eviction, bridge.archive.pipeline
        gates = StageGates(pipeline)
        finalize_s = gates.gate("finalize", "s")
        outcome: dict = {}
        run_tree(bridge, self.request_for(handle("s"), "A"))
        self.assertFalse(gates.was_reached("plan", "s"))  # held by the first request's hold
        real_begin = bridge.begin_request

        def begin_request(request):
            run = real_begin(request)  # closes the stale hold: s's chunk starts
            outcome["inflight_at_open"] = gates.wait("finalize", "s")
            outcome["rows_reserved"] = eviction.reserved_bytes > 0
            return run

        def inside(run):
            outcome["holds"] = self.holds_open(bridge)
            outcome["t_held"] = not gates.was_reached("plan", "t")
            finalize_s.set()  # the in-flight fit completes beside the request ...
            outcome["published_inside"] = until(lambda: eviction.stats()["live_nodes"] == 1
                                                and eviction.reserved_bytes == 0, 10.0)
            outcome["still_inside"] = self.holds_open(bridge)
            outcome["t_still_held"] = not gates.was_reached("plan", "t")
            outcome["live"] = eviction.live_bytes
        result: dict = {}

        def request():
            try:
                with mock.patch.object(bridge, "begin_request", begin_request), self.during(inside):
                    result["summary"] = run_tree(bridge, self.request_for(handle("t"), "E"))[0]
            except BaseException as exc:  # noqa: BLE001 - asserted below
                result["error"] = exc
        thread = threading.Thread(target=request, daemon=True)
        thread.start()
        thread.join(20.0)
        finished = not thread.is_alive()
        gates.open_all()
        thread.join(WAIT)
        self.assertTrue(finished, "t's hold waited for the fit on the GPU (overlap opens at once)")
        self.assertNotIn("error", result)
        node_s = bridge.checkpoint("s")["head"]
        self.assertEqual(outcome, {"inflight_at_open": True, "rows_reserved": True, "holds": 1, "t_held": True,
                                   "published_inside": True, "still_inside": 1, "t_still_held": True,
                                   "live": self.node_bytes(bridge, node_s)})
        record = result["summary"]["archival_defer"]
        self.assertEqual((record["holds"], record["inflight_overlaps"], record["inflight_waits"]), (1, 1, 0))
        self.assertGreater(record["inflight_overlap_seconds"], 0.0)
        node_t = bridge.checkpoint("t")["head"]
        stats = eviction.stats()
        self.assertEqual((stats["evict"]["nodes"], stats["live_nodes"]), (1, 1))  # s's node made t's room
        self.assertTrue(eviction.is_evicted(node_s))
        self.assertFalse(eviction.is_evicted(node_t))
        self.assert_accounting(bridge)


if __name__ == "__main__":  # pragma: no cover
    import unittest

    unittest.main()
