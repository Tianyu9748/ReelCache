"""W1.1 (wl-w1a): the W1 telemetry review's spans on the CPU toy harness (no SGLang, no CUDA: host clocks).

* ``drain``: a fork's source drain and a rewind's (tree mode: switch) drain, in the controller step of
  ``begin_request``: ``prep``, inside ``begin.controller_seconds``, before the run's own ``publish_wait`` and restore
  window (no double counting: the spans ``kv_rebuild_exposed`` sums are disjoint);
* ``recompute_replay``: the recompute arm's replay (``run.begin_replay`` .. ``run.end_replay``, ``prep``), after its
  restore window (``restore_plan``, ``restore_sync`` / ``restore_sync.<raw tier>``) and ``exec_state``;
* the raw_offload tier vocabulary: ``restore_sync.nvme`` (``store_io="nocache"``), ``restore_sync.page_cache``
  (wl-w1b's ``store`` label, buffered reads), ``bytes.h2d`` of store reads; the restore records keep wl-w1b's labels;
* telemetry off: no record, no new summary key, the same outputs; nothing leaks between requests.
"""

import unittest

import torch

from reelcache.integrations.sglang import workload_telemetry as wt
from tests.test_sglang_bridge import handle
from tests.test_wl_raw_offload import OffloadCase
from tests.test_wl_recompute import PATHS, RecomputeTestCase
from tests.test_wl_tree import TreeTestCase, assert_caches_bitwise, run_tree

#: Restore-side spans whose sum is ``kv_rebuild_exposed`` (with the exposed stall), CONTRACT §3 (+ W1.2 session_open).
EXPOSED = ("publish_wait", "drain", "recompute_replay", "session_open")
SLACK_NS = 200_000  # perf_counter vs monotonic_ns reads around the same instant (same clock; call overhead only)


def named(record, name):
    return [span for span in record["spans"] if span["name"] == name]


def width(span):
    return span["host_end_ns"] - span["host_start_ns"]


def exposed_spans(record):
    return [span for span in record["spans"] if span["name"] in EXPOSED or span["name"].startswith("restore_sync")]


class DisjointMixin:
    def assertDisjoint(self, spans):
        ordered = sorted((span["host_start_ns"], span["host_end_ns"], span["name"]) for span in spans)
        for (_, end, left), (start, _, right) in zip(ordered, ordered[1:]):
            self.assertLessEqual(end, start, f"{left} overlaps {right}")


class DrainSpanTests(DisjointMixin, TreeTestCase):
    """Tree mode (``tree_retention="full"``), RAW archive: rewinds and forks right after the previous request."""

    def sequence(self, bridge):
        out = {}
        for name, session, letters in (("generate", handle("s"), "AB"),
                                       ("rewind", handle("s", "rewind", checkpoint=1), "AC"),
                                       ("fork", handle("f", "fork", from_session="s", checkpoint=1), "AD"),
                                       ("resume", handle("f", "resume"), "ADE")):
            req = self.request_for(session, letters)
            summary, latents, caches, rng = run_tree(bridge, req)  # no checkpoint between: the drain waits
            out[name] = (req, summary, latents, {s: [c.k[:, :c.end].clone() for c in v] for s, v in caches.items()})
            self.assertIsNone(wt._REQUEST.get())  # the controller scope never outlives begin_request
        return out

    def test_rewind_and_fork_drains_are_prep_spans_inside_the_controller_step(self):
        bridge = self.bridge("drain-on", workload_telemetry=True)
        out = self.sequence(bridge)
        for name, (req, summary, _, _) in out.items():
            record = bridge.workload_record(req.request_id)
            drains = named(record, "drain")
            if name in ("generate", "resume"):
                self.assertEqual(drains, [], name)
                continue
            self.assertEqual(len(drains), 1, name)
            drain = drains[0]
            self.assertEqual(drain["phase"], "prep")
            self.assertEqual(drain["stream"], "host")  # CPU: host clocks (CUDA: an event pair on the compute stream)
            # inside the controller step: never longer than begin.controller_seconds
            self.assertLessEqual(width(drain), summary["begin"]["controller_seconds"] * 1e9 + SLACK_NS, name)
            # before the run's own publish wait and restore window; the summed spans are disjoint
            for span in record["spans"]:
                if span["name"] in ("publish_wait", "restore_plan") or span["name"].startswith("restore_sync"):
                    self.assertLessEqual(drain["host_end_ns"], span["host_start_ns"], (name, span["name"]))
            self.assertDisjoint(exposed_spans(record))
            if name == "rewind":  # the switch drain: restore.switch.drain_seconds is the drain inside the span
                switch = summary["restore"]["switch"]
                self.assertTrue(switch["tree"])
                self.assertGreaterEqual(width(drain) + SLACK_NS, switch["drain_seconds"] * 1e9)

    def test_telemetry_off_is_unchanged(self):
        off_bridge = self.bridge("drain-off")
        on, off = self.sequence(self.bridge("drain-on", workload_telemetry=True)), self.sequence(off_bridge)
        for name in on:
            req, summary, latents, caches = off[name]
            self.assertIsNone(off_bridge.workload_record(req.request_id))
            self.assertNotIn("workload", summary)
            self.assertTrue(torch.equal(on[name][2], latents), name)
            for stream in caches:
                for a, b in zip(on[name][3][stream], caches[stream]):
                    self.assertTrue(torch.equal(a, b), name)
            self.assertEqual(sorted(set(on[name][1]) - {"workload"}), sorted(summary), name)

    def test_control_plane_rewind_records_nothing(self):
        bridge = self.bridge("drain-control", workload_telemetry=True)
        req = self.request_for(handle("s"), "AB")
        run_tree(bridge, req)
        bridge.control("rewind", {"session_id": "s", "checkpoint": 1})  # a control, not a request: no telemetry
        self.assertIsNone(wt._REQUEST.get())
        self.assertEqual(named(bridge.workload_record(req.request_id), "drain"), [])


class RecomputeSpanTests(DisjointMixin, RecomputeTestCase):
    def test_replay_is_a_prep_span_after_the_restore_window(self):
        bridge = self.recompute_bridge(raw_tier=False, workload_telemetry=True)  # every restore replays
        nodes, _ = self.build_a_to_f(bridge)
        req = self.request_for(handle("s", "rewind", checkpoint=nodes["B"]), PATHS["B"] + "G")
        summary, *_ = run_tree(bridge, req)
        restore = summary["restore"]
        self.assertEqual((restore["mode"], restore["source"]), ("recompute", "replay"))
        record = bridge.workload_record(req.request_id)
        replay, window, state, plan = (named(record, name) for name in (
            "recompute_replay", "restore_sync", "exec_state", "restore_plan"))
        self.assertEqual((len(replay), len(window), len(state), len(plan)), (1, 1, 1, 1))
        self.assertTrue(all(span["phase"] == "prep" for span in replay + window + state + plan))
        self.assertLessEqual(plan[0]["host_end_ns"], window[0]["host_start_ns"])
        self.assertLessEqual(window[0]["host_end_ns"], state[0]["host_start_ns"])
        self.assertLessEqual(state[0]["host_end_ns"], replay[0]["host_start_ns"])
        self.assertGreaterEqual(width(replay[0]) + SLACK_NS, restore["recompute_replay_seconds"] * 1e9)
        self.assertEqual(len(named(record, "drain")), 1)  # the rewind's switch drain
        self.assertFalse(any(span["name"].startswith("restore_sync.") for span in record["spans"]))  # no KV read
        self.assertDisjoint(exposed_spans(record))
        self.assertEqual(record["bytes"], {"read": 0, "decoded": 0, "h2d": 0})

    def test_raw_tier_install_is_a_tier_segment_without_replay(self):
        bridge = self.recompute_bridge(workload_telemetry=True)  # the raw tier serves the history
        nodes, _ = self.build_a_to_f(bridge)
        req = self.request_for(handle("s", "rewind", checkpoint=nodes["B"]), PATHS["B"] + "G")
        summary, *_ = run_tree(bridge, req)
        restore = summary["restore"]
        self.assertEqual((restore["mode"], restore["source"]), ("archive", "raw_tier"))
        record = bridge.workload_record(req.request_id)
        self.assertEqual(named(record, "recompute_replay"), [])
        tiers = {span["name"] for span in record["spans"] if span["name"].startswith("restore_sync.")}
        expected = {f"restore_sync.{wt.raw_tier_label(restore['lease']['tiers']['chunks'])}"}
        self.assertEqual(tiers, expected)
        self.assertEqual(len(named(record, "exec_state")), 1)
        self.assertDisjoint(exposed_spans(record))


class OffloadTierSpanTests(OffloadCase):
    def restored(self, bridge):
        self.build(bridge, "ABC", flush=False)
        bridge.control("raw_tier", {"session_id": "s", "action": "drop"})  # the store serves every attended node
        summary = self.restore_child(bridge, "ABCD")[0]
        return summary, bridge.workload_record(summary["request_key"])

    def test_nocache_store_reads_are_restore_sync_nvme(self):
        summary, record = self.restored(self.bridge("nocache", store_io="nocache", workload_telemetry=True))
        self.assertEqual(set(summary["restore"]["lease"]["tiers"]["chunks"]), {"nvme"})
        self.assertEqual({span["name"] for span in record["spans"] if span["name"].startswith("restore_sync.")},
                         {"restore_sync.nvme"})
        self.assertGreater(record["bytes"]["read"], 0)
        self.assertEqual(record["bytes"]["h2d"], record["bytes"]["read"])

    def test_buffered_store_reads_are_restore_sync_page_cache(self):
        summary, record = self.restored(self.bridge("buffered", workload_telemetry=True))
        self.assertEqual(set(summary["restore"]["lease"]["tiers"]["chunks"]), {"store"})  # wl-w1b's label kept
        self.assertEqual(set(record["tiers"]["chunks"]), {"store"})
        self.assertEqual({span["name"] for span in record["spans"] if span["name"].startswith("restore_sync.")},
                         {"restore_sync.page_cache"})
        self.assertEqual(record["bytes"]["h2d"], record["bytes"]["read"])

    def test_offload_restore_unchanged_with_telemetry_off(self):
        on = self.bridge("nocache-on", store_io="nocache", workload_telemetry=True)
        off = self.bridge("nocache-off", store_io="nocache")
        results = []
        for bridge in (on, off):
            self.build(bridge, "ABC", flush=False)
            bridge.control("raw_tier", {"session_id": "s", "action": "drop"})
            results.append(self.restore_child(bridge, "ABCD"))
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        assert_caches_bitwise(self, results[0][2], results[1][2])
        self.assertNotIn("workload", results[1][0])
        self.assertIsNone(off.workload_record(results[1][0]["request_key"]))


class SessionOpenSpanTests(DisjointMixin, TreeTestCase):
    """W1.2: the first request after a close reopens the session (manifest load, tree registry, prewarm) in the
    controller step: one ``session_open`` span, prep, before the drain and the run's restore, never overlapping."""

    def close(self, bridge, session_id="s"):
        bridge.checkpoint(session_id)
        bridge.control("close", {"session_id": session_id})
        self.assertIsNone(bridge.controller.get(session_id, required=False))

    def sequence(self, bridge):
        out = {}
        run_tree(bridge, self.request_for(handle("s"), "AB"))
        for name, session, letters, reopen in (
                ("resume_after_close", handle("s", "resume"), "ABC", True),
                ("resume_open", handle("s", "resume"), "ABCD", False),
                ("fork_after_close", handle("f", "fork", from_session="s", checkpoint=2), "ABE", True),
                ("rewind_after_close", handle("s", "rewind", checkpoint=1), "AF", True)):
            if reopen:
                self.close(bridge)
            req = self.request_for(session, letters)
            summary, latents, caches, _ = run_tree(bridge, req)
            out[name] = (req, summary, latents, caches, reopen)
            self.assertIsNone(wt._REQUEST.get())
        return out

    def test_the_first_request_after_a_close_records_one_session_open_span(self):
        bridge = self.bridge("open-on", workload_telemetry=True)
        for name, (req, summary, _, _, reopen) in self.sequence(bridge).items():
            record = bridge.workload_record(req.request_id)
            opens = named(record, "session_open")
            if not reopen:
                self.assertEqual(opens, [], name)
                continue
            self.assertEqual(len(opens), 1, name)
            span = opens[0]
            self.assertEqual(span["phase"], "prep")
            self.assertLessEqual(width(span), summary["begin"]["controller_seconds"] * 1e9 + SLACK_NS, name)
            reopened = summary["restore"]["session_open"]
            self.assertEqual([item["session"] for item in reopened], ["s"], name)
            self.assertGreaterEqual(width(span) + SLACK_NS, reopened[0]["seconds"] * 1e9, name)  # the same interval
            for other in record["spans"]:  # before the drain, the publish wait and the restore window
                if other["name"] in ("drain", "publish_wait", "restore_plan") or other["name"].startswith(
                        "restore_sync"):
                    self.assertLessEqual(span["host_end_ns"], other["host_start_ns"], (name, other["name"]))
            self.assertDisjoint(exposed_spans(record))
            self.assertEqual(len(named(record, "drain")), 0 if name.startswith("resume") else 1, name)

    def test_open_control_outside_a_request_records_nothing_and_off_is_unchanged(self):
        bridge = self.bridge("open-control", workload_telemetry=True)
        run_tree(bridge, self.request_for(handle("s"), "AB"))
        self.close(bridge)
        bridge.control("open", {"session_id": "s"})  # a control, not a request: the request after it reopens nothing
        self.assertIsNone(wt._REQUEST.get())
        req = self.request_for(handle("s", "resume"), "ABC")
        summary, *_ = run_tree(bridge, req)
        self.assertEqual(named(bridge.workload_record(req.request_id), "session_open"), [])
        self.assertEqual(summary["restore"]["session_open"], [])
        off_bridge = self.bridge("open-off")
        on, off = self.sequence(self.bridge("open-on2", workload_telemetry=True)), self.sequence(off_bridge)
        for name in on:
            self.assertTrue(torch.equal(on[name][2], off[name][2]), name)
            assert_caches_bitwise(self, on[name][3], off[name][3])
            self.assertNotIn("workload", off[name][1])
            self.assertIsNone(off_bridge.workload_record(off[name][0].request_id))
            self.assertEqual(sorted(set(on[name][1]) - {"workload"}), sorted(off[name][1]), name)


class TierVocabularyTests(unittest.TestCase):
    def test_store_maps_to_page_cache_and_mixed_installs_name_their_slowest_tier(self):
        self.assertEqual([wt.workload_tier(label) for label in ("store", "nvme", "host_raw", "gpu_raw")],
                         ["page_cache", "nvme", "host_raw", "gpu_raw"])
        self.assertEqual(wt.offload_tier_label(["gpu_raw", "host_raw"]), "host_raw")
        self.assertEqual(wt.offload_tier_label(["gpu_raw", "store"]), "page_cache")
        self.assertEqual(wt.offload_tier_label(["store", "nvme", "gpu_raw"]), "nvme")
        self.assertEqual(wt.offload_tier_label(["gpu_raw"]), "gpu_raw")
        for labels in (["nvme"], ["store"], ["host_raw"], ["gpu_raw"], ["gpu_raw", "store"]):
            self.assertIn(f"restore_sync.{wt.offload_tier_label(labels)}", wt.SPAN_NAMES)

    def test_h2d_counts_every_host_to_device_copy(self):
        report = {"mode": "archive", "lease": {"tiers": {"per_chunk": [
            {"tier": "nvme", "tokens": 2}, {"tier": "store", "tokens": 3}, {"tier": "gpu_raw", "tokens": 5},
            {"tier": "host_raw", "tokens": 7}, {"tier": "compressed_prepared", "tokens": 11}]}}}
        self.assertEqual(wt.restore_bytes(report, 10), {"read": 280, "decoded": 110, "h2d": 120})

    def test_span_list_is_the_closed_list(self):
        self.assertEqual(wt.SPAN_NAMES, frozenset({
            "restore_plan", "restore_sync", "restore_sync.gpu_raw", "restore_sync.host_raw",
            "restore_sync.page_cache", "restore_sync.nvme", "restore_sync.compressed_prepared",
            "restore_sync.compressed_cold", "decode", "reconstruct", "rope", "install", "exec_state", "publish_wait",
            "drain", "recompute_replay", "session_open", "prompt_transition", "recache", "context_forward",
            "archival_capture_sync", "guard_wait",  # W1.2: session_open (coordinator 2026-10-19 09:14)
            "promotion_copy", "promotion_fence",  # W1.4 (wl-w1b7): coordinator 2026-10-19 21:00
            "promotion_wait"}))  # W1.4.1: coordinator 2026-10-19 22:40

    def test_drain_span_needs_a_request_scope(self):
        with wt.drain_span():  # no request: nothing, no error
            pass
        with wt.request_scope(None), wt.drain_span():  # telemetry off: nothing
            pass
        telemetry = wt.WorkloadTelemetry("cpu")
        with wt.request_scope(telemetry):
            with wt.drain_span():
                pass
        self.assertIsNone(wt._REQUEST.get())
        spans = telemetry.resolve()["spans"]
        self.assertEqual([(span["name"], span["phase"]) for span in spans], [("drain", "prep")])


if __name__ == "__main__":
    unittest.main()
