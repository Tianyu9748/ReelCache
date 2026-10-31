"""Workload v2 (freeze W1a): the ReelCache bridge's workload telemetry (CONTRACT §3, "ReelCache bridge record").

* The span names are the contract's closed list; the config field ``workload_telemetry`` is run configuration
  (never archive content), default off.
* Off: no telemetry object, no CUDA event, no ``workload`` key, no stall recorder on a readiness, no restore window
  in any context; on vs off: bitwise the same latents, caches, chunk digests and node refs.
* On: the record's schema (spans, exposed stalls by kind and phase, hidden restore work, prompt transition, tiers,
  bytes); ``restore_sync`` and ``restore_sync.<tier>`` partition the synchronous restore; it is resolved only after
  the request (``request_summary`` / ``workload_record``), the decode seam's ``guard_wait`` included.
* Stall accounting: CPU timers where CUDA is unavailable (a producer that issues a layer late gives a ``host_wait``
  of about that delay; an issued layer a ``device_wait``), and the CUDA-event arithmetic with fake events (pairs
  clamped at 0, sums by kind and phase, hidden = union of the restore windows minus the exposed stall).
* Phase: ``generation`` from the first new block's denoise (the stage's mark, else the first readiness wait outside a
  recache forward, a chunk's clean forward or boundary); a LongLive 1.0 recache forward that reads the restored
  history first waits in ``prep``.
"""

from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

import torch

import reelcache.overlapped_restore as overlapped_restore
from reelcache.errors import ConfigurationError
from reelcache.integrations.sglang import workload_telemetry as wt
from reelcache.integrations.sglang.config import CONTENT_FIELDS, ReelCacheRuntimeConfig
from reelcache.overlapped_restore import LayerReadiness
from tests.test_sglang_bridge import LAYERS, BridgeTestCase, handle, request, timeline_request
from tests.test_sglang_bridge_overlap import OverlapStage, OverlapTestCase, drive

#: CONTRACT §3 (coordinator 2026-10-19 01:10), verbatim, plus ``session_open`` (coordinator 2026-10-19 09:14, W1.2).
CONTRACT_SPANS = {"restore_plan", "restore_sync", "restore_sync.gpu_raw", "restore_sync.host_raw",
                  "restore_sync.page_cache", "restore_sync.nvme", "restore_sync.compressed_prepared",
                  "restore_sync.compressed_cold", "decode", "reconstruct", "rope", "install", "exec_state",
                  "publish_wait", "drain", "recompute_replay", "session_open", "prompt_transition", "recache",
                  "context_forward", "archival_capture_sync", "guard_wait",
                  "promotion_copy", "promotion_fence",  # W1.4 (wl-w1b7): coordinator 2026-10-19 21:00
                  "promotion_wait"}  # W1.4.1: coordinator 2026-10-19 22:40
SPAN_KEYS = {"name", "phase", "host_start_ns", "host_end_ns", "gpu_ms", "stream"}


class FakeEvent:
    """A CUDA timing event stand-in: ``elapsed_time`` from programmed device times (ms)."""

    def __init__(self, at_ms: float):
        self.at_ms = at_ms

    def synchronize(self):
        pass

    def elapsed_time(self, other: FakeEvent) -> float:
        return other.at_ms - self.at_ms


def fake_cuda(telemetry: wt.WorkloadTelemetry, times: list[float]):
    """Turn ``telemetry`` into a CUDA one whose next events complete at ``times`` (ms), in creation order."""
    telemetry.cuda, telemetry.clock, telemetry.compute_stream = True, "cuda_events", "compute-stream"
    queue = list(times)
    telemetry.stream_label = lambda stream=None: "compute"
    return mock.patch.object(wt._Event, "record", staticmethod(lambda stream: FakeEvent(queue.pop(0))))


class SpanNameAndConfigTests(unittest.TestCase):
    def test_span_names_are_the_closed_contract_list(self):
        self.assertEqual(set(wt.SPAN_NAMES), CONTRACT_SPANS)
        self.assertEqual(wt.RESTORE_TIERS, ("gpu_raw", "host_raw", "page_cache", "nvme", "compressed_prepared",
                                            "compressed_cold"))
        telemetry = wt.WorkloadTelemetry("cpu")
        for name in ("restore", "restore_sync.disk", "text_encode", ""):
            with self.assertRaises(ValueError):
                telemetry.begin(name)
            with self.assertRaises(ValueError):
                telemetry.add_host_span(name, 0, 1)
        segments = telemetry.segments("restore_sync")
        with self.assertRaises(ValueError):
            segments.switch("restore_sync.ssd")
        segments.close()
        with self.assertRaises(ValueError):
            telemetry.prompt_transition("rewrite")

    def test_the_config_field_is_run_configuration_off_by_default(self):
        config = ReelCacheRuntimeConfig.from_dict({"store_root": "archive"})
        self.assertIs(config.workload_telemetry, False)
        self.assertNotIn("workload_telemetry", CONTENT_FIELDS)
        on = ReelCacheRuntimeConfig.from_dict({"store_root": "archive", "workload_telemetry": True})
        self.assertEqual(on.content_dict(), config.content_dict())  # never archive content (session ids unchanged)
        self.assertIs(on.to_dict()["workload_telemetry"], True)  # the root's unhashed run-configuration sidecar
        self.assertEqual(ReelCacheRuntimeConfig.from_dict(on.to_dict()), on)
        for bad in (1, "yes", None):
            with self.assertRaises(ConfigurationError):
                ReelCacheRuntimeConfig.from_dict({"store_root": "archive", "workload_telemetry": bad})


class StallAccountingTests(unittest.TestCase):
    def test_cpu_timers_where_cuda_is_unavailable(self):
        telemetry = wt.WorkloadTelemetry("cpu")
        self.assertEqual(telemetry.clock, "host")
        readiness = LayerReadiness(3, "cpu")
        readiness.stall_recorder = telemetry
        readiness.mark(1)  # issued before its wait: a device wait (no host block)

        def late():
            time.sleep(0.06)
            readiness.mark(0)

        thread = threading.Thread(target=late)
        thread.start()
        readiness.wait(0)  # blocks until the producer issues layer 0
        readiness.wait(1)
        thread.join()
        telemetry.capture_opened("kv_recache")  # a wait inside a recache forward is prep... once generation started
        readiness.mark(2)
        readiness.wait(2)
        record = telemetry.resolve()
        stall = record["exposed_stall"]
        self.assertEqual(stall["count"], 3)
        kinds = [(item["kind"], item["layer"], item["phase"]) for item in record["stalls"]]
        self.assertEqual(kinds, [("host_wait", 0, "generation"), ("device_wait", 1, "generation"),
                                 ("device_wait", 2, "generation")])
        self.assertGreaterEqual(stall["by_kind"]["host_wait"], 40.0)  # ~the producer's 60 ms delay
        self.assertLess(stall["by_kind"]["device_wait"], 20.0)
        self.assertAlmostEqual(stall["gpu_ms"], sum(stall["by_kind"].values()))
        self.assertAlmostEqual(stall["gpu_ms"], sum(stall["by_phase"].values()))
        self.assertEqual(telemetry.generation_rule, "readiness_wait")
        self.assertEqual(record["cuda_events"], 0)
        # a recache forward that reads restored history first: its waits are prep
        early = wt.WorkloadTelemetry("cpu")
        early.capture_opened("kv_recache")
        token = early.stall_begin(ready=True)
        early.stall_end(token, layer=0)
        early.capture_closed("kv_recache")
        early.stall_end(early.stall_begin(ready=True, source="join"))
        self.assertEqual(early.phase, "prep")  # a join never starts the generation phase
        self.assertEqual(early.resolve()["exposed_stall"]["by_phase"]["generation"], 0.0)

    def test_cuda_event_arithmetic(self):
        """elapsed(before, after) per wait, clamped at 0 (the consumer had queued work longer than the wait); sums by
        kind and phase; hidden = union of the restore windows - the exposed stall; spans take their pair's time."""
        telemetry = wt.WorkloadTelemetry("cpu")
        # creation order: window(restore) begin 0; prep span 1..3; stall A 10..14 (host wait); generation; stall B
        # 20..19 (negative: the forward still had work queued); window end 40; window(raw) 5..30
        with fake_cuda(telemetry, [0.0, 1.0, 3.0, 10.0, 14.0, 20.0, 19.0, 40.0, 5.0, 30.0]):
            window = telemetry.window_begin("restore")
            with telemetry.span("restore_plan"):
                pass
            first = telemetry.stall_begin(ready=False)  # infers the generation phase (no recache forward open)
            telemetry.stall_end(first, layer=0)
            second = telemetry.stall_begin(ready=True)
            telemetry.stall_end(second, layer=1)
            telemetry.window_end(window)
            raw = telemetry.window_begin("raw")
            telemetry.window_end(raw)
            record = telemetry.resolve()
        stall = record["exposed_stall"]
        self.assertEqual(stall["count"], 2)
        self.assertEqual(stall["by_kind"], {"device_wait": 0.0, "host_wait": 4.0})
        self.assertEqual(stall["by_phase"], {"prep": 0.0, "generation": 4.0})
        self.assertEqual(stall["gpu_ms"], 4.0)
        self.assertEqual(record["restore_window_gpu_ms"], 40.0)  # [0, 40] U [5, 30]
        self.assertEqual(record["hidden_rebuild_gpu_ms"], 36.0)
        [plan] = record["spans"]
        self.assertEqual((plan["name"], plan["phase"], plan["gpu_ms"], plan["stream"]),
                         ("restore_plan", "prep", 2.0, "compute"))
        self.assertEqual(record["clock"], "cuda_events")
        self.assertEqual(record["cuda_events"], 10)
        self.assertEqual(telemetry.resolve(), record)  # resolved once: the events are released, the values kept
        self.assertTrue(all(entry[4] is None for entry in telemetry._spans + telemetry._stalls))

    def test_restore_sync_segments_partition_the_synchronous_restore(self):
        telemetry = wt.WorkloadTelemetry("cpu")
        with fake_cuda(telemetry, [0.0, 2.0, 7.0, 7.5, 9.0, 10.0]):
            segments = telemetry.segments("restore_sync")
            self.assertIs(wt.active_telemetry(), telemetry)
            with wt.tier_segment(wt.raw_tier_label([("gpu_raw", 8), ("host_raw", 8)])):
                pass
            with wt.tier_segment(wt.compressed_tier_label(["compressed_prepared"])):
                pass
            segments.switch("restore_sync")  # the same name: no new segment
            segments.close()
        self.assertIsNone(wt.active_segments())
        spans = telemetry.resolve()["spans"]
        self.assertEqual([item["name"] for item in spans], ["restore_sync", "restore_sync.host_raw", "restore_sync",
                                                            "restore_sync.compressed_prepared", "restore_sync"])
        self.assertEqual([item["gpu_ms"] for item in spans], [2.0, 5.0, 0.5, 1.5, 1.0])  # sum = the window, 10 ms
        for left, right in zip(spans, spans[1:]):  # contiguous: one shared boundary
            self.assertEqual(left["host_end_ns"], right["host_start_ns"])
        self.assertEqual(wt.raw_tier_label(["gpu_raw"]), "gpu_raw")
        self.assertEqual(wt.compressed_tier_label(["compressed_prepared", "compressed_cold"]), "compressed_cold")
        with wt.tier_segment("gpu_raw"):  # no window open: nothing recorded
            pass

    def test_prompt_transition_kinds_and_bytes(self):
        telemetry = wt.WorkloadTelemetry("cpu")
        telemetry.prompt_transition("switch")
        telemetry.end(telemetry.recache_commit())
        telemetry.prompt_transition("switch")
        self.assertEqual(telemetry.resolve()["prompt_transition"]["kind"], "recache")
        shots = wt.WorkloadTelemetry("cpu")
        chunk = type("Chunk", (), {"index": 3, "scene_cut": True, "shot_index": 1})()
        shots.end(shots.chunk_boundary(chunk))
        self.assertEqual(shots.resolve()["prompt_transition"]["kind"], "shot")
        report = {"mode": "archive", "lease": {"tiers": {"per_chunk": [
            {"tier": "gpu_raw", "tokens": 10}, {"tier": "host_raw", "tokens": 20},
            {"tier": "compressed_cold", "tokens": 30}, {"tier": "compressed_prepared", "tokens": 40}]}}}
        self.assertEqual(wt.restore_bytes(report, 2), {"read": 200, "decoded": 140, "h2d": 40})
        raw = {"mode": "archive", "lease": {}, "plans": {"cond": {"spans": [{"length": 6}, {"length": 4}]}}}
        self.assertEqual(wt.restore_bytes(raw, 3, representation="raw"), {"read": 30, "decoded": 0, "h2d": 30})
        self.assertEqual(wt.restore_bytes({"mode": "resident"}, 3), {"read": 0, "decoded": 0, "h2d": 0})
        self.assertEqual(wt.bytes_per_token(num_layers=2, num_heads=2, head_dim=4,
                                            capture_dtypes={"K": torch.float32, "V": torch.bfloat16}), 2 * 2 * 4 * 6)


class BridgeTelemetryTests(OverlapTestCase):
    def requests(self, bridge, stage):
        """A new session, a durable pause, an overlapped fork, a plain fork and a resident resume."""
        out = {"new": self.session_with_history(bridge, stage)}
        out["overlapped"] = self.fork(bridge, stage, "o", overlap=True)
        out["plain"] = self.fork(bridge, stage, "p", overlap=False)
        out["resident"] = drive(bridge, request(handle("p", "resume", pause_after_chunks=4)), stage)
        return out

    def test_off_creates_no_telemetry_no_event_and_records_nothing(self):
        bridge, stage = self.bridge(), OverlapStage()
        recorders = []
        real = overlapped_restore.start_overlapped_restore

        def spy(*args, **kwargs):
            recorders.append(kwargs.get("telemetry"))
            readiness = real(*args, **kwargs)
            recorders.append(readiness.stall_recorder)
            return readiness

        boom = mock.Mock(side_effect=AssertionError("telemetry object created with workload_telemetry off"))
        with mock.patch.object(wt.WorkloadTelemetry, "__init__", boom), \
                mock.patch.object(wt.Segments, "__init__", boom), \
                mock.patch.object(wt._Event, "record", boom), \
                mock.patch.object(overlapped_restore, "start_overlapped_restore", spy):
            results = self.requests(bridge, stage)
        self.assertTrue(recorders and all(item is None for item in recorders))
        summaries = {"new": results["new"], **{key: value[0] for key, value in results.items() if key != "new"}}
        for name, summary in summaries.items():
            self.assertNotIn("workload", summary, name)
            self.assertIsNone(bridge.workload_record(summary["request_key"]), name)
            self.assertNotIn("workload", bridge.request_summary(summary["request_key"]), name)
        self.assertIsNone(wt.active_segments())
        boom.assert_not_called()

    def test_on_and_off_generate_the_same_bits_and_node_refs(self):
        stages = OverlapStage(), OverlapStage()
        off, on = self.bridge("off"), self.bridge("on", workload_telemetry=True)
        left, right = self.requests(off, stages[0]), self.requests(on, stages[1])
        for name in ("overlapped", "plain", "resident"):
            (a, latents_a, _, _), (b, latents_b, _, _) = left[name], right[name]
            self.assertTrue(torch.equal(latents_a, latents_b), name)
            self.assertEqual(a["chunk_digests"], b["chunk_digests"], name)
            self.assertEqual((a["head"], a["restore"]["mode"]), (b["head"], b["restore"]["mode"]), name)
            self.assertEqual((set(b) - set(a), set(a) - set(b)), ({"workload"}, set()), name)  # the only difference
        self.assertSameCaches(stages[1].caches, stages[0].caches)  # the stage caches after the last request
        self.assertEqual(left["new"]["head"], right["new"]["head"])
        for session in ("L", "o", "p"):
            self.assertEqual([node.ref for node in off.archive.path(off.checkpoint(session)["head"])],
                             [node.ref for node in on.archive.path(on.checkpoint(session)["head"])], session)

    def test_record_schema_and_resolution_after_the_request(self):
        bridge, stage = self.bridge(workload_telemetry=True, defer_archival_during_decode=True), OverlapStage()
        results = self.requests(bridge, stage)
        summaries = {"new": results["new"], **{key: value[0] for key, value in results.items() if key != "new"}}
        for name, summary in summaries.items():
            self.assertIn("workload", summary, name)
        overlapped = summaries["overlapped"]
        self.assertIsNone(overlapped["workload"])  # resolved only after the request's outputs
        with bridge.memory_critical_section("decode", request_id=overlapped["request_key"]):
            pass  # the decode seam after the request: its entry guard's wait joins the record
        for name, summary in summaries.items():
            record = bridge.workload_record(summary["request_key"])
            self.assertEqual(record["spans"], bridge.request_summary(summary["request_key"])["workload"]["spans"])
            self.assertTrue({"spans", "exposed_stall", "hidden_rebuild_gpu_ms", "prompt_transition", "tiers",
                             "bytes"} <= set(record), name)
            for span in record["spans"]:
                self.assertEqual(set(span), SPAN_KEYS, name)
                self.assertIn(span["name"], wt.SPAN_NAMES)
                self.assertIn(span["phase"], wt.PHASES)
                self.assertIn(span["stream"], ("compute", "restore", "raw", "host"))
                self.assertLessEqual(span["host_start_ns"], span["host_end_ns"])
                self.assertIsNone(span["gpu_ms"])  # CPU: no GPU time
            stall = record["exposed_stall"]
            self.assertEqual(set(stall), {"count", "gpu_ms", "by_kind", "by_phase"})
            self.assertEqual(set(stall["by_kind"]), set(wt.STALL_KINDS))
            self.assertEqual(set(stall["by_phase"]), set(wt.PHASES))
            self.assertAlmostEqual(stall["gpu_ms"], sum(stall["by_kind"].values()))
            self.assertAlmostEqual(stall["gpu_ms"], sum(stall["by_phase"].values()))
            self.assertEqual(stall["count"], len(record["stalls"]))
            self.assertGreaterEqual(record["hidden_rebuild_gpu_ms"], 0.0)
            self.assertEqual(set(record["prompt_transition"]), {"kind", "gpu_ms", "host_ms"})
            self.assertEqual(record["prompt_transition"]["kind"], "none")
            self.assertEqual(set(record["bytes"]), {"read", "decoded", "h2d"})
            names = [span["name"] for span in record["spans"]]
            self.assertIn("context_forward", names, name)
            self.assertIn("archival_capture_sync", names, name)
            self.assertEqual(record["clock"], "host")
        names = lambda key: [span["name"] for span in bridge.workload_record(summaries[key]["request_key"])["spans"]]
        self.assertIn("drain", names("new"))  # the durable pause
        self.assertNotIn("restore_sync", names("new"))  # a new session restores nothing
        self.assertIn("exec_state", names("resident"))
        self.assertNotIn("restore_sync", names("resident"))
        self.assertIn("guard_wait", names("overlapped"))
        for key in ("overlapped", "plain"):
            record = bridge.workload_record(summaries[key]["request_key"])
            spans = record["spans"]
            # W1.1: the fork's source drain (begin_request's controller step) precedes the run's publish wait and plan
            self.assertEqual([span["name"] for span in spans][:3], ["drain", "publish_wait", "restore_plan"])
            sync = [span for span in spans if span["name"].startswith("restore_sync")]
            for left, right in zip(sync, sync[1:]):  # one contiguous window
                self.assertEqual(left["host_end_ns"], right["host_start_ns"])
            self.assertTrue(all(span["phase"] == "prep" for span in sync))
            self.assertEqual(record["tiers"]["mode"], "archive")
            self.assertEqual(record["tiers"]["chunks"], {"compressed_cold": 2})
            self.assertEqual(record["bytes"]["read"], record["bytes"]["decoded"])
            self.assertGreater(record["bytes"]["read"], 0)
        overlapped = bridge.workload_record(summaries["overlapped"]["request_key"])
        layer_waits = [item for item in overlapped["stalls"] if item["source"] == "layer"]
        self.assertEqual([item["layer"] for item in layer_waits], list(range(LAYERS)))
        self.assertTrue(all(item["phase"] == "generation" for item in layer_waits))  # the first denoise forward
        self.assertEqual(sum(item["source"] == "join" for item in overlapped["stalls"]), 1)
        self.assertIsNotNone(overlapped["restore_window_gpu_ms"])
        plain = bridge.workload_record(summaries["plain"]["request_key"])
        self.assertIn("restore_sync.compressed_cold", [span["name"] for span in plain["spans"]])
        self.assertEqual((plain["exposed_stall"]["count"], plain["hidden_rebuild_gpu_ms"]), (0, 0.0))
        resident = bridge.workload_record(summaries["resident"]["request_key"])
        self.assertEqual((resident["tiers"]["mode"], resident["bytes"]), ("resident", {"read": 0, "decoded": 0,
                                                                                         "h2d": 0}))

    def test_a_recache_forward_reading_restored_history_waits_in_prep(self):
        """LongLive 1.0 resumed at its prompt switch: the recache forward is the first to read the restored history,
        so its per-layer waits are prep (before the first new block's denoise); the recache epoch's commit is a
        ``recache`` transition and a nested ``archival_capture_sync``."""
        bridge, stage = self.bridge(workload_telemetry=True), OverlapStage()
        drive(bridge, timeline_request(handle("L", pause_after_chunks=3), ("A", "A")), stage)
        bridge.drain_all()
        summary, _, _, _ = drive(bridge, timeline_request(
            handle("o", "fork", from_session="L", checkpoint=3, pause_after_chunks=4, overlap_restore=True),
            ("A", "C")), stage)
        record = bridge.workload_record(summary["request_key"])
        layer_waits = [item for item in record["stalls"] if item["source"] == "layer"]
        self.assertEqual(len(layer_waits), LAYERS)
        self.assertTrue(all(item["phase"] == "prep" for item in record["stalls"]))  # the join follows the recache
        self.assertAlmostEqual(record["exposed_stall"]["by_phase"]["prep"],
                               sum(item["gpu_ms"] for item in record["stalls"]))
        self.assertEqual(record["exposed_stall"]["by_phase"]["generation"], 0.0)
        names = [(span["name"], span["phase"]) for span in record["spans"]]
        self.assertIn(("recache", "prep"), names)
        self.assertIn(("context_forward", "generation"), names)
        self.assertEqual(record["prompt_transition"]["kind"], "recache")
        self.assertEqual(record["generation_rule"], "clean_forward")


class RawInstallTelemetryTests(BridgeTestCase):
    """Synchronous RAW installs (``restore.DenseInstall``): nested per-layer ``decode`` / ``install`` spans inside the
    synchronous restore; the RAW reference archive's store reads stay ``restore_sync`` (page cache vs NVMe is not
    known to the bridge); a lineage archive's raw-tier ranges are ``restore_sync.<tier>``."""

    def resumed_record(self, bridge):
        from tests.test_sglang_bridge import drive as plain_drive

        plain_drive(bridge, request(handle("s1", pause_after_chunks=2)))
        bridge.drain_all()
        if getattr(bridge.archive, "raw_tier", None) is not None:
            bridge.archive.raw_tier.flush(30)
        resumed, _, _, _ = plain_drive(bridge, request(handle("s1", "resume")))  # new caches: an archive restore
        self.assertEqual(resumed["restore"]["mode"], "archive")
        return bridge.workload_record(resumed["request_key"])

    def assertNestedSteps(self, record, window):
        spans = record["spans"]
        sync = [span for span in spans if span["name"].startswith("restore_sync")]
        low, high = sync[0]["host_start_ns"], sync[-1]["host_end_ns"]
        steps = [span for span in spans if span["name"] in ("decode", "install")]
        self.assertEqual([span["name"] for span in steps], ["decode", "install"] * LAYERS)
        self.assertTrue(all(low <= span["host_start_ns"] <= span["host_end_ns"] <= high for span in steps))
        inside = [span for span in sync if span["name"] == window]
        self.assertEqual(len(inside), 1, [span["name"] for span in sync])
        for step in steps:  # nested in the window that ran the install
            self.assertTrue(inside[0]["host_start_ns"] <= step["host_start_ns"] <= inside[0]["host_end_ns"])

    def test_raw_reference_archive_restore(self):
        record = self.resumed_record(self.bridge(workload_telemetry=True))
        self.assertEqual([span["name"] for span in record["spans"] if span["name"].startswith("restore_sync")],
                         ["restore_sync"])
        self.assertNestedSteps(record, "restore_sync")
        self.assertEqual(record["tiers"], {"mode": "archive"})  # the RAW archive has no tier records
        self.assertEqual(record["bytes"]["read"], record["bytes"]["h2d"])
        self.assertEqual(record["bytes"]["decoded"], 0)
        self.assertGreater(record["bytes"]["read"], 0)

    def test_lineage_raw_tier_ranges(self):
        record = self.resumed_record(self.bridge(representation="svd_hevc", workload_telemetry=True))
        tiers = set(record["tiers"]["chunks"])
        self.assertTrue(tiers and tiers <= {"gpu_raw", "host_raw"}, tiers)
        window = f"restore_sync.{wt.raw_tier_label(tiers)}"
        self.assertNestedSteps(record, window)
        self.assertEqual([span["name"] for span in record["spans"] if span["name"].startswith("restore_sync")],
                         ["restore_sync", window, "restore_sync"])
        self.assertEqual(record["bytes"]["decoded"], 0)


if __name__ == "__main__":
    unittest.main()
