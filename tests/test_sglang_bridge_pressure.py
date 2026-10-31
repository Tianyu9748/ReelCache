"""Freeze F device-pressure response in the SGLang bridge (Step 11 d_bridge rev3 on Step 7 dev22j).

* ``request_kv_geometry``: a request's native KV bytes from its cache geometry (real model shapes) and the bytes it
  still allocates once its run opens: SGLang's stage sizes the caches before ``begin_request``, so that is the first
  new chunk's capture (the native caches too for a request that does not carry ``model.cache_tokens``).
* ``begin_request`` calls the archive's ``relieve_pressure("request_start", needed_bytes=...)``; a failing check
  never fails the request; the summary carries the run's ``pressure`` record (events, bytes freed per component,
  latency).
* Before an archive restore installs anything, ``relieve_pressure("restore", needed_bytes=...)`` runs with the
  restore's working set (``restore_working_set``) plus the KV the request still allocates; the restore then runs
  inside ``archive.memory_critical("restore")`` (with a raw tier or the deferral policy) and its lease is closed before
  the section exits; an OOM inside the restore is relieved, recorded and re-raised.
* The decode seam is entered whenever a raw tier exists (a no-op without a tier and without the deferral policy, as
  in C3); its pressure events and an OOM relief are folded into the summary, also when the section raises.
* A section's fitter re-check deferrals (Step 7 dev22j4; recorded in its report on exit) join the request's pressure
  record as one deferred section each (``recheck_event``).
* Controls report the retained device caches (``retained_caches``: byte fields only, for R-D2b).
* End to end on CPU with a fake device-memory provider (CPU tensors stand in for device copies, as in Step 7's
  ``PressureTests``): a request start under pressure demotes the session's GPU raw copies and trims a registered
  cache; the resume then restores from host raw, bitwise.
"""

import contextlib
import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from reelcache.integrations.sglang.bridge import (RESTORE_DEVICE_FRACTION, add_pressure_event, pressure_record,
                                              recheck_event, request_kv_geometry, restore_working_set)
from reelcache.integrations.sglang.controls import RAW_TIER_KEYS
from tests.test_sglang_bridge import (DIM, FRAMES, HEADS, LAYERS, TPF, BridgeTestCase, FullHistoryCache, drive,
                                     handle, request)

try:
    from reelcache import device_pressure  # noqa: F401 - freeze F (Step 7 dev22j)
except ImportError:  # pragma: no cover - a tree before dev22j
    device_pressure = None

BF16 = {"K": torch.bfloat16, "V": torch.bfloat16}


def kv_request(*, layers, heads, head_dim, blocks, frames, tpf, streams=("cond",), dtype="bfloat16",
               cache_tokens=None):
    model = {} if cache_tokens is None else {"cache_tokens": cache_tokens}
    return SimpleNamespace(streams=streams, num_layers=layers, num_heads=heads, head_dim=head_dim, dtype=dtype,
                           block_sizes=(frames,) * blocks, tokens_per_frame=tpf, model=model)


class FakeDevice:
    """Device-memory provider for the raw tier on CPU (Step 7's ``_FakeDevice``)."""

    def __init__(self, *, total=1 << 30, free=1 << 30):
        self.memory = {"total": total, "free": free, "reserved": 0, "allocated": 0, "inactive_split": 0, "peak": 0}

    def __call__(self, device):
        return dict(self.memory)


def chunk_tiers(summary):
    lease = (summary.get("restore") or {}).get("lease") or {}
    return {(item["stream"], item["chunk"]): item["tier"] for item in (lease.get("tiers") or {}).get("per_chunk", [])}


def tier_event(where, *, freed, trims, seconds=0.25):
    """A raw-tier pressure event as ``RawChunkTier.pressure_check`` returns it (demotion, then the trims)."""
    return {"event": "pressure", "where": where, "pressure": True, "headroom": 10, "target": 20, "reserve": 20,
            "needed_bytes": None, "freed_bytes": freed, "gpu_bytes_after": 0, "seconds": seconds,
            "trims": {"reason": where, "trims": [{"name": name, "bytes": nbytes, "seconds": 0.05}
                                                 for name, nbytes in trims],
                      "bytes": sum(nbytes for _, nbytes in trims), "seconds": 0.05}}


class KvGeometryTests(unittest.TestCase):
    def test_causvid_h12_resume_matches_the_native_caches_and_the_tier_entry(self):
        """5b's carry-over census: 60 x (1, 60840, 12, 128) BF16 = 10.44 GiB of native KV after an h=12 request
        (13 chunks x 3 frames x 1560 tokens); one CausVid chunk's capture = the raw tier's 862617600-byte entry."""
        req = kv_request(layers=30, heads=12, head_dim=128, blocks=13, frames=3, tpf=1560, cache_tokens=60840)
        geometry = request_kv_geometry(req, BF16, 12)
        self.assertEqual(geometry["native_bytes"], 60 * 60840 * 12 * 128 * 2)
        self.assertAlmostEqual(geometry["native_bytes"] / 2 ** 30, 10.44, places=2)
        self.assertTrue(geometry["native_allocated"])
        self.assertEqual((geometry["history_chunks"], geometry["new_chunks"], geometry["chunk_tokens"]), (12, 1, 4680))
        self.assertEqual(geometry["chunk_capture_bytes"], 862617600)
        self.assertEqual(geometry["needed_bytes"], 862617600)  # the caches are on the device already

    def test_a_request_without_cache_tokens_counts_its_native_caches(self):
        req = kv_request(layers=30, heads=12, head_dim=128, blocks=15, frames=3, tpf=1560)
        geometry = request_kv_geometry(req, BF16, 14)
        self.assertFalse(geometry["native_allocated"])
        self.assertEqual(geometry["cache_tokens"], 15 * 3 * 1560)  # every frame of the request
        self.assertEqual(geometry["needed_bytes"], geometry["native_bytes"] + geometry["chunk_capture_bytes"])

    def test_cfg_streams_and_fp32_keys(self):
        cond = request_kv_geometry(kv_request(layers=4, heads=2, head_dim=8, blocks=3, frames=2, tpf=6,
                                              cache_tokens=24), BF16, 1)
        cfg = request_kv_geometry(kv_request(layers=4, heads=2, head_dim=8, blocks=3, frames=2, tpf=6,
                                             cache_tokens=24, streams=("cond", "uncond")), BF16, 1)
        self.assertEqual(cfg["native_bytes"], 2 * cond["native_bytes"])
        self.assertEqual(cfg["chunk_capture_bytes"], 2 * cond["chunk_capture_bytes"])
        fp32_keys = request_kv_geometry(kv_request(layers=4, heads=2, head_dim=8, blocks=3, frames=2, tpf=6,
                                                   cache_tokens=24, dtype="float32"),
                                        {"K": torch.float32, "V": torch.bfloat16}, 1)
        self.assertEqual(fp32_keys["native_bytes"], 2 * cond["native_bytes"])  # FP32 cache
        self.assertEqual(fp32_keys["chunk_capture_bytes"], 4 * 12 * 16 * (4 + 2))  # K FP32 + V BF16 per token

    def test_no_new_chunk_needs_nothing(self):
        geometry = request_kv_geometry(kv_request(layers=2, heads=2, head_dim=4, blocks=2, frames=2, tpf=3,
                                                  cache_tokens=12), BF16, 2)
        self.assertEqual((geometry["new_chunks"], geometry["chunk_capture_bytes"], geometry["needed_bytes"]),
                         (0, 0, 0))

    def test_pressure_record_totals_per_component_and_latency(self):
        record = pressure_record({"needed_bytes": 1}, [
            ("request_start", {**tier_event("request_start", freed=7, trims=[("decoder_pool:0", 5)]),
                               "check_seconds": 1.5}),
            ("request_start", None),
            ("request_start", {"error": "RuntimeError: x", "check_seconds": 0.1})])
        add_pressure_event(record, "vae_decode", tier_event("decode", freed=3, trims=[("restorer:0", 2)]))
        add_pressure_event(record, "vae_decode", {"event": "pressure", "where": "decode", "pressure": False,
                                                  "headroom": 50, "target": 20})
        self.assertEqual(record["freed_by_component"], {"gpu_raw": 10, "decoder_pool:0": 5, "restorer:0": 2})
        self.assertEqual((record["freed_bytes"], record["pressured"], len(record["events"])), (17, 2, 3))
        self.assertAlmostEqual(record["seconds"], 1.5 + 0.25 + 0.05)  # bridge-measured, then demotion + trims
        self.assertEqual(record["errors"], [{"source": "request_start", "error": "RuntimeError: x"}])
        self.assertEqual([item["source"] for item in record["events"]], ["request_start", "vae_decode", "vae_decode"])
        self.assertEqual((record["deferred_sections"], record["archival_deferred_seconds"]), (0, 0.0))
        # Step 7 dev22j2: a section entered under pressure annotates its event with the archival deferral
        add_pressure_event(record, "restore_section", {**tier_event("restore", freed=0, trims=[]),
                                                       "deferred_archival": True, "deferral_wait_seconds": 0.25,
                                                       "archival_deferred_seconds": 1.5})
        add_pressure_event(record, "vae_decode", {**tier_event("vae_decode", freed=0, trims=[]),
                                                  "deferred_archival": False, "deferral_wait_seconds": 30.0,
                                                  "deferral_error": "CapacityError: timeout"})
        self.assertEqual((record["deferred_sections"], record["deferral_wait_seconds"],
                          record["archival_deferred_seconds"]), (1, 30.25, 1.5))
        self.assertTrue(record["events"][-2]["deferred_archival"])
        self.assertEqual(record["events"][-1]["deferral_error"], "CapacityError: timeout")

    def test_fitter_recheck_deferrals_count_as_one_deferred_section(self):
        """Step 7 dev22j4: a section's fitter re-check deferrals (``fitter_rechecks`` in its report) become one event
        of the section, counted as the raw tier counts them; archival waited, not the request (no latency)."""
        self.assertIsNone(recheck_event({"reason": "restore", "deferred": False}))
        self.assertIsNone(recheck_event({"reason": "restore", "deferred": False, "fitter_rechecks": []}))
        rechecks = [{"event": "deferred_archival", "deferred_archival": True, "reason": "fitter_recheck",
                     "section": "restore", "node_id": "c000003", "need_bytes": 40, "headroom": 10, "reserve": 20,
                     "target": 60, "waited_seconds": 0.5, "archival_deferred_seconds": 0.5,
                     "outcome": "sections_closed"},
                    {"event": "deferred_archival", "deferred_archival": True, "reason": "fitter_recheck",
                     "section": "restore", "node_id": "prewarm", "need_bytes": 30, "waited_seconds": 0.25,
                     "archival_deferred_seconds": 0.25, "outcome": "caller_waits"}]
        event = recheck_event({"reason": "restore", "deferred": False, "fitter_rechecks": rechecks,
                               "deferred_archival": True, "deferral_reason": "fitter_recheck",
                               "archival_deferred_seconds": 0.75})
        self.assertEqual((event["where"], event["fitter_recheck_deferrals"]), ("restore", 2))
        record = pressure_record(None, [("restore_section", event)])
        self.assertEqual((record["deferred_sections"], record["fitter_recheck_deferrals"], record["pressured"],
                          record["freed_bytes"], record["seconds"], record["deferral_wait_seconds"]),
                         (1, 2, 0, 0, 0.0, 0.0))
        self.assertAlmostEqual(record["archival_deferred_seconds"], 0.75)
        rechecks.clear()  # the record keeps its own copies
        self.assertEqual([item["node_id"] for item in record["events"][0]["fitter_rechecks"]], ["c000003", "prewarm"])
        self.assertEqual(pressure_record(None, [])["fitter_recheck_deferrals"], 0)


class BridgePressureTests(BridgeTestCase):
    representation = "svd_hevc"
    capture_bytes = LAYERS * FRAMES * TPF * HEADS * DIM * (2 + 2)  # one toy chunk's K + V (BF16), cond stream

    def bridge(self, *args, **kwargs):
        bridge = super().bridge(*args, **kwargs)
        # Freeze F5 (cut 4): a restore that allocates nothing (every attended range a GPU raw copy read in place) runs
        # without the relief and the section (tests/test_sglang_bridge_restore_cuts.py). These contract tests run the
        # staged RAW install, whose restore allocates, so the relief and the section are as in F-F4.
        if hasattr(bridge.archive, "raw_install_in_place"):
            bridge.archive.raw_install_in_place = False
        return bridge

    def test_begin_request_asks_for_the_kv_still_to_allocate_and_records_it(self):
        bridge = self.bridge()
        calls = []

        def relieve(where, *, needed_bytes=None):
            calls.append((where, needed_bytes))
            return tier_event(where, freed=7, trims=[("decoder_pool:0", 5), ("restorer:0", 0)])

        with mock.patch.object(bridge.archive, "relieve_pressure", relieve):
            first, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=2)))
            self.assertEqual(calls, [("request_start", self.capture_bytes)])
            resumed, _, _, _ = drive(bridge, request(handle("s1", "resume")))
        estimate = resumed["restore"]["pressure"]["estimate"]
        needed = estimate["bytes"] + self.capture_bytes
        # request start, the pre-restore relief, then the restore section's own entry check (the archive's reserve;
        # dev22j's memory_critical records its entry check as "decode" whatever the section's reason)
        self.assertEqual(calls[1:3], [("request_start", self.capture_bytes), ("restore", needed)])
        self.assertEqual(len(calls), 4)
        self.assertIn(calls[3], [("restore", None), ("decode", None)])
        self.assertEqual(resumed["restore"]["pressure"]["needed_bytes"], needed)
        for summary, history, sources in ((first, 0, ["request_start"]),
                                          (resumed, 2, ["request_start", "restore", "restore_section"])):
            record = summary["pressure"]
            kv = record["kv"]
            self.assertEqual(kv["native_bytes"], LAYERS * 2 * 8 * FRAMES * TPF * HEADS * DIM * 2)
            self.assertTrue(kv["native_allocated"])
            # W1.3: the chunks each request plans (the first pauses after 2 of its 4 blocks; the resume has 2 left)
            self.assertEqual((kv["history_chunks"], kv["new_chunks"]), (history, 2))
            self.assertEqual(kv["needed_bytes"], self.capture_bytes)
            self.assertEqual([item["source"] for item in record["events"]], sources)
            event = record["events"][0]
            self.assertEqual((event["where"], event["pressure"]), ("request_start", True))
            self.assertEqual(event["freed_by_component"], {"gpu_raw": 7, "decoder_pool:0": 5})
            self.assertEqual((record["freed_bytes"], record["pressured"]), (12 * len(sources), len(sources)))
            self.assertGreaterEqual(event["check_seconds"], 0.0)
        self.assertEqual(first["pressure"]["seconds"], first["pressure"]["events"][0]["check_seconds"])

    def test_the_restore_relief_runs_before_the_install_with_the_working_set_and_inside_the_section(self):
        """Coordinator 07:25 / 07:40: relieve_pressure("restore", needed_bytes = restore working set + the KV the
        request still allocates) runs before anything is installed; the restore runs inside
        memory_critical("restore") and its lease is closed before the section exits."""
        bridge = self.bridge()
        drive(bridge, request(handle("s1", pause_after_chunks=2)))
        archive, order, needed = bridge.archive, [], {}
        real_restore, real_relieve = archive.restore, archive.relieve_pressure

        def relieve(where, *, needed_bytes=None):
            order.append(f"relieve:{where}")
            needed.setdefault(where, needed_bytes)
            return real_relieve(where, needed_bytes=needed_bytes)

        @contextlib.contextmanager
        def section(reason):
            order.append(f"enter:{reason}")
            yield {"reason": reason, "deferred": False}
            order.append(f"exit:{reason}")

        real_install = FullHistoryCache.install_restored_history

        def install(cache, layout, write):
            if order[-1] != "install":
                order.append("install")
            return real_install(cache, layout, write)

        def restore(**kwargs):
            order.append("restore")
            lease = real_restore(**kwargs)
            close = lease.close

            def closing():
                order.append("lease_close")
                return close()

            lease.close = closing
            return lease

        with mock.patch.object(archive, "relieve_pressure", relieve), \
                mock.patch.object(archive, "memory_critical", section), \
                mock.patch.object(archive, "restore", restore), \
                mock.patch.object(FullHistoryCache, "install_restored_history", install):
            resumed, latents, caches, _ = drive(bridge, request(handle("s1", "resume")))
        self.assertEqual(order, ["relieve:request_start", "relieve:restore", "enter:restore", "install", "restore",
                                 "lease_close", "exit:restore"])
        restore_report = resumed["restore"]
        self.assertEqual(restore_report["mode"], "archive")
        attended = 2 * FRAMES * TPF  # full history: both restored chunks, one stream
        expected = restore_working_set(SimpleNamespace(num_layers=LAYERS, num_heads=HEADS, head_dim=DIM,
                                                       block_sizes=(FRAMES,) * 4, tokens_per_frame=TPF),
                                       bridge.capture_dtypes, {"cond": SimpleNamespace(
                                           spans=[SimpleNamespace(length=attended)])},
                                       arena_bytes=bridge.config.restore_arena_bytes,
                                       basis_bytes=bridge.config.basis_budget_bytes)
        estimate = restore_report["pressure"]["estimate"]
        self.assertEqual(estimate, expected)
        self.assertEqual(estimate["attended_tokens"], attended)
        per_token = self.capture_bytes // (FRAMES * TPF)  # raw K + V bytes of one token, every layer
        self.assertEqual(estimate["canvases_and_bases_bytes"],
                         math.ceil(RESTORE_DEVICE_FRACTION * attended * per_token))
        self.assertEqual(estimate["install_bytes"], self.capture_bytes)
        self.assertEqual(needed["restore"], estimate["bytes"] + self.capture_bytes)
        self.assertEqual(restore_report["pressure"]["needed_bytes"], needed["restore"])
        self.assertEqual(restore_report["pressure"]["section"]["reason"], "restore")
        self.assertGreaterEqual(restore_report["pressure"]["section"]["entry_seconds"], 0.0)
        self.assertLifetimesRetired(bridge, resumed)

    def test_an_oom_inside_the_restore_is_relieved_recorded_and_raised(self):
        bridge = self.bridge()
        first, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=2)))
        archive = bridge.archive

        @contextlib.contextmanager
        def section(reason):
            report = {"reason": reason, "deferred": False,
                      "raw_pressure": tier_event("restore", freed=0, trims=[])}
            report["raw_pressure"]["pressure"] = False
            try:
                yield report
            except torch.cuda.OutOfMemoryError:
                report["oom_relief"] = tier_event("oom", freed=5, trims=[("decoder_pool:0", 6)], seconds=0.5)
                raise

        def restore(**kwargs):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (test restore)")

        with mock.patch.object(archive, "memory_critical", section), mock.patch.object(archive, "restore", restore):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                drive(bridge, request(handle("s1", "resume")))
        summary = bridge.request_summary(bridge.requests[-1]["request_id"])
        self.assertIn("OutOfMemoryError", summary["error"])
        sources = [item["source"] for item in summary["pressure"]["events"]]
        self.assertEqual(sources[-2:], ["restore_section", "oom_relief"])
        self.assertEqual(summary["pressure"]["freed_by_component"], {"gpu_raw": 5, "decoder_pool:0": 6})
        self.assertFalse(bridge.active_request)
        resumed, _, _, _ = drive(bridge, request(handle("s1", "resume")))  # the session restores again afterwards
        self.assertEqual(resumed["restore"]["mode"], "archive")

    def test_a_failing_check_never_fails_the_request(self):
        bridge = self.bridge()

        def relieve(where, *, needed_bytes=None):
            raise RuntimeError("probe")

        with mock.patch.object(bridge.archive, "relieve_pressure", relieve):
            summary, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=2)))
            resumed, _, _, _ = drive(bridge, request(handle("s1", "resume")))
        self.assertEqual(summary["status"], "paused")
        self.assertEqual(summary["pressure"]["errors"], [{"source": "request_start", "error": "RuntimeError: probe"}])
        self.assertEqual(summary["pressure"]["events"], [])
        self.assertEqual(resumed["status"], "completed")  # neither check fails the restore
        self.assertEqual([item["source"] for item in resumed["pressure"]["errors"]],
                         ["request_start", "restore", "restore_section"])
        self.assertIsNotNone(resumed["restore"]["pressure"]["needed_bytes"])

    def test_no_event_without_a_device_and_no_check_without_the_hook(self):
        bridge = self.bridge()  # the real dev22j check: no CUDA device on CPU -> None
        summary, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=2)))
        self.assertEqual(summary["pressure"]["events"], [])
        self.assertEqual(summary["pressure"]["kv"]["needed_bytes"], self.capture_bytes)
        raw = self.bridge("raw", representation="raw")  # RAW reference archive: no pressure hook
        summary, _, _, _ = drive(raw, request(handle("r1", pause_after_chunks=1)))
        self.assertEqual(summary["pressure"], pressure_record(None, []))

    def test_the_decode_seam_records_its_events_and_an_oom_relief(self):
        bridge = self.bridge()
        summary, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=2)))
        request_id = summary["request_id"]
        with bridge.memory_critical_section("vae_decode", request_id=request_id) as report:
            self.assertEqual(report, {"reason": "vae_decode", "deferred": False})  # real section, no device
        [entry] = bridge.request_summary(request_id)["memory_critical"]
        self.assertGreaterEqual(entry["entry_seconds"], 0.0)

        @contextlib.contextmanager
        def section(reason):
            report = {"reason": reason, "deferred": False,
                      "raw_pressure": tier_event("decode", freed=9, trims=[("decoder_pool:0", 4)])}
            try:
                yield report
            except torch.cuda.OutOfMemoryError:
                report["oom_relief"] = tier_event("oom", freed=11, trims=[("restorer:0", 3)], seconds=0.5)
                raise

        with mock.patch.object(bridge.archive, "memory_critical", section):
            with bridge.memory_critical_section("vae_decode", request_id=request_id):
                pass
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                with bridge.memory_critical_section("vae_decode", request_id=request_id):
                    raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")
        recorded = bridge.request_summary(request_id)
        self.assertEqual(len(recorded["memory_critical"]), 3)
        self.assertIn("oom_relief", recorded["memory_critical"][2])  # recorded although the section raised
        record = recorded["pressure"]
        self.assertEqual([item["source"] for item in record["events"]], ["vae_decode", "vae_decode", "oom_relief"])
        self.assertEqual(record["freed_by_component"], {"gpu_raw": 29, "decoder_pool:0": 8, "restorer:0": 3})
        self.assertEqual(record["pressured"], 3)
        self.assertAlmostEqual(record["seconds"], 2 * (0.25 + 0.05) + (0.5 + 0.05))

    def test_fitter_recheck_deferrals_of_the_restore_and_decode_sections_join_the_request_record(self):
        """Step 7 dev22j4 completes a section's report on exit with its fitter re-check deferrals. The bridge reads
        the report once the section closed, so the restore section's and the decode seam's deferrals are counted in
        the request's pressure record and kept in the restore report and the memory_critical record."""
        bridge = self.bridge()
        drive(bridge, request(handle("s1", pause_after_chunks=2)))

        @contextlib.contextmanager
        def section(reason):
            report = {"reason": reason, "deferred": False}
            try:
                yield report
            finally:  # as dev22j4's memory_critical: recorded after the watched section closed
                report.update(fitter_rechecks=[{"event": "deferred_archival", "deferred_archival": True,
                                                "reason": "fitter_recheck", "section": reason, "node_id": "c000002",
                                                "need_bytes": 64, "waited_seconds": 0.5,
                                                "archival_deferred_seconds": 0.5, "outcome": "sections_closed"}],
                              deferred_archival=True, deferral_reason="fitter_recheck", archival_deferred_seconds=0.5)

        with mock.patch.object(bridge.archive, "memory_critical", section):
            resumed, _, _, _ = drive(bridge, request(handle("s1", "resume")))
            with bridge.memory_critical_section("vae_decode", request_id=resumed["request_id"]):
                pass
        self.assertEqual(resumed["restore"]["pressure"]["section"]["fitter_rechecks"][0]["node_id"], "c000002")
        summary = bridge.request_summary(resumed["request_id"])
        record = summary["pressure"]
        rechecks = [item for item in record["events"] if item.get("event") == "fitter_recheck"]
        self.assertEqual([(item["source"], item["where"]) for item in rechecks],
                         [("restore_section", "restore"), ("vae_decode", "vae_decode")])
        self.assertEqual((record["deferred_sections"], record["fitter_recheck_deferrals"], record["pressured"]),
                         (2, 2, 0))
        self.assertAlmostEqual(record["archival_deferred_seconds"], 1.0)
        [seam] = summary["memory_critical"]
        self.assertEqual((seam["deferral_reason"], len(seam["fitter_rechecks"])), ("fitter_recheck", 1))

    def test_without_a_raw_tier_the_seams_stay_c3(self):
        bridge = self.bridge(raw_tier=False)
        summary, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=2)))
        with bridge.memory_critical_section("vae_decode", request_id=summary["request_id"]) as report:
            self.assertIsNone(report)
        self.assertNotIn("memory_critical", bridge.request_summary(summary["request_id"]))
        with mock.patch.object(bridge.archive, "memory_critical") as section:
            resumed, _, _, _ = drive(bridge, request(handle("s1", "resume")))
        section.assert_not_called()  # no restore section without a raw tier or the deferral policy
        self.assertIsNone(resumed["restore"]["pressure"]["section"])
        self.assertGreater(resumed["restore"]["pressure"]["needed_bytes"], 0)  # the relief itself still runs

    @unittest.skipIf(device_pressure is None, "needs Step 7 dev22j (reelcache.device_pressure)")
    def test_request_start_under_pressure_demotes_gpu_raw_and_trims_then_restores_bitwise(self):
        from reelcache.device_pressure import register_device_trim
        reference = self.bridge("reference", representation="raw")
        drive(reference, request(handle("s1", pause_after_chunks=2)))
        _, expected_latents, expected_caches, _ = drive(reference, request(handle("s1", "resume")))
        bridge = self.bridge()
        tier = bridge.archive.raw_tier
        device = FakeDevice()
        tier._device_memory = device  # before the first adoption: CPU copies count as device copies
        drive(bridge, request(handle("s1", pause_after_chunks=2)))
        bridge.drain_all()
        tier.flush(30)
        self.assertGreater(tier.stats()["gpu_bytes"], 0)
        trimmed = []

        class Canvases:
            def trim_device_cache(self, reason):
                trimmed.append(reason)
                device.memory["free"] += 1 << 28
                return 1 << 28

        canvases = Canvases()
        registration = register_device_trim("decoder_pool:test", canvases.trim_device_cache)
        self.addCleanup(registration.unregister)
        device.memory["free"] = 0  # pressure at the next request start
        resumed, latents, caches, _ = drive(bridge, request(handle("s1", "resume")))
        record = resumed["pressure"]
        event = record["events"][0]
        self.assertEqual((event["source"], event["pressure"]), ("request_start", True))
        self.assertEqual(event["needed_bytes"], self.capture_bytes)
        self.assertGreaterEqual(event["target"], event["needed_bytes"])
        self.assertGreater(record["freed_by_component"]["gpu_raw"], 0)  # the session's GPU raw copies demoted
        self.assertEqual(record["freed_by_component"]["decoder_pool:test"], 1 << 28)
        self.assertEqual(trimmed, ["request_start"])
        self.assertEqual(set(chunk_tiers(resumed).values()), {"host_raw"})  # then served from host raw
        self.assertTrue(torch.equal(latents, expected_latents))
        self.assertCachesEqual(caches, expected_caches)
        stats = bridge.control("capabilities")["raw_tier"]["pressure"]  # tier samples carry the response
        self.assertGreaterEqual(stats["by_where"]["request_start"], 1)
        self.assertGreater(stats["demoted_bytes"], 0)
        self.assertEqual(stats["trimmed_by_component"]["decoder_pool:test"], 1 << 28)
        self.assertLifetimesRetired(bridge, resumed)

    @unittest.skipIf(device_pressure is None, "needs Step 7 dev22j (reelcache.device_pressure)")
    def test_a_pressured_restore_section_defers_archival_around_the_install_only(self):
        """Step 7 dev22j2: a section entered under pressure defers archival GPU work while it runs (inside it the
        pipeline's drain / prewarm raise and submit refuses). The bridge's restore section wraps only the install and
        the restore (its drain ran before), so a resume under pressure restores bitwise and its chunks commit after."""
        from reelcache.integrations.sglang import lineage_archive
        if not hasattr(lineage_archive.LineageSessionArchive, "_pressure_deferred_section"):
            self.skipTest("needs Step 7 dev22j2 (pressure-triggered archival deferral)")
        reference = self.bridge("reference", representation="raw")
        drive(reference, request(handle("s1", pause_after_chunks=2)))
        _, expected_latents, expected_caches, _ = drive(reference, request(handle("s1", "resume")))
        bridge = self.bridge()
        archive = bridge.archive
        device = FakeDevice()
        archive.raw_tier._device_memory = device
        drive(bridge, request(handle("s1", pause_after_chunks=2)))
        bridge.drain_all()
        archive.raw_tier.flush(30)
        device.memory["free"] = 0  # pressure that no relief can cover: the restore section defers archival
        deferred_during_restore = []
        real_restore = archive.restore

        def restore(**kwargs):
            deferred_during_restore.append(archive.pipeline.stats()["deferred"])
            return real_restore(**kwargs)

        with mock.patch.object(archive, "fitter", object()), mock.patch.object(archive, "restore", restore):
            resumed, latents, caches, _ = drive(bridge, request(handle("s1", "resume", durable=True)))
        self.assertEqual(deferred_during_restore, [True])
        self.assertFalse(archive.pipeline.stats()["deferred"])
        section = resumed["restore"]["pressure"]["section"]
        self.assertEqual((section["reason"], section["deferred_archival"]), ("restore", True))
        self.assertGreaterEqual(section["archival_deferred_seconds"], 0.0)
        record = resumed["pressure"]
        [event] = [item for item in record["events"] if item["source"] == "restore_section"]
        self.assertEqual((event["where"], event["deferred_archival"]), ("restore", True))
        self.assertGreaterEqual(event["deferral_wait_seconds"], 0.0)
        self.assertEqual(record["deferred_sections"], 1)
        self.assertTrue(resumed["published"])  # its chunks were submitted and published after the section
        self.assertTrue(torch.equal(latents, expected_latents))
        self.assertCachesEqual(caches, expected_caches)
        stats = bridge.control("capabilities")["raw_tier"]["pressure"]
        self.assertEqual(stats["deferred_sections"], 1)
        self.assertGreaterEqual(stats["by_where"]["restore"], 2)  # the pre-restore relief and the section's check
        self.assertLifetimesRetired(bridge, resumed)

    @unittest.skipIf(device_pressure is None, "needs Step 7 dev22j (reelcache.device_pressure)")
    def test_controls_report_the_pool_and_the_pressure_response(self):
        self.assertTrue({"pressure", "host_pool_bytes", "host_pool_peak_bytes"} <= set(RAW_TIER_KEYS))
        bridge = self.bridge()
        capabilities = bridge.control("capabilities")
        self.assertIsNone(capabilities["archive_memory"]["gpu_raw_reserve_bytes"])
        pressure = capabilities["raw_tier"]["pressure"]
        self.assertTrue(pressure["enabled"])
        self.assertEqual(pressure["reserve_source"], "auto")
        self.assertIn("host_pool_bytes", capabilities["raw_tier"])
        retained = capabilities["retained_caches"]  # R-D2b: device byte fields only (a verifier sums them)
        self.assertEqual(set(retained), {"canvas_arena_bytes", "resident_bases_bytes", "prepared_packs_bytes"})
        self.assertTrue(all(type(value) is int and value >= 0 for value in retained.values()))
        self.assertEqual(retained["prepared_packs_bytes"], 0)  # a host cache: its bytes are host bytes
        self.assertIn("prepared_packs_bytes", capabilities["retained_host_caches"])
        stats = bridge.control("raw_tier", {"action": "stats"})
        self.assertEqual(set(stats["retained_caches"]), set(retained))
        self.assertIn("gpu_memory", stats)
        off = self.bridge("off", gpu_raw_reserve_bytes=0)  # response off: D2/E behaviour, no check at all
        tier = off.archive.raw_tier
        tier._device_memory = FakeDevice(free=0)
        summary, _, _, _ = drive(off, request(handle("s1", pause_after_chunks=2)))
        self.assertFalse(off.control("capabilities")["raw_tier"]["pressure"]["enabled"])
        resumed, _, _, _ = drive(off, request(handle("s1", "resume")))
        self.assertEqual(resumed["pressure"]["events"], [])
        self.assertEqual(off.control("capabilities")["raw_tier"]["pressure"]["checks"], 0)


class _Ticket:
    """An ``ArchiveTicket`` stand-in: pending until resolved, then its node or its failure."""

    def __init__(self, *, error=None, node=None, done=True):
        self.error, self.node_ref, self._done, self.encoder = error, node, done, None

    def done(self):
        return self._done

    def result(self, timeout=None):
        if self.error is not None:
            raise self.error
        return self.node_ref


class RequestHistoryTests(BridgeTestCase):
    """Freeze F3 (coordinator 10:10, after the F2 census): the bridge's recent-request history (``commit_reports``)
    keeps resolved commits as plain data, never a ticket whose exception and traceback frames could pin memory."""

    representation = "svd_hevc"

    def test_resolved_commits_are_plain_data_and_their_reports_are_unchanged(self):
        bridge = self.bridge()
        summary, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=2, durable=True)))
        stored = bridge._request_tickets[summary["request_id"]]
        self.assertTrue(stored and all(entry.get("plain") and "ticket" not in entry for entry in stored))
        self.assertEqual(bridge.commit_reports(summary["request_id"]), summary["commits"])  # what finish reported

    def test_a_failed_commit_leaves_no_exception_or_frame_in_the_history(self):
        import gc
        import weakref
        from reelcache.errors import StateError
        bridge = self.bridge()

        def failing():
            staged = torch.zeros(4680, 64)  # noqa: F841 - held by the traceback's frame, like _fit's staged sources
            raise StateError("ancestor commit c000001 failed; its rates are unavailable")

        try:
            failing()
        except StateError as exc:
            error = exc
        held = weakref.ref(error.__traceback__.tb_next.tb_frame.f_locals["staged"])
        failure = weakref.ref(error)
        expected = repr(error)
        pending = _Ticket(node="node:pending", done=False)
        bridge._request_tickets["r-failed"] = [
            {"ticket": _Ticket(error=error), "node_id": "c000002", "kind": "chunk", "chunk": 1},
            {"ticket": pending, "node_id": "c000003", "kind": "chunk", "chunk": 2}]
        del error
        bridge._settle_request_history()
        gc.collect()
        self.assertIsNone(failure())  # the exception is gone, and with it the frame and its tensor
        self.assertIsNone(held())
        failed, live = bridge._request_tickets["r-failed"]
        self.assertTrue(failed["plain"])
        self.assertEqual(failed["failure"]["type"], "StateError")
        self.assertIn("failing", failed["failure"]["traceback"])  # the formatted traceback is kept as text
        self.assertIs(live["ticket"], pending)  # pending: still the live ticket
        reports = bridge.commit_reports("r-failed", wait=False)
        self.assertEqual((reports[0]["published"], reports[0]["error"]), (False, expected))
        self.assertFalse(reports[1]["published"])
        pending._done = True  # published later: resolved to plain data at the next settle
        bridge._settle_request_history()
        self.assertEqual(bridge._request_tickets["r-failed"][1]["ticket_fields"]["node_ref"], "node:pending")

    def test_records_without_a_request_id_are_filed_never_dropped(self):
        """5b (F3): a request without an SGLang request id gets an internal key; its decode seam's record is filed
        with it (``request_id_missing``); a seam that matches no summary is kept as an unattributed record."""
        bridge = self.bridge()
        first = request(handle("s1", pause_after_chunks=2, durable=True))
        first.request_id = None
        summary, _, _, _ = drive(bridge, first)
        key = summary["request_key"]
        self.assertTrue(summary["request_id_missing"] and str(key).startswith("bridge-"))
        self.assertIs(bridge.request_summary(key), summary)
        self.assertEqual(len(bridge.commit_reports(key)), 2)  # its commit history is filed under the key too
        with bridge.memory_critical_section("vae_decode", request_id=None) as report:  # the real section (raw tier)
            self.assertEqual(report["reason"], "vae_decode")
        [record] = bridge.request_summary(key)["memory_critical"]
        self.assertTrue(record["request_id_missing"])
        self.assertEqual(record["filed_under"], key)
        with bridge.memory_critical_section("vae_decode", request_id=None):
            pass  # a second seam without an id: attributed to no request, kept
        with bridge.memory_critical_section("vae_decode", request_id="never-seen"):
            pass
        unattributed = bridge.describe()["unattributed_memory_critical"]
        self.assertEqual([(item["request_id"], item.get("request_id_missing", False), item["summary_missing"])
                          for item in unattributed], [(None, True, True), ("never-seen", False, True)])
        self.assertEqual(bridge.control("capabilities")["unattributed_memory_critical"], unattributed)  # over HTTP
        second = request(handle("s2", pause_after_chunks=1, durable=True))
        second.request_id = None
        other, _, _, _ = drive(bridge, second)
        self.assertNotEqual(other["request_key"], key)  # two id-less requests never share records
        self.assertEqual((len(bridge.commit_reports(other["request_key"])), len(bridge.commit_reports(key))), (1, 2))
        self.assertEqual(bridge.requests[-1]["request_key"], other["request_key"])


if __name__ == "__main__":
    unittest.main()
