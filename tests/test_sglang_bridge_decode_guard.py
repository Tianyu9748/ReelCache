"""Step 7 dev22j6's decode entry guard in the bridge's request records (Step 11 F5 overlay; coordinator 19:2x).

dev22j6's ``lineage_archive.memory_critical`` checks a decode section's entry: ``report["decode_guard"]`` holds the
target (``max(reserve, prior)`` plus a node on the GPU), the driver's free memory, the cached-block releases, the
reliefs (GPU raw demotions, trims) and ``short``; ``report["deferral_reason"]`` the deferral cause
(``unlearned_reserve`` / ``inflight_node`` / ``short_after_relief``), annotated in place on the event the archive
deferred on (the guard's relief, else the entry check's event, else an event the report does not hold).

The bridge keeps, per decode section, a plain copy of the guard record and its summary (``decode_guard_summary``) in
the request's ``memory_critical`` entry, and files the guard under the request like the other section records: its
reliefs and a deferral on an event the report does not hold join the request's ``pressure`` events (source
``<reason>:decode_guard``), each deferral counted once; ``pressure["decode_guard"]`` sums the checks. Also without a
request id (``filed_under``) and in the unattributed records. Archives without the guard (F4) record nothing new.
"""

import contextlib
import unittest
from unittest import mock

from reelcache.integrations.sglang.bridge import (DECODE_GUARD_DEFERRALS, decode_guard_events, decode_guard_summary,
                                              pressure_record)
from tests.test_sglang_bridge import BridgeTestCase, drive, handle, request
from tests.test_sglang_bridge_pressure import tier_event

PRIOR = 1000
RESERVE = 600


def guard_record(*, free, actions=(), inflight=0, unlearned=False, after=None):
    """``report["decode_guard"]`` as dev22j6's ``_decode_entry_guard`` writes it (``after``: free / headroom after its
    actions, written only once it acted or deferred)."""
    target = max(RESERVE, PRIOR) + inflight
    record = {"reserve": RESERVE, "prior": PRIOR, "inflight_bytes": inflight, "target": target, "free": free,
              "headroom": free + 50, "unlearned": unlearned, "actions": list(actions)}
    if after is not None:
        record.update(free_after=after, headroom_after=after + 50, short=after + 50 < target)
    return record


def entry_check(where="vae_decode"):
    """The section's entry check (``relieve_pressure(reason)``): no pressure."""
    return {"event": "pressure", "where": where, "pressure": False, "headroom": 2000, "reserve": RESERVE,
            "target": RESERVE, "needed_bytes": None, "freed_bytes": 0, "gpu_bytes_after": 0, "seconds": 0.001}


def deferral_fields(wait=0.25, deferred=1.5):
    """What ``_pressure_deferred_section`` writes on exit (report and, via ``annotate_pressure_event``, the event)."""
    return {"deferred_archival": True, "deferral_wait_seconds": wait, "archival_deferred_seconds": deferred}


def guarded_section(build):
    """A dev22j6-shaped ``memory_critical``: ``build(reason)`` returns ``(report, deferred_on)``; with a deferral, the
    report and the event it deferred on are annotated on exit (``deferred_on`` None: the guard did not defer)."""
    @contextlib.contextmanager
    def section(reason):
        report, deferred_on = build(reason)
        try:
            yield report
        finally:
            if deferred_on is not None:
                fields = deferral_fields()
                report.update(deferred=True, **fields)
                deferred_on.update(fields)

    return section


def relieved_and_short(reason):
    """Short of the prior: cached blocks released, then a relief demoting GPU raw; still short -> deferred on the
    relief (``short_after_relief``)."""
    relief = {**tier_event(f"{reason}:decode_guard", freed=300, trims=[("decoder_pool:0", 20)]), "target": PRIOR}
    report = {"reason": reason, "deferred": False, "raw_pressure": entry_check(reason),
              "decode_guard": guard_record(free=400, after=500, actions=[
                  {"action": "release_cached", "bytes": 64, "seconds": 0.002},
                  {"action": "relief", "event": relief}]),
              "deferral_reason": "short_after_relief"}
    relief["deferral_reason"] = "short_after_relief"
    return report, relief


def unlearned_on_entry_check(reason):
    """Room enough, reserve not learned: deferred on the entry check's event (``unlearned_reserve``)."""
    check = entry_check(reason)
    report = {"reason": reason, "deferred": False, "raw_pressure": check,
              "decode_guard": guard_record(free=5000, unlearned=True, after=5000),
              "deferral_reason": "unlearned_reserve"}
    check["deferral_reason"] = "unlearned_reserve"
    return report, check


def inflight_without_entry_event(reason):
    """A node on the GPU, short after the release, no entry-check event in the report: the archive defers on an event
    of its own (``inflight_node``) that only the report's fields describe."""
    report = {"reason": reason, "deferred": False,
              "decode_guard": guard_record(free=1200, inflight=400, after=1300, actions=[
                  {"action": "release_cached", "bytes": 100, "seconds": 0.001}]),
              "deferral_reason": "inflight_node"}
    return report, report


def room_enough(reason):
    """Checked, nothing to do: the driver's free memory holds the target (no actions, no after fields)."""
    return {"reason": reason, "deferred": False, "raw_pressure": entry_check(reason),
            "decode_guard": guard_record(free=5000)}, None


class DecodeGuardRecordTests(unittest.TestCase):
    def test_no_guard_no_summary(self):
        self.assertIsNone(decode_guard_summary({"reason": "vae_decode", "deferred": False}))
        self.assertIsNone(decode_guard_summary({"reason": "restore", "deferral_reason": "fitter_recheck"}))
        self.assertEqual(DECODE_GUARD_DEFERRALS, ("unlearned_reserve", "inflight_node", "short_after_relief"))

    def finished(self, build):
        report, deferred_on = build("vae_decode")
        if deferred_on is not None:
            report.update(deferred=True, **deferral_fields())
            deferred_on.update(deferral_fields())
        return report

    def test_a_relief_deferral_is_summed_and_counted_once_with_its_event(self):
        report = self.finished(relieved_and_short)
        guard = decode_guard_summary(report)
        self.assertEqual((guard["target"], guard["prior"], guard["reserve"], guard["free"], guard["free_after"],
                          guard["short"]), (PRIOR, PRIOR, RESERVE, 400, 500, True))
        self.assertEqual((guard["cache_releases"], guard["cache_released_bytes"], guard["reliefs"],
                          guard["demoted_bytes"]), (1, 64, 1, 300))
        self.assertEqual(guard["relief_freed_by_component"], {"gpu_raw": 300, "decoder_pool:0": 20})
        self.assertEqual((guard["deferral_reason"], guard["deferred"], guard["deferral_wait_seconds"],
                          guard["archival_deferred_seconds"]), ("short_after_relief", True, 0.25, 1.5))
        events = decode_guard_events("vae_decode", report, guard)
        self.assertEqual([(source, event["where"]) for source, event in events],
                         [("vae_decode:decode_guard", "vae_decode:decode_guard")])  # the relief only
        record = pressure_record(None, [("vae_decode", report["raw_pressure"]), *events])
        self.assertEqual((record["deferred_sections"], record["deferral_wait_seconds"],
                          record["archival_deferred_seconds"]), (1, 0.25, 1.5))
        self.assertEqual(record["freed_by_component"], {"gpu_raw": 300, "decoder_pool:0": 20})

    def test_a_deferral_on_the_entry_check_is_counted_with_that_event(self):
        report = self.finished(unlearned_on_entry_check)
        guard = decode_guard_summary(report)
        self.assertEqual((guard["unlearned"], guard["deferral_reason"], guard["deferred"], guard["reliefs"]),
                         (True, "unlearned_reserve", True, 0))
        self.assertEqual(decode_guard_events("vae_decode", report, guard), [])  # the entry check carries it
        record = pressure_record(None, [("vae_decode", report["raw_pressure"])])
        self.assertEqual(record["deferred_sections"], 1)

    def test_a_deferral_on_an_event_outside_the_report_becomes_one_event(self):
        report = self.finished(inflight_without_entry_event)
        guard = decode_guard_summary(report)
        self.assertEqual((guard["inflight_bytes"], guard["target"], guard["short"], guard["deferral_reason"]),
                         (400, PRIOR + 400, True, "inflight_node"))
        [(source, event)] = decode_guard_events("vae_decode", report, guard)
        self.assertEqual((source, event["event"], event["deferral_reason"]),
                         ("vae_decode:decode_guard", "decode_guard_deferral", "inflight_node"))
        record = pressure_record(None, [(source, event)])
        self.assertEqual((record["deferred_sections"], record["pressured"], record["seconds"]), (1, 0, 0.0))
        self.assertAlmostEqual(record["archival_deferred_seconds"], 1.5)

    def test_a_check_without_actions_or_deferral(self):
        report, _ = room_enough("vae_decode")
        guard = decode_guard_summary(report)
        self.assertEqual((guard["cache_releases"], guard["reliefs"], guard["deferral_reason"], guard["deferred"],
                          guard["free_after"], guard["short"]), (0, 0, None, False, None, None))
        self.assertEqual(decode_guard_events("vae_decode", report, guard), [])

    def test_a_fitter_recheck_deferral_is_not_the_guard_s(self):
        report, _ = room_enough("vae_decode")
        report.update(deferral_reason="fitter_recheck", deferred_archival=True, archival_deferred_seconds=0.5)
        guard = decode_guard_summary(report)
        self.assertEqual((guard["deferral_reason"], guard["deferred"]), (None, False))


class BridgeDecodeGuardTests(BridgeTestCase):
    representation = "svd_hevc"

    def bridge(self, *args, **kwargs):
        bridge = super().bridge(*args, **kwargs)
        if hasattr(bridge.archive, "raw_install_in_place"):
            bridge.archive.raw_install_in_place = False
        return bridge

    def sections(self, bridge, builds, request_id):
        reports = []

        def build_and_keep(build):
            def inner(reason):
                report, deferred_on = build(reason)
                reports.append(report)
                return report, deferred_on
            return inner

        for build in builds:
            with mock.patch.object(bridge.archive, "memory_critical", guarded_section(build_and_keep(build))):
                with bridge.memory_critical_section("vae_decode", request_id=request_id):
                    pass
        return reports

    def test_each_decode_keeps_its_guard_record_and_the_request_counts_them(self):
        bridge = self.bridge()
        summary, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=2)))
        request_id = summary["request_id"]
        before = len(bridge.request_summary(request_id)["pressure"]["events"])
        reports = self.sections(bridge, [relieved_and_short, unlearned_on_entry_check, inflight_without_entry_event,
                                         room_enough], request_id)
        recorded = bridge.request_summary(request_id)
        entries = recorded["memory_critical"]
        self.assertEqual(len(entries), 4)
        for entry, report in zip(entries, reports):
            self.assertEqual(entry["decode_guard"], report["decode_guard"])  # a plain copy, per decode
            self.assertIsNot(entry["decode_guard"], report["decode_guard"])
            self.assertEqual(entry["decode_guard_summary"], decode_guard_summary(report))
        self.assertEqual([entry["decode_guard_summary"]["deferral_reason"] for entry in entries],
                         ["short_after_relief", "unlearned_reserve", "inflight_node", None])
        relief = entries[0]["decode_guard"]["actions"][1]["event"]
        self.assertEqual((relief["deferral_reason"], relief["deferred_archival"]), ("short_after_relief", True))
        pressure = recorded["pressure"]
        events = pressure["events"][before:]
        self.assertEqual([item["source"] for item in events],
                         ["vae_decode", "vae_decode:decode_guard",  # entry check, the guard's relief (deferred)
                          "vae_decode",  # the entry check the unlearned deferral annotated
                          "vae_decode:decode_guard",  # the in-flight deferral, outside the report
                          "vae_decode"])
        self.assertEqual(pressure["decode_guard"], {
            "checks": 4, "cache_releases": 2, "cache_released_bytes": 164, "cache_release_seconds": 0.003,
            "reliefs": 1, "demoted_bytes": 300, "short": 2,
            "deferrals": {"short_after_relief": 1, "unlearned_reserve": 1, "inflight_node": 1}})
        self.assertEqual(pressure["deferred_sections"], 3)  # one per deferred decode, never twice
        self.assertAlmostEqual(pressure["archival_deferred_seconds"], 4.5)
        self.assertAlmostEqual(pressure["deferral_wait_seconds"], 0.75)
        self.assertEqual(pressure["freed_by_component"].get("gpu_raw"), 300)

    def test_without_a_request_id_and_unattributed_the_guard_is_kept(self):
        bridge = self.bridge()
        first = request(handle("s1", pause_after_chunks=2, durable=True))
        first.request_id = None
        summary, _, _, _ = drive(bridge, first)
        key = summary["request_key"]
        self.sections(bridge, [relieved_and_short], None)
        [record] = bridge.request_summary(key)["memory_critical"]
        self.assertTrue(record["request_id_missing"])
        self.assertEqual(record["filed_under"], key)
        self.assertEqual(record["decode_guard_summary"]["deferral_reason"], "short_after_relief")
        self.assertEqual(bridge.request_summary(key)["pressure"]["decode_guard"]["checks"], 1)
        self.sections(bridge, [inflight_without_entry_event], "never-seen")
        unattributed = bridge.unattributed_sections[-1]
        self.assertEqual((unattributed["request_id"], unattributed["summary_missing"]), ("never-seen", True))
        self.assertEqual(unattributed["decode_guard_summary"]["deferral_reason"], "inflight_node")

    def test_an_archive_without_the_guard_records_nothing_new(self):
        bridge = self.bridge()
        summary, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=2)))
        request_id = summary["request_id"]

        @contextlib.contextmanager
        def section(reason):
            yield {"reason": reason, "deferred": False, "raw_pressure": entry_check(reason)}

        with mock.patch.object(bridge.archive, "memory_critical", section):
            with bridge.memory_critical_section("vae_decode", request_id=request_id):
                pass
        recorded = bridge.request_summary(request_id)
        [entry] = recorded["memory_critical"]
        self.assertNotIn("decode_guard_summary", entry)
        self.assertNotIn("decode_guard", recorded["pressure"])


try:  # the real dev22j6 guard (Step 7's CPU stand-ins), when the base tree has it
    from reelcache.integrations.sglang import lineage_archive as la
    from reelcache.raw_tier import RawChunkTier
    from tests.test_step7_raw_tier import _adopt, _fake_archive, _FakeDevice, _matrix, _Pipelines
    REAL_GUARD = hasattr(la.LineageSessionArchive, "_decode_entry_guard")
except ImportError:  # pragma: no cover - a base before dev22j6
    REAL_GUARD = False

    class _Pipelines:
        pass


@unittest.skipUnless(REAL_GUARD, "needs Step 7 dev22j6 (LineageSessionArchive._decode_entry_guard)")
class RealDecodeGuardContractTests(_Pipelines, unittest.TestCase):
    """The summary of reports the real dev22j6 ``memory_critical`` writes (Step 7's ``DecodeEntryGuardTests`` setup:
    CPU pipeline, stand-in fitter, 1 GiB fake device, prior 16 %)."""

    def archive(self, *, free):
        import types
        device = _FakeDevice(free=free)
        tier = RawChunkTier(device_memory=device, gpu_budget_bytes=1 << 30, host_budget_bytes=1 << 30)
        self.addCleanup(tier.close)
        entry = _adopt(tier, "c0", _matrix(0))
        with tier._cond:
            tier.fitted(entry)
            tier.published(entry, "ref-0")
        tier.flush(30)
        archive = _fake_archive(tier, defer=False)
        archive.config.decode_reserve_prior_bytes = None
        for name in ("_decode_entry_guard", "_release_cached_blocks"):
            setattr(archive, name, types.MethodType(getattr(la.LineageSessionArchive, name), archive))
        archive.decode_guard_ = la._decode_guard_state(True)
        _, pipeline = self.pipeline(name="bridge-guard")
        archive.pipeline, archive.fitter = pipeline, object()
        return archive, tier, device, entry

    def test_a_first_decode_short_of_the_prior_demotes_and_is_counted_once(self):
        archive, tier, device, entry = self.archive(free=0)
        prior = la.decode_reserve_prior_bytes(archive.config, {"total": 1 << 30})
        device.memory["free"] = tier.reserve_bytes() + 1  # the reserve check passes; the prior does not
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            pass
        guard = decode_guard_summary(report)
        self.assertEqual((guard["target"], guard["prior"], guard["reliefs"], guard["deferral_reason"],
                          guard["deferred"]), (prior, prior, 1, "unlearned_reserve", True))
        self.assertFalse(entry.on_device)  # the GPU raw copy went to host before the decode
        events = [("vae_decode", report.get("raw_pressure")), *decode_guard_events("vae_decode", report, guard)]
        self.assertEqual([(source, event["where"]) for source, event in events],
                         [("vae_decode", "vae_decode"), ("vae_decode:decode_guard", "vae_decode:decode_guard")])
        record = pressure_record(None, events)
        # deferred on the relief, which the record holds once; the entry check found no pressure
        self.assertEqual((record["deferred_sections"], record["pressured"]), (1, 1))
        self.assertEqual(record["freed_by_component"], guard["relief_freed_by_component"])
        self.assertAlmostEqual(record["archival_deferred_seconds"], guard["archival_deferred_seconds"])

    def test_a_first_decode_with_room_is_neither_deferred_nor_waits(self):
        # Step 7 dev22j8: no unconditional first-decode deferral (only a first decode still short after the pause and
        # the relief defers and waits, ``unlearned_reserve``: the test above)
        archive, tier, device, entry = self.archive(free=1 << 30)
        with la.LineageSessionArchive.memory_critical(archive, "vae_decode") as report:
            self.assertFalse(archive.pipeline.stats()["deferred"])
        guard = decode_guard_summary(report)
        self.assertEqual((guard["unlearned"], guard["reliefs"], guard["deferral_reason"], guard["deferred"],
                          guard["deferral_wait_seconds"]), (True, 0, None, False, None))
        self.assertEqual(decode_guard_events("vae_decode", report, guard), [])
        self.assertEqual(pressure_record(None, [("vae_decode", report.get("raw_pressure"))])["deferred_sections"], 0)


if __name__ == "__main__":
    unittest.main()
