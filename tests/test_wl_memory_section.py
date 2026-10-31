"""Workload v2 W1.1 (wl-w1b): the tier-backed RAW archives (``raw_offload``, ``latents_only``) share the lineage
archive's device-memory seams -- the decode and restore memory-critical sections (the pressure response on entry demotes
the least recently used GPU raw copies; an out-of-memory error inside is relieved, then re-raised), the request-start and
pre-restore reliefs and the in-place restore query -- as the lineage archive's own functions bound to the RAW archive
(not copies), so every arm has the same GPU-tier semantics and out-of-memory protection. F6's ``raw`` (no raw tier) has
none of them, as before."""

import unittest

import torch

from reelcache.integrations.sglang.lineage_archive import LineageSessionArchive
from tests.test_sglang_bridge import BridgeTestCase, drive, handle, request
from tests.test_sglang_bridge_pressure import FakeDevice

SEAMS = ("memory_critical", "relieve_pressure", "relieve_device_memory", "restore_allocates")


class MemorySectionTests(BridgeTestCase):
    representation = "raw_offload"

    def pressured(self, store="archive", **overrides):
        """A session of three chunks, written, with the tier's device memory faked (CPU copies count as GPU raw)."""
        bridge = self.bridge(store, **overrides)
        device = FakeDevice()
        bridge.archive.raw_tier._device_memory = device  # before the first adoption
        summary, *_ = drive(bridge, request(handle("s1", pause_after_chunks=3)))
        bridge.drain_all()
        bridge.archive.raw_tier.flush(30)
        return bridge, device, summary

    def test_f6_raw_keeps_no_seam(self):
        raw = self.bridge("raw", representation="raw")
        for name in SEAMS:
            self.assertIsNone(getattr(raw.archive, name, None), name)
        for name in ("config", "fitter", "raw_install_in_place", "_oom_relief"):  # attribute-identical to F6
            self.assertFalse(hasattr(raw.archive, name), name)
        summary, *_ = drive(raw, request(handle("s1", pause_after_chunks=2)))
        with raw.memory_critical_section("vae_decode", request_id=summary["request_id"]) as report:
            self.assertIsNone(report)
        self.assertNotIn("memory_critical", raw.request_summary(summary["request_id"]))
        resumed, *_ = drive(raw, request(handle("s1", "resume")))
        self.assertIsNone(resumed["restore"]["pressure"]["section"])
        no_tier = self.bridge("no-tier", representation="latents_only", restore_mode="recompute", raw_tier=False)
        self.assertIsNone(getattr(no_tier.archive, "memory_critical", None))

    def test_the_seams_are_the_lineage_archives_own_functions(self):
        for representation, extra in (("raw_offload", {}), ("latents_only", {"restore_mode": "recompute"})):
            archive = self.bridge(representation, representation=representation, **extra).archive
            for name in SEAMS:
                self.assertIs(getattr(archive, name).__func__, getattr(LineageSessionArchive, name), name)
            self.assertIs(archive._oom_relief, LineageSessionArchive._oom_relief)
            self.assertIsNone(archive.fitter)  # no archival deferral, pause or decode entry guard

    def test_the_decode_section_demotes_gpu_raw_copies_under_pressure(self):
        bridge, device, summary = self.pressured()
        tier = bridge.archive.raw_tier
        before = tier.stats()["gpu_bytes"]
        self.assertGreater(before, 0)
        device.memory["free"] = 0  # pressure no relief covers: every demotable GPU raw copy goes to host
        with bridge.memory_critical_section("vae_decode", request_id=summary["request_id"]) as report:
            pass
        tier.flush(30)
        self.assertTrue(report["raw_pressure"]["pressure"])
        self.assertEqual(tier.stats()["gpu_bytes"], 0)
        self.assertGreater(tier.stats()["pressure"]["demoted_bytes"], 0)
        record = bridge.request_summary(summary["request_id"])
        [section] = record["memory_critical"]
        self.assertEqual(section["reason"], "vae_decode")
        self.assertTrue(any(item.get("source") == "vae_decode" for item in record["pressure"]["events"]))
        relaxed, _, _ = self.pressured("relaxed")
        with relaxed.memory_critical_section("vae_decode") as report:  # headroom: nothing demoted
            pass
        self.assertFalse(report["raw_pressure"]["pressure"])
        self.assertGreater(relaxed.archive.raw_tier.stats()["gpu_bytes"], 0)

    def test_an_out_of_memory_error_inside_is_relieved_then_reraised(self):
        bridge, device, summary = self.pressured()
        tier = bridge.archive.raw_tier
        self.assertGreater(tier.stats()["gpu_bytes"], 0)
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            with bridge.memory_critical_section("vae_decode", request_id=summary["request_id"]) as report:
                raise torch.cuda.OutOfMemoryError("injected")
        tier.flush(30)
        self.assertIn("oom_relief", report)
        self.assertEqual(tier.stats()["gpu_bytes"], 0)  # every GPU raw copy demoted
        self.assertEqual(tier.stats()["pressure"]["oom_reliefs"], 1)

    def test_a_resume_under_pressure_takes_the_reliefs_and_the_restore_section_and_stays_bitwise(self):
        reference = self.bridge("reference", representation="raw")
        drive(reference, request(handle("s1", pause_after_chunks=3)))
        _, expected_latents, expected_caches, _ = drive(reference, request(handle("s1", "resume")))
        bridge, device, _ = self.pressured()
        device.memory["free"] = 0
        resumed, latents, caches, _ = drive(bridge, request(handle("s1", "resume")))
        restore = resumed["restore"]
        self.assertEqual(restore["mode"], "archive")
        self.assertTrue(restore["allocates"])  # host raw ranges after the request-start relief
        self.assertEqual(restore["pressure"]["section"]["reason"], "restore")
        sources = {item.get("source") for item in resumed["pressure"]["events"]}
        self.assertTrue({"request_start", "restore_section"} <= sources, sources)
        self.assertTrue(torch.equal(latents, expected_latents))
        self.assertCachesEqual(caches, expected_caches)

    def test_an_all_gpu_raw_history_restores_in_place_without_the_section(self):
        size = 1 << 30  # every chunk keeps its GPU raw copy
        bridge, device, _ = self.pressured(gpu_raw_budget_bytes=size)
        resumed, *_ = drive(bridge, request(handle("s1", "resume")))
        restore = resumed["restore"]
        self.assertFalse(restore["allocates"])  # the lineage archive's in-place query: no relief, no section
        self.assertIsNone(restore["pressure"]["section"])
        self.assertEqual(set(item["tier"] for item in restore["lease"]["tiers"]["per_chunk"]), {"gpu_raw"})

    def test_latents_only_has_the_same_decode_section(self):
        bridge, device, summary = self.pressured("latents", representation="latents_only", restore_mode="recompute")
        tier = bridge.archive.raw_tier
        self.assertGreater(tier.stats()["gpu_bytes"], 0)
        device.memory["free"] = 0
        with bridge.memory_critical_section("vae_decode", request_id=summary["request_id"]) as report:
            pass
        tier.flush(30)
        self.assertTrue(report["raw_pressure"]["pressure"])
        self.assertEqual(tier.stats()["gpu_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
