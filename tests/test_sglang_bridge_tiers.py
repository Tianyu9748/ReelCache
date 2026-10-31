"""Freeze D through the SGLang bridge on CPU (Step 11): the raw-chunk tier's holders, records and controls.

The toy causal model and fake caches of ``test_sglang_bridge`` drive the bridge like SGLang's causal stage. The
archive is the Step 6 lineage runtime with Step 7's raw-chunk tier (``reelcache.raw_tier``, freeze D dev22a):

* every commit records its encoder (``commits[i].encoder``);
* sessions hold raw entries (forks share them); close, delete, replace, rewind, abort and an acknowledged archival
  failure move or drop the holds, so no tier memory outlives the sessions that hold it (I5);
* restores record the tier of every chunk (``restore.lease.tiers``; ``resident`` for a resident resume) and the
  payload verification (``restore.lease.verification``);
* raw-tier restores equal the RAW archive's bitwise; compressed restores equal the ``raw_tier=False`` (C3) restore;
  a mixed placement (``raw_tier`` control, ``place``) restores each chunk from its own tier.
"""

import threading
import unittest
from unittest import mock

import torch

from tests.test_sglang_bridge import (FRAMES, TPF, BridgeTestCase, Stage, control_error, drive, handle,
                                      publication_hook, request, wait_until)

try:
    from reelcache import raw_tier as _raw_tier_module  # noqa: F401 - freeze D (Step 7 dev22a)
    HAVE_RAW_TIER = True
except ImportError:  # a C3 core: the bridge's freeze D hooks are no-ops there
    HAVE_RAW_TIER = False

TOKENS = FRAMES * TPF


def tier_totals(bridge):
    stats = bridge.archive.raw_tier.stats()
    return {"entries": stats["entries"]["total"], "gpu_bytes": stats["gpu_bytes"], "host_bytes": stats["host_bytes"],
            "unpublished_bytes": stats["unpublished_bytes"]}


def held(bridge, session_id):
    return bridge.control("describe", {"session_id": session_id})["raw_tier"]


def chunk_tiers(summary):
    return {(item["stream"], item["chunk"]): item["tier"]
            for item in summary["restore"]["lease"]["tiers"]["per_chunk"]}


def cache_rows(caches, chunk):
    """The K and V rows of one chunk in every layer of the cond stream."""
    rows = slice(chunk * TOKENS, (chunk + 1) * TOKENS)
    return [(cache.k[:, rows].clone(), cache.v[:, rows].clone()) for cache in caches["cond"]]


@unittest.skipUnless(HAVE_RAW_TIER, "needs Step 7's reelcache.raw_tier (freeze D dev22a)")
class RawTierBridgeTests(BridgeTestCase):
    representation = "svd_hevc"

    def settle(self, bridge):
        """Publish everything, then let the tier's worker finish its copies and drops."""
        bridge.drain_all()
        bridge.archive.raw_tier.flush(30)

    def released(self, bridge):
        """Every raw entry is gone and no device or host byte is accounted (no memory ratchet)."""
        wait_until(lambda: tier_totals(bridge) == {"entries": 0, "gpu_bytes": 0, "host_bytes": 0,
                                                   "unpublished_bytes": 0})

    # -- records ---------------------------------------------------------------------------------------------------
    def test_commits_record_their_encoder(self):
        bridge = self.bridge()
        summary, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=2)))
        self.assertTrue(summary["durable"])
        self.assertEqual([item["encoder"] for item in summary["commits"]], ["libx265", "libx265"])
        reports = bridge.commit_reports(summary["request_id"])
        self.assertEqual([item["encoder"] for item in reports], ["libx265", "libx265"])
        raw = self.bridge("raw", representation="raw")
        raw_summary, _, _, _ = drive(raw, request(handle("r1", pause_after_chunks=1)))
        self.assertEqual([item["encoder"] for item in raw_summary["commits"]], [None])
        capabilities = bridge.control("capabilities")
        self.assertEqual(capabilities["encoder"]["selected"], "libx265")
        self.assertTrue(capabilities["raw_tier"]["enabled"])
        self.assertIsNone(raw.control("capabilities")["encoder"])

    def test_archive_restore_from_the_raw_tier_is_bitwise_the_raw_archive(self):
        reference = self.bridge("reference", representation="raw")
        drive(reference, request(handle("s1", pause_after_chunks=2)))
        _, expected_latents, expected_caches, _ = drive(reference, request(handle("s1", "resume")))
        bridge = self.bridge()
        drive(bridge, request(handle("s1", pause_after_chunks=2)))
        self.settle(bridge)
        resumed, latents, caches, _ = drive(bridge, request(handle("s1", "resume")))
        restore = resumed["restore"]
        self.assertEqual(restore["mode"], "archive")
        tiers = chunk_tiers(resumed)
        self.assertEqual(sorted(tiers), [("cond", 0), ("cond", 1)])
        self.assertTrue(set(tiers.values()) <= {"gpu_raw", "host_raw"}, tiers)
        self.assertEqual(restore["lease"]["tiers"]["policy"], "prefer")
        self.assertIn(restore["lease"]["verification"]["mode"], ("sync", "async", "off"))
        self.assertEqual(restore["lease"]["verification"]["payload_wait_seconds"], 0.0)  # raw tiers read no payload
        self.assertTrue(torch.equal(latents, expected_latents))
        for left, right in zip(caches["cond"], expected_caches["cond"]):
            self.assertEqual(left.end, right.end)
            self.assertTrue(torch.equal(left.k[:, :left.end], right.k[:, :right.end]))
            self.assertTrue(torch.equal(left.v[:, :left.end], right.v[:, :right.end]))
        self.assertLifetimesRetired(bridge, resumed)

    def test_compressed_restores_and_nodes_equal_raw_tier_off(self):
        """raw_tier=False is C3; with the tier on, the planner reads re-staged sources: identical node ids, and a
        compressed-only restore (raw_restore="off") installs the same cache bitwise."""
        off = self.bridge("off", raw_tier=False)
        on = self.bridge("on", raw_restore="off")
        results = {}
        for name, bridge in (("off", off), ("on", on)):
            first, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=2)))
            resumed, latents, caches, _ = drive(bridge, request(handle("s1", "resume")))
            results[name] = (first["head"], resumed, latents, caches)
        self.assertEqual(results["off"][0], results["on"][0])  # same published node ids
        self.assertIsNone(off.archive.raw_tier)
        self.assertFalse(off.control("capabilities")["raw_tier"]["enabled"])
        self.assertNotIn("raw_tier", off.control("describe", {"session_id": "s1"}))
        for name in ("off", "on"):
            tiers = chunk_tiers(results[name][1])
            self.assertTrue(set(tiers.values()) <= {"compressed_prepared", "compressed_cold"}, (name, tiers))
        self.assertIsNone(results["off"][1]["restore"]["lease"]["tiers"]["policy"])
        self.assertTrue(torch.equal(results["off"][2], results["on"][2]))
        for left, right in zip(results["off"][3]["cond"], results["on"][3]["cond"]):
            self.assertTrue(torch.equal(left.k[:, :left.end], right.k[:, :right.end]))
            self.assertTrue(torch.equal(left.v[:, :left.end], right.v[:, :right.end]))

    def test_resident_resume_reports_every_chunk_resident(self):
        bridge = self.bridge()
        stage = Stage()
        drive(bridge, request(handle("s1", pause_after_chunks=2), streams=("cond", "uncond")), stage=stage)
        resumed, _, _, _ = drive(bridge, request(handle("s1", "resume"), streams=("cond", "uncond")), stage=stage)
        restore = resumed["restore"]
        self.assertEqual(restore["mode"], "resident")
        self.assertEqual(chunk_tiers(resumed), {(stream, chunk): "resident" for stream in ("cond", "uncond")
                                                for chunk in (0, 1)})
        tiers = restore["lease"]["tiers"]
        self.assertEqual((tiers["chunks"], tiers["bytes"], tiers["policy"]), ({"resident": 4}, {}, "prefer"))
        self.assertEqual([(item["token_start"], item["tokens"]) for item in tiers["per_chunk"][:2]],
                         [(0, TOKENS), (TOKENS, TOKENS)])
        # coordinator 23:35: every restore reports the configured mode and the payloads verified (none here)
        self.assertEqual(restore["lease"]["verification"], {"mode": "sync", "payload_wait_seconds": 0.0,
                                                            "verified_payloads": 0})
        off = self.bridge("unverified", verify_reads=False)
        drive(off, request(handle("s2", pause_after_chunks=1)), stage=stage)
        again, _, _, _ = drive(off, request(handle("s2", "resume")), stage=stage)
        self.assertEqual((again["restore"]["mode"], again["restore"]["lease"]["verification"]["mode"]),
                         ("resident", "off"))

    # -- holders: close, fork, rewind, delete, replace, abort, archival failure ---------------------------------
    def test_close_releases_the_session_raw_entries(self):
        bridge = self.bridge()
        drive(bridge, request(handle("s1", pause_after_chunks=2)))
        self.settle(bridge)
        entry = held(bridge, "s1")
        self.assertEqual((entry["entries"], entry["unpublished_bytes"]), (2, 0))
        self.assertGreater(entry["gpu_bytes"] + entry["host_bytes"], 0)
        self.assertEqual(bridge.control("raw_tier", {"session_id": "s1"})["session"]["entries"], 2)
        closed = bridge.control("close", {"session_id": "s1"})
        self.assertEqual(closed["raw_tier"]["detached"], 2)
        self.released(bridge)
        # Reopened later, the session restores from its compressed nodes.
        resumed, _, _, _ = drive(bridge, request(handle("s1", "resume")))
        self.assertTrue(set(chunk_tiers(resumed).values()) <= {"compressed_prepared", "compressed_cold"})

    def test_fork_shares_entries_until_every_holder_closes(self):
        bridge = self.bridge()
        drive(bridge, request(handle("s1", pause_after_chunks=2)))
        bridge.control("fork", {"session_id": "s1", "target": "f1", "checkpoint": 1})
        self.settle(bridge)
        self.assertEqual(held(bridge, "s1")["entries"], 2)
        fork = held(bridge, "f1")
        self.assertEqual((fork["entries"], fork["shared_entries"]), (1, 1))
        bridge.control("close", {"session_id": "s1"})
        wait_until(lambda: tier_totals(bridge)["entries"] == 1)  # chunk 0 stays: the fork holds it
        bridge.control("open", {"session_id": "s1"})  # a reopened session holds what survives of its path
        self.assertEqual(held(bridge, "s1")["entries"], 1)
        resumed, _, _, _ = drive(bridge, request(handle("f1", "resume")))
        self.assertTrue(set(chunk_tiers(resumed).values()) <= {"gpu_raw", "host_raw"})
        bridge.control("close", {"session_id": "f1"})
        wait_until(lambda: tier_totals(bridge)["entries"] == 1)  # the fork's own chunks go; s1 keeps chunk 0
        bridge.control("close", {"session_id": "s1"})
        self.released(bridge)

    def test_equal_content_shares_one_entry_until_both_sessions_close(self):
        bridge = self.bridge()
        drive(bridge, request(handle("s1", pause_after_chunks=2)))
        drive(bridge, request(handle("s2", pause_after_chunks=2)))  # same prompt: the same node refs
        self.settle(bridge)
        self.assertEqual(tier_totals(bridge)["entries"], 2)
        self.assertEqual((held(bridge, "s1")["shared_entries"], held(bridge, "s2")["shared_entries"]), (2, 2))
        bridge.control("close", {"session_id": "s1"})
        self.assertEqual(held(bridge, "s2")["entries"], 2)
        bridge.control("close", {"session_id": "s2"})
        self.released(bridge)

    def test_rewind_detaches_the_abandoned_entries(self):
        bridge = self.bridge()
        drive(bridge, request(handle("s1", pause_after_chunks=3)))
        self.settle(bridge)
        self.assertEqual(held(bridge, "s1")["entries"], 3)
        bridge.control("rewind", {"session_id": "s1", "checkpoint": 1})
        self.settle(bridge)
        self.assertEqual(held(bridge, "s1")["entries"], 1)
        wait_until(lambda: tier_totals(bridge)["entries"] == 1)
        bridge.control("close", {"session_id": "s1"})
        self.released(bridge)

    def test_delete_and_replace_release_their_entries(self):
        bridge = self.bridge()
        # Distinct prompts: equal content publishes equal node refs, one entry with both sessions as holders.
        drive(bridge, request(handle("s1", pause_after_chunks=2), prompt="A"))
        drive(bridge, request(handle("s2", pause_after_chunks=2), prompt="B"))
        self.settle(bridge)
        self.assertEqual(tier_totals(bridge)["entries"], 4)
        bridge.control("delete", {"session_id": "s1"})
        wait_until(lambda: tier_totals(bridge)["entries"] == 2)
        drive(bridge, request(handle("s2", replace=True, pause_after_chunks=1), prompt="C"))
        self.settle(bridge)
        self.assertEqual(held(bridge, "s2")["entries"], 1)  # only the replacement's own commit
        wait_until(lambda: tier_totals(bridge)["entries"] == 1)
        bridge.control("close", {"session_id": "s2"})
        self.released(bridge)

    def test_abort_retires_unpublished_entries_and_close_releases_the_rest(self):
        bridge = self.bridge()
        gate = threading.Event()
        publication_hook(bridge, gate=gate)  # nothing publishes until the gate opens
        drive(bridge, request(handle("s1"), blocks=3))
        self.assertGreater(tier_totals(bridge)["unpublished_bytes"], 0)  # never-block: submits did not wait
        closer = threading.Thread(target=lambda: bridge.control("close", {"session_id": "s1", "mode": "abort"}))
        closer.start()
        gate.set()
        closer.join(60)
        self.assertFalse(closer.is_alive())
        self.released(bridge)

    def test_acknowledged_archival_failure_holds_the_published_head(self):
        from reelcache import lineage_commit

        bridge = self.bridge()
        real = lineage_commit.LatestMethodPlanner.plan

        def failing(planner, *args, chunk_id=None, **kwargs):
            if chunk_id is not None and "c000001" in chunk_id:
                raise RuntimeError("injected planning failure")
            return real(planner, *args, chunk_id=chunk_id, **kwargs)

        with mock.patch.object(lineage_commit.LatestMethodPlanner, "plan", failing):
            drive(bridge, request(handle("s1"), blocks=3))
            control_error(self, "archival_failure", "pause", {"session_id": "s1"}, bridge)
        bridge.archive.raw_tier.flush(30)
        # The failed commits' entries are dropped once no worker reads them; the session holds chunk 0's only.
        wait_until(lambda: tier_totals(bridge)["entries"] == 1)
        entry = held(bridge, "s1")
        self.assertEqual((entry["entries"], entry["unpublished_bytes"]), (1, 0))
        bridge.control("close", {"session_id": "s1"})
        self.released(bridge)

    # -- the raw_tier control: deterministic mixed placement --------------------------------------------------
    def test_place_raw_mixed_restore_takes_each_chunk_from_its_tier(self):
        reference = self.bridge("reference", representation="raw")
        drive(reference, request(handle("s1", pause_after_chunks=3)))
        _, _, raw_caches, _ = drive(reference, request(handle("s1", "resume")))
        compressed = self.bridge("compressed", raw_tier=False)
        drive(compressed, request(handle("s1", pause_after_chunks=3)))
        _, _, lossy_caches, _ = drive(compressed, request(handle("s1", "resume")))
        bridge = self.bridge()
        drive(bridge, request(handle("s1", pause_after_chunks=3)))
        self.settle(bridge)
        placed = bridge.control("raw_tier", {"session_id": "s1", "action": "place",
                                             "placement": {"0": "host_raw", "1": "compressed"}})
        self.assertTrue(placed["report"]["host_raw"] and placed["report"]["compressed"])
        promoted = bridge.control("raw_tier", {"session_id": "s1", "action": "place", "placement": {"2": "gpu_raw"}})
        self.assertIsNotNone(promoted["report"]["gpu_raw"])
        wait_until(lambda: held(bridge, "s1")["entries"] == 2)  # chunk 1 keeps its compressed node only
        resumed, _, caches, _ = drive(bridge, request(handle("s1", "resume")))
        tiers = chunk_tiers(resumed)
        self.assertEqual(tiers[("cond", 0)], "host_raw")
        self.assertIn(tiers[("cond", 1)], ("compressed_prepared", "compressed_cold"))
        self.assertEqual(tiers[("cond", 2)], "gpu_raw")
        for chunk, expected in ((0, raw_caches), (1, lossy_caches), (2, raw_caches)):
            for (k, v), (ek, ev) in zip(cache_rows(caches, chunk), cache_rows(expected, chunk)):
                self.assertTrue(torch.equal(k, ek) and torch.equal(v, ev), chunk)
        for op_args, code in (({"session_id": "s1", "action": "place", "placement": {"0": "nvme"}},
                               "invalid_argument"),
                              ({"session_id": "s1", "action": "place", "placement": {}}, "invalid_argument"),
                              ({"session_id": "s1", "action": "teleport"}, "invalid_argument"),
                              ({"session_id": "nope", "action": "demote"}, "unknown_session")):
            control_error(self, code, "raw_tier", op_args, bridge)
        demoted = bridge.control("raw_tier", {"session_id": "s1", "action": "demote"})
        self.assertEqual(demoted["session"]["gpu_bytes"], 0)
        bridge.control("raw_tier", {"session_id": "s1", "action": "drop"})  # drains the resume's commits first
        self.released(bridge)
        bridge.control("raw_tier", {"action": "reset_peaks"})
        stats = bridge.control("raw_tier")["raw_tier"]
        self.assertEqual((stats["gpu_peak_bytes"], stats["host_peak_bytes"]), (0, 0))
        off = self.bridge("no_tier", raw_tier=False)
        drive(off, request(handle("s1", pause_after_chunks=1)))
        control_error(self, "invalid_argument", "raw_tier", {"session_id": "s1", "action": "drop"}, off)
        self.assertFalse(off.control("raw_tier")["raw_tier"]["enabled"])


if __name__ == "__main__":
    unittest.main()
