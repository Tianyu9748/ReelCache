"""F5 (Step 11's ``overlap_restore`` spec, integrated in Step 7 dev22j6): ``LineageSessionArchive.restore`` with
``overlapped=True`` starts the compressed calls as layer-overlapped restores (Step 5c's
:mod:`reelcache.overlapped_restore`) and returns at once; ``lease.complete()`` joins them and fills the plain path's
records. CPU: the bridge's packed restorer on the software decoder pool, sessions resumed from a persisted store with
the plain and the overlapped restore (joined at once, as the bridge does before anything derived leaves the request):
latents and KV bitwise equal, the overlap record kept. The config field: default on, bool only, not a content field.
dev22j6v5 (Step 5c's O4): the raw tier's ranges (GPU raw, host raw, mixed with compressed, CFG streams) restored
overlapped are bitwise the plain restore, one raw install per call group on the fast path.
"""

from __future__ import annotations

import copy
import shutil
import unittest

import torch

from reelcache.archive_restore import LineageRestorer
from reelcache.errors import ConfigurationError
from reelcache.integrations.sglang import ReelCacheRuntimeConfig
from reelcache.integrations.sglang import config as config_module
import tests.test_sglang_bridge as t
from tests.test_packed_nvdec import SoftwarePool

STREAMS = ("cond", "uncond")
SESSIONS = (("s1", "A", 3), ("s2", "B", 2))


class OverlapRestoreConfigTests(unittest.TestCase):
    def test_default_validation_and_identity(self):
        config = ReelCacheRuntimeConfig.from_dict({"store_root": "overlap"})
        self.assertTrue(config.overlap_restore)
        self.assertFalse(ReelCacheRuntimeConfig.from_dict({"store_root": "overlap", "overlap_restore": False})
                         .overlap_restore)
        with self.assertRaisesRegex(ConfigurationError, "overlap_restore must be a bool"):
            ReelCacheRuntimeConfig.from_dict({"store_root": "overlap", "overlap_restore": 1})
        self.assertNotIn("overlap_restore", config_module.CONTENT_FIELDS)  # scheduling only: same stored content
        self.assertEqual(ReelCacheRuntimeConfig.from_dict(config.to_dict()), config)


class RestoreAllocatesTests(unittest.TestCase):
    """Step 11 cut 4 (F5): ``restore_allocates(plans)`` is False only when the RAW install reads device copies in
    place (``raw_install_in_place``, Step 5c's O4) and every attended node has a GPU raw copy now."""

    def archive(self, *, in_place, raw_restore="prefer", on_device=("ref-0", "ref-1")):
        from types import SimpleNamespace

        from reelcache.integrations.sglang.lineage_archive import LineageSessionArchive
        entries = [SimpleNamespace(node_ref=ref, on_device=ref in on_device) for ref in ("ref-0", "ref-1")]
        tier = SimpleNamespace(entries=lambda node_refs=None: [entry for entry in entries
                                                               if node_refs is None or entry.node_ref in node_refs])
        return SimpleNamespace(raw_install_in_place=in_place, raw_tier=tier,
                               config=SimpleNamespace(raw_restore=raw_restore)), LineageSessionArchive.restore_allocates

    @staticmethod
    def plans(*refs):
        from types import SimpleNamespace
        return {"cond": SimpleNamespace(spans=tuple(SimpleNamespace(node_ref=ref) for ref in refs))}

    def test_only_an_in_place_install_of_device_copies_allocates_nothing(self):
        from reelcache.integrations.sglang.lineage_archive import LineageSessionArchive
        self.assertTrue(LineageSessionArchive.raw_install_in_place)  # F5 (O4): GPU raw copies install in place
        archive, allocates = self.archive(in_place=True)
        self.assertFalse(allocates(archive, self.plans("ref-0", "ref-1")))
        self.assertTrue(allocates(archive, self.plans()))  # nothing attended: the plain answer
        partial, allocates = self.archive(in_place=True, on_device=("ref-0",))
        self.assertTrue(allocates(partial, self.plans("ref-0", "ref-1")))  # one copy on host: staging
        staged, allocates = self.archive(in_place=False)
        self.assertTrue(allocates(staged, self.plans("ref-0", "ref-1")))
        compressed, allocates = self.archive(in_place=True, raw_restore="off")
        self.assertTrue(allocates(compressed, self.plans("ref-0", "ref-1")))
        no_tier, allocates = self.archive(in_place=True)
        no_tier.raw_tier = None
        self.assertTrue(allocates(no_tier, self.plans("ref-0")))


class RawOverlapArchiveTests(t.BridgeTestCase):
    """Step 5c's O4 through Step 7's archive (dev22j6v5) on F4's bridge: forks whose attended ranges the raw tier
    serves (GPU raw, host raw, mixed with compressed; one or two CFG streams), restored with ``overlapped=True`` and
    joined at once, are bitwise the plain restore. Each call group's raw ranges are ONE raw install on the fast path
    (rows read in place, nothing staged), and ``restore_allocates`` is False only for an all-GPU-raw history."""

    representation = "svd_hevc"

    def placed(self, bridge, placement, *, chunks=3, streams=("cond",)):
        """Session L with ``chunks`` chunks, its raw entries placed per chunk (the ``raw_tier`` control)."""
        t.drive(bridge, t.request(t.handle("L", pause_after_chunks=chunks), streams=streams))
        bridge.drain_all()
        bridge.archive.raw_tier.flush(30)
        bridge.control("raw_tier", {"session_id": "L", "action": "demote"})  # the GPU raw budget for the placement
        bridge.control("raw_tier", {"session_id": "L", "action": "place", "placement": placement})
        compressed = sum(1 for tier in placement.values() if tier == "compressed")
        t.wait_until(lambda: bridge.control("describe", {"session_id": "L"})["raw_tier"]["entries"]
                     == chunks - compressed)

    def forks(self, bridge, *, checkpoint=2, streams=("cond",)):
        """The plain fork of L at ``checkpoint``, then the overlapped one: [(summary, latents, caches, lease record,
        restore_allocates as the restore started)] for each."""
        archive, plain = bridge.archive, bridge.archive.restore
        mode, seen = {"overlapped": False}, []

        def restore(**kwargs):
            allocates = archive.restore_allocates(kwargs["plans"])
            if mode["overlapped"]:
                lease = plain(**(kwargs | {"overlapped": True, "overlap_options": {
                    **(kwargs.get("overlap_options") or {}), "require_prepared": False}}))
                lease.complete()
            else:
                lease = plain(**kwargs)
            seen.append(({key: copy.deepcopy(lease.stats.get(key)) for key in ("tiers", "overlap", "jobs", "spans")},
                         allocates))
            return lease

        results = []
        archive.restore = restore
        try:
            for overlapped in (False, True):
                mode["overlapped"] = overlapped
                summary, latents, caches, _ = t.drive(bridge, t.request(t.handle(
                    f"fork-{overlapped}", "fork", from_session="L", checkpoint=checkpoint,
                    pause_after_chunks=checkpoint + 1), streams=streams))
                results.append((summary, latents.clone(), caches))
                self.assertLifetimesRetired(bridge, summary)
        finally:
            archive.restore = plain
        self.assertEqual(len(seen), 2)
        return [result + item for result, item in zip(results, seen)]

    @staticmethod
    def tiers(record):
        return {(item["stream"], item["chunk"]): item["tier"] for item in record["tiers"]["per_chunk"]}

    def assertBitwise(self, plain, overlapped):
        self.assertLatentsEqual(overlapped[1], plain[1])
        self.assertCachesEqual(overlapped[2], plain[2])
        self.assertEqual(overlapped[0]["chunk_digests"], plain[0]["chunk_digests"])
        self.assertEqual(self.tiers(overlapped[3]), self.tiers(plain[3]))
        self.assertEqual(overlapped[3]["jobs"], plain[3]["jobs"])  # the plain path's records, filled by complete()

    def test_an_all_raw_history_is_one_raw_install_on_the_fast_path(self):
        layers = t.LAYERS
        for tier, allocates in (("gpu_raw", False), ("host_raw", True)):
            with self.subTest(tier=tier):
                bridge = self.bridge(f"archive-{tier}", raw_restore="prefer")
                self.placed(bridge, {"0": tier, "1": tier})
                plain, overlapped = self.forks(bridge)
                self.assertEqual(set(self.tiers(plain[3]).values()), {tier})
                self.assertEqual((plain[4], overlapped[4]), (allocates, allocates))
                record = overlapped[3]["overlap"]
                self.assertEqual((record["requested"], record["calls"], record["raw_calls"], record["fallback"]),
                                 (True, 1, 1, None))
                [raw] = record["raw_install"]
                self.assertEqual((raw["spans"], raw["jobs"], raw["layers"]), (2, 2 * 2 * layers, layers))
                self.assertEqual((raw["fast_layers"], raw["staged_layers"], raw["staged_rows"], raw["in_place_rows"]),
                                 (layers, 0, 0, 2 * 2 * layers))
                self.assertEqual((raw["rope_per"], raw["rope_calls"]), ("span", 2 * layers))
                self.assertGreaterEqual(overlapped[3]["tiers"]["raw_install_seconds"], 0.0)
                self.assertBitwise(plain, overlapped)

    def test_mixed_histories_join_raw_and_compressed_ranges_per_call_group(self):
        """Host raw, compressed and GPU raw chunks in one stream; GPU raw and compressed in two CFG streams (one call
        group per stream, each with its own raw install)."""
        cases = (({"0": "host_raw", "1": "compressed", "2": "gpu_raw"}, 4, 3, ("cond",), [(None, 2)]),
                 ({"0": "gpu_raw", "1": "compressed"}, 3, 2, ("cond", "uncond"), [("cond", 1), ("uncond", 1)]))
        for index, (placement, chunks, checkpoint, streams, groups) in enumerate(cases):
            with self.subTest(placement=placement, streams=streams):
                bridge = self.bridge(f"archive-mixed-{index}", raw_restore="prefer")
                self.placed(bridge, placement, chunks=chunks, streams=streams)
                plain, overlapped = self.forks(bridge, checkpoint=checkpoint, streams=streams)
                self.assertIn(self.tiers(plain[3])[(streams[0], 1)], ("compressed_prepared", "compressed_cold"))
                self.assertEqual((plain[4], overlapped[4]), (True, True))
                record = overlapped[3]["overlap"]
                self.assertEqual((record["calls"], record["raw_calls"], record["fallback"]),
                                 (len(groups), len(groups), None))
                self.assertEqual([(item["stream"], item["spans"]) for item in record["raw_install"]], groups)
                self.assertEqual(sum(item["staged_rows"] for item in record["raw_install"]), 0)
                self.assertBitwise(plain, overlapped)


class OverlapRestoreBridgeTests(t.BridgeTestCase):
    representation = "svd_hevc"

    def packed(self, bridge):
        pool = SoftwarePool()
        self.addCleanup(pool.close)
        bridge.archive.restorer = LineageRestorer(bridge.archive.runtime, pool, arena_bytes=1 << 30)

    def overlapped(self, bridge, records):
        """Every archive restore overlapped, joined at once (the bridge joins before anything leaves the request)."""
        archive = bridge.archive
        plain = archive.restore

        def restore(**kwargs):
            lease = plain(**(kwargs | {"overlapped": True}))
            records.append(dict(lease.stats["overlap"]))
            lease.complete()
            records[-1]["completed_restorer_reports"] = len(lease.stats["restorer"])
            return lease

        archive.restore = restore

    def test_overlapped_restores_are_bitwise_the_plain_restore(self):
        source = self.bridge(store="source")
        for session_id, prompt, chunks in SESSIONS:
            t.drive(source, t.request(t.handle(session_id, pause_after_chunks=chunks), prompt=prompt, streams=STREAMS))
        source.close()
        results, records = {}, []
        for overlapped in (False, True):
            name = f"copy-{overlapped}"
            shutil.copytree(self.workspace / "source", self.workspace / name)
            bridge = self.bridge(store=name)
            self.packed(bridge)
            if overlapped:
                self.overlapped(bridge, records)
            for session_id, prompt, chunks in SESSIONS:
                summary, latents, caches, _ = t.drive(
                    bridge, t.request(t.handle(session_id, "resume"), prompt=prompt, streams=STREAMS))
                self.assertEqual((summary["resumed_from_chunks"], summary["generated_chunks"]), (chunks, 4))
                results[(overlapped, session_id)] = (latents, caches)
            self.assertLifetimesRetired(bridge)
            bridge.close()
        self.assertEqual(len(records), len(SESSIONS))
        for record in records:  # prepared histories (the session-open prewarm): one overlapped call per CFG stream
            self.assertEqual((record["requested"], record["calls"], record["fallback"]), (True, len(STREAMS), None))
            self.assertEqual(record["completed_restorer_reports"], len(STREAMS))
        for session_id, _, _ in SESSIONS:
            latents, caches = results[(True, session_id)]
            reference_latents, reference_caches = results[(False, session_id)]
            with self.subTest(session=session_id):
                self.assertTrue(torch.equal(latents, reference_latents))
                for stream in reference_caches:
                    for got, want in zip(caches[stream], reference_caches[stream]):
                        self.assertEqual(got.end, want.end)
                        self.assertTrue(torch.equal(got.k[:, :got.end], want.k[:, :want.end]))
                        self.assertTrue(torch.equal(got.v[:, :got.end], want.v[:, :want.end]))


if __name__ == "__main__":
    unittest.main()
