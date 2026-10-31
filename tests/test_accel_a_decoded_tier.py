"""accel A1 (2026-10-30): the decoded-coefficient tier (``reelcache.decoded_tier``) on CPU with the software decoder.

The archival validation decode fills the tier (normative HEVC output, tile rectangles); restores route cached nodes
past the decoder (copies into the canvases, then the unchanged kernels). Routed restores must equal the decoder path
bitwise, also with every canvas pixel outside the tile rectangles poisoned; the stored bytes never change; the LRU
honours its budget and leases; forgotten / collected / archive-evicted packs leave the tier; the configuration is
validated; without a tier nothing changes.
"""

from __future__ import annotations

import shutil
import unittest
import uuid
from types import SimpleNamespace

import numpy as np
import torch

from reelcache import decoded_tier as dt
from reelcache.archive_restore import LineageRestorer
from reelcache.decoded_tier import DecodedTier, offer_decoded
from reelcache.errors import ConfigurationError
from reelcache.integrations.sglang.config import ReelCacheRuntimeConfig
from reelcache.integrations.sglang.lineage_archive import LineageSessionArchive
from reelcache.manager import KVRuntime
from reelcache.types import ModelIdentity
from tests import step6_support as s6
from tests.test_packed_nvdec import SoftwarePool
from tests.test_step7_store_faults import _commit, _destination


def _same(test, left, right):
    test.assertEqual(set(left), set(right))
    for key in left:
        test.assertTrue(torch.equal(left[key], right[key]), key)


def _poison(restorer):
    """Every canvas the arena hands out is filled with 0xAB first: pixels a routed copy does not write stay poisoned."""
    arena = restorer.restorer.arena
    acquire = arena.acquire

    def poisoned(shape):
        canvas = acquire(shape)
        canvas.fill_(0xAB)
        return canvas

    arena.acquire = poisoned


class TierCase(unittest.TestCase):
    """Lineage ``a`` (4 chunks) archived while a tier is open (filled by the validation decode), ``b`` (2 chunks).

    ``shared``: the store, the archival-filled tier and the reference caches are built once per class (tests that only
    restore); otherwise once per test (tests that change the store)."""

    budget = 1 << 30
    shared = True
    with_tier = True

    @classmethod
    def setUpClass(cls):
        if cls.shared:
            cls._build(cls, cls.addClassCleanup)

    def setUp(self):
        if not self.shared:
            self._build(self, self.addCleanup)

    @staticmethod
    def _build(target, cleanup):
        target.root = s6.ROOT / f"accel-a-{uuid.uuid4().hex}"
        cleanup(shutil.rmtree, target.root, True)
        target.tier = None
        if target.with_tier:
            target.tier = DecodedTier(target.budget, device="cpu", model=s6.MODEL)
            cleanup(target.tier.close)
        runtime = target.runtime = KVRuntime(target.root, s6.MODEL)
        cleanup(lambda: runtime._closed or runtime.close())
        registry = s6.registry()
        target.heads = {}
        for lineage, first, count in (("a", 0, 4), ("b", 8, 2)):
            runtime.create_lineage(lineage)
            target.heads[lineage] = _commit(runtime, registry, lineage, first, count=count)
        target.reference = {}
        for name, head in target.heads.items():
            caches, destination = _destination(4 * s6.TOKENS)
            LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 30).restore(head, destination)
            target.reference[name] = caches

    def make_tier(self, **options):
        tier = DecodedTier(options.pop("budget", self.budget), device="cpu", model=s6.MODEL, **options)
        self.addCleanup(tier.close)
        return tier

    def set_layer_model(self, a):
        """c(h) = a ms for this test's routing of the shared tier (restored at cleanup)."""
        self.addCleanup(setattr, self.tier, "layer_model", self.tier.layer_model)
        self.tier.layer_model = (a, 0.0, None, "test")

    def restore(self, restorer, head):
        caches, destination = _destination(4 * s6.TOKENS)
        report = restorer.restore(head, destination)
        return caches, report

    def restorer(self, tier, poison=True):
        pool = SoftwarePool()
        restorer = LineageRestorer(self.runtime, pool, arena_bytes=1 << 30, decoded_tier=tier)
        if poison:
            _poison(restorer)
        return restorer, pool


class RoutedRestoreTests(TierCase):
    def test_validation_decode_fills_the_tier_with_tile_rectangles(self):
        stats = self.tier.stats()
        self.assertEqual(stats["fills"], 6)  # one entry per archived node pack (6 chunks)
        self.assertEqual(stats["entries"], 6)
        self.assertEqual(stats["fill_errors"], 0)
        self.assertLessEqual(stats["resident_bytes"], self.budget)
        history = LineageRestorer.history(SimpleNamespace(runtime=self.runtime), self.heads["a"])
        packet = self.runtime.load_packet(history[0]["packet"])
        canvas = packet["metadata"]["canvas"]
        rects = sum(tile["pictures"][1] * canvas["planes"] * tile["rect"][2] * tile["rect"][3]
                    for tile in packet["metadata"]["tiles"])
        entry = self.tier._entries[packet["metadata"]["payloads"]["stream.hevc"]["sha256"]]
        self.assertEqual(entry.nbytes, rects)
        self.assertLess(rects, canvas["pictures"] * canvas["planes"] * canvas["height"] * canvas["width"])

    def test_routing_every_node_is_bitwise_the_decoder_path_and_decodes_nothing(self):
        tier = self.make_tier(route="all")
        for packet in self._packs():
            self._fill(tier, packet)
        restorer, pool = self.restorer(tier)
        caches, report = self.restore(restorer, self.heads["a"])
        _same(self, caches, self.reference["a"])
        record = report["decoded_tier"]
        self.assertEqual((record["routed_nodes"], record["hits"], record["misses"]), (4, 4, 0))
        self.assertEqual(pool.jobs, [])  # nothing reached the decoder
        self.assertGreater(record["routed_bytes"], 0)
        self.assertEqual(tier.stats()["leased"], 0)  # leases released with the restore
        # the second lineage was never bound: every node misses and is decoded
        caches, report = self.restore(restorer, self.heads["b"])
        _same(self, caches, self.reference["b"])
        self.assertEqual(report["decoded_tier"]["routed_nodes"], 0)

    def test_tier_filled_while_archiving_serves_the_restore(self):
        restorer, pool = self.restorer(self.tier)  # route "deficit" with c = 0: every cached node is routed
        self.set_layer_model(0.0)
        caches, report = self.restore(restorer, self.heads["a"])
        _same(self, caches, self.reference["a"])
        self.assertEqual(report["decoded_tier"]["routed_nodes"], 4)
        self.assertEqual(pool.jobs, [])

    def test_deficit_routes_the_oldest_nodes_until_nvdec_fits_the_layer_compute(self):
        restorer, pool = self.restorer(self.tier)
        self.set_layer_model(1e9)  # compute dwarfs decoding: nothing routed
        caches, report = self.restore(restorer, self.heads["a"])
        _same(self, caches, self.reference["a"])
        record = report["decoded_tier"]
        self.assertEqual(record["routed_nodes"], 0)
        before = record["predicted_nvdec_ms_per_layer"]
        jobs = len(pool.jobs)
        self.assertGreater(jobs, 0)
        # c such that about half the decode must go: the two oldest nodes are routed
        self.tier.layer_model = (0.55 * before / self.tier.target, 0.0, None, "test")  # restored at cleanup
        pool.jobs.clear()
        caches, report = self.restore(restorer, self.heads["a"])
        _same(self, caches, self.reference["a"])
        record = report["decoded_tier"]
        self.assertEqual(record["routed_order"], list(range(record["routed_nodes"])))  # oldest first
        self.assertIn(record["routed_nodes"], (2, 3))
        self.assertLessEqual(record["predicted_nvdec_ms_per_layer_after"], 0.55 * before + 1e-9)
        self.assertEqual(len(pool.jobs), jobs * (4 - record["routed_nodes"]) // 4)

    def test_routed_resident_and_key_subsets_equal_the_decoder_path(self):
        tier = self.make_tier(route="all")
        for packet in self._packs():
            self._fill(tier, packet)
        history = LineageRestorer.history(SimpleNamespace(runtime=self.runtime), self.heads["a"])
        keys = [key for key in s6.KEYS if key.layer == 1]
        options = {"keys": keys, "resident": [history[1]["node_id"]]}
        reference_restorer = LineageRestorer(self.runtime, SoftwarePool(), arena_bytes=1 << 30)
        expected, destination = _destination(4 * s6.TOKENS)
        reference_restorer.restore(self.heads["a"], destination, **options)
        restorer, pool = self.restorer(tier)
        caches, destination = _destination(4 * s6.TOKENS)
        report = restorer.restore(self.heads["a"], destination, **options)
        _same(self, caches, expected)
        self.assertEqual(report["decoded_tier"]["routed_nodes"], 3)
        self.assertEqual(pool.jobs, [])

    def test_bounded_layer_ahead_restores_route_identically(self):
        tier = self.make_tier(route="all")
        for packet in self._packs():
            self._fill(tier, packet)
        restorer, pool = self.restorer(tier)
        caches, destination = _destination(4 * s6.TOKENS)
        report = restorer.restore(self.heads["a"], destination, layer_ahead=0)
        _same(self, caches, self.reference["a"])
        self.assertEqual(report["layer_ahead"], 0)
        self.assertEqual(report["decoded_tier"]["routed_nodes"], 4)

    def test_archive_eviction_drops_the_evicted_nodes_pictures(self):
        tier = self.make_tier()
        history = LineageRestorer.history(SimpleNamespace(runtime=self.runtime), self.heads["b"])
        packets = tuple(item["packet"] for item in history)
        for packet in packets:
            self._fill(tier, packet)
        restorer, _ = self.restorer(tier, poison=False)
        self.restore(restorer, self.heads["b"])  # binds both packs (store loads)
        fake = SimpleNamespace(restorer=restorer, _records={})
        LineageSessionArchive._payload_evicted(fake, "manifest:x", packets[:1], ())
        stats = tier.stats()
        self.assertEqual((stats["entries"], stats["forgotten"], stats["bindings"]), (1, 1, 1))

    def _packs(self):
        history = LineageRestorer.history(SimpleNamespace(runtime=self.runtime), self.heads["a"])
        return [item["packet"] for item in history]

    def _fill(self, tier, packet_ref):
        """Fill ``tier`` for one stored pack the way the archival validation decode does."""
        from reelcache.node_packets import _validation_decode

        packet = self.runtime.load_packet(packet_ref)
        metadata = packet["metadata"]
        stream = self.runtime.store.get(packet["payloads"]["stream.hevc"])
        canvas = metadata["canvas"]
        decoded, _ = _validation_decode(stream, pictures=canvas["pictures"], height=canvas["height"],
                                        width=canvas["width"], pix_fmt=canvas["pix_fmt"], threads=1)
        self.assertTrue(tier.offer(metadata["payloads"]["stream.hevc"]["sha256"], decoded, metadata["tiles"],
                                   model=s6.MODEL))


class DefaultPathTests(TierCase):
    with_tier = False

    def test_without_a_tier_nothing_is_offered_routed_or_recorded(self):
        self.assertEqual(dt.open_tiers(), [])
        self.assertEqual(offer_decoded("x" * 64, np.zeros((1, 3, 32, 32), np.uint8),
                                       [{"pictures": [0, 1], "rect": [0, 0, 32, 32]}]), 0)
        restorer = LineageRestorer(self.runtime, SoftwarePool(), arena_bytes=1 << 30)
        caches, report = self.restore(restorer, self.heads["a"])
        _same(self, caches, self.reference["a"])
        self.assertNotIn("decoded_tier", report)
        self.assertNotIn("decoded_tier", restorer.stats())
        self.assertNotIn("decoded_tier", restorer.forget_unstored())

    def test_an_open_tier_does_not_change_the_stored_bytes(self):
        tier = self.make_tier()
        root = s6.ROOT / f"accel-a-{uuid.uuid4().hex}"
        self.addCleanup(shutil.rmtree, root, True)
        runtime = KVRuntime(root, s6.MODEL)
        self.addCleanup(runtime.close)
        runtime.create_lineage("a")
        head = _commit(runtime, s6.registry(), "a", 0, count=2)
        self.assertEqual(tier.stats()["fills"], 2)
        mine = LineageRestorer.history(SimpleNamespace(runtime=runtime), head)
        theirs = LineageRestorer.history(SimpleNamespace(runtime=self.runtime), self.heads["a"])[:2]
        self.assertEqual(len(mine), 2)
        for left, right in zip(mine, theirs, strict=True):  # content-addressed packs: the same bytes
            self.assertEqual(left["packet"], right["packet"])


class EvictionTests(TierCase):
    shared = False  # changes the store

    def test_forget_unstored_drops_collected_packs_and_keeps_live_ones(self):
        restorer, _ = self.restorer(self.tier, poison=False)
        for head in self.heads.values():  # binds every pack (store loads)
            self.restore(restorer, head)
        self.assertEqual(self.tier.stats()["bindings"], 6)
        self.runtime.delete_lineage("b")
        self.runtime.collect_garbage()
        report = restorer.forget_unstored()
        self.assertEqual(report["decoded_tier"]["entries"], 2)
        stats = self.tier.stats()
        self.assertEqual((stats["entries"], stats["bindings"]), (4, 4))
        self.set_layer_model(0.0)
        caches, report = self.restore(restorer, self.heads["a"])
        _same(self, caches, self.reference["a"])
        self.assertEqual(report["decoded_tier"]["routed_nodes"], 4)

class BudgetTests(unittest.TestCase):
    model = ModelIdentity("accel-a", "fp", 1, 8, "tokens_channels", "float32")

    def tier(self, budget, **options):
        tier = DecodedTier(budget, device="cpu", model=self.model, **options)
        self.addCleanup(tier.close)
        return tier

    @staticmethod
    def offer(tier, key, value=1, model=None):
        pictures = np.full((2, 3, 32, 32), value, np.uint8)
        return tier.offer(key, pictures, [{"pictures": [0, 2], "rect": [0, 0, 32, 16]}], model=model)

    def test_shared_streams_stay_until_their_last_pack_is_forgotten(self):
        tier = self.tier(1 << 20)
        self.assertTrue(tier.offer("k" * 64, np.ones((2, 3, 32, 32), np.uint8),
                                   [{"pictures": [0, 2], "rect": [0, 0, 16, 32]}]))
        tier.bind("p1", "k" * 64)
        tier.bind("p2", "k" * 64)
        self.assertEqual(tier.forget_packets(["p1"])["entries"], 0)
        self.assertEqual(tier.stats()["entries"], 1)
        self.assertEqual(tier.forget_packets(["p2"])["entries"], 1)
        self.assertEqual(tier.stats()["entries"], 0)

    def test_lru_eviction_within_the_budget(self):
        tier = self.tier(2 * 4096 + 100)  # room for two entries (3072 bytes, in 4 KiB slab units)
        for key in ("a", "b"):
            self.assertTrue(self.offer(tier, key))
        self.assertTrue(self.offer(tier, "a"))  # a duplicate touches "a": "b" is now least recently used
        self.assertTrue(self.offer(tier, "c"))
        self.assertEqual(list(tier._entries), ["a", "c"])
        stats = tier.stats()
        self.assertEqual((stats["evictions"], stats["fill_duplicates"], stats["freed"]), (1, 1, 1))
        self.assertLessEqual(stats["resident_bytes"], stats["budget_bytes"])
        self.assertTrue(torch.equal(tier._entries["a"].view(0), torch.ones((2, 3, 16, 32), dtype=torch.uint8)))

    def test_slab_first_fit_with_coalescing_frees(self):
        from reelcache.decoded_tier import _Slab

        slab = _Slab(5 * 4096 + 7, pinned=False)
        self.assertEqual(slab.size, 5 * 4096)
        a, b, c = slab.allocate(100), slab.allocate(2 * 4096), slab.allocate(4096)
        self.assertEqual((a, b, c), (0, 4096, 3 * 4096))
        slab.free(b, 2 * 4096)
        self.assertIsNone(slab.allocate(3 * 4096))  # holes of 2 + 1 units, not adjacent
        slab.free(a, 100)  # coalesces with the freed range
        self.assertEqual(slab.allocate(3 * 4096), 0)
        slab.free(c, 4096)
        slab.free(0, 3 * 4096)
        self.assertEqual(slab._free, [(0, 5 * 4096)])
        self.assertEqual(slab.free_bytes(), 5 * 4096)

    def test_too_large_foreign_and_malformed_offers_are_skipped_never_raised(self):
        tier = self.tier(100)
        self.assertFalse(self.offer(tier, "big"))
        other = ModelIdentity("other", "fp", 1, 8, "tokens_channels", "float32")
        self.assertFalse(self.tier(1 << 20).offer("k", np.zeros((1, 3, 32, 32), np.uint8),
                                                   [{"pictures": [0, 1], "rect": [0, 0, 32, 32]}], model=other))
        bad = self.tier(1 << 20)
        self.assertFalse(bad.offer("k", np.zeros((1, 3, 32, 32), np.uint8),
                                   [{"pictures": [0, 2], "rect": [0, 0, 8, 8]}]))
        stats = bad.stats()
        self.assertEqual(stats["fill_errors"], 1)
        self.assertIn("IntegrityError", stats["last_fill_error"])
        self.assertEqual(tier.stats()["fill_skipped_large"], 1)

    def test_leased_entries_are_neither_evicted_nor_freed_until_released(self):
        tier = self.tier(4096 + 100)  # room for one entry
        self.assertTrue(self.offer(tier, "a"))
        entry = tier._entries["a"]
        entry.pins += 1  # a restore in flight leases it
        self.assertFalse(self.offer(tier, "b"))  # no room next to the leased entry
        self.assertEqual(tier.stats()["fill_skipped_full"], 1)
        self.assertEqual(tier.forget_packets([]), {"entries": 0, "bytes": 0})
        tier.bind("p", "a")
        tier.forget_packets(["p"])  # evicted while leased: kept alive (retiring) until released
        self.assertIsNotNone(entry.buffer)
        self.assertEqual(tier.stats()["retiring"], 1)
        tier.release({"_entries": [entry]})
        self.assertIsNone(entry.buffer)
        self.assertEqual(tier.stats()["retiring"], 0)

    def test_closed_tier_takes_no_offers_and_leaves_the_registry(self):
        tier = DecodedTier(1 << 20, device="cpu")
        self.assertIn(tier, dt.open_tiers())
        self.assertEqual(offer_decoded("k", np.zeros((1, 3, 32, 32), np.uint8),
                                       [{"pictures": [0, 1], "rect": [0, 0, 32, 32]}]), 1)
        tier.close()
        self.assertNotIn(tier, dt.open_tiers())
        self.assertFalse(self.offer(tier, "x"))
        self.assertEqual(tier.stats()["entries"], 0)
        self.assertIsNone(tier.route([], SimpleNamespace(jobs=[SimpleNamespace(submitted=False, node_id="n",
                                                                               nbytes=1)], layers=[0])))

    def test_layer_compute_defaults_and_overrides(self):
        self.assertEqual(dt.layer_compute_model("raven")[:3], dt.LAYER_COMPUTE_MS["raven"])
        self.assertEqual(dt.layer_compute_model("longlive2")[:3], dt.LAYER_COMPUTE_MS["longlive2"])
        self.assertEqual(dt.layer_compute_model("longlive1")[:3], dt.LAYER_COMPUTE_MS["longlive1"])
        self.assertTrue(dt.layer_compute_model("mystery")[3].startswith("default:causvid"))
        self.assertEqual(dt.layer_compute_model("raven", (1, 2))[:3], (1.0, 2.0, None))
        tier = self.tier(1 << 20, layer_ms=(1.0, 0.5))
        self.assertEqual(tier.layer_compute_ms(8), 5.0)
        windowed = DecodedTier(1 << 20, device="cpu", model=ModelIdentity("longlive2", "fp", 1, 8, "t", "float32"))
        self.addCleanup(windowed.close)
        self.assertEqual(windowed.layer_compute_ms(30), windowed.layer_compute_ms(3))
        for bad in ({"route": "some"}, {"nvdec_gbps": 0}, {"target": 0}):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                DecodedTier(1 << 20, device="cpu", **bad)
        with self.assertRaises(ConfigurationError):
            DecodedTier(0, device="cpu")
        with self.assertRaises(ConfigurationError):
            LineageRestorer(None, SoftwarePool(), decoded_tier=object())


class ConfigTests(unittest.TestCase):
    def test_fields_defaults_and_validation(self):
        config = ReelCacheRuntimeConfig(store_root="/tmp/accel-a")
        self.assertEqual((config.decoded_tier_bytes, config.decoded_tier_route, config.decoded_tier_layer_ms),
                         (0, "deficit", None))
        data = config.to_dict()
        tuned = ReelCacheRuntimeConfig.from_dict({**data, "decoded_tier_bytes": 8 << 30, "decoded_tier_route": "all",
                                               "decoded_tier_layer_ms": [2.5, 0.5]})
        self.assertEqual((tuned.decoded_tier_bytes, tuned.decoded_tier_route, tuned.decoded_tier_layer_ms),
                         (8 << 30, "all", (2.5, 0.5)))
        self.assertEqual(ReelCacheRuntimeConfig.from_dict(tuned.to_dict()), tuned)
        for bad in ({"decoded_tier_bytes": -1}, {"decoded_tier_bytes": True}, {"decoded_tier_bytes": 1.5},
                    {"decoded_tier_route": "oldest"}, {"decoded_tier_layer_ms": [1.0]},
                    {"decoded_tier_layer_ms": [1.0, -1.0]}, {"decoded_tier_layer_ms": "1,2"},
                    {"decoded_tier_layer_ms": [True, 1]}, {"decoded_tier_layer_ms": [float("nan"), 1]},
                    {"decoded_tier_bytes": 1 << 30, "validate_decode": False}):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                ReelCacheRuntimeConfig(store_root="/tmp/accel-a", **bad)


if __name__ == "__main__":
    unittest.main()
