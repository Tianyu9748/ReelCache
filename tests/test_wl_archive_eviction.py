"""W1.4.3 (E1): LRU eviction of the compressed archive for the ablation arms (``archive_budget_bytes``, ``reuse=False``)
on the CPU toy harness (the Step 6 lineage archive, svd_hevc, software decoder).

* A node's payload (packet payload blobs, basis row blocks) is evicted WHOLE, least recently used first (access:
  publication, or a restore reading it from the archive); manifests, refs, latents and execution state stay.
* ``live_bytes + reserved_bytes`` never exceeds the budget: commits reserve before their writes reach the store.
* A restore whose attended history misses an evicted node installs the deepest available ancestor through the archive's
  normal path and replays the rest from the stored clean latents: (ii) the installed ancestor equals a normal archive
  restore of it; (i) the replayed part equals a replay of the same steps from that installed state, and a full replay
  when the ancestor came from the raw tiers (exact).
* Whole-tree delete (and the warm-up purge) leaves nothing; a reopened archive knows what it evicted; identical
  content written again restores a node.
"""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import tests.test_sglang_bridge as tsb
import tests.test_wl_tree as tree_module
from reelcache.errors import CapacityError, ConfigurationError
from reelcache.integrations.sglang import ReelCacheRuntimeConfig
from reelcache.integrations.sglang import bridge as bridge_module
from reelcache.lineage_commit import identity_object
from reelcache.store import EvictedObjectError
from tests.test_sglang_bridge import CHANNELS, handle
from tests.test_wl_tree import Chunk, TreeStage, TreeTestCase, assert_caches_bitwise, run_tree, toy_forward

PATHS = {"A": "A", "B": "AB", "C": "ABC", "D": "ABCD", "E": "ABE", "F": "ABCDF"}
HUGE = 1 << 40


def snapshot(caches) -> dict:
    """Every layer's filled slots (the toy caches are full history: all of them are attended)."""
    for items in caches.values():
        for cache in items:
            cache.await_restored_history()
    return {stream: [(cache.k[:, :cache.end].clone(), cache.v[:, :cache.end].clone(), cache.end) for cache in items]
            for stream, items in caches.items()}


def until(predicate, timeout: float = 30.0, step: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(step)
    return True


def physical_payload_bytes(store) -> int:
    """The KV payload bytes physically in the store now -- what ``live_bytes + reserved_bytes`` must never understate
    and the budget bounds: every basis row block and every packet payload blob indexed and not evicted (the sidecar --
    manifests, records, execution-state blobs -- is not counted)."""
    with store._lock:
        rows = dict(store._db.execute("SELECT id, size FROM objects WHERE kind IN ('blob', 'basis')").fetchall())
        manifests = [row[0] for row in store._db.execute("SELECT id FROM objects WHERE kind = 'manifest'")]
        evicted = set(store._evicted)
    payloads = set()
    for ref in manifests:
        data = store.get_manifest(ref)
        if data.get("type") == "packet":
            payloads.update((data.get("payloads") or {}).values())
    return sum(size for object_id, size in rows.items()
               if object_id not in evicted and (object_id.startswith("basis:") or object_id in payloads))


def same(test, left: dict, right: dict) -> None:
    test.assertEqual(set(left), set(right))
    for stream in left:
        for (k1, v1, e1), (k2, v2, e2) in zip(left[stream], right[stream]):
            test.assertEqual(e1, e2)
            test.assertTrue(torch.equal(k1, k2) and torch.equal(v1, v2))


class Capture:
    """The caches at the replay's begin (the installed ancestor) and end (the replayed history)."""

    def __init__(self, test):
        self.begin = self.end = None
        real_begin, real_end = bridge_module._BridgeRun.begin_replay, bridge_module._BridgeRun.end_replay
        capture = self

        def begin(run, *args, **kwargs):
            plan = real_begin(run, *args, **kwargs)  # the guards are lifted: the caches are readable
            capture.begin = snapshot({stream: run.target.caches[stream] for stream in run.request.streams})
            return plan

        def end(run, *args, **kwargs):
            capture.end = snapshot({stream: run.target.caches[stream] for stream in run.request.streams})
            return real_end(run, *args, **kwargs)
        patcher = mock.patch.multiple(bridge_module._BridgeRun, begin_replay=begin, end_replay=end)
        patcher.start()
        test.addCleanup(patcher.stop)


class EvictionCase(TreeTestCase):
    representation = "svd_hevc"

    def evicting(self, store="evicting", budget=HUGE, **overrides):
        values = {"reuse": False, "archive_budget_bytes": budget, "raw_tier": False, **overrides}
        return self.bridge(store, **values)

    def chain(self, bridge, letters="ABCDEF", stage=None):
        """A linear session (one new chunk per request, resident continuations: no restore touches the LRU)."""
        stage = stage or TreeStage()
        nodes = {}
        for depth, letter in enumerate(letters, start=1):
            run_tree(bridge, self.request_for(handle("s", "generate" if depth == 1 else "resume"), letters[:depth]),
                     stage=stage)
            nodes[letter] = self.head(bridge)
        bridge.checkpoint("s")
        return nodes

    def evict(self, bridge, *refs):
        eviction = bridge.archive.eviction
        with eviction._cond:
            for ref in refs:
                eviction._evict(ref)

    @staticmethod
    def set_budget(bridge, budget: int) -> None:
        """The budget for the next phase; the peak restarts with it (as the E1 smoke's calibration)."""
        eviction = bridge.archive.eviction
        with eviction._cond:
            eviction.budget = int(budget)
            eviction.peak_bytes = eviction.live_bytes + eviction.reserved_bytes
            eviction._cond.notify_all()

    def evict_now(self, bridge, ref) -> bool:
        """One targeted eviction attempt (all or nothing): whether ``ref`` went."""
        eviction = bridge.archive.eviction
        with eviction._cond:
            return eviction._evict(ref)

    def hevc_geometry(self):
        """The HEVC-only arm: its identity basis needs a chunk of at least as many tokens as channels (64) -- the toy
        chunk at 96 tokens (as tests.test_wl_baselines)."""
        for module in (tsb, tree_module):
            patcher = mock.patch.multiple(module, GRID=(6, 8), TPF=48)
            patcher.start()
            self.addCleanup(patcher.stop)

    def child(self, bridge, node, letter, child="G"):
        return run_tree(bridge, self.request_for(handle("s", "rewind", checkpoint=node), PATHS[letter] + child))

    def node_bytes(self, bridge, ref) -> int:
        payload = bridge.archive.eviction.node_payload(ref)
        return sum(bridge.archive.store.object_sizes(payload.objects).values())


class ArchiveEvictionConfigTests(unittest.TestCase):
    base = {"store_root": "x", "representation": "svd_hevc", "reuse": False}

    def test_budget_needs_a_compressed_arm_without_reuse(self):
        config = ReelCacheRuntimeConfig.from_dict({**self.base, "archive_budget_bytes": 1 << 30})
        self.assertEqual(config.archive_budget_bytes, 1 << 30)
        self.assertEqual(config.to_dict()["archive_budget_bytes"], 1 << 30)
        self.assertNotIn("archive_budget_bytes", ReelCacheRuntimeConfig.from_dict(self.base).to_dict())
        for bad in ({"reuse": True}, {"representation": "raw"},
                    {"representation": "latents_only", "restore_mode": "recompute"}, {"archive_budget_bytes": 0},
                    {"archive_budget_bytes": -1}, {"archive_budget_bytes": 1.5}):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                ReelCacheRuntimeConfig.from_dict({**self.base, "archive_budget_bytes": 1 << 30, **bad})
        with self.assertRaises(ConfigurationError) as caught:  # ReelCache (reuse=True) never evicts
            ReelCacheRuntimeConfig.from_dict({"store_root": "x", "representation": "svd_hevc",
                                           "archive_budget_bytes": 1 << 30})
        self.assertIn("reuse=False", str(caught.exception))

    def test_compressed_arms_promote_only_when_they_evict(self):
        ReelCacheRuntimeConfig.from_dict({**self.base, "archive_budget_bytes": 1 << 30, "raw_tier_promote": "host"})
        for representation in ("svd_hevc", "svd", "hevc"):
            with self.subTest(representation=representation), self.assertRaises(ConfigurationError):
                ReelCacheRuntimeConfig.from_dict({**self.base, "representation": representation,
                                               "raw_tier_promote": "host"})
        with self.assertRaises(ConfigurationError):  # ReelCache rejects both
            ReelCacheRuntimeConfig.from_dict({"store_root": "x", "representation": "svd_hevc",
                                           "raw_tier_promote": "host", "archive_budget_bytes": 1 << 30})


class ArchiveEvictionTests(EvictionCase):
    def test_lru_order_and_the_budget_hold_including_in_flight_writes(self):
        probe = self.evicting("probe")
        sizes = {letter: self.node_bytes(probe, ref) for letter, ref in self.chain(probe).items()}
        budget = int(3.5 * max(sizes.values()))
        bridge = self.evicting(budget=budget)
        eviction = bridge.archive.eviction
        seen = []
        real = eviction.reserve

        def reserve(key, nbytes):  # the invariant at every write: live + reserved within the budget
            real(key, nbytes)
            seen.append(eviction.live_bytes + eviction.reserved_bytes)
        with mock.patch.object(eviction, "reserve", reserve):
            nodes = self.chain(bridge)
        order = [nodes[letter] for letter in "ABCDEF"]
        evicted = [ref for ref in order if eviction.is_evicted(ref)]
        self.assertEqual(evicted, order[:len(evicted)])  # least recently published first
        self.assertGreaterEqual(len(evicted), 2)
        stats = eviction.stats()
        self.assertTrue(seen and max(seen) <= budget)
        self.assertLessEqual(stats["peak_bytes"], budget)
        # every write was reserved before the store saw it: each node's basis rows and its packet payloads, once
        self.assertEqual((stats["reservations"], stats["reserved_total_bytes"]), (2 * len(sizes), sum(sizes.values())))
        self.assertEqual(stats["evict"]["nodes"], len(evicted))
        self.assertEqual(stats["evict"]["bytes"], sum(sizes[letter] for letter, ref in nodes.items()
                                                      if ref in evicted))
        live = [ref for ref in order if ref not in evicted]
        self.assertEqual(stats["live_bytes"], sum(sizes[letter] for letter, ref in nodes.items() if ref in live))
        store = bridge.archive.store
        for ref in evicted:  # the whole payload left the store; the node (manifest, latents, state) did not
            payload = eviction.node_payload(ref)
            self.assertEqual(store.evicted_ids(payload.objects), set(payload.objects))
            with self.assertRaises(EvictedObjectError):
                store.get(next(iter(payload.objects)))
            bridge.archive.chunk_latents(bridge.archive.load(ref))
            bridge.archive.execution_state(ref)
        # A restore that READS nodes from the archive makes them most recently used: with A..D live, B's restore
        # reads A and B, so E's publication (one node over the budget) evicts C, the least recently used.
        touched = self.evicting("touched", budget=int(4.5 * max(sizes.values())))
        nodes = self.chain(touched, "ABCD")
        restore = self.child(touched, nodes["B"], "B", child="E")[0]["restore"]
        self.assertEqual(restore["mode"], "archive")
        touched.checkpoint("s")
        eviction = touched.archive.eviction
        self.assertEqual({letter for letter, ref in nodes.items() if eviction.is_evicted(ref)}, {"C"})

    def test_an_evicted_node_restores_through_its_ancestor_and_a_replay(self):
        bridge = self.evicting()
        nodes, _ = self.build_a_to_f(bridge)
        bridge.checkpoint("s")
        self.evict(bridge, nodes["D"])
        capture = Capture(self)
        summary, latents, caches, _ = self.child(bridge, nodes["F"], "F")
        restore = summary["restore"]
        self.assertEqual((restore["mode"], restore["source"], restore["archive_evicted_hits"]),
                         ("recompute", "replay", 1))
        record = restore["recompute"]
        self.assertEqual((record["cached_ancestor_depth"], record["installed_nodes"], record["replayed_nodes"],
                          record["anchor"]), (3, 3, 2, nodes["C"]))
        self.assertEqual(restore["replay"]["chunk_forwards"], 2)
        # the anchor restore decodes: it gets the pre-restore relief (and the section) as a normal archive restore
        self.assertIsNotNone(restore["pressure"]["estimate"])
        self.assertNotIn("skipped", restore["pressure"])
        self.assertEqual(record["anchor_source"], "compressed")  # C installed from the (lossy) archive
        installed, replayed = capture.begin, capture.end
        # (ii) the installed ancestor is exactly a normal archive restore of C (same archive, same decoder)
        mode, reference = self.restored(bridge, nodes["C"], "C")
        self.assertEqual(mode, "archive")
        same(self, installed, reference)
        # (i) the replayed part is a replay of D and F from that installed state -- not a clean full replay (the
        # anchor is lossy; review C1: the intended semantics)
        same(self, replayed, self.toy_replay(installed, "ABCDF", [3, 4]))
        stats = bridge.archive.eviction.stats()
        self.assertEqual((stats["misses"], stats["evicted_hits"]), (1, 1))

    def restored(self, bridge, node, letter):
        """The restore mode and the caches right after a normal restore of ``node`` (before any new chunk)."""
        real = bridge_module._BridgeRun.restore
        out = {}

        def restore(run, target):
            done = real(run, target)
            out["caches"] = snapshot({stream: target.caches[stream] for stream in run.request.streams})
            return done
        with mock.patch.object(bridge_module._BridgeRun, "restore", restore):
            summary = self.child(bridge, node, letter, child="H")[0]
        return summary["restore"]["mode"], out["caches"]

    def toy_replay(self, installed, letters, blocks):
        """The toy model's clean forwards of ``blocks`` on top of an installed snapshot (the replay's reference)."""
        from tests.test_wl_tree import GuardedCache, PROMPTS
        from tests.test_sglang_bridge import FRAMES, TPF

        caches = {}
        for stream, items in installed.items():
            caches[stream] = []
            for k, v, end in items:
                cache = GuardedCache()
                cache.k[:, :end] = k
                cache.v[:, :end] = v
                cache.end = end
                caches[stream].append(cache)
        for block in blocks:
            start = block * FRAMES
            chunk = Chunk(block, start, FRAMES, start * TPF, FRAMES * TPF, float(start), False, 0)
            toy_forward(caches, list(caches), PROMPTS[letters[block]], chunk)
        return snapshot(caches)

    def test_an_evicted_node_in_a_raw_tier_is_restored_from_it(self):
        """Availability includes the raw tiers: an evicted node whose raw copy is still cached is no miss."""
        bridge = self.evicting(raw_tier=True, gpu_raw_budget_bytes=1 << 30, host_raw_budget_bytes=1 << 30)
        nodes, _ = self.build_a_to_f(bridge)
        bridge.checkpoint("s")
        bridge.archive.raw_tier.flush()
        self.evict(bridge, nodes["D"])
        restore = self.child(bridge, nodes["F"], "F")[0]["restore"]
        self.assertEqual((restore["mode"], restore["archive_evicted_hits"]), ("archive", 0))
        tiers = {item["node"]: item["tier"] for item in restore["lease"]["tiers"]["per_chunk"]}
        self.assertIn(tiers[nodes["D"]], ("gpu_raw", "host_raw"))

    def test_raw_tier_ancestors_make_the_replay_exact(self):
        """With the ancestor's visible set in the raw tiers (exact), the selective replay is bitwise the RAW restore."""
        bridge = self.evicting(raw_tier=True, gpu_raw_budget_bytes=1 << 30, host_raw_budget_bytes=1 << 30)
        nodes, _ = self.build_a_to_f(bridge)
        bridge.checkpoint("s")
        tier = bridge.archive.raw_tier
        tier.flush()
        tier.drop(tier.entries(node_refs=[nodes["D"], nodes["F"]]))
        self.evict(bridge, nodes["D"])
        got = self.child(bridge, nodes["F"], "F")
        record = got[0]["restore"]["recompute"]
        self.assertEqual((record["cached_ancestor_depth"], record["anchor"], record["anchor_source"]),
                         (3, nodes["C"], "raw"))
        self.assertTrue(all(item["tier"] in ("gpu_raw", "host_raw")
                            for item in got[0]["restore"]["lease"]["tiers"]["per_chunk"]))
        raw = self.bridge("raw", representation="raw")
        raw_nodes, _ = self.build_a_to_f(raw)
        expected = self.child(raw, raw_nodes["F"], "F")
        self.assertTrue(torch.equal(got[1], expected[1]))
        assert_caches_bitwise(self, got[2], expected[2])

    def test_mid_path_and_root_evictions(self):
        bridge = self.evicting()
        nodes, _ = self.build_a_to_f(bridge)
        bridge.checkpoint("s")
        self.evict(bridge, nodes["B"])  # full history: every candidate from B on attends B
        record = self.child(bridge, nodes["F"], "F")[0]["restore"]["recompute"]
        self.assertEqual((record["cached_ancestor_depth"], record["anchor"], record["replayed_nodes"],
                          record["anchor_source"]), (1, nodes["A"], 4, "compressed"))
        self.evict(bridge, nodes["A"])  # the first chunk: no ancestor, a full replay
        restore = self.child(bridge, nodes["E"], "E")[0]["restore"]
        self.assertEqual((restore["recompute"]["cached_ancestor_depth"], restore["recompute"]["anchor"],
                          restore["replay"]["chunk_forwards"], restore["archive_evicted_hits"],
                          restore["recompute"]["anchor_source"]), (0, None, 3, 2, "none"))
        self.assertIsNone(restore["pressure"])  # nothing restored

    def test_an_evicted_parent_takes_children_and_identical_content_restores_it(self):
        bridge = self.evicting()
        eviction, store = bridge.archive.eviction, bridge.archive.store
        nodes = self.chain(bridge, "ABCD")
        payload = eviction.node_payload(nodes["D"])
        saved = {object_id: store.get(object_id) for object_id in payload.objects}
        self.evict(bridge, nodes["D"])
        # D's child: D replayed from C; the child's commit plans from the session's in-memory state of D (D is the
        # head: its rows are not read again; a parent loaded from the store: the restart test below)
        summary = self.child(bridge, nodes["D"], "D", child="G")[0]
        self.assertEqual(summary["restore"]["recompute"]["anchor"], nodes["C"])
        child = bridge.checkpoint("s")["head"]
        self.assertFalse(eviction.is_evicted(child))
        restore = run_tree(bridge, self.request_for(handle("s", "rewind", checkpoint=child), "ABCDGH"))[0]["restore"]
        self.assertEqual((restore["recompute"]["anchor"], restore["replay"]["chunk_forwards"]), (nodes["C"], 2))
        # Identical content written again (a deterministic regeneration; here D's own bytes) restores D: the store
        # makes the payload readable again and the publication makes the node live again.
        for object_id, data in saved.items():
            store.put(data, kind=object_id.split(":")[0])
        self.assertEqual(store.evicted_ids(payload.objects), set())
        eviction.published(object(), nodes["D"])
        self.assertFalse(eviction.is_evicted(nodes["D"]))
        self.assertEqual(eviction.stats()["republished"], 1)
        self.assertEqual(self.child(bridge, nodes["D"], "D")[0]["restore"]["mode"], "archive")

    def test_whole_tree_delete_and_the_warmup_purge_leave_nothing(self):
        bridge = self.evicting()
        store, eviction = bridge.archive.store, bridge.archive.eviction
        empty = (store.list_refs(""), store.stats()["object_count"])
        warm = self.chain(bridge, "ABC")  # the warm-up session ("s"), partly evicted
        self.evict(bridge, warm["B"])
        bridge.control("delete", {"session_id": "s"})
        self.assertEqual((store.list_refs(""), store.stats()["object_count"]), empty)
        self.assertEqual(store.evicted_ids(), set())
        stats = eviction.stats()
        self.assertEqual((stats["live_bytes"], stats["live_nodes"], stats["evicted_nodes"]), (0, 0, 0))
        self.assertNotIn("evicted_objects", store.stats())

    def test_a_reopened_archive_knows_what_it_evicted(self):
        bridge = self.evicting("reopen")
        nodes, _ = self.build_a_to_f(bridge)
        bridge.checkpoint("s")
        self.evict(bridge, nodes["D"])
        live = bridge.archive.eviction.stats()["live_bytes"]
        bridge.close()
        again = self.evicting("reopen")
        eviction = again.archive.eviction
        self.assertTrue(eviction.is_evicted(nodes["D"]))
        self.assertEqual(eviction.stats()["live_bytes"], live)
        record = self.child(again, nodes["F"], "F")[0]["restore"]["recompute"]
        self.assertEqual(record["anchor"], nodes["C"])

    def test_an_evicting_arm_plans_every_commit_without_a_parent_basis(self):
        """Review I5: an evicting arm (no basis reuse) plans every commit as one with no active basis, so no commit
        depends on another node's payload. A zero matrix of a child whose evicted parent is still in memory (the
        lineage's last job) fits its own one-row basis instead of reusing the parent's (evicted) version -- W1.4.3 v1
        reused it and the child was registered evicted the moment it published; and no parent rows are ever
        assembled from the store (a rewind to an older, evicted node; a restart)."""
        bridge = self.evicting("restart")
        nodes = self.chain(bridge, "ABCDE")
        eviction, store = bridge.archive.eviction, bridge.archive.store
        assembled = []

        def spied(target):
            runtime = target.archive.pipeline.runtime
            real = runtime.assemble_parent_rows
            return mock.patch.object(runtime, "assemble_parent_rows",
                                     lambda parent: assembled.append(parent) or real(parent))
        # in memory: E is the head (its commit is the lineage's last job), evicted; F has a zero matrix
        self.evict(bridge, nodes["E"])
        real_kv = tree_module.toy_kv

        def kv(prompt, chunk, layer, stream, cache):
            key, value = real_kv(prompt, chunk, layer, stream, cache)
            return (key, torch.zeros_like(value)) if chunk.index == 5 and layer == 0 else (key, value)
        with spied(bridge), mock.patch.object(tree_module, "toy_kv", kv):
            run_tree(bridge, self.request_for(handle("s", "resume"), "ABCDEF"))
            zero_child = bridge.checkpoint("s")["head"]
        payload = eviction.node_payload(zero_child)
        self.assertFalse(eviction.is_evicted(zero_child))
        self.assertEqual(store.evicted_ids(payload.objects), set())
        self.assertFalse(set(payload.versions) & set(eviction.node_payload(nodes["E"]).versions))
        # from the store: a rewind to C, an older evicted node
        self.evict(bridge, nodes["C"], nodes["D"])
        with spied(bridge):
            summary = self.child(bridge, nodes["C"], "C", child="G")[0]
            child = bridge.checkpoint("s")["head"]
        self.assertEqual(summary["restore"]["recompute"]["anchor"], nodes["B"])
        self.assertNotIn(child, (nodes["C"], None))
        self.assertFalse(eviction.is_evicted(child))
        bridge.close()
        again = self.evicting("restart")  # a restart: D's state comes from the store
        with spied(again):
            summary = self.child(again, nodes["D"], "D", child="H")[0]
            child = again.checkpoint("s")["head"]
        self.assertEqual(assembled, [])  # no parent's rows were ever read
        self.assertEqual(summary["restore"]["recompute"]["anchor"], nodes["B"])
        self.assertNotIn(child, (nodes["D"], None))
        self.assertTrue(again.archive.eviction.is_evicted(nodes["D"]))
        self.assertFalse(again.archive.eviction.is_evicted(child))
        restore = run_tree(again, self.request_for(handle("s", "rewind", checkpoint=child), "ABCDHG"))[0]["restore"]
        self.assertEqual((restore["recompute"]["anchor"], restore["replay"]["chunk_forwards"]), (nodes["B"], 3))

    def test_replayed_nodes_are_promoted_not_the_installed_ones(self):
        bridge = self.evicting(raw_tier=True, gpu_raw_budget_bytes=1 << 30, host_raw_budget_bytes=1 << 30,
                               raw_tier_retain="lru", raw_tier_promote="host")
        self.assertEqual(bridge.describe()["raw_tier_policy"]["promotion_scope"], "replay")
        nodes, _ = self.build_a_to_f(bridge)
        bridge.checkpoint("s")
        bridge.control("raw_tier", {"action": "drop"})  # the archive serves (lossy); D is evicted
        self.evict(bridge, nodes["D"])
        capture = Capture(self)
        restore = self.child(bridge, nodes["F"], "F")[0]["restore"]
        self.assertEqual(restore["recompute"]["anchor"], nodes["C"])
        self.assertEqual(restore["promotion"]["nodes"], 2)  # D and F (replayed); A, B, C came from the archive
        tier = bridge.archive.raw_tier
        tier.flush()
        promoted = {entry.node_ref for entry in tier.entries() if entry.origin == "replay"}
        self.assertEqual(promoted, {nodes["D"], nodes["F"]})
        # D and F from host raw -- installed-form rows (post-RoPE K) copied as they are, never rotated again -- and
        # A, B, C from the archive: bitwise the replay's output, on the plain and on the overlapped restore
        for plain in (True, False):
            with self.subTest(plain=plain):
                again, caches = self.restored_rows(bridge, nodes["F"], "F", plain=plain)
                self.assertEqual((again["mode"], again["archive_evicted_hits"]), ("archive", 0))
                served = {(item["node"], item["tier"], item.get("origin"), item.get("form"))
                          for item in again["lease"]["tiers"]["per_chunk"] if item["node"] in promoted}
                self.assertEqual(served, {(ref, "host_raw", "replay", "installed") for ref in promoted})
                same(self, caches, capture.end)

    def restored_rows(self, bridge, node, letter, *, plain):
        """The restore record and the caches right after a normal restore of ``node`` (``plain``: not
        layer-overlapped; else overlapped, its layers awaited)."""
        real = bridge_module._BridgeRun.restore
        out = {}

        def restore(run, target):
            done = real(run, target)
            out["caches"] = snapshot({stream: target.caches[stream] for stream in run.request.streams})
            return done
        session = handle("s", "rewind", checkpoint=node, **({"overlap_restore": False} if plain else {}))
        with mock.patch.object(bridge_module._BridgeRun, "restore", restore):
            summary = run_tree(bridge, self.request_for(session, PATHS[letter] + "H"))[0]
        return summary["restore"], out["caches"]

    def test_a_budget_smaller_than_a_write_fails_typed(self):
        bridge = self.evicting(budget=64)
        with self.assertRaises(Exception) as caught:
            self.chain(bridge, "AB")
        text = f"{caught.exception} {caught.exception.__cause__}"
        self.assertIn("archive_budget_bytes", text)

    def test_commits_in_flight_beyond_the_budget_fail_typed_instead_of_deadlocking(self):
        """Two commits in flight whose reservations fill a tiny budget: nothing published can be evicted and neither
        can publish, so one fails at once (typed) and the other proceeds -- no wait forever, no budget overrun."""
        probe = self.evicting("probe")
        size = max(self.node_bytes(probe, ref) for ref in self.chain(probe, "AB").values())
        bridge = self.evicting(budget=int(1.3 * size))
        eviction = bridge.archive.eviction
        eviction.wait_seconds = 5.0  # without the detection a writer would wait this long, then fail differently
        stage = TreeStage()
        run_tree(bridge, self.request_for(handle("s"), "A"), stage=stage)
        run_tree(bridge, self.request_for(handle("s", "resume"), "AB"), stage=stage)  # both in flight
        with self.assertRaises(Exception) as caught:  # B's archival fails, typed at its cause; A published
            bridge.checkpoint("s")
        self.assertIn("archive_budget_bytes is below the commits in flight",
                      f"{caught.exception} {caught.exception.__cause__}")
        self.assertLessEqual(eviction.stats()["peak_bytes"], eviction.budget)

    def test_a_write_only_a_later_commit_could_make_room_for_fails_at_once(self):
        """The reservation deadlock beyond "every commit in flight waits": the first commit's packet payloads find
        nothing to evict while the NEXT commit of its session already holds its basis rows' reservation (the fitter
        ran ahead). That commit publishes only after the first (ordered publication), so nothing can make room: the
        write fails at once, typed and actionable, instead of waiting ``wait_seconds`` (300 s in a run) -- and the
        budget is never overrun. (The W1.4.3 E1 CPU smoke hit this at a calibrated budget, in one pass in three.)"""
        probe = self.evicting("probe")
        size = max(self.node_bytes(probe, ref) for ref in self.chain(probe, "AB").values())
        bridge = self.evicting(budget=int(1.3 * size))
        eviction, pipeline = bridge.archive.eviction, bridge.archive.pipeline
        eviction.wait_seconds = 60.0
        later_planned, stored = threading.Event(), []

        def hooks(stage, node_id):
            if stage == "finalize" and node_id == "c000001":
                stored.append(eviction.reserved_bytes)  # B's basis rows are reserved (and written)
                later_planned.set()
            if stage == "store" and node_id == "c000000":
                later_planned.wait(30)  # A writes its packet payloads only after B's rows are reserved
            if stage == "store" and node_id == "c000001":  # B writes its payloads once A's outcome is known (or
                deadline = time.monotonic() + 30  # after 30 s: waiting for room, B would then end the deadlock)
                while time.monotonic() < deadline and not pipeline.stats_.get("failed"):
                    time.sleep(0.01)
        pipeline.hooks = hooks
        stage = TreeStage()
        started = time.monotonic()
        run_tree(bridge, self.request_for(handle("s"), "A"), stage=stage)
        run_tree(bridge, self.request_for(handle("s", "resume"), "AB"), stage=stage)
        with self.assertRaises(Exception) as caught:
            bridge.checkpoint("s")
        self.assertLess(time.monotonic() - started, 20.0)  # at once, not after wait_seconds
        self.assertTrue(later_planned.is_set() and stored and stored[0] > 0)
        self.assertIn("archive_budget_bytes is below the commits in flight",
                      f"{caught.exception} {caught.exception.__cause__}")
        stats = eviction.stats()
        self.assertLessEqual(stats["peak_bytes"], eviction.budget)
        self.assertEqual((stats["reserved_bytes"], stats["live_nodes"], stats["reserve_waits"]), (0, 0, 0))

    def test_a_write_waits_for_a_commit_that_can_still_publish(self):
        """No false deadlock: a write that finds nothing to evict while another commit in flight is not waiting (it
        can still publish or end, which makes room) waits for it instead of failing."""
        from reelcache.integrations.sglang.archive_eviction import ArchiveEviction

        eviction = ArchiveEviction(None, 100, wait_seconds=30.0)
        first, second, outcome = object(), object(), {}
        eviction.reserve(first, 60)  # a commit in flight (writing, not waiting)

        def write():
            try:
                eviction.reserve(second, 60)
                outcome["reserved"] = True
            except CapacityError as exc:
                outcome["error"] = exc
        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        writer.join(0.5)
        self.assertTrue(writer.is_alive(), outcome)  # it waits for the first commit
        eviction.release(first)  # the first commit ends: room
        writer.join(10)
        self.assertEqual(outcome, {"reserved": True})
        self.assertEqual((eviction.reserved_bytes, eviction.stats()["reserve_waits"]), (60, 1))

    def test_a_writer_that_reserves_nothing_still_counts_as_in_flight(self):
        """Review round 2 (writes admitted by content address): a commit whose objects are all present reserves
        nothing, yet it is in flight -- its staging leases may keep a node from going until it publishes -- so a write
        that cannot fit meanwhile waits for it instead of failing at once."""
        from reelcache.integrations.sglang.archive_eviction import ArchiveEviction

        store = SimpleNamespace(indexed=lambda ids: set(ids), evicted_ids=lambda ids: set())  # everything present
        eviction = ArchiveEviction(store, 100, wait_seconds=30.0)
        duplicate, writer, outcome = SimpleNamespace(budget_order=("L", 1)), SimpleNamespace(), {}
        with eviction._cond:
            eviction.live_bytes = 50  # live, not evictable now (e.g. protected by the duplicate's staging leases)
        with eviction.writing(duplicate, {"blob:x": 40}):
            self.assertEqual((eviction.reserved_bytes, eviction._reserved), (0, {id(duplicate): 0}))

        def write():
            try:
                eviction.reserve(writer, 80)
                outcome["reserved"] = True
            except CapacityError as exc:
                outcome["error"] = exc
        thread = threading.Thread(target=write, daemon=True)
        thread.start()
        thread.join(0.5)
        self.assertTrue(thread.is_alive(), outcome)  # it waits for the duplicate (it can still publish)
        with eviction._cond:
            eviction.live_bytes = 0  # the duplicate published; the protected node went
            eviction._cond.notify_all()
        eviction.release(duplicate)
        thread.join(10)
        self.assertEqual(outcome, {"reserved": True})

    def test_a_write_waits_for_an_earlier_commit_of_its_lineage(self):
        """The deadlock rule follows publication order: a write that finds nothing to evict while an EARLIER commit of
        its lineage holds a reservation waits (that commit can still publish, which makes room); the same write of the
        EARLIER commit, with only a later one of its lineage holding room, fails at once (no publication can come)."""
        from reelcache.integrations.sglang.archive_eviction import ArchiveEviction

        eviction = ArchiveEviction(None, 100, wait_seconds=30.0)
        first, second = SimpleNamespace(budget_order=("L", 1)), SimpleNamespace(budget_order=("L", 2))
        other = SimpleNamespace(budget_order=("M", 1))
        eviction.reserve(first, 60)
        outcome = {}

        def write():
            try:
                eviction.reserve(second, 60)
                outcome["reserved"] = True
            except CapacityError as exc:
                outcome["error"] = exc
        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        writer.join(0.5)
        self.assertTrue(writer.is_alive(), outcome)  # it waits for the earlier commit
        eviction.release(first)
        writer.join(10)
        self.assertEqual(outcome, {"reserved": True})
        started = time.monotonic()  # the earlier commit of a lineage behind a later one's reservation: at once
        with self.assertRaisesRegex(CapacityError, "below the commits in flight"):
            eviction.reserve(SimpleNamespace(budget_order=("L", 1)), 60)
        self.assertLess(time.monotonic() - started, 5.0)
        eviction.reserve(other, 40)  # another lineage's commit holding room: an earlier one of L waits for it
        waiter = threading.Thread(target=lambda: outcome.update(late=eviction.reserve(first, 1)), daemon=True)
        waiter.start()
        waiter.join(0.5)
        self.assertTrue(waiter.is_alive())
        eviction.release(other)
        waiter.join(10)
        self.assertIn("late", outcome)
        self.assertEqual(eviction.reserved_bytes, 61)

    def test_a_restore_pins_what_it_reads_and_audits_skip_evicted_payloads(self):
        bridge = self.evicting()
        nodes = self.chain(bridge, "ABCD")
        eviction = bridge.archive.eviction
        self.evict(bridge, nodes["A"])
        bridge.archive.store.verify(nodes["D"])  # a full audit of a tree with an evicted payload passes
        restorer, seen = bridge.archive.restorer, []
        real = restorer.restore

        def restore(*args, **kwargs):  # mid-restore: the nodes it reads cannot be evicted
            seen.append(eviction.stats()["pinned_nodes"])
            return real(*args, **kwargs)
        with mock.patch.object(restorer, "restore", restore):
            self.child(bridge, nodes["B"], "B", child="E")  # B's history: A evicted -> full replay, nothing read
            self.child(bridge, nodes["D"], "D", child="E")  # D's: A evicted -> full replay
        self.assertEqual(seen, [])
        fresh = self.evicting("fresh")
        nodes = self.chain(fresh, "ABC")
        restorer = fresh.archive.restorer
        real = restorer.restore
        eviction = fresh.archive.eviction
        real_hold = fresh.archive.hold_available

        def unheld(refs):  # the bridge's availability hold out of the picture: the restore's own pins only
            available, hold = real_hold(refs)
            fresh.archive.release_available(hold)
            return available, None
        with mock.patch.object(restorer, "restore", restore), mock.patch.object(fresh.archive, "hold_available",
                                                                                unheld):
            self.child(fresh, nodes["C"], "C", child="D")
        self.assertTrue(seen and min(seen) == 3)  # A, B, C pinned while read
        self.assertEqual(eviction.stats()["pinned_nodes"], 0)  # and released with the lease

    def test_the_restore_holds_its_choice_until_it_pinned_what_it_reads(self):
        """Between a restore's choice (archive, raw copy or replay) and its reads nothing it chose can go: the path's
        archived nodes are pinned (a write evicts other nodes or waits) and an evicted node's raw copy is held (a drop
        waits); the restore releases the hold once it pinned what it reads (also when it fails: ``finish``)."""
        bridge = self.evicting(raw_tier=True, gpu_raw_budget_bytes=1 << 30, host_raw_budget_bytes=1 << 30)
        nodes = self.chain(bridge, "ABC")
        archive, eviction, tier = bridge.archive, bridge.archive.eviction, bridge.archive.raw_tier
        tier.flush()
        self.evict(bridge, nodes["B"])  # B's raw copy stays: available
        refs = [nodes[letter] for letter in "ABC"]
        available, hold = archive.hold_available(iter(refs))
        self.assertEqual(available, set(refs))
        eviction.budget = eviction.stats()["live_bytes"]  # full: any write must evict
        eviction.wait_seconds = 0.2
        with self.assertRaises(CapacityError):  # A and C are held: the write waits, then fails typed
            eviction.reserve(object(), 1)
        self.assertEqual(eviction.evicted(refs), {nodes["B"]})
        entry = tier.entries(node_refs=[nodes["B"]])[0]
        tier.drop([entry])  # B's held raw copy: the drop waits for the hold
        self.assertFalse(entry.dropped)
        archive.release_available(hold)
        self.assertTrue(entry.dropped)
        self.assertEqual(eviction.stats()["pinned_nodes"], 0)
        key = object()
        eviction.reserve(key, 1)  # released: the write evicts A, the least recently used
        self.assertEqual(eviction.evicted(refs), {nodes["A"], nodes["B"]})
        eviction.release(key)
        # a restore takes the hold and releases it: nothing stays pinned or held after the request
        seen = []
        real = archive.release_available

        def release(held):
            seen.append(held)
            return real(held)
        with mock.patch.object(archive, "release_available", release):
            restore = self.child(bridge, nodes["C"], "C", child="D")[0]["restore"]
        self.assertEqual(restore["recompute"]["anchor"], nodes["A"])  # B evicted, its raw copy dropped; A's held
        self.assertEqual(len(seen), 1)
        self.assertEqual(set(seen[0][0]) & set(refs), {nodes["C"]})  # the path's archived nodes were pinned
        self.assertEqual({entry.node_ref for entry in seen[0][1]}, {nodes["A"]})  # A's raw copy was held
        self.assertEqual(eviction.stats()["pinned_nodes"], 0)
        self.assertFalse(any(entry.pins.get("availability") for entry in tier.entries()))

    def test_preparations_pin_what_they_read_without_touching_the_lru(self):
        """Prepare-on-drop and a session prewarm read archived payloads: pinned while they read (no eviction under
        them), skipped for evicted nodes, and no LRU access (only publications and restores are)."""
        bridge = self.evicting()
        nodes = self.chain(bridge, "ABC")
        archive, eviction = bridge.archive, bridge.archive.eviction
        order, seen = list(eviction._nodes), []
        with mock.patch.object(archive.restorer, "ensure_prepared", create=True,
                               new=lambda packets: seen.append(eviction.stats()["pinned_nodes"])):
            archive._prepare_on_drop(nodes["A"])
            self.evict(bridge, nodes["B"])
            archive._prepare_on_drop(nodes["B"])  # evicted: nothing left to prepare
        self.assertEqual(seen, [1])
        self.assertEqual(list(eviction._nodes), [ref for ref in order if ref != nodes["B"]])  # A not touched
        self.assertEqual(eviction.stats()["pinned_nodes"], 0)
        seen.clear()
        with mock.patch.object(archive.restorer, "prewarm", create=True,
                               new=lambda history: seen.append(eviction.stats()["pinned_nodes"]) or 0.0):
            self.assertEqual(archive.prewarm(nodes["C"]), 0.0)  # B evicted: not prepared
            self.assertEqual(seen, [])
            fresh = self.evicting("fresh")
            fresh_nodes = self.chain(fresh, "AB")
            fresh_eviction = fresh.archive.eviction
            order = list(fresh_eviction._nodes)
            with mock.patch.object(fresh.archive.restorer, "prewarm", create=True,
                                   new=lambda history: seen.append(fresh_eviction.stats()["pinned_nodes"]) or 0.0):
                fresh.archive.prewarm(fresh_nodes["B"])
        self.assertTrue(seen and seen[0] >= 2)  # A and B pinned while prepared
        self.assertEqual(list(fresh_eviction._nodes), order)
        self.assertEqual(fresh_eviction.stats()["pinned_nodes"], 0)

    def test_a_pending_node_is_never_evicted_and_its_reservation_holds(self):
        """A node written but not yet published (its publication held at the pipeline's publish step) is no eviction
        candidate: its reservation stays counted, the writes it needed evicted the published node, and a further
        write that finds only it waits and fails typed instead of touching it. Published, it becomes live."""
        probe = self.evicting("probe")
        size = max(self.node_bytes(probe, ref) for ref in self.chain(probe, "AB").values())
        bridge = self.evicting(budget=int(1.5 * size))
        eviction, pipeline = bridge.archive.eviction, bridge.archive.pipeline
        reached, gate, calls = threading.Event(), threading.Event(), []

        def hooks(stage, node_id):
            if stage == "publish":
                calls.append(node_id)
                if len(calls) == 2:
                    reached.set()
                    gate.wait(60)
        pipeline.hooks = hooks
        stage = TreeStage()
        run_tree(bridge, self.request_for(handle("s"), "A"), stage=stage)
        node_a = bridge.checkpoint("s")["head"]  # A published: B is the only commit in flight
        try:
            run_tree(bridge, self.request_for(handle("s", "resume"), "AB"), stage=stage)
            self.assertTrue(reached.wait(60))  # B is stored, its publication is held
            stats = eviction.stats()
            self.assertTrue(eviction.is_evicted(node_a))  # B's reservations evicted the published A
            self.assertGreater(stats["reserved_bytes"], 0)  # B's reservation holds while it is pending
            self.assertEqual((stats["live_nodes"], stats["live_bytes"]), (0, 0))
            self.assertLessEqual(stats["live_bytes"] + stats["reserved_bytes"], eviction.budget)
            eviction.wait_seconds = 0.2
            with self.assertRaises(CapacityError):  # only the pending B could make room: it is never evicted
                eviction.reserve(object(), size)
            self.assertEqual(eviction.stats()["reserved_bytes"], stats["reserved_bytes"])
        finally:
            gate.set()
        node_b = bridge.checkpoint("s")["head"]
        stats = eviction.stats()
        self.assertEqual((stats["reserved_bytes"], stats["live_nodes"]), (0, 1))
        self.assertEqual(stats["live_bytes"], self.node_bytes(bridge, node_b))
        self.assertFalse(eviction.is_evicted(node_b))

    def test_pinned_nodes_are_never_evicted_and_a_write_waits_for_them(self):
        bridge = self.evicting()
        nodes = self.chain(bridge, "ABC")
        eviction = bridge.archive.eviction
        token = eviction.pin_available([nodes["A"]])
        sizes = {letter: self.node_bytes(bridge, ref) for letter, ref in nodes.items()}
        eviction.budget = sizes["A"] + sizes["B"] + sizes["C"]  # full
        key = object()
        eviction.reserve(key, sizes["B"])  # evicts B, the least recently used unpinned node (A is pinned)
        self.assertEqual((eviction.is_evicted(nodes["A"]), eviction.is_evicted(nodes["B"])), (False, True))
        eviction.release(key)
        eviction.wait_seconds = 0.2
        token2 = eviction.pin_available([nodes["C"]])
        with self.assertRaises(CapacityError):  # nothing unpinned to evict: it waits, then fails typed
            eviction.reserve(key, sizes["A"] + sizes["C"])
        eviction.unpin(token)
        eviction.unpin(token2)
        self.assertEqual(eviction.pin_available([nodes["B"]]), None)  # an evicted node cannot be pinned

    # -- W1.4.3 review round 1 ---------------------------------------------------------------------------------------
    def test_the_anchor_source_says_what_a_miss_equals(self):
        """Review C1 (decided: the semantics stay, the contract is explicit): a miss installs its deepest available
        ancestor from any tier and replays the rest on it. It is bitwise a clean full replay only when that anchor is
        exact raw (``anchor_source="raw"``) or there is none (``"none"``); with a compressed anchor (``"compressed"``)
        it equals a replay from the installed snapshot, and the rows promoted from that replay are inexact -- an
        anchor built on such rows is reported compressed too, though every range of it comes from a raw tier."""
        bridge = self.evicting(raw_tier=True, gpu_raw_budget_bytes=1 << 30, host_raw_budget_bytes=1 << 30,
                               raw_tier_retain="lru", raw_tier_promote="host")
        nodes, _ = self.build_a_to_f(bridge)
        bridge.checkpoint("s")
        tier = bridge.archive.raw_tier
        tier.flush()
        reference = self.bridge("raw", representation="raw")  # the clean history (exact RAW archive)
        reference_nodes, _ = self.build_a_to_f(reference)
        clean = self.child(reference, reference_nodes["F"], "F")
        # raw: D evicted without a raw copy; C's visible set from its captured raw copies
        tier.drop(tier.entries(node_refs=[nodes["D"]]))
        self.evict(bridge, nodes["D"])
        got = self.child(bridge, nodes["F"], "F")
        record = got[0]["restore"]["recompute"]
        self.assertEqual((record["anchor"], record["anchor_source"]), (nodes["C"], "raw"))
        self.assertTrue(torch.equal(got[1], clean[1]))  # bitwise a clean full replay
        assert_caches_bitwise(self, got[2], clean[2])
        tier.flush()
        promoted = {entry.node_ref: entry.exact for entry in tier.entries() if entry.origin == "replay"}
        self.assertEqual(promoted, {nodes["D"]: True})  # F kept its captured copy (promotion skip "exists")
        # derived: D's promoted rows marked inexact (as if replayed on a lossy history), F evicted without a raw copy:
        # the anchor D comes wholly from raw tiers, yet one range is inexact -- compressed
        with tier._cond:
            next(entry for entry in tier.entries(node_refs=[nodes["D"]])).exact = False
        tier.drop(tier.entries(node_refs=[nodes["F"]]))
        self.evict(bridge, nodes["F"])
        restore = self.child(bridge, nodes["F"], "F")[0]["restore"]
        self.assertEqual((restore["recompute"]["anchor"], restore["recompute"]["anchor_source"]),
                         (nodes["D"], "compressed"))
        served = {item["node"]: (item["tier"], item.get("exact")) for item in restore["lease"]["tiers"]["per_chunk"]}
        self.assertEqual(served[nodes["D"]], ("host_raw", False))
        self.assertTrue(all(tier_name in ("gpu_raw", "host_raw") for tier_name, _ in served.values()))
        # compressed: every raw copy dropped; the anchor C from the lossy archive -- its replay's promoted rows inexact
        bridge.control("raw_tier", {"action": "drop"})
        capture = Capture(self)
        restore = self.child(bridge, nodes["F"], "F")[0]["restore"]
        self.assertEqual((restore["recompute"]["anchor"], restore["recompute"]["anchor_source"]),
                         (nodes["C"], "compressed"))
        same(self, capture.end, self.toy_replay(capture.begin, "ABCDF", [3, 4]))  # a replay from the snapshot
        tier.flush()
        promoted = {entry.node_ref: entry.exact for entry in tier.entries() if entry.origin == "replay"}
        self.assertEqual(promoted, {nodes["D"]: False, nodes["F"]: False})
        # none: the first chunk evicted too, no raw copy: a full replay -- bitwise the clean history again
        bridge.control("raw_tier", {"action": "drop"})
        self.evict(bridge, nodes["A"])
        got = self.child(bridge, nodes["F"], "F")
        self.assertEqual((got[0]["restore"]["recompute"]["anchor"], got[0]["restore"]["recompute"]["anchor_source"]),
                         (None, "none"))
        self.assertTrue(torch.equal(got[1], clean[1]))
        assert_caches_bitwise(self, got[2], clean[2])

    def test_a_capture_on_a_lossy_history_is_no_exact_anchor(self):
        """Review C1, exactness at capture: a chunk generated on a history the session installed from the lossy archive
        is computed on lossy KV -- its captured raw copy is no clean replay (``exact`` False), and neither are the
        session's next captures until a restore installs an exact history again (exact raw ranges, or a full replay).
        An anchor built on such a copy is reported ``compressed`` although every range of it comes from a raw tier:
        the miss equals a replay from its installed snapshot, which here differs from a clean full replay."""
        bridge = self.evicting(raw_tier=True, gpu_raw_budget_bytes=1 << 30, host_raw_budget_bytes=1 << 30,
                               raw_tier_retain="lru", raw_tier_promote="host")
        tier = bridge.archive.raw_tier
        nodes = self.chain(bridge, "AB")  # native (one session, resident continuations): exact captures
        tier.flush()
        self.assertEqual({entry.node_ref: entry.exact for entry in tier.entries()},
                         {nodes["A"]: True, nodes["B"]: True})
        bridge.control("raw_tier", {"action": "drop"})  # the rewind to B installs it from the lossy archive
        lossy = self.child(bridge, nodes["B"], "B", child="C")
        self.assertEqual(lossy[0]["restore"]["mode"], "archive")
        self.assertFalse({item["tier"] for item in lossy[0]["restore"]["lease"]["tiers"]["per_chunk"]}
                         & {"gpu_raw", "host_raw"})
        node_c = self.head(bridge)
        run_tree(bridge, self.request_for(handle("s", "resume"), "ABCD"))  # a resident continuation: still lossy
        node_d = self.head(bridge)
        bridge.checkpoint("s")
        tier.flush()
        self.assertEqual({entry.node_ref: entry.exact for entry in tier.entries()}, {node_c: False, node_d: False})
        # an exact history again: A and B evicted without a raw copy -- the rewind to B replays them in full
        self.evict(bridge, nodes["A"], nodes["B"])
        replayed = self.child(bridge, nodes["B"], "B", child="E")
        self.assertEqual(replayed[0]["restore"]["recompute"]["anchor_source"], "none")
        node_e = self.head(bridge)
        bridge.checkpoint("s")
        tier.flush()
        self.assertEqual({entry.node_ref: entry.exact for entry in tier.entries()},
                         {nodes["A"]: True, nodes["B"]: True, node_c: False, node_d: False, node_e: True})
        # D evicted without a raw copy: the anchor C comes wholly from raw tiers, yet C was captured on a lossy B
        tier.drop(tier.entries(node_refs=[node_d]))
        self.evict(bridge, node_d)
        capture = Capture(self)
        got = run_tree(bridge, self.request_for(handle("s", "rewind", checkpoint=node_d), "ABCDF"))
        restore = got[0]["restore"]
        self.assertEqual((restore["recompute"]["anchor"], restore["recompute"]["anchor_source"]),
                         (node_c, "compressed"))
        served = {item["node"]: item.get("exact") for item in restore["lease"]["tiers"]["per_chunk"]
                  if item["tier"] in ("gpu_raw", "host_raw")}
        self.assertEqual(served, {nodes["A"]: True, nodes["B"]: True, node_c: False})
        same(self, capture.end, self.toy_replay(capture.begin, "ABCD", [3]))  # D replayed on the installed snapshot
        _, clean, _ = self.fresh("ABCDF")  # ... which is no clean full replay: C's rows carry B's lossy history
        self.assertFalse(all(torch.equal(a.k[:, :a.end], b.k[:, :b.end]) for stream in clean
                             for a, b in zip(got[2][stream], clean[stream])))
        tier.flush()
        self.assertEqual({entry.node_ref: entry.exact for entry in tier.entries() if entry.origin == "replay"},
                         {nodes["A"]: True, nodes["B"]: True, node_d: False})  # D's replay rows: inexact
        self.assertTrue(bridge.controller.get("s").lossy_history)
        # back to the root: a history computed natively from the start again (its captures exact)
        run_tree(bridge, self.request_for(handle("s", "rewind", checkpoint=0), "AG"))
        bridge.checkpoint("s")
        tier.flush()
        self.assertFalse(bridge.controller.get("s").lossy_history)
        fresh = [entry for entry in tier.entries() if entry.origin == "captured" and entry.label[1] == "c000001"]
        self.assertTrue(fresh and all(entry.exact for entry in fresh))

    def test_a_pending_commit_keeps_the_shared_payload_it_references(self):
        """Review C2: a pending publication's shared payload -- the HEVC-only arm's identity basis, written once and
        referenced by every node -- is reachable from the pending commit's staging leases: evicting the only live node
        that holds it is all or nothing, so nothing of it goes while the commit is pending (W1.4.3 v1 took the basis
        from under the commit, which then published a node whose payload was gone)."""
        self.hevc_geometry()
        bridge = self.evicting("hevc", representation="hevc")
        eviction, pipeline, store = bridge.archive.eviction, bridge.archive.pipeline, bridge.archive.store
        identity = identity_object(CHANNELS)[0]
        reached, gate, calls = threading.Event(), threading.Event(), []

        def hooks(stage, node_id):
            if stage == "publish":
                calls.append(node_id)
                if len(calls) == 2:
                    reached.set()
                    gate.wait(60)
        pipeline.hooks = hooks
        stage = TreeStage()
        run_tree(bridge, self.request_for(handle("s"), "A"), stage=stage)
        node_a = bridge.checkpoint("s")["head"]
        self.assertIn(identity, eviction.node_payload(node_a).objects)
        try:
            run_tree(bridge, self.request_for(handle("s", "resume"), "AB"), stage=stage)
            self.assertTrue(reached.wait(60))  # B is stored and references the basis; its publication is held
            self.assertFalse(self.evict_now(bridge, node_a))  # A's own objects include the basis: nothing goes
            self.assertEqual(store.evicted_ids(eviction.node_payload(node_a).objects), set())
            self.assertFalse(eviction.is_evicted(node_a))
            self.assertGreaterEqual(eviction.stats()["evict"]["skipped_leased"], 1)
        finally:
            gate.set()
        node_b = bridge.checkpoint("s")["head"]
        self.assertFalse(eviction.is_evicted(node_b))
        self.assertEqual(store.evicted_ids(eviction.node_payload(node_b).objects), set())
        self.assertTrue(self.evict_now(bridge, node_a))  # B published: the basis is shared, A goes without it
        self.assertNotIn(identity, store.evicted_ids())

    def test_a_publication_registers_its_payload_atomically_with_evictions(self):
        """Review C2: ``published`` checks a node's payload and registers it under the eviction lock: an eviction racing
        it -- of the only other node holding a shared object -- waits, so a node is never registered live with an
        object gone (W1.4.3 v1 checked outside the lock). Called directly, without the staging leases that also
        protect a real publication."""
        self.hevc_geometry()
        bridge = self.evicting("hevc", representation="hevc")
        nodes = self.chain(bridge, "AB")
        eviction, store = bridge.archive.eviction, bridge.archive.store
        identity = identity_object(CHANNELS)[0]
        with eviction._cond:  # B as if still pending: out of the accounting
            payload = eviction._nodes.pop(nodes["B"])
            for object_id in payload.objects:
                holders = eviction._holders[object_id]
                holders.discard(nodes["B"])
                if not holders:
                    del eviction._holders[object_id]
                    eviction.live_bytes -= eviction._sizes.pop(object_id)
        real, racer, outcome = store.object_sizes, [], {}

        def object_sizes(object_ids):
            if not racer:  # inside the publication: evict A (the basis's only registered holder) concurrently
                racer.append(threading.Thread(target=lambda: outcome.update(evicted=self.evict_now(bridge,
                                                                                                   nodes["A"]))))
                racer[0].start()
                racer[0].join(0.5)
            return real(object_ids)
        with mock.patch.object(store, "object_sizes", object_sizes):
            eviction.published(object(), nodes["B"])
        racer[0].join(30)
        self.assertEqual(outcome, {"evicted": True})  # A went once B was registered -- without the shared basis
        self.assertFalse(eviction.is_evicted(nodes["B"]))
        self.assertEqual(store.evicted_ids(eviction.node_payload(nodes["B"]).objects), set())
        self.assertNotIn(identity, store.evicted_ids())

    def test_the_identity_basis_is_reserved_whenever_it_is_physically_absent(self):
        """Review C3: the HEVC-only arm's identity basis (one shared block per channel count) is reserved whenever the
        store does not hold it -- the first node, and a rewrite after its last holder was evicted: a commit whose
        payload fits the budget but whose payload + basis does not fails typed, the budget never exceeded (W1.4.3 v1
        wrote the basis unreserved, past the budget)."""
        self.hevc_geometry()
        probe = self.evicting("probe", representation="hevc")
        probe_nodes = self.chain(probe, "AB")
        stats = probe.archive.eviction.stats()
        self.assertEqual(stats["reserved_total_bytes"], stats["live_bytes"])  # the basis reserved exactly once
        identity, basis_bytes = identity_object(CHANNELS)
        own_b = self.node_bytes(probe, probe_nodes["B"]) - basis_bytes
        bridge = self.evicting("tight", representation="hevc")
        eviction = bridge.archive.eviction
        nodes = self.chain(bridge, "A")
        self.evict(bridge, nodes["A"])  # the basis goes with its only holder
        self.assertIn(identity, bridge.archive.store.evicted_ids())
        with eviction._cond:
            eviction.budget = own_b + basis_bytes - 1  # B's payload fits; payload + basis does not
            eviction.peak_bytes = eviction.live_bytes + eviction.reserved_bytes
        with self.assertRaises(Exception) as caught:
            run_tree(bridge, self.request_for(handle("s", "resume"), "AB"))
            bridge.checkpoint("s")
        self.assertIn("archive_budget_bytes", f"{caught.exception} {caught.exception.__cause__}")
        stats = eviction.stats()
        self.assertLessEqual(stats["peak_bytes"], eviction.budget)
        # the basis was rewritten within its reservation, then the commit failed: it stays in the store for no live
        # node, counted as an orphan (evicted first), so the accounting matches the store
        self.assertNotIn(identity, bridge.archive.store.evicted_ids())
        self.assertEqual((stats["orphan_objects"], stats["orphan_bytes"], stats["live_bytes"]),
                         (1, basis_bytes, basis_bytes))
        with eviction._cond:
            self.assertTrue(eviction._evict_one())  # the orphan goes first
        self.assertIn(identity, bridge.archive.store.evicted_ids())
        stats = eviction.stats()
        self.assertEqual((stats["orphan_objects"], stats["live_bytes"], stats["evict"]["orphans"]), (0, 0, 1))

    def test_a_partially_leased_node_is_not_evicted_at_all(self):
        """Review I4: a node is evicted all or nothing -- one leased payload object keeps every object of it (W1.4.3 v1
        evicted the others and kept the node registered live with a partial payload)."""
        bridge = self.evicting()
        nodes = self.chain(bridge, "AB")
        eviction, store = bridge.archive.eviction, bridge.archive.store
        objects = sorted(eviction.node_payload(nodes["A"]).objects)
        self.assertGreater(len(objects), 1)
        lease = store.lease([objects[0]], verify=False)
        try:
            self.assertFalse(self.evict_now(bridge, nodes["A"]))
            self.assertEqual(store.evicted_ids(objects), set())
            self.assertFalse(eviction.is_evicted(nodes["A"]))
            self.assertEqual(eviction.stats()["evict"]["skipped_leased"], 1)
        finally:
            lease.release()
        self.assertTrue(self.evict_now(bridge, nodes["A"]))
        self.assertEqual(store.evicted_ids(objects), set(objects))

    def test_reopening_an_evicted_store_without_a_budget_is_refused(self):
        """Review I6: a store whose archive evicted payloads reopens only with an archive budget -- without one a
        typed, actionable configuration error (W1.4.3 v1 opened it and failed later, on the first read)."""
        bridge = self.evicting("refuse")
        nodes = self.chain(bridge, "ABC")
        self.evict(bridge, nodes["B"])
        bridge.close()
        with self.assertRaises(ConfigurationError) as caught:
            self.bridge("refuse", reuse=False)
        self.assertIn("archive_budget_bytes", str(caught.exception))
        again = self.evicting("refuse")
        self.assertTrue(again.archive.eviction.is_evicted(nodes["B"]))
        self.assertEqual(self.child(again, nodes["C"], "C")[0]["restore"]["recompute"]["anchor"], nodes["A"])

    def test_a_cancelled_commit_stops_waiting_for_budget_room(self):
        """Review M8: a commit waiting in ``reserve`` for room (every candidate pinned) stops at once when its lineage
        is cancelled -- a whole-tree delete or the warm-up purge, which every run does -- instead of waiting out
        ``RESERVE_WAIT_SECONDS``."""
        probe = self.evicting("probe")
        size = max(self.node_bytes(probe, ref) for ref in self.chain(probe, "AB").values())
        bridge = self.evicting(budget=int(1.5 * size))
        eviction, pipeline = bridge.archive.eviction, bridge.archive.pipeline
        stage = TreeStage()
        run_tree(bridge, self.request_for(handle("s"), "A"), stage=stage)
        node_a = bridge.checkpoint("s")["head"]
        token = eviction.pin_available([node_a])  # A cannot go: B's reservation must wait
        eviction.wait_seconds = 60.0
        try:
            run_tree(bridge, self.request_for(handle("s", "resume"), "AB"), stage=stage)
            deadline = time.monotonic() + 30
            while not eviction._waiting and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(eviction._waiting)  # B waits for room
            started = time.monotonic()
            pipeline.cancel("s", timeout=30)
            self.assertLess(time.monotonic() - started, 5.0)
        finally:
            eviction.unpin(token)
        self.assertEqual((eviction.stats()["reserved_bytes"], len(eviction._waiting)), (0, 0))

    def test_a_reopened_archive_adopts_parents_before_children(self):
        """Review M9 (decided: the LRU order resets on reopen -- runs do not reopen mid-run): a reopened archive
        registers its nodes parents before children, the root first (W1.4.3 v1 could register a child before its
        parent)."""
        bridge = self.evicting("order")
        nodes, _ = self.build_a_to_f(bridge)
        bridge.checkpoint("s")
        bridge.close()
        again = self.evicting("order")
        archive = again.archive
        order = archive._archived_nodes()
        self.assertTrue(set(nodes.values()) <= set(order))
        position = {ref: index for index, ref in enumerate(order)}
        for ref in order:
            parent = archive.store.get_manifest(ref)["node"].get("parent")
            if parent:
                self.assertLess(position[parent], position[ref])
        self.assertEqual(list(archive.eviction._nodes), [ref for ref in order if ref in archive.eviction._nodes])

    # -- review round 2 (rubber-duck a6263240): failures keep their bytes counted, prune is atomic, writes deduplicate
    def test_a_failed_commits_written_objects_stay_counted_until_they_leave_the_store(self):
        """Review round 2, finding 1: a commit that wrote its basis rows and was then cancelled (a whole-tree delete,
        the warm-up purge) gives its reservation back in the same step its written objects become orphans (counted
        live, evicted first) -- before its staging leases go and before the failed objects' collection. At that
        barrier another lineage reserves: ``live + reserved`` never understates the payload bytes in the store, and
        the writer's room comes from the orphans (c5cb1aba dropped the reservation and left the rows untracked: the
        writer was admitted past the budget)."""
        bridge = self.evicting()
        eviction, store, pipeline = bridge.archive.eviction, bridge.archive.store, bridge.archive.pipeline
        self.chain(bridge, "AB")
        reached, gate = threading.Event(), threading.Event()

        def hooks(stage, node_id):
            if stage == "store" and node_id == "c000000" and not reached.is_set():
                reached.set()
                gate.wait(60)
        pipeline.hooks = hooks
        run_tree(bridge, self.request_for(handle("t"), "E"))
        self.assertTrue(reached.wait(60))  # t's commit: its rows written (reserved), its payloads not yet
        rows = eviction.reserved_bytes
        self.assertGreater(rows, 0)
        barrier, real_gc = {}, store.collect_garbage

        def collect_garbage(*args, **kwargs):
            if kwargs.get("candidates") and "physical" not in barrier:  # the failed objects' collection, about to run
                physical = physical_payload_bytes(store)
                barrier["physical"] = physical
                barrier["counted"] = eviction.live_bytes + eviction.reserved_bytes
                need = 2 * rows
                self.set_budget(bridge, physical + need - 1)  # the writer fits only once t's rows leave the store
                key = SimpleNamespace(budget_order=("writer", 1))
                try:
                    eviction.reserve(key, need)
                    barrier["admitted_within"] = physical_payload_bytes(store) + need <= eviction.budget
                finally:
                    eviction.release(key)
            return real_gc(*args, **kwargs)
        with mock.patch.object(store, "collect_garbage", collect_garbage):
            deleting = threading.Thread(target=lambda: bridge.control("delete", {"session_id": "t"}), daemon=True)
            deleting.start()
            self.assertTrue(until(lambda: pipeline.stats_.get("cancelled", 0) >= 1, 30.0))
            gate.set()  # the cancelled commit stops at its payload write; retired, its staging leases go
            deleting.join(60)
        self.assertFalse(deleting.is_alive())
        self.assertGreaterEqual(barrier["counted"], barrier["physical"])
        self.assertTrue(barrier["admitted_within"])
        self.assertGreaterEqual(eviction.stats()["evict"]["orphans"], 1)
        self.set_budget(bridge, HUGE)
        self.assertEqual(eviction.live_bytes, physical_payload_bytes(store))
        self.assertEqual(eviction.reserved_bytes, 0)

    def test_a_write_waits_while_a_failed_commits_orphans_are_still_protected(self):
        """Review round 2: a cancelled commit's written rows become orphans the moment its reservation goes, while its
        staging leases still protect them for a moment (the pipeline releases them next). A write that needs their
        room meanwhile waits -- their protection lifts -- instead of failing as "below the commits in flight", and
        takes their room once the leases go."""
        import reelcache.lineage_commit as lineage_commit

        bridge = self.evicting()
        eviction, pipeline = bridge.archive.eviction, bridge.archive.pipeline
        reached, gate = threading.Event(), threading.Event()

        def hooks(stage, node_id):
            if stage == "store" and node_id == "c000000" and not reached.is_set():
                reached.set()
                gate.wait(60)
        pipeline.hooks = hooks
        run_tree(bridge, self.request_for(handle("t"), "E"))
        self.assertTrue(reached.wait(60))
        rows = eviction.reserved_bytes
        real_release, outcome = lineage_commit.CommitJob.release_staging, {}

        def release_staging(job, *, collect):
            if "waiting" not in outcome and job.error is not None:  # retired: reservation gone, rows orphaned, leased
                outcome["orphaned"] = eviction.stats()["orphan_bytes"]
                self.set_budget(bridge, eviction.live_bytes + eviction.reserved_bytes + rows - 1)

                def write():
                    key = SimpleNamespace(budget_order=("writer", 1))
                    try:
                        eviction.reserve(key, rows)
                        outcome["reserved"] = True
                    except BaseException as exc:  # noqa: BLE001 - recorded
                        outcome["error"] = exc
                    finally:
                        eviction.release(key)
                writer = threading.Thread(target=write, daemon=True)
                writer.start()
                writer.join(0.5)
                outcome["waiting"], outcome["writer"] = writer.is_alive(), writer
            return real_release(job, collect=collect)
        with mock.patch.object(lineage_commit.CommitJob, "release_staging", release_staging):
            cancelling = threading.Thread(target=lambda: pipeline.cancel("t", timeout=60), daemon=True)
            cancelling.start()
            self.assertTrue(until(lambda: pipeline.stats_.get("cancelled", 0) >= 1, 30.0))
            gate.set()
            cancelling.join(60)
            outcome["writer"].join(30)
        self.assertEqual({key: outcome.get(key) for key in ("orphaned", "waiting", "reserved")},
                         {"orphaned": rows, "waiting": True, "reserved": True})
        self.assertGreaterEqual(eviction.stats()["evict"]["orphans"], 1)

    def test_a_publication_whose_bookkeeping_fails_stays_counted_and_is_registered_before_the_next_write(self):
        """Review round 2, finding 1: the publication is durable; when its bookkeeping fails (a manifest read here) the
        reservation keeps its bytes counted, and the node is registered before the next write is admitted
        (c5cb1aba released the reservation: the durable payload was untracked)."""
        bridge = self.evicting()
        eviction, store = bridge.archive.eviction, bridge.archive.store
        real, failures = eviction.node_payload, []

        def flaky(node_ref):
            if not failures:
                failures.append(node_ref)
                raise OSError("injected: the node's manifest could not be read")
            return real(node_ref)
        with mock.patch.object(eviction, "node_payload", flaky):
            run_tree(bridge, self.request_for(handle("s"), "A"))
            node_a = bridge.checkpoint("s")["head"]
            self.assertEqual(failures, [node_a])
            self.assertGreaterEqual(eviction.live_bytes + eviction.reserved_bytes, physical_payload_bytes(store))
            stats = eviction.stats()
            self.assertEqual((stats["live_nodes"], stats["unreconciled"], stats["errors"]), (0, 1, 1))
            run_tree(bridge, self.request_for(handle("s", "rewind", checkpoint=node_a), "AB"))
            node_b = bridge.checkpoint("s")["head"]
        stats = eviction.stats()
        self.assertEqual((stats["live_nodes"], stats["unreconciled"], stats["reconciled"], stats["reserved_bytes"]),
                         (2, 0, 1, 0))
        self.assertFalse(eviction.is_evicted(node_a) or eviction.is_evicted(node_b))
        self.assertEqual(stats["live_bytes"], physical_payload_bytes(store))

    def test_a_prune_keeps_a_freed_object_counted_while_it_is_in_the_store(self):
        """Review round 2, finding 2: a whole-tree delete removes the last live holder of the HEVC-only arm's identity
        basis while an evicted node of another session still references it. ``prune`` checks presence before it
        subtracts anything, in one step under the eviction lock: a writer reserving meanwhile waits for it and sees
        the basis counted (an orphan, evicted first) -- c5cb1aba subtracted it first and checked presence outside the
        lock, so the writer was admitted past the budget in that gap."""
        self.hevc_geometry()
        bridge = self.evicting(representation="hevc")
        eviction, store = bridge.archive.eviction, bridge.archive.store
        identity = identity_object(CHANNELS)[0]
        run_tree(bridge, self.request_for(handle("s"), "A"))
        bridge.checkpoint("s")
        run_tree(bridge, self.request_for(handle("t"), "E"))
        node_t = bridge.checkpoint("t")["head"]
        self.assertTrue(self.evict_now(bridge, node_t))  # t's own objects go; the basis stays (s holds it)
        self.assertEqual(store.evicted_ids([identity]), set())
        basis = store.object_sizes([identity])[identity]
        budget = eviction.live_bytes + 4 * basis
        self.set_budget(bridge, budget)
        need = budget - basis + 1  # fits only once the basis is not in the store
        armed, gap, real_indexed, real_prune = threading.Event(), {}, store.indexed, eviction.prune

        def indexed(ids):
            ids = list(ids)
            if armed.is_set() and identity in ids and "admitted" not in gap:
                outcome = {}

                def write():
                    key = SimpleNamespace(budget_order=("writer", 1))
                    try:
                        eviction.reserve(key, need)
                        outcome["within"] = physical_payload_bytes(store) + need <= budget
                    except BaseException as exc:  # noqa: BLE001 - recorded
                        outcome["error"] = exc
                    finally:
                        eviction.release(key)
                writer = threading.Thread(target=write, daemon=True)
                writer.start()
                writer.join(1.0)
                gap["admitted"] = not writer.is_alive()  # admitted while prune was between its steps
                gap["writer"], gap["outcome"] = writer, outcome
            return real_indexed(ids)

        def prune():
            armed.set()
            try:
                return real_prune()
            finally:
                armed.clear()
        with mock.patch.object(store, "indexed", indexed), mock.patch.object(eviction, "prune", prune):
            bridge.control("delete", {"session_id": "s"})
        gap["writer"].join(30)
        self.assertFalse(gap["admitted"])
        self.assertEqual(gap["outcome"], {"within": True})
        stats = eviction.stats()
        self.assertEqual((stats["reserved_bytes"], stats["live_nodes"]), (0, 0))
        self.assertTrue(eviction.is_evicted(node_t))
        self.assertEqual(eviction.live_bytes, physical_payload_bytes(store))

    def test_identical_row_blocks_are_reserved_once(self):
        """Review round 2, finding 3: a chunk of all-zero matrices -- every one a zero-input plan fitting its own
        one-row basis, all of them the same content -- admits each content once. Under a budget of exactly its
        node's bytes it publishes (c5cb1aba reserved the block once per matrix and failed)."""
        real_kv = tree_module.toy_kv

        def zeros(prompt, chunk, layer, stream, cache):
            key, value = real_kv(prompt, chunk, layer, stream, cache)
            return torch.zeros_like(key), torch.zeros_like(value)
        with mock.patch.object(tree_module, "toy_kv", zeros):
            probe = self.evicting("probe")
            run_tree(probe, self.request_for(handle("s"), "A"))
            size = self.node_bytes(probe, probe.checkpoint("s")["head"])
            bridge = self.evicting(budget=size)
            run_tree(bridge, self.request_for(handle("s"), "A"))
            node = bridge.checkpoint("s")["head"]
        stats = bridge.archive.eviction.stats()
        self.assertEqual((stats["live_nodes"], stats["live_bytes"], stats["reserved_total_bytes"]), (1, size, size))
        self.assertFalse(bridge.archive.eviction.is_evicted(node))
        self.assertEqual(stats["writes"]["bytes"], size)  # one block for all four matrices

    def test_a_duplicate_or_republished_write_reserves_only_what_is_absent(self):
        """Review round 2, finding 3: writes are admitted by content address. A second session writing the identical
        chunk finds every object present and reserves nothing; once every holder of those objects is evicted, a third
        writing it again reserves exactly the evicted bytes (the store restores them). c5cb1aba reserved the full
        write each time -- over-reservation that made the ablation arms evict more than necessary."""
        bridge = self.evicting()
        eviction = bridge.archive.eviction
        run_tree(bridge, self.request_for(handle("s"), "A"))
        node_s = bridge.checkpoint("s")["head"]
        size = self.node_bytes(bridge, node_s)
        self.assertEqual(eviction.stats()["reserved_total_bytes"], size)
        run_tree(bridge, self.request_for(handle("t"), "A"))
        node_t = bridge.checkpoint("t")["head"]
        self.assertEqual(eviction.stats()["reserved_total_bytes"], size)  # identical content: nothing reserved
        self.assertEqual(eviction.stats()["writes"]["present_bytes"], size)
        self.evict(bridge, *dict.fromkeys((node_s, node_t)))
        self.assertTrue(eviction.is_evicted(node_s) and eviction.is_evicted(node_t))
        run_tree(bridge, self.request_for(handle("u"), "A"))
        node_u = bridge.checkpoint("u")["head"]
        self.assertEqual(eviction.stats()["reserved_total_bytes"], 2 * size)  # exactly the evicted bytes again
        self.assertFalse(eviction.is_evicted(node_u))
        self.assertEqual(eviction.live_bytes, size)


if __name__ == "__main__":
    unittest.main()
