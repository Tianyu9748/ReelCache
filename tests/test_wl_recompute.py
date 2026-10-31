"""Workload v2 W1b: the exact recompute arm (``restore_mode="recompute"``, ``representation="latents_only"``) on the CPU
toy harness, without SGLang.

The replay rebuilds the native cache from the stored clean latents by rerunning the target path's clean forwards in
native order (chunk context forwards and, here, a LongLive-1.0-like recache at every prompt change), each under its
block's historical conditioning; it must equal a RAW restore of the same node bitwise (the toy kernels are
deterministic). The raw tiers sit in front: a full hit installs from them, any miss replays.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from reelcache.errors import ConfigurationError, IntegrityError, StateError, UnsupportedError
from reelcache.integrations.sglang import ReelCacheRuntimeConfig
from reelcache.integrations.sglang.nodes import replay_steps
from reelcache.types import MatrixKey
from reelcache_sglang_models.causal_wan.lifecycle import (block_prompt_texts, recache_matches, replay_schedule,
                                                       segment_of)
from tests.test_sglang_bridge import FRAMES, LAYERS, TPF, Span, handle
from tests.test_wl_tree import (CAPACITY, PROMPTS, GuardedCache, SessionMaxChecks, TreeStage, TreeTestCase,
                                assert_caches_bitwise, run_tree, tree_request)

PATHS = {"A": "A", "B": "AB", "C": "ABC", "D": "ABCD", "E": "ABE", "F": "ABCDF"}


def spans(restore):
    """The attended layout of a restore (node ids differ between archives: RAW nodes hold packets)."""
    return {stream: [(span["chunk"], span["token_start"], span["length"], span["slot_start"], span["source"])
                     for span in plan["spans"]] for stream, plan in restore["plans"].items()}


class RecomputeTestCase(TreeTestCase):
    """Tree mode + the recompute arm; ``raw`` builds the RAW reference of the same tree."""

    representation = "latents_only"

    def recompute_bridge(self, store="latents", **overrides):
        return self.bridge(store, restore_mode="recompute", **overrides)

    def raw_bridge(self, store="raw", **overrides):
        values = {"representation": "raw", **overrides}
        return self.bridge(store, **values)

    def restore_child(self, bridge, node, letter, child="G"):
        """Rewind to ``node`` (path ``PATHS[letter]``) and generate one child: (summary, latents, caches, rng)."""
        return run_tree(bridge, self.request_for(handle("s", "rewind", checkpoint=node), PATHS[letter] + child))


class RecomputeChecks:
    def test_replay_equals_raw_restore_for_every_node(self):
        raw = self.raw_bridge()
        raw_nodes, _ = self.build_a_to_f(raw)
        recompute = self.recompute_bridge(raw_tier=False)  # no raw tier: every restore replays
        nodes, _ = self.build_a_to_f(recompute)
        for letter in "ABCDEF":
            expected = self.restore_child(raw, raw_nodes[letter], letter)
            got = self.restore_child(recompute, nodes[letter], letter)
            restore = got[0]["restore"]
            self.assertEqual((restore["mode"], restore["source"]), ("recompute", "replay"), letter)
            self.assertEqual(expected[0]["restore"]["mode"], "archive")
            self.assertEqual(restore["replay"]["chunk_forwards"], len(PATHS[letter]))
            self.assertTrue(restore["replay"]["completed"])
            self.assertGreaterEqual(restore["recompute_replay_seconds"], 0.0)
            self.assertEqual(spans(restore), spans(expected[0]["restore"]), letter)  # same attended layout
            self.assertTrue(torch.equal(got[1], expected[1]), letter)
            assert_caches_bitwise(self, got[2], expected[2])
            self.assertTrue(torch.equal(got[3].get_state(), expected[3].get_state()), letter)
        # The RAW store replayed directly (validation configuration: representation="raw" + recompute) agrees too.
        raw.close()
        replaying = self.bridge("raw", representation="raw", restore_mode="recompute")
        summary, latents, caches, _ = self.restore_child(replaying, raw_nodes["F"], "F", child="H")
        reference = self.raw_bridge("raw-reference")
        reference_nodes, _ = self.build_a_to_f(reference)
        expected = self.restore_child(reference, reference_nodes["F"], "F", child="H")
        self.assertEqual(summary["restore"]["mode"], "recompute")
        self.assertTrue(torch.equal(latents, expected[1]))
        assert_caches_bitwise(self, caches, expected[2])

    def test_raw_tier_hits_install_without_replay_and_misses_replay(self):
        raw = self.raw_bridge()
        raw_nodes, _ = self.build_a_to_f(raw)
        bridge = self.recompute_bridge(gpu_raw_budget_bytes=1 << 30, host_raw_budget_bytes=1 << 30)
        nodes, _ = self.build_a_to_f(bridge)
        tier = bridge.archive.raw_tier
        # D's branch was left by E's rewind: its entries stayed in the LRU (owned), so restoring D is a full tier hit.
        hit = self.restore_child(bridge, nodes["D"], "D")
        restore = hit[0]["restore"]
        self.assertEqual((restore["mode"], restore["source"], restore["replay"]), ("archive", "raw_tier", None))
        self.assertEqual(sum(restore["lease"]["tiers"]["chunks"].values()), 4)
        expected = self.restore_child(raw, raw_nodes["D"], "D")
        self.assertTrue(torch.equal(hit[1], expected[1]))
        assert_caches_bitwise(self, hit[2], expected[2])
        # One missing range is a miss for the whole history: replay (never a partial install). E's own range is attended
        # whatever the epochs (with recache epochs, B's chunk range is superseded by E's epoch on E's path).
        victim = [entry for entry in tier.entries(node_refs=[nodes["E"]])]
        self.assertEqual(len(victim), 1)
        tier.drop(victim)
        missed = self.restore_child(bridge, nodes["E"], "E")
        self.assertEqual((missed[0]["restore"]["mode"], missed[0]["restore"]["source"]), ("recompute", "replay"))
        expected = self.restore_child(raw, raw_nodes["E"], "E")
        self.assertTrue(torch.equal(missed[1], expected[1]))
        assert_caches_bitwise(self, missed[2], expected[2])
        stats = bridge.archive.stats()["raw_tier"]
        self.assertGreaterEqual(stats["archive"]["misses"], 1)
        self.assertGreaterEqual(stats["archive"]["restores"], 1)

    def test_raw_tier_control_drop_forces_replay(self):
        bridge = self.recompute_bridge(gpu_raw_budget_bytes=1 << 30, host_raw_budget_bytes=1 << 30)
        nodes, _ = self.build_a_to_f(bridge)
        stats = bridge.control("raw_tier", {"action": "stats", "session_id": "s"})
        self.assertTrue(stats["raw_tier"]["enabled"])
        self.assertGreater(stats["session"]["entries"], 0)
        bridge.control("raw_tier", {"action": "drop"})
        self.assertEqual(bridge.control("raw_tier", {"action": "stats"})["raw_tier"]["gpu_bytes"], 0)
        summary = self.restore_child(bridge, nodes["D"], "D")[0]
        self.assertEqual((summary["restore"]["mode"], summary["restore"]["source"]), ("recompute", "replay"))

    def test_latents_only_archive_contains_no_kv_objects(self):
        bridge = self.recompute_bridge()
        nodes, _ = self.build_a_to_f(bridge)
        store, archive = bridge.archive.store, bridge.archive
        with store.transaction():
            objects = store._walk(list(nodes.values()), verify=False)
        kinds = {}
        for object_id in objects:
            if object_id.startswith("manifest:"):
                kind = store.get_manifest(object_id).get("type")
                kinds[kind] = kinds.get(kind, 0) + 1
        self.assertNotIn("sglang_packet", kinds)
        self.assertGreater(kinds["sglang_causal_chunk"], 0)
        for node in archive.path(nodes["F"])[1:]:
            manifest = archive._manifest(node.ref)
            self.assertTrue(all(refs == [] for refs in manifest["packets"].values()), node.ref)
            self.assertEqual(node.record["representation"], "latents_only")
        self.assertEqual(archive.storage(nodes["F"])["packets"], 0)
        with self.assertRaises(StateError):
            archive.raw_matrix(nodes["F"], "cond", None)
        raw = self.raw_bridge()
        raw_nodes, _ = self.build_a_to_f(raw)
        self.assertLess(store.stats()["object_bytes"] * 4, raw.archive.store.stats()["object_bytes"])
        # Latents and RNG states are stored (the replay's inputs).
        state = archive.execution_state(archive.load(nodes["F"]))
        self.assertIn("latents", state)
        self.assertTrue(any(name.startswith("generator.") for name in state))

    def test_a_stage_without_replay_fails_typed_and_the_worker_continues(self):
        bridge = self.recompute_bridge(raw_tier=False)
        nodes, _ = self.build_a_to_f(bridge)
        with self.assertRaises(UnsupportedError) as caught:
            run_tree(bridge, self.request_for(handle("s", "rewind", checkpoint=nodes["C"]), "ABCG"), do_replay=False)
        self.assertIn("replays the restored history", str(caught.exception))
        self.assertFalse(bridge.active_request)
        summary, _, _, _ = self.restore_child(bridge, nodes["C"], "C")
        self.assertEqual(summary["restore"]["mode"], "recompute")

    def test_an_incomplete_replay_fails_the_cursor_check(self):
        bridge = self.recompute_bridge(raw_tier=False)
        nodes, _ = self.build_a_to_f(bridge)
        steps = replay_steps(bridge.archive.path(nodes["D"])[1:])
        with self.assertRaises(IntegrityError) as caught:
            run_tree(bridge, self.request_for(handle("s", "rewind", checkpoint=nodes["D"]), "ABCDG"),
                     replay_skip=len(steps) - 1)
        self.assertIn("replayed cache cursors", str(caught.exception))
        self.assertFalse(bridge.active_request)

    def test_resident_continuation_needs_no_replay(self):
        bridge = self.recompute_bridge(raw_tier=False)
        stage = TreeStage()
        run_tree(bridge, self.request_for(handle("s", pause_after_chunks=2), "ABC"), stage=stage)
        summary, _, _, _ = run_tree(bridge, self.request_for(handle("s", "resume"), "ABC"), stage=stage)
        self.assertEqual(summary["restore"]["mode"], "resident")


class RecomputeTests(RecomputeChecks, RecomputeTestCase):
    """Cross-attention-only style prompt changes (no cache epochs)."""


class RecomputeRecacheTests(RecomputeChecks, RecomputeTestCase):
    """LongLive-1.0-like recache epochs at every prompt change, on both branches of the tree."""

    recache = True

    def test_recache_epochs_on_branches_are_replayed_at_their_positions(self):
        bridge = self.recompute_bridge(raw_tier=False)
        nodes, _ = self.build_a_to_f(bridge)
        summary, _, _, _ = self.restore_child(bridge, nodes["F"], "F")
        replay = summary["restore"]["replay"]
        self.assertEqual((replay["chunk_forwards"], replay["recache_forwards"]), (5, 3))
        self.assertEqual(len(replay["epochs"]), 3)
        summary, _, _, _ = self.restore_child(bridge, nodes["E"], "E")
        replay = summary["restore"]["replay"]
        self.assertEqual((replay["chunk_forwards"], replay["recache_forwards"]), (3, 1))
        self.assertIn("switch-2-pe", replay["epochs"][0])  # E's branch has its own epoch (prompt pe), not C's


class SessionMaxRecomputeTests(SessionMaxChecks, RecomputeTestCase):
    """W1.2: the runner's session_max geometry on the recompute arm without a raw tier (every rewind replays, with
    LongLive-1.0-like recache epochs), bitwise the fresh native runs."""

    recache = True

    def bridge(self, store="archive", profile=None, **overrides):
        return super().bridge(store, profile=profile, **{"restore_mode": "recompute", "raw_tier": False, **overrides})


class LifecycleReplayTests(unittest.TestCase):
    """The profile stages' plain-data replay decisions (``reelcache_sglang_models.causal_wan.lifecycle``)."""

    def steps(self):
        return [{"kind": "chunk", "block": 0}, {"kind": "chunk", "block": 1}, {"kind": "chunk", "block": 2},
                {"kind": "chunk", "block": 3}, {"kind": "chunk", "block": 4},
                {"kind": "recache", "precedes_block": 5, "window": {"token_start": 1560 * 3, "num_tokens": 1560 * 12},
                 "written": [[1560 * 6, 1560 * 9]]}, {"kind": "chunk", "block": 5}, {"kind": "chunk", "block": 6}]

    def test_replay_schedule_segments_and_cross_attention_resets(self):
        schedule = replay_schedule(self.steps(), [5])
        self.assertEqual([item.get("segment") for item in schedule], [0, 0, 0, 0, 0, None, 1, 1])
        self.assertEqual([item["reset_cross_attention"] for item in schedule],
                         [True, False, False, False, False, True, True, False])
        self.assertEqual((schedule[5]["previous_segment"], schedule[5]["new_segment"]), (0, 1))
        # A switch without a recache on the path (cross-attention only): the prompt changes, the cache resets.
        plain = replay_schedule([step for step in self.steps() if step["kind"] == "chunk"], [5])
        self.assertEqual([item["reset_cross_attention"] for item in plain], [True, False, False, False, False, True, False])
        with self.assertRaises(ValueError):
            replay_schedule(self.steps(), [6])  # the path's epoch precedes block 5, not a switch of this request
        with self.assertRaises(ValueError):
            replay_schedule([{"kind": "bogus"}], [])

    def test_recache_matches_and_block_prompts(self):
        plan = {"frame_seq": 1560, "recompute_tokens": [1560 * 3, 1560 * 15], "written_global_frames": [6, 15]}
        self.assertTrue(recache_matches(self.steps()[5], plan))
        self.assertFalse(recache_matches(self.steps()[5], {**plan, "written_global_frames": [3, 15]}))
        switches = [{"chunk": 5, "prompt": "q"}, {"chunk": 7, "prompt": "r"}]
        self.assertEqual(block_prompt_texts("p", switches, 9), ["p"] * 5 + ["q", "q", "r", "r"])
        self.assertEqual(segment_of(7, [5, 7]), 2)

    def test_runtime_config_of_the_recompute_arm(self):
        from reelcache_sglang_models.causal_wan.lifecycle import bridge_runtime_config
        from reelcache_sglang_models.causal_wan.profiles import get_profile

        config = bridge_runtime_config(get_profile("longlive1"), store_root="/tmp/x", representation="latents_only",
                                       restore_mode="recompute", tree_retention="full")
        self.assertNotIn("max_staged_canvases", config)
        parsed = ReelCacheRuntimeConfig.from_dict(config)
        self.assertEqual((parsed.representation, parsed.restore_mode, parsed.tree_retention),
                         ("latents_only", "recompute", "full"))
        for bad in ({"representation": "latents_only"}, {"restore_mode": "recompute"},
                    {"restore_mode": "replay"}):
            with self.assertRaises(ConfigurationError):
                ReelCacheRuntimeConfig.from_dict({"store_root": "x", **bad})

    def test_runtime_config_accepts_every_bridge_representation(self):
        """W1c compatibility (coordinator): raw, svd_hevc, svd, hevc and latents_only pass (the bridge configuration's
        own validation stays authoritative); the LongLive 1.0 staging rule ``max(2, recache chunks)`` applies to the
        canvas packs (svd_hevc, hevc) only, and a caller's value still wins."""
        from reelcache_sglang_models.causal_wan.lifecycle import bridge_runtime_config
        from reelcache_sglang_models.causal_wan.profiles import get_profile

        longlive1, causvid = get_profile("longlive1"), get_profile("causvid")
        staged = max(2, -(-longlive1.recache.recache_latent_frames // longlive1.chunk_latent_frames))
        self.assertEqual(staged, 4)
        for representation in ("raw", "svd_hevc", "svd", "hevc", "latents_only"):
            config = bridge_runtime_config(longlive1, store_root="/tmp/x", representation=representation)
            self.assertEqual(config["representation"], representation)
            self.assertEqual(config.get("max_staged_canvases"),
                             staged if representation in ("svd_hevc", "hevc") else None, representation)
            self.assertNotIn("max_staged_canvases",
                             bridge_runtime_config(causvid, store_root="/tmp/x", representation=representation))
        self.assertEqual(bridge_runtime_config(longlive1, store_root="/tmp/x", representation="hevc",
                                               max_staged_canvases=8)["max_staged_canvases"], 8)
        for representation in ("lossless", "HEVC", "svd-hevc", "", None):
            with self.assertRaises(ValueError):
                bridge_runtime_config(causvid, store_root="/tmp/x", representation=representation)


# -- W1.4.1 (N2): selective recompute -- the deepest cached ancestor installed, only the rest replayed ---------------
SINK, WINDOW = FRAMES * TPF, 2 * FRAMES * TPF  # toy rolling geometry: block 0 is the sink, the window two chunks
CHUNK_ENTRY_BYTES = FRAMES * TPF * 64 * 2 * 2 * LAYERS  # one chunk's K and V (bf16, 2 heads x 32) over the layers


class WindowedCache(GuardedCache):
    """A sink + rolling-window toy cache (LongLive-like visibility; SGLang's own ring buffer runs in the GPU checks):
    the next chunk attends block 0 (the sink) and the two newest chunks; tokens the window rolls past are zeroed (a
    ring buffer overwrites them), so a chunk's K/V depend on its visible set only -- what a RAW restore installs."""

    def __init__(self, capacity=CAPACITY):
        super().__init__(capacity)
        self.sink_tokens = SINK

    def update(self, key, value, start):
        super().update(key, value, start)
        stale = self.end - WINDOW
        if stale > SINK:
            self.k[:, SINK:stale] = 0
            self.v[:, SINK:stale] = 0

    def replay_layout(self, appends, *, next_chunk=None):
        layout = super().replay_layout(appends, next_chunk=next_chunk)
        end = layout.local_end_index
        layout.sink_tokens = SINK
        if end - WINDOW > SINK:
            layout.live = (Span(0, SINK, 0), Span(end - WINDOW, end, end - WINDOW))
            layout.attended = layout.live if next_chunk else None
        return layout


class WindowedStage(TreeStage):
    def caches_for(self, streams):
        if self.caches is None:
            self.caches = {stream: [WindowedCache() for _ in range(LAYERS)] for stream in streams}
        return super().caches_for(streams)


class SelectiveRecomputeChecks:
    """A recompute restore that misses the raw tiers installs the deepest ancestor whose visible set the tiers hold (a
    RAW restore of it) and replays only the rest of the path: bitwise a full replay and the RAW restore of the same
    node (K/V, the next chunk's latents, RNG). ``CASES``: name -> (target, dropped path positions, (cached ancestor
    depth, installed nodes, replayed nodes, plans built by the search or None))."""

    windowed = False
    CASES: dict = {}
    _references: dict = {}  # (class, letter) -> the full replay's and the RAW restore's child (plain tensors)

    def stage(self):
        return WindowedStage() if self.windowed else TreeStage()

    def tiered(self, **overrides):
        values = {"gpu_raw_budget_bytes": 1 << 30, "host_raw_budget_bytes": 1 << 30, **overrides}
        bridge = self.recompute_bridge("tiered", **values)
        nodes, _ = self.build_a_to_f(bridge, stage=self.stage())
        bridge.archive.raw_tier.flush()
        return bridge, nodes

    def child(self, bridge, node, letter):
        bridge.drop_resident()
        return run_tree(bridge, self.request_for(handle("s", "rewind", checkpoint=node), PATHS[letter] + "G"),
                        stage=self.stage())

    def references(self, letter):
        """The same child from a full replay (no raw tier) and from the RAW archive (once per class and target)."""
        key = (type(self).__name__, letter)
        if key not in SelectiveRecomputeChecks._references:
            results = []
            for bridge in (self.recompute_bridge("full", raw_tier=False), self.raw_bridge("raw")):
                nodes, _ = self.build_a_to_f(bridge, stage=self.stage())
                results.append(self.child(bridge, nodes[letter], letter))
            SelectiveRecomputeChecks._references[key] = results
        return SelectiveRecomputeChecks._references[key]

    def drop(self, bridge, refs):
        tier = bridge.archive.raw_tier
        tier.drop(tier.entries(node_refs=set(refs)))

    def assert_exact(self, got, references):
        for expected in references:
            self.assertTrue(torch.equal(got[1], expected[1]))
            assert_caches_bitwise(self, got[2], expected[2])
            self.assertTrue(torch.equal(got[3].get_state(), expected[3].get_state()))

    def run_case(self, name):
        letter, drop, (depth, installed, replayed, plans), anchor_at = self.CASES[name]
        bridge, nodes = self.tiered()
        path = bridge.archive.path(nodes[letter])[1:]
        self.drop(bridge, [path[index].ref for index in drop])
        got = self.child(bridge, nodes[letter], letter)
        restore = got[0]["restore"]
        record = restore["recompute"]
        self.assertEqual((record["cached_ancestor_depth"], record["installed_nodes"], record["replayed_nodes"]),
                         (depth, installed, replayed), name)
        if plans is not None:
            self.assertEqual(record["search"]["plans_built"], plans, name)
        done = len(PATHS[letter])
        if depth == done:  # the target itself is cached: the install, no replay
            self.assertEqual((restore["mode"], restore["source"], restore["replay"]), ("archive", "raw_tier", None))
        else:
            self.assertEqual((restore["mode"], restore["source"]), ("recompute", "replay"))
            self.assertEqual(record["anchor"], None if anchor_at is None else path[anchor_at].ref, name)
            self.assertEqual(replayed, len(path) - (0 if anchor_at is None else anchor_at + 1))  # every later node
            replay = restore["replay"]
            self.assertTrue(replay["completed"])
            self.assertEqual(replay["chunk_forwards"], done - depth)
        self.assert_exact(got, self.references(letter))
        return bridge, nodes, got

    def test_prefix_cached(self):
        self.run_case("prefix")

    def test_middle_missing(self):
        self.run_case("middle")

    def test_none_cached(self):
        self.run_case("none")

    def test_all_cached(self):
        self.run_case("all")


class SelectiveRecomputeTests(SelectiveRecomputeChecks, RecomputeTestCase):
    """Full history (CausVid / RAVEN): every chunk of the ancestor is attended. Path F = A0 B1 C2 D3 F4."""

    CASES = {"prefix": ("F", [4], (4, 4, 1, 1), 3), "middle": ("F", [1], (1, 1, 4, 2), 0),
             "none": ("F", [0], (0, 0, 5, 1), None), "all": ("F", [], (5, 5, 0, None), None)}

    def test_a_corrupted_installed_ancestor_is_detected(self):
        """Negative control: the bitwise comparison sees the installed ancestor (a wrong tier copy changes it)."""
        bridge, nodes = self.tiered()
        path = bridge.archive.path(nodes["F"])[1:]
        self.drop(bridge, [path[4].ref])
        entry = bridge.archive.raw_tier.entries(node_refs=[path[2].ref])[0]  # C: installed for anchor D
        entry.groups[0].device[(MatrixKey(0, "K"), "cond")].add_(1)
        got = self.child(bridge, nodes["F"], "F")
        self.assertEqual(got[0]["restore"]["recompute"]["cached_ancestor_depth"], 4)
        full = self.references("F")[0]
        self.assertFalse(torch.equal(got[1], full[1]) and all(
            torch.equal(a.k[:, :a.end], b.k[:, :b.end]) for a, b in zip(got[2]["cond"], full[2]["cond"])))

    def test_a_replayed_promotion_hits_after_a_selective_replay(self):
        """N2 + promotion: the replayed nodes are promoted (LRU insert), the installed ones skip as present; the next
        restore of the same node is a full install, bitwise."""
        bridge, nodes = self.tiered(raw_tier_retain="lru", raw_tier_promote="host")
        path = bridge.archive.path(nodes["F"])[1:]
        self.drop(bridge, [path[3].ref, path[4].ref])
        got = self.child(bridge, nodes["F"], "F")
        restore = got[0]["restore"]
        self.assertEqual(restore["recompute"]["cached_ancestor_depth"], 3)
        self.assertEqual(restore["promotion"]["nodes"], 2)  # D and F (A, B, C were installed: present)
        self.assertEqual(restore["promotion"]["skips"].get("exists"), 3)
        bridge.archive.raw_tier.flush()
        again = self.child(bridge, nodes["F"], "F")
        self.assertEqual(again[0]["restore"]["recompute"]["cached_ancestor_depth"], 5)
        self.assert_exact(again, self.references("F"))


class SelectiveRecomputeWindowedTests(SelectiveRecomputeChecks, RecomputeTestCase):
    """Sink + rolling window: the ancestor's visible set is the sink and its two newest chunks, so chunks older than
    the window may be missing. Path F = A0 B1 C2 D3 F4; the next block of chunk k >= 3 attends A, k-1, k."""

    windowed = True
    CASES = {"prefix": ("F", [4], (4, 3, 1, 1), 3), "middle": ("F", [1], (5, 3, 0, None), None),
             "none": ("F", [0], (0, 0, 5, 1), None), "all": ("F", [], (5, 3, 0, None), None),
             "middle_and_target": ("F", [1, 4], (4, 3, 1, 1), 3), "window_gap": ("F", [2, 4], (2, 2, 3, 2), 1)}

    def test_middle_and_target_missing(self):
        self.run_case("middle_and_target")  # full history would replay from A

    def test_a_window_gap_skips_to_below_it(self):
        self.run_case("window_gap")


class SelectiveRecomputeRecacheTests(SelectiveRecomputeChecks, RecomputeTestCase):
    """LongLive-1.0-like recache epochs (full history): path F = A0 B1 R2a2 R2b3 C4 R3a5 R3b6 D7 R4a8 R4b9 F10 (``Rk``:
    the epoch before block k, one range node per re-materialized chunk k-2, k-1); an epoch's last node (R2b, R3b, R4b)
    is a state too. An ancestor's visible set holds the versions valid at it (R4b: R2a R3a R4a R4b; D: R2a R3a R3b D),
    later epochs are replayed at their positions."""

    recache = True
    CASES = {"prefix": ("F", [10], (4, 4, 1, 1), 9), "middle": ("F", [5], (3, 3, 6, 2), 4),
             "none": ("F", [0, 2], (0, 0, 11, 2), None), "all": ("F", [], (5, 5, 0, None), None),
             "branch": ("E", [3, 4], (2, 2, 3, 1), 1), "branch_epoch": ("E", [4], (2, 2, 1, 1), 3)}

    def test_epochs_on_a_branch_are_replayed_from_the_branch_point(self):
        bridge, nodes, got = self.run_case("branch")
        replay = got[0]["restore"]["replay"]
        self.assertEqual((replay["chunk_forwards"], replay["recache_forwards"]), (1, 1))
        self.assertIn("switch-2-pe", replay["epochs"][0])  # E's own epoch (prompt pe), not C's

    def test_a_cached_epoch_end_is_the_ancestor(self):
        bridge, nodes, got = self.run_case("branch_epoch")  # E's epoch installed from the tiers: only E replayed
        replay = got[0]["restore"]["replay"]
        self.assertEqual((replay["chunk_forwards"], replay["recache_forwards"]), (1, 0))


class SelectiveRecomputeWindowedRecacheTests(SelectiveRecomputeChecks, RecomputeTestCase):
    """Rolling window + recache epochs: F's next block attends R2a (the sink chunk), R4b (chunk 3) and F; R4b's attends
    R2a, R4a and R4b (positions as in :class:`SelectiveRecomputeRecacheTests`)."""

    windowed = True
    recache = True
    CASES = {"prefix": ("F", [10], (4, 3, 1, 1), 9), "middle": ("F", [1], (5, 3, 0, None), None),
             "none": ("F", [0, 2], (0, 0, 11, 2), None), "all": ("F", [], (5, 3, 0, None), None)}


def tmpfs_store(test) -> Path:
    """A fresh store directory on tmpfs (``/dev/shm``). W1.4.3 (delta review, moderate b): the tmpfs confirmations
    REQUIRE it -- the test is skipped, with this reason, only when ``/dev/shm`` is not a writable tmpfs mount (the
    receipt nodes have one; the filesystem-independent variants run everywhere)."""
    try:
        with open("/proc/mounts", encoding="utf-8") as handle:
            mounts = {fields[1]: fields[2] for fields in (line.split() for line in handle) if len(fields) > 2}
    except OSError:
        mounts = {}
    if mounts.get("/dev/shm") != "tmpfs" or not os.access("/dev/shm", os.W_OK):
        test.skipTest(f"/dev/shm is not a writable tmpfs mount on this host (type {mounts.get('/dev/shm')!r}): the "
                      "tmpfs store confirmation needs one")
    root = Path(tempfile.mkdtemp(prefix="reelcache-w143-", dir="/dev/shm"))
    test.addCleanup(shutil.rmtree, root, True)
    return root / "store"


def object_file_bytes(root: Path) -> int:
    """Bytes the store's object files occupy (allocated blocks: on tmpfs, memory)."""
    return sum(path.stat().st_blocks * 512 for path in (root / "objects").rglob("*") if path.is_file())


class NossdConfirmationTests(RecomputeTestCase):
    """W1.4.1 ("nossd" confirmations): the recompute arm GPU raw only (``host_raw_budget_bytes=0``) under tree mode,
    retain-on-release and the drain-free switch; stores on tmpfs with ``store_io="nocache"``."""

    def test_a_zero_host_budget_is_gpu_raw_only(self):
        bridge = self.recompute_bridge("gpu-only", gpu_raw_budget_bytes=3 * CHUNK_ENTRY_BYTES,
                                       host_raw_budget_bytes=0, raw_tier_retain="lru", tree_switch="detach")
        tier = bridge.archive.raw_tier
        nodes, _ = self.build_a_to_f(bridge)
        raw = self.raw_bridge()
        raw_nodes, _ = self.build_a_to_f(raw)

        def host_free():
            stats = tier.stats()
            self.assertTrue(stats["enabled"], stats["disabled_reason"])
            self.assertEqual((stats["gpu_only"], stats["degraded"], stats["host_bytes"], stats["host_peak_bytes"],
                              stats["host_pool_bytes"], stats["entries"]["host_only"], stats["entries"]["both"],
                              stats["spills"]), (True, 0, 0, 0, 0, 0, 0, 0))
            self.assertLessEqual(stats["gpu_bytes"], 3 * CHUNK_ENTRY_BYTES)
            return stats

        stats = host_free()
        self.assertGreater(stats["entries"]["device_only"], 0)
        self.assertGreater(stats["evictions_gpu"], 0)  # beyond the GPU budget: deleted, not demoted
        described = bridge.describe()["archive"]["raw_tier"]  # the configured budgets, as the runner records them
        self.assertEqual((described["gpu_raw_budget_bytes"], described["host_raw_budget_bytes"],
                          described["gpu_only"]), (3 * CHUNK_ENTRY_BYTES, 0, True))
        sources = set()
        for letter in "ABCDEF":
            got = self.restore_child(bridge, nodes[letter], letter)
            expected = self.restore_child(raw, raw_nodes[letter], letter)
            sources.add(got[0]["restore"]["source"])
            self.assertTrue(torch.equal(got[1], expected[1]), letter)
            assert_caches_bitwise(self, got[2], expected[2])
            host_free()
        self.assertIn("replay", sources)  # evicted nodes are recomputed
        self.assertEqual(ReelCacheRuntimeConfig.from_dict({"store_root": "x", "representation": "latents_only",
                                                        "restore_mode": "recompute",
                                                        "host_raw_budget_bytes": 0}).host_raw_budget_bytes, 0)

    def test_a_tmpfs_store_with_nocache_publishes_restores_and_frees_on_delete(self):
        """Required tmpfs confirmation (``/dev/shm``): see :meth:`_nocache_store`."""
        root = tmpfs_store(self)
        self._nocache_store(root)
        self.assertEqual(os.stat(root).st_dev, os.stat("/dev/shm").st_dev)  # the store is on the tmpfs mount

    def test_a_nocache_store_publishes_restores_and_frees_on_delete(self):
        """The same, filesystem-independent (the test workspace)."""
        self._nocache_store(self.workspace / "store-nocache")

    def _nocache_store(self, root: Path) -> None:
        """``store_io="nocache"`` with the nossd raw_offload configuration: the store's own object bytes grow with
        publication, restores read the stored latents (every raw copy dropped: a replay, bitwise RAW), the advice
        reports no error, and a whole-tree delete brings the object bytes and refs back to the empty store's."""
        bridge = self.bridge("unused", store_root=str(root), restore_mode="recompute", store_io="nocache",
                             gpu_raw_budget_bytes=1 << 30, host_raw_budget_bytes=1 << 30, raw_tier_retain="lru",
                             raw_tier_promote="host")
        store = bridge.archive.store
        objects = Path(store.root)
        self.assertTrue(str(objects.resolve()).startswith(str(Path(root).resolve())))
        empty = (object_file_bytes(objects), store.list_refs(""))
        nodes, _ = self.build_a_to_f(bridge)
        written = object_file_bytes(objects)
        self.assertGreater(written, empty[0])
        bridge.control("raw_tier", {"action": "drop"})  # every raw copy gone: the restores read the stored latents
        raw = self.raw_bridge()
        raw_nodes, _ = self.build_a_to_f(raw)
        for letter in "DF":
            got = self.restore_child(bridge, nodes[letter], letter)
            self.assertEqual(got[0]["restore"]["source"], "replay")
            expected = self.restore_child(raw, raw_nodes[letter], letter)
            self.assertTrue(torch.equal(got[1], expected[1]), letter)
            assert_caches_bitwise(self, got[2], expected[2])
        io = store.io_stats()
        self.assertEqual((io["policy"], io["errors"]), ("nocache", 0))  # fadvise (a no-op on tmpfs) never fails
        self.assertGreater(io["read_calls"], 0)
        self.assertGreater(io["write_calls"], 0)
        bridge.control("delete", {"session_id": "s"})
        self.assertEqual((object_file_bytes(objects), store.list_refs("")), empty)  # the tree's (tmpfs) bytes freed

    def test_the_recompute_arm_without_a_raw_tier_on_tmpfs(self):
        """Required tmpfs confirmation (``/dev/shm``): see :meth:`_no_tier`."""
        root = tmpfs_store(self)
        self._no_tier(root)
        self.assertEqual(os.stat(root).st_dev, os.stat("/dev/shm").st_dev)

    def test_the_recompute_arm_without_a_raw_tier(self):
        """The same, filesystem-independent (the test workspace)."""
        self._no_tier(self.workspace / "store-no-tier")

    def _no_tier(self, root: Path) -> None:
        """User decision (coordinator 00:30): the recompute arm keeps no KV (``raw_tier=False``) -- with tree mode, the
        drain-free switch and ``store_io="nocache"`` every restore is a full replay, bitwise the RAW restore, and a
        whole-tree delete frees the tree's object bytes."""
        bridge = self.bridge("unused", store_root=str(root), restore_mode="recompute", raw_tier=False,
                             tree_switch="detach", store_io="nocache")
        self.assertIsNone(bridge.archive.raw_tier)
        store = bridge.archive.store
        objects = Path(store.root)
        empty = (object_file_bytes(objects), store.list_refs(""))
        nodes, _ = self.build_a_to_f(bridge)
        self.assertGreater(object_file_bytes(objects), empty[0])
        raw = self.raw_bridge()
        raw_nodes, _ = self.build_a_to_f(raw)
        for letter in "ABCDEF":
            got = self.restore_child(bridge, nodes[letter], letter)
            restore = got[0]["restore"]
            self.assertEqual((restore["mode"], restore["source"], restore["recompute"]["cached_ancestor_depth"],
                              restore["replay"]["chunk_forwards"]), ("recompute", "replay", 0, len(PATHS[letter])))
            expected = self.restore_child(raw, raw_nodes[letter], letter)
            self.assertTrue(torch.equal(got[1], expected[1]), letter)
            assert_caches_bitwise(self, got[2], expected[2])
        self.assertIsNone(bridge.describe()["archive"]["raw_tier"])
        self.assertEqual(store.io_stats()["errors"], 0)
        bridge.control("delete", {"session_id": "s"})
        self.assertEqual((object_file_bytes(objects), store.list_refs("")), empty)


def stage_replay_functions() -> dict:
    """The profile stage's replay functions, executed from ``sglang_stages.py``'s source with stub SGLang names (the
    CPU harness has no SGLang; the module itself is not importable here). Returns the namespace."""
    import ast

    path = Path(__file__).resolve().parent.parent / "reelcache_sglang_models/causal_wan/sglang_stages.py"
    tree = ast.parse(path.read_text())
    wanted = {"_private_cross_caches", "_check_rope_start", "_replay_profile", "_replay_blocks"}
    module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted],
                        type_ignores=[])

    class CrossAttentionKVCache:
        def __init__(self, k, v):
            self.k, self.v, self.is_init = k, v, False

    namespace = {"__name__": "reelcache_sglang_models.causal_wan._replay_extract", "torch": torch,
                 "__package__": "reelcache_sglang_models.causal_wan", "Req": object, "ServerArgs": object,
                 "SEGMENTS_KEY": "segments", "CrossAttentionKVCache": CrossAttentionKVCache}
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - the repository's own source
    return namespace


class RecordingStage:
    """Records the conditioning of every replayed forward (profile and LongLive 2.0 stage seams)."""

    def __init__(self):
        self.calls = []
        self.transformer = SimpleNamespace(num_attention_heads=1, attention_head_dim=1)
        self.num_transformer_blocks = 1
        self.profile = SimpleNamespace(chunk_latent_frames=3)
        self._rope_temporal_offset = 0.0

    def _get_max_text_len(self, server_args):
        return 1

    def _causal_rope_start_frame(self, start_frame):
        return float(start_frame)

    def _segment_embeds(self, embeds, segment, count):
        return f"prompt-segment-{segment}"

    def _reset_crossattn_cache(self, cross):
        return None

    def _update_causal_context_cache(self, batch, server_args, **kwargs):
        context = kwargs["context_input"]
        self.calls.append(("forward", kwargs["prompt_embeds"], kwargs["current_start_tokens"], kwargs["start_frame"],
                           tuple(context.shape), float(context.sum()), str(kwargs.get("pos_cond_kwargs"))))

    def run_prompt_switch_recache(self, batch, server_args, **kwargs):
        self.calls.append(("recache", kwargs["switch_before_chunk"], kwargs["new_prompt_embeds"],
                           kwargs["previous_segment"], kwargs["new_segment"]))
        return SimpleNamespace(plan={"frame_seq": 1560, "recompute_tokens": [1560 * 3, 1560 * 15],
                                     "written_global_frames": [6, 15]})

    # LongLive 2.0 (SGLang's causal stage) seams
    def _set_rope_temporal_offset(self, batch, shot):
        self.calls.append(("offset", shot))

    def _shot_index(self, batch, block):
        return block // 2

    def _is_scene_cut(self, batch, block):
        return block % 2 == 0

    def _reset_crossattn_cache_for_block(self, batch, *caches):
        return None

    def _select_block_prompt_embeds(self, batch, embeds, block):
        return f"{embeds}-block-{block}"

    def _select_block_cond_kwargs(self, batch, kwargs, block):
        return f"cond-{block}"

    def _get_negative_prompt_embeds(self, batch):
        return "negative"

    def _prepare_causal_dmd_neg_cond_kwargs(self, batch, server_args, dtype):
        return "neg-cond"

    def _pin_current_chunk(self, kv_cache, frames):
        self.calls.append(("pin", kv_cache, frames))


class StageReplaySkipTests(unittest.TestCase):
    """W1.4.1 (N2), the stage side: with ``first_step`` the profile stage and LongLive 2.0's loop replay exactly the
    plan's steps from it on, each under the same conditioning as in the full replay (the GPU checks run them)."""

    def setUp(self):
        self.functions = stage_replay_functions()
        self.latents = torch.arange(1 * 2 * 30, dtype=torch.float32).reshape(1, 2, 30, 1, 1)
        self.ctx = SimpleNamespace(latents=self.latents, batch_size=1, target_dtype=torch.float32, prompt_embeds="p",
                                   image_kwargs=None, pos_cond_kwargs="pos", autocast_enabled=False)

    def chunk(self, block):
        return {"kind": "chunk", "block": block, "start_frame": 3 * block, "num_frames": 3,
                "token_start": 1560 * 3 * block, "rope_start_frame": float(3 * block), "scene_cut": block % 2 == 0}

    def profile_steps(self):
        recache = {"kind": "recache", "precedes_block": 5, "epoch": "e5",
                   "window": {"token_start": 1560 * 3, "num_tokens": 1560 * 12}, "written": [[1560 * 6, 1560 * 9]]}
        return [self.chunk(0), self.chunk(1), self.chunk(2), self.chunk(3), self.chunk(4), recache, self.chunk(5),
                self.chunk(6)]

    def replay(self, name, steps, first_step, **extra):
        stage = RecordingStage()
        forwards = {"chunk": 0, "recache": 0}
        batch = SimpleNamespace(extra={"segments": {"switch_chunks": [5], "count": 2}})
        self.functions[name](stage, batch, None, steps=steps, ctx=self.ctx, kv_cache="kv", latents=self.latents,
                             forwards=forwards, first_step=first_step, **extra)
        return stage.calls, forwards

    def check(self, name, steps, **extra):
        full, counts = self.replay(name, steps, 0, **extra)
        self.assertEqual(sum(counts.values()), len(steps))
        per_step = []  # the full replay's calls, split per step
        for step in steps:
            forwards = 1 + int(extra.get("use_cfg", False)) if step["kind"] == "chunk" else 1
            calls = []
            while full and len([c for c in calls if c[0] in ("forward", "recache")]) < forwards:
                calls.append(full.pop(0))
            while full and full[0][0] == "pin":
                calls.append(full.pop(0))
            per_step.append(calls)
        for start in range(1, len(steps)):
            calls, counts = self.replay(name, steps, start, **extra)
            self.assertEqual(sum(counts.values()), len(steps) - start, start)
            self.assertEqual(calls, [call for item in per_step[start:] for call in item], start)

    def test_profile_stage_replays_from_start_with_the_full_replays_conditioning(self):
        self.check("_replay_profile", self.profile_steps())

    def test_longlive2_loop_replays_from_start_with_the_full_replays_conditioning(self):
        self.check("_replay_blocks", [self.chunk(block) for block in range(6)], kv_cache_neg="kv-neg", use_cfg=True)


if __name__ == "__main__":
    unittest.main()
