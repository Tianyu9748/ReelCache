"""Workload v2 W1b: durable session trees (``tree_retention="full"``) on the CPU toy harness, without SGLang.

The toy causal model of ``tests.test_sglang_bridge`` (K/V depend on the prompt, the block, the layer, the stream and
the cached history) drives the bridge like SGLang's stage, with a per-block prompt schedule (one new prompt per
request) and, optionally, a LongLive-1.0-like recache epoch at every prompt change. The workload plan's A-F example:
A1 -> B2 -> C3 -> D4; rewind to B (k=2) and generate E5; target D (k=1, not an ancestor of E) and generate F6.
"""

import itertools
import threading
import unittest
from types import SimpleNamespace

import torch

from reelcache.errors import ConfigurationError, IntegrityError, StateError, UnsupportedError
from reelcache.integrations.sglang import ReelCacheRuntimeConfig
from reelcache.integrations.sglang.nodes import (TREE_HOLDER_SUFFIX, partial_recache_node, replay_steps, tree_entries,
                                              tree_prefix, tree_ref)
from tests.test_sglang_bridge import (FRAMES, GRID, LAYERS, PROFILE, TPF, BridgeTestCase, Capture, Chunk,
                                      FullHistoryCache, Recache, control_error, handle, request, rope_cos_sin,
                                      rotate_key, stored, toy_kv)

CAPACITY = 8 * FRAMES * TPF
PROMPTS = {"A": "pa", "B": "pb", "C": "pc", "D": "pd", "E": "pe", "F": "pf", "G": "pg", "H": "ph"}
_IDS = itertools.count()


class GuardedCache(FullHistoryCache):
    """``FullHistoryCache`` + SGLang's ``restore_pending`` seam: the first K/V access waits on it (once)."""

    def __init__(self, capacity=CAPACITY):
        super().__init__(capacity)
        self.restore_pending = None

    def await_restored_history(self):
        pending = self.restore_pending
        if pending is not None:
            self.restore_pending = None
            pending.wait()

    def update(self, key, value, start):
        self.await_restored_history()
        super().update(key, value, start)

    def overwrite(self, start, key, value):
        self.await_restored_history()
        super().overwrite(start, key, value)


class TreeStage:
    """Caches kept across requests (cursors reset per request), like SGLang's stage."""

    def __init__(self):
        self.caches = None

    def caches_for(self, streams):
        if self.caches is None or tuple(self.caches) != tuple(streams):
            self.caches = {stream: [GuardedCache() for _ in range(LAYERS)] for stream in streams}
        else:
            for caches in self.caches.values():
                for cache in caches:
                    cache.reset_indices()
        return self.caches


def segment_ids(prompts):
    """Per-block conditioning ids like ``lifecycle.block_conditioning_ids``: ``s<segment>:<prompt>``."""
    ids, segment = [], 0
    for index, prompt in enumerate(prompts):
        if index and prompt != prompts[index - 1]:
            segment += 1
        ids.append(f"s{segment}:{prompt}")
    return tuple(ids)


def tree_request(session, prompts, *, recache=False, streams=("cond",), **metadata):
    """A request generating ``len(prompts)`` blocks, block ``i`` under ``prompts[i]`` (per-block conditioning; the
    request conditioning stays the session's). ``recache``: a toy LongLive-1.0 recache epoch at every prompt change
    from block 2 on. ``metadata``: optional ``generation_index`` / ``workload`` fields (wl-sglang)."""
    req = request(session, blocks=len(prompts), streams=streams)
    req.request_id = f"r-{next(_IDS)}"
    req.block_conditioning = segment_ids(prompts)
    req.block_prompts = list(prompts)
    req.recache_switches = recache
    for key, value in metadata.items():
        setattr(req, key, value)
    return req


def toy_forward(caches, streams, prompt, chunk, run=None, capture=None):
    cos, sin = rope_cos_sin(chunk.rope_start_frame, chunk.num_frames)
    for stream in streams:
        item = None if capture is None else Capture(chunk, run.stream_of(caches[stream]), capture)
        for layer in range(LAYERS):
            cache = caches[stream][layer]
            cache.await_restored_history()  # SGLang: the cache's first access, before any read
            key, value = toy_kv(prompt, chunk, layer, stream, cache)
            if item is not None:
                run.capture_layer(item, layer, key, value, (cos, sin))
            if chunk.index < 0:
                cache.overwrite(chunk.token_start, rotate_key(key, cos, sin), value)
            else:
                cache.update(rotate_key(key, cos, sin), value, chunk.token_start)
        if item is not None:
            run.end_clean_forward(item)


def recache_window(at):
    first = at - 2
    return Chunk(-1, first * FRAMES, 2 * FRAMES, first * FRAMES * TPF, 2 * FRAMES * TPF, float(first * FRAMES),
                 False, 0)


def toy_recache(run, req, caches, at, *, capture=True):
    """Toy LongLive-1.0 prompt-switch recache before block ``at``: chunks at-2..at-1 under block ``at``'s prompt."""
    prompt = req.block_prompts[at]
    window = recache_window(at)
    toy_forward(caches, req.streams, prompt, window, run, "kv_recache" if capture else None)
    if capture:
        return run.end_recache(Recache(window, ((window.token_start, window.num_tokens),), f"switch-{at}-{prompt}",
                                       details={"prompt": prompt}))
    return None


def replay(run, req, caches, *, skip=None):
    """The stage side of the recompute arm: the replay plan's clean forwards, without capture."""
    plan = run.begin_replay()
    for index, step in enumerate(plan["steps"]):
        if index == skip or index < plan.get("start_step", 0):  # W1.4.1 (N2): an installed ancestor's steps
            continue
        if step["kind"] == "chunk":
            chunk = Chunk(step["block"], step["start_frame"], step["num_frames"], step["token_start"],
                          step["num_tokens"], step["rope_start_frame"], False, 0)
            toy_forward(caches, req.streams, req.block_prompts[step["block"]], chunk)
        else:
            toy_recache(None, req, caches, step["precedes_block"], capture=False)
    return run.end_replay(details={"toy": True})


def run_tree(bridge, req, *, stage=None, do_replay=True, replay_skip=None):
    """One request through the bridge like SGLang's stage; returns (summary, latents, caches, generator)."""
    caches = stage.caches_for(req.streams) if stage is not None else {
        stream: [GuardedCache() for _ in range(LAYERS)] for stream in req.streams}
    latents = torch.zeros(1, 4, sum(req.block_sizes), *GRID)
    generator = torch.Generator().manual_seed(req.conditioning["seed"])
    torch.randn(3, generator=generator)  # latent preparation draws
    run = bridge.begin_request(req)
    target = SimpleNamespace(caches=caches, latents=latents, generator=generator, rope_cos_sin=rope_cos_sin,
                             rotate_key=rotate_key, device=torch.device("cpu"))
    try:
        resume = run.restore(target)
        run.block_prompts = list(req.block_prompts)  # the stage hands the run its per-block prompt texts
        if do_replay and getattr(run, "replay_plan", None) is not None:
            replay(run, req, caches, skip=replay_skip)
        start = sum(req.block_sizes[:resume])
        for index in range(resume, len(req.block_sizes)):
            frames = req.block_sizes[index]
            prompt = req.block_prompts[index]
            if req.recache_switches and index >= 2 and prompt != req.block_prompts[index - 1]:
                report = run.restore_report or {}
                restored = index == resume and report.get("head_kind") == "recache" and report.get("head_block") == index
                if not restored:
                    toy_recache(run, req, caches, index)
            chunk = Chunk(index, start, frames, start * TPF, frames * TPF, float(start), False, 0)
            reference = caches["cond"][0]
            reference.await_restored_history()  # (N5) an overlapped restore of it lands before this read, as in SGLang
            history = reference.k[:, :reference.end].float().mean() if reference.end else 0.0
            noise = torch.randn(1, 4, frames, *GRID, generator=generator)
            latents[:, :, start:start + frames] = noise + float(history)
            toy_forward(caches, req.streams, prompt, chunk, run, "clean_commit")
            start += frames
            if run.end_chunk(chunk, latents[:, :, start - frames:start].clone()):
                break
        return run.finish(error=None), latents, caches, generator
    except BaseException as exc:
        run.finish(error=exc)
        raise


def assert_caches_bitwise(test, left, right):
    for stream in left:
        for a, b in zip(left[stream], right[stream]):
            test.assertEqual(a.end, b.end)
            test.assertTrue(torch.equal(a.k[:, :a.end], b.k[:, :b.end]))
            test.assertTrue(torch.equal(a.v[:, :a.end], b.v[:, :b.end]))


class TreeTestCase(BridgeTestCase):
    representation = "raw"
    tree = "full"
    recache = False
    streams = ("cond",)

    def bridge(self, store="archive", profile=None, **overrides):
        values = {"tree_retention": self.tree, **overrides}
        if profile is None:
            return super().bridge(store, **values)
        return super().bridge(store, profile=profile, **values)

    def request_for(self, session, letters, **metadata):
        return tree_request(session, [PROMPTS[letter] for letter in letters], recache=self.recache,
                            streams=self.streams, **metadata)

    def head(self, bridge, session_id="s"):
        return bridge.checkpoint(session_id)["head"]

    def fresh(self, letters):
        """A fresh native run of the whole path (a new session in its own store, generated in one request)."""
        bridge = self.bridge(f"fresh-{''.join(letters)}-{next(_IDS)}")
        summary, latents, caches, rng = run_tree(bridge, self.request_for(handle("ref"), letters))
        return latents, caches, rng

    def build_a_to_f(self, bridge, *, explicit_indices=False, stage=None):
        """A1 B2 C3 D4 (one resume each), then E5 from B (k=2) and F6 from D (k=1, not an ancestor of E)."""
        nodes, results = {}, {}
        for depth, letter in enumerate("ABCD", start=1):
            letters = "ABCD"[:depth]
            action = "generate" if depth == 1 else "resume"
            metadata = {"generation_index": depth} if explicit_indices else {}
            results[letter] = run_tree(bridge, self.request_for(handle("s", action), letters, **metadata), stage=stage)
            nodes[letter] = self.head(bridge)
        metadata = {"generation_index": 5} if explicit_indices else {}
        results["E"] = run_tree(bridge, self.request_for(handle("s", "rewind", checkpoint=nodes["B"]), "ABE",
                                                         **metadata), stage=stage)
        nodes["E"] = self.head(bridge)
        metadata = {"generation_index": 6} if explicit_indices else {}
        results["F"] = run_tree(bridge, self.request_for(handle("s", "rewind", checkpoint=nodes["D"]), "ABCDF",
                                                         **metadata), stage=stage)
        nodes["F"] = self.head(bridge)
        return nodes, results


class TreeChecks:
    """Tree retention on the exact RAW archive (both prompt-change styles)."""

    def test_a_to_f_example_is_exact_and_every_node_is_owned(self):
        bridge = self.bridge()
        nodes, results = self.build_a_to_f(bridge)
        controller = bridge.controller
        # The registry: generation index -> node, durable in the store as tree/<session>/<index:08d>.
        expected = {index: nodes[letter] for index, letter in enumerate("ABCDEF", start=1)}
        self.assertEqual(controller.tree("s"), expected)
        self.assertEqual(controller.get("s").tree, expected)
        self.assertEqual(sorted(bridge.archive.store.list_refs(tree_prefix("s"))),
                         [tree_ref("s", index) for index in range(1, 7)])
        # E and F are siblings / cousins, not ancestors of the head path of each other.
        path_e = [node.ref for node in bridge.archive.path(nodes["E"])]
        path_f = [node.ref for node in bridge.archive.path(nodes["F"])]
        self.assertIn(nodes["B"], path_e)
        self.assertNotIn(nodes["D"], path_e)
        self.assertIn(nodes["D"], path_f)
        self.assertEqual(results["F"][0]["restore"]["switch"]["ancestor"], False)
        self.assertEqual(results["E"][0]["restore"]["switch"]["ancestor"], True)
        self.assertGreaterEqual(results["F"][0]["restore"]["switch"]["drain_seconds"], 0.0)
        # Every new chunk equals a fresh native run along the same path with the same prompts (bitwise).
        for letter, path in (("A", "A"), ("B", "AB"), ("C", "ABC"), ("D", "ABCD"), ("E", "ABE"), ("F", "ABCDF")):
            latents, caches, rng = self.fresh(path)
            _, got_latents, got_caches, got_rng = results[letter]
            self.assertTrue(torch.equal(got_latents, latents), letter)
            assert_caches_bitwise(self, got_caches, caches)
            self.assertTrue(torch.equal(got_rng.get_state(), rng.get_state()), letter)
        # Sidecar (unhashed): generation index, parent, request id, depth, conditioning ids, prompt schedule.
        sidecar = bridge.archive.store.telemetry(nodes["E"])["bridge"]["tree"]
        self.assertEqual((sidecar["generation_index"], sidecar["parent"], sidecar["depth"]), (5, nodes["B"], 3))
        self.assertEqual(sidecar["conditioning_ids"], list(segment_ids(["pa", "pb", "pe"])))
        self.assertEqual(sidecar["prompt_schedule"]["source"], "stage")
        self.assertEqual([item["text"] for item in sidecar["prompt_schedule"]["blocks"]], ["pa", "pb", "pe"])
        self.assertEqual(len(sidecar["prompt_schedule"]["blocks"][2]["sha256"]), 64)
        self.assertEqual(sidecar["request_id"], results["E"][0]["request_id"])
        self.assertFalse(sidecar["deduplicated"])

    def test_every_owned_node_stays_restorable_and_old_branches_are_targets(self):
        bridge = self.bridge()
        nodes, _ = self.build_a_to_f(bridge)
        stage = TreeStage()
        for index, letter in enumerate("ABCDEF", start=1):
            path = {"A": "A", "B": "AB", "C": "ABC", "D": "ABCD", "E": "ABE", "F": "ABCDF"}[letter]
            summary, latents, caches, _ = run_tree(bridge, self.request_for(
                handle("s", "rewind", checkpoint=nodes[letter]), path + "G"), stage=stage)
            self.assertEqual(summary["resumed_from_chunks"], len(path))
            fresh_latents, fresh_caches, _ = self.fresh(path + "G")
            self.assertTrue(torch.equal(latents, fresh_latents), letter)
            assert_caches_bitwise(self, caches, fresh_caches)
        self.assertEqual(len(bridge.controller.tree("s")), 12)  # six G children, each with its own index

    def test_regeneration_with_the_same_prompt_deduplicates_and_records_the_new_index(self):
        bridge = self.bridge()
        nodes, _ = self.build_a_to_f(bridge)
        # From B again under C's prompt: the restored generator state reproduces C bit for bit -> the same node.
        run_tree(bridge, self.request_for(handle("s", "rewind", checkpoint=nodes["B"]), "ABC"))
        again = self.head(bridge)
        self.assertEqual(again, nodes["C"])
        tree = bridge.controller.tree("s")
        self.assertEqual((tree[7], tree[3]), (nodes["C"], nodes["C"]))
        sidecar = bridge.archive.store.telemetry(nodes["C"])["bridge"]["tree"]
        self.assertEqual(sidecar["generation_index"], 7)
        self.assertEqual([(item["session_id"], item["generation_index"]) for item in sidecar["generations"]],
                         [("s", 3), ("s", 7)])
        self.assertTrue(sidecar["deduplicated"])

    def test_restart_rebuilds_the_registry_and_restores_old_branches(self):
        first = self.bridge("shared")
        nodes, _ = self.build_a_to_f(first, explicit_indices=True)
        registry = first.controller.tree("s")
        first.close()
        second = self.bridge("shared")  # a new worker on the same store
        state = second.controller.get("s")  # reopened from the persisted head; registry from the refs
        self.assertEqual(state.tree, registry)
        self.assertEqual(state.next_generation, 7)
        self.assertEqual(state.head_ref, nodes["F"])
        # E's branch is not on F's path: still a target after the restart; the new chunk is the fresh native one.
        summary, latents, caches, _ = run_tree(second, self.request_for(
            handle("s", "rewind", checkpoint=nodes["E"]), "ABEH"))
        self.assertEqual((summary["restore"]["mode"], summary["resumed_from_chunks"]), ("archive", 3))
        fresh_latents, fresh_caches, _ = self.fresh("ABEH")
        self.assertTrue(torch.equal(latents, fresh_latents))
        assert_caches_bitwise(self, caches, fresh_caches)
        self.assertEqual(second.controller.tree("s")[7], self.head(second))

    def test_gc_keeps_owned_nodes_and_whole_tree_delete_reclaims(self):
        bridge = self.bridge()
        store = bridge.archive.store
        empty = (store.list_refs(""), store.stats()["object_count"])
        nodes, _ = self.build_a_to_f(bridge)
        bridge.control("rewind", {"session_id": "s", "checkpoint": nodes["E"]})  # C, D and F are off the head path
        owned = set()
        for ref in nodes.values():
            owned.update(node.ref for node in bridge.archive.path(ref))
        objects = set()
        with store.transaction():
            objects = set(store._walk(list(nodes.values()), verify=False))
        collected = bridge.control("gc")["gc"]
        self.assertEqual(stored(bridge, owned), owned)
        self.assertEqual(stored(bridge, objects), objects)  # every packet / state of every owned node
        self.assertGreaterEqual(collected["live_objects"], len(objects))
        deleted = bridge.control("delete", {"session_id": "s"})
        self.assertEqual(sorted(deleted["tree_deleted"]), [tree_ref("s", index) for index in range(1, 7)])
        self.assertEqual(stored(bridge, objects), set())
        self.assertEqual((store.list_refs(""), store.stats()["object_count"]), empty)

    def test_targets_must_be_owned_clean_nodes(self):
        bridge = self.bridge()
        nodes, _ = self.build_a_to_f(bridge)
        run_tree(bridge, self.request_for(handle("t"), "AH"))  # another session: its node is not s's
        other = self.head(bridge, "t")
        head = self.head(bridge)
        for checkpoint in (other, "manifest:" + "0" * 64):
            with self.assertRaises(StateError):
                bridge.controller.rewind("s", checkpoint)
            control_error(self, "illegal_transition", "rewind", {"session_id": "s", "checkpoint": checkpoint},
                          bridge)
        self.assertEqual(self.head(bridge), head)
        # The core and the RAW archive refuse a non-owned node themselves (compare-and-swap of the head).
        with self.assertRaises(StateError):
            bridge.archive.rewind("s", other)
        self.assertEqual(self.head(bridge), head)


class RawTreeTests(TreeChecks, TreeTestCase):
    """RAW archive, cross-attention-only style prompt changes (no recache epochs)."""


class RawRecacheTreeTests(TreeChecks, TreeTestCase):
    """RAW archive with a LongLive-1.0-like recache epoch at every prompt change (chained range nodes)."""

    recache = True

    def test_partial_epoch_nodes_are_never_targets(self):
        bridge = self.bridge()
        nodes, _ = self.build_a_to_f(bridge)
        path = bridge.archive.path(nodes["D"])
        partial = [node for node in path if partial_recache_node(node)]
        whole = [node for node in path if node.kind == "recache" and not partial_recache_node(node)]
        self.assertEqual(len(partial), 2)  # epochs before blocks 2 and 3: two range nodes each, the first partial
        self.assertEqual(len(whole), 2)
        bridge.control("rewind", {"session_id": "s", "checkpoint": nodes["E"]})  # D's epochs are off the head path
        for node in partial:
            with self.assertRaises(StateError) as caught:
                bridge.controller.rewind("s", node.ref)
            self.assertIn("partially re-materialized", str(caught.exception))
        # The epoch's last node is a clean state: legal, and the restore reports it as the head.
        summary, _, _, _ = run_tree(bridge, self.request_for(handle("s", "rewind", checkpoint=whole[-1].ref),
                                                             "ABCD"))
        self.assertEqual((summary["restore"]["head_kind"], summary["restore"]["head_block"]), ("recache", 3))
        self.assertEqual(self.head(bridge), nodes["D"])  # the same block 3 content: deduplicated to D

    def test_replay_steps_merge_chained_epoch_ranges(self):
        bridge = self.bridge()
        nodes, _ = self.build_a_to_f(bridge)
        steps = replay_steps(bridge.archive.path(nodes["F"])[1:])
        self.assertEqual([step["kind"] for step in steps],
                         ["chunk", "chunk", "recache", "chunk", "recache", "chunk", "recache", "chunk"])
        self.assertEqual([step["precedes_block"] for step in steps if step["kind"] == "recache"], [2, 3, 4])
        self.assertTrue(all(len(step["nodes"]) == 2 and len(step["ranges"]) == 2
                            for step in steps if step["kind"] == "recache"))


#: W1.2 (pilot, wl-gen): the runner's ``session_max`` geometry -- every request sized for a fixed session total.
SESSION_TOTAL = 6
SESSION_MAX_STEPS = (("A", "generate", None, "A", 1), ("B", "resume", None, "AB", 2), ("C", "resume", None, "ABC", 3),
                     ("D", "resume", None, "ABCD", 4), ("E", "rewind", "B", "ABE", 5), ("F", "rewind", "D", "ABCDF", 6))


class SessionMaxChecks:
    """W1.2 regression: requests sized for a fixed session total (``SESSION_TOTAL`` blocks, the path's prompts padded
    with its last one) that pause after their one new chunk (``pause_after_chunks = depth + 1``), each naming that chunk
    with ``generation_index``. W1.1 counted ``len(block_sizes) - resume_blocks`` new chunks: every such request was
    refused (``generation_index`` below the count), or an accepted index was shifted down."""

    exact = True  # the archive restores exact history: rewinds equal fresh native runs bitwise

    def session_max(self, session, path, **metadata):
        return self.request_for(session, path + path[-1] * (SESSION_TOTAL - len(path)), **metadata)

    def build_session_max(self, bridge):
        nodes, results = {}, {}
        for letter, action, target, path, index in SESSION_MAX_STEPS:
            extra = {} if target is None else {"checkpoint": nodes[target]}
            session = handle("s", action, pause_after_chunks=len(path), **extra)
            results[letter] = run_tree(bridge, self.session_max(session, path, generation_index=index))
            nodes[letter] = self.head(bridge)
        return nodes, results

    def test_session_max_requests_name_their_one_new_chunk(self):
        bridge = self.bridge()
        nodes, results = self.build_session_max(bridge)
        self.assertEqual(bridge.controller.tree("s"), {index: nodes[letter] for letter, _, _, _, index
                                                      in SESSION_MAX_STEPS})
        self.assertEqual(sorted(bridge.archive.store.list_refs(tree_prefix("s"))),
                         [tree_ref("s", index) for index in range(1, 7)])
        for letter, _, _, path, _ in SESSION_MAX_STEPS:
            summary = results[letter][0]
            self.assertEqual((summary["paused"], summary["generated_chunks"]), (True, len(path)), letter)
            self.assertEqual(sum(node.kind == "chunk" for node in bridge.archive.path(nodes[letter])[1:]), len(path))
        sidecar = bridge.archive.store.telemetry(nodes["E"])["bridge"]["tree"]
        self.assertEqual((sidecar["generation_index"], sidecar["parent"], sidecar["depth"]), (5, nodes["B"], 3))
        self.assertEqual(bridge.controller.get("s").next_generation, 7)
        if self.exact:  # the rewinds (E from B, F from D off the head path) equal fresh native runs of their paths
            for letter, path in (("E", "ABE"), ("F", "ABCDF")):
                latents, caches, rng = self.fresh(path)
                _, got_latents, got_caches, got_rng = results[letter]
                self.assertTrue(torch.equal(got_latents[:, :, :latents.shape[2]], latents), letter)
                self.assertFalse(got_latents[:, :, latents.shape[2]:].any())  # the planned rest was never generated
                assert_caches_bitwise(self, got_caches, caches)
                self.assertTrue(torch.equal(got_rng.get_state(), rng.get_state()), letter)


class SessionMaxRawTests(SessionMaxChecks, TreeTestCase):
    """RAW archive (synchronous publication)."""


class SessionMaxRecacheTests(SessionMaxChecks, TreeTestCase):
    """RAW archive with LongLive-1.0-like recache epochs at every prompt change."""

    recache = True


class SessionMaxOffloadTests(SessionMaxChecks, TreeTestCase):
    """``raw_offload`` (write-behind: the ownership refs publish with the background write)."""

    representation = "raw_offload"


class SessionMaxLineageTests(SessionMaxChecks, TreeTestCase):
    """The lineage archive (``svd_hevc``, asynchronous publication; lossy, so no bitwise comparison)."""

    representation = "svd_hevc"
    exact = False


class SessionMaxIndexTests(TreeTestCase):
    """W1.2: the planned window of a paused request, beyond one chunk."""

    def test_paused_requests_name_their_last_planned_chunk(self):
        bridge = self.bridge()
        padded = lambda path: path + path[-1] * (SESSION_TOTAL - len(path))  # noqa: E731
        run_tree(bridge, self.request_for(handle("s", pause_after_chunks=2), padded("AB"), generation_index=2))
        self.assertEqual(sorted(bridge.controller.tree("s")), [1, 2])
        # Two new chunks (blocks 2 and 3) of six, the second named 10: indices 9 and 10 (W1.1 counted four new chunks
        # and assigned 7 and 8).
        summary, *_ = run_tree(bridge, self.request_for(handle("s", "resume", pause_after_chunks=4), padded("ABCD"),
                                                        generation_index=10))
        self.assertEqual((summary["paused"], summary["generated_chunks"]), (True, 4))
        head = self.head(bridge)
        tree = bridge.controller.tree("s")
        self.assertEqual((sorted(tree), tree[10]), ([1, 2, 9, 10], head))
        self.assertEqual(bridge.archive.load(tree[10]).parent_ref, tree[9])
        # A pause point at or below the restored depth still generates one chunk: it takes the requested index.
        run_tree(bridge, self.request_for(handle("s", "resume", pause_after_chunks=1), padded("ABCDE"),
                                          generation_index=11))
        self.assertEqual(bridge.controller.tree("s")[11], self.head(bridge))
        # No pause: every remaining block is new (unchanged); its last one is named.
        summary, *_ = run_tree(bridge, self.request_for(handle("s", "resume"), "ABCDEF", generation_index=12))
        self.assertEqual((summary["paused"], summary["generated_chunks"]), (False, 6))
        self.assertEqual(bridge.controller.tree("s")[12], self.head(bridge))
        # An index below the planned new chunks is still refused (two planned chunks cannot end at index 1).
        with self.assertRaises(ConfigurationError):
            run_tree(bridge, self.request_for(handle("u", pause_after_chunks=2), padded("AB"), generation_index=1))

    def test_planned_new_chunks(self):
        from reelcache.integrations.sglang.session import planned_new_chunks

        cases = {(6, 0, 1): 1, (6, 2, 3): 1, (6, 2, 4): 2, (6, 2, None): 4, (6, 4, 1): 1, (6, 4, 4): 1, (6, 5, 10): 1,
                 (3, 0, 3): 3, (3, 0, None): 3}
        for (blocks, resume, pause), expected in cases.items():
            self.assertEqual(planned_new_chunks(blocks, resume, pause), expected, (blocks, resume, pause))


class HeadModeTests(TreeTestCase):
    """``tree_retention="head"`` (default) is F6: rewinds follow the head path, no ownership refs."""

    tree = "head"

    def test_head_mode_rejects_non_ancestors_exactly_as_f6(self):
        bridge = self.bridge()
        self.assertFalse(bridge.controller.tree_mode)
        nodes = {}
        for depth, letter in enumerate("ABCD", start=1):
            run_tree(bridge, self.request_for(handle("s", "generate" if depth == 1 else "resume"), "ABCD"[:depth]))
            nodes[letter] = self.head(bridge)
        run_tree(bridge, self.request_for(handle("s", "rewind", checkpoint=nodes["B"]), "ABE"))
        with self.assertRaises(StateError) as caught:
            bridge.controller.rewind("s", nodes["D"])
        self.assertEqual(str(caught.exception),
                         f"checkpoint {nodes['D']!r} is not on session 's''s path")
        control_error(self, "illegal_transition", "rewind", {"session_id": "s", "checkpoint": nodes["D"]}, bridge)
        with self.assertRaises(StateError) as caught:
            bridge.archive.rewind("s", nodes["D"])
        self.assertEqual(str(caught.exception), "rewind target is not an ancestor of the session head")
        self.assertEqual(bridge.archive.store.list_refs("tree/"), {})
        # GC reclaims the abandoned branch (C, D): F6's ownership is the head path.
        bridge.control("gc")
        self.assertEqual(stored(bridge, {nodes["C"], nodes["D"]}), set())

    def test_options_off_equal_f6_node_for_node(self):
        """Explicit A/B: the same requests with the defaults, with the W1b options spelled at their defaults and in
        tree mode produce identical node ids and identical outputs; only tree mode writes ownership refs. W1.3: tree
        mode with ``tree_switch`` spelled ``drain``, and ``detach`` with nothing in flight at its switches (durable
        pauses), are node for node the default tree run; the default stays out of the configuration record."""
        runs = {}
        for name, overrides in (("default", {}), ("explicit", {"tree_retention": "head", "restore_mode": "archive"}),
                                ("tree", {"tree_retention": "full"}),
                                ("tree_drain", {"tree_retention": "full", "tree_switch": "drain"}),
                                ("tree_detach", {"tree_retention": "full", "tree_switch": "detach"})):
            config = {**overrides}
            bridge = super(TreeTestCase, self).bridge(name, **config)
            outputs, heads = [], []
            for depth in range(1, 4):
                result = run_tree(bridge, self.request_for(handle("s", "generate" if depth == 1 else "resume"),
                                                           "ABC"[:depth]))
                outputs.append(result[1])
                heads.append(self.head(bridge))
            run_tree(bridge, self.request_for(handle("s", "rewind", checkpoint=heads[0]), "AE"))
            heads.append(self.head(bridge))
            runs[name] = (heads, outputs, bridge.archive.store.list_refs("tree/"))
        self.assertEqual(ReelCacheRuntimeConfig.from_dict({"store_root": "x"}).tree_retention, "head")
        self.assertEqual(runs["default"][0], runs["explicit"][0])
        self.assertEqual(runs["default"][0], runs["tree"][0])
        for left, right in zip(runs["default"][1], runs["tree"][1]):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual((runs["default"][2], runs["explicit"][2]), ({}, {}))
        self.assertEqual(len(runs["tree"][2]), 4)
        for name in ("tree_drain", "tree_detach"):
            self.assertEqual(runs[name][0], runs["tree"][0], name)
            self.assertEqual(runs[name][2], runs["tree"][2], name)
            for left, right in zip(runs["tree"][1], runs[name][1]):
                self.assertTrue(torch.equal(left, right), name)
        default = ReelCacheRuntimeConfig.from_dict({"store_root": "x", "tree_retention": "full"})
        self.assertEqual((default.tree_switch, "tree_switch" in default.to_dict()), ("drain", False))
        detach = ReelCacheRuntimeConfig.from_dict({"store_root": "x", "tree_retention": "full", "tree_switch": "detach"})
        self.assertEqual(detach.to_dict()["tree_switch"], "detach")
        self.assertEqual(ReelCacheRuntimeConfig.from_dict(detach.to_dict()), detach)
        for bad in ({"tree_switch": "detach"}, {"tree_retention": "full", "tree_switch": "lazy"}):
            with self.assertRaises(ConfigurationError):
                ReelCacheRuntimeConfig.from_dict({"store_root": "x", **bad})


class TreeCasTests(TreeTestCase):
    def test_generation_index_conflict_fails_the_publication_atomically(self):
        bridge = self.bridge()
        nodes, _ = self.build_a_to_f(bridge, explicit_indices=True)
        before = (self.head(bridge), bridge.controller.tree("s"))
        # Index 3 already names C: a different node may not take it (compare-and-swap in the publishing transaction).
        with self.assertRaises(StateError) as caught:
            run_tree(bridge, self.request_for(handle("s", "rewind", checkpoint=nodes["E"]), "ABEH",
                                              generation_index=3))
        self.assertIn("already names", str(caught.exception))
        state = bridge.controller.get("s")
        self.assertEqual((state.head_ref, bridge.controller.tree("s")), (nodes["E"], before[1]))
        self.assertEqual(bridge.archive.head("s"), nodes["E"])  # the head and the ref moved together, or neither
        # The same index for the SAME node (an idempotent retry) is accepted.
        run_tree(bridge, self.request_for(handle("s", "rewind", checkpoint=nodes["B"]), "ABE", generation_index=5))
        self.assertEqual(bridge.controller.tree("s")[5], nodes["E"])
        with self.assertRaises(ConfigurationError):
            run_tree(bridge, self.request_for(handle("s", "rewind", checkpoint=nodes["B"]), "ABE",
                                              generation_index="5"))

    def test_concurrent_head_change_loses_the_rewind(self):
        bridge = self.bridge()
        nodes, _ = self.build_a_to_f(bridge)
        archive = bridge.archive
        real_path, raced = archive.path, []

        def racing_path(ref):
            result = real_path(ref)
            if not raced:  # another writer moves the head between the read and the compare-and-swap
                raced.append(ref)
                archive.store.set_ref("sglang/sessions/s", nodes["C"])
            return result

        archive.path = racing_path
        try:
            with self.assertRaises(StateError) as caught:
                archive.rewind("s", nodes["E"])
        finally:
            archive.path = real_path
        self.assertIn("changed concurrently", str(caught.exception))
        self.assertEqual(archive.head("s"), nodes["C"])

    def test_requested_indices_name_the_last_new_chunk(self):
        bridge = self.bridge()
        run_tree(bridge, self.request_for(handle("s"), "ABC", generation_index=3))  # setup: three chunks at once
        tree = bridge.controller.tree("s")
        self.assertEqual(sorted(tree), [1, 2, 3])
        run_tree(bridge, self.request_for(handle("s", "resume"), "ABCD"))  # no metadata: the next index
        self.assertEqual(max(bridge.controller.tree("s")), 4)
        with self.assertRaises(ConfigurationError):  # 2 new chunks cannot end at index 1
            run_tree(bridge, self.request_for(handle("u"), "AB", generation_index=1))


class LineageTreeTests(TreeTestCase):
    """The Step 6 lineage archive (svd_hevc, asynchronous publication) in tree mode: refs set with publication, owned
    rewinds at the core, the raw tier keeps the branch it leaves."""

    representation = "svd_hevc"

    def test_lineage_tree_refs_owned_rewinds_and_raw_entries_of_old_branches(self):
        bridge = self.bridge(host_raw_budget_bytes=1 << 30, gpu_raw_budget_bytes=1 << 30)
        nodes, results = self.build_a_to_f(bridge)
        expected = {index: nodes[letter] for index, letter in enumerate("ABCDEF", start=1)}
        self.assertEqual(bridge.controller.tree("s"), expected)
        self.assertEqual(results["F"][0]["restore"]["switch"]["ancestor"], False)
        tier = bridge.archive.raw_tier
        # E's rewind moved the head off C/D: their raw entries stayed (the tree holder), so F restored D from them.
        tiers = results["F"][0]["restore"]["lease"]["tiers"]["chunks"]
        self.assertEqual(sum(tiers.get(name, 0) for name in ("gpu_raw", "host_raw")), 4)
        held = {entry.node_ref: entry.holders for entry in tier.entries()}
        # The tree holder took every owned node's entries at the last head move (before F existed); F is the head's.
        self.assertTrue(all("s" + TREE_HOLDER_SUFFIX in held[nodes[letter]] for letter in "ABCDE"))
        self.assertIn("s", held[nodes["F"]])
        # Core level: an owned non-ancestor is accepted (compare-and-swap), a node of another session is not.
        runtime = bridge.archive.runtime
        bridge.checkpoint("s")
        bridge.archive.release("s")  # the pipeline holds a live lineage; the controller releases it before a rewind
        runtime.rewind_lineage("s", nodes["E"])
        self.assertEqual(bridge.archive.head("s"), nodes["E"])
        run_tree(bridge, self.request_for(handle("t"), "AH"))
        other = bridge.checkpoint("t")["head"]
        with self.assertRaises(StateError):
            runtime.rewind_lineage("s", other)
        runtime.set_lineage_ownership(None)  # the F6 rule: ancestors only
        with self.assertRaises(StateError) as caught:
            runtime.rewind_lineage("s", nodes["D"])
        self.assertEqual(str(caught.exception), "rewind target is not an ancestor of the lineage head")
        # Whole-tree delete releases the refs and both holders; GC reclaims the session.
        bridge.control("delete", {"session_id": "s"})
        self.assertEqual(bridge.archive.store.list_refs(tree_prefix("s")), {})
        self.assertFalse(any("s" + TREE_HOLDER_SUFFIX in entry.holders for entry in tier.entries()))

    def test_lineage_publication_conflict_is_an_archival_failure(self):
        bridge = self.bridge()
        run_tree(bridge, self.request_for(handle("s"), "A", generation_index=1))
        head = self.head(bridge)
        run_tree(bridge, self.request_for(handle("s", "resume"), "AB", generation_index=1))
        with self.assertRaises(StateError) as caught:
            bridge.checkpoint("s")  # the publication's compare-and-swap failed: the commit is not published
        self.assertIn("archival failed", str(caught.exception))
        self.assertEqual(bridge.archive.head("s"), head)
        self.assertEqual(bridge.controller.tree("s"), {1: head})


#: A tree-mode worker SIGKILLed inside the transaction that publishes E5: the head compare-and-swap ran, the ownership
#: ref is being written. Neither may survive (one store transaction).
TREE_CRASH_CHILD = """
import json, sys, time
import tests.test_wl_tree as t
from reelcache.integrations.sglang import CausalWanKVBridge, ReelCacheRuntimeConfig

spec = json.loads(sys.argv[1])
values = {"store_root": spec["root"], "representation": "raw", "tree_retention": "full", "fit_device": "cpu",
          "decoder": "software", "energy_key": 0.98, "energy_value": 0.95}
bridge = CausalWanKVBridge(ReelCacheRuntimeConfig.from_dict(values), t.PROFILE, device="cpu", model_path="toy/causal-wan")
nodes = {}
for depth, letter in enumerate("ABCD", start=1):
    t.run_tree(bridge, t.tree_request(t.handle("s", "generate" if depth == 1 else "resume"),
                                      [t.PROMPTS[x] for x in "ABCD"[:depth]]))
    nodes[letter] = bridge.archive.head("s")
with open(spec["result"], "w") as handle:
    json.dump(nodes, handle)
store = bridge.archive.store
real = store.set_ref

def blocking(name, object_id, **kwargs):
    if name.startswith("tree/s/00000005"):
        assert store._depth > 0  # inside the publishing transaction, after the head's compare-and-swap
        with open(spec["marker"], "w") as handle:
            handle.write("gate")
        time.sleep(3600)
    return real(name, object_id, **kwargs)

store.set_ref = blocking
t.run_tree(bridge, t.tree_request(t.handle("s", "rewind", checkpoint=nodes["B"]), ["pa", "pb", "pe"]))
"""


class TreeCrashTests(TreeTestCase):
    def test_kill_inside_the_publishing_transaction_keeps_head_and_refs_consistent(self):
        import json
        import os
        import signal
        import subprocess
        import sys
        import time

        root, marker, result = self.workspace / "crashed", self.workspace / "gate", self.workspace / "result.json"
        spec = {"root": str(root), "marker": str(marker), "result": str(result)}
        log = self.workspace / "crash_child.log"
        with open(log, "w") as output:
            child = subprocess.Popen([sys.executable, "-c", TREE_CRASH_CHILD, json.dumps(spec)],
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, stdout=output,
                                     stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 600
                while not marker.exists():
                    if child.poll() is not None:
                        self.fail(f"crash child exited early: {log.read_text()[-3000:]}")
                    if time.monotonic() > deadline:
                        self.fail("crash child never reached its gate")
                    time.sleep(0.05)
                child.send_signal(signal.SIGKILL)
                child.wait(60)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(60)
        self.assertEqual(child.returncode, -signal.SIGKILL)
        nodes = json.loads(result.read_text())
        restarted = self.bridge("crashed")  # a new worker on the killed worker's store
        state = restarted.controller.get("s")
        # The rewind to B had moved the head (its own transaction); E5's publication rolled back as a whole.
        self.assertEqual(state.head_ref, nodes["B"])
        self.assertEqual(state.tree, {index: nodes[letter] for index, letter in enumerate("ABCD", start=1)})
        self.assertNotIn(tree_ref("s", 5), restarted.archive.store.list_refs("tree/"))
        # Every owned node is restorable after the crash: D (not an ancestor of the head B) and B again.
        summary, latents, caches, _ = run_tree(restarted, self.request_for(
            handle("s", "rewind", checkpoint=nodes["D"]), "ABCDF"))
        self.assertEqual(summary["resumed_from_chunks"], 4)
        fresh_latents, fresh_caches, _ = self.fresh("ABCDF")
        self.assertTrue(torch.equal(latents, fresh_latents))
        assert_caches_bitwise(self, caches, fresh_caches)
        _, latents, caches, _ = run_tree(restarted, self.request_for(handle("s", "rewind", checkpoint=nodes["B"]),
                                                                     "ABE"))
        tree = restarted.controller.tree("s")
        self.assertEqual(sorted(tree), [1, 2, 3, 4, 5, 6])  # F took index 5, the regenerated E index 6
        self.assertEqual(tree[6], self.head(restarted))
        fresh_latents, fresh_caches, _ = self.fresh("ABE")
        self.assertTrue(torch.equal(latents, fresh_latents))
        assert_caches_bitwise(self, caches, fresh_caches)
        restarted.control("gc")
        self.assertEqual(stored(restarted, set(tree.values())), set(tree.values()))


class TreeHelpersTests(unittest.TestCase):
    def test_tree_ref_names(self):
        self.assertEqual(tree_ref("s1", 5), "tree/s1/00000005")
        self.assertEqual(tree_entries({"tree/s1/00000005": "n5", "tree/s1/00000012": "n12"}, "s1"), {5: "n5", 12: "n12"})
        for bad in (0, -1, 10 ** 8, "1", 1.0):
            with self.assertRaises(IntegrityError):
                tree_ref("s1", bad)
        with self.assertRaises(IntegrityError):
            tree_prefix("a/b")
        with self.assertRaises(IntegrityError):
            tree_entries({"tree/s1/x": "n"}, "s1")

    def test_config_fields(self):
        config = ReelCacheRuntimeConfig.from_dict({"store_root": "x"})
        self.assertEqual((config.tree_retention, config.restore_mode), ("head", "archive"))
        self.assertNotIn("tree_retention", config.content_dict())
        self.assertNotIn("restore_mode", config.content_dict())
        with self.assertRaises(ConfigurationError):
            ReelCacheRuntimeConfig.from_dict({"store_root": "x", "tree_retention": "branch"})


if __name__ == "__main__":
    unittest.main()
