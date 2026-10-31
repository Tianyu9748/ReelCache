"""CPU tests of the LongCat-Video session provider (Step 14 CR-14-3; no SGLang, no GPU).

``reelcache_sglang_models.longcat_video.session_provider`` serves SGLang's ReelCache session controls for the LongCat port:
the Step 14 control ops with the segment as the boundary unit, generation actions (generate / resume / fork /
rewind) planned from the session archive, segment boundaries committed to the ReelCache store, in-flight pause through
the boundary hook, restart safety. These tests drive it the way the port's stages do (``begin_request`` ->
``bind_request`` -> per segment: the session cache backend's ``restore`` / ``on_cache_built`` -> ``end_segment`` ->
``finish``) with a small deterministic stand-in for the model: a segment's conditioning cache depends only on its
conditioning frames, its new frames on the conditioning frames, the prompt and the generator's draws -- the LongCat
dependency structure.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

import numpy as np
import torch

from reelcache_sglang_models.longcat_video import conditioning_cache as cc
from reelcache_sglang_models.longcat_video import reelcache_backend as mb
from reelcache_sglang_models.longcat_video import session_provider as sp
from reelcache_sglang_models.longcat_video.segment_resume import generator_state_hex, set_generator_state_hex
from reelcache_sglang_models.longcat_video.session_archive import SessionArchive
from reelcache_sglang_models.longcat_video.spec import LongCatVideoArch

LAYERS, HEADS, HEAD_DIM = 2, 2, 64
CHANNELS = HEADS * HEAD_DIM
GRID = (2, 6, 8)
TOKENS = GRID[0] * GRID[1] * GRID[2]
FINGERPRINT = "test-longcat-sessions"
ARCH = LongCatVideoArch(hidden_size=CHANNELS, depth=LAYERS, num_heads=HEADS)
FRAMES_SHAPE = (13, 8, 12, 3)
GEOMETRY = {"height": 64, "width": 96, "num_frames": 93, "num_cond_frames": 13, "bucket": "test"}
REPO = Path(__file__).resolve().parents[1]


def initial_frames(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, FRAMES_SHAPE, dtype=np.uint8)


def frames_digest(frames: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()


def conditioning_cache(frames: np.ndarray, segment: int, session: str) -> cc.LongCatConditioningCache:
    """The segment's conditioning cache: a function of its conditioning frames only (compressible content)."""
    digest = frames_digest(frames)
    generator = torch.Generator().manual_seed(int(digest[:8], 16))
    keys, values = [], []
    for _layer in range(LAYERS):
        out = []
        for target in ("K", "V"):
            basis = torch.linalg.qr(torch.randn(CHANNELS, 32, generator=generator)).Q.T
            value = (torch.randn(TOKENS, 32, generator=generator) * torch.linspace(2, 0.02, 32)) @ basis
            value = value + 0.01 * torch.randn(TOKENS, CHANNELS, generator=generator)
            if target == "K":
                value = torch.nn.functional.layer_norm(value, (CHANNELS,))
            out.append(value.to(torch.bfloat16).reshape(1, TOKENS, HEADS, HEAD_DIM).contiguous())
        keys.append(out[0])
        values.append(out[1])
    cond = torch.randn(1, 16, GRID[0], 3, 4, generator=generator).to(torch.bfloat16)
    epoch = cc.ConditioningCacheEpoch(segment_index=segment, token_grid=GRID, cond_latents_sha256=digest,
                                      model_fingerprint=FINGERPRINT, session_id=session,
                                      parent_segment_index=segment - 1 if segment else None)
    return cc.LongCatConditioningCache(epoch=epoch, keys_pre_rope=keys, values=values, source=cc.SOURCE_NATIVE_BUILD,
                                       metadata={"build_seconds": 0.0}, cond_latents=cond)


def next_frames(frames: np.ndarray, prompt: str, generator: torch.Generator) -> np.ndarray:
    """The segment's last 13 decoded frames: conditioning frames, prompt and the generator's draws."""
    noise = torch.randn(16, generator=generator).numpy().tobytes()
    seed = hashlib.sha256(np.ascontiguousarray(frames).tobytes() + prompt.encode() + noise).digest()
    return np.random.default_rng(int.from_bytes(seed[:8], "little")).integers(0, 256, FRAMES_SHAPE, dtype=np.uint8)


def make_provider(store: Path, **config) -> sp.LongCatSessionProvider:
    from reelcache.archive_pipeline import ArchivePipeline
    from reelcache.archive_restore import SoftwareLineageRestorer
    from reelcache.lineage_commit import ArchiveOptions
    from reelcache.predictor_registry import RankPredictorRegistry

    cfg = sp.LongCatSessionConfig.from_dict({"store_root": str(store), **config})
    runtime = mb.open_runtime(store, mb.model_identity(FINGERPRINT, ARCH), **cfg.integrity())  # as create_provider
    pipeline = None
    if cfg.representation == "svd_hevc":  # as create_provider builds it on a worker without CUDA
        pipeline = ArchivePipeline(runtime, method=mb.default_method(), registry=RankPredictorRegistry("s14-longcat"),
                                   options=ArchiveOptions(encode_threads=1, plan_memory_bytes=cfg.plan_memory_bytes),
                                   device="cpu", encode_workers=1)
    decision = sp.resolve_schedule(cfg.schedule, representation=cfg.representation, defer_planning=cfg.defer_planning,
                                   plan_device=cfg.resolved_plan_device(), plan_memory_bytes=cfg.plan_memory_bytes,
                                   resident_bytes=0, total_bytes=0)
    backend = mb.ReelCacheConditioningCacheBackend(runtime, pipeline=pipeline, restorer=SoftwareLineageRestorer(runtime),
                                                archive=(cfg.representation,), heads=HEADS, pin_memory=False,
                                                capture_slots=cfg.c3_slots(), schedule=decision["chosen"])
    return sp.LongCatSessionProvider(cfg, backend=backend, owned=(sp._Closer(pipeline, runtime),),
                                     startup={"schedule": decision})


def join_restore(cache: cc.LongCatConditioningCache, issue_seconds: float = 0.0) -> dict:
    """What the stage does with an overlapped restore (d5 default): the first denoising step waits for every layer,
    then joins the restore (its failure re-raised) before anything leaves the segment; the segment record's fields."""
    ready = cache.ready
    if ready is None:
        return {}
    for layer in range(ready.layers):
        ready.wait(layer)
    joined = ready.join()
    cache.ready = None
    return {"cache_restore_overlapped": True, "cache_restore_issue_seconds": issue_seconds,
            "overlapped_restore": joined, "cache_restore_exposed_seconds": joined["consumer_waits"]["seconds"]}


def same_cache(a: cc.LongCatConditioningCache, b: cc.LongCatConditioningCache) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a.keys_pre_rope + a.values, b.keys_pre_rope + b.values))


def run_request(provider, handle: dict, *, prompt: str = "a cat", frames: np.ndarray | None = None, seed: int = 0,
                num_segments: int = 1, crash_after: int | None = None, **request) -> dict:
    """One session request, as the port's stages run it (an overlapped restore is joined as the stage joins it)."""
    fields = {"request_id": None, "prompt": prompt, "seed": seed, "num_segments": num_segments, "num_frames": 93,
              "session_id": None, "start_segment_index": 0, "generator_state": None,
              "conditioning_cache_backend": None, "has_client_frames": frames is not None, "injected": [], **request}
    run = provider.begin_request(fields, handle)
    generator = torch.Generator().manual_seed(seed)
    if run.plan.generator_state is not None:
        set_generator_state_hex(generator, run.plan.generator_state)
    cond = frames if run.plan.frames is None else run.plan.frames
    run.bind_request(frames=cond, generator_state=generator_state_hex(generator), geometry=GEOMETRY)
    outputs, restored, error = [], [], None
    try:
        for seg in range(run.plan.start_segment, run.plan.start_segment + run.plan.num_segments):
            epoch = cc.ConditioningCacheEpoch(segment_index=seg, token_grid=GRID, cond_latents_sha256=frames_digest(cond),
                                              model_fingerprint=FINGERPRINT, session_id=run.session.session_id,
                                              parent_segment_index=seg - 1 if seg else None)
            issued = time.perf_counter()
            cache = provider.cache_backend.restore(epoch)
            issue_seconds = time.perf_counter() - issued
            overlap = {}
            if cache is None:
                cache = conditioning_cache(cond, seg, run.session.session_id)
                provider.cache_backend.on_cache_built(cache)
                source = cc.SOURCE_NATIVE_BUILD
            else:
                source = cc.SOURCE_RESTORED
                overlap = join_restore(cache, issue_seconds)
                restored.append(cache)
            cond = next_frames(cond, prompt, generator)
            outputs.append(cond)
            if crash_after is not None and seg + 1 - run.plan.start_segment >= crash_after:
                raise RuntimeError("simulated failure")
            record = {"epoch": epoch.as_dict(), "cache_source": source, "generator_state": "start-state", **overlap}
            if run.end_segment(segment_index=seg, record=record, next_frames=cond,
                               generator_state=generator_state_hex(generator), prompt=prompt):
                break
    except BaseException as exc:  # noqa: BLE001 - the stage reports it, then re-raises
        error = exc
        raise
    finally:
        summary = run.finish(error=error)
    return {"summary": summary, "outputs": outputs, "final_state": generator_state_hex(generator),
            "restored": restored}


def boundary_epochs(provider, sid: str) -> list[cc.ConditioningCacheEpoch]:
    archive = provider.archive
    return [cc.ConditioningCacheEpoch.from_dict(boundary.segment["epoch"])
            for boundary in archive.path(archive.session(sid)["head"])[1:]]


class SessionProviderTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="longcat-sessions-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.store = self.root / "store"
        self.provider = make_provider(self.store)
        self.addCleanup(self._close, self.provider)

    @staticmethod
    def _close(provider):
        try:
            provider.close()
        except Exception:  # noqa: BLE001 - already closed by the test
            pass

    def control(self, op, **args):
        return self.provider.control(op, args)

    def assert_error(self, code, op, **args):
        from reelcache.integrations.sglang.controls import SessionControlError

        with self.assertRaises(SessionControlError) as caught:
            self.provider.control(op, args)
        self.assertEqual(caught.exception.code, code, caught.exception.to_dict())
        return caught.exception

    def test_capabilities_config_and_attachment(self):
        caps = self.control("capabilities")
        self.assertEqual((caps["boundary"], caps["model"], caps["outputs"]), ("segment", mb.MODEL_NAME, ["new"]))
        self.assertTrue(caps["prompt_change"]["supported"])
        self.assertEqual(caps["generation_actions"], list(sp.GENERATION_ACTIONS))
        self.assertEqual(set(caps["ops"]), {"capabilities", "list", "describe", "checkpoints", "request", "open",
                                            "pause", "checkpoint", "fork", "rewind", "close", "delete", "gc",
                                            "archive_stats", "drop_raw", "demote_raw", "reset_raw_peaks"})
        from reelcache.errors import ConfigurationError
        for bad in ({}, {"store_root": "x", "durability": "sometimes"}, {"store_root": "x", "energy_key": 0.99},
                    {"store_root": "x", "unknown": 1}, {"store_root": "x", "representation": "fp8"},
                    {"store_root": "x", "plan_device": "tpu"}, {"store_root": "x", "plan_memory_bytes": 0},
                    {"store_root": "x", "plan_device": "cuda", "plan_memory_bytes": None},
                    {"store_root": "x", "schedule": "now"},
                    {"store_root": "x", "defer_planning": True, "plan_device": "cpu"}):
            with self.assertRaises(ConfigurationError):
                sp.LongCatSessionConfig.from_dict(bad)
        default = sp.LongCatSessionConfig.from_dict({"store_root": "x"})
        self.assertEqual((default.plan_device, default.plan_memory_bytes, default.schedule, default.defer_planning),
                         ("auto", 8 << 30, "auto", False))
        if not torch.cuda.is_available():
            self.assertEqual(default.resolved_plan_device(), "cpu")
            with self.assertRaises(ConfigurationError):
                sp.LongCatSessionConfig.from_dict({"store_root": "x", "defer_planning": True}).resolved_plan_device()
        reason = caps["archival"].pop("schedule_reason")
        self.assertTrue(reason.startswith("auto: the archive plans on the CPU"), reason)
        self.assertTrue(caps["archival"].pop("defer_planning_reason").startswith("not needed"))
        self.assertEqual(caps["archival"], {"plan_device": "cpu", "plan_memory_bytes": 8 << 30,
                                            "schedule": "deferred", "schedule_requested": "auto",
                                            "defer_planning": False, "release_planner_memory": False,
                                            "planning_gate": None, "deferred_sources": "device",
                                            "raw_tier": None, "encoder": None})
        self.assertEqual(default.planning_gate, "group")  # g7: CUDA workers gate the planner group by group
        with self.assertRaisesRegex(ConfigurationError, "planning_gate"):
            sp.LongCatSessionConfig.from_dict({"store_root": "x", "planning_gate": "chunk"})
        # eager GPU planning overlapping denoising: refused at startup without the measured headroom
        gib = 1 << 30
        rtx, a100 = int(94.97 * gib), int(79.25 * gib)
        resident = int(37.4 * gib)
        record = sp.check_eager_overlap(schedule="eager", defer_planning=False, plan_device="cuda",
                                        plan_memory_bytes=8 * gib, resident_bytes=resident, total_bytes=rtx)
        self.assertTrue(record["applies"])
        self.assertAlmostEqual(record["headroom_gib"], 43.0)
        with self.assertRaisesRegex(ConfigurationError, "defer_planning"):
            sp.check_eager_overlap(schedule="eager", defer_planning=False, plan_device="cuda",
                                   plan_memory_bytes=8 * gib, resident_bytes=resident, total_bytes=a100)
        for schedule, defer, device in (("eager", True, "cuda"), ("deferred", False, "cuda"), ("eager", False, "cpu")):
            self.assertFalse(sp.check_eager_overlap(schedule=schedule, defer_planning=defer, plan_device=device,
                                                    plan_memory_bytes=8 * gib, resident_bytes=resident,
                                                    total_bytes=a100)["applies"])
        # the default schedule "auto" (coordinator decision): eager where the check fits, else deferred; logged reason

        def chosen(requested, total, *, device="cuda", defer=False, representation="svd_hevc"):
            decision = sp.resolve_schedule(requested, defer_planning=defer, plan_device=device,
                                           plan_memory_bytes=8 * gib, resident_bytes=resident, total_bytes=total,
                                           representation=representation)
            self.assertEqual((decision["requested"], set(decision)),
                             (requested, {"requested", "chosen", "reason", "eager_overlap", "defer_planning",
                                          "release_planner_memory", "defer_planning_reason"}))
            return decision["chosen"], decision["reason"]

        self.assertEqual(chosen("auto", rtx)[0], "eager")  # RTX PRO 6000
        self.assertIn("fits", chosen("auto", rtx)[1])
        self.assertEqual(chosen("auto", a100)[0], "deferred")  # A100-80GB
        self.assertIn("does not fit", chosen("auto", a100)[1])
        self.assertEqual(chosen("auto", a100, defer=True)[0], "eager")  # eager + defer_planning fits everywhere
        self.assertEqual(chosen("auto", rtx, device="cpu")[0], "deferred")  # CPU planner
        self.assertEqual(chosen("auto", rtx, representation="raw")[0], "deferred")  # nothing to plan
        self.assertEqual(chosen("deferred", rtx), ("deferred", "schedule 'deferred' set explicitly"))
        self.assertEqual(chosen("eager", rtx)[0], "eager")
        self.assertEqual(chosen("eager", a100, defer=True)[0], "eager")
        with self.assertRaisesRegex(ConfigurationError, "defer_planning"):
            chosen("eager", a100)  # explicit eager-overlap without the headroom: refused
        with self.assertRaises(ConfigurationError):
            sp.LongCatSessionConfig.from_dict({"store_root": "x", "schedule": "eagerly"})
        # SGLang's memory-critical seam (causal_kv_memory_critical_section): the backend's generation section
        with self.provider.memory_critical_section("vae_decode", request_id="r"):  # defer_planning off: no-op
            self.assertFalse(self.provider.backend.pipeline.stats()["deferred"])
        self.provider.backend.defer_planning = True  # the CUDA check is bypassed: section semantics are device-free
        try:
            with self.provider.memory_critical_section("vae_decode", request_id="r"):
                self.assertTrue(self.provider.backend.pipeline.stats()["deferred"])
            self.assertFalse(self.provider.backend.pipeline.stats()["deferred"])
        finally:
            self.provider.backend.defer_planning = False
        self.assertIs(cc.get_cache_backend(sp.CACHE_BACKEND_NAME), self.provider.cache_backend)
        self.assertEqual(sp.PROVIDER_SPEC.split(":"), [sp.__name__, "create_provider"])
        self.assertTrue(callable(sp.create_provider))

    def test_generate_resume_equals_one_request_and_describes_its_segments(self):
        frames = initial_frames()
        first = run_request(self.provider, {"session_id": "a", "action": "generate"}, frames=frames, num_segments=2)
        self.assertEqual([item["provider_cache"] for item in first["summary"]["segments"]], ["build", "build"])
        resumed = run_request(self.provider, {"session_id": "a", "action": "resume"}, num_segments=1)
        self.assertEqual(resumed["summary"]["start_segment"], 2)
        self.control("checkpoint", session_id="a")  # drain: a's epochs are published (restorable)
        whole = run_request(self.provider, {"session_id": "b", "action": "generate"}, frames=frames, num_segments=3)
        # resuming from the archived boundary continues exactly (frames and generator state)
        self.assertTrue(np.array_equal(resumed["outputs"][0], whole["outputs"][2]))
        self.assertEqual(resumed["final_state"], whole["final_state"])
        listed = {item["session_id"]: item for item in self.control("list")["sessions"]}
        self.assertEqual((listed["a"]["generated_segments"], listed["b"]["generated_segments"]), (3, 3))
        described = self.control("describe", session_id="a")
        self.assertEqual([item["index"] for item in described["segments"]], [0, 1, 2])
        self.assertEqual(described["segments"][2]["prompt"], "a cat")
        self.assertEqual(described["geometry"], GEOMETRY)
        described = self.control("describe", session_id="a")
        self.assertEqual({item["cache_epoch_now"] for item in described["segments"]}, {"published"})
        # session b replays a's timeline exactly: every segment's conditioning cache is a's published epoch
        self.assertEqual([item["provider_cache"] for item in whole["summary"]["segments"]],
                         ["restore", "restore", "restore"])
        checkpoints = self.control("checkpoints", session_id="a")
        self.assertEqual([item["segments"] for item in checkpoints["checkpoints"]], [0, 1, 2, 3])
        request = self.control("request", request_id=resumed["summary"]["request_id"])
        self.assertEqual(request["summary"]["generated_segments"], 3)

    def test_rewind_with_a_new_prompt_restores_the_archived_conditioning_cache(self):
        frames = initial_frames(1)
        run_request(self.provider, {"session_id": "s", "action": "generate", "durable": True}, frames=frames,
                    num_segments=3)
        before = self.control("checkpoints", session_id="s")["checkpoints"]
        self.control("checkpoint", session_id="s", name="three")
        rewound = run_request(self.provider, {"session_id": "s", "action": "rewind", "checkpoint": 1},
                              prompt="a dog", num_segments=1)
        segment = rewound["summary"]["segments"][0]
        self.assertEqual((segment["index"], segment["provider_cache"]), (1, "restore"))  # prompt-independent cache
        after = self.control("checkpoints", session_id="s")
        self.assertEqual([item["node"] for item in after["checkpoints"][:2]], [item["node"] for item in before[:2]])
        self.assertNotEqual(after["checkpoints"][2]["node"], before[2]["node"])  # new prompt, new segment 1
        self.assertFalse(after["named"]["three"]["on_head_path"])
        self.assertEqual(self.control("describe", session_id="s")["segments"][1]["prompt"], "a dog")
        # a rewind needs an ancestor; a fork may use any named boundary
        self.assert_error("illegal_transition", "rewind", session_id="s", checkpoint="three")
        forked = self.control("fork", session_id="s", target="t", checkpoint="three")
        self.assertEqual(forked["generated_segments"], 3)
        self.assert_error("unknown_checkpoint", "rewind", session_id="s", checkpoint=7)
        self.assert_error("unknown_checkpoint", "fork", session_id="s", target="u", checkpoint="nope")

    def test_fork_is_copy_on_write_and_survives_deleting_the_source(self):
        frames = initial_frames(2)
        run_request(self.provider, {"session_id": "src", "action": "generate"}, frames=frames, num_segments=2)
        src_head = self.control("describe", session_id="src")["head"]
        forked = run_request(self.provider, {"session_id": "dst", "action": "fork", "from_session": "src",
                                             "checkpoint": 1}, prompt="a fox")
        self.assertEqual(forked["summary"]["segments"][0]["provider_cache"], "restore")
        self.assertEqual(self.control("describe", session_id="src")["head"], src_head)  # source untouched
        src_path = [item["node"] for item in self.control("checkpoints", session_id="src")["checkpoints"]]
        dst_path = [item["node"] for item in self.control("checkpoints", session_id="dst")["checkpoints"]]
        self.assertEqual(dst_path[:2], src_path[:2])  # shared boundaries 0 and 1
        self.assert_error("illegal_transition", "fork", session_id="src", target="dst")
        before = self.provider.runtime.store.stats()["object_count"]
        deleted = self.control("delete", session_id="src")
        self.assertGreater(deleted["gc"]["objects_removed"], 0)
        self.assertLess(self.provider.runtime.store.stats()["object_count"], before)
        self.assertNotIn("src", [item["session_id"] for item in self.control("list")["sessions"]])
        self.assert_error("unknown_session", "describe", session_id="src")
        # the fork keeps its shared history: its boundaries and segment 0's cache epoch (forked lineage) stay
        resumed = run_request(self.provider, {"session_id": "dst", "action": "rewind", "checkpoint": 0},
                              prompt="a fox")
        self.assertEqual(resumed["summary"]["segments"][0]["provider_cache"], "restore")
        self.assertEqual([item["node"] for item in self.control("checkpoints", session_id="dst")
                          ["checkpoints"]][:1], dst_path[:1])

    # -- CR-14-6: the published-epoch index after gc / delete / restart (both representations) ----------------------
    def _gone(self, provider, representation: str, epoch: cc.ConditioningCacheEpoch, node_ref: str | None) -> None:
        from reelcache.errors import IntegrityError, StateError

        self.assertIsNone(provider.backend.archive_state(epoch, representation))
        if representation == "raw":
            with self.assertRaises(StateError):
                provider.runtime.branch_info(mb.raw_branch(epoch))
        else:
            with self.assertRaises(IntegrityError):
                provider.runtime.store.object_sizes([node_ref])

    def _index_after_gc_delete_restart(self, representation: str) -> None:
        store = self.root / f"store-{representation}"
        provider = make_provider(store, representation=representation)
        self.addCleanup(self._close, provider)
        reference = run_request(provider, {"session_id": "a", "action": "generate", "durable": True},
                                frames=initial_frames(11), num_segments=3)
        run_request(provider, {"session_id": "u", "action": "generate", "durable": True}, frames=initial_frames(12),
                    num_segments=2)
        a_epochs = provider._segment_epochs(provider._head("a"))
        u_epochs = provider._segment_epochs(provider._head("u"))
        u_nodes = {index: provider.backend.node_ref(epoch) for index, epoch in u_epochs.items()}
        live = len(a_epochs) + len(u_epochs)
        # Step 14 M3 controls: a fork op (no generation) rewound to boundary 0, then delete + gc
        provider.control("fork", {"session_id": "a", "target": "fo", "checkpoint": 1})
        provider.control("rewind", {"session_id": "fo", "checkpoint": 0})
        gc = provider.control("delete", {"session_id": "fo"})["gc"]
        self.assertEqual((gc["live_epochs"], gc["reindexed"]), (live, live))  # was "reindexed": 0 on raw
        self.assertEqual(gc["reindexed_by_representation"][representation], live)
        for epoch in list(a_epochs.values()) + list(u_epochs.values()):
            self.assertEqual(provider.backend.archive_state(epoch, representation), "published")
        # Step 14 M3 prompt change: a fork under another prompt restores segment 1's prompt-independent cache
        changed = run_request(provider, {"session_id": "pb1", "action": "fork", "from_session": "a", "checkpoint": 1},
                              prompt="a dog")
        self.assertEqual(changed["summary"]["segments"][0]["provider_cache"], "restore")
        # deleting an unrelated session: its epochs are gone, every live one stays indexed
        gc = provider.control("delete", {"session_id": "u"})["gc"]
        self.assertEqual((gc["live_epochs"], gc["reindexed"]), (len(a_epochs), len(a_epochs)))
        for index, epoch in u_epochs.items():
            self._gone(provider, representation, epoch, u_nodes[index])
        for epoch in a_epochs.values():
            self.assertEqual(provider.backend.archive_state(epoch, representation), "published")
        # restart: a new provider on the store re-indexes exactly the live epochs; requests restore, never rebuild
        provider.close()
        provider = make_provider(store, representation=representation)
        self.addCleanup(self._close, provider)
        self.assertEqual(provider.startup["reindexed"], len(a_epochs))
        self.assertEqual(provider.startup["reindexed_by_representation"][representation], len(a_epochs))
        for index, epoch in u_epochs.items():
            self._gone(provider, representation, epoch, u_nodes[index])
        rewound = run_request(provider, {"session_id": "a", "action": "rewind", "checkpoint": 1}, num_segments=2)
        self.assertEqual([item["provider_cache"] for item in rewound["summary"]["segments"]], ["restore", "restore"])
        self.assertTrue(np.array_equal(rewound["outputs"][1], reference["outputs"][2]))  # exact continuation
        again = run_request(provider, {"session_id": "pb2", "action": "fork", "from_session": "pb1", "checkpoint": 1},
                            prompt="a dog")
        self.assertEqual(again["summary"]["segments"][0]["provider_cache"], "restore")

    def test_raw_epochs_stay_indexed_after_gc_delete_and_restart(self):
        self._index_after_gc_delete_restart("raw")

    def test_svd_hevc_epochs_stay_indexed_after_gc_delete_and_restart(self):
        self._index_after_gc_delete_restart("svd_hevc")

    def _deleting_the_archiving_session(self, representation: str) -> None:
        """Another session restored the source's epochs (a fork regenerating segment 1): deleting the source keeps
        what the fork's boundaries record (svd_hevc: a holder lineage at the node only the source's lineage reached),
        drops the rest, and survives a restart; deleting the fork then releases the holder."""
        store = self.root / f"store-held-{representation}"
        provider = make_provider(store, representation=representation)
        self.addCleanup(self._close, provider)
        run_request(provider, {"session_id": "src", "action": "generate", "durable": True}, frames=initial_frames(21),
                    num_segments=3)
        src = provider._segment_epochs(provider._head("src"))
        src_nodes = {index: provider.backend.node_ref(epoch) for index, epoch in src.items()}
        forked = run_request(provider, {"session_id": "dst", "action": "fork", "from_session": "src", "checkpoint": 1,
                                        "durable": True}, prompt="a fox")
        self.assertEqual(forked["summary"]["segments"][0]["provider_cache"], "restore")  # src's segment-1 epoch
        deleted = provider.control("delete", {"session_id": "src"})
        holders = [branch for branch in provider.runtime.list_lineages() if branch.startswith(mb.HOLDER_PREFIX)]
        if representation == "svd_hevc":
            self.assertEqual(deleted["held_epochs"], [mb.holder_branch(mb.epoch_key(src[1]))])
            self.assertEqual(holders, deleted["held_epochs"])
        else:
            self.assertEqual((deleted["held_epochs"], holders), ([], []))
        self.assertEqual(deleted["gc"]["live_epochs"], 2)  # dst records src's segments 0 and 1
        self._gone(provider, representation, src[2], src_nodes[2])
        for index in (0, 1):
            self.assertEqual(provider.backend.archive_state(src[index], representation), "published")
        rewound = run_request(provider, {"session_id": "dst", "action": "rewind", "checkpoint": 1}, prompt="a fox")
        self.assertEqual(rewound["summary"]["segments"][0]["provider_cache"], "restore")
        provider.close()
        provider = make_provider(store, representation=representation)
        self.addCleanup(self._close, provider)
        self.assertEqual(provider.startup["reindexed"], 2)
        rewound = run_request(provider, {"session_id": "dst", "action": "rewind", "checkpoint": 1}, prompt="a fox")
        self.assertEqual(rewound["summary"]["segments"][0]["provider_cache"], "restore")
        self._gone(provider, representation, src[2], src_nodes[2])
        gc = provider.control("delete", {"session_id": "dst"})["gc"]
        self.assertEqual((gc["live_epochs"], gc["reindexed"]), (0, 0))
        self.assertEqual(gc["holders_released"], 1 if representation == "svd_hevc" else 0)
        self.assertEqual([branch for branch in provider.runtime.list_lineages()
                          if branch.startswith(mb.BRANCH_PREFIX)], [])
        for index in (0, 1):
            self._gone(provider, representation, src[index], src_nodes[index])

    def test_deleting_the_archiving_session_keeps_restored_epochs_raw(self):
        self._deleting_the_archiving_session("raw")

    def test_deleting_the_archiving_session_keeps_restored_epochs_svd_hevc(self):
        self._deleting_the_archiving_session("svd_hevc")

    def test_inflight_pause_stops_at_the_next_segment_boundary(self):
        seen = []

        def hook():
            seen.append(self.provider.inflight_control("list", {})["sessions"][0]["status"])
            self.assertIsNone(self.provider.inflight_control("fork", {"session_id": "p", "target": "q"}))
            with self.assertRaises(Exception) as busy:
                self.provider.control("rewind", {"session_id": "p", "checkpoint": 0})
            self.assertEqual(getattr(busy.exception, "code", None), "busy")
            reply = self.provider.inflight_control("pause", {"session_id": "p"})
            self.assertEqual((reply["status"], reply["deferred"]), ("pause_requested", True))

        self.provider.set_boundary_hook(hook)
        result = run_request(self.provider, {"session_id": "p", "action": "generate"}, frames=initial_frames(3),
                             num_segments=4)
        self.assertEqual((result["summary"]["generated_segments"], result["summary"]["paused_by"]), (1, "control"))
        self.assertTrue(result["summary"]["durable"])  # an explicit pause drains (durability "pause")
        self.assertEqual(seen, ["running"])
        self.provider.set_boundary_hook(None)
        ack = self.control("pause", session_id="p")  # the scheduler's acknowledgement after the request
        self.assertEqual((ack["deferred"], ack["published"], ack["generated_segments"]), (False, True, 1))
        # pause_after_chunks = the session's segment count at which a request stops
        stopped = run_request(self.provider, {"session_id": "p", "action": "resume", "pause_after_chunks": 3},
                              num_segments=5)
        self.assertEqual((stopped["summary"]["generated_segments"], stopped["summary"]["paused_by"]),
                         (3, "pause_after_chunks"))

    def test_requests_and_controls_are_refused_explicitly(self):
        from reelcache.integrations.sglang.controls import SessionControlError

        frames = initial_frames(4)
        run_request(self.provider, {"session_id": "e", "action": "generate"}, frames=frames)

        def refused(code, handle, **request):
            with self.assertRaises(SessionControlError) as caught:
                run_request(self.provider, handle, **request)
            self.assertEqual(caught.exception.code, code, caught.exception.to_dict())

        refused("illegal_transition", {"session_id": "e", "action": "generate"}, frames=frames)
        refused("invalid_argument", {"session_id": "e", "action": "resume"}, frames=frames)  # client conditioning
        refused("incompatible_model", {"session_id": "e", "action": "resume", "output": "full"})
        refused("incompatible_model", {"session_id": "e", "action": "resume", "decode_context_latents": 8})
        refused("invalid_argument", {"session_id": "e", "action": "resume"}, generator_state="00")
        refused("invalid_argument", {"session_id": "e", "action": "resume"}, start_segment_index=1)
        refused("invalid_argument", {"session_id": "e", "action": "resume"}, conditioning_cache_backend="x")
        refused("invalid_argument", {"session_id": "e", "action": "resume", "pause_after_chunks": 1})
        refused("illegal_transition", {"session_id": "e", "action": "resume", "checkpoint": 0})
        refused("unknown_session", {"session_id": "nobody", "action": "resume"})
        refused("invalid_argument", {"session_id": "e", "action": "generate"})  # no conditioning video
        self.assert_error("invalid_argument", "fork", session_id="e", target="../x")
        self.assert_error("invalid_argument", "checkpoint", session_id="e", name="a/b")
        self.assert_error("invalid_argument", "describe", session_id="a/b")
        self.assert_error("invalid_argument", "nonsense")
        self.assert_error("invalid_argument", "list", extra=1)
        self.assert_error("unknown_request", "request", request_id="none")
        self.assert_error("unknown_checkpoint", "delete", session_id="e", checkpoint="missing")
        # replace drops the old session
        replaced = run_request(self.provider, {"session_id": "e", "action": "generate", "replace": True},
                               frames=initial_frames(5), num_segments=2)
        self.assertEqual(replaced["summary"]["generated_segments"], 2)
        self.assertEqual(self.control("describe", session_id="e")["generated_segments"], 2)
        # a failed request keeps every committed boundary; the session is idle again
        with self.assertRaisesRegex(RuntimeError, "simulated failure"):
            run_request(self.provider, {"session_id": "e", "action": "resume"}, num_segments=3, crash_after=2)
        state = self.control("describe", session_id="e")
        self.assertEqual((state["generated_segments"], state["status"]), (3, "idle"))

    def test_eager_schedule_leaves_only_the_last_segment_for_a_durable_pause(self):
        provider = make_provider(self.root / "eager", schedule="eager")
        self.addCleanup(self._close, provider)
        result = run_request(provider, {"session_id": "same", "action": "generate", "durable": True},
                             frames=initial_frames(9), num_segments=3)
        summary = result["summary"]
        self.assertTrue(summary["durable"])
        self.assertEqual(summary["archival"], "published")
        self.assertEqual(provider.control("describe", {"session_id": "same"})["pending"], [])
        self.assertEqual({item["cache_epoch_now"] for item in provider.control("describe", {"session_id": "same"})[
            "segments"]}, {"published"})
        self.assertEqual(provider.control("capabilities", {})["archival"]["schedule"], "eager")
        # the eager provider archives the same nodes as the deferred one (same session and frames, other store)
        deferred = run_request(self.provider, {"session_id": "same", "action": "generate", "durable": True},
                               frames=initial_frames(9), num_segments=3)
        self.assertEqual(deferred["summary"]["archival"], "published")
        eager_nodes = [provider.backend.node_ref(cc.ConditioningCacheEpoch.from_dict(item["epoch"]))
                       for item in self._segments(provider, "same")]
        deferred_nodes = [self.provider.backend.node_ref(cc.ConditioningCacheEpoch.from_dict(item["epoch"]))
                          for item in self._segments(self.provider, "same")]
        self.assertEqual(eager_nodes, deferred_nodes)
        self.assertNotIn(None, eager_nodes)

    @staticmethod
    def _segments(provider, sid):
        archive = provider.archive
        return [boundary.segment for boundary in archive.path(archive.session(sid)["head"])[1:]]

    def test_named_checkpoints_close_reopen_and_gc(self):
        run_request(self.provider, {"session_id": "n", "action": "generate", "durable": True},
                    frames=initial_frames(6), num_segments=2)
        pinned = self.control("checkpoint", session_id="n", name="two")
        self.assertEqual(pinned["name"], "two")
        self.control("checkpoint", session_id="n", name="two")  # same boundary: idempotent
        run_request(self.provider, {"session_id": "n", "action": "rewind", "checkpoint": 0}, prompt="other",
                    num_segments=1)
        with self.assertRaises(Exception) as exists:
            self.provider.control("checkpoint", {"session_id": "n", "name": "two"})
        self.assertEqual(getattr(exists.exception, "code", None), "illegal_transition")
        self.control("checkpoint", session_id="n", name="two", replace=True)
        closed = self.control("close", session_id="n")
        self.assertEqual(closed["status"], "closed")
        self.assertEqual(self.control("list")["sessions"][0]["status"], "closed")
        opened = self.control("open", session_id="n")
        self.assertFalse(opened["was_open"])
        self.assertEqual(opened["prepared"]["epochs"], [0])  # the prepared state of the restore requirement
        self.assertEqual(self.control("delete", session_id="n", checkpoint="two")["deleted_checkpoint"], "two")
        collected = self.control("gc")["gc"]
        self.assertIn("objects_removed", collected)
        store = self.provider.runtime.store
        refs = store.list_refs("longcat/")
        self.assertTrue(refs)
        for ref in refs.values():
            store.verify(ref)  # every live session record, boundary and frames blob is intact after gc

    def test_restart_lists_and_resumes_from_the_last_committed_boundary(self):
        frames = initial_frames(7)
        whole = run_request(self.provider, {"session_id": "w", "action": "generate"}, frames=frames, num_segments=3)
        self.provider.close()
        script = textwrap.dedent(f"""
            import os, sys
            sys.path.insert(0, {str(REPO)!r})
            import tests.test_longcat_video_session_provider as T
            from pathlib import Path
            provider = T.make_provider(Path({str(self.store)!r}), durability="never")
            def kill():
                if provider._run.session.head.segments >= 2:
                    os.kill(os.getpid(), 9)
            provider.set_boundary_hook(kill)
            T.run_request(provider, {{"session_id": "k", "action": "generate"}}, frames=T.initial_frames(7),
                          num_segments=3)
        """)
        child = subprocess.run([sys.executable, "-c", script], env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                               capture_output=True, text=True, timeout=600)
        self.assertEqual(child.returncode, -9, child.stderr[-2000:])
        provider = make_provider(self.store)
        self.addCleanup(self._close, provider)
        self.provider = provider
        listed = {item["session_id"]: item for item in self.control("list")["sessions"]}
        self.assertEqual((listed["k"]["generated_segments"], listed["k"]["status"]), (2, "closed"))
        self.assertEqual(listed["w"]["generated_segments"], 3)
        resumed = run_request(provider, {"session_id": "k", "action": "resume"})
        self.assertTrue(np.array_equal(resumed["outputs"][0], whole["outputs"][2]))  # exact continuation
        self.assertEqual(resumed["final_state"], whole["final_state"])
        self.assertEqual(provider.startup["sessions"], 2)
        archive = SessionArchive(provider.runtime.store)
        self.assertEqual(len(archive.path(archive.session("k")["head"])), 4)
        # nothing half-published: every lineage head and every session record verifies after the kill
        store = provider.runtime.store
        for head in list(provider.runtime.list_lineages().values()) + list(store.list_refs("longcat/").values()):
            store.verify(head)


class OverlapRestoreTests(unittest.TestCase):
    """d5 (freeze F5): an svd_hevc session restore overlaps the first denoising step by default (``overlap_restore``,
    the SGLang bridge's name); a session handle overrides it per request; outside a request a restore is plain; the
    result is bitwise the plain restore and its tier is recorded when it completes."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="longcat-overlap-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def provider(self, name: str = "store", **config):
        provider = make_provider(self.root / name, **config)
        self.addCleanup(SessionProviderTests._close, provider)
        return provider

    def generate(self, provider, sid: str = "a", seed: int = 7):
        run_request(provider, {"session_id": sid, "action": "generate", "durable": True}, frames=initial_frames(seed),
                    num_segments=2)

    def test_default_overlaps_session_restores_bitwise_with_completion_tiers(self):
        from reelcache.errors import ConfigurationError
        from reelcache.integrations.sglang.controls import SessionControlError

        self.assertTrue(sp.LongCatSessionConfig.from_dict({"store_root": "x"}).overlap_restore)
        for bad in (1, "yes", None):
            with self.assertRaises(ConfigurationError):
                sp.LongCatSessionConfig.from_dict({"store_root": "x", "overlap_restore": bad})
        provider = self.provider()
        record = provider.control("capabilities", {})["overlap_restore"]
        self.assertEqual((record["configured"], record["effective"]), (True, True))
        self.assertEqual(provider.startup["overlap_restore"], record)
        self.generate(provider)
        overlapped = run_request(provider, {"session_id": "b", "action": "fork", "from_session": "a", "checkpoint": 1})
        plain = run_request(provider, {"session_id": "c", "action": "fork", "from_session": "a", "checkpoint": 1,
                                       "overlap_restore": False})
        self.assertEqual((overlapped["summary"]["overlap_restore"], plain["summary"]["overlap_restore"]), (True, False))
        seg_o, seg_p = overlapped["summary"]["segments"][0], plain["summary"]["segments"][0]
        self.assertEqual((seg_o["provider_cache"], seg_p["provider_cache"]), ("restore", "restore"))
        self.assertIs(seg_o["cache_restore_overlapped"], True)
        self.assertIsNone(seg_p.get("cache_restore_overlapped"))
        # the software restorer keeps no prepared packs: every restore reads the store (cold), labelled at completion
        self.assertEqual((seg_o["tier"], seg_p["tier"]), ("compressed_cold", "compressed_cold"))
        self.assertNotIn("tier_pending", seg_o)
        self.assertGreater(seg_o["tier_seconds"], 0.0)
        self.assertEqual(seg_o["tier_bytes"], seg_p["tier_bytes"])
        self.assertGreaterEqual(seg_o["cache_restore_exposed_seconds"], 0.0)
        self.assertTrue(same_cache(overlapped["restored"][0], plain["restored"][0]))  # bitwise the plain restore
        self.assertTrue(np.array_equal(overlapped["outputs"][0], plain["outputs"][0]))
        described = provider.control("describe", {"session_id": "b"})["segments"]
        self.assertIs(described[1]["cache_restore_overlapped"], True)  # kept in the boundary record
        tiers = provider.backend._restore_tiers["compressed_cold"]
        self.assertGreaterEqual(tiers["count"], 2)
        self.assertGreater(tiers["seconds"], 0.0)  # the overlapped restore's real duration, not its issue time
        with self.assertRaises(SessionControlError) as caught:
            run_request(provider, {"session_id": "d", "action": "fork", "from_session": "a", "checkpoint": 1,
                                   "overlap_restore": "yes"})
        self.assertEqual(caught.exception.code, "invalid_argument")
        self.assertNotIn("d", [item["session_id"] for item in provider.control("list", {})["sessions"]])
        # outside a request (tools, timing harnesses: nothing waits per layer or joins) a restore is plain
        epoch = boundary_epochs(provider, "a")[1]
        self.assertIsNone(provider._run)
        outside = provider.cache_backend.restore(epoch)
        self.assertIsNone(outside.ready)
        self.assertTrue(same_cache(outside, plain["restored"][0]))

    def test_config_off_handle_on_and_the_raw_representation(self):
        off = self.provider("off", overlap_restore=False)
        record = off.control("capabilities", {})["overlap_restore"]
        self.assertEqual((record["configured"], record["effective"]), (False, False))
        self.generate(off)
        plain = run_request(off, {"session_id": "b", "action": "fork", "from_session": "a", "checkpoint": 1})
        self.assertIsNone(plain["summary"]["segments"][0].get("cache_restore_overlapped"))
        forced = run_request(off, {"session_id": "c", "action": "fork", "from_session": "a", "checkpoint": 1,
                                   "overlap_restore": True})
        self.assertIs(forced["summary"]["segments"][0]["cache_restore_overlapped"], True)
        self.assertTrue(same_cache(plain["restored"][0], forced["restored"][0]))
        raw = self.provider("raw", representation="raw")
        record = raw.control("capabilities", {})["overlap_restore"]
        self.assertEqual((record["configured"], record["effective"]), (True, False))
        self.generate(raw)
        forked = run_request(raw, {"session_id": "b", "action": "fork", "from_session": "a", "checkpoint": 1,
                                   "overlap_restore": True})  # RAW restores are plain even when asked
        self.assertFalse(forked["summary"]["overlap_restore"])
        segment = forked["summary"]["segments"][0]
        self.assertEqual(segment["provider_cache"], "restore")
        self.assertIsNone(segment.get("cache_restore_overlapped"))

    def test_an_explicit_backend_setting_applies_as_in_d4(self):
        """An explicit ``backend.set_overlap`` (tools, harnesses, 5c's D-E tests: the d4 mechanism) applies to session
        requests without a handle override and to restores outside a request; the handle still overrides it;
        ``set_overlap(None)`` restores the config's default."""
        provider = self.provider()
        self.generate(provider)
        epoch = boundary_epochs(provider, "a")[1]
        provider.backend.set_overlap(False)
        plain = run_request(provider, {"session_id": "b", "action": "fork", "from_session": "a", "checkpoint": 1})
        self.assertIsNone(plain["summary"]["segments"][0].get("cache_restore_overlapped"))
        self.assertIs(plain["summary"]["overlap_restore"], False)
        forced = run_request(provider, {"session_id": "c", "action": "fork", "from_session": "a", "checkpoint": 1,
                                        "overlap_restore": True})
        self.assertIs(forced["summary"]["segments"][0]["cache_restore_overlapped"], True)
        provider.backend.set_overlap(True)
        outside = provider.cache_backend.restore(epoch)  # outside a request: the explicit setting
        self.assertIsNotNone(outside.ready)
        join_restore(outside)
        self.assertTrue(same_cache(outside, plain["restored"][0]))
        provider.backend.set_overlap(None)
        default = run_request(provider, {"session_id": "d", "action": "fork", "from_session": "a", "checkpoint": 1})
        self.assertIs(default["summary"]["segments"][0]["cache_restore_overlapped"], True)  # the config again
        self.assertIs(default["summary"]["overlap_restore"], True)
        self.assertIsNone(provider.cache_backend.restore(epoch).ready)  # unset: plain outside a request
        self.assertTrue(same_cache(default["restored"][0], forced["restored"][0]))

    def test_completion_tier_resolution(self):
        resolved = sp._completion_tier({"provider_cache": "restore", "tier_pending": True},
                                       {"tier": "compressed_cold", "bytes": 10, "seconds": 0.5}, None)
        self.assertEqual(resolved, {"provider_cache": "restore", "tier": "compressed_cold", "tier_bytes": 10,
                                    "tier_seconds": 0.5})
        fallback = sp._completion_tier({"tier_pending": True}, None,
                                       {"tier": "compressed_prepared", "bytes": 3, "seconds": 0.1})
        self.assertEqual(fallback["tier"], "compressed_prepared")
        pending = sp._completion_tier({"tier_pending": True}, {"tier": None}, {"tier": None, "pending": True})
        self.assertEqual(pending, {"tier_pending": True})
        plain = {"provider_cache": "restore", "tier": "host_raw", "tier_bytes": 1, "tier_seconds": 0.2}
        self.assertEqual(sp._completion_tier(plain, {"tier": "compressed_cold"}, None), plain)


class StageWiringTests(unittest.TestCase):
    def test_pipeline_declares_the_provider_and_http_fields(self):
        """With SGLang: the real classes; without it (CPU env): the same declarations read from the sources."""
        try:
            from reelcache_sglang_models.longcat_video.sglang_configs import LongCatVideoSamplingParams
            from reelcache_sglang_models.longcat_video.sglang_pipeline import LongCatVideoContinuationPipeline
        except ImportError:
            import ast

            package = Path(sp.__file__).resolve().parent
            pipeline = ast.parse((package / "sglang_pipeline.py").read_text())
            cls = next(node for node in pipeline.body if isinstance(node, ast.ClassDef)
                       and node.name == "LongCatVideoContinuationPipeline")
            assigned = {target.id: node.value for node in cls.body if isinstance(node, ast.Assign)
                        for target in node.targets if isinstance(target, ast.Name)}
            self.assertEqual(ast.unparse(assigned["reelcache_session_provider"]), "PROVIDER_SPEC")
            configs = ast.parse((package / "sglang_configs.py").read_text())
            params = next(node for node in configs.body if isinstance(node, ast.ClassDef)
                          and node.name == "LongCatVideoSamplingParams")
            method = next(node for node in params.body if isinstance(node, ast.FunctionDef)
                          and node.name == "video_request_extra_fields")
            self.assertIn("'num_segments'", ast.unparse(method))
            self.assertIn("session_provider", (package / "sglang_stages.py").read_text())
            return
        self.assertEqual(LongCatVideoContinuationPipeline.reelcache_session_provider, sp.PROVIDER_SPEC)
        self.assertIn("num_segments", LongCatVideoSamplingParams.video_request_extra_fields())

    def test_session_requests_without_a_video_pass_sglang_f2v_validation(self):
        """SGLang requires a video for F2V requests (``_validate_with_pipeline_config``); a session resume/fork/rewind
        carries none (its boundary comes from the archive; the provider refuses a request video), so the LongCat
        sampling params accept it there and only there (Step 14 M3 job 4403138 hit the plain check over HTTP)."""
        try:
            from reelcache_sglang_models.longcat_video.sglang_configs import (
                LongCatVideoContinuationConfig,
                LongCatVideoSamplingParams,
            )
        except ImportError:
            import ast

            package = Path(sp.__file__).resolve().parent
            configs = ast.parse((package / "sglang_configs.py").read_text())
            params = next(node for node in configs.body if isinstance(node, ast.ClassDef)
                          and node.name == "LongCatVideoSamplingParams")
            method = next(node for node in params.body if isinstance(node, ast.FunctionDef)
                          and node.name == "_validate_with_pipeline_config")
            source = ast.unparse(method)
            for needle in ("causal_kv_session", "'generate'", "self.video_path", "self.image_path",
                           "super()._validate_with_pipeline_config(pipeline_config)"):
                self.assertIn(needle, source)
            return
        config = LongCatVideoContinuationConfig()

        def params(handle=None, *, video=None, image=None):
            value = LongCatVideoSamplingParams(prompt="p", video_path=video)
            value.causal_kv_session = handle  # as SGLang's validate_causal_kv_session leaves it (a dict)
            value.image_path = image
            return value

        for action in ("resume", "fork", "rewind"):
            params({"session_id": "s", "action": action})._validate_with_pipeline_config(config)
        params({"session_id": "s", "action": "generate"}, video="v.mp4")._validate_with_pipeline_config(config)
        params(video="v.mp4")._validate_with_pipeline_config(config)
        for refused in (params(), params({"session_id": "s", "action": "generate"}), params({"session_id": "s"})):
            with self.assertRaisesRegex(ValueError, "video_path"):
                refused._validate_with_pipeline_config(config)
        with self.assertRaisesRegex(ValueError, "input_reference"):
            params({"session_id": "s", "action": "resume"}, image="x.png")._validate_with_pipeline_config(config)


if __name__ == "__main__":
    unittest.main()
