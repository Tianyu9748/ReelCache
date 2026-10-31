"""SGLang bridge (``reelcache.integrations.sglang``) on CPU, without SGLang.

A toy causal model drives the bridge exactly like SGLang's causal stage does
(restore -> per-chunk clean capture before RoPE -> commit at the chunk
boundary -> optional clean recache forward -> pause/finish). Its K/V and
latents depend on the cached history, so a wrong restore changes the
continuation. The fake cache implements the full-history subset of
``CausalSelfAttentionKVCache``'s restore API; SGLang's real
sink/window/pinned replay is tested in SGLang's own unit tests and the GPU
jobs. ``raw`` runs the exact synchronous reference archive; ``svd_hevc`` runs
the Step 6 runtime (lineages, ``ArchivePipeline`` on the CPU planner,
``SoftwareLineageRestorer``).
"""

import dataclasses
import gc
import itertools
import json
import os
import signal
import subprocess
import sys
import threading
import time
import unittest
import zlib
from unittest import mock
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from reelcache import restore_kernels as rk
from reelcache.errors import (CapacityError, ConfigurationError, IntegrityError, PendingError, StateError,
                           UnsupportedError)
from reelcache.integrations.sglang import CausalWanKVBridge, ModelProfile, ReelCacheRuntimeConfig, profile_for
from reelcache.integrations.sglang.restore import (GC_PAUSES, ROPE_WARM_FRAMES, GcPauses, RopeTableCache,
                                               decoder_pool_record, decoder_pool_snapshot, gc_pause_record,
                                               install_gc_pauses)
from reelcache.integrations.sglang.controls import ERROR_STATUS, SessionControlError, prompt_change_capability
from reelcache.integrations.sglang.lineage_archive import LineageSessionArchive
from reelcache.integrations.sglang.profiles import LONGLIVE2, register_profile
from tests.core_support import CoreWorkspaceTest

HEADS, DIM, FRAMES, GRID = 2, 32, 2, (4, 6)
TPF = GRID[0] * GRID[1]
LAYERS, CHANNELS = 2, HEADS * DIM
PROFILE = ModelProfile(
    name="fixture", backbone="toy-causal-wan", transformers=("ToyCausalWan",), model_paths=("toy/causal-wan",),
    checkpoint={"repo": "toy"}, num_layers=LAYERS, num_heads=HEADS, head_dim=DIM, chunk_frames=FRAMES,
    visibility="full_history", sink_frames=0, window_frames=8 * FRAMES, energy_profile="custom",
    cfg_streams=("cond", "uncond"),
)


@dataclass(frozen=True)
class Chunk:
    index: int
    start_frame: int
    num_frames: int
    token_start: int
    num_tokens: int
    rope_start_frame: float
    scene_cut: bool
    shot_index: int


@dataclass(frozen=True)
class Capture:
    chunk: Chunk
    stream: str
    kind: str = "clean_commit"


@dataclass(frozen=True)
class Recache:
    window: Chunk
    written: tuple
    epoch_id: str
    cause: str = "kv_recache"
    details: dict = None


@dataclass(frozen=True)
class Span:
    local_start: int
    local_end: int
    global_start: int

    @property
    def length(self):
        return self.local_end - self.local_start


_MUTATIONS = itertools.count(1)


class FullHistoryCache:
    """Full-history subset of SGLang's CausalSelfAttentionKVCache (no sink/window/pinned), including its
    resident-KV API: ``mutation_generation`` renewed by every K/V write, ``index_state`` /
    ``holds_index_state`` / ``restore_index_state``."""

    def __init__(self, capacity):
        self.k = torch.zeros(1, capacity, HEADS, DIM, dtype=torch.bfloat16)
        self.v = torch.zeros_like(self.k)
        self.cache_size = self.attention_window_size = capacity
        self.sink_tokens = self.global_sink_tokens = 0
        self.end, self.pinned_start, self.pinned_len = 0, -1, 0
        self.mutation_generation = next(_MUTATIONS)

    def _read_indices(self):
        return self.end, self.end

    def reset_indices(self):
        self.end = 0

    def index_state(self):
        return SimpleNamespace(mutation_generation=self.mutation_generation, end=self.end,
                               cache_size=self.cache_size)

    def holds_index_state(self, state):
        return (state.mutation_generation, state.cache_size) == (self.mutation_generation, self.cache_size)

    def restore_index_state(self, state):
        if not self.holds_index_state(state):
            raise ValueError("cache contents changed")
        self.end = state.end

    def update(self, key, value, start):
        tokens = key.shape[1]
        if start != self.end:
            raise AssertionError("toy cache only appends")
        self.mutation_generation = next(_MUTATIONS)
        self.k[:, start:start + tokens] = key
        self.v[:, start:start + tokens] = value
        self.end = start + tokens

    def overwrite(self, start, key, value):
        tokens = key.shape[1]
        if start + tokens > self.end:
            raise AssertionError("recache rewrites existing slots only")
        self.mutation_generation = next(_MUTATIONS)
        self.k[:, start:start + tokens] = key
        self.v[:, start:start + tokens] = value

    def replay_layout(self, appends, *, next_chunk=None):
        end = 0
        for start, tokens, _ in appends:
            assert start == end
            end += tokens
        live = (Span(0, end, 0),)
        return SimpleNamespace(cache_size=self.cache_size, sink_tokens=0, global_sink_tokens=0,
                               attention_window_size=self.cache_size, global_end_index=end, local_end_index=end,
                               pinned_start=-1, pinned_len=0, live=live, attended=live if next_chunk else None)

    def install_restored_history(self, layout, write):
        if self.end != 0:
            raise RuntimeError("install needs a reset cache")
        self.mutation_generation = next(_MUTATIONS)
        self.k[:, :layout.local_end_index].zero_()
        self.v[:, :layout.local_end_index].zero_()
        for span in layout.attended or layout.live:
            write(span, self.k[:, span.local_start:span.local_end], self.v[:, span.local_start:span.local_end])
        self.end = layout.local_end_index


def rope_cos_sin(start_frame, frames):
    positions = torch.arange(frames * TPF, dtype=torch.float32) + float(start_frame) * TPF
    angles = positions[:, None] * torch.linspace(0.01, 0.5, DIM // 2)[None, :]
    return angles.cos(), angles.sin()


def rotate_key(key, cos, sin):
    pairs = key.float().reshape(*key.shape[:-1], DIM // 2, 2)
    cos, sin = cos[:, None, :], sin[:, None, :]
    even = pairs[..., 0] * cos - pairs[..., 1] * sin
    odd = pairs[..., 0] * sin + pairs[..., 1] * cos
    return torch.stack((even, odd), dim=-1).reshape(key.shape).to(key.dtype)


def toy_kv(prompt, chunk, layer, stream, cache, key_dtype=torch.bfloat16):
    """Low-rank pre-RoPE K / V that depend on the prompt, chunk, layer, stream and the cached history."""
    seed = zlib.crc32(f"{prompt}:{chunk.index}:{layer}:{stream}".encode())
    generator = torch.Generator().manual_seed(seed)
    tokens = chunk.num_tokens
    rank = min(tokens, CHANNELS)  # a multi-chunk recache window has more tokens than channels
    left = torch.linalg.qr(torch.randn(tokens, rank, generator=generator, dtype=torch.float64))[0]
    right = torch.linalg.qr(torch.randn(CHANNELS, CHANNELS, generator=generator, dtype=torch.float64))[0][:rank]
    spectrum = torch.logspace(0, -2.5, rank, dtype=torch.float64)
    base = (left * spectrum) @ right
    history = cache.k[:, :cache.end].float().mean() if cache.end else torch.tensor(0.0)
    key = (base * 4 + history.double()).float().reshape(1, tokens, HEADS, DIM).to(key_dtype)
    value = (base.flip(0) * 3 + 0.5 * history.double()).float().reshape(1, tokens, HEADS, DIM).to(torch.bfloat16)
    return key, value


def pool_stats(*, jobs, started, cold_jobs, created, waits, creation, warm_up, maxima, queued=(0, 0),
               warm_ups=(0, 0), running=0, started_warm_ups=0, moot=0, warming=(False, False), warm_queued=(0, 0)):
    """``DecoderPool.stats()`` of a two-worker pool A6r (pool level = sums over the workers, maximum of the maxima)."""
    workers = [{"index": index, "jobs": jobs[index], "jobs_started": started[index], "cold_jobs": cold_jobs[index],
                "sessions_created": created[index], "queue_wait_seconds": waits[index],
                "creation_wait_seconds": creation[index], "warm_up_wait_seconds": warm_up[index],
                "queue_wait_max_seconds": maxima[index], "queued_jobs": queued[index],
                "warm_ups_in_flight": warm_ups[index], "warming": warming[index], "warm_queued": warm_queued[index],
                "auto_warms": 0, "busy_seconds": 1.0} for index in range(2)]
    pool = {key: sum(worker[key] for worker in workers)
            for key in ("jobs", "jobs_started", "cold_jobs", "sessions_created", "queue_wait_seconds",
                        "creation_wait_seconds", "warm_up_wait_seconds", "queued_jobs", "warm_ups_in_flight")}
    return {**pool, "queue_wait_max_seconds": max(maxima), "warm_up_running": running, "warm_ups": started_warm_ups,
            "warm_ups_moot": moot, "pictures": 0, "failures": 0, "background_warm_ups": "none", "auto_warms": 0,
            "workers": workers}


class RecordingPool:
    """A decoder pool that only answers ``stats()`` (scripted), so any other use by a restore raises."""

    def __init__(self, *snapshots):
        self.snapshots, self.calls = list(snapshots), 0

    def stats(self):
        self.calls += 1
        return self.snapshots[min(self.calls, len(self.snapshots)) - 1]


class Stage:
    """SGLang's causal stage keeps its caches across requests and only resets their cursors per request."""

    def __init__(self):
        self.caches = None

    def caches_for(self, streams):
        if self.caches is None or tuple(self.caches) != tuple(streams):
            self.caches = {stream: [FullHistoryCache(8 * FRAMES * TPF) for _ in range(LAYERS)] for stream in streams}
        else:
            for caches in self.caches.values():
                for cache in caches:
                    cache.reset_indices()
        return self.caches


_REQUESTS = itertools.count()


def request(session, *, prompt="A", blocks=4, streams=("cond",), sink_tokens=0):
    return SimpleNamespace(
        request_id=f"r-{prompt}-{next(_REQUESTS)}", session=session, conditioning={"prompt": prompt, "seed": 7},
        block_sizes=(FRAMES,) * blocks, tokens_per_frame=TPF, grid=GRID, pos_start_base=0, streams=streams,
        num_layers=LAYERS, num_heads=HEADS, head_dim=DIM, dtype="bfloat16",
        model={"transformer": "ToyCausalWan", "model_path": "toy/causal-wan", "cache_tokens": 8 * FRAMES * TPF,
               "sink_tokens": sink_tokens, "global_sink_tokens": 0, "attention_window_tokens": 8 * FRAMES * TPF},
    )


def handle(session_id, action="generate", **extra):
    return {"session_id": session_id, "action": action, "replace": False, **extra}


def recache(run, req, caches, at, *, written=None):
    """Toy prompt switch before block ``at``: one clean forward over chunks at-2..at-1 under the new prompt,
    rewritten in place (no sink in the toy cache); ``written`` defaults to both chunks."""
    switch = switch_of(req)
    first = at - 2
    window = Chunk(-1, first * FRAMES, 2 * FRAMES, first * FRAMES * TPF, 2 * FRAMES * TPF, float(first * FRAMES),
                   False, 0)
    if written is None:
        written = ((window.token_start, window.num_tokens),)
    cos, sin = rope_cos_sin(window.rope_start_frame, window.num_frames)
    for stream in req.streams:
        capture = Capture(window, run.stream_of(caches[stream]), "kv_recache")
        for layer in range(LAYERS):
            key, value = toy_kv(switch["prompt"], window, layer, stream, caches[stream][layer],
                                key_dtype=getattr(req, "key_dtype", torch.bfloat16))
            run.capture_layer(capture, layer, key, value, (cos, sin))
            caches[stream][layer].overwrite(window.token_start, rotate_key(key, cos, sin), value)
        run.end_clean_forward(capture)
    return run.end_recache(Recache(window, tuple(written), f"switch-{at}", details={"prompt": switch["prompt"]}))


def switch_of(req):
    """A prompt switch: in the conditioning (one identity) or a timeline field with per-block conditioning."""
    return getattr(req, "switch", None) or req.conditioning.get("switch")


def drive(bridge, req, *, fail_at=None, stage=None):
    """One request through the bridge like SGLang's causal stage; returns (summary, latents, caches, generator).

    Without ``stage`` every request gets new caches (the session KV is never resident); with a ``Stage`` the
    caches persist across requests like SGLang's. A request without a session handle generates without the
    bridge (and still writes the stage caches)."""
    if stage is None:
        caches = {stream: [FullHistoryCache(8 * FRAMES * TPF) for _ in range(LAYERS)] for stream in req.streams}
    else:
        caches = stage.caches_for(req.streams)
    latents = torch.zeros(1, 4, sum(req.block_sizes), *GRID)
    generator = torch.Generator().manual_seed(req.conditioning["seed"])
    torch.randn(3, generator=generator)  # latent preparation draws
    run = bridge.begin_request(req)
    if run is None and stage is None:
        return None, latents, caches, generator
    target = SimpleNamespace(caches=caches, latents=latents, generator=generator, rope_cos_sin=rope_cos_sin,
                             rotate_key=rotate_key, device=torch.device("cpu"))
    try:
        resume = 0 if run is None else run.restore(target)
        start = 0
        for index, frames in enumerate(req.block_sizes):
            if index < resume:
                start += frames
                continue
            switch = switch_of(req)
            if run is not None and switch is not None and index == switch["at"] and index >= resume:
                # The restored head says whether the switch's recache epoch was restored (rerun it otherwise).
                restored_at_switch = index == resume and run.restore_report.get("head_kind") == "recache"
                if not restored_at_switch:
                    recache(run, req, caches, index, written=switch.get("written"))
            chunk = Chunk(index, start, frames, start * TPF, frames * TPF, float(start), False, 0)
            history = caches["cond"][0].k[:, :caches["cond"][0].end].float().mean() if caches["cond"][0].end else 0.0
            noise = torch.randn(1, 4, frames, *GRID, generator=generator)
            latents[:, :, start:start + frames] = noise + float(history)
            cos, sin = rope_cos_sin(chunk.rope_start_frame, frames)
            for stream in req.streams:
                capture = None if run is None else Capture(chunk, run.stream_of(caches[stream]))
                if run is not None and hasattr(run, "begin_clean_forward"):
                    run.begin_clean_forward(capture)  # optional seam call (SGLang's capture scope makes it)
                for layer in range(LAYERS):
                    prompt = req.conditioning["prompt"] if switch is None or index < switch["at"] else switch["prompt"]
                    key, value = toy_kv(prompt, chunk, layer, stream, caches[stream][layer],
                                        key_dtype=getattr(req, "key_dtype", torch.bfloat16))
                    if run is not None:
                        run.capture_layer(capture, layer, key, value, (cos, sin))
                    caches[stream][layer].update(rotate_key(key, cos, sin), value, chunk.token_start)
                if run is not None:
                    run.end_clean_forward(capture)
            if fail_at == index:
                raise RuntimeError("forward failed")
            start += frames
            if run is not None and run.end_chunk(chunk, latents[:, :, start - frames:start].clone()):
                break
        return (None if run is None else run.finish(error=None)), latents, caches, generator
    except BaseException as exc:
        if run is not None:
            run.finish(error=exc)
        raise



def rk_key(layer, target):
    from reelcache.types import MatrixKey
    return MatrixKey(layer, target)


SNAPSHOT = ("/data/hf/hub/models--Rabinovich--LongLive-2.0-5B-Diffusers/snapshots/"
            "37664a64cfc38cdc5fabe88576961ef10646e4f1")


class BridgeTestCase(CoreWorkspaceTest):
    representation = "raw"

    def bridge(self, store="archive", profile=PROFILE, **overrides):
        values = {"store_root": str(self.workspace / store), "representation": self.representation,
                  "fit_device": "cpu", "decoder": "software", "energy_key": 0.98, "energy_value": 0.95,
                  "x265_threads": 2, "encode_workers": 2}
        values.update(overrides)
        bridge = CausalWanKVBridge(ReelCacheRuntimeConfig.from_dict(values), profile, device="cpu",
                                   model_path="toy/causal-wan")
        self.addCleanup(bridge._close_at_exit)
        return bridge

    def assertCachesEqual(self, left, right):
        for stream in left:
            for a, b in zip(left[stream], right[stream]):
                self.assertEqual(a.end, b.end)
                self.assertTrue(torch.equal(a.k[:, :a.end], b.k[:, :b.end]))
                self.assertTrue(torch.equal(a.v[:, :a.end], b.v[:, :b.end]))

    def assertLatentsEqual(self, left, right):
        self.assertTrue(torch.equal(left, right))

    def assertLifetimesRetired(self, bridge, summary=None):
        bridge.drain_all()  # archival continues behind requests; a checkpoint publishes and retires it
        lease = ((summary or {}).get("restore") or {}).get("lease") or {}
        self.assertTrue(lease.get("closed", True))
        self.assertEqual(lease.get("open_handles", 0), 0)
        self.assertEqual(rk.pending_deferred(), 0)
        self.assertEqual(bridge.archive.runtime.residency.stats()["reserved_bytes"], 0)
        self.assertFalse(bridge.active_request)


class ConfigAndProfileTests(unittest.TestCase):
    def test_runtime_config_defaults_and_validation(self):
        config = ReelCacheRuntimeConfig.from_dict({"store_root": "archive"})
        self.assertEqual((config.mode, config.representation, config.hevc_inter_prediction, config.decoder),
                         ("archive_native", "svd_hevc", False, "nvdec_packed"))
        self.assertEqual((config.packing, config.reuse, config.refit, config.rate_lag, config.canvas),
                         ("node", True, True, 3, "canonical"))  # F4: rate_lag 3 (was 1)
        self.assertEqual(config.codec_template().inter_prediction, False)
        self.assertTrue(config.targets("longlive2").provisional)
        self.assertEqual((config.targets("longlive2").key, config.targets("longlive2").value), (0.98, 0.95))
        self.assertEqual(ReelCacheRuntimeConfig.from_dict(config.to_dict()), config)
        for bad in ({}, {"store_root": "a", "unknown": 1}, {"store_root": "a", "hevc_inter_prediction": True},
                    {"store_root": "a", "energy_key": 0.98}, {"store_root": "a", "representation": "svd"},
                    {"store_root": "a", "mode": "reconstructed_hot"}, {"store_root": "a", "rate_lag": 9},
                    {"store_root": "a", "max_source_chunks": 4, "max_unpublished": 2},
                    {"store_root": "a", "verify_post_codec": True, "validate_decode": False},
                    {"store_root": "a", "profile": "x", "profile_fields": {}}, {"store_root": "a", "canvas": "big"}):
            with self.assertRaises(ConfigurationError, msg=str(bad)):
                ReelCacheRuntimeConfig.from_dict(bad)
        raw = ReelCacheRuntimeConfig.from_dict({"store_root": "a", "representation": "raw",
                                             "hevc_inter_prediction": True})
        self.assertEqual(raw.codec_template().kind, "raw")
        custom = ReelCacheRuntimeConfig.from_dict({"store_root": "a", "energy_key": 0.99, "energy_value": 0.9})
        self.assertEqual((custom.targets("x").key, custom.targets("x").value), (0.99, 0.9))

    def test_matmul_precision_policy_reaches_the_rank_method(self):
        with self.assertRaises(ConfigurationError):
            ReelCacheRuntimeConfig.from_dict({"store_root": "x", "matmul_precision": "tf32"})
        with self.assertRaises(ConfigurationError):
            ModelProfile.from_dict({**PROFILE.to_dict(), "matmul_precision": "tf32"})
        config = ReelCacheRuntimeConfig.from_dict({"store_root": "x", "matmul_precision": "fp32_verified_fp64"})
        self.assertEqual(config.content_dict()["matmul_precision"], "fp32_verified_fp64")
        self.assertNotIn("store_root", config.content_dict())

    def test_profiles_match_by_exact_model_path(self):
        llm2 = {"transformer": "LongLive2Transformer3DModel"}
        self.assertIs(profile_for({**llm2, "model_path": "Rabinovich/LongLive-2.0-5B-Diffusers"}), LONGLIVE2)
        self.assertIs(profile_for({**llm2, "model_path": SNAPSHOT}), LONGLIVE2)  # HF cache snapshot, bound revision
        for model in (llm2, {**llm2, "model_path": SNAPSHOT.replace("37664a64", "00000000")},
                      {**llm2, "model_path": "/x/Rabinovich/LongLive-2.0-5B-Diffusers"},
                      {"transformer": "CausalWanTransformer3DModel", "model_path": "Rabinovich/LongLive-2.0-5B-Diffusers"}):
            with self.assertRaises(UnsupportedError, msg=str(model)):
                profile_for(model)
        self.assertIs(profile_for({**llm2, "model_path": "/local/converted"}, "longlive2"), LONGLIVE2)  # explicit
        with self.assertRaises(UnsupportedError):
            profile_for({"transformer": "Other", "model_path": SNAPSHOT}, "longlive2")
        self.assertEqual(LONGLIVE2.channels, 3072)
        bad = request(handle("s"))
        bad.block_sizes = (3, 3)
        with self.assertRaises(UnsupportedError):
            PROFILE.check_request(bad)
        PROFILE.check_request(request(handle("s")))

    def test_profile_fields_window_none_aliases_and_registration(self):
        fields = {  # the shape of Step 12's bridge_profile_fields() for a full-history 1.3B profile
            "name": "causvid", "backbone": "wan2.1-t2v-1.3b-causal", "transformers": (),
            "model_paths": ("/models/profiles/reelcache-causvid-480p-s12v1",), "checkpoint": {"binding_fingerprint": "f"},
            "num_layers": 30, "num_heads": 12, "head_dim": 128, "chunk_frames": 3, "visibility": "full_history",
            "sink_frames": 0, "window_frames": None, "energy_profile": "causvid", "native_dtype": "bfloat16",
            "rope": "sglang_causal_wan", "cfg_streams": ("cond",), "recache": "none", "lifecycle": (),
            "defaults": {"height": 480, "width": 832, "tokens_per_frame": 1560},
        }
        causvid = ModelProfile.from_dict(fields)
        self.assertIsNone(causvid.window_frames)
        self.assertEqual(ModelProfile.from_dict(causvid.to_dict()), causvid)
        shared = {"transformer": "ReelCacheCausalWanTransformer3DModel"}
        self.assertTrue(causvid.matches({**shared, "model_path": "/models/profiles/reelcache-causvid-480p-s12v1/"}))
        self.assertFalse(causvid.matches({**shared, "model_path": "/models/profiles/reelcache-raven-480p-s12v1"}))
        self.assertIs(profile_for({**shared, "model_path": fields["model_paths"][0]}, fields=fields).name, "causvid")
        with self.assertRaises(UnsupportedError):  # a path-bound profile needs its exact path even when named
            profile_for({**shared, "model_path": "/elsewhere"}, fields=fields)
        longlive1 = ModelProfile.from_dict({**fields, "name": "longlive1", "visibility": "sink_rolling_window",
                                            "sink_frames": 3, "window_frames": 12, "energy_profile": "longlive1",
                                            "recache": "kv_recache_on_prompt_switch",
                                            "model_paths": ("/models/profiles/reelcache-longlive1-480p-s12v1",)})
        self.assertEqual(longlive1.visibility, "sink_window")
        for bad in ({**fields, "window_frames": None, "visibility": "sink_window"},
                    {**fields, "window_frames": None, "sink_frames": 3}, {**fields, "window_frames": 3,
                                                                          "sink_frames": 3},
                    {**fields, "model_paths": ()}, {**fields, "unknown": 1}, {**fields, "native_dtype": "int8"}):
            with self.assertRaises(ConfigurationError, msg=str(bad)):
                ModelProfile.from_dict(bad)
        req = request(handle("s"), blocks=4)
        full = ModelProfile.from_dict({**fields, "num_layers": LAYERS, "num_heads": HEADS, "head_dim": DIM,
                                       "chunk_frames": FRAMES, "model_paths": ("toy/full",)})
        full.check_request(req)  # request-sized buffer: 8 chunks of room >= 4 requested
        req.block_sizes = (FRAMES,) * 9
        with self.assertRaises(UnsupportedError):
            full.check_request(req)
        from reelcache.integrations.sglang import profiles as registry

        self.addCleanup(registry._PROFILES.pop, "causvid", None)
        register_profile(causvid)
        register_profile(causvid)  # idempotent
        with self.assertRaises(ConfigurationError):  # another name on the same exact path
            register_profile(ModelProfile.from_dict({**fields, "name": "causvid-copy"}))
        self.assertIs(profile_for({**shared, "model_path": fields["model_paths"][0]}), causvid)

    def test_core_and_bridge_never_import_sglang(self):
        script = (
            "import sys, reelcache, reelcache.manager, reelcache.codecs, reelcache.restore_kernels, reelcache.packed_decode\n"
            "assert not [n for n in sys.modules if n.startswith('reelcache.integrations')], 'core imported integrations'\n"
            "import reelcache.integrations.sglang\n"
            "assert not [n for n in sys.modules if n == 'sglang' or n.startswith('sglang.')], 'bridge imported sglang'\n"
            "print('ok')\n"
        )
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


class SessionLifecycleChecks:
    """Lifecycle checks shared by the RAW reference and the Step 6 lineage backend."""

    def test_no_session_handle_means_ordinary_generation(self):
        bridge = self.bridge()
        summary, latents, caches, _ = drive(bridge, request(None))
        self.assertIsNone(summary)
        self.assertEqual(bridge.archive.sessions(), {})

    def test_cfg_streams_commit_atomically_and_restore_separately(self):
        reference = self.bridge("reference")
        _, expected_latents, expected_caches, _ = drive(reference, request(handle("ref"), streams=("cond", "uncond")))
        bridge = self.bridge()
        first, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=1), streams=("cond", "uncond")))
        node = bridge.archive.load(first["head"])
        self.assertEqual(node.streams, ("cond", "uncond"))
        if self.representation == "raw":
            self.assertEqual(sorted(bridge.archive._manifest(first["head"])["packets"]), ["cond", "uncond"])
        _, latents, caches, _ = drive(bridge, request(handle("s1", "resume"), streams=("cond", "uncond")))
        self.assertLatentsEqual(latents, expected_latents)
        self.assertCachesEqual(caches, expected_caches)

    def test_fork_and_rewind_are_isolated_branches(self):
        reference = self.bridge("reference")
        _, expected_latents, expected_caches, _ = drive(reference, request(handle("ref")))
        bridge = self.bridge()
        drive(bridge, request(handle("s1", pause_after_chunks=3)))
        head_s1 = bridge.archive.head("s1")
        checkpoints = bridge.controller.checkpoints("s1")
        self.assertEqual([item["chunks"] for item in checkpoints], [0, 1, 2, 3])
        forked, latents, caches, _ = drive(bridge, request(handle("s2", "fork", from_session="s1", checkpoint=1)))
        self.assertEqual((forked["resumed_from_chunks"], forked["generated_chunks"]), (1, 4))
        self.assertLatentsEqual(latents, expected_latents)
        self.assertCachesEqual(caches, expected_caches)
        self.assertEqual(bridge.archive.head("s1"), head_s1)  # the source is untouched
        rewound, latents, caches, _ = drive(bridge, request(handle("s1", "rewind", checkpoint=2)))
        self.assertEqual(rewound["resumed_from_chunks"], 2)
        self.assertLatentsEqual(latents, expected_latents)
        self.assertNotEqual(bridge.archive.head("s1"), head_s1)
        bridge.drain_all()
        path_s2 = [node.ref for node in bridge.archive.path(bridge.archive.head("s2"))]
        path_s1 = [node.ref for node in bridge.archive.path(bridge.archive.head("s1"))]
        self.assertEqual(path_s1[:2], path_s2[:2])  # shared root and first chunk, stored once
        if self.representation == "raw":
            # Exact continuations regenerate identical content: content-only ids deduplicate the branches' nodes.
            self.assertEqual(path_s1[2:], path_s2[2:])
        else:
            self.assertNotEqual(path_s1[2:], path_s2[2:])  # continuations of lossy restores are new content
        with self.assertRaises(StateError):
            bridge.controller.fork("s1", "s2")

    def test_two_sessions_keep_separate_state(self):
        expected = {}
        for prompt in ("A", "B"):
            reference = self.bridge(f"reference-{prompt}")
            _, latents, caches, _ = drive(reference, request(handle("ref"), prompt=prompt))
            expected[prompt] = (latents, caches)
        bridge = self.bridge()
        drive(bridge, request(handle("a", pause_after_chunks=1), prompt="A"))
        drive(bridge, request(handle("b", pause_after_chunks=2), prompt="B"))
        _, latents_a, caches_a, _ = drive(bridge, request(handle("a", "resume"), prompt="A"))
        _, latents_b, caches_b, _ = drive(bridge, request(handle("b", "resume"), prompt="B"))
        self.assertLatentsEqual(latents_a, expected["A"][0])
        self.assertLatentsEqual(latents_b, expected["B"][0])
        self.assertCachesEqual(caches_a, expected["A"][1])
        self.assertCachesEqual(caches_b, expected["B"][1])
        with self.assertRaises(UnsupportedError):
            drive(bridge, request(handle("a", "rewind", checkpoint=1), prompt="B"))
        self.assertEqual(bridge.controller.get("a").status, "completed")

    def test_failure_keeps_the_previous_checkpoint_and_lifetimes(self):
        bridge = self.bridge()
        drive(bridge, request(handle("s1", pause_after_chunks=1)))
        head = bridge.archive.head("s1")
        with self.assertRaises(RuntimeError):
            drive(bridge, request(handle("s1", "resume")), fail_at=2)
        bridge.checkpoint("s1")  # chunk 1 was submitted before the failure: it publishes behind the request
        state = bridge.controller.get("s1")
        self.assertEqual(state.status, "failed")
        self.assertEqual(sum(node.kind == "chunk" for node in bridge.archive.path(state.head_ref)), 2)
        self.assertNotEqual(state.head_ref, head)  # chunk 1 completed before the failure and was published
        self.assertFalse(bridge.active_request)
        resumed, _, _, _ = drive(bridge, request(handle("s1", "resume")))
        self.assertEqual(resumed["generated_chunks"], 4)
        # A running request blocks close and a second request; pause requests reach the running session.
        req = request(handle("s1", "rewind", checkpoint=1))
        run = bridge.begin_request(req)
        with self.assertRaises(PendingError):
            bridge.close()
        with self.assertRaises(PendingError):
            bridge.begin_request(request(handle("s3")))
        bridge.controller.request_pause("s1")
        run.finish(error=None)
        with self.assertRaises(StateError):
            bridge.controller.request_pause("s1")
        bridge.close()
        self.assertEqual(rk.pending_deferred(), 0)
        with self.assertRaises(StateError):
            bridge.begin_request(request(handle("s4")))

    def test_restore_rejects_inconsistent_requests(self):
        bridge = self.bridge()
        drive(bridge, request(handle("s1")))
        with self.assertRaises(ConfigurationError):  # nothing left to generate
            drive(bridge, request(handle("s1", "resume")))
        with self.assertRaises(StateError):
            drive(bridge, request(handle("s1")))  # generate again without replace
        replaced, _, _, _ = drive(bridge, request(handle("s1", replace=True)))
        self.assertEqual(replaced["generated_chunks"], 4)
        with self.assertRaises(StateError):
            drive(bridge, request(handle("missing", "resume")))
        shifted = request(handle("s1", "rewind", checkpoint=2))
        shifted.pos_start_base = 1
        head = bridge.checkpoint("s1")["head"]  # the replaced session's commits publish behind its request
        self.assertEqual(bridge.archive.head("s1"), head)
        with self.assertRaises(UnsupportedError):  # geometry differs; the head is not rewound
            drive(bridge, shifted)
        self.assertEqual(bridge.archive.head("s1"), head)
        if self.representation == "raw":
            with self.assertRaises(IntegrityError):
                bridge.archive.load(bridge.archive.store.put_manifest({"type": "other"}))

    def test_request_path_reopen_is_reported_with_the_restore(self):
        """A request that reopens a persisted session (a new worker, no ``open_session``) reports the reopen
        (manifest load + session prewarm when configured) next to its restore: the cold resume's restore time
        includes its per-session work (restore accounting rule); ``open_session`` first = pre-staged."""
        first = self.bridge("reopen")
        for name in ("s1", "s2", "s3", "src"):
            drive(first, request(handle(name, pause_after_chunks=2)))
        first.close()
        cases = (("s1", {}, "resume"), ("s2", {"prewarm": False}, "resume"), ("fork", {}, "fork"))
        for name, settings, action in cases:
            worker = self.bridge("reopen", **settings)
            extra = {"from_session": "src", "checkpoint": 2} if action == "fork" else {}
            summary, *_ = drive(worker, request(handle(name, action, **extra)))
            restore = summary["restore"]
            opened = restore["session_open"]
            self.assertEqual([item["session"] for item in opened], ["src" if action == "fork" else name])
            self.assertGreater(opened[0]["seconds"], 0.0)
            if not settings.get("prewarm", True):
                self.assertEqual(opened[0]["prewarm_seconds"], 0.0)
            self.assertGreaterEqual(restore["total_with_session_open_seconds"], restore["total_seconds"])
            worker.close()
        staged = self.bridge("reopen")
        staged.open_session("s3")  # pre-staged before the request: nothing reopened on the request path
        summary, *_ = drive(staged, request(handle("s3", "resume")))
        self.assertEqual(summary["restore"]["session_open"], [])
        self.assertAlmostEqual(summary["restore"]["total_with_session_open_seconds"],
                               summary["restore"]["total_seconds"], delta=0.05)

    def test_reopen_after_restart_restores_from_the_store(self):
        reference = self.bridge("reference")
        _, expected_latents, expected_caches, _ = drive(reference, request(handle("ref")))
        first = self.bridge("shared")
        drive(first, request(handle("s1", pause_after_chunks=2)))
        first.close()
        second = self.bridge("shared")  # a new worker on the same store
        opened = second.open_session("s1")
        self.assertEqual((opened["status"], opened["generated_chunks"]), ("paused", 2))
        self.assertEqual(opened["history"][0]["event"], "reopened")
        resumed, latents, caches, _ = drive(second, request(handle("s1", "resume")))
        self.assertEqual(resumed["resumed_from_chunks"], 2)
        if self.representation == "raw":
            self.assertLatentsEqual(latents, expected_latents)
            self.assertCachesEqual(caches, expected_caches)
        self.assertLifetimesRetired(second, resumed)


    def test_fp32_keys_with_a_bf16_cache(self):
        """RAVEN: FP32 pre-RoPE keys at the attention input, BF16 values and cache (capture_dtypes)."""
        raven = ModelProfile.from_dict({**PROFILE.to_dict(), "name": "fixture-fp32k",
                                        "capture_dtypes": {"K": "float32", "V": "bfloat16"}})
        self.assertEqual(raven.to_dict()["capture_dtypes"], {"K": "float32", "V": "bfloat16"})
        self.assertNotIn("capture_dtypes", PROFILE.to_dict())  # unset optional fields keep fingerprints

        def fp32(req):
            req.key_dtype = torch.float32
            return req

        reference = self.bridge("reference", profile=raven)
        _, expected_latents, expected_caches, _ = drive(reference, fp32(request(handle("ref"))))
        bridge = self.bridge(profile=raven)
        self.assertEqual(bridge.model.dtype_for("K"), "float32")
        first, _, _, _ = drive(bridge, fp32(request(handle("s1", pause_after_chunks=2))))
        self.assertEqual(bridge.archive.path(first["head"])[-1].record["captured_dtype"],
                         {"K": "float32", "V": "bfloat16"})
        resumed, latents, caches, _ = drive(bridge, fp32(request(handle("s1", "resume"))))
        self.assertEqual(resumed["restore"]["mode"], "archive")
        self.assertLatentsEqual(latents, expected_latents)
        self.assertCachesEqual(caches, expected_caches)
        if self.representation == "raw":  # the archived key is the FP32 input of the engine's rotation
            key = bridge.archive.raw_matrix(first["head"], "cond", rk_key(0, "K"))
            self.assertEqual(key.dtype, torch.float32)
        else:  # native bytes count 4 bytes per FP32 key element and 2 per BF16 value element
            tokens = FRAMES * TPF
            self.assertEqual(first["commits"][0]["native_bytes"], LAYERS * tokens * CHANNELS * (4 + 2))
        with self.assertRaises(IntegrityError):  # a BF16 key is not what this profile captures
            drive(self.bridge("mismatch", profile=raven), request(handle("s2")))

    def test_identical_sessions_produce_identical_ids(self):
        """Manifests hash content only: identical sessions archived by different workers/stores (different request
        ids, timings, store paths) produce identical root, node, record and packet ids; run telemetry (request
        ids, timings, configuration) lives in the stores' unhashed sidecars."""
        runs = []
        for store in ("run-a", "run-b"):
            bridge = self.bridge(store)
            drive(bridge, request(handle("s1", pause_after_chunks=2), streams=("cond", "uncond")))
            drive(bridge, request(handle("s1", "resume"), streams=("cond", "uncond")))
            head = bridge.checkpoint("s1")["head"]
            nodes = [node.ref for node in bridge.archive.path(head)]
            with bridge.archive.store.transaction():  # every object reachable from the head
                objects = sorted(bridge.archive.store._walk([head], verify=False))
            runs.append((bridge, nodes, objects))
        (first, nodes_a, objects_a), (second, nodes_b, objects_b) = runs
        self.assertEqual(len(nodes_a), 5)
        self.assertEqual(nodes_a, nodes_b)
        self.assertGreater(len(objects_a), len(nodes_a))
        self.assertEqual(objects_a, objects_b)
        for bridge in (first, second):
            store = bridge.archive.store
            root = store.telemetry(nodes_a[0])["bridge"]
            self.assertEqual(root["config"]["store_root"], bridge.config.store_root)  # run configuration, unhashed
            chunk = store.telemetry(nodes_a[-1])["bridge"]
            self.assertTrue(chunk["request_id"].startswith("r-A-"))
        self.assertNotEqual(first.archive.store.telemetry(nodes_a[-1])["bridge"]["request_id"],
                            second.archive.store.telemetry(nodes_a[-1])["bridge"]["request_id"])
        if self.representation == "raw":
            packet = first.archive._manifest(nodes_a[-1])["packets"]["cond"][0]
            self.assertNotIn("encode_seconds", first.archive.store.get_manifest(packet)["metrics"])
            self.assertIn("encode_seconds", first.archive.store.telemetry(packet)["metrics"])
            # Within one store, an identical second session shares every node of the first.
            drive(first, request(handle("s2"), streams=("cond", "uncond")))
            self.assertEqual(first.archive.head("s2"), nodes_a[-1])

    # -- resident native KV across requests ---------------------------------------------------------------------
    def assertCachesBitwise(self, left, right):
        BridgeTestCase.assertCachesEqual(self, left, right)

    def test_resume_on_resident_kv_is_exact_and_restores_nothing(self):
        reference = self.bridge("reference")
        _, expected_latents, expected_caches, expected_rng = drive(reference, request(handle("ref")))
        bridge = self.bridge()
        stage = Stage()
        first, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=2)), stage=stage)
        self.assertTrue(first["durable"] and first["published"])  # an explicit pause publishes before returning
        self.assertEqual(bridge.resident["session_id"], "s1")
        resumed, latents, caches, rng = drive(bridge, request(handle("s1", "resume")), stage=stage)
        self.assertEqual((resumed["restore"]["mode"], resumed["resumed_from_chunks"]), ("resident", 2))
        self.assertNotIn("plans", resumed["restore"])
        # The session continued on its native KV: exact for both representations (nothing was decoded).
        self.assertTrue(torch.equal(latents, expected_latents))
        self.assertCachesBitwise(caches, expected_caches)
        self.assertTrue(torch.equal(rng.get_state(), expected_rng.get_state()))
        self.assertLifetimesRetired(bridge, resumed)

    def test_resident_kv_is_not_reused_after_other_writes_or_head_moves(self):
        expected = {}
        for prompt in ("A", "B"):
            _, latents, caches, _ = drive(self.bridge(f"reference-{prompt}"), request(handle("ref"), prompt=prompt))
            expected[prompt] = (latents, caches)
        bridge = self.bridge()
        stage = Stage()
        # An ordinary request (no session: the bridge is not called) writes the stage caches in between.
        drive(bridge, request(handle("a", pause_after_chunks=2), prompt="A"), stage=stage)
        drive(bridge, request(None, prompt="B"), stage=stage)
        resumed, latents, caches, _ = drive(bridge, request(handle("a", "resume"), prompt="A"), stage=stage)
        self.assertEqual(resumed["restore"]["mode"], "archive")
        self.assertIn("changed", resumed["restore"]["resident"])
        self.assertLatentsEqual(latents, expected["A"][0])
        self.assertCachesEqual(caches, expected["A"][1])
        # Two sessions interleaved on one stage: each resume finds the other session's KV.
        drive(bridge, request(handle("b", pause_after_chunks=1), prompt="B"), stage=stage)
        drive(bridge, request(handle("c", pause_after_chunks=1), prompt="A"), stage=stage)
        resumed, latents, _, _ = drive(bridge, request(handle("b", "resume"), prompt="B"), stage=stage)
        self.assertEqual(resumed["restore"]["mode"], "archive")
        self.assertIn("held by session 'c'", resumed["restore"]["resident"])
        self.assertLatentsEqual(latents, expected["B"][0])
        # A rewind moves the head: the resident KV (generated head) is never used for the earlier checkpoint.
        drive(bridge, request(handle("d", pause_after_chunks=3), prompt="A"), stage=stage)
        rewound, latents, _, _ = drive(bridge, request(handle("d", "rewind", checkpoint=1), prompt="A"), stage=stage)
        self.assertEqual((rewound["restore"]["mode"], rewound["resumed_from_chunks"]), ("archive", 1))
        self.assertLatentsEqual(latents, expected["A"][0])
        # Disabled reuse always restores from the archive.
        plain = self.bridge("plain", resident_reuse=False)
        plain_stage = Stage()
        drive(plain, request(handle("e", pause_after_chunks=2), prompt="A"), stage=plain_stage)
        resumed, _, _, _ = drive(plain, request(handle("e", "resume"), prompt="A"), stage=plain_stage)
        self.assertEqual((resumed["restore"]["mode"], resumed["restore"]["resident"]), ("archive", "disabled"))


class RawSessionTests(SessionLifecycleChecks, BridgeTestCase):
    representation = "raw"

    def test_inference_mode_capture_restores_bitwise(self):
        """SGLang denoises under torch.inference_mode: its tensors have no version counter."""
        reference = self.bridge("reference")
        _, expected_latents, expected_caches, _ = drive(reference, request(handle("ref")))
        bridge = self.bridge()
        with torch.inference_mode():
            first, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=2)))
            resumed, latents, caches, _ = drive(bridge, request(handle("s1", "resume")))
        node = bridge.archive.path(first["head"])[-1]
        self.assertEqual(node.record["mutation_check"], "unavailable_inference_tensors")
        self.assertEqual(resumed["resumed_from_chunks"], 2)
        self.assertLatentsEqual(latents, expected_latents)
        self.assertCachesEqual(caches, expected_caches)

    def test_raw_pause_resume_is_bitwise_identical(self):
        reference = self.bridge("reference")
        expected, expected_latents, expected_caches, expected_rng = drive(reference, request(handle("ref")))
        self.assertEqual((expected["status"], expected["generated_chunks"]), ("completed", 4))
        bridge = self.bridge()
        first, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=2)))
        self.assertTrue(first["paused"])
        self.assertEqual(first["generated_chunks"], 2)
        self.assertEqual(first["archive"]["backend"], "raw_exact")
        resumed, latents, caches, rng = drive(bridge, request(handle("s1", "resume")))
        self.assertEqual((resumed["resumed_from_chunks"], resumed["generated_chunks"]), (2, 4))
        self.assertLatentsEqual(latents, expected_latents)
        self.assertCachesEqual(caches, expected_caches)
        self.assertTrue(torch.equal(rng.get_state(), expected_rng.get_state()))
        plan = resumed["restore"]["plans"]["cond"]
        self.assertEqual(plan["attended_tokens"], 2 * FRAMES * TPF)
        self.assertEqual([span["chunk"] for span in plan["spans"]], [0, 1])
        self.assertEqual(resumed["restore"]["lease"]["rope_mode"], "native")
        self.assertLifetimesRetired(bridge, resumed)


def switch_request(session, written=None, **kwargs):
    req = request(session, blocks=5, **kwargs)
    switch = {"at": 3, "prompt": "B"}
    if written is not None:
        switch["written"] = written
    req.conditioning = {**req.conditioning, "switch": switch}
    return req


class RecacheEpochChecks:
    def test_recache_epoch_restores_newest_kv_and_rewind_restores_the_old(self):
        reference = self.bridge("reference")
        expected, expected_latents, expected_caches, _ = drive(reference, switch_request(handle("ref")))
        self.assertEqual([item.get("recache") for item in expected["commits"]].count("switch-3"), 1)
        bridge = self.bridge()
        first, _, _, _ = drive(bridge, switch_request(handle("s1", pause_after_chunks=4)))
        kinds = [node.kind for node in bridge.archive.path(first["head"])]
        recache_nodes = 2 if self.representation == "raw" else 1  # RAW: one node per range; lineage: one epoch node
        self.assertEqual(kinds, ["root", "chunk", "chunk", "chunk"] + ["recache"] * recache_nodes + ["chunk"])
        resumed, latents, caches, _ = drive(bridge, switch_request(handle("s1", "resume")))
        plan = resumed["restore"]["plans"]["cond"]
        recached = {node.ref for node in bridge.archive.path(first["head"]) if node.kind == "recache"}
        self.assertEqual([span["chunk"] for span in plan["spans"]], [0, 1, 2, 3])
        self.assertEqual([span["source"] for span in plan["spans"]], ["chunk", "recache", "recache", "chunk"])
        self.assertEqual(sum(span["node"] in recached for span in plan["spans"]), 2)  # c1, c2 from epoch switch-3
        self.assertEqual(latents.shape, expected_latents.shape)
        if self.representation == "raw":
            self.assertTrue(torch.equal(latents, expected_latents))
            self.assertCachesEqual(caches, expected_caches)
        # Rewind before the switch: the old KV of c1/c2 is restored and the branch recomputes its own epoch.
        pre_switch = bridge.archive.path(first["head"])[3].ref
        rewound, latents, caches, _ = drive(bridge, switch_request(handle("s1", "rewind", checkpoint=pre_switch)))
        plan = rewound["restore"]["plans"]["cond"]
        self.assertFalse(any(span["node"] in recached for span in plan["spans"]))
        if self.representation == "raw":
            self.assertTrue(torch.equal(latents, expected_latents))
            self.assertCachesEqual(caches, expected_caches)
        bridge.checkpoint("s1")  # the completed request returned before its commits were published
        new_epoch = [node.ref for node in bridge.archive.path(bridge.archive.head("s1")) if node.kind == "recache"]
        self.assertEqual(len(new_epoch), recache_nodes)

    def test_recache_archives_only_written_whole_chunks_never_the_sink(self):
        bridge = self.bridge()
        # Only the second chunk of the window was written back: one archived range.
        first, _, _, _ = drive(bridge, switch_request(handle("s1", pause_after_chunks=4),
                                                      written=((2 * FRAMES * TPF, FRAMES * TPF),)))
        recache = [item for item in first["commits"] if item.get("recache")]
        self.assertEqual([item["chunk_index"] for item in recache[0]["ranges"]], [2])
        with self.assertRaises(IntegrityError):  # half a chunk is not an archived range
            drive(self.bridge("half"), switch_request(handle("s2"), written=((2 * FRAMES * TPF, TPF),)))
        sinked = ModelProfile.from_dict({**PROFILE.to_dict(), "name": "fixture-sink", "sink_frames": FRAMES,
                                         "visibility": "sink_window"})
        with self.assertRaises(IntegrityError):  # the window starts at chunk 1; chunk 0 (the sink) is not in it
            drive(self.bridge("sink", profile=sinked),
                  switch_request(handle("s3"), written=((0, FRAMES * TPF),), sink_tokens=FRAMES * TPF))
        early = switch_request(handle("s4"), sink_tokens=FRAMES * TPF)
        early.conditioning["switch"]["at"] = 2  # window = chunks 0-1: the sink chunk would be re-archived
        with self.assertRaises(IntegrityError):
            drive(self.bridge("sink2", profile=sinked), early)


def timeline_request(session, prompts, *, at=3, **kwargs):
    """A prompt switch carried as a timeline field: the conditioning excludes the schedule, the request names each
    block's conditioning (``CausalKVRequest.block_conditioning``) — the stage contract of prompt-switch profiles."""
    req = request(session, blocks=5, **kwargs)
    req.switch = {"at": at, "prompt": prompts[1]}
    req.block_conditioning = tuple(prompts[0] if index < at else prompts[1] for index in range(5))
    return req


class TimelineConditioningChecks:
    def test_fork_under_another_future_prompt_keeps_the_restored_blocks(self):
        reference = self.bridge("reference")
        _, expected_c, _, _ = drive(reference, timeline_request(handle("ref"), ("A", "C")))
        bridge = self.bridge()
        first, _, _, _ = drive(bridge, timeline_request(handle("s1", pause_after_chunks=4), ("A", "B")))
        nodes = bridge.archive.path(bridge.checkpoint("s1")["head"])
        self.assertEqual([node.kind for node in nodes][:4], ["root", "chunk", "chunk", "chunk"])
        self.assertEqual(nodes[-1].record.get("block_conditioning"), "B")
        recache_ref = [node.ref for node in nodes if node.kind == "recache"][-1]  # the epoch's (last) node
        self.assertEqual(bridge.archive.load(recache_ref).record["precedes_block"], 3)
        # Fork at the pre-switch chunk node under prompt C: blocks 0-2 match, the fork runs its own C epoch. (The
        # chunk-count checkpoint 3 names the newest node with three chunks: the B epoch, rejected below.)
        pre_switch = nodes[3].ref
        forked, latents, _, _ = drive(bridge, timeline_request(
            handle("s2", "fork", from_session="s1", checkpoint=pre_switch), ("A", "C")))
        self.assertEqual((forked["resumed_from_chunks"], forked["generated_chunks"]), (3, 5))
        self.assertEqual((forked["restore"]["head_kind"], forked["restore"]["head_block"]), ("chunk", 2))
        self.assertLatentsEqual(latents, expected_c)
        fork_nodes = bridge.archive.path(bridge.checkpoint("s2")["head"])
        fork_epoch = [node for node in fork_nodes if node.kind == "recache"]
        self.assertTrue(fork_epoch and all(node.record.get("block_conditioning") == "C" for node in fork_epoch))
        self.assertNotIn(recache_ref, {node.ref for node in fork_nodes})
        # The B epoch itself cannot be restored under C; nothing is created or moved.
        for checkpoint in (recache_ref, 3):
            with self.assertRaises(UnsupportedError):
                drive(bridge, timeline_request(handle("s3", "fork", from_session="s1", checkpoint=checkpoint),
                                               ("A", "C")))
        self.assertIsNone(bridge.archive.head("s3"))
        head = bridge.archive.head("s1")
        with self.assertRaises(UnsupportedError):
            drive(bridge, timeline_request(handle("s1", "resume"), ("A", "C")))
        self.assertEqual(bridge.archive.head("s1"), head)
        # A rewind to the epoch reports it as the restored head: the stage must not rerun that recache.
        rewound, _, _, _ = drive(bridge, timeline_request(handle("s1", "rewind", checkpoint=recache_ref),
                                                          ("A", "B")))
        self.assertEqual((rewound["restore"]["head_kind"], rewound["restore"]["head_epoch"],
                          rewound["restore"]["head_block"]), ("recache", "switch-3", 3))
        self.assertEqual(rewound["resumed_from_chunks"], 3)


class RecacheEpochTests(RecacheEpochChecks, TimelineConditioningChecks, BridgeTestCase):
    representation = "raw"

    def test_partial_epoch_nodes_are_not_checkpoints(self):
        """The RAW reference chains one recache node per range; only the epoch's last node is a clean state."""
        bridge = self.bridge()
        drive(bridge, switch_request(handle("s1", pause_after_chunks=4)))
        listed = bridge.controller.checkpoints("s1")
        epoch = [item for item in listed if item["kind"] == "recache"]
        self.assertEqual([item["checkpoint"] for item in epoch], [False, True])
        with self.assertRaises(StateError):
            drive(bridge, switch_request(handle("s2", "fork", from_session="s1", checkpoint=epoch[0]["node"])))
        self.assertIsNone(bridge.archive.head("s2"))
        forked, _, _, _ = drive(bridge, switch_request(handle("s3", "fork", from_session="s1",
                                                              checkpoint=epoch[-1]["node"])))
        self.assertEqual((forked["restore"]["head_kind"], forked["restore"]["head_epoch"]), ("recache", "switch-3"))


class LineageSessionTests(SessionLifecycleChecks, BridgeTestCase):
    """The same lifecycle on the Step 6 runtime (lossy: exactness is replaced by the energy targets)."""

    representation = "svd_hevc"

    def bridge(self, store="archive", profile=PROFILE, **overrides):
        # The lossy restore path (freeze D: with the default raw tier, restores after generation are served from raw
        # copies, bitwise; tests/test_step7_raw_tier.py covers those).
        overrides.setdefault("raw_restore", "off")
        return super().bridge(store, profile, **overrides)

    def assertCachesEqual(self, left, right):
        # Lossy restores reproduce the native cache within the K98/V95 targets, not bitwise.
        for stream in left:
            for a, b in zip(left[stream], right[stream]):
                self.assertEqual(a.end, b.end)
                for name in ("k", "v"):
                    native = getattr(b, name)[:, :b.end].double()
                    restored = getattr(a, name)[:, :a.end].double()
                    self.assertLess(float((restored - native).norm() / native.norm()), 0.3)

    def assertLatentsEqual(self, left, right):
        # Continuations after a lossy restore drift within the toy model's tolerance.
        self.assertEqual(left.shape, right.shape)
        self.assertLess(float((left - right).norm() / right.norm()), 0.05)

    def test_lossy_pause_resume_meets_the_targets(self):
        reference = self.bridge("reference", representation="raw")
        _, _, raw_caches, _ = drive(reference, request(handle("ref"), blocks=2))
        # C3 source leases (freeze D: the raw tier releases a producer's tensors when it gives up its copy).
        bridge = self.bridge(verify_post_codec=True, raw_tier=False)
        first, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=1), blocks=2))
        self.assertEqual((first["archive"]["backend"], first["archive"]["asynchronous"]), ("lineage", True))
        self.assertEqual(first["archive"]["released_sources"], 1)
        commit = first["commits"][0]
        self.assertEqual((commit["representation"], commit["kind"], commit["node"]), ("svd_hevc", "chunk",
                                                                                     first["head"]))
        self.assertEqual(len(commit["matrices"]), 2 * LAYERS)
        for row in commit["matrices"]:
            self.assertGreaterEqual(row["stored_factor_energy"], row["target"])
            self.assertLessEqual(row["rank_view"], FRAMES * TPF)
            self.assertIsNotNone(row["post_codec_relative_error"])
        self.assertEqual(commit["options"]["packing"], "node")
        node = bridge.archive.runtime.load_node(first["head"])
        record = bridge.archive.runtime.load_record(node.matrix_records[0][1])
        packet = bridge.archive.runtime.load_packet(record.packet_ref)
        self.assertEqual(packet["metadata"]["schema"], "reelcache.tensor_pack.v2")
        self.assertIs(packet["metadata"]["codec"]["inter_prediction"], False)
        resumed, _, caches, _ = drive(bridge, request(handle("s1", "resume"), blocks=2))
        self.assertEqual(resumed["restore"]["lease"]["decoder"], "software")
        self.assertEqual(resumed["restore"]["lease"]["rope_mode"], "fused")
        tokens = FRAMES * TPF
        for layer in range(LAYERS):
            for name, bound in (("v", 0.3), ("k", 0.2)):
                restored = getattr(caches["cond"][layer], name)[:, :tokens].double()
                native = getattr(raw_caches["cond"][layer], name)[:, :tokens].double()
                error = float((restored - native).norm() / native.norm())
                self.assertGreater(error, 0.0)
                self.assertLess(error, bound)
        self.assertEqual(bridge.archive.registry.registry_id, "sglang")
        self.assertLifetimesRetired(bridge, resumed)

    def test_inference_mode_sources_archive_on_the_pipeline_thread(self):
        """Inference tensors captured under torch.inference_mode are read by the fitter thread outside it."""
        bridge = self.bridge()
        with torch.inference_mode():
            first, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=2), streams=("cond", "uncond")))
            resumed, _, _, _ = drive(bridge, request(handle("s1", "resume"), streams=("cond", "uncond")))
        self.assertEqual([item["kind"] for item in first["commits"]], ["chunk", "chunk"])
        self.assertTrue(all(item.get("node") for item in first["commits"]))
        self.assertEqual(bridge.archive.path(first["head"])[-1].record["mutation_check"],
                         "unavailable_inference_tensors")
        self.assertEqual((resumed["resumed_from_chunks"], resumed["status"]), (2, "completed"))
        self.assertLifetimesRetired(bridge, resumed)

    def test_native_rope_restore_inside_inference_mode(self):
        """The restore runs in the caller's torch.inference_mode (no workaround left): the engine's own RoPE and the
        in-place PyTorch writers fill the inference-tensor caches there."""
        reference = self.bridge("reference")
        with torch.inference_mode():
            _, expected_latents, expected_caches, _ = drive(reference, request(handle("ref")))
        bridge = self.bridge(rope_mode="native")
        with torch.inference_mode():
            drive(bridge, request(handle("s1", pause_after_chunks=2)))
            resumed, latents, caches, _ = drive(bridge, request(handle("s1", "resume")))
        self.assertTrue(caches["cond"][0].k.is_inference())
        self.assertEqual((resumed["restore"]["mode"], resumed["restore"]["lease"]["rope_mode"]), ("archive", "native"))
        self.assertLatentsEqual(latents, expected_latents)
        self.assertCachesEqual(caches, expected_caches)

    def test_rope_tables_from_inference_mode_engines_are_used_without_copies(self):
        """SGLang computes the engine RoPE tables under torch.inference_mode: they are used as returned
        (RopeTable treats inference tables as immutable), inside and outside inference mode."""
        from reelcache.integrations.sglang.restore import LineageInstall
        from reelcache.integrations.sglang.visibility import TokenSpan

        span = TokenSpan("n", 0, "cond", 0, FRAMES * TPF, 0, 0, 0.0, FRAMES, "window", FRAMES * TPF, "chunk")
        plan = SimpleNamespace(spans=(span,))
        caches = {"cond": [FullHistoryCache(8 * FRAMES * TPF) for _ in range(LAYERS)]}
        with torch.inference_mode():
            install = LineageInstall(plans={"cond": plan}, caches=caches, rope_cos_sin=rope_cos_sin,
                                     rotate_key=rotate_key, rope_mode="fused")
            positions = install.rope("n", (rk_key(0, "K"), "cond"), SimpleNamespace(token_start=0))
            inside = positions.table.kernel_tables("cpu")
        engine_cos, engine_sin = install.tables[("cond", 0)]
        self.assertTrue(engine_cos.is_inference() and engine_sin.is_inference())  # no normal-tensor copy
        outside = positions.table.kernel_tables("cpu")
        expected_cos, expected_sin = rope_cos_sin(0, FRAMES)
        for cos, sin in (inside, outside):
            self.assertTrue(torch.equal(cos, expected_cos) and torch.equal(sin, expected_sin))
        self.assertEqual(tuple(outside[0].shape), (FRAMES * TPF, DIM // 2))
        self.assertIsNone(install.rope("n", (rk_key(0, "V"), "cond"), SimpleNamespace(token_start=0)))

    def test_asynchronous_commits_publish_in_order_and_reuse_bases(self):
        bridge = self.bridge(rate_lag=1, max_source_chunks=2, max_unpublished=4)
        req = request(handle("s1"))
        summary, _, _, _ = drive(bridge, req)
        self.assertEqual((summary["status"], summary["generated_chunks"], summary["durable"]), ("completed", 4, False))
        head = bridge.checkpoint("s1")["head"]
        nodes = bridge.archive.path(head)
        self.assertEqual([node.kind for node in nodes], ["root"] + ["chunk"] * 4)
        self.assertEqual([node.chunk_index for node in nodes[1:]], [0, 1, 2, 3])
        commits = bridge.commit_reports(req.request_id)
        self.assertEqual([item["node"] for item in commits], [node.ref for node in nodes[1:]])
        decisions = {row["decision"] for item in commits[1:] for row in item["matrices"]}
        self.assertTrue(decisions - {"fresh"}, decisions)  # later chunks inherit/augment/refit basis versions
        frontiers = bridge.archive.frontiers("s1")
        self.assertEqual(frontiers["published_frontier"], frontiers["generated_frontier"])
        self.assertEqual(frontiers["published_ref"], head)
        stats = bridge.archive.stats()
        self.assertEqual(stats["pipeline"]["submitted"], stats["pipeline"]["published"])


def publication_hook(bridge, *, stage="publish", seconds=0.0, gate=None, fail=None):
    """Pipeline test seam: delay (or gate) one archival stage; ``fail`` makes that node's stage raise."""

    def hook(name, node_id):
        if name != stage:
            return
        if gate is not None:
            gate.wait(30)
        if seconds:
            time.sleep(seconds)
        if fail is not None and node_id == fail:
            raise RuntimeError(f"injected {name} failure of {node_id}")

    bridge.archive.pipeline.hooks = hook


def wait_until(predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.01)


class MatmulPrecisionTests(BridgeTestCase):
    representation = "svd_hevc"

    def test_staged_canvases_cover_the_largest_recache_node(self):
        """A node stages one host canvas per range before it encodes: fewer canvases than ranges would hang."""
        switching = ModelProfile.from_dict({**PROFILE.to_dict(), "name": "fixture-switch",
                                            "recache": "kv_recache_on_prompt_switch"})
        self.assertEqual(switching.max_recache_ranges, 8)  # 8-chunk buffer, no sink
        self.assertEqual(PROFILE.max_recache_ranges, 0)
        self.assertEqual(self.bridge("auto", profile=switching).archive.max_staged_canvases, 8)
        self.assertEqual(self.bridge("default").archive.max_staged_canvases, 2)
        with self.assertRaises(ConfigurationError):
            self.bridge("too-small", profile=switching, max_staged_canvases=2)
        # A profile without a bound gets an explicit error instead of a hang when a recache exceeds the staging.
        scene = ModelProfile.from_dict({**PROFILE.to_dict(), "name": "fixture-scene",
                                        "recache": "optional_scene_cut_clean_recache"})
        bridge = self.bridge("guard", profile=scene, max_staged_canvases=1, max_source_chunks=1, max_unpublished=1)
        with self.assertRaises(CapacityError):
            drive(bridge, switch_request(handle("s1")))  # its recache writes 2 chunk ranges
        self.assertEqual(bridge.controller.get("s1").status, "failed")

    def test_plan_memory_budget_reaches_the_archive_options(self):
        bridge = self.bridge(plan_memory_bytes=1 << 30, max_source_chunks=1)
        self.assertEqual(bridge.archive.options.plan_memory_bytes, 1 << 30)
        self.assertIsNone(self.bridge("unbounded").archive.options.plan_memory_bytes)
        self.assertIsNone(bridge.startup["cuda_memory_cap"])  # the cap applies to CUDA bridges only
        defaults = ReelCacheRuntimeConfig(store_root=str(self.workspace / "d"))
        self.assertEqual((defaults.decode_context_latents, defaults.plan_memory_bytes,
                          defaults.cuda_memory_limit_bytes), (40, None, None))
        for name, value in (("plan_memory_bytes", 0), ("cuda_memory_limit_bytes", -1), ("plan_memory_bytes", 1.5)):
            with self.assertRaises(ConfigurationError):
                ReelCacheRuntimeConfig(store_root=str(self.workspace / "d"), **{name: value})

    def test_profile_or_config_matmul_policy_reaches_the_rank_method(self):
        tf32 = ModelProfile.from_dict({**PROFILE.to_dict(), "name": "fixture-tf32",
                                       "matmul_precision": "fp32_verified_fp64"})
        for name, profile, settings, expected in (
                ("profile", tf32, {}, "fp32_verified_fp64"), ("default", PROFILE, {}, "fp32"),
                ("config", PROFILE, {"matmul_precision": "fp32_verified_fp64"}, "fp32_verified_fp64")):
            bridge = self.bridge(name, profile=profile, **settings)
            self.assertEqual(bridge.archive.method.matmul_precision, expected)


class RestoreWarmupTests(BridgeTestCase):
    """CR-12p2-11: the restore path's process-level cold start is paid when the backend attaches."""

    representation = "svd_hevc"
    GEOMETRY = ModelProfile.from_dict({**PROFILE.to_dict(), "name": "fixture-geometry",
                                       "defaults": {"height": 64, "width": 96, "tokens_per_frame": TPF}})

    def test_profile_default_grid(self):
        self.assertEqual(self.GEOMETRY.default_grid(), GRID)
        self.assertEqual(LONGLIVE2.default_grid(), (22, 40))
        self.assertIsNone(PROFILE.default_grid())  # no defaults: no synthetic chunk
        for defaults in ({"height": 64, "width": 96, "tokens_per_frame": 25},  # no integer square stride
                         {"height": 64, "width": 96}, {"height": 64.0, "width": 96, "tokens_per_frame": TPF}):
            profile = ModelProfile.from_dict({**PROFILE.to_dict(), "name": "fixture-odd", "defaults": defaults})
            self.assertIsNone(profile.default_grid())

    def test_kernel_warmup_shapes_cover_every_launch_group(self):
        """``restore_kernels.warmup`` shapes: lineage restores (pictures; fused or native RoPE, ranks up to
        min(channels, chunk tokens)) and RAW restores (dense; K and V split when their capture dtypes differ)."""
        from reelcache.integrations.sglang.restore import kernel_warmup_shapes

        model = self.bridge("shapes", profile=self.GEOMETRY).archive.model
        (fused,) = kernel_warmup_shapes(self.GEOMETRY, model, path="pictures", grid=GRID)
        self.assertEqual((fused.heads, fused.head_dim, fused.dtype, fused.layout, fused.source, fused.rope,
                          fused.ranks), (HEADS, DIM, torch.bfloat16, "BTHD", "pictures", "fused",
                                         (1, min(CHANNELS, FRAMES * TPF))))
        (native,) = kernel_warmup_shapes(self.GEOMETRY, model, path="pictures", rope_mode="native")
        self.assertEqual((native.rope, native.native_rope_dtype, native.ranks), ("native", torch.bfloat16,
                                                                                 (1, CHANNELS)))
        (dense,) = kernel_warmup_shapes(self.GEOMETRY, model, path="dense")
        self.assertEqual((dense.source, dense.rope, dense.source_dtype, dense.native_rope_dtype),
                         ("dense", "native", torch.bfloat16, torch.bfloat16))
        fp32 = ModelProfile.from_dict({**self.GEOMETRY.to_dict(), "name": "fixture-fp32-keys",
                                       "capture_dtypes": {"K": "float32", "V": "bfloat16"}})
        mixed = self.bridge("mixed", profile=fp32).archive.model
        keys, values = kernel_warmup_shapes(fp32, mixed, path="dense")
        self.assertEqual((keys.rope, keys.source_dtype, keys.native_rope_dtype, values.rope, values.source_dtype),
                         ("native", torch.float32, torch.float32, "none", torch.bfloat16))
        (rotated,) = kernel_warmup_shapes(fp32, mixed, path="pictures", rope_mode="native", grid=GRID)
        self.assertEqual((rotated.dtype, rotated.native_rope_dtype), (torch.bfloat16, torch.float32))
        report = rk.warmup("cpu", [fused, native, dense, keys, values, rotated])  # reference backend: validates
        self.assertEqual(report["calls"], 0)

    def test_attach_reports(self):
        # A software-decoder bridge restores with the reference kernels: nothing to warm at attach.
        software = self.bridge("software", profile=self.GEOMETRY, warmup=True).startup["warmup"]
        self.assertEqual(software["components"], {})
        # RAW: the dense / engine-RoPE kernel shapes (on CPU the reference backend compiles nothing).
        raw = self.bridge("raw", profile=self.GEOMETRY, representation="raw", warmup=True).startup["warmup"]
        kernels = raw["components"]["restore_kernels"]
        self.assertNotIn("error", kernels)
        self.assertEqual(kernels["detail"]["calls"], 0)
        self.assertGreaterEqual(raw["seconds"], kernels["seconds"])
        json.dumps(raw)
        # Automatic: CUDA bridges only (these CPU bridges stay cold); a bool or None.
        for representation in ("raw", "svd_hevc"):
            self.assertEqual(self.bridge(f"auto-{representation}", profile=self.GEOMETRY,
                                         representation=representation).startup["warmup"],
                             {"skipped": "warmup disabled"})
        with self.assertRaises(ConfigurationError):
            ReelCacheRuntimeConfig.from_dict({"warmup": "yes"})
        # A failing warm-up is reported, never raised: the bridge attaches and works cold.
        with mock.patch.object(rk, "warmup", side_effect=RuntimeError("boom")):
            failing = self.bridge("failing", profile=self.GEOMETRY, representation="raw", warmup=True)
        self.assertIn("boom", failing.startup["warmup"]["components"]["restore_kernels"]["error"])
        drive(failing, request(handle("s1"), blocks=2))

    def test_engine_rope_warmup_for_native_rope_restores(self):
        calls = []

        def tables(frames, height, width, *, start_frame, device):
            calls.append(("tables", frames, height, width, start_frame, str(device)))
            return rope_cos_sin(start_frame, frames)

        def rotate(key, cos, sin):
            calls.append(("rotate", tuple(key.shape), key.dtype))
            return rotate_key(key, cos, sin)

        transformer = SimpleNamespace(causal_rotary_cos_sin=tables, causal_rotate_key=rotate)
        spans = [("tables", FRAMES, GRID[0], GRID[1], float(start), "cpu")
                 for start in range(0, ROPE_WARM_FRAMES, FRAMES)]  # CR-17-2: the restore's span tables, every bridge
        raw = self.bridge("raw-rope", profile=self.GEOMETRY, representation="raw", warmup=True)
        raw.warm_engine_rope(transformer)
        rope = raw.startup["warmup"]["components"]["engine_rope"]
        self.assertNotIn("error", rope)
        # F6 cut 9: both classes of the Triton kernel's num_tokens % 16 specialization (one chunk; the next count of
        # the other class), rotated from two chunks' tables
        from reelcache.integrations.sglang.restore import engine_rope_token_classes
        chunk, other = engine_rope_token_classes(FRAMES * TPF)
        self.assertEqual((chunk, other, chunk % 16 == 0, other % 16 == 0), (FRAMES * TPF, 56, True, False))
        self.assertEqual(rope["detail"], {"tokens": [chunk, other], "dtype": "torch.bfloat16"})
        self.assertEqual(calls, spans + [("tables", 2 * FRAMES, GRID[0], GRID[1], 0, "cpu"),
                                         ("rotate", (1, chunk, HEADS, DIM), torch.bfloat16),
                                         ("rotate", (1, other, HEADS, DIM), torch.bfloat16)])
        # Fused-RoPE lineage restores without raw-tier ranges never call the engine's rotation (only its tables); with
        # a raw tier (F6 cut 9) its ranges do; a disabled warm-up warms nothing.
        fused = self.bridge("fused-rope", profile=self.GEOMETRY, warmup=True, raw_restore="off")
        fused.warm_engine_rope(transformer)
        self.assertNotIn("engine_rope", fused.startup["warmup"]["components"])
        self.assertEqual(fused.startup["warmup"]["components"]["rope_tables"]["detail"]["spans"], len(spans))
        tiered = self.bridge("fused-tiers", profile=self.GEOMETRY, warmup=True)  # raw_restore "prefer" (default)
        self.assertIsNotNone(tiered.archive.raw_tier)
        tiered.warm_engine_rope(transformer)
        self.assertEqual(tiered.startup["warmup"]["components"]["engine_rope"]["detail"]["tokens"], [chunk, other])
        native = self.bridge("native-rope", profile=self.GEOMETRY, warmup=True, rope_mode="native")
        native.warm_engine_rope(transformer)
        self.assertIn("engine_rope", native.startup["warmup"]["components"])
        cold = self.bridge("cold-rope", profile=self.GEOMETRY, representation="raw", warmup=False)
        cold.warm_engine_rope(transformer)
        self.assertEqual(cold.startup["warmup"], {"skipped": "warmup disabled"})
        self.assertEqual(len(cold.rope_tables), 0)
        # span tables of four bridges; the engine RoPE (tables + 2 rotations) of RAW, fused with tiers and native
        self.assertEqual(len(calls), 4 * len(spans) + 3 * 3)

    def test_engine_rope_token_classes_cover_both_specializations(self):
        from reelcache.integrations.sglang.restore import engine_rope_token_classes
        for chunk in (4680, 12480, 14040, 48, 8, 1560 * 3, 1560 * 8):
            first, second = engine_rope_token_classes(chunk)
            self.assertEqual(first, chunk)
            self.assertNotEqual(first % 16 == 0, second % 16 == 0)  # one count of each class
            self.assertTrue(chunk < second <= 2 * chunk)  # within two chunks' tables

    def test_rope_span_tables_built_at_attach_equal_the_engine_tables(self):
        """CR-17-2: the attach warm-up keeps the engine's own (cos, sin) of every chunk span below ROPE_WARM_FRAMES at
        the profile's default grid, computed exactly as a restore computes them (bitwise, same dtype and device);
        torch's global thread settings are unchanged."""
        def tables(frames, height, width, *, start_frame, device):
            return rope_cos_sin(start_frame, frames)

        threads = torch.get_num_threads()
        bridge = self.bridge("rope-tables", profile=self.GEOMETRY, warmup=True)
        bridge.warm_engine_rope(SimpleNamespace(causal_rotary_cos_sin=tables))
        self.assertEqual(torch.get_num_threads(), threads)
        starts = range(0, ROPE_WARM_FRAMES, FRAMES)
        self.assertEqual(len(bridge.rope_tables), len(starts))
        for start in starts:
            cached = bridge.rope_tables.get(RopeTableCache.key(float(start), FRAMES, GRID, "cpu"))
            expected = rope_cos_sin(float(start), FRAMES)
            for got, want in zip(cached, expected):
                self.assertTrue(got.is_inference())  # never written (built like SGLang's, under inference_mode)
                self.assertEqual((got.dtype, got.device, got.shape), (want.dtype, want.device, want.shape))
                self.assertTrue(torch.equal(got, want))
        self.assertEqual(bridge.rope_tables.stats()["warm_builds"], len(starts))

    def test_first_restore_after_attach_builds_no_rope_table(self):
        """CR-17-2: a session within the warmed positions restores with no table build; the installed KV equals a
        restore that builds its tables on demand (the tables are the same)."""
        calls = []

        def tables(frames, height, width, *, start_frame, device):
            calls.append((frames, start_frame))
            return rope_cos_sin(start_frame, frames)

        bridge = self.bridge("rope-first-restore", profile=self.GEOMETRY, warmup=True)
        bridge.warm_engine_rope(SimpleNamespace(causal_rotary_cos_sin=tables))
        warmed = len(calls)
        drive(bridge, request(handle("s1", pause_after_chunks=3), blocks=5))
        # stage=None: every request gets new caches, so the fork restores the archived checkpoint
        forked, _, warm_caches, _ = drive(bridge, request(handle("f1", "fork", from_session="s1", checkpoint=3,
                                                                 pause_after_chunks=4), blocks=5))
        self.assertEqual(forked["restore"]["mode"], "archive")
        self.assertEqual(forked["restore"]["rope_tables"]["builds"], 0)
        self.assertGreater(forked["restore"]["rope_tables"]["hits"], 0)
        self.assertEqual(len(calls), warmed)
        # The same checkpoint restored with an empty cache: the restore builds its tables (as before CR-17-2).
        bridge.rope_tables = RopeTableCache()
        cold, _, cold_caches, _ = drive(bridge, request(handle("f2", "fork", from_session="s1", checkpoint=3,
                                                               pause_after_chunks=4), blocks=5))
        self.assertEqual(cold["restore"]["rope_tables"]["hits"], 0)
        self.assertEqual(cold["restore"]["rope_tables"]["builds"], forked["restore"]["rope_tables"]["hits"])
        for stream in warm_caches:
            for warm_cache, cold_cache in zip(warm_caches[stream], cold_caches[stream]):
                self.assertTrue(torch.equal(warm_cache.k, cold_cache.k))
                self.assertTrue(torch.equal(warm_cache.v, cold_cache.v))

    def test_decoder_pool_record_reports_the_state_at_submit_and_the_counters_change(self):
        """Records only (pool A5, Step 19 interference cases): pool level and per worker, the gauges when a restore
        is submitted, the counters' change over it (queue waits and their parts behind a session creation or a
        warm-up), the mean wait per started job and the running maximum; no pool, no stats or failing stats give no
        record and never an error."""
        pool = RecordingPool(
            pool_stats(jobs=(10, 8), started=(9, 8), cold_jobs=(1, 0), created=(3, 3), waits=(0.5, 0.25),
                       creation=(0.1, 0.0), warm_up=(0.0, 0.0), maxima=(0.2, 0.1), queued=(1, 0), warm_ups=(0, 1),
                       running=1, started_warm_ups=2, warming=(False, True), warm_queued=(0, 2)),
            pool_stats(jobs=(100, 98), started=(100, 98), cold_jobs=(1, 0), created=(3, 4), waits=(2.0, 0.75),
                       creation=(0.1, 0.05), warm_up=(0.0, 0.3), maxima=(0.2, 0.35), started_warm_ups=4, moot=1))
        record = decoder_pool_record(decoder_pool_snapshot(pool), decoder_pool_snapshot(pool))
        self.assertTrue(record["available"])
        self.assertEqual(record["at_submit"], {"queued_jobs": 1, "warm_ups_in_flight": 1, "warm_up_running": 1})
        during = record["during_restore"]
        self.assertEqual((during["jobs"], during["jobs_started"], during["cold_jobs"], during["sessions_created"]),
                         (180, 181, 0, 1))
        self.assertAlmostEqual(during["queue_wait_seconds"], 2.0)
        self.assertAlmostEqual(during["creation_wait_seconds"], 0.05)
        self.assertAlmostEqual(during["warm_up_wait_seconds"], 0.3)
        self.assertEqual((during["warm_ups"], during["warm_ups_moot"]), (2, 1))  # A6r: a prewarm overlapped
        self.assertAlmostEqual(record["mean_queue_wait_seconds"], 2.0 / 181)
        self.assertEqual(record["queue_wait_max_seconds"], {"before": 0.2, "after": 0.35, "rose": True})
        self.assertEqual(record["after"], {"queued_jobs": 0, "warm_ups_in_flight": 0, "warm_up_running": 0})
        first, second = record["workers"]
        self.assertEqual(first["at_submit"], {"queued_jobs": 1, "warm_ups_in_flight": 0, "warming": False,
                                              "warm_queued": 0})
        self.assertEqual(second["at_submit"], {"queued_jobs": 0, "warm_ups_in_flight": 1, "warming": True,
                                               "warm_queued": 2})
        self.assertNotIn("warm_ups", first["during_restore"])  # warm-ups started / moot: pool level only
        self.assertEqual((first["during_restore"]["jobs_started"], second["during_restore"]["jobs_started"]),
                         (91, 90))
        self.assertFalse(first["queue_wait_max_seconds"]["rose"])
        self.assertTrue(second["queue_wait_max_seconds"]["rose"])
        self.assertAlmostEqual(second["during_restore"]["warm_up_wait_seconds"], 0.3)
        self.assertGreaterEqual(record["stats_seconds"], 0.0)
        self.assertEqual(decoder_pool_record(decoder_pool_snapshot(None), None), {"available": False})
        self.assertIsNone(decoder_pool_snapshot(SimpleNamespace()))
        self.assertIsNone(decoder_pool_snapshot(SimpleNamespace(stats=lambda: 1 / 0)))

    def test_archive_restore_records_the_decoder_pool_and_only_reads_it(self):
        """An archive restore reports its decoder-pool record: unavailable without a pool (CPU); with one, its stats
        are read twice (at submit, after the restore) and nothing else, and the restored KV is the same."""
        bridge = self.bridge("pool-records", profile=self.GEOMETRY)
        drive(bridge, request(handle("s1", pause_after_chunks=3), blocks=5))
        plain, _, plain_caches, _ = drive(bridge, request(handle("f1", "fork", from_session="s1", checkpoint=3,
                                                                 pause_after_chunks=4), blocks=5))
        self.assertEqual(plain["restore"]["mode"], "archive")
        self.assertEqual(plain["restore"]["decoder_pool"], {"available": False})
        pool = RecordingPool(
            pool_stats(jobs=(0, 0), started=(0, 0), cold_jobs=(0, 0), created=(1, 1), waits=(0.0, 0.0),
                       creation=(0.0, 0.0), warm_up=(0.0, 0.0), maxima=(0.0, 0.0), warm_ups=(1, 0), running=1,
                       warming=(True, False)),
            pool_stats(jobs=(6, 6), started=(6, 6), cold_jobs=(0, 0), created=(1, 1), waits=(0.1, 0.4),
                       creation=(0.0, 0.0), warm_up=(0.0, 0.3), maxima=(0.03, 0.31)))
        self.assertIsNone(bridge.archive.pool)
        bridge.archive.pool = pool
        try:
            recorded, _, recorded_caches, _ = drive(bridge, request(handle("f2", "fork", from_session="s1",
                                                                           checkpoint=3, pause_after_chunks=4),
                                                                    blocks=5))
        finally:
            bridge.archive.pool = None
        self.assertEqual(pool.calls, 2)
        record = recorded["restore"]["decoder_pool"]
        self.assertTrue(record["available"])
        self.assertEqual((record["at_submit"]["warm_ups_in_flight"], record["at_submit"]["warm_up_running"]), (1, 1))
        self.assertTrue(record["workers"][0]["at_submit"]["warming"])
        self.assertEqual(record["during_restore"]["jobs_started"], 12)
        self.assertAlmostEqual(record["during_restore"]["warm_up_wait_seconds"], 0.3)
        self.assertTrue(record["queue_wait_max_seconds"]["rose"])
        for stream in plain_caches:
            for plain_cache, recorded_cache in zip(plain_caches[stream], recorded_caches[stream]):
                self.assertTrue(torch.equal(plain_cache.k, recorded_cache.k))
                self.assertTrue(torch.equal(plain_cache.v, recorded_cache.v))

    def test_gc_pause_hook_is_installed_once_per_process_and_counts_per_generation(self):
        """Records only: one gc.callbacks hook per process whatever the number of bridges; it counts collections and
        their seconds per generation, costs about a microsecond per collection and never freezes the heap."""
        frozen = gc.get_freeze_count()
        install_gc_pauses()
        self.bridge("gc-hook-a", profile=self.GEOMETRY)
        self.bridge("gc-hook-b", profile=self.GEOMETRY)
        self.assertIs(install_gc_pauses(), GC_PAUSES)
        self.assertEqual(sum(callback is GC_PAUSES for callback in gc.callbacks), 1)
        before = GC_PAUSES.snapshot()
        gc.collect(0)
        gc.collect(2)
        record = gc_pause_record(before, GC_PAUSES.snapshot())
        self.assertTrue(record["available"])
        self.assertGreaterEqual(record["collections"][0], 1)
        self.assertGreaterEqual(record["collections"][2], 1)
        self.assertTrue(all(seconds >= 0.0 for seconds in record["seconds"]))
        self.assertAlmostEqual(record["total_seconds"], sum(record["seconds"]))
        self.assertEqual(gc.get_freeze_count(), frozen)
        probe, pairs = GcPauses(), 20000  # the hook's own cost: one start/stop pair per collection
        started = time.perf_counter()
        for _ in range(pairs):
            probe("start", {"generation": 0})
            probe("stop", {"generation": 0, "collected": 0, "uncollectable": 0})
        self.assertLess((time.perf_counter() - started) / pairs, 20e-6)
        self.assertEqual(probe.collections, [pairs, 0, 0])
        self.assertEqual(gc_pause_record(None, before), {"available": False})

    def test_archive_restore_records_the_collections_inside_its_window(self):
        """A gen-2 collection inside an archive restore shows in its record; the restored KV is unchanged."""
        bridge = self.bridge("gc-restore", profile=self.GEOMETRY)
        drive(bridge, request(handle("s1", pause_after_chunks=3), blocks=5))
        plain, _, plain_caches, _ = drive(bridge, request(handle("f1", "fork", from_session="s1", checkpoint=3,
                                                                 pause_after_chunks=4), blocks=5))
        self.assertTrue(plain["restore"]["gc"]["available"])
        self.assertEqual(len(plain["restore"]["gc"]["collections"]), 3)
        original = bridge.archive.restore

        def restore_with_a_collection(**kwargs):
            gc.collect(2)  # a full collection inside the restore window
            return original(**kwargs)

        bridge.archive.restore = restore_with_a_collection
        try:
            collected, _, collected_caches, _ = drive(bridge, request(handle("f2", "fork", from_session="s1",
                                                                             checkpoint=3, pause_after_chunks=4),
                                                                      blocks=5))
        finally:
            del bridge.archive.restore
        self.assertGreaterEqual(collected["restore"]["gc"]["collections"][2], 1)
        self.assertGreater(collected["restore"]["gc"]["seconds"][2], 0.0)
        for stream in plain_caches:
            for plain_cache, collected_cache in zip(plain_caches[stream], collected_caches[stream]):
                self.assertTrue(torch.equal(plain_cache.k, collected_cache.k))
                self.assertTrue(torch.equal(plain_cache.v, collected_cache.v))

    def test_capabilities_report_the_integrity_settings_and_quarantine(self):
        """CR-17-1: the serving defaults (hash-checked reads, no open-time audit, quarantine) and the quarantined
        sessions are visible through capabilities; RAW archives get audit_on_open / on_open_error from the config."""
        caps = self.bridge("integrity-caps", profile=self.GEOMETRY).control("capabilities", {})
        self.assertEqual(caps["integrity"], {"verify_reads": True, "audit_on_open": False,
                                             "on_open_error": "quarantine", "quarantined": {}})
        raw = self.bridge("integrity-raw", profile=self.GEOMETRY, representation="raw", audit_on_open=True)
        self.assertIs(raw.archive.runtime.audit_on_open, True)
        self.assertEqual(raw.control("capabilities", {})["integrity"]["audit_on_open"], True)

    def test_rope_table_cache_bound_and_keys(self):
        cache = RopeTableCache(max_bytes=3 * 2 * FRAMES * TPF * (DIM // 2) * 4)  # three spans of (cos, sin)
        for start in range(0, 10, FRAMES):
            cache.put(RopeTableCache.key(start, FRAMES, GRID, "cpu"), rope_cos_sin(start, FRAMES))
        self.assertEqual(len(cache), 3)
        self.assertEqual(cache.stats()["evictions"], 2)
        self.assertIsNone(cache.get(RopeTableCache.key(0, FRAMES, GRID, "cpu")))  # least recently used, dropped
        self.assertIsNotNone(cache.get(RopeTableCache.key(8.0, FRAMES, GRID, torch.device("cpu"))))  # int == float
        bound = cache.bind(lambda start, frames: rope_cos_sin(start, frames), grid=(GRID[0] + 1, GRID[1]),
                           device="cpu")
        bound(4, FRAMES)  # another grid: a build, then a hit
        bound(4, FRAMES)
        self.assertEqual(bound.report()["hits"], 1)
        self.assertEqual(bound.report()["builds"], 1)

    def test_session_staging_is_not_part_of_the_process_warmup(self):
        """Per-session staging (persisted sessions' packs/bases) is opt-in, never part of the default attach."""
        first = self.bridge("staging", profile=self.GEOMETRY)
        drive(first, request(handle("s1", pause_after_chunks=1)))
        first.close()
        archive = self.bridge("staging", profile=self.GEOMETRY).archive
        staged = mock.Mock(return_value={"nodes": 1})
        with mock.patch.object(archive.restorer, "prewarm_lineages", staged, create=True):
            self.assertNotIn("sessions", archive.warmup(self.GEOMETRY)["components"])
            staged.assert_not_called()
            self.assertIn("sessions", archive.warmup(self.GEOMETRY, sessions=True)["components"])
            staged.assert_called_once_with()

    def test_worker_heap_is_frozen_once_after_attach(self):
        """``gc.freeze()`` after model load and attach (Step 5b stall diagnosis: ~0.5 s full collections in the
        workers); once per process, run in a child interpreter so this test process keeps its own GC state."""
        code = (
            "import gc\n"
            "from reelcache.integrations.sglang import freeze_long_lived_objects as freeze\n"
            "first = freeze(); again = freeze()\n"
            "assert first['frozen_objects'] == gc.get_freeze_count() > 0, first\n"
            "assert again['repeat'] and again['frozen_objects'] == first['frozen_objects'], again\n"
            "assert set(first) == {'collected', 'collect_seconds', 'frozen_objects', 'seconds'}, first\n"
            "print('ok')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("ok", result.stdout)

    def test_synthetic_restore_in_a_scratch_store(self):
        """The opt-in end-to-end diagnostic: a synthetic chunk archived and restored in a scratch store."""
        bridge = self.bridge("warm", profile=self.GEOMETRY, warmup=False)
        report = bridge.archive.warmup(self.GEOMETRY, synthetic=True)
        synthetic = report["components"]["synthetic_restore"]
        self.assertNotIn("error", synthetic)
        detail = synthetic["detail"]
        self.assertEqual((detail["tokens"], detail["grid"], detail["layers"], detail["rope"]),
                         (FRAMES * TPF, list(GRID), min(4, LAYERS), "fused"))
        self.assertGreater(detail["stored_bytes"], 0)
        self.assertEqual(set(detail["seconds"]), {"sources", "archive", "restore_first", "restore_shifted",
                                                 "restore_second"})
        self.assertNotIn("archive", report["components"])  # the pipeline prewarm is opt-in (and CUDA only)
        self.assertNotIn("sessions", report["components"])  # no persisted session (and no packed restorer)
        self.assertGreaterEqual(report["seconds"], synthetic["seconds"])
        json.dumps(report)
        # Nothing of the warm-up remains: no lineage, no scratch store, and the shared online learner saw nothing.
        self.assertEqual(bridge.archive.runtime.list_lineages(), {})
        self.assertEqual([name for name in os.listdir(self.workspace / "warm") if name.startswith(".warmup")], [])
        self.assertEqual(bridge.archive.registry.sequence, 0)
        # Sessions afterwards are unaffected: identical ids with and without the warm-up.
        cold = self.bridge("cold", profile=self.GEOMETRY, warmup=False)
        self.assertEqual(cold.startup["warmup"], {"skipped": "warmup disabled"})
        for each in (bridge, cold):
            drive(each, request(handle("s1"), blocks=2))
            each.drain_all()
        self.assertEqual(bridge.archive.runtime.list_lineages(), cold.archive.runtime.list_lineages())
        # Without geometry defaults the synthetic chunk is skipped; a failing one is reported, never raised.
        self.assertIn("skipped", bridge.archive.warmup(PROFILE, synthetic=True)["components"]["synthetic_restore"])
        with mock.patch.object(LineageSessionArchive, "_synthetic_restore", side_effect=RuntimeError("boom")):
            failed = bridge.archive.warmup(self.GEOMETRY, synthetic=True)
        self.assertIn("boom", failed["components"]["synthetic_restore"]["error"])

    def test_warmup_pipeline_never_takes_the_session_fitter(self):
        """A pipeline's fitter thread adopts its GPU fitter (``GPUFitter.adopt_current_thread``): a warm-up
        pipeline sharing the session pipeline's fitter would leave it owned by the closed warm-up thread and every
        later commit would fail (seen on the GPU). The warm-up pipeline gets a private fitter."""
        from reelcache.integrations.sglang import lineage_archive

        archive = self.bridge("fitter", profile=self.GEOMETRY, warmup=False).archive
        self.assertIsNone(archive._scratch_fitter())  # CPU fitting: no GPU fitter at all
        captured, session_fitter, private = {}, object(), object()

        def construct(*args, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop after construction")

        with mock.patch.object(lineage_archive, "ArchivePipeline", side_effect=construct), \
                mock.patch.object(archive, "fitter", session_fitter), \
                mock.patch.object(type(archive), "_scratch_fitter", return_value=private):
            report = archive.warmup(self.GEOMETRY, synthetic=True)
        self.assertIn("stop after construction", report["components"]["synthetic_restore"]["error"])
        self.assertIs(captured["fitter"], private)
        self.assertIsNot(captured["registry"], archive.registry)


class AsyncPublicationTests(BridgeTestCase):
    """Publication behind generation and across requests (report section 3.4, ``archive_native``)."""

    representation = "svd_hevc"
    assertLatentsEqual = LineageSessionTests.assertLatentsEqual

    def test_requests_return_before_publication(self):
        bridge = self.bridge()
        drive(bridge, request(handle("warm"), blocks=2))  # first-use costs of the fitter and encoders
        bridge.checkpoint("warm")
        started = time.perf_counter()
        drive(bridge, request(None), stage=Stage())  # archival off: the same generation without a session
        off = time.perf_counter() - started
        publication = 0.3
        publication_hook(bridge, seconds=publication)  # every publication now takes >= 0.3 s, in order
        started = time.perf_counter()
        summary, _, _, _ = drive(bridge, request(handle("s1")))
        on = time.perf_counter() - started
        self.assertEqual((summary["status"], summary["generated_chunks"]), ("completed", 4))
        self.assertFalse(summary["durable"] or summary["published"])
        self.assertTrue(summary["pending"])
        self.assertEqual(summary["archive"]["publish_wait_seconds"], 0.0)
        # The request did not wait for the >= 1.2 s of ordered publication: its latency stays near archival-off.
        self.assertLess(on, 2 * publication, (on, off))
        self.assertLess(on - off, publication, (on, off))
        checkpoint = bridge.checkpoint("s1")
        self.assertTrue(checkpoint["published"])
        self.assertGreater(checkpoint["wait_seconds"], 0.0)
        nodes = bridge.archive.path(checkpoint["head"])
        self.assertEqual([node.chunk_index for node in nodes[1:]], [0, 1, 2, 3])
        state = bridge.controller.get("s1")
        self.assertEqual([item["event"] for item in state.history].count("published"), 4)
        # An idle, fully published session's pipeline state is released when another session runs.
        self.assertTrue(bridge.archive.live("s1"))
        drive(bridge, request(handle("s2"), blocks=1))
        self.assertFalse(bridge.archive.live("s1"))
        self.assertLifetimesRetired(bridge)

    def test_bounded_queue_applies_and_reports_backpressure(self):
        # The C3 bounds (freeze D: raw_tier=False; with the raw tier submit waits only as a last resort).
        bridge = self.bridge(max_source_chunks=1, max_unpublished=1, raw_tier=False)
        publication_hook(bridge, seconds=0.2)
        summary, _, _, _ = drive(bridge, request(handle("s1")))
        backpressure = summary["archive"]["backpressure"]
        self.assertGreaterEqual(backpressure["waits"], 3)  # chunks 1..3 each waited for the previous publication
        self.assertGreater(backpressure["seconds"], 0.3)
        self.assertTrue(all(item["queue_depth"] <= 1 for item in summary["commits"]))
        self.assertEqual(sum(item["backpressure_waits"] for item in summary["commits"]), backpressure["waits"])
        state = bridge.controller.get("s1")
        self.assertEqual(state.backpressure["waits"], backpressure["waits"])
        self.assertEqual(len(bridge.archive.path(bridge.checkpoint("s1")["head"])), 5)
        # With submit_timeout a full queue raises CapacityError instead of waiting; the session keeps its
        # submitted chunks and resumes from them.
        strict = self.bridge("strict", max_source_chunks=1, max_unpublished=1, submit_timeout=0.05, raw_tier=False)
        publication_hook(strict, seconds=0.5)
        with self.assertRaises(CapacityError):
            drive(strict, request(handle("t1")))
        state = strict.controller.get("t1")
        self.assertEqual((state.status, state.generated_chunks), ("failed", 1))
        strict.archive.pipeline.hooks = None
        strict.archive.pipeline.submit_timeout = None  # wait again (a 50 ms bound is below one CPU encode)
        resumed, _, _, _ = drive(strict, request(handle("t1", "resume")))
        self.assertEqual((resumed["resumed_from_chunks"], resumed["restore"]["mode"]), (1, "archive"))
        self.assertEqual(resumed["generated_chunks"], 4)

    def test_pause_checkpoint_and_close_wait_for_the_published_frontier(self):
        bridge = self.bridge()
        publication_hook(bridge, seconds=0.2)
        paused, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=2)))
        self.assertTrue(paused["durable"] and paused["published"])
        self.assertEqual(paused["pending"], [])
        self.assertGreater(paused["archive"]["publish_wait_seconds"], 0.0)
        self.assertEqual([node.kind for node in bridge.archive.path(paused["head"])], ["root", "chunk", "chunk"])
        self.assertTrue(all(item["published"] and item.get("matrices") for item in paused["commits"]))
        # The handle overrides the policy: this pause returns before publication.
        quick, _, _, _ = drive(bridge, request(handle("s2", pause_after_chunks=2, durable=False)))
        self.assertFalse(quick["durable"])
        self.assertTrue(quick["pending"])
        described = bridge.controller.pause("s2")  # explicit pause of an idle session: waits for publication
        self.assertTrue(described["published"])
        self.assertEqual((described["generated_chunks"], described["pending"]), (2, []))
        # close_session publishes pending commits before dropping the session.
        drive(bridge, request(handle("s3")))
        bridge.controller.close_session("s3")
        self.assertEqual(len(bridge.archive.path(bridge.archive.head("s3"))), 5)
        # durability="always": every request publishes before it returns (the synchronous behavior).
        always = self.bridge("always", durability="always")
        publication_hook(always, seconds=0.05)
        summary, _, _, _ = drive(always, request(handle("s4")))
        self.assertTrue(summary["durable"] and summary["published"])
        self.assertEqual(len(always.archive.path(summary["head"])), 5)

    def test_archival_failure_surfaces_on_the_next_request_and_on_pause(self):
        reference = self.bridge("reference")
        _, expected_latents, _, _ = drive(reference, request(handle("ref")))
        bridge = self.bridge()
        gate = threading.Event()
        publication_hook(bridge, stage="encode", gate=gate, fail="c000001")
        first, _, _, _ = drive(bridge, request(handle("s1"), blocks=2))
        self.assertEqual((first["status"], first["published"]), ("completed", False))  # nothing surfaced yet
        gate.set()
        state = bridge.controller.get("s1")
        wait_until(lambda: bridge.controller.failure(state) is not None)
        with self.assertRaises(StateError) as caught:
            drive(bridge, request(handle("s1", "resume")))
        self.assertIn("archival failed", str(caught.exception))
        self.assertEqual((state.status, state.generated_chunks, state.pending), ("failed", 1, []))
        self.assertEqual([node.kind for node in bridge.archive.path(state.head_ref)], ["root", "chunk"])
        self.assertIsNone(bridge.resident)
        # Surfaced once: the session continues from its last published checkpoint.
        bridge.archive.pipeline.hooks = None
        resumed, latents, _, _ = drive(bridge, request(handle("s1", "resume")))
        self.assertEqual((resumed["resumed_from_chunks"], resumed["generated_chunks"]), (1, 4))
        self.assertLatentsEqual(latents, expected_latents)
        # A durable pause waits for its commits and raises their failure.
        publication_hook(bridge, stage="encode", fail="c000001")
        with self.assertRaises(StateError):
            drive(bridge, request(handle("s2", pause_after_chunks=2)))
        state = bridge.controller.get("s2")
        self.assertEqual((state.status, state.generated_chunks), ("failed", 1))
        self.assertEqual(len(bridge.archive.path(state.head_ref)), 2)
        # ... and so does an explicit checkpoint.
        drive(bridge, request(handle("s3"), blocks=2))
        state = bridge.controller.get("s3")
        wait_until(lambda: bridge.controller.failure(state) is not None)
        with self.assertRaises(StateError):
            bridge.checkpoint("s3")
        self.assertEqual(state.generated_chunks, 1)

    def test_resume_waits_only_when_it_restores_unpublished_nodes(self):
        reference = self.bridge("reference")
        _, expected_latents, expected_caches, _ = drive(reference, request(handle("ref")))
        bridge = self.bridge()
        stage = Stage()
        gate = threading.Event()
        publication_hook(bridge, gate=gate)  # nothing publishes until the gate opens
        first, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=2, durable=False)), stage=stage)
        self.assertEqual(len(first["pending"]), 2)
        # Resident native KV: the resume continues at once although nothing is published yet.
        resumed, latents, caches, _ = drive(bridge, request(handle("s1", "resume")), stage=stage)
        self.assertEqual((resumed["restore"]["mode"], resumed["restore"]["publish_wait_seconds"]), ("resident", 0.0))
        self.assertEqual(len(resumed["restore"]["pending_at_resume"]), 2)
        self.assertTrue(torch.equal(latents, expected_latents))
        BridgeTestCase.assertCachesEqual(self, caches, expected_caches)
        gate.set()
        # A restore from the archive needs the session's published frontier: it waits for it.
        other = self.bridge("other")
        publication_hook(other, seconds=0.3)
        drive(other, request(handle("s2", pause_after_chunks=2, durable=False)))
        restored, latents, _, _ = drive(other, request(handle("s2", "resume")))  # new caches: not resident
        self.assertEqual(restored["restore"]["mode"], "archive")
        self.assertGreater(restored["restore"]["publish_wait_seconds"], 0.1)
        self.assertLatentsEqual(latents, expected_latents)


class LineageRecacheTests(RecacheEpochChecks, TimelineConditioningChecks, BridgeTestCase):
    representation = "svd_hevc"

    def assertCachesEqual(self, left, right):
        LineageSessionTests.assertCachesEqual(self, left, right)

    def assertLatentsEqual(self, left, right):
        LineageSessionTests.assertLatentsEqual(self, left, right)



# ------------------------------------------------------------------------------------------ Step 14 session controls
def control_error(test, code, op, args, bridge):
    with test.assertRaises(SessionControlError) as caught:
        bridge.control(op, args)
    test.assertEqual(caught.exception.code, code, str(caught.exception))
    test.assertEqual(caught.exception.status, ERROR_STATUS[code])
    return caught.exception


def stored(bridge, object_ids):
    """The subset of ``object_ids`` still indexed in the bridge's store."""
    present = set()
    for object_id in object_ids:
        try:
            bridge.archive.store.object_size(object_id)
        except IntegrityError:
            continue
        present.add(object_id)
    return present


#: The toy request driver run in a child interpreter that is SIGKILLed mid-archival (crash tests).
CRASH_CHILD = """
import json, os, sys, threading, time
from unittest import mock
import tests.test_sglang_bridge as t
from reelcache.integrations.sglang import CausalWanKVBridge, ReelCacheRuntimeConfig

spec = json.loads(sys.argv[1])
values = {"store_root": spec["root"], "representation": spec["representation"], "fit_device": "cpu",
          "decoder": "software", "energy_key": 0.98, "energy_value": 0.95, "x265_threads": 2, "encode_workers": 2}
if spec["representation"] == "svd_hevc":
    values["raw_restore"] = "off"  # lossy restores before and after the restart (freeze D raw tier: bitwise before)
bridge = CausalWanKVBridge(ReelCacheRuntimeConfig.from_dict(values), t.PROFILE, device="cpu",
                           model_path="toy/causal-wan")
streams = ("cond", "uncond")
t.drive(bridge, t.request(t.handle("s1", pause_after_chunks=2), streams=streams))
_, latents, caches, _ = t.drive(bridge, t.request(t.handle("pre", "fork", from_session="s1", checkpoint=2,
                                                           pause_after_chunks=3), streams=streams))
result = {"pre_latents": latents[:, :, 2 * t.FRAMES:3 * t.FRAMES].tolist(),
          "pre_key": caches["cond"][0].k[:, :caches["cond"][0].end].float().tolist(),
          "s1_head": bridge.archive.head("s1")}
with open(spec["result"], "w") as handle:
    json.dump(result, handle)

def reached():
    with open(spec["marker"], "w") as handle:
        handle.write("gate")
    time.sleep(3600)

if spec["representation"] == "raw":
    from reelcache.integrations.sglang import archive as raw_archive
    real, calls = raw_archive.encode_tensor, []

    def blocking(tensor, context, codec):
        calls.append(1)
        if len(calls) == 2 * t.LAYERS + 1:  # chunk 2: the cond stream is written, the uncond stream blocks
            reached()
        return real(tensor, context, codec)

    with mock.patch.object(raw_archive, "encode_tensor", blocking):
        t.drive(bridge, t.request(t.handle("s1", "resume"), streams=streams))
else:
    def hook(name, node_id):
        if name == "publish" and node_id == "c000002":
            reached()

    bridge.archive.pipeline.hooks = hook
    t.drive(bridge, t.request(t.handle("s1", "resume", durable=False), streams=streams))
    time.sleep(3600)
"""


class SessionControlChecks:
    """Step 14 worker-level session controls (``bridge.control``), shared by the RAW reference and the lineage
    backend."""

    def test_list_describe_checkpoints_request_and_typed_errors(self):
        bridge = self.bridge()
        summary, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=2)))
        listed = bridge.control("list")
        self.assertEqual([item["session_id"] for item in listed["sessions"]], ["s1"])
        entry = listed["sessions"][0]
        self.assertEqual((entry["open"], entry["status"], entry["generated_chunks"], entry["published"]),
                         (True, "paused", 2, True))
        described = bridge.control("describe", {"session_id": "s1", "history": 3})
        self.assertEqual(len(described["history"]), 3)
        self.assertEqual([item["request_id"] for item in described["requests"]], [summary["request_id"]])
        checkpoints = bridge.control("checkpoints", {"session_id": "s1"})
        self.assertEqual([item["chunks"] for item in checkpoints["checkpoints"]], [0, 1, 2])
        report = bridge.control("request", {"request_id": summary["request_id"]})["summary"]
        self.assertEqual(sorted(report["chunk_digests"]), ["0", "1"])
        self.assertEqual([(item["chunk"], item["stream"]) for item in report["clean_forwards"]],
                         [(0, "cond"), (1, "cond")])
        json.dumps(bridge.control("describe", {"session_id": "s1"}))  # plain JSON throughout
        capabilities = bridge.control("capabilities")
        self.assertEqual((capabilities["boundary"], capabilities["prompt_change"]["supported"]), ("chunk", False))
        self.assertIn("pause", capabilities["inflight_ops"])
        for op, args, code in (
                ("describe", {"session_id": "nope"}, "unknown_session"),
                ("frobnicate", {}, "invalid_argument"),
                ("describe", {"session_id": "a/b"}, "invalid_argument"),
                ("describe", {"session_id": "s1", "path": "/tmp/x"}, "invalid_argument"),
                ("fork", {"session_id": "s1", "target": "../x"}, "invalid_argument"),
                ("checkpoint", {"session_id": "s1", "name": "/etc/passwd"}, "invalid_argument"),
                ("rewind", {"session_id": "s1", "checkpoint": "missing"}, "unknown_checkpoint"),
                ("rewind", {"session_id": "s1", "checkpoint": 7}, "unknown_checkpoint"),
                ("rewind", {"session_id": "s1", "checkpoint": "/abs/path"}, "invalid_argument"),
                ("request", {"request_id": "nope"}, "unknown_request"),
                ("close", {"session_id": "s1", "mode": "explode"}, "invalid_argument")):
            control_error(self, code, op, args, bridge)
        # Persisted sessions are listed and described from their head without reopening them.
        bridge.control("close", {"session_id": "s1"})
        bridge.close()
        restarted = self.bridge()
        entry = restarted.control("list")["sessions"][0]
        self.assertEqual((entry["open"], entry["status"], entry["generated_chunks"]), (False, "closed", 2))
        self.assertIsNone(restarted.controller.state("s1"))
        self.assertEqual([item["chunks"] for item in restarted.control(
            "checkpoints", {"session_id": "s1"})["checkpoints"]], [0, 1, 2])
        self.assertIsNone(restarted.controller.state("s1"))
        opened = restarted.control("open", {"session_id": "s1"})
        self.assertEqual((opened["was_open"], opened["status"], opened["generated_chunks"]), (False, "paused", 2))

    def test_named_checkpoints_fork_rewind_delete_and_gc_reachability(self):
        reference = self.bridge("reference")
        _, expected_latents, expected_caches, _ = drive(reference, request(handle("ref")))
        bridge = self.bridge()
        empty = (bridge.archive.store.list_refs(""), bridge.archive.store.stats()["object_count"])
        drive(bridge, request(handle("s1", pause_after_chunks=2)))
        c2 = bridge.control("checkpoint", {"session_id": "s1", "name": "c2"})
        drive(bridge, request(handle("s1", "resume")))
        c4 = bridge.control("checkpoint", {"session_id": "s1", "name": "c4"})
        self.assertEqual((c2["generated_chunks"], c4["generated_chunks"], c4["replaced"]), (2, 4, None))
        control_error(self, "illegal_transition", "checkpoint", {"session_id": "s1", "name": "c2"}, bridge)
        self.assertIsNone(bridge.control("checkpoint", {"session_id": "s1", "name": "c4"})["replaced"])
        named = bridge.control("checkpoints", {"session_id": "s1"})["named"]
        self.assertEqual({name: item["chunks"] for name, item in named.items()}, {"c2": 2, "c4": 4})
        forked = bridge.control("fork", {"session_id": "s1", "target": "f", "checkpoint": "c2"})
        self.assertEqual((forked["generated_chunks"], forked["node"], forked["status"]), (2, c2["head"], "paused"))
        control_error(self, "illegal_transition", "fork", {"session_id": "s1", "target": "f"}, bridge)
        rewound = bridge.control("rewind", {"session_id": "s1", "checkpoint": "c2"})
        self.assertEqual((rewound["node"], rewound["previous_head"], rewound["generated_chunks"]),
                         (c2["head"], c4["head"], 2))
        # c4 is off the head path now: a rewind cannot reach it (fork from it instead).
        control_error(self, "illegal_transition", "rewind", {"session_id": "s1", "checkpoint": "c4"}, bridge)
        tail = {node.ref for node in bridge.archive.path(c4["head"])[3:]}  # chunks 3 and 4: only c4 reaches them
        self.assertEqual(len(tail), 2)
        bridge.control("gc")
        self.assertEqual(stored(bridge, tail), tail)  # pinned by the named checkpoint
        deleted = bridge.control("delete", {"session_id": "s1", "checkpoint": "c4"})
        self.assertGreater(deleted["gc"]["objects_removed"], 0)
        self.assertEqual(stored(bridge, tail), set())  # unreachable: reclaimed
        shared = {node.ref for node in bridge.archive.path(c2["head"])}
        deleted = bridge.control("delete", {"session_id": "s1"})
        self.assertEqual((deleted["deleted"], deleted["checkpoints_deleted"]), (True, {"c2": c2["head"]}))
        self.assertEqual(stored(bridge, shared), shared)  # still reached by the fork: never collected
        self.assertEqual([item["session_id"] for item in bridge.control("list")["sessions"]], ["f"])
        control_error(self, "unknown_session", "describe", {"session_id": "s1"}, bridge)
        _, latents, caches, _ = drive(bridge, request(handle("f", "resume")))
        self.assertLatentsEqual(latents, expected_latents)
        self.assertCachesEqual(caches, expected_caches)
        bridge.control("delete", {"session_id": "f"})
        # Every session object is reclaimed: the store holds what it held before the first session.
        self.assertEqual((bridge.archive.store.list_refs(""), bridge.archive.store.stats()["object_count"]), empty)

    def test_inflight_pause_stops_at_the_next_boundary_and_acknowledges_durably(self):
        reference = self.bridge("reference")
        _, expected_latents, expected_caches, _ = drive(reference, request(handle("ref")))
        bridge = self.bridge()
        seen = []

        def poll():  # SGLang's scheduler polls for controls at every chunk boundary of a session request
            if seen:
                return
            seen.append(bridge.inflight_control("pause", {"session_id": "s1"}))
            seen.append(bridge.inflight_control("describe", {"session_id": "s1", "history": 0}))
            seen.append(bridge.inflight_control("checkpoint", {"session_id": "s1"}))  # would drain: queued
            seen.append(bridge.inflight_control("pause", {"session_id": "other"}))  # not running: queued
            with self.assertRaises(SessionControlError) as caught:
                bridge.control("close", {"session_id": "s1"})
            seen.append(caught.exception.code)

        bridge.set_boundary_hook(poll)
        summary, _, _, _ = drive(bridge, request(handle("s1")))
        self.assertEqual((seen[0]["status"], seen[0]["deferred"]), ("pause_requested", True))
        self.assertEqual(seen[1]["status"], "running")
        self.assertEqual(seen[2:], [None, None, "busy"])
        self.assertEqual((summary["paused"], summary["generated_chunks"]), (True, 1))
        self.assertGreaterEqual(summary["boundary_hook_seconds"], 0.0)
        acknowledged = bridge.control("pause", {"session_id": "s1"})
        self.assertEqual((acknowledged["deferred"], acknowledged["published"], acknowledged["generated_chunks"]),
                         (False, True, 1))
        self.assertEqual(sum(node.kind == "chunk" for node in bridge.archive.path(acknowledged["head"])), 1)
        bridge.set_boundary_hook(None)
        _, latents, caches, _ = drive(bridge, request(handle("s1", "resume")))
        self.assertLatentsEqual(latents, expected_latents)
        self.assertCachesEqual(caches, expected_caches)
        with self.assertRaises(ConfigurationError):
            bridge.set_boundary_hook("not callable")

    def test_controls_on_a_running_session_are_busy(self):
        bridge = self.bridge()
        drive(bridge, request(handle("s1", pause_after_chunks=1)))
        run = bridge.begin_request(request(handle("s1", "resume")))
        try:
            for op, args in (("close", {"session_id": "s1"}), ("rewind", {"session_id": "s1", "checkpoint": 0}),
                             ("checkpoint", {"session_id": "s1"}), ("fork", {"session_id": "s1", "target": "f"}),
                             ("delete", {"session_id": "s1"}), ("checkpoints", {"session_id": "s1"}),
                             ("gc", {})):
                control_error(self, "busy", op, args, bridge)
            self.assertTrue(bridge.control("pause", {"session_id": "s1"})["deferred"])
            self.assertEqual(bridge.control("describe", {"session_id": "s1"})["status"], "running")
        finally:
            run.finish(error=None)
        self.assertEqual(bridge.control("close", {"session_id": "s1"})["status"], "closed")

    def test_prompt_change_without_native_support_is_rejected_explicitly(self):
        bridge = self.bridge()
        drive(bridge, request(handle("s1", pause_after_chunks=2), prompt="A"))
        for action, extra in (("resume", {}), ("rewind", {"checkpoint": 1})):
            with self.assertRaises(UnsupportedError) as caught:
                drive(bridge, request(handle("s1", action, **extra), prompt="B"))
            self.assertIn("different conditioning", str(caught.exception))
            self.assertIn("has no native prompt change", str(caught.exception))
        with self.assertRaises(UnsupportedError):
            drive(bridge, request(handle("f", "fork", from_session="s1"), prompt="B"))
        self.assertEqual(bridge.controller.get("s1").generated_chunks, 2)  # nothing moved
        self.assertEqual(bridge.archive.head("f"), None)
        self.assertFalse(bridge.control("capabilities")["prompt_change"]["supported"])
        self.assertEqual(prompt_change_capability(LONGLIVE2)["mode"], "multi_shot")
        switching = dataclasses.replace(PROFILE, recache="kv_recache_on_prompt_switch")
        self.assertEqual(prompt_change_capability(switching)["request_fields"], ["prompt_switches"])

    def test_requests_that_restore_history_decode_only_their_new_chunks(self):
        bridge = self.bridge(decode_context_latents=3)
        first, _, _, _ = drive(bridge, request(handle("s1", pause_after_chunks=2)))
        self.assertEqual(first["decode"], {"output": "full"})  # nothing restored: the clip is new
        resumed, _, _, _ = drive(bridge, request(handle("s1", "resume", pause_after_chunks=3)))
        self.assertEqual(resumed["decode"], {"output": "new", "start_latent": 1, "context_latents": 3,
                                             "first_new_latent": 4, "exact": False})
        whole, _, _, _ = drive(bridge, request(handle("s1", "resume", decode_context_latents=64)))
        self.assertEqual(whole["decode"], {"output": "new", "start_latent": 0, "context_latents": 6,
                                           "first_new_latent": 6, "exact": True})
        full, _, _, _ = drive(bridge, request(handle("f", "fork", from_session="s1", checkpoint=1, output="full")))
        self.assertEqual(full["decode"], {"output": "full"})

    def test_observed_plain_requests_time_clean_forwards_without_archiving(self):
        reference = self.bridge("reference")
        _, expected, _, _ = drive(reference, request(None), stage=Stage())  # generation without the bridge
        bridge = self.bridge(observe_plain_requests=True)
        self.assertTrue(bridge.observes_plain_requests)
        req = request(None)
        req.dispatched_at = time.perf_counter()
        summary, latents, _, _ = drive(bridge, req)
        self.assertTrue(summary["observer"])
        self.assertEqual((summary["chunks"], summary["status"]), (4, "completed"))
        self.assertTrue(torch.equal(latents, expected))  # observation changes nothing
        self.assertEqual(bridge.archive.sessions(), {})
        self.assertFalse(bridge.active_request)
        report = bridge.control("request", {"request_id": req.request_id})["summary"]
        self.assertEqual([item["chunk"] for item in report["clean_forwards"]], [0, 1, 2, 3])
        self.assertTrue(all(item["seconds"] >= 0.0 for item in report["clean_forwards"]))
        timing = report["timing"]
        self.assertGreater(timing["time_to_first_new_chunk_seconds"], timing["first_new_chunk_from_begin_seconds"])
        self.assertGreaterEqual(timing["pre_denoise_seconds"], 0.0)


class SessionProviderTests(unittest.TestCase):
    """Pipelines without causal denoising stages (e.g. LongCat's conditioning cache) attach the session provider
    they declare; it serves the same worker interface (control, inflight_control, set_boundary_hook)."""

    def test_pipelines_without_causal_stages_use_their_declared_provider(self):
        import reelcache.integrations.sglang as package

        made = []

        def provider(*, config, pipeline, server_args):
            made.append((config, server_args))
            return SimpleNamespace(startup={})

        with mock.patch.object(package, "freeze_long_lived_objects", return_value={"frozen": 1}) as freeze:
            backend = package.create_sglang_backend(config={"store_root": "x"}, server_args="args",
                                                    pipeline=SimpleNamespace(stages=[],
                                                                             reelcache_session_provider=provider))
            self.assertEqual(made, [({"store_root": "x"}, "args")])
            self.assertEqual(backend.startup["gc_freeze"], {"frozen": 1})
            named = package.create_sglang_backend(
                config={}, server_args=None,
                pipeline=SimpleNamespace(stages=[], reelcache_session_provider="types:SimpleNamespace"))
            self.assertEqual((named.config, named.server_args), ({}, None))
            self.assertEqual(freeze.call_count, 2)
            with self.assertRaises(ConfigurationError):
                package.create_sglang_backend(config={}, server_args=None,
                                              pipeline=SimpleNamespace(stages=[], reelcache_session_provider="bad"))

    def test_sglang_codec_flag_reaches_a_provider_only_when_set(self):
        # SGLang's ReelCacheArgs.backend_config() always adds its --reelcache-hevc-inter-prediction flag to the
        # --reelcache-config dict; a provider (strict about its keys) gets it only when the flag is set.
        import reelcache.integrations.sglang as package

        made = []

        def provider(*, config, pipeline, server_args):
            made.append(dict(config))
            return SimpleNamespace(startup={})

        pipeline = SimpleNamespace(stages=[], reelcache_session_provider=provider)
        with mock.patch.object(package, "freeze_long_lived_objects", return_value={}):
            for flag in (False, True):
                package.create_sglang_backend(config={"store_root": "x", "hevc_inter_prediction": flag},
                                              server_args=None, pipeline=pipeline)
        self.assertEqual(made, [{"store_root": "x"}, {"store_root": "x", "hevc_inter_prediction": True}])


class RawControlTests(SessionControlChecks, BridgeTestCase):
    representation = "raw"

    def test_cfg_group_commit_failure_in_one_stream_publishes_nothing(self):
        streams = ("cond", "uncond")
        reference = self.bridge("reference")
        _, expected_latents, expected_caches, _ = drive(reference, request(handle("ref"), streams=streams))
        bridge = self.bridge()
        from reelcache.integrations.sglang import archive as raw_archive

        real, calls = raw_archive.encode_tensor, []
        per_stream = 2 * LAYERS

        def failing(tensor, context, codec):
            calls.append(1)
            if len(calls) == 2 * per_stream + per_stream + 1:  # chunk 1: cond written, first uncond matrix fails
                raise RuntimeError("injected uncond-stream encode failure")
            return real(tensor, context, codec)

        with mock.patch.object(raw_archive, "encode_tensor", failing):
            with self.assertRaises(RuntimeError):
                drive(bridge, request(handle("s1"), streams=streams))
        path = bridge.archive.path(bridge.archive.head("s1"))
        self.assertEqual([node.kind for node in path], ["root", "chunk"])  # the previous checkpoint
        self.assertEqual(path[-1].streams, streams)
        self.assertEqual(bridge.archive.store.collect_garbage()["objects_removed"], 0)  # no orphan cond packets
        self.assertEqual(bridge.control("describe", {"session_id": "s1"})["status"], "failed")
        _, latents, caches, _ = drive(bridge, request(handle("s1", "resume"), streams=streams))
        self.assertLatentsEqual(latents, expected_latents)
        self.assertCachesEqual(caches, expected_caches)

    def test_sigkill_mid_commit_never_exposes_a_half_written_node(self):
        streams = ("cond", "uncond")
        reference = self.bridge("reference")
        _, expected_latents, expected_caches, _ = drive(reference, request(handle("ref"), streams=streams))
        report = run_crash_child(self, "raw")
        bridge = self.bridge("crashed")
        listed = {item["session_id"]: item for item in bridge.control("list")["sessions"]}
        self.assertEqual(listed["s1"]["generated_chunks"], 2)  # chunk 2 was mid-commit: never exposed
        self.assertEqual(listed["s1"]["head"], report["s1_head"])
        for item in listed.values():
            bridge.archive.store.verify(item["head"], recursive=True)
        bridge.control("gc")
        _, latents, caches, _ = drive(bridge, request(handle("s1", "resume"), streams=streams))
        self.assertLatentsEqual(latents, expected_latents)
        self.assertCachesEqual(caches, expected_caches)


def run_crash_child(test, representation):
    """Run ``CRASH_CHILD`` on ``<workspace>/crashed`` and SIGKILL it once it blocks inside archival."""
    root = test.workspace / "crashed"
    marker, result = test.workspace / "gate", test.workspace / "result.json"
    spec = {"root": str(root), "representation": representation, "marker": str(marker), "result": str(result)}
    log = test.workspace / "crash_child.log"
    with open(log, "w") as output:
        child = subprocess.Popen([sys.executable, "-c", CRASH_CHILD, json.dumps(spec)],
                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, stdout=output,
                                 stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 600
            while not marker.exists():
                if child.poll() is not None:
                    test.fail(f"crash child exited early: {log.read_text()[-3000:]}")
                if time.monotonic() > deadline:
                    test.fail("crash child never reached its gate")
                time.sleep(0.05)
            child.send_signal(signal.SIGKILL)
            child.wait(60)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(60)
    test.assertEqual(child.returncode, -signal.SIGKILL)
    with open(result) as handle:
        return json.load(handle)


class LineageControlTests(SessionControlChecks, BridgeTestCase):
    representation = "svd_hevc"
    assertLatentsEqual = LineageSessionTests.assertLatentsEqual
    assertCachesEqual = LineageSessionTests.assertCachesEqual

    def test_cfg_group_plan_failure_in_one_stream_publishes_nothing(self):
        streams = ("cond", "uncond")
        bridge = self.bridge()
        from reelcache import lineage_commit

        real = lineage_commit.LatestMethodPlanner.plan

        def failing(planner, *args, chunk_id=None, **kwargs):
            if chunk_id is not None and "c000001" in chunk_id and chunk_id.endswith(":uncond"):
                raise RuntimeError("injected uncond-stream planning failure")
            return real(planner, *args, chunk_id=chunk_id, **kwargs)

        with mock.patch.object(lineage_commit.LatestMethodPlanner, "plan", failing):
            first, _, _, _ = drive(bridge, request(handle("s1"), streams=streams, blocks=3))
            self.assertEqual(first["status"], "completed")  # archival runs behind the request
            error = control_error(self, "archival_failure", "pause", {"session_id": "s1"}, bridge)
        self.assertIn("c000001", str(error))
        path = bridge.archive.path(bridge.archive.head("s1"))
        self.assertEqual([node.kind for node in path], ["root", "chunk"])  # the previous checkpoint
        self.assertEqual(path[-1].streams, streams)
        self.assertEqual(bridge.control("describe", {"session_id": "s1"})["generated_chunks"], 1)
        resumed, _, _, _ = drive(bridge, request(handle("s1", "resume"), streams=streams, blocks=3))
        self.assertEqual((resumed["resumed_from_chunks"], resumed["generated_chunks"]), (1, 3))

    def test_close_abort_retires_unpublished_commits_and_drain_publishes_them(self):
        bridge = self.bridge()
        gate = threading.Event()
        publication_hook(bridge, gate=gate)  # nothing publishes until the gate opens
        drive(bridge, request(handle("s1"), blocks=3))
        drive(bridge, request(handle("s2"), blocks=3))
        self.assertEqual(len(bridge.controller.get("s1").pending), 3)
        results = {}

        def close(session_id, mode):
            results[session_id] = bridge.control("close", {"session_id": session_id, "mode": mode})

        closer = threading.Thread(target=close, args=("s1", "abort"))
        closer.start()
        time.sleep(0.3)
        gate.set()  # the in-flight stage finishes, then sees its cancellation
        closer.join(60)
        aborted = results["s1"]
        self.assertEqual((aborted["status"], aborted["mode"], aborted["was_open"]), ("closed", "abort", True))
        # A commit already inside its publication may land; every later one is retired, never published.
        published = sum(node.kind == "chunk" for node in bridge.archive.path(bridge.archive.head("s1")))
        self.assertEqual(aborted["generated_chunks"], published)
        self.assertGreaterEqual(len(aborted["retired"]), 1)
        self.assertEqual(sorted(aborted["retired"]), [f"c{index:06d}" for index in range(published, 3)])
        drained = bridge.control("close", {"session_id": "s2", "mode": "drain"})
        self.assertEqual((drained["retired"], drained["generated_chunks"]), ([], 3))
        self.assertEqual(sum(node.kind == "chunk" for node in bridge.archive.path(drained["head"])), 3)
        self.assertEqual(bridge.control("describe", {"session_id": "s1"})["open"], False)

    def test_sigkill_between_submission_and_publication_never_exposes_a_half_published_node(self):
        streams = ("cond", "uncond")
        report = run_crash_child(self, "svd_hevc")
        bridge = self.bridge("crashed", raw_restore="off")
        listed = {item["session_id"]: item for item in bridge.control("list")["sessions"]}
        self.assertEqual(sorted(listed), ["pre", "s1"])
        self.assertEqual(listed["s1"]["head"], report["s1_head"])  # c000002 was never published
        self.assertEqual(listed["s1"]["generated_chunks"], 2)
        for item in listed.values():
            bridge.archive.store.verify(item["head"], recursive=True)
        bridge.control("gc")
        # Lossy: a restore of the same checkpoint after the restart equals the restore before it.
        _, latents, caches, _ = drive(bridge, request(handle("post", "fork", from_session="s1", checkpoint=2,
                                                             pause_after_chunks=3), streams=streams))
        self.assertTrue(torch.equal(latents[:, :, 2 * FRAMES:3 * FRAMES], torch.tensor(report["pre_latents"])))
        self.assertTrue(torch.equal(caches["cond"][0].k[:, :caches["cond"][0].end].float(),
                                    torch.tensor(report["pre_key"])))
        resumed, _, _, _ = drive(bridge, request(handle("s1", "resume"), streams=streams))
        self.assertEqual((resumed["resumed_from_chunks"], resumed["generated_chunks"]), (2, 4))

    def test_archival_yields_to_the_bridge_restorer_without_seal_time_preparation(self):
        # Step 7: archival yields while a restore is pending; without seal-time preparation the pipeline gets the
        # bridge's restorer explicitly (yield_to), so restores always get the yield.
        for seal_prepare in (False, True):
            archive = self.bridge(f"yield_{seal_prepare}", seal_prepare=seal_prepare).archive
            gate = getattr(archive.pipeline, "_yield", None)
            self.assertIsNotNone(gate)
            self.assertIs(gate.activity, archive.restorer.activity)

    def test_memory_critical_section_defers_gpu_fitting_and_is_recorded(self):
        bridge = self.bridge()
        archive = bridge.archive
        with bridge.memory_critical_section("vae_decode", request_id="r0") as report:
            # Freeze F: with a raw tier (on by default) the archive's section always runs (its device-pressure
            # response); without the deferral policy it defers nothing (no device on CPU: no pressure event).
            self.assertEqual(report, {"reason": "vae_decode", "deferred": False})
        self.assertEqual(archive.pipeline.stats()["deferrals"], 0)
        off = self.bridge("tier_off", raw_tier=False)
        with off.memory_critical_section("vae_decode", request_id="r0") as report:
            self.assertIsNone(report)  # C3: no raw tier and no deferral policy: a no-op context
        self.assertEqual(archive.pipeline.stats()["deferrals"], 0)
        bridge._remember("r1", "s1", {"status": "completed"})
        enabled = SimpleNamespace(defer_archival_during_decode=True)
        # The GPU path needs fit_device="cuda" (validated below); on CPU the same pipeline section runs with a
        # stand-in fitter.
        with mock.patch.object(bridge, "config", enabled), mock.patch.object(archive, "config", enabled), \
                mock.patch.object(archive, "fitter", object()):
            with bridge.memory_critical_section("vae_decode", request_id="r1") as report:
                self.assertTrue(archive.pipeline.stats()["deferred"])
                self.assertTrue(report["deferred"])
            self.assertFalse(archive.pipeline.stats()["deferred"])
        self.assertEqual(archive.pipeline.stats()["deferrals"], 1)
        recorded = bridge.request_summary("r1")["memory_critical"]
        self.assertEqual([(item["reason"], item["deferred"]) for item in recorded], [("vae_decode", True)])
        self.assertGreaterEqual(recorded[0]["section_seconds"], 0.0)
        self.assertEqual(bridge.control("capabilities")["archive_memory"]["defer_archival_during_decode"], False)
        # A memory policy (one per GPU deployment): accepted with RAW commits or CPU fitting, where it has nothing
        # to defer (the RAW archive declares no section; a lossy archive without GPU fitter yields no deferral).
        base = {"store_root": str(self.workspace / "cfg"), "energy_key": 0.98, "energy_value": 0.95,
                "defer_archival_during_decode": True}
        for other in ({"representation": "raw"}, {"representation": "svd_hevc", "fit_device": "cpu"}):
            self.assertTrue(ReelCacheRuntimeConfig.from_dict({**base, **other}).defer_archival_during_decode)
        raw = self.bridge("raw_policy", representation="raw", defer_archival_during_decode=True)
        with raw.memory_critical_section("vae_decode", request_id="r2") as report:
            self.assertIsNone(report)
        cpu = self.bridge("cpu_policy", defer_archival_during_decode=True)  # fit_device cpu
        with cpu.memory_critical_section("vae_decode") as report:
            self.assertEqual(report, {"reason": "vae_decode", "deferred": False})
        self.assertEqual(cpu.archive.pipeline.stats()["deferrals"], 0)


if __name__ == "__main__":
    unittest.main()
