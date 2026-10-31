"""SGLang stages of the causal-Wan profiles (imports SGLang).

* ``ProfileTextEncodingStage``      - encodes the initial prompt and every prompt-switch prompt, each in its own
                                      text-encoder call as upstream LongLive 1.0 does (one embedding row per
                                      segment; Step 14 CR-14-4).
* ``ProfileLatentPreparationStage`` - draws the initial noise exactly as the original repository does (layout,
                                      dtype, request generator), then hands SGLang its [B, C, T, H, W] view.
* ``ProfileCausalDenoisingStage``   - the causal block loop of ``CausalDMDDenoisingStage`` with request-scoped
                                      caches sized by the visibility policy (CausVid ``full_history`` /
                                      ``window_21_frames``, LongLive sink + rolling window), per-chunk observer
                                      hooks and the LongLive 1.0 prompt-switch KV-recache as an explicit step
                                      (``run_prompt_switch_recache``) whose ``RecacheRecord`` Step 14 publishes as
                                      a CacheEpoch. Chunks are still denoised and committed through the base
                                      helpers (``_denoise_and_update_causal_block``, ``_update_causal_context_cache``)
                                      so seams added there by the bridge apply unchanged.
* ``RavenProfileDenoisingStage``    - RAVEN's consistency sampler (integer timesteps, float32 v-lerp step,
                                      forked step-noise generator drawn at every step).

ReelCache bridge (Step 12 phase 2; SGLang branch ``reelcache-integration`` with ``--reelcache-enable``): the denoising
stage opens the backend run of a ``causal_kv_session`` request through the base-stage seam
(``_begin_causal_kv_run``), skips the chunks the bridge restored (latents, generator states and the attended KV,
re-rotated with the engine's RoPE at the original positions), hands every clean chunk forward to the bridge's
attention-level capture (``_causal_kv_pending`` -> ``_update_causal_context_cache``), commits/pauses at each chunk
boundary (``end_chunk``) and publishes a LongLive 1.0 prompt-switch recache as a cache epoch carrying only the
frames the recache wrote (``_causal_kv_recache_scope``). Without a session handle (or on a SGLang tree without the
seam) the stage runs exactly as before.

Resident session KV (branch commit 1663578341): like SGLang's own causal stage, the profile stage keeps its caches
across requests (``causal_kv_cache`` / ``crossattn_cache``) and, when a request's cache geometry is the same, only
resets their cursors and the cross-attention caches (``request_caches``). The bridge records the caches'
``index_state`` at the end of a session request; the session's next ``resume`` continues on that KV with no restore
and no wait for publication when nothing wrote it since (``mutation_generation``; any other request, restore or a new
geometry renews it, and the next resume then restores from the archive). Profile code that writes K/V directly
renews the tag (``sglang_cache.note_kv_write``). On e00930c (no such seam) every request allocates its own caches.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any

import torch

from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalSelfAttentionKVCache,
    CrossAttentionKVCache,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import CausalDMDDenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationSpec,
    LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import TextEncodingStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from .lifecycle import (block_conditioning_ids, recache_epoch_details, recache_epoch_id, request_cache_tokens,
                        resume_plan, segment_of, switch_recache_restored)
from .profiles import ModelProfile, VisibilityPolicy
from .recache import CLEAN_COMMIT_FORWARD_KIND, RECACHE_FORWARD_KIND, RecacheRecord, plan_recache
from .sampling import raven_forward, raven_x0, yield_seed
from .sglang_cache import RESIDENT_KV_SEAM, RecacheAwareCausalKVCache, note_kv_write
from .sglang_configs import normalize_prompt_switches

logger = init_logger(__name__)

# SGLang branch ``reelcache-integration`` (Step 11 CR b) moved LongLive's sink-protected recompute write into the base
# cache (``protect_sink_on_recompute``); e00930c lacks it and keeps using ``RecacheAwareCausalKVCache``.
BASE_CACHE_PROTECTS_SINK = "protect_sink_on_recompute" in getattr(CausalSelfAttentionKVCache, "__dataclass_fields__", {})
# Branch commit 8cbb8d1d9f (Step 11, CR-12p2-1): per-block conditioning identity. With it, prompt switches are
# compared per restored block instead of being part of the whole request's conditioning.
SEAM_BLOCK_CONDITIONING = callable(getattr(CausalDMDDenoisingStage, "_causal_kv_block_conditioning", None))

SEGMENTS_KEY = "reelcache.causal_wan.segments"
OBSERVER_KEY = "reelcache.causal_wan.observer"
RECORDS_KEY = "reelcache.causal_wan.records"
FORWARD_KIND_KEY = "reelcache.causal_wan.forward_kind"
NOISE_LAYOUT_KEY = "reelcache.causal_wan.initial_noise_layout"
INJECT_PROMPT_EMBEDS_KEY = "reelcache.causal_wan.diagnostic_inject_prompt_embeds"
INJECT_REPORT_KEY = "reelcache.causal_wan.diagnostic_inject_report"
# Diagnostics only (teacher forcing for parity runs): {chunk_index: [B, C, T, H, W] latents} replace the denoised
# chunk before it is committed, so the next chunk is predicted from the reference history; the replaced
# predictions are kept under FORCED_PREDICTIONS_KEY. Without the key the base helper runs unchanged.
FORCE_CHUNKS_KEY = "reelcache.causal_wan.diagnostic_force_chunks"
FORCED_PREDICTIONS_KEY = "reelcache.causal_wan.diagnostic_forced_predictions"


def _request_seed(batch: Req) -> int:
    seeds = getattr(batch, "seeds", None)
    if isinstance(seeds, list) and len(seeds) == 1:
        return int(seeds[0])
    seed = batch.seed
    if isinstance(seed, list):
        if len(seed) != 1:
            raise ValueError("causal-Wan profiles run one sample per request")
        seed = seed[0]
    return int(seed)


def _observer(batch: Req):
    return batch.extra.get(OBSERVER_KEY)


def _notify(batch: Req, name: str, *args) -> None:
    obs = _observer(batch)
    fn = getattr(obs, name, None) if obs is not None else None
    if fn is not None:
        fn(*args)


def concat_encoded_rows(parts: list):
    """Merge the outputs of per-prompt ``TextEncodingStage.encode_text`` calls into the structure one batched call
    returns: tensors concatenated along the batch (row) dimension in prompt order, per-row value lists (sequence
    lengths) concatenated, tuples / per-encoder lists / dicts merged element-wise, ``None`` kept."""
    first = parts[0]
    if first is None:
        if any(part is not None for part in parts):
            raise ValueError("per-prompt text encodings differ in structure")
        return None
    if isinstance(first, torch.Tensor):
        return torch.cat(parts, dim=0)
    if isinstance(first, tuple):
        return tuple(concat_encoded_rows([part[i] for part in parts]) for i in range(len(first)))
    if isinstance(first, dict):
        return {key: concat_encoded_rows([part[key] for part in parts]) for key in first}
    if isinstance(first, list):
        if all(isinstance(item, (int, float)) for item in first):  # per-row values (sequence lengths)
            return [item for part in parts for item in part]
        if any(len(part) != len(first) for part in parts):
            raise ValueError("per-prompt text encodings differ in structure")
        return [concat_encoded_rows([part[i] for part in parts]) for i in range(len(first))]
    raise TypeError(f"cannot merge text-encoder output of type {type(first).__name__}")


class ProfileTextEncodingStage(TextEncodingStage):
    """Encodes ``[prompt] + [switch prompts]`` (one embedding row per segment), each prompt in its own call.

    Upstream LongLive 1.0 encodes every segment's prompts in a separate text-encoder call
    (``pipeline/interactive_causal_inference.py:128`` at c6a27e7: ``cond_list = [self.text_encoder(text_prompts=p)
    for p in text_prompts_list]``), and the umt5 encoder is not batch-invariant: inside one batched ``[A, B]`` call
    A's embedding differs from A alone (bf16 max |d| 0.043), which changed every chunk of a switch request, those
    before the switch included. ``encode_text`` therefore encodes a list of prompts one prompt per call and stacks the
    rows (``concat_encoded_rows``), so a segment's embedding never depends on the schedule (Step 14 CR-14-4). A
    single prompt takes SGLang's own path unchanged.

    Parity diagnostic (in-process only, never set by the request API): when ``batch.extra[INJECT_PROMPT_EMBEDS_KEY]``
    holds a tensor [segments, 512, 4096], the encoder still runs, its output is compared with the injected
    tensor (``batch.extra[INJECT_REPORT_KEY]``) and the injected embeddings are used for denoising.
    """

    deduplicated_output_fields = ()

    def encode_text(self, text, server_args, *args, **kwargs):
        if isinstance(text, str) or len(text) <= 1:
            return super().encode_text(text, server_args, *args, **kwargs)
        return_type = kwargs.get("return_type", args[2] if len(args) > 2 else "list")
        if return_type == "stack":  # stacked per encoder: rows are not the leading dimension
            raise NotImplementedError("per-prompt text encoding supports return_type 'list' and 'dict'")
        return concat_encoded_rows([super(ProfileTextEncodingStage, self).encode_text([item], server_args, *args,
                                                                                       **kwargs)
                                    for item in text])

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        switches = normalize_prompt_switches(getattr(batch, "prompt_switches", None))
        if isinstance(batch.prompt, list):
            raise ValueError("causal-Wan profiles take one prompt per request")
        batch.extra[SEGMENTS_KEY] = {"switch_chunks": [s["chunk"] for s in switches], "count": 1 + len(switches)}
        if not switches:
            out = super().forward(batch, server_args)
        else:
            original = batch.prompt
            batch.prompt = [original] + [s["prompt"] for s in switches]
            try:
                out = super().forward(batch, server_args)
            finally:
                batch.prompt = original
        injected = batch.extra.get(INJECT_PROMPT_EMBEDS_KEY)
        if injected is not None:
            computed = out.prompt_embeds[0]
            injected = injected.to(device=computed.device, dtype=computed.dtype)
            if injected.shape != computed.shape:
                raise ValueError(f"injected prompt embeddings {tuple(injected.shape)} != {tuple(computed.shape)}")
            diff = (computed.float() - injected.float())
            batch.extra[INJECT_REPORT_KEY] = {
                "max_abs": float(diff.abs().max()), "relative_frobenius": float(diff.norm() / injected.float().norm()),
                "bitwise_equal": bool(torch.equal(computed, injected)), "dtype": str(computed.dtype)}
            out.prompt_embeds[0] = injected
        return out


class ProfileLatentPreparationStage(LatentPreparationStage):
    """Initial noise drawn with the original repository's layout/dtype from the request generator."""

    def __init__(self, scheduler, transformer, profile: ModelProfile) -> None:
        super().__init__(scheduler=scheduler, transformer=transformer)
        self.profile = profile

    def get_latent_preparation_spec(self, batch, server_args, batch_size, num_frames, device) -> LatentPreparationSpec:
        b, c, t, h, w = server_args.pipeline_config.prepare_latent_shape(batch, batch_size, num_frames)
        noise = self.profile.sampler.noise
        shape = (b, t, c, h, w) if noise.initial_layout == "B,T,C,H,W" else (b, c, t, h, w)
        return LatentPreparationSpec(shape=shape, dtype=getattr(torch, noise.initial_dtype), device=device,
                                     prepare_latent_ids=False, pack_latents=False)

    def should_scale_initial_noise(self, batch, server_args) -> bool:
        return False

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.latents is not None:
            raise ValueError(f"{self.profile.profile_id}: preset latents are not supported; the profile draws the "
                             "native initial noise from the request seed")
        if batch.batch_size != 1:
            raise ValueError("causal-Wan profiles run one sample per request")
        batch = super().forward(batch, server_args)
        layout = self.profile.sampler.noise.initial_layout
        if layout == "B,T,C,H,W":
            batch.latents = batch.latents.permute(0, 2, 1, 3, 4).contiguous()
        batch.raw_latent_shape = batch.latents.shape
        batch.extra[NOISE_LAYOUT_KEY] = layout
        return batch

    def run_grouped_requests(self, batches: list[Req], server_args: ServerArgs) -> list[Req]:
        return [self(batch, server_args) for batch in batches]


class ProfileCausalDenoisingStage(CausalDMDDenoisingStage):
    """Causal block loop with profile visibility, observer hooks and the LongLive 1.0 recache step."""

    if SEAM_BLOCK_CONDITIONING:
        # The prompt-switch schedule is the per-block conditioning (``_causal_kv_block_conditioning``); without
        # that seam it stays part of the request conditioning (a session never resumes under another schedule).
        _CAUSAL_KV_IDENTITY_EXCLUDE = CausalDMDDenoisingStage._CAUSAL_KV_IDENTITY_EXCLUDE | frozenset({"prompt_switches"})

    #: Keep the caches across requests (reset cursors only) where SGLang's caches carry the resident-KV seam; a
    #: harness may set it False on an instance to get fresh caches per request.
    REUSE_REQUEST_CACHES = RESIDENT_KV_SEAM

    def __init__(self, transformer, scheduler, profile: ModelProfile) -> None:
        super().__init__(transformer=transformer, scheduler=scheduler)
        self.profile = profile
        self.num_frames_per_block = profile.chunk_latent_frames
        self.sink_size = profile.visibility().sink_latent_frames

    # ------------------------------------------------------------------ policy
    def resolve_visibility(self, batch: Req, server_args: ServerArgs) -> VisibilityPolicy:
        name = getattr(batch, "context_policy", None) or getattr(server_args.pipeline_config, "context_policy", None)
        return self.profile.visibility(name)

    def _cache_spec(self) -> tuple[type[CausalSelfAttentionKVCache], dict]:
        """Cache class + extra fields: a prompt-switch profile needs the upstream sink-protected recompute write."""
        if self.profile.recache.trigger != "prompt_switch":
            return CausalSelfAttentionKVCache, {}
        if BASE_CACHE_PROTECTS_SINK:
            return CausalSelfAttentionKVCache, {"protect_sink_on_recompute": True}
        return RecacheAwareCausalKVCache, {}

    def allocate_request_caches(self, *, batch_size: int, total_frames: int, policy: VisibilityPolicy,
                                dtype: torch.dtype, device, max_text_len: int) -> tuple[list, list]:
        """New zeroed self- and cross-attention caches for a request (the stage keeps them: ``request_caches``)."""
        geometry = request_cache_tokens(self.profile, policy, total_frames=total_frames,
                                        tokens_per_frame=self.num_token_per_frame)
        size = geometry["cache_tokens"]
        heads, head_dim = self.transformer.num_attention_heads, self.transformer.attention_head_dim
        cls, extra = self._cache_spec()
        kv = [cls(k=torch.zeros((batch_size, size, heads, head_dim), dtype=dtype, device=device),
                  v=torch.zeros((batch_size, size, heads, head_dim), dtype=dtype, device=device),
                  global_end_index=torch.zeros(1, dtype=torch.long, device=device),
                  local_end_index=torch.zeros(1, dtype=torch.long, device=device),
                  global_end_index_int=0, local_end_index_int=0, cache_size=size,
                  sink_tokens=geometry["sink_tokens"], attention_window_size=geometry["attention_window_tokens"],
                  **extra)
              for _ in range(self.num_transformer_blocks)]
        cross = [CrossAttentionKVCache(k=torch.zeros((batch_size, max_text_len, heads, head_dim), dtype=dtype, device=device),
                                       v=torch.zeros((batch_size, max_text_len, heads, head_dim), dtype=dtype, device=device))
                 for _ in range(self.num_transformer_blocks)]
        return kv, cross

    def _stage_caches_fit(self, *, batch_size: int, total_frames: int, policy: VisibilityPolicy, dtype: torch.dtype,
                          device, max_text_len: int) -> bool:
        """Whether the stage's caches have exactly the geometry ``allocate_request_caches`` would give the request."""
        kv, cross = self.causal_kv_cache, self.crossattn_cache
        layers = self.num_transformer_blocks
        if kv is None or cross is None or len(kv) != layers or len(cross) != layers:
            return False
        geometry = request_cache_tokens(self.profile, policy, total_frames=total_frames,
                                        tokens_per_frame=self.num_token_per_frame)
        heads, head_dim = self.transformer.num_attention_heads, self.transformer.attention_head_dim
        cls, _ = self._cache_spec()
        device = torch.device(device)
        shape = (batch_size, geometry["cache_tokens"], heads, head_dim)

        def same_tensor(t):
            return tuple(t.shape) == shape and t.dtype == dtype and t.device == device

        # Cross-attention caches are reset (``is_init`` False) and their K/V replaced by the first forward
        # (``CrossAttentionKVCache.store``), so only their number matters.
        return all(
            type(c) is cls and same_tensor(c.k) and same_tensor(c.v)
            and (c.cache_size, c.sink_tokens, c.attention_window_size, c.global_sink_tokens, c.allow_growth)
            == (geometry["cache_tokens"], geometry["sink_tokens"], geometry["attention_window_tokens"], 0, False)
            and c.global_end_index_int is not None and c.local_end_index_int is not None
            for c in kv) and all(isinstance(x, CrossAttentionKVCache) for x in cross)

    def request_caches(self, *, batch_size: int, total_frames: int, policy: VisibilityPolicy, dtype: torch.dtype,
                       device, max_text_len: int) -> tuple[list, list, bool]:
        """The request's self- and cross-attention caches and whether the stage's own caches were reused.

        Like SGLang's causal stage (``_cache_needs_reinit_for_batch``): caches whose geometry is the request's are
        reused with only their cursors, pinned sink and the cross-attention caches reset, so the K/V a session left
        there (and its ``mutation_generation``) survives until another write; otherwise the stage releases them and
        allocates new ones (a new geometry, e.g. a full-history request of another length, is never resident)."""
        args = dict(batch_size=batch_size, total_frames=total_frames, policy=policy, dtype=dtype, device=device,
                    max_text_len=max_text_len)
        if not self.REUSE_REQUEST_CACHES:
            return (*self.allocate_request_caches(**args), False)
        if self._stage_caches_fit(**args):
            kv, cross = self.causal_kv_cache, self.crossattn_cache
            _, extra = self._cache_spec()
            for c in kv:
                c.reset_indices()  # cursors and pinned sink; K/V and its mutation tag stay
                for name, value in extra.items():
                    setattr(c, name, value)
            self._reset_crossattn_cache(cross)
            return kv, cross, True
        self.causal_kv_cache = self.crossattn_cache = None  # release first: never hold two cache sets
        kv, cross = self.allocate_request_caches(**args)
        self.causal_kv_cache, self.crossattn_cache = kv, cross
        return kv, cross, False

    @staticmethod
    def _set_sink_recompute(kv_cache, sink_recache: bool) -> None:
        """During a recache without a global sink (upstream ``sink_recache_after_switch``) the sink is rewritten too."""
        for c in kv_cache:
            if isinstance(c, RecacheAwareCausalKVCache):
                c.sink_recache_after_switch = sink_recache
            elif BASE_CACHE_PROTECTS_SINK:
                c.protect_sink_on_recompute = not sink_recache

    # ------------------------------------------------------------------ recache (LongLive 1.0)
    def run_prompt_switch_recache(self, batch: Req, server_args: ServerArgs, *, ctx, kv_cache, crossattn_cache,
                                  latents: torch.Tensor, current_start_frame: int, new_prompt_embeds,
                                  switch_before_chunk: int, previous_segment: int, new_segment: int,
                                  kv_run=None, new_prompt: str | None = None) -> RecacheRecord | None:
        """Upstream ``_recache_after_switch``: one clean unmasked forward over the last min(12, n) frames under the
        new prompt; the sink keeps its KV (global_sink); only the frames beside it are written back.

        With a ReelCache session run (``kv_run``) the recache forward is captured and the frames it wrote (never the
        sink) are archived as a new cache epoch (``_causal_kv_recache_scope``); restore then resolves those frames
        to their newest version on the session path."""
        rp = self.profile.recache
        if rp.trigger != "prompt_switch":
            raise RuntimeError(f"{self.profile.profile_id} has no prompt-switch recache")
        fs = self.num_token_per_frame
        g, l = kv_cache[0]._read_indices()
        plan = plan_recache(current_start_frame=current_start_frame, local_attn_frames=rp.recache_latent_frames,
                            sink_frames=self.profile.visibility().sink_latent_frames, frame_seq=fs,
                            global_end_tokens=g, local_end_tokens=l, global_sink=rp.global_sink)
        if plan is None:
            self._reset_crossattn_cache(crossattn_cache)
            return None
        record = RecacheRecord(profile_id=self.profile.profile_id, switch_before_chunk=switch_before_chunk,
                               previous_segment=previous_segment, new_segment=new_segment, plan=plan.to_dict(),
                               context_timestep=int(getattr(server_args.pipeline_config, "context_noise", 0)),
                               cross_attention_reset=True)
        epoch = None
        if kv_run is not None:
            if not new_prompt:
                raise ValueError("a captured prompt-switch recache needs the new prompt text (epoch identity)")
            if plan.sink_recache or not plan.is_recompute:
                raise NotImplementedError(
                    f"{self.profile.profile_id}: the recache before chunk {switch_before_chunk} rewrites the sink "
                    f"(written frames {plan.written_global_frames}); archiving it as a cache epoch needs whole "
                    "archived chunks outside the sink")
            epoch_id = recache_epoch_id(switch_before_chunk, new_prompt)
            epoch = self._causal_kv_recache(
                batch, window_start_frame=plan.recache_start_frame, window_frames=plan.num_recache_frames,
                written_frames=tuple(plan.written_global_frames), pos_start_base=0, epoch_id=epoch_id,
                details=recache_epoch_details(record.cache_epoch_fields(), prompt=new_prompt, segment=new_segment,
                                              profile_id=self.profile.profile_id))
        _notify(batch, "before_recache", plan.to_dict(), kv_cache, crossattn_cache)
        if plan.zero_kv_before:
            for c in kv_cache:
                # W1.4.1: a direct write: whatever still reads these slots (a restore, a promotion copy) goes first
                c.await_restored_history()
                c.k.zero_()
                c.v.zero_()
                note_kv_write(c)
        self._reset_crossattn_cache(crossattn_cache)
        rs, cs = plan.recache_start_frame, plan.current_start_frame
        self._set_sink_recompute(kv_cache, plan.sink_recache)
        _before_device_sync(kv_run, "recache_start")
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        batch.extra[FORWARD_KIND_KEY] = RECACHE_FORWARD_KIND
        archived = None
        try:
            with self._causal_kv_recache_scope(kv_run, epoch) if epoch is not None else nullcontext():
                self._update_causal_context_cache(
                    batch, server_args, context_input=latents[:, :, rs:cs], prompt_embeds=new_prompt_embeds,
                    kv_cache=kv_cache, crossattn_cache=crossattn_cache, current_start_tokens=rs * fs, start_frame=rs,
                    image_kwargs=ctx.image_kwargs, pos_cond_kwargs=ctx.pos_cond_kwargs, attn_metadata=None,
                    target_dtype=ctx.target_dtype, autocast_enabled=ctx.autocast_enabled)
                _before_device_sync(kv_run, "recache_end")
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                seconds = time.perf_counter() - t0
                t1 = time.perf_counter()
            if epoch is not None:
                archived = {"epoch_id": epoch.epoch_id, "written_tokens": [list(w) for w in epoch.written],
                            "window": {"token_start": epoch.window.token_start,
                                       "num_tokens": epoch.window.num_tokens,
                                       "rope_start_frame": float(epoch.window.rope_start_frame)},
                            "submit_seconds": time.perf_counter() - t1}
        finally:
            batch.extra[FORWARD_KIND_KEY] = CLEAN_COMMIT_FORWARD_KIND
            self._set_sink_recompute(kv_cache, False)
        self._reset_crossattn_cache(crossattn_cache)
        after = [c._read_indices() for c in kv_cache]
        if any(a != (g, l) for a in after):
            raise RuntimeError(f"recache moved the cache indices: {(g, l)} -> {after[0]}")
        record.indices_after = {"global_end_tokens": g, "local_end_tokens": l}
        record.seconds = seconds
        record.cache_epoch_archive = archived
        return record

    # ------------------------------------------------------------------ diagnostics
    def _denoise_and_update_causal_block(self, batch: Req, server_args: ServerArgs, **kwargs) -> torch.Tensor:
        force = batch.extra.get(FORCE_CHUNKS_KEY)
        if not force:
            return super()._denoise_and_update_causal_block(batch, server_args, **kwargs)
        prepare_context_input = kwargs.pop("prepare_context_input")
        current, attn_metadata = self._denoise_causal_dmd_chunk(batch, server_args, **kwargs)
        ci = kwargs["start_frame"] // self.profile.chunk_latent_frames
        forced = force.get(ci)
        if forced is not None:
            batch.extra.setdefault(FORCED_PREDICTIONS_KEY, {})[ci] = current.detach().clone()
            if forced.shape != current.shape:
                raise ValueError(f"forced chunk {ci} has shape {tuple(forced.shape)}, expected {tuple(current.shape)}")
            current = forced.to(device=current.device, dtype=current.dtype)
        self._update_causal_context_cache(
            batch, server_args, context_input=prepare_context_input(current), prompt_embeds=kwargs["prompt_embeds"],
            kv_cache=kwargs["kv_cache"], crossattn_cache=kwargs["crossattn_cache"],
            current_start_tokens=kwargs["current_start_tokens"], start_frame=kwargs["start_frame"],
            image_kwargs=kwargs["image_kwargs"], pos_cond_kwargs=kwargs["pos_cond_kwargs"], attn_metadata=attn_metadata,
            target_dtype=kwargs["target_dtype"], autocast_enabled=kwargs["autocast_enabled"])
        return current

    # ------------------------------------------------------------------ main loop
    @staticmethod
    def _segment_embeds(prompt_embeds, segment: int, count: int):
        if count == 1:
            return prompt_embeds
        return [pe[segment:segment + 1] for pe in prompt_embeds]

    def _causal_kv_block_conditioning(self, batch: Req, server_args: ServerArgs, block_sizes) -> tuple[str, ...] | None:
        """Seam hook (branch 8cbb8d1d9f): block i's id = its prompt segment (``lifecycle.block_conditioning_ids``).
        CausVid / RAVEN (workload v2, cross-attention prompt switch): the same ids when the request carries
        ``prompt_switches`` (possibly empty), none without it (F6; ``profiles.switch_block_conditioning``)."""
        from .profiles import switch_block_conditioning

        raw = getattr(batch, "prompt_switches", None)
        switches = None if raw is None else normalize_prompt_switches(raw)
        return switch_block_conditioning(self.profile, batch.prompt, switches, len(block_sizes))

    def _checkpoint_generators(self, batch: Req, device) -> list | None:
        """Generators whose states a session checkpoint must carry, if not just ``batch.generator``."""
        return None

    def _begin_profile_kv_run(self, batch: Req, server_args: ServerArgs, *, ctx, num_chunks: int, kv_cache,
                              latents: torch.Tensor) -> tuple[Any, int]:
        """Open the ReelCache bridge run of a ``causal_kv_session`` request (SGLang seam ``_begin_causal_kv_run``).

        Returns ``(None, 0)`` for ordinary generation (no session handle). Otherwise the run and the number of
        leading chunks the bridge restored into ``latents``/``kv_cache`` (and the checkpoint's generator states).
        A session handle without an attached backend (a server without ``--reelcache-enable``, or a SGLang tree
        without the seam) is refused explicitly, never ignored (the HTTP server rejects it too; Step 14 CR-14-2)."""
        begin = getattr(self, "_begin_causal_kv_run", None)
        if begin is None or getattr(self, "_causal_kv_backend", None) is None:
            if getattr(batch, "causal_kv_session", None) is not None:
                raise ValueError("causal_kv_session requires a server started with --reelcache-enable: no causal KV "
                                 f"backend is attached to the {self.profile.profile_id} denoising stage")
            return None, 0
        generators = self._checkpoint_generators(batch, latents.device)
        saved = batch.generator
        if generators is not None:
            batch.generator = generators  # the restore target (and every checkpoint) carries all of them
        try:
            return begin(batch, server_args, ctx=ctx, block_sizes=[self.profile.chunk_latent_frames] * num_chunks,
                         pos_start_base=0, use_cfg=False, kv_cache=kv_cache, kv_cache_neg=None, latents=latents)
        finally:
            batch.generator = saved

    if hasattr(CausalDMDDenoisingStage, "_begin_causal_kv_run"):
        def _begin_causal_kv_run(self, batch: Req, server_args: ServerArgs, **kwargs):
            """SGLang seam + workload v2 (W1b): after the bridge's restore, a ``restore_mode="recompute"`` run whose
            attended history missed the raw tiers carries a replay plan, which the stage replays here (exactly, before
            anything else reads the caches: :func:`replay_causal_history`); every run learns the request's per-block
            prompt texts (node sidecars of session trees)."""
            run, resume = super()._begin_causal_kv_run(batch, server_args, **kwargs)
            after_causal_kv_begin(self, batch, server_args, run, kwargs)
            return run, resume

        _begin_causal_kv_run._reelcache_replay_seam = True  # lifecycle.REPLAY_SEAM_MARKER: installed on the class

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        ctx = self._prepare_causal_dmd_forward_context(batch, server_args)
        if ctx.batch_size != 1:
            raise ValueError("causal-Wan profiles run one sample per request")
        if getattr(batch, "image_latent", None) is not None:
            raise ValueError("causal-Wan profiles are text-to-video only")
        if server_args.kv_cache_quant_config.enabled:
            raise ValueError("QVG KV-cache quantization is not supported by the causal-Wan profiles")
        chunk = self.profile.chunk_latent_frames
        t, h, w = ctx.num_frames, ctx.height, ctx.width
        if t % chunk:
            raise ValueError(f"{t} latent frames are not whole {chunk}-frame chunks")
        num_chunks = t // chunk
        policy = self.resolve_visibility(batch, server_args)
        segments = batch.extra.get(SEGMENTS_KEY) or {"switch_chunks": [], "count": 1}
        switch_chunks = list(segments["switch_chunks"])
        from .profiles import PROMPT_SWITCH_CROSS_ATTENTION, prompt_switch_mode

        mode = prompt_switch_mode(self.profile)  # recache (LongLive 1.0), cross_attention_only (CausVid, RAVEN), None
        if switch_chunks and mode is None:
            raise ValueError(f"{self.profile.profile_id} does not support prompt switches")
        if any(c >= num_chunks for c in switch_chunks):
            raise ValueError(f"prompt switch at chunk {switch_chunks} beyond the {num_chunks} requested chunks")
        switch_prompts = {s["chunk"]: s["prompt"]
                          for s in normalize_prompt_switches(getattr(batch, "prompt_switches", None))}
        count = segments["count"]
        for pe in ctx.prompt_embeds:
            if pe.shape[0] != count:
                raise ValueError(f"expected {count} prompt-embedding rows (one per segment), got {pe.shape[0]}")
        kv_cache, cross, caches_reused = self.request_caches(
            batch_size=ctx.batch_size, total_frames=t, policy=policy, dtype=ctx.target_dtype,
            device=ctx.latents.device, max_text_len=self._get_max_text_len(server_args))
        latents = ctx.latents
        fs = self.num_token_per_frame
        records: dict[str, Any] = {
            "profile_id": self.profile.profile_id, "profile_fingerprint": self.profile.fingerprint(),
            "visibility": policy.name, "cache_latent_frames": kv_cache[0].cache_size // fs,
            "sink_latent_frames": policy.sink_latent_frames, "num_chunks": num_chunks,
            "timesteps": [float(x) for x in ctx.timesteps.flatten().tolist()],
            "extrapolated_chunks": [i for i in range(num_chunks) if policy.is_extrapolated(i, chunk)],
            "switch_chunks": switch_chunks, "recaches": [], "chunk_seconds": [], "generated_chunks": [],
            "stage_caches": {"reused": caches_reused, "reuse_enabled": bool(self.REUSE_REQUEST_CACHES),
                             "resident_kv_seam": RESIDENT_KV_SEAM}}
        batch.extra[RECORDS_KEY] = records
        batch.extra[FORWARD_KIND_KEY] = CLEAN_COMMIT_FORWARD_KIND
        kv_run, resume = self._begin_profile_kv_run(batch, server_args, ctx=ctx, num_chunks=num_chunks,
                                                    kv_cache=kv_cache, latents=latents)
        paused = False
        telemetry = getattr(kv_run, "workload_telemetry", None)  # workload telemetry of the bridge run (None: off)
        with self._causal_kv_run_scope(kv_run) if kv_run is not None else nullcontext():
            switch_restored = None
            if kv_run is not None and resume in switch_chunks and mode == PROMPT_SWITCH_CROSS_ATTENTION:
                switch_restored = False  # cross-attention only: nothing of a switch is archived; it applies anew
            if kv_run is not None and resume in switch_chunks and switch_restored is None:
                switch_restored = switch_recache_restored(getattr(kv_run, "restore_report", None), resume)
                if switch_restored is None:
                    raise RuntimeError(f"cannot tell whether the prompt-switch recache before chunk {resume} is on "
                                       "the restored session path (the backend reported no visibility plans)")
            plan = resume_plan(num_chunks=num_chunks, switch_chunks=switch_chunks, resume_blocks=resume,
                               switch_restored=switch_restored)
            records["session"] = None if kv_run is None else {
                "resumed_from_chunks": resume, "restored_switches": list(plan.restored_switches),
                "recache_before_first_chunk": plan.recache_before_first_chunk and mode != PROMPT_SWITCH_CROSS_ATTENTION}
            _notify(batch, "on_request_start", {**records, "initial_latents": latents}, kv_cache, cross)
            # Chunks restored by the bridge are skipped; their segment switches (and recache epochs, when restored)
            # are already part of the restored state.
            segment = segment_of(resume, [s for s in switch_chunks if s < resume or s in plan.restored_switches])
            embeds = self._segment_embeds(ctx.prompt_embeds, segment, count)
            remaining = num_chunks - resume
            batch.record_stage_iterations(remaining * len(ctx.timesteps), remaining * len(ctx.timesteps))
            with self.progress_bar(total=remaining * len(ctx.timesteps), batch=batch) as progress_bar:
                for ci in range(resume, num_chunks):
                    start = ci * chunk
                    if (mode == PROMPT_SWITCH_CROSS_ATTENTION and ci in switch_chunks
                            and ci not in plan.restored_switches):
                        # CausVid / RAVEN (workload v2, user decision "cross_attention_only"): chunks ci.. cross-attend
                        # to the new prompt. The cross-attention caches are reset, so the next forward computes their
                        # K/V from the new prompt's embeddings; the self-attention KV (restored, resident or generated)
                        # is never recomputed and nothing is archived for the switch itself (no recache epoch).
                        transition = None if telemetry is None else telemetry.begin("prompt_transition")
                        prev, segment = segment, segment + 1
                        embeds = self._segment_embeds(ctx.prompt_embeds, segment, count)
                        self._reset_crossattn_cache(cross)
                        records.setdefault("cross_attention_switches", []).append(
                            {"before_chunk": ci, "previous_segment": prev, "segment": segment})
                        if telemetry is not None:
                            telemetry.end(transition)
                            telemetry.prompt_transition("switch")
                    elif ci in switch_chunks and ci not in plan.restored_switches:
                        # workload telemetry: reset + recache (+ its epoch's capture), distinct from chunk forwards
                        transition = None if telemetry is None else telemetry.begin("prompt_transition")
                        prev, segment = segment, segment + 1
                        embeds = self._segment_embeds(ctx.prompt_embeds, segment, count)
                        rec = self.run_prompt_switch_recache(
                            batch, server_args, ctx=ctx, kv_cache=kv_cache, crossattn_cache=cross, latents=latents,
                            current_start_frame=start, new_prompt_embeds=embeds, switch_before_chunk=ci,
                            previous_segment=prev, new_segment=segment, kv_run=kv_run,
                            new_prompt=switch_prompts.get(ci))
                        if rec is not None:
                            records["recaches"].append(rec.to_dict())
                            _notify(batch, "on_recache", rec, kv_cache, cross)
                        elif self.profile.recache.reset_cross_attention:
                            self._reset_crossattn_cache(cross)
                        if telemetry is not None:
                            telemetry.end(transition)
                            telemetry.prompt_transition("switch")  # "recache" once its epoch is committed
                    kv_chunk = None
                    if kv_run is not None:
                        kv_chunk = self._causal_kv_chunk(batch, block_index=ci, start_frame=start, num_frames=chunk,
                                                         pos_start_base=0, scene_cut=False)
                        self._causal_kv_pending = (kv_run, kv_chunk)
                    _before_device_sync(kv_run, "chunk_start")
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    t0 = time.perf_counter()
                    if telemetry is not None:
                        telemetry.mark_generation()  # the first denoise forward of a new block starts (t2)
                    try:
                        out = self._denoise_and_update_causal_block(
                            batch, server_args, chunk_latents=latents[:, :, start:start + chunk],
                            scheduler=ctx.scheduler, timesteps=ctx.timesteps, prompt_embeds=embeds, kv_cache=kv_cache,
                            crossattn_cache=cross, current_start_tokens=start * fs, start_frame=start,
                            image_kwargs=ctx.image_kwargs, pos_cond_kwargs=ctx.pos_cond_kwargs,
                            target_dtype=ctx.target_dtype, autocast_enabled=ctx.autocast_enabled, device=ctx.device,
                            attn_raw_latent_shape=(chunk, h, w), prepare_model_input=lambda x: x,
                            prepare_context_input=lambda x: x, progress_bar=progress_bar)
                    finally:
                        if kv_run is not None:
                            self._causal_kv_pending = None
                    latents[:, :, start:start + chunk] = out
                    _before_device_sync(kv_run, "chunk_end")
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    records["chunk_seconds"].append(time.perf_counter() - t0)
                    records["generated_chunks"].append(ci)
                    _notify(batch, "on_chunk_committed",
                            {"chunk_index": ci, "start_frame": start, "num_frames": chunk, "segment": segment,
                             "extrapolated": policy.is_extrapolated(ci, chunk),
                             "attended_latent_frames": [list(r) for r in policy.attended_latent_frames(ci, chunk)],
                             "latents": out}, kv_cache, cross)
                    if kv_run is not None and kv_run.end_chunk(kv_chunk, out):
                        # Paused at a clean boundary: only the generated chunks are output.
                        paused = True
                        latents = latents[:, :, :start + chunk]
                        records["paused_after_chunks"] = ci + 1
                        break
        batch.extra.pop(FORWARD_KIND_KEY, None)
        if kv_run is not None:
            batch.extra["causal_kv"] = kv_run.finish(error=None)
        records["paused"] = paused
        _notify(batch, "on_request_done", dict(records), kv_cache, cross)
        batch.latents = latents
        return batch


class RavenProfileDenoisingStage(ProfileCausalDenoisingStage):
    """RAVEN ConsistencySampler + LinearInterpolationSchedule(v_lerp) in float32 with its forked step-noise RNG."""

    STEP_GENERATOR_KEY = "reelcache.causal_wan.raven_step_generator"

    def _step_generator(self, batch: Req, device) -> torch.Generator:
        gen = batch.extra.get(self.STEP_GENERATOR_KEY)
        if gen is None:
            gen = torch.Generator(device=device).manual_seed(yield_seed(_request_seed(batch)))
            batch.extra[self.STEP_GENERATOR_KEY] = gen
        return gen

    def _checkpoint_generators(self, batch: Req, device) -> list:
        """RAVEN draws its step noise from a forked generator: a checkpoint carries its state as well."""
        request = batch.generator if isinstance(batch.generator, (list, tuple)) else [batch.generator]
        return [g for g in request if g is not None] + [self._step_generator(batch, device)]

    def _denoise_causal_dmd_chunk(self, batch: Req, server_args: ServerArgs, *, chunk_latents, scheduler, timesteps,
                                  prompt_embeds, kv_cache, crossattn_cache, current_start_tokens, start_frame,
                                  image_kwargs, pos_cond_kwargs, target_dtype, autocast_enabled, device,
                                  attn_raw_latent_shape, prepare_model_input, progress_bar=None):
        T = self.profile.sampler.num_train_timesteps
        steps = [int(t) for t in timesteps.flatten().tolist()]
        gen = self._step_generator(batch, chunk_latents.device)
        x = chunk_latents.to(torch.float32)
        b = x.shape[0]
        for i, t in enumerate(steps):
            s = steps[i + 1] if i + 1 < len(steps) else 0
            model_in = prepare_model_input(x).to(target_dtype)
            tt = torch.full((b, 1), t, device=x.device, dtype=torch.long)
            pred = self._forward_causal_transformer(
                batch, latent_model_input=model_in, prompt_embeds=prompt_embeds, timestep=tt, kv_cache=kv_cache,
                crossattn_cache=crossattn_cache, current_start_tokens=current_start_tokens, start_frame=start_frame,
                image_kwargs=image_kwargs, pos_cond_kwargs=pos_cond_kwargs, current_timestep=i, attn_metadata=None,
                target_dtype=target_dtype, autocast_enabled=autocast_enabled)
            outs = []
            for j in range(b):  # RAVEN steps each sample [C, T, H, W] separately
                x0 = raven_x0(pred[j].float(), x[j], t, T)
                noise = torch.randn(x0.shape, device=x0.device, generator=gen)
                outs.append(raven_forward(x0, noise, s, T).to(torch.float32))
            x = torch.stack(outs, dim=0)
            if progress_bar is not None:
                progress_bar.update()
        return x, None


# ---------------------------------------------------------------------------------------------------------------------
# Workload v2 (W1b): the exact recompute arm (``restore_mode="recompute"``). A restore whose attended history missed the
# raw tiers leaves the caches reset (and guarded) and hands the stage the path's clean forwards in native order
# (``run.replay_plan``); the stage replays them from the stored clean latents the bridge installed, each under its
# block's historical conditioning (validated by the bridge against the recorded per-block conditioning ids), without
# capture, then the bridge checks every cache's cursors against the RAW restore plan of the same node (``end_replay``).
# Exactness: a block's KV depends on everything it attended when it was generated, so the replay is the FULL chain
# from block 0 for every visibility policy (full history: CausVid / RAVEN; sink + rolling window: LongLive), including
# the LongLive 1.0 prompt-switch recache epochs at their positions and LongLive 2.0 shot conditioning (per-block
# prompts, scene-cut pinned sinks, multi-shot RoPE offsets). Clean forwards draw no noise; the RNG states are the
# checkpoint's (restored by the bridge). W1.4.1 (N2, selective recompute): the chain may start after the deepest
# ancestor whose visible set the raw tiers hold -- its RAW install is bitwise the native state every later block attends
# -- so the stage skips the plan's first ``start_step`` steps.

def _private_cross_caches(stage, server_args: ServerArgs, ctx) -> list:
    """Fresh cross-attention caches for the replay (the request's own stay as SGLang reset them: the first new forward
    computes them exactly as after any restore)."""
    heads = stage.transformer.num_attention_heads
    head_dim = stage.transformer.attention_head_dim
    text_len = stage._get_max_text_len(server_args)
    device = ctx.latents.device
    return [CrossAttentionKVCache(k=torch.zeros((ctx.batch_size, text_len, heads, head_dim), dtype=ctx.target_dtype,
                                                device=device),
                                  v=torch.zeros((ctx.batch_size, text_len, heads, head_dim), dtype=ctx.target_dtype,
                                                device=device))
            for _ in range(stage.num_transformer_blocks)]


def _check_rope_start(stage, step: dict) -> None:
    rope = float(stage._causal_rope_start_frame(int(step["start_frame"])))
    if rope != float(step["rope_start_frame"]):
        raise RuntimeError(f"replay of block {step['block']}: RoPE start {rope} != the archived "
                           f"{step['rope_start_frame']}")


def _replay_profile(stage, batch: Req, server_args: ServerArgs, *, steps: list, ctx, kv_cache, latents,
                    forwards: dict, first_step: int = 0) -> None:
    """Profile stages (CausVid, RAVEN, LongLive 1.0): segment prompt embeddings per block, the prompt-switch recache
    of every cache epoch on the path (``run_prompt_switch_recache`` without capture), cross-attention resets where the
    prompt changes. W1.4.1 (N2): steps before ``first_step`` are installed already (the schedule still spans every
    step; the fresh cross-attention caches compute their first replayed step's prompt like a reset)."""
    from .lifecycle import recache_matches, replay_schedule

    segments = batch.extra.get(SEGMENTS_KEY) or {"switch_chunks": [], "count": 1}
    count = segments["count"]
    schedule = replay_schedule(steps, list(segments["switch_chunks"]))
    cross = _private_cross_caches(stage, server_args, ctx)
    chunk = stage.profile.chunk_latent_frames
    for index, (step, item) in enumerate(zip(steps, schedule)):
        if index < first_step:
            continue
        if step["kind"] == "chunk":
            _check_rope_start(stage, step)
            start, frames = int(step["start_frame"]), int(step["num_frames"])
            if item["reset_cross_attention"]:
                stage._reset_crossattn_cache(cross)
            stage._update_causal_context_cache(
                batch, server_args, context_input=latents[:, :, start:start + frames],
                prompt_embeds=stage._segment_embeds(ctx.prompt_embeds, item["segment"], count), kv_cache=kv_cache,
                crossattn_cache=cross, current_start_tokens=int(step["token_start"]), start_frame=start,
                image_kwargs=ctx.image_kwargs, pos_cond_kwargs=ctx.pos_cond_kwargs, attn_metadata=None,
                target_dtype=ctx.target_dtype, autocast_enabled=ctx.autocast_enabled)
            forwards["chunk"] += 1
        else:
            block = item["precedes_block"]
            record = stage.run_prompt_switch_recache(
                batch, server_args, ctx=ctx, kv_cache=kv_cache, crossattn_cache=cross, latents=latents,
                current_start_frame=block * chunk,
                new_prompt_embeds=stage._segment_embeds(ctx.prompt_embeds, item["new_segment"], count),
                switch_before_chunk=block, previous_segment=item["previous_segment"],
                new_segment=item["new_segment"], kv_run=None)
            if record is None or not recache_matches(step, record.plan):
                raise RuntimeError(f"the replayed recache before block {block} does not re-materialize the archived "
                                   f"epoch {step.get('epoch')!r} (window {step.get('window')}, written "
                                   f"{step.get('written')})")
            forwards["recache"] += 1


def _replay_blocks(stage, batch: Req, server_args: ServerArgs, *, steps: list, ctx, kv_cache, kv_cache_neg, latents,
                   use_cfg: bool, forwards: dict, first_step: int = 0) -> None:
    """SGLang's causal stages (LongLive 2.0): the native loop's per-block conditioning (block prompts, shot index RoPE
    offset, cross-attention reset per block prompt), both CFG streams, scene-cut chunks pinned after their forward.
    W1.4.1 (N2): steps before ``first_step`` are installed already (with their pinned shot sink)."""
    cross = _private_cross_caches(stage, server_args, ctx)
    cross_neg = _private_cross_caches(stage, server_args, ctx) if use_cfg else None
    negative = stage._get_negative_prompt_embeds(batch) if use_cfg else None
    neg_cond = stage._prepare_causal_dmd_neg_cond_kwargs(batch, server_args, ctx.target_dtype) if use_cfg else None
    set_offset = getattr(stage, "_set_rope_temporal_offset", None)
    try:
        for step in steps[first_step:]:
            if step["kind"] != "chunk":
                raise NotImplementedError(f"{type(stage).__name__}: cache-epoch (recache) replay runs on the profile "
                                          "stages only")
            block, start, frames = int(step["block"]), int(step["start_frame"]), int(step["num_frames"])
            if callable(set_offset):
                set_offset(batch, stage._shot_index(batch, block))
            _check_rope_start(stage, step)
            scene_cut = bool(stage._is_scene_cut(batch, block))
            if scene_cut != bool(step["scene_cut"]):
                raise RuntimeError(f"replay of block {block}: scene cut {scene_cut} != the archived {step['scene_cut']}")
            stage._reset_crossattn_cache_for_block(batch, *([cross] + ([cross_neg] if use_cfg else [])))
            common = dict(context_input=latents[:, :, start:start + frames], current_start_tokens=int(step["token_start"]),
                          start_frame=start, image_kwargs=ctx.image_kwargs, attn_metadata=None,
                          target_dtype=ctx.target_dtype, autocast_enabled=ctx.autocast_enabled)
            stage._update_causal_context_cache(
                batch, server_args, prompt_embeds=stage._select_block_prompt_embeds(batch, ctx.prompt_embeds, block),
                kv_cache=kv_cache, crossattn_cache=cross,
                pos_cond_kwargs=stage._select_block_cond_kwargs(batch, ctx.pos_cond_kwargs, block), **common)
            if use_cfg:
                stage._update_causal_context_cache(
                    batch, server_args, prompt_embeds=stage._select_block_prompt_embeds(batch, negative, block),
                    kv_cache=kv_cache_neg, crossattn_cache=cross_neg,
                    pos_cond_kwargs=stage._select_block_cond_kwargs(batch, neg_cond, block), **common)
            if scene_cut:
                stage._pin_current_chunk(kv_cache, frames)
                if use_cfg:
                    stage._pin_current_chunk(kv_cache_neg, frames)
            forwards["chunk"] += 1
    finally:
        if hasattr(stage, "_rope_temporal_offset"):
            stage._rope_temporal_offset = 0.0  # the loop sets each new block's own offset


def _before_device_sync(kv_run, where: str) -> None:
    """W1.4.1 (N1): before a device-wide synchronize of the stage, a replay-promotion copy still running is waited
    for explicitly and recorded by the bridge run (``promotion_wait``) -- the sync would absorb it unseen. The sync
    itself is unchanged; with no pending copy (every default run) this does nothing."""
    hook = getattr(kv_run, "before_device_sync", None)
    if hook is not None:
        hook(where)


@torch.no_grad()
def replay_causal_history(stage, batch: Req, server_args: ServerArgs, *, run, ctx, kv_cache, latents,
                          kv_cache_neg=None, use_cfg: bool = False) -> dict:
    """Replay ``run.replay_plan`` into the stage's reset caches (see the section comment); returns the restore record
    the bridge completed (``recompute_replay_seconds``: CUDA-synchronized wall time of the replay)."""
    if getattr(stage, "_causal_kv_pending", None) is not None:
        raise RuntimeError("a recompute replay must not run inside a capture scope")
    plan = run.begin_replay()
    device = latents.device
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    observer = batch.extra.pop(OBSERVER_KEY, None)  # diagnostics hooks see generation, not the replay
    forwards = {"chunk": 0, "recache": 0}
    synchronize()
    started = time.perf_counter()
    try:
        first = int(plan.get("start_step", 0))  # W1.4.1 (N2): the installed ancestor's steps are skipped
        if isinstance(stage, ProfileCausalDenoisingStage):
            _replay_profile(stage, batch, server_args, steps=plan["steps"], ctx=ctx, kv_cache=kv_cache,
                            latents=latents, forwards=forwards, first_step=first)
        else:
            _replay_blocks(stage, batch, server_args, steps=plan["steps"], ctx=ctx, kv_cache=kv_cache,
                           kv_cache_neg=kv_cache_neg, latents=latents, use_cfg=use_cfg, forwards=forwards,
                           first_step=first)
        synchronize()
    finally:
        if observer is not None:
            batch.extra[OBSERVER_KEY] = observer
    seconds = time.perf_counter() - started
    return run.end_replay(seconds=seconds, details={"forwards": forwards, "stage": type(stage).__name__,
                                                    "synchronized": device.type == "cuda"})


def _block_prompts(stage, batch: Req, num_blocks: int) -> list | None:
    """Per-block prompt texts of the request (node sidecars of session trees); None when unknown."""
    if isinstance(stage, ProfileCausalDenoisingStage):
        from .lifecycle import block_prompt_texts

        return block_prompt_texts(batch.prompt, normalize_prompt_switches(getattr(batch, "prompt_switches", None)),
                                  num_blocks)
    prompts = batch.extra.get("causal_block_prompts")  # SGLang's CAUSAL_BLOCK_PROMPTS_KEY (multi-shot)
    if isinstance(prompts, list) and len(prompts) == num_blocks:
        return [str(item) for item in prompts]
    return [batch.prompt] * num_blocks if isinstance(batch.prompt, str) else None


def after_causal_kv_begin(stage, batch: Req, server_args: ServerArgs, run, kwargs: dict) -> None:
    """After SGLang's ``_begin_causal_kv_run`` (the bridge restored): hand the run the per-block prompts and run a
    pending recompute replay. A failing replay finishes the run with the error (the stage's run scope is not open
    yet) and raises."""
    if run is None:
        return
    try:
        prompts = _block_prompts(stage, batch, len(kwargs.get("block_sizes") or ()))
        if prompts is not None:
            run.block_prompts = prompts
    except Exception:  # noqa: BLE001 - sidecar provenance only
        pass
    if getattr(run, "replay_plan", None) is None:
        return
    try:
        replay_causal_history(stage, batch, server_args, run=run, ctx=kwargs["ctx"], kv_cache=kwargs["kv_cache"],
                              latents=kwargs["latents"], kv_cache_neg=kwargs.get("kv_cache_neg"),
                              use_cfg=bool(kwargs.get("use_cfg")))
    except BaseException as exc:
        run.finish(error=exc)
        raise


def install_replay_seam(stage) -> bool:
    """For causal stages ReelCache does not own (SGLang's LongLive 2.0 stage): wrap the INSTANCE's
    ``_begin_causal_kv_run`` with :func:`after_causal_kv_begin` (recompute replay + per-block prompts). Returns False
    when the stage has no seam or is wrapped already. Process-local; without it a recompute replay on such a stage
    fails at its first attention access (the bridge's guard), never silently."""
    original = getattr(stage, "_begin_causal_kv_run", None)
    if original is None or getattr(original, "_reelcache_replay_seam", False):
        return False

    def begin(batch, server_args, **kwargs):
        run, resume = original(batch, server_args, **kwargs)
        after_causal_kv_begin(stage, batch, server_args, run, kwargs)
        return run, resume

    begin._reelcache_replay_seam = True
    stage._begin_causal_kv_run = begin
    return True


def install_replay_seams(pipeline) -> list[str]:
    """The runner's call (coordinator decision 2026-10-19: the runner installs the replay seams): wrap every causal
    stage of ``pipeline`` (an SGLang ``ComposedPipelineBase``, a worker exposing ``pipeline``, or an iterable of
    stages) whose seam still lacks the replay hook (:func:`install_replay_seam`; ``lifecycle.needs_replay_seam``) --
    SGLang's own causal stages (LongLive 2.0, any model ``CausalDMDDenoisingStage`` serves); ReelCache's profile stages
    carry it on their class. Returns the registered names of the stages wrapped now (``[]`` when called again:
    idempotent). A recompute request on a stage without it keeps failing at its first attention access."""
    from .lifecycle import needs_replay_seam, pipeline_stages

    installed = []
    for stage in pipeline_stages(pipeline):
        if needs_replay_seam(stage) and install_replay_seam(stage):
            installed.append(getattr(stage, "_registered_stage_name", None) or type(stage).__name__)
    return installed
