"""SGLang stages of the LongCat-Video direct-720p continuation pipeline (imports SGLang).

Stage chain (one request = one or more continuation segments, batch size 1, no CFG):

1. :class:`LongCatVideoInputStage` -- conditioning frames, LongCat bucket geometry, request generator.
2. :class:`LongCatVideoTextEncodingStage` -- native UMT5 conditioning (prompt_clean, 512 tokens + mask)
   on SGLang's UMT5 encoder.
3. :class:`LongCatVideoContinuationStage` -- per segment: noise draw and VAE posterior sample of the last
   13 RGB frames (same generator order as native ``prepare_latents``), the conditioning-cache build
   (timestep 0, cross-attention skipped, pre-RoPE K + V of 48 layers) or a restore through the
   :mod:`.conditioning_cache` seam, 16 distilled Euler steps reading the cache, native decode tail.
   The next segment conditions on the decoded last 13 frames and builds a new cache epoch.

A request may start at a segment boundary (``start_segment_index`` > 0, the boundary's conditioning frames as
input, and the ``generator_state`` recorded at that segment's start): resume/rewind replays the segment exactly,
and an armed conditioning-cache backend restores the segment's archived epoch instead of rebuilding it.

Context parallelism (``sp_degree = ulysses_degree = N``): every rank runs this stage chain (SPMD); as in native
``generate_vc`` the prompt embeddings and each segment's prepared latents are broadcast from CP rank 0, the DiT
runs native LongCat CP (:mod:`.context_parallel`) and every rank steps the scheduler and decodes the full latents.
Each rank's conditioning cache holds its token rows only: the epoch carries the rank's ``RowShard`` and the backend
seam captures/installs that rank's rows (per-rank seam of :mod:`.conditioning_cache`).
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import ComponentUse
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs

from .conditioning_cache import (
    SOURCE_NATIVE_BUILD,
    ConditioningCacheEpoch,
    LongCatConditioningCache,
    accept_restored_cache,
    get_cache_backend,
    tensor_sha256,
)
from .schedule import continuation_schedule, distill_sigmas, dit_timestep
from .segment_resume import generator_state_hex, generator_state_sha256, set_generator_state_hex
from .spec import MAX_SEQUENCE_LENGTH, ContinuationGeometry, DistilledSampler, bucket_for_condition
from .text import prompt_clean, tokenizer_kwargs
from .video_io import denormalize_latents, frames_to_vae_input, load_condition_frames, normalize_latents, postprocess_decoded
from .weights import MODEL_FINGERPRINT

STATE_KEY = "longcat_video"
STEP_OBSERVER_KEY = "longcat_step_observer"


@contextlib.contextmanager
def native_precision_flags():
    """Native LongCatRuntime disables TF32 for matmul and cuDNN; keep that during LongCat compute."""
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextlib.contextmanager
def _generation_section(backend, rec: dict):
    """The memory-critical section around denoising / decoding (a GPU archive pipeline defers its planning there when
    configured): the request's cache backend's and, when that is another backend (a plain request), the worker's
    session provider's (``session_provider.memory_critical_section``; CR-17-6). The time spent entering it (the
    fitter finishing its current node) is recorded."""
    from .session_provider import memory_critical_section

    section = memory_critical_section(backend)
    if section is None:
        yield
        return
    started = time.perf_counter()
    with section:
        rec["planning_deferral_wait_seconds"] = rec.get("planning_deferral_wait_seconds", 0.0) + (
            time.perf_counter() - started)
        yield


def _cache_released(backend, epoch) -> None:
    """The segment dropped its conditioning cache (after denoising): a backend with raw tiers (freeze D) turns its
    retained GPU copy into ``gpu_raw`` under the GPU raw budget (A100-80GB: 0, so it is freed now)."""
    hook = getattr(backend, "cache_released", None) if backend is not None else None
    if hook is not None:
        hook(epoch)


@dataclass
class LongCatVideoRequestState:
    """Request-local state (never shared through pipeline stages)."""

    geometry: ContinuationGeometry
    sampler: DistilledSampler
    resolution: str
    bucket_key: str
    cond_frames: np.ndarray
    generator: torch.Generator
    seed: int
    num_segments: int
    session_id: str | None = None
    cache_backend_name: str | None = None
    requested_hw: tuple[int | None, int | None] = (None, None)
    prompt_embeds: torch.Tensor | None = None
    prompt_attention_mask: torch.Tensor | None = None
    injected: dict = field(default_factory=dict)
    segments: list[dict] = field(default_factory=list)
    start_segment: int = 0



def request_state(batch: Req) -> LongCatVideoRequestState:
    state = batch.extra.get(STATE_KEY)
    if not isinstance(state, LongCatVideoRequestState):
        raise RuntimeError("LongCat-Video request state missing; LongCatVideoInputStage must run first")
    return state


def _single(value, name: str):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise NotImplementedError(f"LongCat-Video port serves one sample per request; got {len(value)} {name}")
        return value[0]
    return value


class LongCatVideoInputStage(PipelineStage):
    """Validate the request and build its conditioning frames, geometry and generator."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        holder: dict = {}
        try:
            return self._forward(batch, server_args, holder)
        except BaseException as exc:
            run = holder.get("run")
            if run is not None and not run.finished:  # the session is idle again; committed boundaries stay
                run.finish(error=exc)
            raise

    def _forward(self, batch: Req, server_args: ServerArgs, holder: dict) -> Req:
        if int(getattr(batch, "num_outputs_per_prompt", 1) or 1) != 1:
            raise NotImplementedError("num_outputs_per_prompt must be 1")
        prompt = _single(batch.prompt, "prompts")
        if not isinstance(prompt, str):
            raise ValueError("LongCat-Video continuation needs a text prompt")
        sampler = DistilledSampler(num_inference_steps=int(batch.num_inference_steps or 16),
                                   guidance_scale=float(batch.guidance_scale))
        resolution = str(getattr(batch, "resolution", "720p"))
        num_cond = int(getattr(batch, "num_cond_frames", 13))
        num_segments = int(getattr(batch, "num_segments", 1))
        if num_segments < 1:
            raise ValueError("num_segments must be >= 1")
        run = None
        handle = getattr(batch, "causal_kv_session", None)
        if handle is not None:
            # ReelCache session request (Step 14): the session provider plans it; resume/fork/rewind take the boundary
            # conditioning (frames + generator state) from the session archive, never from the request.
            from .session_provider import active_provider, request_fields

            provider = active_provider()
            if provider is None:
                raise RuntimeError("causal_kv_session requires a server started with --reelcache-enable (the LongCat "
                                   "session provider is not attached to this worker)")
            run = provider.begin_request(request_fields(batch), handle)
            holder["run"] = run
            num_segments = run.plan.num_segments
        if run is not None and run.plan.frames is not None:
            frames = run.plan.frames
        else:
            source = (batch.condition_inputs or {}).get("cond_frames")
            if source is None:
                source = _single(batch.video_path, "video paths")
                if source is None:
                    raise ValueError("LongCat-Video continuation needs conditioning frames (video_path or "
                                     "condition_inputs['cond_frames'])")
            frames = load_condition_frames(source)
        if frames.shape[0] < num_cond:
            raise ValueError(f"need at least {num_cond} conditioning frames, got {frames.shape[0]}")
        key, height, width = bucket_for_condition(int(frames.shape[1]), int(frames.shape[2]), resolution)
        requested = (batch.height, batch.width)
        batch.height, batch.width = height, width
        geometry = ContinuationGeometry(height=height, width=width, num_frames=int(batch.num_frames),
                                        num_cond_frames=num_cond)
        seed = int(_single(batch.seed, "seeds"))
        device = get_local_torch_device()
        generator = torch.Generator(device=device).manual_seed(seed)
        start = getattr(batch, "start_segment_index", 0)
        start = 0 if start is None else int(start)
        if start < 0:
            raise ValueError("start_segment_index must be >= 0")
        resume_state = getattr(batch, "generator_state", None)
        if run is not None:
            start, resume_state = run.plan.start_segment, run.plan.generator_state
        resume = None
        if start or resume_state is not None:
            # Resume/rewind at a segment boundary: the request's frames are that boundary's conditioning frames
            # (the previous segment's last decoded frames); the recorded generator state reproduces the noise.
            if resume_state is not None:
                set_generator_state_hex(generator, resume_state)
            resume = {"start_segment_index": start, "generator_state_restored": resume_state is not None,
                      "generator_state_sha256": None if resume_state is None
                      else generator_state_sha256(resume_state)}
        batch.generator = generator
        cond_frames = np.ascontiguousarray(frames[-num_cond:])
        session_id = getattr(batch, "session_id", None)
        cache_backend_name = getattr(batch, "conditioning_cache_backend", None)
        if run is not None:
            from .session_provider import CACHE_BACKEND_NAME, SESSION_RUN_KEY

            # generate: boundary 0 (the request's frames, the seed's generator state) creates the session;
            # resume/fork/rewind: the request must match the boundary's geometry
            run.bind_request(frames=cond_frames, generator_state=generator_state_hex(generator),
                             geometry={"height": height, "width": width, "num_frames": int(batch.num_frames),
                                       "num_cond_frames": num_cond, "bucket": key, "resolution": resolution})
            session_id, cache_backend_name = run.plan.session_id, CACHE_BACKEND_NAME
            batch.extra[SESSION_RUN_KEY] = run
        batch.extra[STATE_KEY] = LongCatVideoRequestState(
            geometry=geometry, sampler=sampler, resolution=resolution, bucket_key=key,
            cond_frames=cond_frames, generator=generator, seed=seed,
            num_segments=num_segments, session_id=session_id,
            cache_backend_name=cache_backend_name, requested_hw=requested,
            start_segment=start)
        if resume is not None:
            batch.extra[STATE_KEY].injected["resume"] = resume
        if run is not None:
            batch.extra[STATE_KEY].injected["session"] = {"session_id": run.plan.session_id, "action": run.plan.action,
                                                          "start_segment": start, "request_id": run.request_id}
        self.log_info("bucket %s %dx%d, %d segment(s) from segment %d, seed %d", key, height, width, num_segments,
                      start, seed)
        return batch


class LongCatVideoTextEncodingStage(PipelineStage):
    """Native UMT5 prompt conditioning (no CFG: guidance 1.0) on SGLang's text encoder."""

    def __init__(self, text_encoder, tokenizer, prompt_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.prompt_dtype = prompt_dtype  # native: encode_prompt(dtype=dit.dtype)

    def component_uses(self, server_args: ServerArgs, stage_name: str | None = None) -> list[ComponentUse]:
        return [ComponentUse(stage_name=self._component_stage_name(stage_name), component_name="text_encoder",
                             preferred_ready_after_request=True, start_at_stage_entry=False)]

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        state = request_state(batch)
        device = get_local_torch_device()
        inputs = batch.condition_inputs or {}
        if inputs.get("prompt_embeds") is not None:
            pe = inputs["prompt_embeds"].to(device=device, dtype=self.prompt_dtype)
            mask = inputs.get("prompt_attention_mask")
            if mask is None:
                raise ValueError("injected prompt_embeds need prompt_attention_mask")
            mask = mask.to(device)
            state.injected["prompt_embeds"] = tensor_sha256(pe)
        else:
            prompt = prompt_clean(_single(batch.prompt, "prompts"))
            tok = self.tokenizer([prompt], **tokenizer_kwargs(MAX_SEQUENCE_LENGTH))
            ids = tok.input_ids.to(device)
            mask = tok.attention_mask.to(device)
            with self.use_declared_component(component_name="text_encoder", module=self.text_encoder) as encoder:
                with set_forward_context(current_timestep=0, attn_metadata=None):
                    out = encoder(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            hidden = out.last_hidden_state
            pe = hidden.to(dtype=self.prompt_dtype, device=device)
            pe = pe.view(pe.shape[0], 1, pe.shape[1], -1)
        if pe.ndim != 4 or pe.shape[2] != MAX_SEQUENCE_LENGTH or tuple(mask.shape) != (pe.shape[0], pe.shape[2]):
            raise ValueError(f"prompt embeddings {tuple(pe.shape)} / mask {tuple(mask.shape)} not [B,1,512,C]/[B,512]")
        state.prompt_embeds, state.prompt_attention_mask = pe, mask
        batch.prompt_embeds = [pe]
        batch.prompt_attention_mask = [mask]
        return batch


class LongCatVideoContinuationStage(PipelineStage):
    """Segment loop: latents, conditioning cache (build or restore), denoising, decode."""

    def __init__(self, transformer, vae, scheduler) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def component_uses(self, server_args: ServerArgs, stage_name: str | None = None) -> list[ComponentUse]:
        name = self._component_stage_name(stage_name)
        return [ComponentUse(stage_name=name, component_name="vae", phase="encode"),
                ComponentUse(stage_name=name, component_name="transformer", memory_intensive=True),
                ComponentUse(stage_name=name, component_name="vae", phase="decode")]

    # -- per-segment pieces --------------------------------------------------------------------------
    def _prepare_latents(self, batch: Req, state: LongCatVideoRequestState, frames: np.ndarray, seg: int,
                         rec: dict, device: torch.device, dtype: torch.dtype, cp=None):
        geo = state.geometry
        inputs = batch.condition_inputs or {}
        n_cond = geo.cond_latent_frames
        if seg == 0 and inputs.get("initial_latents") is not None:
            latents = inputs["initial_latents"].to(device=device, dtype=dtype).clone()
            if tuple(latents.shape) != geo.latent_shape:
                raise ValueError(f"initial_latents {tuple(latents.shape)} != {geo.latent_shape}")
            if cp is not None:
                cp.broadcast(latents)
                rec["latents_broadcast_from_cp_rank0"] = True
            state.injected["initial_latents"] = tensor_sha256(latents)
            rec["latents_source"] = "injected_initial_latents"
            return latents, latents[:, :, :n_cond]
        _sync(device)
        t0 = time.perf_counter()
        latents = torch.randn(geo.latent_shape, generator=state.generator, device=device, dtype=dtype)
        if seg == 0 and inputs.get("cond_latents") is not None:
            cond = inputs["cond_latents"].to(device=device, dtype=dtype)
            state.injected["cond_latents"] = tensor_sha256(cond)
            rec["latents_source"] = "noise_drawn_cond_latents_injected"
        else:
            video = frames_to_vae_input(frames, geo.height, geo.width).to(device=device, dtype=dtype)
            with self.use_declared_component(component_name="vae", module=self.vae, phase="encode") as vae:
                posterior = vae.encode(video)
                dist = posterior.latent_dist if hasattr(posterior, "latent_dist") else posterior
                sample = dist.sample(state.generator)
            cond = normalize_latents(sample.to(dtype))
            rec["latents_source"] = "native_noise_then_vae_posterior_sample"
        if tuple(cond.shape) != (1, latents.shape[1], n_cond, *latents.shape[3:]):
            raise ValueError(f"conditioning latents {tuple(cond.shape)} do not match {geo.latent_shape}")
        latents[:, :, :n_cond] = cond
        if cp is not None:
            cp.broadcast(latents)  # native generate_vc: cp_broadcast(latents) after prepare_latents
            rec["latents_broadcast_from_cp_rank0"] = True
        _sync(device)
        rec["latent_prep_seconds"] = time.perf_counter() - t0
        return latents, latents[:, :, :n_cond]

    def _conditioning_cache(self, batch: Req, state: LongCatVideoRequestState, cond: torch.Tensor, seg: int,
                            backend, rec: dict, device: torch.device, dtype: torch.dtype,
                            cp=None) -> LongCatConditioningCache:
        geo = state.geometry
        arch = self.transformer.longcat_arch
        epoch = ConditioningCacheEpoch(segment_index=seg, token_grid=geo.cond_token_grid,
                                       cond_latents_sha256=tensor_sha256(cond), model_fingerprint=MODEL_FINGERPRINT,
                                       session_id=state.session_id, parent_segment_index=seg - 1 if seg else None,
                                       row_shard=None if cp is None else cp.row_shard())
        cache = None
        if backend is not None:
            _sync(device)
            t0 = time.perf_counter()
            restored = backend.restore(epoch)
            if restored is not None:
                cache = accept_restored_cache(restored, epoch, layers=arch.depth, heads=arch.num_heads,
                                              head_dim=arch.head_dim, dtype=dtype, device=device)
                if cache.ready is not None:
                    # overlapped restore: the layers are still being written; the first denoising step waits per
                    # layer (layer-ahead) instead of the stage waiting for the whole cache here
                    rec["cache_restore_overlapped"] = True
                    rec["cache_restore_issue_seconds"] = time.perf_counter() - t0
                else:
                    _sync(device)
                    rec["cache_restore_seconds"] = time.perf_counter() - t0
        if cache is None:
            _sync(device)
            t0 = time.perf_counter()
            timestep0 = torch.zeros(cond.shape[0], cond.shape[2]).to(device=cond.device, dtype=dtype)
            with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=batch):
                # native _cache_clean_latents: timestep 0, cross-attention skipped, pre-RoPE K/V returned
                keys, values = self.transformer(cond, None, timestep0, return_kv=True, skip_crs_attn=True)
            _sync(device)
            build_s = time.perf_counter() - t0
            cache = LongCatConditioningCache(epoch=epoch, keys_pre_rope=keys, values=values,
                                             source=SOURCE_NATIVE_BUILD,
                                             metadata={"build_seconds": build_s,
                                                       "segment_generator_state": rec.get("generator_state")},
                                             cond_latents=cond)
            cache.validate(layers=arch.depth, heads=arch.num_heads, head_dim=arch.head_dim, dtype=dtype, device=device)
            rec["cache_build_seconds"] = build_s
            if backend is not None:
                t1 = time.perf_counter()
                backend.on_cache_built(cache)
                _sync(device)
                rec["cache_observer_seconds"] = time.perf_counter() - t1
        rec["epoch"] = epoch.as_dict()
        rec["cache_source"] = cache.source
        rec["cache_bytes"] = cache.nbytes
        return cache

    def _denoise(self, batch: Req, state: LongCatVideoRequestState, cache: LongCatConditioningCache,
                 latents: torch.Tensor, seg: int, rec: dict, device: torch.device, dtype: torch.dtype,
                 t_cache: float | None = None):
        sampler = state.sampler
        steps = sampler.num_inference_steps
        scheduler = copy.deepcopy(self.scheduler)
        cfg = scheduler.config
        if (float(scheduler.shift) != sampler.flow_shift or float(cfg.shift) != sampler.flow_shift
                or cfg.use_dynamic_shifting or cfg.invert_sigmas):
            raise RuntimeError(f"scheduler config {dict(cfg)} is not the LongCat distilled schedule")
        scheduler.set_timesteps(steps, sigmas=distill_sigmas(steps, sampler.num_train_timesteps,
                                                             sampler.num_distill_sample_steps), device=device)
        ref_t, ref_s = continuation_schedule(steps, sampler.flow_shift, sampler.num_train_timesteps,
                                             sampler.num_distill_sample_steps, device=device)
        if not (torch.equal(scheduler.timesteps, ref_t) and torch.equal(scheduler.sigmas, ref_s)):
            raise RuntimeError("SGLang scheduler timesteps/sigmas differ from the native distilled schedule")
        scheduler.set_begin_index(0)
        observer = batch.extra.get(STEP_OBSERVER_KEY)
        kv_layers = list(zip(cache.keys_pre_rope, cache.values))
        n_cond = state.geometry.cond_latent_frames
        step_seconds = []
        for i, t in enumerate(scheduler.timesteps):
            _sync(device)
            t0 = time.perf_counter()
            latent_model_input = latents.to(dtype)
            timestep = dit_timestep(t, latent_model_input.shape[0], latent_model_input.shape[2], dtype)
            ready = cache.ready if i == 0 else None  # an overlapped restore completes within the first step
            with set_forward_context(current_timestep=i, attn_metadata=None, forward_batch=batch):
                noise_pred = self.transformer(latent_model_input, state.prompt_embeds, timestep,
                                              encoder_attention_mask=state.prompt_attention_mask,
                                              num_cond_latents=n_cond, kv_cache_layers=kv_layers,
                                              kv_cache_ready=ready)
            noise_pred = -noise_pred
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            _sync(device)
            step_seconds.append(time.perf_counter() - t0)
            if i == 0:
                if t_cache is not None:
                    rec["time_to_step0_end_seconds"] = time.perf_counter() - t_cache
                if ready is not None:
                    joined = ready.join()  # re-raises a restore failure
                    rec["overlapped_restore"] = joined
                    rec["cache_restore_exposed_seconds"] = (joined.get("consumer_waits") or {}).get("seconds")
                    cache.ready = None
            if observer is not None:
                observer(segment=seg, step=i, timestep=t, model_output=noise_pred, latents=latents)
        rec["timesteps"] = [float(x) for x in scheduler.timesteps.tolist()]
        rec["step_seconds"] = step_seconds
        return latents

    def _decode(self, latents: torch.Tensor, rec: dict, device: torch.device) -> torch.Tensor:
        _sync(device)
        t0 = time.perf_counter()
        with self.use_declared_component(component_name="vae", module=self.vae, phase="decode") as vae:
            vae_dtype = next(vae.parameters()).dtype
            lat = denormalize_latents(latents.to(vae_dtype))
            video = vae.decode(lat)
            video = video.sample if hasattr(video, "sample") else video
        frames01 = postprocess_decoded(video.to(vae_dtype))
        _sync(device)
        rec["decode_seconds"] = time.perf_counter() - t0
        return frames01

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        state = request_state(batch)
        if state.prompt_embeds is None:
            raise RuntimeError("prompt embeddings missing; LongCatVideoTextEncodingStage must run first")
        device = get_local_torch_device()
        dtype = self.transformer.compute_dtype
        geo = state.geometry
        backend = get_cache_backend(state.cache_backend_name) if state.cache_backend_name else None
        cp = self.transformer.context_parallel()
        if cp is not None:
            # native generate_vc: the prompt is encoded on CP rank 0 and broadcast to the group
            cp.broadcast(state.prompt_embeds)
            cp.broadcast(state.prompt_attention_mask)
            state.injected["context_parallel"] = {**cp.describe(), "prompt_broadcast_from_cp_rank0": True,
                                                  "row_shard": cp.row_shard().as_dict()}
        frames = state.cond_frames
        pieces = []
        from .session_provider import SESSION_RUN_KEY

        run = batch.extra.get(SESSION_RUN_KEY)
        if run is not None and cp is not None:
            run.finish(error=RuntimeError("context-parallel session request"))
            raise RuntimeError("LongCat sessions run on one GPU (per-rank sessions belong to the factor-sharded "
                               "Step 16 integration)")
        error = None
        try:
            self._segments(batch, state, frames, pieces, backend, device, dtype, geo, cp, run)
        except BaseException as exc:  # noqa: BLE001 - the session run records it; re-raised
            error = exc
            raise
        finally:
            if run is not None:
                state.injected["session"] = {**state.injected.get("session", {}),
                                             **{key: value for key, value in run.finish(error=error).items()
                                                if key in ("generated_segments", "head", "paused_by", "durable",
                                                           "archival")}}
        video = torch.cat(pieces, dim=1).permute(0, 4, 1, 2, 3).contiguous()
        return OutputBatch(output=video, metrics=batch.metrics, usage=batch.usage)

    def _segments(self, batch: Req, state: LongCatVideoRequestState, frames: np.ndarray, pieces: list, backend,
                  device: torch.device, dtype: torch.dtype, geo, cp, run) -> None:
        with native_precision_flags(), torch.no_grad():
            for seg in range(state.start_segment, state.start_segment + state.num_segments):
                rec = {"segment_index": seg,
                       "cond_frames_sha256": hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest(),
                       "generator_state": generator_state_hex(state.generator)}
                latents, cond = self._prepare_latents(batch, state, frames, seg, rec, device, dtype, cp)
                with self.use_declared_component(component_name="transformer", module=self.transformer):
                    _sync(device)
                    t_cache = time.perf_counter()  # cache build/restore + step 0 = time to the first new step
                    cache = self._conditioning_cache(batch, state, cond, seg, backend, rec, device, dtype, cp)
                    with _generation_section(backend, rec):
                        new = self._denoise(batch, state, cache, latents[:, :, geo.cond_latent_frames:], seg, rec,
                                            device, dtype, t_cache)
                released = cache.epoch
                del cache
                _cache_released(backend, released)
                full = torch.cat([cond, new], dim=2)
                rec["final_latents_sha256"] = tensor_sha256(full)
                with _generation_section(backend, rec):
                    frames01 = self._decode(full, rec, device)
                frames_u8 = (frames01 * 255).to(torch.uint8)  # native (x * 255).astype(uint8)
                rec["frames_uint8_sha256"] = tensor_sha256(frames_u8[0])
                pieces.append((frames01 if seg == 0 else frames01[:, geo.num_cond_frames:]).cpu())
                frames = frames_u8[0, -geo.num_cond_frames:].cpu().numpy()
                state.segments.append(rec)
                del latents, cond, new, full, frames01, frames_u8
                if run is not None and run.end_segment(segment_index=seg, record=rec, next_frames=frames,
                                                       generator_state=generator_state_hex(state.generator),
                                                       prompt=_single(batch.prompt, "prompts")):
                    break
