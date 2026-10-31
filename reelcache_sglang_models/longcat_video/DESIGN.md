# LongCat-Video in SGLang Diffusion: design note (Step 13 `sglang-longcat-port`, phases 1-4)

Scope: the native **direct 720p video continuation** of LongCat-Video 13.6B (commit `6b3f4b8`, HF revision
`03b5552`), i.e. `LongCatVideoPipeline.generate_vc(use_distill=True, enhance_hf=False)` with `cfg_step_lora`,
16 steps, guidance 1.0, 93 frames per segment with 13 conditioning frames, bucket `0.54` = 704x1312. This is
the configuration of the M06 720p grid, the energy sweeps and the feasibility rebuild baseline. Out of scope
(rejected explicitly, never silently substituted): the 50-step CFG sampler and `enhance_hf`, T2V/I2V entry
points (the DiT supports their forward, the pipeline does not expose them), the 480p->720p refinement path
(`refinement_lora`, block-sparse attention), tensor parallelism, ring / K-V-gather sequence parallelism, CFG
parallel, batches > 1. Native context parallelism (Ulysses, 2-D spatial split) was added in phase 4 (section 10).

Package: `ReelCache/reelcache_sglang_models/` (external SGLang model package, loaded through
`SGLANG_EXTERNAL_MODEL_PACKAGE=reelcache_sglang_models`). Core `reelcache` never imports it.

## 1. Native workload (what must be preserved)

Per segment (`pipeline_longcat_video.py:837-1095`, `modules/attention.py`, `modules/longcat_video_dit.py`):

1. `prepare_latents`: draw noise `torch.randn((1,16,24,88,164), generator, cuda, bf16)` **first**, then VAE-encode
   the last 13 RGB frames (`[-1,1]`, BF16) and draw the posterior sample **with the same generator**;
   normalize `(z - mean) * (1/std)` in BF16; overwrite latent frames 0-3 (4 conditioning latents).
2. `_cache_clean_latents`: one DiT forward over the 4 conditioning latents (14,432 tokens) at timestep 0 with
   cross-attention skipped; every layer returns `(k, v)` where **k is cloned after `k_norm` and before
   `rope_3d`** (pre-RoPE) and v is the raw value; 48 layers x 2 x `[1, 32, 14432, 128]` BF16 = 11.35 GB.
   The cache is prompt-independent.
3. 16 Euler steps over the 20 new latents: DiT input BF16, timestep `t.to(bf16)` per frame; every layer
   concatenates cache + current pre-RoPE K/V and applies 3D RoPE to the concatenation over the
   `(4 + 20, 44, 82)` grid (segment-local positions: cache at t=0..3, new latents at t=4..23); text
   cross-attention for the new tokens only; `noise_pred` FP32, negated, scheduler step in FP32.
4. Concatenate cache latents + new latents, decode all 24 latents (93 frames), uint8 by truncation.
5. Next segment: condition on the last 13 decoded frames, build a **new** cache (no cross-segment append),
   same generator.

Dtype flow: BF16 weights everywhere (LoRA cast to BF16 too); FP32 regions under native
`autocast('cuda', float32)`: timestep MLP, adaLN modulation, final linear; LayerNorm/RMSNorm in FP32 and
cast back; gated residuals formed in FP32 and cast to BF16; RoPE computed in FP32. TF32 off.

## 2. Mapping to SGLang components

| Native piece | SGLang port | Reused / new |
|---|---|---|
| `AutoTokenizer` (`tokenizer/`) | `TokenizerLoader` (same transformers class, right padding) | reused |
| HF `UMT5EncoderModel` BF16 | SGLang `UMT5EncoderModel` (`TextEncoderLoader`, `T5Config`, precision `bf16`) | reused; stage new |
| `prompt_clean`, 512-token padding, mask, `text_tokens_zero_pad` | `LongCatVideoTextEncodingStage` + DiT `_embed_text` (zero padded tokens after `y_proj`, attend all 512) | new |
| `AutoencoderKLWan` (LongCat copy), BF16 | SGLang `AutoencoderKLWan` (`VAELoader`, `WanVAEConfig`: encoder+decoder, `use_feature_cache=True` = native causal chunking, no tiling/parallel) at `bf16` | reused |
| VAE posterior `sample(generator)` after the noise draw; BF16 (de)normalization; decode tail | `LongCatVideoContinuationStage._prepare_latents/_decode`, `video_io.py` | new (bitwise replicas) |
| `FlowMatchEulerDiscreteScheduler` (shift 12) | SGLang `FlowMatchEulerDiscreteScheduler` (`SchedulerLoader`, `flow_shift=None` keeps the checkpoint's 12); request-local deep copy per segment; schedule checked bitwise against `schedule.continuation_schedule` | reused + check |
| `get_timesteps_sigmas(16, use_distill=True)` | `schedule.distill_sigmas` | new (bitwise) |
| `LongCatVideoTransformer3DModel` | `sglang_dit.LongCatVideoTransformer3DModel(CachableDiT)`, registered in `ModelRegistry` under the native `_class_name`; math in `dit_core.LongCatVideoDiTMixin` | new |
| flash-attn 2 self/cross attention | SGLang `LocalAttention` (FA, FA2 or torch SDPA chosen by SGLang; SM120 defaults to SDPA, SM80 to FA) | reused layer |
| `rope_3d.RotaryPositionalEmbedding` | `rope.py` (CPU FP32 angle table, device cos/sin cached per grid) | new (bitwise) |
| `cfg_step_lora` via `enable_loras` (unmerged) | `lora.LoRALinear`/`LoRABranch`, attached after the base load by the pipeline-local loader | new (bitwise op order) |
| `_cache_clean_latents` | `build_conditioning_kv` (final layer skipped: its output is discarded natively) in the continuation stage | new |
| `forward_with_kv_cache` | `LongCatSelfAttention` cache branch (concat + read-time RoPE) | new |
| segment loop of `run_demo_long_video.py` | `LongCatVideoContinuationStage` (`num_segments`) | new |
| weights directory without diffusers `model_index.json` | `LongCatVideoContinuationPipeline._load_config` returns a synthetic index; `transformer` slot -> `dit/`; full path+size+SHA256 binding (`weights.py`, item 9 hashes) | new |

Stage chain: `LongCatVideoInputStage` (frames from `video_path` .npz/.npy/png-dir or
`condition_inputs["cond_frames"]`; bucket; request generator on the worker device) ->
`LongCatVideoTextEncodingStage` -> `LongCatVideoContinuationStage` (per segment: latents, cache build or
restore, denoising, decode). Output: `OutputBatch.output` `[1, 3, T, 704, 1312]` FP32 in [0,1] (segment 0:
93 frames, later segments: their 80 new frames); SGLang's writer converts with `(x*255).to(uint8)`, the
native truncation. Per-segment records (epochs, timings, hashes) stay on `req.extra["longcat_video"]`.

Registration (`reelcache_sglang_models/__init__.py` -> `longcat_video/registration.py`):
`ModelRegistry.register_model("LongCatVideoTransformer3DModel", "<module>:<class>")` and
`register_pipeline(LongCatVideoContinuationPipeline, hf_model_paths=["meituan-longcat/LongCat-Video"])`.
SGLang therefore maps any model path whose last component is `LongCat-Video` to this pipeline; the pipeline
then accepts only a local directory whose files match the binding (`REELCACHE_LONGCAT_VIDEO_WEIGHT_VERIFY=full`
default hashes 80 GB; `stat` = sizes + small-file hashes, recorded as weaker). The pipeline-local DiT loader
has `component_names = []`, so SGLang's global `transformer` loader is untouched; after loading it checks
that the 1,022 checkpoint tensors map one-to-one onto the module (identity names) and attaches the 336 LoRA
branches. `native_only_components` blocks a silent diffusers fallback for transformer/VAE/text encoder.

## 3. Conditioning-cache seam for ReelCache (typed, no compression yet)

`conditioning_cache.py` (pure torch; phase 3 adds the ReelCache backend in `reelcache_backend.py`, Section 9):

* `ConditioningCacheEpoch` (frozen): segment index, parent segment, cause `longcat_segment`, token grid
  `(4, 44, 82)`, SHA256 of the 4 conditioning latents, model fingerprint (hash of the weight binding),
  session id, `key_domain="pre_rope"`, `position_offset=0` (segment-local), layout.
* `LongCatConditioningCache`: 48 pre-RoPE K and 48 V tensors, `[1, 14432, 32, 128]` BF16 (`BTHD`, the
  FlashAttention layout; native is `BHTD`), `source` = `native_build` | `restored`. `layer_matrices(i)`
  returns ReelCache's canonical `[N, C] = [14432, 4096]` matrices as **views** (no copy);
  `layer_native(i)` gives the native layout for parity; `from_native_layout` / `from_matrices` build a
  restored cache.
* `ConditioningCacheBackend` protocol: `restore(epoch) -> cache | None` is called before the build
  (a returned cache is validated against the epoch, shapes, dtype, device and must be `restored`;
  anything malformed raises), `on_cache_built(cache)` right after a native build (capture/archive).
  Backends are registered by name in the worker process; requests carry only the name
  (`conditioning_cache_backend`) and `session_id`. An unknown name raises.

Because the native cache already stores pre-RoPE keys and applies RoPE at read time, capture needs no
un-rotation and restore needs no re-rotation: installing a restored cache is a pointer swap. A ReelCache
backend (Step 11's `reelcache.integrations.sglang` owns it) archives each epoch, and on resume either decodes
the archived epoch or lets the pipeline rebuild it from the stored 4 latents / 13 frames; Section 3.5's
restore requirement compares those two per GPU (native rebuild: RTX PRO 6000 2.42-2.53 s, A100 see README).

## 4. Parity expectations and phase-2 plan

Bitwise cross-engine identity is not expected: attention kernels (native flash-attn 2.8.3 vs SGLang
FA/FA2/SDPA), Torch 2.8/cu128 vs 2.13/cu130 cuBLAS/cuDNN, and SGLang's own UMT5/VAE implementations differ.
Isolation hooks (in-process requests only): `condition_inputs["prompt_embeds"|"prompt_attention_mask"]`,
`condition_inputs["cond_latents"]` (segment 0; noise still drawn), `condition_inputs["initial_latents"]`
(segment 0; replaces noise + VAE encode), `extra["longcat_step_observer"]` (per-step callback), and a
capturing cache backend. Each injection is recorded in the request state. Phase 2 compares, per GPU, against
`artifacts/sglang_integration/sglang/step13_longcat/native_ref/results/`: prompt embeddings, conditioning
latents, per-layer cache K/V (max/mean abs and relative error, all 48 layers for vb0294), step-0 model output
and latents, every step's latents, final latents, decoded frames (SSIM/PSNR), and SGLang cache rebuild time.

## 5. Verified in phase 1 (CPU)

`tests/test_longcat_video_port.py` imports the hash-pinned native modules on the CPU and checks, bitwise:
RoPE angle tables and rotation, segment-local rows, distilled sigmas/timesteps/Euler step/timestep tensor,
prompt cleaning, frame preprocessing vs diffusers `VideoProcessor`, latent (de)normalization, decode tail,
bucket tables and lookup, LoRA application, and a tiny FP32 DiT (plain, conditioned, cache build = native
pre-RoPE K/V, cache read, LoRA on/off) against the native `LongCatVideoTransformer3DModel` with SDPA.
The real 1,022 checkpoint names and the real LoRA header (336 targets, qkv split 3, kv split 2) are checked
against the full-size model on the meta device. `tests/test_longcat_video_port_isolation.py` checks import
isolation. A CPU smoke in the in-progress SGLang env (`sglang_smoke/`) registered the package, resolved the
native checkpoint path to this pipeline and ran a tiny model through SGLang's `LocalAttention` (bitwise equal
to the core).

## 6. Known costs not optimized yet (native behavior kept)

RoPE of the 86,592-token concatenation recomputed every step and layer; `y_proj` recomputed every step;
unmerged LoRA (~4% extra FLOPs); cache + current K/V concatenated per layer per step. Merging LoRA or caching
rotated cache keys changes rounding or memory and must be measured as a separate, labeled mode.

## 7. Packaging

`pyproject.toml`'s `[tool.setuptools.packages.find] include = ["reelcache*"]` already matches
`reelcache_sglang_models*`; no change is needed. The Step 8 env re-installed ReelCache, so `reelcache_sglang_models`
imports in `venvs/sglang-cu132` once `build/env_cu132_reelcache.sh` is sourced; phase-2 jobs prepend an immutable
snapshot of the package to `PYTHONPATH` to pin the code they run.

## 8. Phase 2 on GPU (SGLang, RTX PRO 6000 / A100-80GB)

Evidence: `artifacts/sglang_integration/sglang/step13_longcat/phase2/README.md`. The port runs unchanged
through SGLang's worker-style in-process path (registry, `ServerArgs`, `build_pipeline`, `prepare_request`,
`pipeline.forward`) and through `DiffGenerator` (scheduler + GPU-worker processes; frames bitwise equal to the
in-process run), warmup off. Attention backend: FA2 is not importable in the cu132 env (only flash-attn-4's
CuTe package), so `--attention-backend torch_sdpa` is used; Torch dispatches FLASH_ATTENTION for every LongCat
attention shape (on SM120 SGLang's `fa` request falls back to SDPA anyway; on SM80 SGLang's default is
sgl-kernel FA3-Sm80, measured as a comparison). Exact: schedule, noise, injected inputs, layer-0 V (K within one
BF16 ulp in rare elements), determinism of builds, runs and the RAW restore seam (a restored cache gives
bitwise-identical latents and frames). Not bitwise: attention/GEMM/reduction kernels of Torch 2.13 vs the
native Torch 2.8 + flash-attn 2.8.3 (switching SDPA kernels inside SGLang moves results as far as SGLang is
from native), SGLang's UMT5 and Wan-VAE implementations (differences comparable to the native cross-GPU ones). The
per-layer cache error vs native is below the native RTX-vs-A100 cache difference on every compared layer except
one e2e layer (ratio 1.011); step-0 model outputs differ by 3-6e-3. Final frames: RTX samples SSIM >= 0.9966; the
A100 reference samples (different CUDA noise: SM-count-dependent `torch.randn`) are rounding-sensitive (final
latents 1.5-10 %) and an attention-kernel swap inside SGLang diverges equally. SGLang cache rebuild: RTX 2.38 s
warm (native 2.42-2.53 s), A100 3.20 s `torch_sdpa` / 3.16 s `fa` (native 3.25-3.27 s). No package change was
needed.

## 9. Phase 3: ReelCache-compressed per-segment conditioning-cache epochs

Evidence: `artifacts/sglang_integration/sglang/step13_longcat/phase3/` (README, PROGRESS).

* `reelcache_backend.py` (SGLang-free; core `reelcache` is imported lazily and never imports the package):
  `ReelCacheConditioningCacheBackend` implements the seam. `on_cache_built` captures the 48 x {pre-RoPE K, V} canonical
  `[14432, 4096]` views of a natively built cache (plus the 4 conditioning latents and the segment's generator
  state); archival writes one ReelCache lineage per session (`longcat-<session>`) whose nodes are `segment` commits of
  the latest method at `energy_targets("longcat_720p")` (K98/V95), all-intra svd_hevc node packs on canonical
  canvases, published by `ArchivePipeline` as `longcat_segment` cache epochs (one segment-local range `[0, T)`;
  parent = the previous segment's epoch; the runtime refuses a cross-segment append; a rewind that regenerates
  segment k forks the lineage after segment k-1). A RAW reference branch per epoch (`seal_chunk`, `CodecSpec()`) serves
  the identity gate. `restore(epoch)` resolves the epoch by (model fingerprint, conditioning-latent digest, token
  grid), checks the stored node/epoch, and restores into fresh BTHD tensors: svd_hevc through `LineageRestorer`
  (packed NVDEC + fused `restore_kernels`, `rope=None` because the cache holds pre-RoPE keys) or the software
  restorer on the CPU; raw through the RAW history + `restore_kernels` dense install. Restores happen only when armed
  (`arm_restore`); an armed restore that cannot be served raises (no silent rebuild); captures are never dropped
  (bounded pinned slots raise when full). Restores run with InferenceMode disabled: SGLang's executor runs stages under
  `torch.inference_mode()`, and the decoder pool's worker threads write the decode canvases in place, which inference
  tensors forbid outside InferenceMode (found by the first SGLang e2e run, regression-tested).
  Warm-up has two explicit levels (restore accounting, user rule of 2026-10-15 22:51: the requirement is the prepared
  state; process warm-up and the session's CPU packing are excluded, the CPU packing is reported as an optimization goal):
  `prewarm_process(token_grids=((4, 44, 82),), max_rank=None)` is the process level (backend attach; reads no archived
  data): `restore_kernels.warmup` compiles/loads every restore-kernel variant of
  LongCat's launch groups (BTHD BF16, picture sources, no rotation -- pre-RoPE K and V -- ranks 1..max_rank) and
  `PackedRestorer.warmup` creates a decoder session for every canonical node canvas of the conditioning-cache tile
  geometry up to `max_rank` (17 canvases up to rank 3072, 18 up to 4096 for 4x44x82 yuv444p; `gpu_archive_stack` allows
  20 sessions per worker, the core raises `CapacityError` above the pool's limit); a core without these APIs raises. `prepare_epochs(epochs=None)` is the session level: node packs prepared,
  basis versions device-resident and decoder sessions for the epochs' canvases (`LineageRestorer.prewarm(history)`,
  budget-bounded) is the session's CPU packing -- restores after it run in the **prepared state** (the requirement row);
  its own time is the CPU-packing figure. `prewarm(epochs)` = both (session open). `warmup()` (one discarded restore)
  remains for the archiving process. `set_overlap(True)` (the backend's default for direct calls; session requests
  pass their own choice, default on since d5: §18) makes svd_hevc
  restores layer-ahead: `restore()` returns the allocated cache at once with a `LayerReadiness` (per-layer host flag +
  CUDA event); a background thread runs the `LineageRestorer` restore on a side stream and marks layer `L-1` when the
  restorer starts issuing layer `L` (its reconstruction loop is layer-ordered; out-of-order issue raises), the rest
  after it returns; the DiT's first step waits per layer (`kv_cache_ready`), so decode/reconstruction of later layers
  overlaps the forward of earlier ones; the stage joins the restore after step 0 (a failure re-raises there or in a
  waiting layer, never a silent read of unwritten layers) and records `time_to_step0_end_seconds` for builds, plain
  and overlapped restores. The node pack read, its preparation and the basis upload still precede layer 0 (one node
  pack per epoch; per-layer reads/basis assembly need a core API). `reload_index()` reindexes the svd_hevc segment
  nodes of a reopened store and the RAW reference branches of those epochs (RAW resumes after a restart).
  `diagnostic_method(k, v)` builds the latest method at explicit K/V energies for sensitivity studies (labeled
  provisional; not a profile).
* Archival schedule: `deferred` (default; async D2H at capture, archive at `flush()` when generation is idle) or
  `eager` (submit at capture; archival runs behind the following segments). Phase 3 planned on the CPU in SGLang (the
  unbounded GPU planner's working set, ~57 GB then, did not fit next to the served model); since Step 7 bounds it
  (`ArchiveOptions(plan_memory_bytes=B)`), the session provider plans on the GPU next to the model (section 12).
  Restore always runs on the GPU.
* `segment_resume.py` + stage changes: every segment record carries `generator_state` (hex); a request may start at a
  segment boundary (`start_segment_index`, the boundary's 13 frames, `generator_state`), which replays the segment
  exactly and lets an armed backend restore its epoch. Defaults leave plain requests unchanged.

## 10. Phase 4: native context parallelism and the per-rank conditioning-cache seam

Evidence: `artifacts/sglang_integration/sglang/step13_longcat/phase4/` (README, PROGRESS).

* `context_parallel.py` (pure torch.distributed) reimplements native `context_parallel_util` / `ulysses_wrapper`:
  `optimal_split` (= `get_optimal_split`; (1, 2) for 2 ranks), `RowShard` (rank `hi * split_w + wi` owns spatial block
  `(hi, wi)` of every latent frame, frame-major then row then column = `NativeOwners.spatial_rows`), `split_tokens` /
  `assemble_tokens` (= `split_cp_2d` after the patch embed / `GatherFunction2D`), `all_to_all` (the native algorithm:
  chunk `r` of the scattered axis to rank `r`, received chunks concatenated rank-major on the gathered axis) and
  `LongCatContextParallel` (a process group + split: `ulysses_in`/`ulysses_out` in this port's `[B, S, H, D]` layout,
  `gather_tokens`, `broadcast` from CP rank 0).
* `dit_core.py`: under CP the patch-embedded tokens are split to the rank's block, every self-attention call runs the
  kernel on `H/N` heads of the Ulysses-gathered sequence between the two all-to-alls (plain, conditioned and cache-read
  paths alike), RoPE uses the rank's rows (`RopeTables.cos_sin(..., shard)`: the FP32 angle table is split before
  cos/sin, as native), cross-attention and the per-token layers run on local rows, and the final-layer output is
  all-gathered so every rank holds the full prediction. A cache build returns, and a cache read expects, the rank's
  rows only. Weights stay replicated.
* SGLang: `sglang_dit.sglang_context_parallel()` resolves CP from SGLang's SP group at first use: native Ulysses over
  the whole group (`ulysses_degree == sp_degree`, ring 1, tp 1); other layouts raise. `validate_server_args` accepts
  `sp_degree = ulysses_degree` in {1, 2, 4} (native rounds buckets to `16 * max(split)`; larger splits select bucket
  tables that are not ported) and rejects ring, TP, K/V-gather (also SGLang's automatic `kv_gather_degree=2` at
  `sp_degree=2`), CFG parallel and DP. The stages stay REPLICATED (SPMD on every rank, as SGLang's GPU workers run
  them); as in native `generate_vc`, the continuation stage broadcasts the prompt embeddings and every segment's
  prepared latents from CP rank 0, and every rank steps the scheduler and decodes the full latents. For native-parity
  numerics run SGLang with `encoder_parallel="replicate"` (folding the UMT5 encoder changes its numerics; native CP
  encodes on rank 0 with the whole encoder).
* Per-rank seam (`conditioning_cache.py`): under CP the epoch carries the rank's `RowShard` (`row_shard`;
  `local_tokens`, `row_index()`, `full_epoch()`, `as_dict`/`from_dict`); a rank's cache is `[1, T_local, H, D]` pre-RoPE
  K and V of exactly `row_index()` (LongCat 720p: 7216 of 14432 rows per rank, 5.67 GB). Capture: `on_cache_built`
  sees the rank's shard; `layer_matrices(i)` are the `[T_local, 4096]` row views (summing `row_shard_gram` over ranks
  gives the full Gram for an owner fit). Install: `LongCatConditioningCache.from_matrices(epoch, keys, values)` takes
  `[T_local, C]` rows in `row_index()` order -- exactly what `FactorShardDomain.restore_node(node_map,
  NativeOwners.spatial_rows(**row_shard.native_owners_kwargs(grid, kv_heads=32, head_dim=128)), rope=None)` returns per
  entry -- and `from_native_layout` takes native `[1, H, T_local, D]`. `CacheRowSeam(capture=..., install=...)` adapts
  such callables to the backend protocol. RoPE is never applied at install: keys stay pre-RoPE and rotate at read
  time. The single-GPU `ReelCacheConditioningCacheBackend` refuses row-sharded epochs explicitly (a factor-sharded owner
  backend is the Step 16/17 integration).


## 11. Step 14 CR-14-3: session provider (SGLang session controls, segment boundaries)

Evidence: `artifacts/sglang_integration/sglang/step13_longcat/cr14_3/` (README, PROGRESS).

* Attach: `LongCatVideoContinuationPipeline.reelcache_session_provider = "reelcache_sglang_models.longcat_video.session_provider:create_provider"`.
  With `--reelcache-enable` (SGLang branch `reelcache-integration`), `reelcache.integrations.sglang.create_sglang_backend`
  dispatches a pipeline without causal stages to that factory with the `--reelcache-config` dict; the returned
  `LongCatSessionProvider` is the worker backend (`control`, `inflight_control`, `set_boundary_hook`, `close`) with the
  ops, arguments, results and `SessionControlError` codes of `reelcache/integrations/sglang/controls.py` and the boundary
  unit `segment`. `LongCatSessionConfig` validates the config (`store_root` required; `representation`
  svd_hevc|raw, `plan_device` auto|cuda|cpu with `plan_memory_bytes`, `schedule`, `defer_planning` (section 12),
  `durability` pause|always|never, `prewarm`, `warmup`, `capture_slots`, explicit
  provisional energy targets, restore budgets; unknown keys refused). On CUDA it builds `gpu_archive_stack` and runs
  `prewarm_process()`; without CUDA a CPU pipeline + software restorer (tests).
* Session archive (`session_archive.py`, SGLang-free): a session is a chain of segment boundaries stored in the cache
  archive's ReelCache store through its public object API -- boundary `k` = blob of the 13 conditioning frames (uint8) +
  manifest (generator state at segment `k`'s start, geometry, parent boundary, record of segment `k-1`: prompt,
  conditioning-cache epoch, cache source, digests, times); session = ref `longcat/sessions/<id>` (compare-and-swap) to
  a record naming its head; named checkpoints = refs `longcat/checkpoints/<id>/<name>`. Forks share boundaries
  (copy-on-write); `gc` = store garbage collection from refs (+ RAW reference branches no live boundary uses).
* Generation (stages): a request's `causal_kv_session` reaches `LongCatVideoInputStage`, which asks
  `provider.begin_request` for the plan -- `generate` (the request's video is boundary 0; `replace` drops an existing
  session), `resume` (head), `fork` (`from_session` + checkpoint: new session at that boundary), `rewind`
  (checkpoint: an ancestor of the head). Resume/fork/rewind take frames + generator state from the archive (requests
  carrying a video, start segment, generator state, cache backend or injected latents are refused); `num_segments`
  (a declared `/v1/videos` field) and `pause_after_chunks` (= the session's segment count) bound the request;
  `output` is `new` only. The continuation stage reports every segment (`end_segment`: boundary committed, session ref
  moved, then the boundary hook -- the scheduler's in-flight poll; a `pause` of the running session stops there) and
  `finish`es the run (durability). The session requests' conditioning-cache backend restores only *published* epochs
  and builds the rest (recorded per segment as `provider_cache`), so a rewind/fork under a new prompt restores the
  segment's prompt-independent cache instead of rebuilding it. Context-parallel session requests are refused (one GPU).
* Backend support (`reelcache_backend.py`): lineage parenting by the session's *timeline* (`seed_timeline`; captures
  record their parent epoch at capture time, so deferred archival after a rewind still parents correctly; a parent on
  another branch or session -> a fork `~rN` at its published node; an unarchived parent -> a fresh lineage, recorded);
  `archive_state`, `pending_epochs`, `discard_pending` (close/delete abort), `session_branches`, `forget_session`,
  `refresh_index` (after gc).
* Restart safety: everything is in the store; a new provider re-indexes the published epochs and lists every session
  at its last committed boundary (a kill between boundaries loses at most the running segment; nothing half-written is
  referenced).
* Live epochs, gc and the index (CR-14-6, Step 14 M3): the *live epochs* are the conditioning-cache epochs the live
  boundaries record (session heads and named checkpoints with their ancestors). Garbage collection keeps exactly
  their archives -- RAW reference branches of other epochs are deleted, svd_hevc nodes no lineage reaches are
  collected -- and re-indexes exactly them (`refresh_index(live_epochs)`); a restart does the same
  (`reload_index(live_epochs)`). Before, `reload_index` found RAW branches only through svd_hevc nodes (a RAW branch is
  named by a digest and carries no epoch identity): on the RAW representation every gc, delete and restart left the
  index empty and later requests rebuilt caches that were still archived (M3 `prompt_change_new_segment`). Deleting a
  session first *holds* the svd_hevc epochs other sessions still record but only its lineages reach -- epochs a fork
  or rewind restored instead of archiving again: a holder lineage `longcat-held~<digest>` forked at that node
  (`hold_epochs`, deepest first); gc releases holders whose epoch is no longer live or that a session lineage reaches
  (`release_holders`) and re-holds live epochs only a released holder reached. Unreachable epochs are gone (not
  indexed, archives collected). Tests: `tests/test_longcat_video_session_provider.py` (both representations), real
  SGLang stages: `artifacts/sglang_integration/sglang/step13_longcat/cr14_6/cpu_stage_index.py`.

## 12. Serving UX: archival planned on the GPU next to the served model

Evidence: `artifacts/sglang_integration/sglang/step13_longcat/gpu_planner/` (README, PROGRESS).

* Why: with the CPU planner a durable request (a pause under `durability: pause`, or `durable: true`) waited minutes per
  archived segment. Step 7's `ArchiveOptions(plan_memory_bytes=B)` plans (and finalizes) a node's matrices in groups
  whose working set fits `B` (eigensolver `B/2`): a LongCat node at `B` = 8 GiB peaks ~1.9 GiB above its 11.35 GB of
  sources instead of ~35 GiB unbounded.
* Provider config (`LongCatSessionConfig`): `plan_device` `auto` (default: the GPU when the worker has one) | `cuda` |
  `cpu`; `plan_memory_bytes` (default 8 GiB; required for GPU planning -- unbounded GPU planning next to the model is
  refused); `schedule` `auto` (default since 2026-10-16 06:33, coordinator decision, reversible: `eager` for a GPU
  planner whose eager-overlap memory check fits -- or with `defer_planning` --, else `deferred`: RTX PRO 6000 ->
  eager, A100-80GB / CPU planner / RAW -> deferred; `resolve_schedule`, logged at attach and reported as
  `capabilities.archival.schedule` / `schedule_requested` / `schedule_reason`) | `deferred` (a request's epochs are
  planned when it pauses/ends, or when the capture slots are full) | `eager` (each epoch is submitted at capture and
  planned while the following segments generate, so a durable pause waits only for what is still unpublished;
  allowed for a CPU pipeline or a bounded GPU pipeline);
  `defer_planning` (GPU only): the continuation stage opens the backend's `generation_section()` around each
  segment's denoising and its VAE decode (`ArchivePipeline.defer_gpu_work(wait=True)`): no GPU planning starts inside
  them, entering waits until the fitter finishes its current node (recorded per segment as
  `planning_deferral_wait_seconds`); the cache build and its capture stay outside (submit/drain must not run inside a
  section). `LongCatSessionProvider.memory_critical_section(reason, request_id=None)` exposes the same section through
  SGLang's seam (branch bb6f7c5116; LongCat's own stages do not use SGLang's DecodingStage). `capabilities.archival`
  reports the effective settings. `check_eager_overlap` refuses at worker start an `eager` schedule without
  `defer_planning` on a device without the measured headroom (resident + 41 GiB + `plan_memory_bytes`/4: allowed on
  the RTX PRO 6000, refused on the A100-80GB, where the fitter ran out of memory next to denoising and the session's
  lineage failed).
* Measured (vb0294, 2 / 1 segments per durable pause; `gpu_planner/README.md`): RTX PRO 6000 CPU planner 617 / 283 s,
  GPU deferred 80-91 / 41-44 s (~43 s per node: finalize ~20 s, x265 ~11 s, plan ~8-18 s), GPU eager 32 / 0.06 s
  (+0.9 % step time, +13 GiB peak), eager + defer ~0.1 s at the pause with ~28-57 s of planning waits per segment;
  A100-80GB CPU 1173 / 422 s, GPU deferred 203 / 103 s, eager + defer 0.3 / 0.2 s with 68-145 s waits per segment.
  GPU-planned nodes match CPU-planned ones in every decision and rank view (energies within 9.1e-5); every matrix
  meets K98/V95.
* Device memory: the provider's GPU stack hands the planner the pinned host copies of captured epochs
  (`deferred_sources="host"`): the planner uploads one node's sources (11.35 GB) when it plans that node, so archival
  holds one node's sources plus the bounded working set on the device, never every captured epoch at once (the
  phase-3 `"device"` mode re-uploads every deferred epoch at flush).
* Identity: bounded planning stores the node of unbounded planning wherever the eigensolves are batch-independent
  (CPU; CUDA exact solver one matrix per call from n = 4096); with the default `topk_rr` solver its k x k
  Rayleigh-Ritz eigensolves are whole-batch calls, so a group size can change last bits (Step 7 section 8.7).
  `node_matrix_rows` / `compare_node_rows` compare two plans of one epoch matrix by matrix: the documented planner
  tolerance (CPU vs GPU planner) is identical decisions and rank views (docs/validation.md, Step 4), with every
  stored energy at or above its K98/V95 target.

## 13. Integrity defaults: verify on read, no open-time audit, quarantine (CR-17-1, g5)

Evidence: `artifacts/sglang_integration/sglang/step13_longcat/cr17_1/` (README, PROGRESS). Spec: CR-17-1 (Step 17
matrix: with the old default `verify_reads=False` a corrupted packet restored into wrong output; with
`verify_reads=True` the open-time hash audit made one corrupt object kill the server at startup).

* Defaults (`reelcache_backend.INTEGRITY_DEFAULTS`; provider config `verify_reads`, `audit_on_open`, `on_open_error`):
  `open_runtime` / `gpu_archive_stack` open the store with `KVRuntime(verify_reads=True, verify_references=False,
  audit_on_open=False, on_open_error="quarantine")` (Step 7 dev20 core API): every object read for use is
  hash-verified (boundary frames, cache packets, RAW packets; NVDEC does not detect bitstream damage and RAW packets
  have no decoder), graph operations stay at index cost, opening checks every head's graph at index cost (manifests
  parse and verify, references indexed) instead of hashing every stored byte, and a damaged head is quarantined
  (`runtime.quarantined`, `QuarantinedError` when loaded; GC keeps damaged graphs). On a core without these flags
  only the old behavior (`verify_reads=False, audit_on_open=False, on_open_error="raise"`) opens; anything else
  raises (no silent weaker integrity). Reported at attach (`startup["integrity"]`, log line) and in
  `capabilities.integrity` (settings + the damage known now).
* Restore of a damaged archive (`ReelCacheConditioningCacheBackend.restore`): an `IntegrityError` from the restorer or
  the RAW history (hash mismatch, missing object, quarantined head) is re-raised as a typed `IntegrityError` naming
  the segment and digest; the epoch is quarantined in the index (`archive_state` -> `"quarantined"`, `lookup` raises
  without reading; the next re-index re-derives it) and the frames of the failed restore are cleared
  (`traceback.clear_frames`), so its 11 GB of destination tensors and staging buffers are released while the error
  propagates. Nothing is returned and nothing is rebuilt silently. `prepare_epochs` (session-level packing, the
  `open` op) isolates damaged epochs (prepared one by one after a failure; damaged ones quarantined, the rest
  prepared). An `IntegrityError` caused by the store's index database (the core wraps sqlite errors) is not damage:
  it is re-raised as is and quarantines nothing (`is_damage`).
* Session requests: the boundary frames are read and verified before a fork or rewind changes any session; a
  generation request that fails with an `IntegrityError` before its first new boundary is rolled back (a rewind gets
  its previous session record back, a fork target and its cache lineage are removed), so the published state is
  unchanged; the request summary carries `error_code: "integrity"` (also for refused requests, `refused: true`), the
  SGLang job fails with no output.
* Damage found at attach (`on_open_error="quarantine"`): `SessionArchive.scan` reads every session record and named
  checkpoint with its boundary chain on its own (`SessionArchiveIntegrityError` for foreign/malformed records, broken
  chains, frames that do not match their digest); a damaged session is listed with `status: "quarantined"`,
  `describe` reports it, `delete` removes it without reading it, every other op on it fails with `integrity`; a
  damaged named checkpoint fails typed when used. `reload_index` skips lineages the runtime quarantined or whose path
  does not load and quarantines the live epochs they may hold (not on any healthy lineage, and the lineage belongs to
  a session that records the epoch, or is its holder); RAW branches that do not load quarantine their epoch's RAW
  archive. A session whose own cache lineage is quarantined fails resume typed; a fork from it fails when it needs a
  quarantined epoch (and is rolled back). Garbage collection while a session record or named checkpoint is damaged is
  conservative: it keeps every RAW branch and holder lineage (the damaged records may reach them) and the store keeps
  damaged graphs. `on_open_error="raise"` keeps the old attach failure.
* Costs: verification hashes what a restore reads -- nothing on the prepared path (packs prepared at seal or `open`),
  the pack bytes on a cold path (Step 5c: within +-2 ms on staged CausVid), the full 11.35 GB on a RAW restore (RAW is
  the exact reference, not the product path).
* Tests: `tests/test_longcat_video_integrity.py` (defaults and the open-time audit flags; index failures are not
  damage; corrupt svd_hevc packet ->
  typed error, no output, fork/rewind rolled back, quarantined epoch fails again without reading; buffers released
  while the error lives; corrupt RAW packet typed with verification and silently wrong without it; damaged RAW chunk
  manifest, session record, boundary chain, named checkpoint and cache lineage at open -> quarantined, other sessions
  work, conservative gc, quarantined delete; `raise` policy; damaged-epoch isolation in `prepare_epochs`);
  `tests/test_longcat_video_session_provider.py` runs the provider with the serving defaults.

## 14. Cold-session staging engine in the packed restorer (CR-17-4, g5)

Evidence: `artifacts/sglang_integration/sglang/step13_longcat/cr17_1/` (README, PROGRESS). Spec: CR-17-4 (Step 12 /
Step 11: the production restorers did not use Step 5c's `StagingEngine`, so cold LongCat restores took the store path
-- `basis_versions.assemble` / `rows_*`, thousands of tiny CPU ops on the intra-op pool, CR-17-2-sensitive).

* `gpu_archive_stack(..., staging_engine=True, staging_threads=None, staging_prepare_threads=None,
  staging_pinned_bytes=512 MiB)` (`STAGING_DEFAULTS`; Step 5c's recommended configuration, the SGLang bridge's names)
  builds the packed restorer through `lineage_restorer`: one `reelcache.restore_staging.StagingEngine` on the worker GPU
  (`cuda:<index>`, the decoder pool's) with threads = min(8, max(2, cpus // 2)) of the process's CPU affinity and
  prepare threads = max(1, threads // 2) unless given, passed as `LineageRestorer(staging_engine=...)`; the stack keeps
  it (the restorer does not own it) and closes it after the restorer, before the pool and the runtime.
  `staging_engine=False` restores cold histories through the store path.
* Every restore and session-level preparation (`prepare_epochs` / the `open` op: `LineageRestorer.prewarm`) whose
  history is not all prepared goes through the engine (parallel GIL-released reads, hash checks, memmem scans into the
  bounded pinned ring; FP16 basis tips uploaded on the engine's stream; no torch CPU compute ops); when the restore
  finishes, the staged packs and complete basis versions are adopted into the prepared and device basis caches
  (Step 7 hook). Prepared restores are untouched. Restored caches are bitwise those of the store path.
* Attach: `prewarm_process` (process level) runs `StagingEngine.warmup()` (allocates the ring, starts the threads;
  5c: 0.08 s RTX, 0.16 s A100 for 512 MiB); reported in `startup["staging_engine"]` / `capabilities.staging_engine`
  (`enabled`, `device`, `threads`, `prepare_threads`, `pinned_bytes`, plus the engine's counters) and the warm-up
  report. A worker without CUDA (software restorer) reports `enabled: false` with the reason.
* Provider config (`LongCatSessionConfig`): `staging_engine` (bool, default true), `staging_threads` /
  `staging_prepare_threads` (None or 1..64), `staging_pinned_bytes` (>= 1 MiB).
* Memory (5c): about 289 MB of FP16 basis tips on the device per cold segment restore (adopted into the device basis
  cache, `basis_budget_bytes` 2 GiB); host: the fixed ring plus the 385 MB stream bytes per segment, kept by the
  prepared cache like the store path (`prepared_budget_bytes`). One cold restore at a time per worker: a 512 MiB
  ring never waits (0 ring waits / oversize in every LongCat row; two concurrent cold restores would want 1 GiB).
* Integrity (CR-17-1 interplay): with verified reads the engine hashes the 385 MB stream (one thread; +161-171 ms on
  the RTX, 5c 4400146) and always hashes basis blocks and views; a mismatch raises `IntegrityError` before any
  reconstruction reads the rows, nothing is adopted, and the backend quarantines the epoch (section 13).
* Expected (5c, dev17 T1, LongCat segment cache incl. CPU packing; staged / store path / rebuild): RTX PRO 6000
  504 / 959 / 2380 ms; A100-40GB 729 / 1598 / 3162 ms; A100-80GB 710 / 1466 / 3162 ms. 5c measures the production
  path after freeze B.
* Tests: `tests/test_longcat_video_staging.py` (staged vs store path bitwise through the backend on a CPU decoder pool
  with a CPU engine, warm second restore without staging; a corrupt packet fails the staged restore typed and
  quarantines the epoch while a healthy one restores bitwise; defaults, thread bounds, validation, attach report).

## 15. Archival planning gated across sessions where eager overlap does not fit (CR-17-6, g6)

Evidence: `artifacts/sglang_integration/sglang/step13_longcat/cr17_6/` (README, PROGRESS). Finding (Step 17 freeze A,
A100-80GB job 4410238, step IL): `schedule: auto` chose `deferred` (the eager-overlap check does not fit); a
non-durable request's epoch was submitted when it ended (`flush(wait=False)`) and planned on the GPU at once; the next
session's request started 0.7 s later and ran out of memory in its first segment (69.1 GiB allocated, 78.0 GiB in
use; 10.6 GiB -- the planner's node sources -- still allocated after the failure). `deferred` chose *when* epochs are
submitted, nothing gated the planner against another request's denoising.

* `resolve_schedule` (attach): for a GPU planner where the eager-overlap check does not fit, `defer_planning` is
  forced on whatever the schedule (`auto` -> `deferred` + gate; explicit `deferred` -> + gate; explicit `eager`
  without it stays refused), with `release_planner_memory`; logged at attach (`defer_planning ... : forced: ...`) and
  reported in `capabilities.archival` (`defer_planning`, `defer_planning_reason`, `release_planner_memory`). RTX PRO
  6000 (check fits: auto -> eager), CPU planners and RAW are unchanged.
* One gated planner per worker: the provider's backend and its archive pipeline serve every session; the stages open
  `session_provider.memory_critical_section(backend)` around denoising and VAE decoding -- the request's cache
  backend's section and, when that is another backend, the worker provider's (a plain request, or one naming another
  backend, is gated too). Inside any open
  section the pipeline's fitter takes no node (`ArchivePipeline.defer_gpu_work(wait=True)`); entering waits until
  the fitter is outside GPU work -- the node being planned and finalized finishes first (at most one node), which
  frees its device sources and workspace (`CommitJob.finalize` drops the sources; the planner's bounded groups are
  freed per group). `release_cached_memory=True` then synchronizes the planner's stream and returns the caching
  allocator's unoccupied blocks (the planner's own stream pool) to the device, so nothing of the planner stays
  resident or reserved while a section runs. Measured headroom (gpu_planner 4403960, A100-80GB): generation alone
  peaks at 58.6-59.0 GiB allocated, the planner alone at 50.6-50.7 GiB (37.4 GiB resident + 13.2 GiB: 10.6 GiB node
  sources + its bounded working set); together ~72 GiB allocated + allocator/context overhead -> the 78 GiB OOM;
  gated (eager + defer_planning, same planner) the request's peak was 61.2 GiB allocated (70.2 GiB reserved, without
  the release), 18 GiB below the device.
* Progress and bounds: planning runs between sections (VAE encode, cache build or restore, between segments and
  requests) and after requests; a section never waits for more than the planner's current node. Captured epochs are
  bounded by the pinned capture slots (`capture_slots`): a capture with no free slot drains first
  (`flush(wait=True)`, outside every section: no deadlock; reported per request as `backpressure_seconds`); the
  pipeline keeps its own bounds (`max_unpublished`, `max_source_chunks`). Per segment the stage records
  `planning_deferral_wait_seconds` (boundary records and request summaries).
* Latency: a request that starts while the previous request's archival plans waits at its first section for the
  planner's current node (A100-80GB, gpu_planner 4403960: plan 7.4-15.9 s + finalize 39.7-89.2 s per node; eager +
  defer measured 68/145/81 s of waits per segment, i.e. up to two nodes planned in the gaps of one segment); the CPU
  encode and publication of that node continue during its generation. Admission control would wait for the whole
  archival (177 s in 4410238); a CPU fallback would take 324-599 s of planning per node on the A100 host and compete
  for the serving process's CPUs.
* Tests: `tests/test_longcat_video_planning_gate.py` (forced gating by device; a non-durable session's planning never
  holds its node's mocked device memory inside another session's or a plain request's sections, which wait for it;
  a request naming another backend enters both sections; the ungated control overlaps; three sessions back to back
  with two capture slots: bounded backlog, back-pressure, every node published, no deadlock).

## 16. Group-level planning gate: a section waits for one planner group, not a node (CR-17-6 follow-up, g7)

g6 (§15) makes a memory-critical section wait until the archive fitter leaves its *node*
(`ArchivePipeline.defer_gpu_work`). On A100-80GB that is 47-105 s for a second session, yet most of a LongCat node
never touches the GPU. The sources are the pinned host copies of the captured epoch (`deferred_sources="host"`).
The node's source fingerprints (12.6 s, host hashing) and its finalization (39.7-89.2 s: finalization runs on the
sources' device, the host) are host stages. Only the planner's matrix groups use the GPU: 24 groups,
plan 7.4-15.9 s per node (gpu_planner 4403960).

* `planning_gate.GatedPlanner` plans like `LatestMethodPlanner`, with the same groups, kernels and per-matrix
  arithmetic. Admission (validation, the finiteness check: ~1.4 s of host time per node) runs on the host copies.
  Each group's sources and inherited rows are copied to the device when the group starts, with the same
  `tensor.to(device)` / `deferred_reads.upload` the planner's admission used for the whole chunk. When the group
  ends, the copies are dropped together with the group's working tensors and new rows (downloaded, as the planner
  does). Between groups the planner holds no device memory (`residency.held_bytes_max`, measured), and the sources
  stay in the pinned capture slots, so there is nothing to offload when a section opens.
  * Plans are bitwise identical: LongCat's groups are independent (one matrix per eigensolver call from n = 4096,
    the bounded-planner guarantee of Step 7), so node ids do not change.
  * Peak planner memory drops from all 10.6 GiB of sources + one group's working set to one group (4 matrices,
    0.44 GiB) + its working set.
* `planning_gate.PlanningGate` (one per worker, on the fitter) treats each planner group as a *unit*.
  * A unit starts only while no section is open or waiting (sections have priority).
  * A section waits only for the unit in progress (≤ one group, ~0.3-0.7 s on A100), then releases the planner's
    cached blocks (`release_cached_memory`) and runs.
  * The fitter keeps taking nodes and running their host stages (fingerprints, finalization) during sections. The
    encode workers and publication run as before.
  * Deadlock-free: a section never waits for planner progress beyond the current unit, and the planner waits only
    for sections to close.
  * Starvation and backlog bounds are unchanged from §15: groups run in the gaps between sections, and the capture
    slots bound the captured epochs (back-pressure outside sections).
* `planning_gate.GatedGPUFitter` is `GPUFitter` with `GatedPlanner`. `gpu_archive_stack(gated_planner=True)` (the
  default) builds it. The backend's `generation_section` uses the gate (`planning_gate`, for host sources) when
  `defer_planning` is on, else g6's node-level `defer_gpu_work`.
  * Provider config `planning_gate: group` (default) or `node` (g6).
  * Reported at attach (`planning gate {...}` log line) and in `capabilities.archival.planning_gate`.
  * `planning_gate_stats()` reports units, unit seconds, deferred units, section waits, released bytes and the
    planner's residency.
* Where schedules apply: RTX PRO 6000 (eager, gate disabled) plans with lazy sources and an ungated planner (lower
  peak, same nodes). A100-80GB (deferred + forced gate) gates per group.
* The prototype overrides `LatestMethodPlanner._admit` / `_plan_group` and mirrors `GPUFitter.plan_chunk`.
  `tests/test_longcat_video_planning_gate_g7.py` pins their source (dev20c = freeze B2), so a core change fails the
  test and asks for re-validation. The natural home is the core planner (Step 7): group-wise source residency and
  a unit gate in `LatestMethodPlanner`, and a fitter-busy flag that covers GPU work only.
* Cross-session node ids follow archival timing (by design; plan §3.2). The rank-predictor registry is shared by
  every session of a worker and pinned when a node starts planning. A node's matrix records keep the predictor
  revision and the prediction records (search width) it was planned with, so a node planned before or after
  another session's node is published gets a different id, while its rows, ranks, decisions and energies are
  bitwise identical. A100 4412738 vs 4412376: the IL F1 node 083f4cd9 vs 00e5005a had the same bytes,
  decisions and margins. Same-lineage provenance is deterministic. Byte-identical new encodings are not promised
  under arbitrary cross-session timing; deterministic tests use replay mode, which pins revisions and order.
  CPU reproduction with the core planner alone:
  `artifacts/sglang_integration/sglang/step13_longcat/cr17_6/registry_timing_repro.py`.
* Tests:
  * gate semantics: waits, priority, disabled gate, timeout, no nesting;
  * `GatedPlanner` plans equal the planner's, with a placement ledger: at most one group's sources alive and none
    after a chunk;
  * the backend's gate selection;
  * two sessions on a CPU pipeline: g7 section wait 0.06 s (one 0.08 s group) vs 0.80 s for the g6 node-level
    control, no placed source alive inside any section, and identical node ids;
  * the core-internals pin;
  * CUDA tests (`REELCACHE_LONGCAT_G7_GPU_TESTS=1`): bitwise plans with host sources, and the planner holding one
    group (peak vs the base planner).

## 17. Freeze D: per-epoch raw tiers, compression as soon as generated, NVENC auto (Step 7 `DESIGN_D_tiers` §6/§9/§11)

* **Unit = one segment epoch** (96 pre-RoPE K/V matrices, 10.57 GiB at 720p; `mb.epoch_nbytes`). Raw copies are
  bitwise the native build and need no re-rope (`rope=None`). `raw_tiers.py` holds the tier index; the names are
  Step 7's: tiers `resident` / `gpu_raw` / `host_raw` / `compressed`, records `compressed_prepared` /
  `compressed_cold`.
  * `resident`: the cache the stage holds for its own segment.
  * `gpu_raw`: the built cache kept on its device after `del cache`. The stage calls `cache_released(epoch)`;
    zero-copy restore.
  * `host_raw`: the capture slot. It is the planner's source until publication, then an LRU entry (by access:
    restore hits touch). Restore = one direct H2D into fresh tensors; no store read.
  * `compressed`: the published node.
* **Restore order** (`_SessionCacheBackend.restore`, `raw_restore: prefer`): `gpu_raw` > `host_raw` (published or
  not) > the published compressed node > build. Records go in the segment record (`tier`, `tier_bytes`,
  `tier_seconds`; `provider_cache_detail` = the tier). `raw_restore: off` = compressed only. Lossy-quality and
  FAULT checks must set it, or call `drop_raw`: a raw copy is served bitwise and never reads the (possibly damaged)
  store.
* **Identity** (review amendment 4): an entry is keyed by a unique submission token until publication, then also by
  its `node_ref`. Restores look up the epoch's content key `(model fingerprint, conditioning-latents digest, token
  grid)`. Captures are de-duplicated by that key, so a regenerated or rewound epoch with other content never shares
  an entry, and a retired entry never clears a newer one's key.
* **Invariants:**
  * I1: an unpublished entry keeps a complete copy. A GPU copy is dropped only after the host D2H event completes;
    an unpublished host copy is never evicted.
  * The planner only ever receives the host copy (`deferred_sources="host"`). Finalize runs on the sources' device,
    so a GPU source would change the node bytes.
  * GPU raw beyond the budget is dropped at request admission, at finish, at every memory-critical section entry
    and at `cache_released`.
* **Budgets** (review amendment 2; Step 7 R5; `session_provider.plan_host_slots` / `plan_gpu_raw`):
  * Host: `0.5 × (cgroup headroom / raw_tier_workers − fixed)`, capped at 64 GiB.
    * Headroom = the tightest `memory.max − memory.current` on the process's cgroup path (v1 analogues; else
      `MemAvailable`).
    * `raw_tier_workers` default `LOCAL_WORLD_SIZE`, else 1.
    * Fixed = staging ring + 2 staged canvases (1.72 GiB each, measured) + the prepared cache + the NVENC probe's
      host delta.
    * Slots = budget // epoch. No minimum: below one epoch the tier is disabled (recorded). Captures then use C3
      back-pressure in `min(capture_slots or 4, physical share // epoch)` slots, and with none, conditioning-cache
      archival is off (resumes rebuild).
    * An explicit `capture_slots` / `host_raw_budget_bytes` above the worker's share is refused at attach.
  * GPU: whole epochs above the resident model + the measured generation need (37 GiB: A100 4414175 74.25 GiB worst
    device use, RTX 4412739 74.05 GiB) + 4 GiB margin + the NVENC probe's device delta.
    * A100-80GB: 0 (nothing is retained).
    * RTX PRO 6000: 1 epoch.
    * An explicit value is taken as given (`fits` recorded).
* **Schedule** (`resolve_schedule(raw_tier=True)`): `auto` → `eager`. Each epoch is submitted when its cache is
  built, so archival never waits for a pause.
  * Where the eager-overlap check fails (A100-80GB), `defer_planning` and the g7 group gate stay forced, so GPU
    planning never overlaps denoising or decoding.
  * The last resort (every slot holds an unpublished epoch) waits for the oldest submitted epoch to publish, outside
    sections (at capture). It is recorded with cause `host_full`; without the tier, `c3_bounds`.
  * The A100 backlog under continuous generation (~0.9 node per segment, 4414175) therefore meets the host budget
    there.
* **Prepare on evict** (amendment 3; Step 7 R6): when a published entry loses its last raw copy, its epoch is queued.
  At the next idle point (request finish, `drop_raw`), its node packs go through `LineageRestorer.ensure_prepared`
  (dev22a: touch or prepare host-side, no device work), else `prepare_epochs`. Compressed restores are labeled
  `compressed_cold` when the restorer staged or missed its prepared cache, else `compressed_prepared`.
* **NVENC** (`encoder: auto|x265|nvenc`; amendment 5): `select_encoder` probes once at attach.
  * The encoder lives as long as the stack, and there is no silent fallback: an explicit `nvenc` without NVENC is
    refused. `auto` records the reason it fell back to libx265.
  * The method's codec becomes the NVENC spec (qp 8, p1/hq, the method's pix_fmt/normalization), so the method
    fingerprint, node bytes and ids change on RTX.
  * Packets carry the encoder record with library/driver versions. The probe's device/host memory is recorded and
    budgeted.
  * K98/V95 must be re-verified post-codec on the 18 canvases (GPU check).
* **CP=2** (`cp_tiers.TieredRowArchive` over Step 16's `RowShardedCacheArchive`; amendment 1; Step 7 R4):
  * Per-rank host rows: the archive's capture buffer is referenced, not copied. Entries are evictable only after the
    collective two-phase publication on all owners.
  * Install picks one epoch-wide tier:
    1. all-gather availability;
    2. if every rank holds `host_raw`, each rank stages and verifies its rows (preflight);
    3. agree; any failure rolls back on every rank;
    4. otherwise all take the archive's collective compressed install (`compressed_cold`), or all build.
  * First cut: no CP GPU raw, and no owner-pack prepare-on-evict.
* **`raw_tier: false` = C3 exactly**: 4 unbudgeted capture slots freed at publication, no retention, C3 schedule
  decisions.
* **Layer-ahead restore (default for session requests since d5, §18) under asynchronous verification (5c D-E):** a payload found corrupt
  after step 0 started fails the restore thread with the typed `IntegrityError`; the backend quarantines the epoch.
  `LayerReadiness.wait` / `join` re-raise it as `reelcache.errors.IntegrityError` (same message, `from` the original),
  not `ConditioningCacheError`. The request therefore fails with code `integrity` and a fork or rewind is rolled
  back (session heads unchanged), exactly as in the decode-failure path. Tests: `test_longcat_video_port`
  `OverlappedCacheTests` and 5c's `LongCatGateTests` (also run without their mock of wait/join).
* **Records and controls:** Step 7 §9 names, in `artifacts/sglang_integration/sglang/step13_longcat/freezeD/RECORD_PATHS.md`.
  * A = `archive_stats` (`pipeline.raw_tier`, `pipeline.commit_to_publish`, `pipeline.encoder`).
  * B = `drop_raw`, `demote_raw`, `reset_raw_peaks`.
  * C = config, with effective values in `capabilities.archival.raw_tier` / `.encoder`.
* **Tests:** `tests/test_longcat_video_raw_tiers.py`, covering:
  * budgets from fake cgroup v1/v2 trees and the slot policy;
  * GPU budgets; encoder selection with a fake probe; the schedule;
  * the tier index (identity, eviction only after publication, LRU, I1, prepare-on-evict);
  * bitwise `gpu_raw` / `host_raw` restores, the planner receiving host copies only, and a deterministic last resort;
  * provider records/controls and `create_provider`'s CPU path;
  * a 2-rank Gloo test of the epoch-wide tier (mixed availability → compressed on both ranks; injected preflight
    failure → rollback on both; budget eviction).

## 18. Freeze F5 (d5): overlapped svd_hevc restore by default for session requests

Coordinator decision (F5; reversible), from the A100-80GB measurement 4424386 (freeze E, 2-step forks of L at boundary
1): prepared T1 40.23 s plain vs 39.79 s overlapped (-0.44 s), cold 40.44 vs 39.93 s (-0.51 s); all 10 restored arms
bitwise identical; only layer 0 waits (26-29 ms prepared, 0.15 s cold); +0.3-0.5 GiB allocated, about +0.6 GiB
device use inside step 0.

* **Switch.** `LongCatSessionConfig.overlap_restore` (default true; the SGLang bridge's config name,
  `ReelCacheRuntimeConfig.overlap_restore`). A request's `causal_kv_session.overlap_restore` (bool; SGLang 7168db64e3
  validates and forwards it) overrides it for that request. `LongCatSessionProvider.overlap_for(run)` decides per
  restore. The precedence is: the handle's value; then an explicit `ReelCacheConditioningCacheBackend.set_overlap(True |
  False)`, the d4 mechanism still used by tools, harnesses and tests (5c's D-E `LongCatGateTests`); then the config's
  value. **Outside a request a restore is plain** unless `set_overlap` was set: tools and timing harnesses (Step 13's
  TS, Step 19's `longcat_cold`) call `cache_backend.restore` with nothing waiting per layer or joining. Direct
  `restore()` calls likewise follow the explicit setting, else plain. `set_overlap(None)` unsets it, and
  `restore(epoch, overlapped=...)` takes a per-call choice. Records: `startup.overlap_restore` /
  `capabilities.overlap_restore` `{configured, effective, reason}`, R `overlap_restore` (what the request did).
* **Scope.** svd_hevc only. Raw tiers (`gpu_raw` zero-copy, `host_raw`), the RAW representation and RAW reference
  restores are plain (synchronous, already at copy speed). Context parallelism: sessions run on one GPU (a CP
  session request is refused); plain CP requests restore through `cp_tiers` (factor-sharded, collective) -- plain,
  no overlap path.
* **Semantics** (as in §9): the cache returns at once with a `LayerReadiness`; the restore thread marks layers on its
  side stream; the DiT's first step waits per layer; the stage joins after step 0 and re-raises a failure (a damaged
  payload: typed `IntegrityError`, the epoch quarantined, a fork or rewind rolled back -- before any output).
  Bitwise the plain restore.
* **Tier label at completion.** d4 labelled an overlapped restore when it was issued, before its thread read the
  store, so a cold overlapped restore read `compressed_prepared` (4424386). d5 evaluates the label
  (`_compressed_state`: prepared-cache misses and staging) on the restore thread when it completes and notes the tier
  then (`_note_restore`; the per-tier counters get the restore's real seconds). `last_restore_tier` reads `{tier:
  None, pending: True, overlapped: True}` until then. The provider notes `tier_pending` at issue and resolves it at
  the segment boundary from the stage's join record (`overlapped_restore.tier/bytes/seconds`), else the backend's
  completed record.
* **Records.** Segment records gain `cache_restore_overlapped`, `cache_restore_issue_seconds` and
  `cache_restore_exposed_seconds` (the first step's host waits on unready layers, from
  `LayerReadiness.consumer_waits`); the stage's `overlapped_restore` record gains `tier`, `bytes`, `seconds` and
  `consumer_waits` `{count, seconds, max_seconds, over_1ms, layers_over_1ms}`. `cache_restore_seconds` stays the
  blocking time of a **plain** restore and is absent for an overlapped one: consumers use T1
  (`time_to_step0_end_seconds`), `tier_seconds` (the restore's own duration) or `cache_restore_exposed_seconds`.
  A harness that needs the plain restore time sets the handle's `overlap_restore: false` for that request.
* **Budgets.** The GPU raw budget's generation need gains `OVERLAP_RESTORE_NEED_BYTES` (640 MiB) while the overlap is
  effective (A100-80GB: 0 epochs either way; RTX PRO 6000: 1 epoch either way). The fixed host consumers gain
  `finalize_workspace` = `FINALIZE_WORKSPACE_BYTES` (768 MiB; Step 7 dev22i's tile workspaces, E 4421998 peak
  741,826,560 B) for svd_hevc archival, so the raw tier's auto host budget leaves it free.
* **Decode entry guard (Step 7 dev22j6).** It lives in the bridge's `LineageArchive` memory-critical section (its
  `RawChunkTier` pressure response with a GPU fitter). LongCat's stage opens its own sections
  (`session_provider.memory_critical_section` -> `ReelCacheConditioningCacheBackend.generation_section`: the planning
  gate, `defer_planning`); LongCat builds no `RawChunkTier`, so the guard never runs for LongCat requests. LongCat's
  VAE decode on A100-80GB (records of 4418960 / 4420645 / 4420728 / 4421998 / 4424386, 35 decodes): +10.17 GiB
  allocated, +11.7-13.7 GiB reserved above the section entry, 30.5-32.8 GiB driver-free at entry, against the
  guard's 16 % prior of 12.68 GiB.
* **Tests:** `test_longcat_video_reelcache_backend` (completion label prepared / cold with a delayed miss, per-call
  override, consumer waits), `test_longcat_video_session_provider` (config and handle switch, default overlapped
  session restores bitwise the plain ones, completion tiers in the records, plain outside a request, raw
  representation), `test_longcat_video_raw_tiers` (the finalize workspace part, the overlap need). The test drivers
  mirror the stage: wait per layer, then join, before the segment boundary.
