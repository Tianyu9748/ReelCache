# ReelCache

**KV-cache archival, restore and branching for autoregressive video diffusion.**

Autoregressive ("causal") video diffusion models generate a video chunk by chunk, and every new chunk attends to
the self-attention KV cache of the chunks before it. In interactive use, a session pauses and resumes, rewinds to
an earlier moment, or forks into alternative continuations, and each of those needs the history's KV again.
Keeping that KV raw costs gigabytes per second of video; recomputing it from saved latents puts a long replay on
the critical path of the next chunk.

ReelCache archives the clean KV of every generated chunk as an immutable, compressed node in a per-session tree,
and restores any node's history directly into the engine's KV cache:

- **Compact.** Per-layer low-rank factors at a target energy (pre-RoPE K 98 %, V 95 % by default). The bases are
  shared and augmented across chunks, and the 8-bit coefficients are coded as all-intra HEVC.
- **Branchable and durable.** Content-addressed, immutable chunk nodes form lineage trees. Resume, rewind or fork
  to any committed chunk; restarts recover from the store.
- **Fast to restore.** NVDEC decode, fused reconstruction and the engine's own RoPE write straight into the
  native cache slots. The restore runs layer by layer, overlapped with the first forward of the next chunk.
- **Out of generation's way.** Generation keeps reading its native KV (`archive_native`). Archival is asynchronous,
  ordered and deferred into idle time, and recent chunks stay in raw GPU/host tiers.
- **Integrated.** A backend for SGLang Diffusion's GPU worker (`--reelcache-enable`) with a session API, plus a
  standalone integration for LingBot-World-Fast.

The Python packages are `reelcache` (the runtime) and `reelcache_sglang_models` (the SGLang model package), in one
distribution named `reelcache`. Serving uses **SGLang 0.5.21** with the 26 patches in `third_party/sglang/`; every
dependency and its evaluated version is listed in [§4.1](#41-dependencies).

This repository contains **code only**. The experiment harnesses, workloads, evaluation results, figures and model
weights are not included (see [What is not included](#12-what-is-not-included)).

## Contents

1. [How it works](#1-how-it-works)
2. [Supported models](#2-supported-models)
3. [Repository structure](#3-repository-structure)
4. [Dependencies and installation](#4-dependencies-and-installation)
5. [Quick start without a GPU](#5-quick-start-without-a-gpu)
6. [Serving with SGLang Diffusion](#6-serving-with-sglang-diffusion)
7. [Configuration](#7-configuration)
8. [LingBot-World-Fast](#8-lingbot-world-fast)
9. [Tests](#9-tests)
10. [Deployment-specific paths](#10-deployment-specific-paths)
11. [Provenance](#11-provenance)
12. [What is not included](#12-what-is-not-included)

## 1. How it works

```
 SGLang GPU worker (generation)                       ReelCache runtime (reelcache)
 ─────────────────────────────                        ────────────────────────────────────────────────────────
 chunk i: denoising steps
          clean t=0 context forward ──── capture ───► normalized pre-RoPE K and V, every layer and CFG stream
          (native KV stays live; the                  │ fit, or project onto the parent's basis (fitter stream)
           next chunk proceeds)                       │ 8-bit coefficient canvases ─► all-intra HEVC (NVENC | x265)
                                                      │ node pack + new basis rows ─► content-addressed store
                                                      ▼ ordered publication (published frontier ≤ generated)
                                                      tiers: GPU raw ─► host raw (LRU) · prepared packs · decoded
 resume / rewind / fork request
   first forward of the new chunk ◄─ layer l ready ── restore: NVDEC ─► dequantize ─► P·Q ─► RoPE ─► KV slots
```

**Capture.** SGLang's causal attention calls a seam during each chunk's *clean* context forward (t = 0, never
during denoising steps). The seam hands over the normalized pre-RoPE K and the V of every layer, in native dtype.
Only clean, complete chunks are archived. A LongLive 1.0 prompt-switch recache is archived as a new *cache epoch*
for the frames it rewrote.

**Compress (intra-chunk).** Each `N × C` matrix (tokens × KV channels) is factored as `A ≈ P·Q`, with an FP16 row
basis `Q` (`r × C`) and FP32 coefficients `P`. The rank `r` is the smallest that keeps the energy target:
- An online predictor estimates `r` from a 32-direction sketch, using the anchor energy
  `E_m = ‖A S_m‖² / ‖A‖²` and a two-coefficient ridge regression in log space. Its state is shared across prompts
  per model, layer and K/V (`predictor_registry.py`).
- The fit is verified against the *stored* factors and grown or trimmed in 32-direction buckets
  (`rank_method.py`).

**Share (inter-chunk).** A chunk first projects onto its parent's basis, and the actual residual decides:
- If the inherited span covers the target, only coefficients are stored.
- Otherwise new rows are appended (*augmentation*), or a fresh basis starts a new group (*refit*), whichever a
  storage-aware estimate says is cheaper.

Bases are immutable row blocks with version manifests, so forks share them and old chunks never change
(`basis_versions.py`, `reuse.py`).

**Encode.** Coefficients are quantized to 8 bits per column (affine) and mosaicked into canvases. Each node is
encoded as an independent all-intra HEVC pack (QP 8, yuv444p): NVENC where the GPU has it, libx265 otherwise.
Encode-time validation decodes it back. There are no cross-chunk codec references (`node_packets.py`, `nvenc.py`,
`_hevc.py`).

**Store.** Immutable content-addressed objects with transactional roots and GC pins (`store.py`). A node records
its parent, cache epoch, matrices, basis versions and execution state (generator/RNG, latents) (`lineage.py`).
With `tree_retention="full"`, every node a session generated stays a legal rewind target. Archival is
asynchronous and ordered (`archive_pipeline.py`):
- generation runs ahead (the *generated* frontier) while nodes publish in order (the *published* frontier);
- pause and checkpoint wait until it is durable;
- a failed ancestor never publishes descendants.

**Tiers.** Several tiers sit in front of the compressed archive:
- Recent chunks also stay as raw KV in an LRU tier, GPU raw then host raw (`raw_tier.py`).
- Published packs can be staged in pinned host memory (the *prepared* tier) for warm restores
  (`restore_staging.py`).
- An optional *decoded-coefficient* tier keeps NVDEC's normative output in host RAM and bypasses the decoder
  entirely, with bitwise-identical KV (`decoded_tier.py`).

**Restore.** A resume, rewind or fork restores the target's visible history (`visibility.py`):
1. raw-tier rows where available, otherwise NVDEC-decoded packs from a persistent decoder pool;
2. dequantization and a fused `P·Q` reconstruction (`restore_kernels.py`);
3. the engine's RoPE at native positions, written directly into the native cache slots.

With `overlap_restore` (the default), the restore runs in layer order on its own CUDA stream while the first
forward of the new chunk runs; each attention layer waits only for its own layer (`overlapped_restore.py`).

**Interference control.** Archival work interferes with generation mainly through SM contention:
- *Generation-first deferral* moves archival into idle gaps between requests (`generation_defer.py`).
- The fitter runs on a dedicated CUDA stream.
- Optional levers cut archival's GPU and host footprint (§7).

**Baselines** for evaluation are built in:
- `raw`: the exact reference;
- `raw_offload`: raw KV behind the same LRU tiers;
- `restore_mode="recompute"` with `latents_only`: exact replay of the clean forwards from stored latents;
- `svd`: fixed-rank FP16 factors;
- `hevc`: HEVC on the normalized KV, without SVD;
- `rank_policy` `fixed_uniform` / `fixed_fraction`, and `reuse=false`: the ablation ladder.

## 2. Supported models

| Model | Size / resolution | Code | Cache lifecycle |
|---|---|---|---|
| CausVid | Wan2.1-T2V 1.3B, 480×832 | `reelcache_sglang_models/causal_wan` (profile `causvid`) | 3-latent-frame chunks, full history |
| RAVEN | Wan2.1 1.3B, 480×832 | `causal_wan` (profile `raven`) | 3-frame chunks, full history (SGLang profile) |
| LongLive 1.0 | Wan2.1 1.3B + LoRA r256, 480×832 | `causal_wan` (profile `longlive1`) | sink + rolling window; a prompt switch recaches the window (new cache epoch) |
| LongLive 2.0 | Wan2.2-TI2V 5B, 704×1280 | SGLang's LongLive2 pipeline, bridge profile `longlive2` | 8-frame chunks; global sink + pinned shot sinks + 32-frame window; multi-shot RoPE |
| LongCat-Video | 13.6B, 704×1312 (720p) | `reelcache_sglang_models/longcat_video` | per-segment conditioning-cache epochs (13 frames); context parallel |
| LingBot-World-Fast | Wan2.2-based (task `i2v-A14B`), 480P / 720P | `integrations/lingbot` (outside SGLang) | persistent full-history cache, 3-latent-frame chunks |

CausVid, RAVEN and LongLive 1.0 share one SGLang causal-Wan path, configured by *profiles* (data, not code paths).
LongCat-Video is a separate port: its conditioning cache is rebuilt each segment and is not a growing history (see
`reelcache_sglang_models/longcat_video/DESIGN.md`). Every model's default energy targets are pre-RoPE K 98 % / V 95 %
(`reelcache/types.py`, `ENERGY_TARGET_PROFILES`).

## 3. Repository structure

```
ReelCache/
├── README.md                       this file
├── pyproject.toml                  distribution `reelcache`: packages reelcache*, reelcache_sglang_models*; optional extras
├── .gitignore
├── reelcache/                      the ReelCache runtime: engine-independent core (numpy + torch; never imports SGLang)
│   ├── adapters/                   native (non-SGLang) clean-boundary adapters: CausVid, MAGI (SP=2), and study helpers
│   ├── integrations/
│   │   ├── sglang/                 the SGLang Diffusion bridge: the backend behind `--reelcache-enable` (README inside)
│   │   └── longcat_factor_shard.py   factor-sharded archive of a context-parallel LongCat conditioning cache
│   └── _native/                    loader of the optional Rust extension (build it from native/; nothing prebuilt)
├── reelcache_sglang_models/        out-of-tree SGLang model package (SGLANG_EXTERNAL_MODEL_PACKAGE=reelcache_sglang_models)
│   ├── causal_wan/                 CausVid, LongLive 1.0 and RAVEN as profiles of SGLang's causal-Wan path
│   └── longcat_video/              LongCat-Video 13.6B direct-720p continuation port (DESIGN.md inside)
├── integrations/
│   └── lingbot/                    LingBot-World-Fast: runtime wrapper, model/archive/overlap adapters, setup, tests
├── native/
│   └── reelcache_native/           Rust/PyO3 crate: GIL-free HEVC NAL scanning and tile unpacking (README inside)
├── third_party/
│   └── sglang/                     SGLang 0.5.21 patches: 26 commits on e00930c5 (release/v0.5.21); README inside
├── examples/                       development drivers and benchmarks
├── scripts/                        CPU receipt (verify.py), official VBench evaluation, rank-study tools
└── tests/                          CPU test suite: 139 modules, helpers and fixtures
```

### `reelcache/`: core runtime

**Contracts, storage and lineage**

| Module | Purpose |
|---|---|
| `types.py` | Validated contracts (`ModelIdentity`, `MatrixKey`, `TensorContext`, `CodecSpec`, `EnergyTargets`, `ENERGY_TARGET_PROFILES`); no model imports |
| `errors.py` | Explicit failure modes for storage, representations and native state |
| `serialization.py` | Portable, bit-preserving tensor state; never executes pickle payloads |
| `store.py` | Immutable content-addressed objects, transactional roots, GC pins |
| `basis_versions.py` | Immutable basis row blocks and ordered version manifests |
| `cache_epochs.py` | Cache epochs (e.g. recache) and newest-version-on-path history resolution |
| `lineage.py` | Lineages: immutable state-tree nodes in the store |
| `lineage_commit.py` | Staged lineage commits, shared by synchronous and pipelined archival |
| `manager.py` | Clean-chunk publication, copy-on-write branches, bounded restoration (the general manager API) |
| `residency.py` | Bounded decoded residency with explicit lifetimes |
| `run_telemetry.py`, `telemetry.py` | Run telemetry kept out of content-addressed manifests; opt-in CPU-wall attribution |

**Rank selection and fitting**

| Module | Purpose |
|---|---|
| `rank_method.py` | Rank selection, verification and basis reuse (the paper's method) |
| `online_rank.py` | Online rank estimation state and fit growth |
| `predictor_registry.py` | Model-scoped online rank predictors with immutable, pinnable revisions |
| `reuse.py` | First-parent, projection-gated reuse of immutable FP16 bases |
| `factorization.py` | Deterministic factorization in the canonical channel domain |
| `gpu_fitting.py` | Owner-thread GPU fitting backends with matched stored-basis projection |
| `cusolver_eigh.py` | Batched symmetric eigensolver through cuSOLVER `cusolverDnXsyevBatched` |
| `exact_blas.py` | Explicit-precision BLAS: IEEE FP32 products on CUDA regardless of TF32 flags |
| `deferred_reads.py` | Lockstep per-matrix routines with batched host reads (no per-matrix synchronization) |
| `spectra.py` | Uncentered, complete-channel FP64 energy spectra (reference measurements) |
| `policy.py`, `calibration.py` | Measured whole-model representation profiles; profile registries from measured outcomes |
| `rank_analysis.py`, `study.py` | Numeric analysis and protocol/launch gates of the rank study |

**Codecs and packing**

| Module | Purpose |
|---|---|
| `codecs.py` | RAW, SVD and SVD+HEVC tensor representations |
| `_hevc.py` | File-free software HEVC (FFmpeg/libx265) for coefficient tiles; `REELCACHE_FFMPEG_BIN` |
| `node_packets.py` | Packed node streams: canonical canvases, device finalization, CPU encode |
| `nvenc.py` | NVENC HEVC encoder for packed all-intra streams (capability-qualified) |
| `nvdec.py` | Explicit, thread-owned NVDEC sessions |
| `factor_packs.py` | FP16 factor packs of the SVD-only baseline |

**Archival pipeline and scheduling**

| Module | Purpose |
|---|---|
| `archive_pipeline.py` | Ordered asynchronous archival of complete clean chunks |
| `execution.py` | Bounded CPU work with admission, cancellation and failure semantics |
| `raw_tier.py` | GPU and host raw-chunk tiers |
| `device_pressure.py` | Registry of trimmable device caches; GPU-memory pressure response |
| `accel_b.py` | Opt-in fitter GPU-footprint levers (FP32 passes, fused fingerprints, GPU finalize, green contexts) |
| `accel_c.py` | Opt-in, bytes-identical host-footprint levers (NVDEC validation, hash once, GIL fast paths, Rust) |
| `graph_capture.py` | CUDA-graph replay of fitter kernel sequences (experimental; keep off, see §7) |

**Restore**

| Module | Purpose |
|---|---|
| `archive_restore.py` | `LineageRestorer`: device restore of lineage history through the packed NVDEC path |
| `packed_decode.py` | NVDEC canvases of all-intra packets feeding the restore kernels |
| `decoder_pool.py` | Persistent NVDEC decoder pool with asynchronous submission and bounded canvases |
| `paired_decode.py` | K/V restoration on independently owned decoder sessions and streams |
| `restore_kernels.py` | Batched, fused reconstruction of restored matrices into native caches |
| `restore_staging.py` | Staging of node packs for device restore (the prepared tier) |
| `overlapped_restore.py` | Layer-overlapped restore: install layer by layer while the forward reads it |
| `prefetch.py` | One-layer-ahead restoration with charged leases |
| `decoded_tier.py` | Host-RAM tier of NVDEC-decoded coefficient canvases |

**Distributed and workloads**

| Module | Purpose |
|---|---|
| `distributed.py` | Ordered, owner-only KV transport (never creates process groups) |
| `sp_kv.py` | Tiered sequence-parallel KV history: hot local shards plus distributed cold decode |
| `factor_sharding.py` | Non-replicated distributed factor ownership |
| `session_workloads.py` | Session-tree rewind-and-generate workloads: settings, sweeps, feasibility, frozen bundles |
| `synthetic_workloads.py`, `workloads.py` | Synthetic session schedules; recorded browser events compiled into clean-boundary workloads |

### `reelcache/adapters/`: native adapters

These predate the SGLang bridge. They attach the runtime directly to native model code, and importing `reelcache`
never loads them or a model.

| Module | Purpose |
|---|---|
| `causvid.py` | Instance-local, clean-boundary integration with native CausVid |
| `causvid_benchmark.py` | Fresh CausVid execution and accounting helpers |
| `magi.py` | Instance-local MAGI Ulysses (sequence-parallel) clean-history adapter |
| `magi_native.py`, `magi_rank_worker.py`, `magi_preflight.py`, `magi_provenance.py`, `magi_vbench.py` | MAGI verification, isolated rank-study workers, model-free preflight bundles, provenance, and official-VBench cases |

### `reelcache/integrations/sglang/`: SGLang bridge

| Module | Purpose |
|---|---|
| `__init__.py` | Backend factory that SGLang's seam loads under `--reelcache-enable` |
| `bridge.py` | `CausalWanKVBridge`: the shared causal-Wan archive backend (capture, commit, restore, sessions) |
| `config.py` | `ReelCacheRuntimeConfig`: validated plain-data configuration (what `--reelcache-config` resolves to) |
| `profiles.py` | Bridge model profiles (data). LongLive 2.0 is built in; the others are registered by `reelcache_sglang_models` |
| `session.py` | Worker-owned session state and the worker-level controller |
| `controls.py` | Typed, plain-JSON session controls (pause, checkpoint, fork, rewind, close, raw tier, ...) |
| `capture.py`, `commit.py` | Clean-chunk capture transaction; lineage-keyed commit inputs |
| `lineage_archive.py` | Compressed session archive (`svd_hevc` and the baselines) on the runtime |
| `archive.py` | RAW session archive (`representation="raw"`, the exact reference) |
| `archive_eviction.py` | LRU eviction of compressed node payloads under `archive_budget_bytes` |
| `nodes.py` | Plain-data view of committed session nodes |
| `restore.py` | Archived history → engine RoPE → native `CausalSelfAttentionKVCache` slots |
| `visibility.py` | Visibility plans: which archived token ranges the next chunk attends to, and where they live |
| `generation_defer.py` | Generation-first archival deferral |
| `workload_telemetry.py` | Per-request workload telemetry |

### `reelcache_sglang_models/`: SGLang model package

- `causal_wan/`
  - `profiles.py`: profile data, with exact weight bindings (path, size, SHA-256) and RoPE/visibility/recache
    specifications;
  - `convert.py`: native-to-diffusers conversion, the LongLive LoRA merge, and `build_profile_dir`;
  - `registration.py`: pipeline registration;
  - `sglang_{configs,dit,stages,pipelines,cache}.py`: SGLang classes;
  - `sampling.py`, `recache.py`, `lifecycle.py`: torch-only or pure-Python semantics.
- `longcat_video/`: the direct-720p continuation port.
  - `spec.py`, `weights.py`: specification and exact weight binding;
  - pure-torch DiT math, RoPE, LoRA, schedule, text and video I/O;
  - context parallelism and the conditioning-cache seam;
  - `reelcache_backend.py`, `raw_tiers.py`, `cp_tiers.py`, `planning_gate.py`: archival;
  - `segment_resume.py`, `session_archive.py`, `session_provider.py`: segment resume and sessions;
  - the SGLang classes, and `DESIGN.md`.

Importing the package does nothing unless SGLang names it through `SGLANG_EXTERNAL_MODEL_PACKAGE`. In that case it
registers the CausVid/RAVEN/LongLive 1.0 profile pipelines and the LongCat-Video pipeline, and it raises on any
registration failure.

### `examples/`, `scripts/`, `tests/`

| Path | Contents |
|---|---|
| `examples/runtime_demo.py` | Durable clean chunks, restart and branching on small synthetic KV (CPU; §5) |
| `examples/benchmark_gpu_fitting.py`, `benchmark_gpu_restore.py`, `benchmark_nvdec.py`, `benchmark_paired_nvdec.py` | Fitting backends; software vs host/device NVDEC restore; NVDEC verification and timing |
| `examples/benchmark_interactive_sessions.py` | Session-tree workloads: `expand` / `check` / `generate` / `slots` / `validate-bank`; `run` needs the evaluation runner (not included) |
| `examples/native_causvid.py`, `causvid_native_reference.py`, `causvid_runtime_controls.py`, `causvid_rank_worker.py` | Native CausVid verification and study workers (bind a local CausVid source tree) |
| `examples/native_magi.py`, `benchmark_magi_layer_streaming.py`, `verify_sp_tiered_gpu.py`, `measure_sp_overlap.py`, `vbench_magi.py` | MAGI SP=2 and tiered sequence-parallel history checks (two GPUs) |
| `examples/vbench_causvid.py`, `rank_metrics.py` | Paired generation for VBench; strict paired SSIM and official VBench dimensions |
| `examples/interactive_workloads.py`, `prepare_synthetic_workloads.py` | Trace-mapped and synthetic workload DAGs |
| `scripts/verify.py` | Runs the CPU suite and writes a receipt bound to the hashed source tree (§9) |
| `scripts/evaluate_vbench.py`, `_vbench_official.py`, `_vbench_protocol.py` | Official VBench evaluation from pinned local implementations |
| `scripts/rank_study.py`, `rank_calibrate.py`, `rank_report.py`, `rank_preflight_compare.py` | Rank-study protocols, calibration and plots |
| `tests/` | 139 `test_*.py` modules (stdlib `unittest`), `core_support.py`, `step6_support.py`, `assets.py` (optional evaluation artifacts, §9), `fixtures/` |

## 4. Dependencies and installation

### 4.1 Dependencies

`import reelcache` needs only NumPy and PyTorch. Every other package is imported lazily by the feature that uses
it, so a missing optional package fails only that feature. The versions below are the ones the paper's
measurements ran with; other recent versions should work but are untested. `pyproject.toml` declares the core and
the optional extras, marked in the last column.

**Runtime (`reelcache`)**

| Package | Evaluated version | Used for | Declared |
|---|---|---|---|
| Python | 3.12.13 | everything | `requires-python >= 3.10` |
| `numpy` | 2.3.5 | core | required |
| `torch` | 2.13.0 (CUDA 13.0 build) | core: tensors, fitting, restore | required |
| `triton` | 3.7.1 (installed with PyTorch) | fused restore kernels (`restore_kernels.py`); fused fingerprints and FP32 passes (`accel_b.py`) | — |
| `PyNvVideoCodec` | 2.2.3 | NVENC encode, NVDEC decode (`nvenc.py`, `nvdec.py`) | extra `nvdec` |
| `nvidia-ml-py` (`pynvml`) | 13.615.71 | NVENC capability probe (`nvenc.py`) | — |
| FFmpeg with libx265 | 7.0.2 (static build) | software HEVC encode/decode and validation decode (`_hevc.py`) | via `REELCACHE_FFMPEG_BIN` or `imageio-ffmpeg` |
| `imageio-ffmpeg` | 0.5.1 | locating an FFmpeg executable | extra `hevc` |
| `cupy-cuda13x` | 14.2.0 | optional CuPy fitting backends (`factorization.py`) | extra `gpu-fit-cu13` (`gpu-fit` for CUDA 12) |
| `matplotlib`, `scikit-image`, `Pillow` | 3.11.2, 0.25.2, 11.3.0 | rank-study plots and metrics | extra `study` |

**Serving: SGLang Diffusion**

| Component | Evaluated version |
|---|---|
| **SGLang** | **0.5.21**: upstream commit `e00930c5489053f26d86b179cee0d087f846acbb` (branch `release/v0.5.21`, 2026-09-29) plus the 26 patches in [`third_party/sglang/`](third_party/sglang/README.md) |
| `sglang-kernel` | 0.4.7 |
| `flashinfer-python` | 0.6.18 |
| `torch` / `triton` | 2.13.0 (CUDA 13.0 build) / 3.7.1 |
| `transformers` / `diffusers` | 5.12.1 / 0.37.0 |
| CUDA toolkit / NVIDIA driver | 13.2 / ≥ 580 (qualified on 595.91) |
| GPU | RTX PRO 6000 Blackwell (sm_120); earlier runtime freezes also on A100 (sm_80) |
| Used by `reelcache_sglang_models` | `safetensors` 0.8.0 (profile conversion), `ftfy` 6.3.1 (prompt cleaning), `Pillow` 11.3.0 (LongCat-Video frames) |

**Tests** (§9; CPU only, run in a separate environment)

| Package | Version | Used by |
|---|---|---|
| Python / `torch` / `numpy` | 3.12.13 / 2.8.0 / 2.4.6 | the whole suite |
| FFmpeg with libx265 | 7.0.2 | codec tests |
| `decord`, `easydict` | 0.6.0, 1.13 | VBench evaluation tests |
| `scikit-image`, `imageio-ffmpeg`, `PyNvVideoCodec` | 0.26.0, 0.6.0, 2.2.2 | metric, VBench and decoder-exit tests |
| `safetensors`, `diffusers`, `Pillow` | 0.8.0, 0.39.0, 12.3.0 | LongCat-Video port tests (with the native LongCat-Video source) |
| `reelcache_native` | built from `native/reelcache_native` | `test_accel_c_native.py` |

**LingBot-World-Fast** (§8; its own environment)

| Component | Evaluated version |
|---|---|
| Official code | `Robbyant/lingbot-world` @ `a43bec7f8091c83e9b30b16b912f6fc906236fa6`, unmodified |
| Python / `torch` / `torchvision` | 3.12.13 / 2.8.0 (CUDA 12.8 build) / 0.23.0 |
| `flash-attn` | 2.8.3 (with sm_120 kernels) |
| `diffusers` / `einops` / `easydict` | 0.39.0 / 0.8.2 / 1.13 |
| `PyNvVideoCodec` | 2.2.3 |

**Optional Rust extension** (§4.4)

| Tool | Version |
|---|---|
| Rust (`rustc`, `cargo`) | 1.92.0 |
| `maturin` | 1.16.0 |
| Crates | `pyo3` 0.23.5, `memchr` 2.8.3 (pinned in `Cargo.lock`) |

**Other optional features**
- Official VBench evaluation (`scripts/evaluate_vbench.py`): pinned local VBench implementations, OpenCV,
  `decord` and `scikit-image` (`scripts/_vbench_official.py` lists them).
- Native MAGI adapters (`reelcache/adapters/magi*.py`): the MAGI source tree (its `inference` package),
  `flash-attn`, `flashinfer`, `transformers`, `einops` and `safetensors`.
- Example drivers: `omegaconf`, `clip`, `scikit-image` and `Pillow`.

### 4.2 Runtime

```bash
cd ReelCache
pip install -e ".[hevc]"                       # CPU: software HEVC (imageio-ffmpeg ships an FFmpeg with libx265)
pip install -e ".[hevc,nvdec,gpu-fit-cu13]"    # GPU: + PyNvVideoCodec (NVENC/NVDEC) + CuPy for CUDA 13
```

| Extra | Packages |
|---|---|
| `hevc` | `imageio-ffmpeg>=0.5` |
| `nvdec` | `PyNvVideoCodec>=2.2,<3` |
| `gpu-fit` / `gpu-fit-cu13` | `cupy-cuda12x>=13.6,<14` / `cupy-cuda13x>=14.2,<15` |
| `study` | `matplotlib`, `scikit-image`, `Pillow` |

The A100 has no NVENC, so `encoder="auto"` falls back to libx265 there. `REELCACHE_FFMPEG_BIN` selects the FFmpeg
executable explicitly.

### 4.3 SGLang Diffusion

```bash
git clone https://github.com/sgl-project/sglang.git && cd sglang
git checkout -b reelcache e00930c5489053f26d86b179cee0d087f846acbb      # SGLang 0.5.21
git am /path/to/ReelCache/third_party/sglang/patches/*.patch
pip install -e "python[diffusion]"
```

See [`third_party/sglang/README.md`](third_party/sglang/README.md) for what the 26 patches change. Without
`--reelcache-enable`, no archive backend is attached and no ReelCache code is imported. Two patches do fix stock
causal-Wan paths that the seam relies on (patches 5 and 7). The text-embedding cache and the workload timer stay off
unless enabled.

### 4.4 Optional Rust extension

`native_ext=true` uses a compiled `reelcache_native` module for three host-side hot spots. It is optional: without
it, the Python paths run, with identical outputs. To build it and place it in `reelcache/_native/`, see
[`native/reelcache_native/README.md`](native/reelcache_native/README.md). The test suite expects it (§9).

### 4.5 Model weights

- **CausVid, RAVEN, LongLive 1.0.**
  - The profiles in `reelcache_sglang_models/causal_wan/profiles.py` bind exact native checkpoint files (path, size,
    SHA-256) and a pinned Wan2.1-T2V-1.3B-Diffusers snapshot.
  - Every path derives from `MODEL_ROOT`, a placeholder (`/models`). Put the files there, or change `MODEL_ROOT`
    (profile fingerprints include the paths, so archives record which files were bound):

    | Path under `MODEL_ROOT` | Contents |
    |---|---|
    | `Wan2.1-T2V-1.3B/` | Wan2.1-T2V-1.3B: `config.json`, `diffusion_pytorch_model.safetensors`, T5, VAE, tokenizer (CausVid, LongLive 1.0) |
    | `causvid/autoregressive_checkpoint/model.pt` | CausVid generator |
    | `LongLive-1.3B/models/{longlive_base.pt,lora.pt}` | LongLive 1.0 base and rank-256 LoRA |
    | `RAVEN-model-weights/raven_model.pt`, `raven_weights/` | RAVEN generator; its T5, VAE and tokenizer |
    | `huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780…/` | the Diffusers components |
    | `reelcache_sglang_profiles/` | `PROFILE_MODEL_ROOT`: converted SGLang model directories |

  - `build_profile_dir` converts them into an SGLang model directory (diffusers names, merged LoRA, a
    `reelcache_profile.json` binding). It is re-verified at load time:

    ```python
    from reelcache_sglang_models.causal_wan.profiles import get_profile
    from reelcache_sglang_models.causal_wan.convert import build_profile_dir
    build_profile_dir(get_profile("causvid"))  # -> /models/reelcache_sglang_profiles/reelcache-causvid-480p-s12v1
    ```

    The SGLang registration matches these exact directories (`profile.sglang.model_path`), so serve them in place.
- **LongLive 2.0:** `Rabinovich/LongLive-2.0-5B-Diffusers` at revision
  `37664a64cfc38cdc5fabe88576961ef10646e4f1` (bridge profile `longlive2`).
- **LongCat-Video:** `reelcache_sglang_models/longcat_video/weights.py` (path + size + SHA-256; see `DESIGN.md`).
- **LingBot-World-Fast:** `integrations/lingbot/setup/` (pinned Hugging Face revisions plus a hash manifest).

## 5. Quick start without a GPU

`examples/runtime_demo.py` exercises the runtime on small synthetic KV, without a model:
1. commit clean chunks;
2. restart from the store;
3. restore, and branch with parent isolation;
4. report the restore error and the store bytes.

```bash
export REELCACHE_FFMPEG_BIN=/path/to/ffmpeg       # optional with the `hevc` extra installed
python examples/runtime_demo.py --output /tmp/reelcache-demo-raw  --codec raw
python examples/runtime_demo.py --output /tmp/reelcache-demo-hevc --codec svd_hevc --allow-lossy
```

The `raw` run restarts with relative KV errors of exactly 0. The `svd_hevc` run is lossy: about 2 % relative error
on these synthetic matrices. Both report `"parent_isolation": true`.

## 6. Serving with SGLang Diffusion

The bridge must know the served model's *profile*:
- LongLive 2.0's profile (`longlive2`) is built in and matched by model path.
- CausVid, RAVEN and LongLive 1.0 are passed inline as `profile_fields`.

`bridge_runtime_config` writes such a configuration. For LongLive 1.0 it also sizes the canvas staging that its
recache nodes need:

```python
import json
from reelcache_sglang_models.causal_wan.profiles import get_profile
from reelcache_sglang_models.causal_wan.lifecycle import bridge_runtime_config
cfg = bridge_runtime_config(get_profile("causvid"), store_root="/local/nvme/reelcache-store",
                            defer_archival_during_generation=True, archival_defer_inflight="overlap")
json.dump(cfg, open("reelcache.json", "w"), indent=1)
```

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE=reelcache_sglang_models      # registers the causal-Wan profiles and LongCat-Video
sglang serve --model-path /models/reelcache_sglang_profiles/reelcache-causvid-480p-s12v1 \
    --reelcache-enable --reelcache-config reelcache.json
```

- `--reelcache-enable`: archive clean causal KV chunks and accept session requests. Startup fails if
  `reelcache.integrations.sglang` cannot be imported; there is no silent dense fallback.
- `--reelcache-config`: a JSON object, or the path of a JSON file, holding `ReelCacheRuntimeConfig` fields (§7).
- `--reelcache-hevc-inter-prediction`: inter-coded HEVC instead of all-intra. Off by default.

**Generation requests** are ordinary `POST /v1/videos` jobs that carry a `causal_kv_session` handle:

```jsonc
{"prompt": "...", "causal_kv_session": {"session_id": "s1"}}                                    // new session
{"prompt": "...", "causal_kv_session": {"session_id": "s1", "action": "resume"}}                // continue the head
{"prompt": "...", "causal_kv_session": {"session_id": "s1", "action": "rewind", "checkpoint": 4}} // after chunk 4
{"prompt": "...", "causal_kv_session": {"session_id": "s2", "action": "fork", "from_session": "s1", "checkpoint": 6}}
```

The handle accepts these keys:

| Key | Meaning |
|---|---|
| `session_id` | the session |
| `action` | `generate` (default), `resume`, `fork` or `rewind` |
| `from_session` | the fork source |
| `checkpoint` | a node reference or a chunk count |
| `pause_after_chunks` | pause at the clean boundary after that many chunks |
| `durable` | return only once the chunks are archived |
| `output` | `new` (only the new chunks; default) or `full` |
| `decode_context_latents` | latent frames decoded first, to warm the causal VAE |
| `overlap_restore` | per-request override of the overlapped restore |
| `replace` | `generate` only |

**Session controls** need no generation:
- `GET /v1/sessions`, `GET` / `DELETE /v1/sessions/{id}`;
- `POST /v1/sessions/{id}/{open,pause,fork,rewind,close}`;
- `GET` / `POST /v1/sessions/{id}/checkpoints`, `GET` / `POST /v1/sessions/{id}/raw_tier`;
- `GET /v1/session_capabilities`, `GET /v1/session_requests/{request_id}`, `POST /v1/session_gc`.

No endpoint accepts a filesystem path.

The paper's measurements drove SGLang's GPU worker in-process, through the evaluation runner, which is not
included. The HTTP surface is covered by the SGLang unit tests in the patch series. The bridge's detailed
semantics are documented in [`reelcache/integrations/sglang/README.md`](reelcache/integrations/sglang/README.md):
publication across requests, resident KV, the seam protocol, session controls, profiles, and attaching a stage.

## 7. Configuration

`--reelcache-config` resolves to `reelcache.integrations.sglang.config.ReelCacheRuntimeConfig`: a frozen, validated
dataclass of plain values, with `store_root` required and every other field defaulted. *Content fields*
(method, codec, targets) are hashed into session identities. Everything else is run configuration: paths,
budgets, workers, decoders, verification and scheduling.

| Group | Main fields (defaults) |
|---|---|
| Method | `representation` (`svd_hevc`; also `raw`, `raw_offload`, `svd`, `hevc`, `latents_only`), `rank_policy` (`adaptive`), `energy_profile` or `energy_key` + `energy_value`, `reuse` / `refit` (`true`), `refit_rule` (`paper`), `rate_lag` (3) |
| Codec | `hevc_qp` (8), `hevc_pix_fmt` (`yuv444p`), `hevc_preset` (`ultrafast`), `encoder` (`auto` = NVENC if present, else x265), `validate_decode` (`true`), `validation_decoder` (`ffmpeg`) |
| Store and sessions | `store_root`, `store_io` (`buffered`; `nocache` keeps the page cache from acting as an extra tier), `durability` (`pause`), `tree_retention` (`head`; `full` keeps every node a rewind target), `tree_switch` (`drain`; `detach`), `archive_budget_bytes`, `max_store_bytes` |
| Tiers | `raw_tier` (`true`), `gpu_raw_budget_bytes`, `host_raw_budget_bytes`, `raw_tier_retain` (`drop`; `lru`), `prepared_budget_bytes` (4 GiB), `decoded_tier_bytes` (0) |
| Restore | `restore_mode` (`archive`; `recompute`), `decoder` (`nvdec_packed`), `nvdec_workers` (8), `overlap_restore` (`true`), `rope_mode` (`fused`), `verify_reads` / `verify_bases` (`true`) |
| Scheduling | `defer_archival_during_generation` (`false`), `archival_defer_inflight` (`wait`; `overlap`), `fitter_stream` (`dedicated`), `archival_cpus` (`job`), `defer_archival_during_decode`, `max_unpublished` (4) |

The evaluation's ReelCache arm below is the resolved LongLive 1.0 configuration, with its store path replaced and
its inline `profile_fields` omitted; its budgets are per model. The full resolved file ships as
`integrations/lingbot/reelcache_lingbot/bridge_config_template.json`.

```json
{
  "store_root": "/local/nvme/reelcache-store",
  "representation": "svd_hevc", "reuse": true, "refit": true, "refit_rule": "paper", "rate_lag": 3,
  "hevc_qp": 8, "hevc_pix_fmt": "yuv444p", "encoder": "auto",
  "decoder": "nvdec_packed", "nvdec_workers": 8, "overlap_restore": true,
  "raw_tier": true, "raw_tier_retain": "lru",
  "gpu_raw_budget_bytes": 5175705600, "host_raw_budget_bytes": 12591583964, "prepared_budget_bytes": 12591583964,
  "tree_retention": "full", "tree_switch": "detach", "store_io": "nocache",
  "defer_archival_during_generation": true, "archival_defer_inflight": "overlap",
  "archival_cpus": "inherit", "workload_telemetry": true
}
```

### Acceleration levers (all opt-in; default off)

These reduce archival's interference with generation and the exposed restore time. The defaults reproduce the
evaluated runtime bit for bit. Identity classes:
- **(i)** bitwise: identical stored bytes and restored KV;
- **(ii)** gated numerics: identical decisions and ranks, `|ΔE| ≤ 1e-6`.

| Field | What it does | Class |
|---|---|---|
| `decoded_tier_bytes`, `decoded_tier_route="all"` (or `"deficit"`) | Keeps NVDEC-decoded coefficient canvases of hot nodes in pinned host RAM and routes restores around the decoder. Size it to the hot working set: a tier smaller than the working set mostly misses | (i) |
| `fitter_fp64="off"` (or `"fp32:<pass>+..."`) | FP32 instead of FP64 for the fitter's gated measurement passes (FP64 runs at 1/64 rate on RTX-class GPUs) | (ii); stored payloads were identical on all tested nodes |
| `fused_fingerprints` | Source fingerprints as one exact-integer Triton kernel per matrix | (i) |
| `finalize_on_fit_device` | Finalize on the GPU even for host-tensor sources (LingBot) | (ii): affines may differ in the last bits |
| `validation_decoder="nvdec"` | NVDEC instead of FFmpeg for the encode-time validation decode | (i) |
| `hash_once`, `seal_prepare_after_publish`, `gil_fastpaths`, `nvenc_device_input`, `native_ext` | Host-side savings: digests reused, prepared-pack sealing after publication, vectorized NAL/unmosaic/record paths, NVENC from device memory, the Rust extension | (i) |
| `fitter_green_context_sms`, `fitter_green_workqueue` | Cap the fitter's SMs with a CUDA green context (`torch.cuda.green_contexts`; the work-queue option needs CUDA ≥ 13.1). This is a rate limiter, not an efficiency gain | (ii) |
| `fitter_cuda_graphs` | **Do not enable.** Online graph capture breaks generation's device synchronization | — |

Recommended settings:
- **Wan models:** `fitter_fp64="off"`, `fused_fingerprints`, `validation_decoder="nvdec"`, `hash_once`,
  `seal_prepare_after_publish`, `gil_fastpaths`, and `decoded_tier_route="all"` with `decoded_tier_bytes` ≥ the
  hot working set. Optionally `nvenc_device_input` and `native_ext`.
- **LingBot:** the same compression flags, `finalize_on_fit_device`, and its overlap options (§8).

Measured on RTX PRO 6000:
- **CausVid / RAVEN:**
  - with a 48 GiB decoded tier, end-to-end next-chunk P50 dropped 13–15 %, with bit-identical frames;
  - under fully overlapped archival, RAVEN's generation slowdown fell from 32.8 % to 21.5 %;
  - CUDA graphs, green contexts and the Rust extension did not reduce interference, which is SM contention.
- **LingBot:** P50 dropped 6.5 %.

## 8. LingBot-World-Fast

LingBot runs the unmodified official pipeline outside SGLang. `integrations/lingbot/` contains:
- the runtime wrapper `lingbot_rt.py`, with its `API.md`;
- the `reelcache_lingbot` adapters: `LingBotModel`, `ReelCacheArchive`, and the layer-overlapped restore;
- weight setup and a portable `env.sh`;
- 9 CPU tests.

See [`integrations/lingbot/README.md`](integrations/lingbot/README.md).

## 9. Tests

```bash
# from the repository root, CPU only
CUDA_VISIBLE_DEVICES= python scripts/verify.py --output /tmp/reelcache-receipt.json
python integrations/lingbot/tests/test_lb_overlap.py      # LingBot adapters (CPU)
```

`scripts/verify.py` runs every `tests/test_*.py` module (stdlib `unittest`). It writes a receipt bound to the
SHA-256 of `reelcache/`, `reelcache_sglang_models/`, `tests/`, `examples/`, `scripts/` and `pyproject.toml`.
`--require-no-skips` turns any skip into a failure; use it when the test assets below are configured.

The suite's requirements:
- an FFmpeg 7 build with libx265: `REELCACHE_FFMPEG_BIN`, else `ffmpeg` on `PATH`, else imageio-ffmpeg's binary;
- `decord==0.6.0` and `easydict==1.13` importable, for the VBench evaluation tests (`test_vbench_evaluation.py`);
- the Rust extension built into `reelcache/_native/`, for `tests/test_accel_c_native.py`;
- optionally, the test assets below.

Tests create the scratch directories `.test-work/`, `.test-artifacts/` and `.core-test-artifacts/` under the
repository root; they are git-ignored. The SGLang-side unit tests live in the patch series:
`python/sglang/multimodal_gen/test/unit/test_causal_kv_*.py` and the others listed in `third_party/sglang/README.md`.

**Test assets.** A few tests compare against artifacts of the original evaluation that this repository does not
ship. They read them from the directory `REELCACHE_TEST_ASSETS` (helper `tests/assets.py`) or from a per-asset
override, and are skipped when neither is set. With the directory set, a missing file fails the test.

| Path under `$REELCACHE_TEST_ASSETS` | Contents | Read by | Per-asset override |
|---|---|---|---|
| `frozen/freeze_w141/{tree,TREE_ID}` | the frozen W1.4.1 runtime tree under this release's names, run beside this one for record equality | `test_generation_first.py` | `REELCACHE_GENFIRST_BASE_TREE` (tree) |
| `frozen/freeze_w12/tree` | the frozen W1.2 runtime tree under this release's names (drain-mode record comparison) | `test_wl_detach.py` | `REELCACHE_W12_TREE` |
| `feasibility/item09.json` | the weight-hash receipt of the causal-Wan bindings, with paths under `MODEL_ROOT` | `test_causal_wan_profiles.py` | |
| `upstream/causvid/modeling/flow_match.py` | CausVid's scheduler source | `test_causal_wan_profiles.py` | |
| `upstream/LongLive1.0/utils/scheduler.py` | LongLive 1.0's scheduler source | `test_causal_wan_profiles.py` | |
| `longcat/source`, `longcat/weights` | LongCat-Video's native source @ `6b3f4b8` and its weights | `test_longcat_video_port.py`; the source also `test_longcat_video_context_parallel.py` | `REELCACHE_LONGCAT_VIDEO_SOURCE`, `REELCACHE_LONGCAT_VIDEO_WEIGHTS` |
| `feasibility/weights_sha256_table.csv` | the LongCat-Video weight-hash table, with paths under `/models/LongCat-Video` | `test_longcat_video_port.py` | |
| `step4/native_causvid/gpu-optimization-controls-v1/`, `step4/sglang_integration/runtime_gate/step4_gpu/capture/causvid_t00_4387197/` | real CausVid KV captures in this release's tensor format (bit-identical tensors) | `test_step4_real_data.py` | |
| `lingbot/repo` | the official LingBot-World source @ `a43bec7` | `integrations/lingbot/tests/test_lb_overlap.py` (lean RoPE) | `LB_REPO` |

## 10. Deployment-specific paths

Machine-specific locations are placeholders or settings; nothing points at the original cluster.

| Where | What | Default |
|---|---|---|
| `reelcache_sglang_models/causal_wan/profiles.py` | `MODEL_ROOT`, from which `WAN_BASE_DIR`, `RAVEN_WEIGHTS_DIR`, `HF_HUB_CACHE` (`DIFFUSERS_SNAPSHOT`), `PROFILE_MODEL_ROOT` (the output root of `build_profile_dir`) and the checkpoint `FileBinding` paths derive (§4.5) | `/models` |
| `reelcache_sglang_models/longcat_video/weights.py` | `DEFAULT_CHECKPOINT_DIR` | `/models/LongCat-Video` |
| `ReelCacheRuntimeConfig.store_root` | the archive store (§7) | required |
| `examples/benchmark_interactive_sessions.py` | the root of the evaluation runner, which is not included (`--runner-root` / `WL_RUNNER_ROOT`) | none; `run` requires it |
| `integrations/lingbot/` | `LB_REPO`, `LB_CKPT`, `LB_WORK`, `LB_PY`, `LB_SCRATCH`, `HF_BIN` | placeholders (its README) |
| the tests in §9 | `REELCACHE_TEST_ASSETS` and the per-asset overrides | unset: those tests skip |

## 11. Provenance

The paper's runtime was developed under an internal working name. This release is that code with the name replaced
by ReelCache everywhere: packages, modules, classes, functions, CLI flags, environment variables, file names,
documentation, and format identifiers such as `reelcache.tensor.v1` and the tensor-file magic. The paths of the
original cluster were then replaced by placeholders and settings (§10); the tests that compare against evaluation
artifacts read them from `REELCACHE_TEST_ASSETS` (§9).

- **Runtime** (`reelcache/`, `reelcache_sglang_models/`, `native/`, `examples/`, `scripts/`, `tests/`, `pyproject.toml`).
  - Source: the composed tree `ALL_v1` (tree id `b1f55ab7573dd856`). That tree is the paper's W1.4.3 runtime freeze
    (`63b731cf19f6b384`), plus two opt-in restore flags (`pipelined_restore`, and `verify_bases=false` to skip
    hashing) and the acceleration levers of §7. With every new flag at its default, its archives (every stored object
    kind) and generated latents are bit-identical to that freeze. This was checked on GPU for CausVid and RAVEN.
  - Checked: applying the rename to `ALL_v1` reproduces 278 of its 298 code files byte for byte; its prebuilt `.so`
    is not shipped. The other 20 differ only by these hand edits.
    - For the rename:
      - `pyproject.toml`: the distribution name.
      - Six test modules: hashes they pin that the rename changes. Each new value is proven to be the digest of the
        same input with the renamed identifiers.
      - `test_generation_first.py`, `test_wl_detach.py`: they compare against renamed copies of the frozen W1.4.1
        and W1.2 trees.
      - `test_step4_real_data.py`: it reads converted copies of the real CausVid captures, with bit-identical tensors.
      - `test_step7_store_faults.py`: one test's corruption is pinned to a structural byte. It used to flip the
        manifest's middle byte, which the longer identifiers moved.
    - For the cluster paths:
      - `causal_wan/profiles.py` (`MODEL_ROOT`), `longcat_video/weights.py` (`DEFAULT_CHECKPOINT_DIR`) and
        `examples/benchmark_interactive_sessions.py` (no default runner root).
      - `test_wl_prompt_switch.py`: the profile identities it pins include the bound weight paths. Each new value is
        proven to be the digest of the same input with the paths mapped to `/models`, nested fingerprints recomputed.
      - `test_generation_first.py`, `test_wl_detach.py`, `test_causal_wan_profiles.py`, `test_longcat_video_port.py`,
        `test_step4_real_data.py`: they find their artifacts through `tests/assets.py` (new);
        `test_longcat_video_context_parallel.py` skips its Gloo tests, whose workers load the LongCat-Video source,
        when that asset is not configured.
      - `test_causal_wan_profiles.py`: the profile-directory tampering test binds stand-in component files, not the
        Diffusers snapshot on disk.
      - `test_sglang_bridge.py`, `test_vbench_causvid.py`, `test_step7_cpu_pool.py`: synthetic paths, a mocked mount
        point and a comment.
  - Content ids differ from the evaluation's only through the renamed identifiers and, for the causal-Wan profiles,
    the placeholder weight paths (a profile's fingerprint includes them). Stores written by the evaluated runtime are
    not readable by this release.
- **SGLang:** the evaluated SGLang commit `043c0204cffef4c75262ec7e5ae2a2141645fe85` is upstream `e00930c5`
  (0.5.21) plus 26 commits. The patches here are those commits under the same rename, plus one test line (see
  [`third_party/sglang/README.md`](third_party/sglang/README.md)). They apply to give git tree `06209d51…`.
- **LingBot integration:** the runtime wrapper and the model/archive/overlap adapters used in the evaluation. The
  edits are import paths, environment variables with placeholder defaults in place of cluster paths, the shipped
  config template and the rename (see its README).
- **This repository's checks** (final tree):
  - CPU suite with the test assets (§9): 2,431 tests, 0 failures, 0 errors, 0 skips (`scripts/verify.py
    --require-no-skips` on a node-local copy, with the Rust extension built from `native/reelcache_native`;
    CPython 3.12.13, PyTorch 2.8, CPU only), plus the LingBot tests (9/9).
  - The same without the test assets, as in a fresh clone: 2,424 tests run, 0 failures, 0 errors, 29 skipped, all of
    them the asset-dependent tests of the six modules in §9 (a class skipped in its setup counts once). LingBot: 8
    passed, 1 skipped.
  - SGLang unit tests (seam tests plus related upstream modules, CPU), renamed SGLang with this repository versus
    the evaluated SGLang with `ALL_v1`: 632 passed, 33 failed, 40 skipped in both, with identical failing tests.
    None of the 12 seam test modules fails. The 33 failures are upstream `test_server_args.py` cases that depend
    on the machine (offload placement decided without a GPU; a hard-coded `/root/.cache` path).
  - GPU smoke (one RTX PRO 6000, serving environment): the 20 test modules with CUDA-dependent paths, 413 tests,
    run with the GPU visible on this repository and on `ALL_v1`. Results are identical: 409 passed, 1 skipped, and
    the same 3 tests fail in both. Two are LongCat-Video session-provider tests whose eager-overlap memory check
    finds no room on this node (43 GiB needed, 0 GiB counted as available). The third asserts that CUDA is not yet
    initialized, which depends on test order within one process.

## 12. What is not included

- Experiment harnesses and runners: the interactive session runner, the sustained-overlap and acceleration
  benchmarks, the LingBot evaluation runner, Slurm jobs.
- Workload bundles, authored prompt banks and datasets.
- Evaluation results, figures, decks, reports and the paper sources.
- Frozen runtime trees, model weights, stores and any built binaries (`.so`, wheels).

The packages here declare no license. The SGLang patches derive from SGLang (Apache-2.0); the Rust crate's
manifest declares MIT.
