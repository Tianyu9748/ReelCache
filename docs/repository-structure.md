# Repository structure

```
ReelCache/
├── README.md                       overview: what ReelCache is for, how it works, the commands
├── docs/                           detailed documentation (this folder)
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

## `reelcache/`: core runtime

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
| `graph_capture.py` | CUDA-graph replay of fitter kernel sequences (experimental; keep off, see [configuration](configuration.md#acceleration-levers)) |

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

## `reelcache/adapters/`: native adapters

These predate the SGLang bridge. They attach the runtime directly to native model code, and importing `reelcache`
never loads them or a model.

| Module | Purpose |
|---|---|
| `causvid.py` | Instance-local, clean-boundary integration with native CausVid |
| `causvid_benchmark.py` | Fresh CausVid execution and accounting helpers |
| `magi.py` | Instance-local MAGI Ulysses (sequence-parallel) clean-history adapter |
| `magi_native.py`, `magi_rank_worker.py`, `magi_preflight.py`, `magi_provenance.py`, `magi_vbench.py` | MAGI verification, isolated rank-study workers, model-free preflight bundles, provenance, and official-VBench cases |

## `reelcache/integrations/sglang/`: SGLang bridge

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

## `reelcache_sglang_models/`: SGLang model package

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

## `examples/`, `scripts/`, `tests/`

| Path | Contents |
|---|---|
| `examples/runtime_demo.py` | Durable clean chunks, restart and branching on small synthetic KV (CPU; [usage](usage.md#quick-start-without-a-gpu)) |
| `examples/benchmark_gpu_fitting.py`, `benchmark_gpu_restore.py`, `benchmark_nvdec.py`, `benchmark_paired_nvdec.py` | Fitting backends; software vs host/device NVDEC restore; NVDEC verification and timing |
| `examples/benchmark_interactive_sessions.py` | Session-tree workloads: `expand` / `check` / `generate` / `slots` / `validate-bank`; `run` needs the evaluation runner (not included) |
| `examples/native_causvid.py`, `causvid_native_reference.py`, `causvid_runtime_controls.py`, `causvid_rank_worker.py` | Native CausVid verification and study workers (bind a local CausVid source tree) |
| `examples/native_magi.py`, `benchmark_magi_layer_streaming.py`, `verify_sp_tiered_gpu.py`, `measure_sp_overlap.py`, `vbench_magi.py` | MAGI SP=2 and tiered sequence-parallel history checks (two GPUs) |
| `examples/vbench_causvid.py`, `rank_metrics.py` | Paired generation for VBench; strict paired SSIM and official VBench dimensions |
| `examples/interactive_workloads.py`, `prepare_synthetic_workloads.py` | Trace-mapped and synthetic workload DAGs |
| `scripts/verify.py` | Runs the CPU suite and writes a receipt bound to the hashed source tree ([testing](testing.md)) |
| `scripts/evaluate_vbench.py`, `_vbench_official.py`, `_vbench_protocol.py` | Official VBench evaluation from pinned local implementations |
| `scripts/rank_study.py`, `rank_calibrate.py`, `rank_report.py`, `rank_preflight_compare.py` | Rank-study protocols, calibration and plots |
| `tests/` | 139 `test_*.py` modules (stdlib `unittest`), `core_support.py`, `step6_support.py`, `assets.py` (optional evaluation artifacts, [testing](testing.md#test-assets)), `fixtures/` |
