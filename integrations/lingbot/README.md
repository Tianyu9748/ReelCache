# LingBot-World-Fast integration

[LingBot-World-Fast](https://github.com/Robbyant/lingbot-world) is an image-to-video world model with camera
control, built on Wan2.2 (task `i2v-A14B` in its code). Its model geometry:
- 40 layers, 40 heads, head dim 128, so 5,120 KV channels per layer;
- chunks of 3 latent frames;
- one persistent self-attention cache: each chunk's KV is written once, by a clean t = 0 pass, and never
  recomputed;
- attention over the full history by default.

There is no SGLang pipeline for it. ReelCache therefore drives the unmodified official pipeline
(`wan.WanI2VFast`, `Robbyant/lingbot-world` @ `a43bec7f8091c83e9b30b16b912f6fc906236fa6`) chunk by chunk, and
archives and restores its KV with the same runtime the SGLang bridge uses
(`reelcache.integrations.sglang.lineage_archive.LineageSessionArchive`).

## Layout

```
integrations/lingbot/
├── runtime/
│   ├── lingbot_rt.py      # wraps the official pipeline: env/distributed setup, pipeline construction, camera
│   │                      # action paths, start-image preparation, uint8 video conversion, and ChunkKV (hooks
│   │                      # that capture a chunk's pre-RoPE K and V from its clean pass and inject restored K/V)
│   ├── API.md             # lingbot_rt reference: constants, functions, ChunkKV, performance notes
│   └── env.sh             # runtime environment (PYTHONPATH, caches, offline HF); source it inside a job
├── reelcache_lingbot/        # importable package (put integrations/lingbot on PYTHONPATH)
│   ├── lb_model.py        # LingBotModel: the official generate() loop, one chunk at a time, on ONE shared
│   │                      # self-attention cache; window installs, restore destinations, lean RoPE.
│   │                      # FakeModel: a CPU stand-in with the same interface (tests)
│   ├── lb_archive.py      # ReelCacheArchive: one lineage node per chunk; fork on rewind; GPU-resident sources;
│   │                      # bounded planning memory; ablation arms (REPRESENTATIONS). FakeArchive: CPU stand-in
│   ├── lb_overlap.py      # OverlapSession / StallEvents: layer-overlapped restore through forward pre-hooks on
│   │                      # blocks[l].self_attn, with per-layer CUDA events (no device synchronize)
│   └── bridge_config_template.json   # the resolved Wan ReelCache-arm config that archive_config() starts from
├── setup/
│   ├── download_weights.sh   # pinned Hugging Face revisions; only T5, VAE and lingbot_world_fast/ (no MoE experts)
│   ├── build_manifest.py     # path / bytes / sha256 manifest, cross-checked against the HF LFS oids
│   ├── weights.sha256sum     # expected hashes (29 files)
│   └── weights_manifest.json # the verified manifest of the evaluation's checkpoint directory (paths as $LB_CKPT/...)
└── tests/test_lb_overlap.py  # 9 CPU tests: overlap hooks, overlapped == synchronous restore (bitwise),
                              # option parsing, lean RoPE == the model's RoPE (bitwise)
```

## Setup

1. Clone the official repository at `a43bec7f8091c83e9b30b16b912f6fc906236fa6` (unmodified) and point `LB_REPO`
   at it.
2. Create a Python 3.12 environment with:
   - torch 2.8.0+cu128, flash-attn 2.8.3 (built with sm_120 kernels for Blackwell), diffusers 0.39;
   - `easydict==1.13`;
   - `PyNvVideoCodec==2.2.3`, for NVENC/NVDEC;
   - ReelCache's own dependencies.
   Point `LB_PY` at its interpreter.
3. Download the weights with `setup/download_weights.sh` (set `LB_CKPT` or `LB_REPO`, `LB_WORK` and `HF_BIN`;
   the checkpoint directory name must contain `cam`). Then check them with `python setup/build_manifest.py`.
4. `source integrations/lingbot/runtime/env.sh` from the ReelCache root, inside your job. It sets:
   - `PYTHONPATH=<ReelCache>:<ReelCache>/integrations/lingbot:<runtime>:$LB_REPO`;
   - node-local caches (`LB_SCRATCH`) and offline Hugging Face; `REELCACHE_FFMPEG_BIN` is passed on only if you set it.

   The defaults are placeholders, so export your own `LB_*` values first.

Path overrides:

| Variable | Used by | Default |
|---|---|---|
| `LB_REPO` | `lingbot_rt.REPO`, tests, `env.sh` | `/models/lingbot-world` (placeholder; the tests skip the source check when unset, or use `$REELCACHE_TEST_ASSETS/lingbot/repo`) |
| `LB_WORK` | `lingbot_rt.WORK_ROOT` (camera paths, Triton cache), downloads | `~/.cache/reelcache/lingbot` |
| `LB_CKPT` | `env.sh`, `setup/*` | `$LB_REPO/lingbot-world-base-cam` |
| `LB_PY` | `env.sh`, `lingbot_rt.VENV_PYTHON` | `env.sh`: the `python3` on `PATH`; `lingbot_rt`: the running interpreter |
| `LB_SCRATCH` | `env.sh`, `lingbot_rt.prepare_env` (temp and caches) | `${TMPDIR:-/tmp}/reelcache-lingbot-<Slurm job id or pid>` |
| `REELCACHE_LINGBOT_RT` | `lb_model.LINGBOT_RT` | `integrations/lingbot/runtime` |
| `REELCACHE_LINGBOT_TEMPLATE` | `lb_archive.TEMPLATE` | `reelcache_lingbot/bridge_config_template.json` |
| `REELCACHE_FFMPEG_BIN` | software HEVC (CPU tests) | unset: `ffmpeg` on `PATH`, then imageio-ffmpeg's binary (the evaluation used FFmpeg 7.0.2) |

## Using it

`LingBotModel` (GPU) or `FakeModel` (CPU) holds the shared cache. `ReelCacheArchive` archives committed chunks and
restores histories into the model's cache:

1. `model.build()` and `model.prepare_session(sid, prompt, image_path, max_chunks)` once per session.
2. Per request:
   - restore the target's history with `archive.restore(head_key, chunks, model.destination_tensors, ...)`,
     either synchronously or overlapped with the first forward;
   - `model.generate(sid, chunk_index, generator, capture_pre_k=True)`;
   - `archive.commit(session, key, parent_key, chunk_index, k_pre, v, latents, ready_event=...)`;
   - `model.decode(...)`.
3. Choose the archive arm with `ReelCacheArchive(..., arm="reelcache")` (`abl_a` … `abl_e` are the ablation ladder).
4. Pass options through `extra`:
   - ReelCache config overrides (any `ReelCacheRuntimeConfig` field);
   - the harness options in `lb_archive.HARNESS_EXTRA`, which are popped before the ReelCache config is built:
     - `lb_overlap_restore` (default `false`);
     - `lb_overlap_release_behind` (default 2);
     - `lb_native_rope` (`"model"` or `"lean"`);
     - `lb_restore_group_chunks` (default 0, all chunks in one restore call; `g > 0` means synchronous calls of at
       most `g` chunks);
     - `lb_overlap_stream_priority`.

Recommended settings:
- the Wan compression flags ([acceleration levers](../../docs/configuration.md#acceleration-levers));
- `finalize_on_fit_device=true`;
- `lb_overlap_restore=true`, `lb_overlap_release_behind=2`, `lb_native_rope="lean"`.

`finalize_on_fit_device` is identity class (ii): the decisions and the basis are identical, but the coefficient
affines differ in the last bits. Overlapped restore and lean RoPE are bitwise-identical to the synchronous path.

The evaluation's request runner is not part of this repository: arms, workload bundles, timing and records.
`lb_model.py` and `lb_archive.py` are the model and archive layers it drove.

## Changes from the evaluated harness copies

Behavior is unchanged. The edits are:
- package name `wl.` → `reelcache_lingbot.`;
- the runtime's former internal name replaced by ReelCache, as everywhere in this repository (the archive class
  `ReelCacheArchive`, the method arm `"reelcache"`, the config template's identifiers);
- `LINGBOT_RT` points at `runtime/` beside the package;
- the config template ships with the package;
- cluster paths became environment-overridable, with placeholder defaults (`/models/...`, `~/.cache/reelcache/lingbot`,
  `$TMPDIR`); the config template's store root and profile model paths are placeholders as well, with the profile
  fingerprints recomputed for those paths, and the weights manifest writes paths as `$LB_CKPT/<file>`;
- the comments were updated to match.

`env.sh` is a portable adaptation of the evaluation's script.
