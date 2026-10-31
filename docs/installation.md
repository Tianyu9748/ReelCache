# Installation

The Python packages are `reelcache` (the runtime) and `reelcache_sglang_models` (the SGLang model package), in one
distribution named `reelcache`. Serving uses **SGLang 0.5.21** with the 26 patches in
[`third_party/sglang/`](../third_party/sglang/README.md).

## Dependencies

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
| **SGLang** | **0.5.21**: upstream commit `e00930c5489053f26d86b179cee0d087f846acbb` (branch `release/v0.5.21`, 2026-09-29) plus the 26 patches in [`third_party/sglang/`](../third_party/sglang/README.md) |
| `sglang-kernel` | 0.4.7 |
| `flashinfer-python` | 0.6.18 |
| `torch` / `triton` | 2.13.0 (CUDA 13.0 build) / 3.7.1 |
| `transformers` / `diffusers` | 5.12.1 / 0.37.0 |
| CUDA toolkit / NVIDIA driver | 13.2 / ≥ 580 (qualified on 595.91) |
| GPU | RTX PRO 6000 Blackwell (sm_120); earlier runtime freezes also on A100 (sm_80) |
| Used by `reelcache_sglang_models` | `safetensors` 0.8.0 (profile conversion), `ftfy` 6.3.1 (prompt cleaning), `Pillow` 11.3.0 (LongCat-Video frames) |

**Tests** ([testing](testing.md); CPU only, run in a separate environment)

| Package | Version | Used by |
|---|---|---|
| Python / `torch` / `numpy` | 3.12.13 / 2.8.0 / 2.4.6 | the whole suite |
| FFmpeg with libx265 | 7.0.2 | codec tests |
| `decord`, `easydict` | 0.6.0, 1.13 | VBench evaluation tests |
| `scikit-image`, `imageio-ffmpeg`, `PyNvVideoCodec` | 0.26.0, 0.6.0, 2.2.2 | metric, VBench and decoder-exit tests |
| `safetensors`, `diffusers`, `Pillow` | 0.8.0, 0.39.0, 12.3.0 | LongCat-Video port tests (with the native LongCat-Video source) |
| `reelcache_native` | built from `native/reelcache_native` | `test_accel_c_native.py` |

**LingBot-World-Fast** ([its README](../integrations/lingbot/README.md); its own environment)

| Component | Evaluated version |
|---|---|
| Official code | `Robbyant/lingbot-world` @ `a43bec7f8091c83e9b30b16b912f6fc906236fa6`, unmodified |
| Python / `torch` / `torchvision` | 3.12.13 / 2.8.0 (CUDA 12.8 build) / 0.23.0 |
| `flash-attn` | 2.8.3 (with sm_120 kernels) |
| `diffusers` / `einops` / `easydict` | 0.39.0 / 0.8.2 / 1.13 |
| `PyNvVideoCodec` | 2.2.3 |

**Optional Rust extension** ([below](#optional-rust-extension))

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

## Installing the runtime

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

## SGLang Diffusion

```bash
git clone https://github.com/sgl-project/sglang.git && cd sglang
git checkout -b reelcache e00930c5489053f26d86b179cee0d087f846acbb      # SGLang 0.5.21
git am /path/to/ReelCache/third_party/sglang/patches/*.patch
pip install -e "python[diffusion]"
```

See [`third_party/sglang/README.md`](../third_party/sglang/README.md) for what the 26 patches change. Without
`--reelcache-enable`, no archive backend is attached and no ReelCache code is imported. Two patches do fix stock
causal-Wan paths that the seam relies on (patches 5 and 7). The text-embedding cache and the workload timer stay off
unless enabled.

## Optional Rust extension

`native_ext=true` uses a compiled `reelcache_native` module for three host-side hot spots. It is optional: without
it, the Python paths run, with identical outputs. To build it and place it in `reelcache/_native/`, see
[`native/reelcache_native/README.md`](../native/reelcache_native/README.md). The test suite expects it ([testing](testing.md)).

## Model weights

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

## Machine-specific paths

Machine-specific locations are placeholders or settings; nothing points at the original cluster.

| Where | What | Default |
|---|---|---|
| `reelcache_sglang_models/causal_wan/profiles.py` | `MODEL_ROOT`, from which `WAN_BASE_DIR`, `RAVEN_WEIGHTS_DIR`, `HF_HUB_CACHE` (`DIFFUSERS_SNAPSHOT`), `PROFILE_MODEL_ROOT` (the output root of `build_profile_dir`) and the checkpoint `FileBinding` paths derive ([model weights](#model-weights)) | `/models` |
| `reelcache_sglang_models/longcat_video/weights.py` | `DEFAULT_CHECKPOINT_DIR` | `/models/LongCat-Video` |
| `ReelCacheRuntimeConfig.store_root` | the archive store ([configuration](configuration.md)) | required |
| `examples/benchmark_interactive_sessions.py` | the root of the evaluation runner, which is not included (`--runner-root` / `WL_RUNNER_ROOT`) | none; `run` requires it |
| `integrations/lingbot/` | `LB_REPO`, `LB_CKPT`, `LB_WORK`, `LB_PY`, `LB_SCRATCH`, `HF_BIN` | placeholders ([its README](../integrations/lingbot/README.md)) |
| the tests with [test assets](testing.md#test-assets) | `REELCACHE_TEST_ASSETS` and the per-asset overrides | unset: those tests skip |
