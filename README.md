# ReelCache

**KV-cache archival, restore and branching for autoregressive video diffusion, built on SGLang Diffusion.**

## What it is for

Autoregressive ("causal") video diffusion models generate a video chunk by chunk, and every new chunk attends to
the self-attention KV cache of the chunks before it. Interactive sessions pause and resume, rewind to an earlier
moment, or fork into alternative continuations, and each of those needs the history's KV again. Keeping that KV raw
costs gigabytes per second of video; recomputing it from saved latents puts a long replay in front of the next
chunk.

ReelCache stores every generated chunk's KV as a compressed, immutable node in a per-session tree. A session can
then resume, rewind or fork at any committed chunk: ReelCache restores that node's history straight into the
engine's KV cache instead of recomputing it.

Supported models: CausVid, RAVEN, LongLive 1.0, LongLive 2.0 and LongCat-Video through SGLang Diffusion, and
LingBot-World-Fast through a standalone integration.

## How it works

```
 generation (SGLang GPU worker)                 ReelCache
 chunk i: clean forward ── K, V per layer ────► low-rank factors at an energy target (K 98 %, V 95 %),
          (generation keeps its native KV)      bases shared across chunks ─► 8-bit coefficients ─► HEVC
                                                ─► immutable node in the session tree (content-addressed store)
 resume / rewind / fork ◄──── layer by layer ── NVDEC decode ─► reconstruct P·Q ─► RoPE ─► native KV cache
```

1. **Capture.** At each chunk's clean (t = 0) forward, the engine hands ReelCache every layer's normalized pre-RoPE
   K and V.
2. **Compress.** Each K or V matrix is factored as `P·Q` at the smallest rank that keeps the energy target. A chunk
   reuses or extends its parent's basis when that is cheaper than fitting a new one.
3. **Encode and store.** The coefficients are quantized to 8 bits and coded as all-intra HEVC (NVENC, or libx265
   without it). Nodes are published in order, in the background, to a content-addressed store; recent chunks also
   stay as raw KV in GPU and host tiers.
4. **Restore.** A resume, rewind or fork decodes the target's visible history with NVDEC, reconstructs it, applies
   the engine's RoPE and writes it into the native cache, layer by layer, while the next chunk's first forward
   runs.

Generation keeps reading its native KV throughout. Archival runs asynchronously on its own CUDA stream and can be
deferred into idle time between requests. Details: [docs/design.md](docs/design.md).

## Commands

**Install** the runtime (Python ≥ 3.10; `import reelcache` needs only NumPy and PyTorch):

```bash
pip install -e ".[hevc]"                       # CPU: software HEVC (imageio-ffmpeg ships an FFmpeg with libx265)
pip install -e ".[hevc,nvdec,gpu-fit-cu13]"    # GPU: + PyNvVideoCodec (NVENC/NVDEC) + CuPy for CUDA 13
```

**Patch and install SGLang Diffusion 0.5.21** for serving:

```bash
git clone https://github.com/sgl-project/sglang.git && cd sglang
git checkout -b reelcache e00930c5489053f26d86b179cee0d087f846acbb
git am /path/to/ReelCache/third_party/sglang/patches/*.patch
pip install -e "python[diffusion]"
```

The optional Rust extension is built with maturin
([native/reelcache_native/README.md](native/reelcache_native/README.md)). All dependencies and their evaluated
versions are listed in [docs/installation.md](docs/installation.md).

**Try the runtime without a GPU** (synthetic KV: commit chunks, restart, restore, branch):

```bash
python examples/runtime_demo.py --output /tmp/reelcache-demo --codec svd_hevc --allow-lossy
```

**Serve a model**, here CausVid (its weights go under `/models`; see
[model weights](docs/installation.md#model-weights)):

```python
import json
from reelcache_sglang_models.causal_wan.profiles import get_profile
from reelcache_sglang_models.causal_wan.convert import build_profile_dir
from reelcache_sglang_models.causal_wan.lifecycle import bridge_runtime_config
profile = get_profile("causvid")
build_profile_dir(profile)  # -> /models/reelcache_sglang_profiles/reelcache-causvid-480p-s12v1
cfg = bridge_runtime_config(profile, store_root="/local/nvme/reelcache-store",
                            defer_archival_during_generation=True, archival_defer_inflight="overlap")
json.dump(cfg, open("reelcache.json", "w"), indent=1)
```

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE=reelcache_sglang_models
sglang serve --model-path /models/reelcache_sglang_profiles/reelcache-causvid-480p-s12v1 \
    --reelcache-enable --reelcache-config reelcache.json
```

**Generate, resume, rewind and fork** with ordinary `POST /v1/videos` requests that carry a session handle:

```jsonc
{"prompt": "...", "causal_kv_session": {"session_id": "s1"}}                                    // new session
{"prompt": "...", "causal_kv_session": {"session_id": "s1", "action": "resume"}}                // continue the head
{"prompt": "...", "causal_kv_session": {"session_id": "s1", "action": "rewind", "checkpoint": 4}} // after chunk 4
{"prompt": "...", "causal_kv_session": {"session_id": "s2", "action": "fork", "from_session": "s1", "checkpoint": 6}}
```

The other request fields and the session endpoints are in [docs/usage.md](docs/usage.md); the configuration
fields are in [docs/configuration.md](docs/configuration.md). LingBot-World-Fast runs outside SGLang:
[integrations/lingbot/README.md](integrations/lingbot/README.md).

**Run the tests** (CPU only; requirements and optional test assets: [docs/testing.md](docs/testing.md)):

```bash
CUDA_VISIBLE_DEVICES= python scripts/verify.py --output /tmp/reelcache-receipt.json
python integrations/lingbot/tests/test_lb_overlap.py
```

## Documentation

The [docs/](docs/README.md) folder holds the details: [design](docs/design.md),
[repository structure](docs/repository-structure.md), [installation](docs/installation.md),
[usage](docs/usage.md), [configuration](docs/configuration.md), [testing](docs/testing.md) and
[provenance](docs/provenance.md).

This repository contains code only: no experiment harnesses, evaluation results or model weights.
