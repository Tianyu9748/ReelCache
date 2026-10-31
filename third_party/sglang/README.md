# SGLang Diffusion patch series for ReelCache

ReelCache runs inside SGLang Diffusion's GPU worker. The engine-side changes are small, opt-in "seams" that are
off unless the server is started with `--reelcache-enable`. They are kept here as a `git format-patch` series:

| | |
|---|---|
| Upstream | https://github.com/sgl-project/sglang |
| SGLang version | **0.5.21** (package `sglang` 0.5.21 with `sglang-kernel` 0.4.7) |
| Base commit | `e00930c5489053f26d86b179cee0d087f846acbb` (branch `release/v0.5.21`, 2026-09-29) |
| Patches | 26 (`patches/0001-*.patch` … `patches/0026-*.patch`); 35 files, +5,891 / −97 lines, about half of them unit tests |
| Result | git tree `06209d51e5b6623ec9e90dfa8b7b368bc2fe92c7` |

Applying the series to the base commit gives git tree `06209d51e5b6623ec9e90dfa8b7b368bc2fe92c7`. Check it with
`git rev-parse HEAD^{tree}` after `git am`.

The paper's measurements ran on SGLang commit `043c0204cffef4c75262ec7e5ae2a2141645fe85`, where the integration still
used the runtime's former internal name. These patches are those 26 commits with the rename applied to the code,
file names and messages, and with the commits' author set to this repository's account. Line counts and patch
boundaries are unchanged. One test line changed beyond the rename:
`test_causal_kv_backend.py`'s import-isolation check (patches 4 and 9) now names the packages `reelcache` and
`reelcache_sglang_models` exactly. A plain prefix check on `reelcache` would also match unrelated modules, such as an
environment helper with that prefix that the evaluation's environment loads at interpreter start.

## Apply

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout -b reelcache e00930c5489053f26d86b179cee0d087f846acbb
git am /path/to/ReelCache/third_party/sglang/patches/*.patch
git rev-parse HEAD^{tree}     # 06209d51e5b6623ec9e90dfa8b7b368bc2fe92c7
```

Then build and install SGLang Diffusion from this checkout (`pip install -e "python[diffusion]"`, see SGLang's
`docs/docs/sglang-diffusion/installation.mdx`). The evaluation used a source build with CUDA 13.2, PyTorch 2.13
(cu130) and CPython 3.12.13 on RTX PRO 6000 Blackwell (sm_120) and A100 (sm_80) GPUs.

## What the patches add

With `--reelcache-enable` absent, no backend is attached and the seam's attention, denoising and decoding hooks do
nothing: no copies and no ReelCache import. Some patches change stock behavior:
- **Patch 5** fixes the stock causal-Wan block. At `e00930c` its cross-attention raised a `TypeError` on the first
  forward.
- **Patch 7** reads the local attention window from the DiT config; it was never read before. It also stops
  reusing stage caches across requests with a different geometry.

Both are no-ops for LongLive 2.0, whose blocks have their own cross-attention and whose window equals the default.
Two opt-in additions stay off unless enabled:
- the text-embedding cache (`SGLANG_WL_EMBED_CACHE_BYTES` > 0);
- the workload timer (`SGLANG_WL_TIMER=1`, or `workload_timing` on a request).

The additions, by area:

- **KV archive seam** (`runtime/layers/kvcache/causal_kv_backend.py`, `causal_attention_cache.py`,
  `models/dits/causal_wanvideo.py`, `pipelines_core/stages/causal_denoising.py`):
  - hands the backend the normalized pre-RoPE K and the V of every causal self-attention layer, during the clean
    (t = 0) context forward of each chunk only. LongLive 1.0 prompt-switch recache forwards are handed over too.
  - chunk-boundary callbacks, where a running session can be paused.
  - `install_restored_history` writes restored KV into the native `CausalSelfAttentionKVCache` slots, re-rotated
    with the engine's own RoPE (`rotary_embedding` gains `out=` writes).
  - per-layer readiness objects, so a restore can overlap the first forward layer by layer.
  - resident session KV that is reused across requests when nothing has written to it since.
- **Server flags** (`configs/reelcache.py`, `server_args.py`):
  - `--reelcache-enable`; startup fails if `reelcache.integrations.sglang` cannot be imported.
  - `--reelcache-config <JSON object | path to a JSON file>`: fields of `ReelCacheRuntimeConfig`.
  - `--reelcache-hevc-inter-prediction`: off by default, so streams are all-intra.
- **Session API**:
  - Requests: `POST /v1/videos` accepts a `causal_kv_session` handle with `session_id`, `action` (`generate`,
    `resume`, `fork` or `rewind`), `from_session`, `checkpoint`, `pause_after_chunks`, `durable`, `output`,
    `decode_context_latents` and `overlap_restore`.
  - Session controls (`causal_kv_session_api.py`) run through the scheduler:
    - `GET /v1/sessions`, `GET` / `DELETE /v1/sessions/{id}`;
    - `/v1/sessions/{id}/{open,pause,fork,rewind,close,checkpoints,raw_tier}`;
    - `GET /v1/session_capabilities`, `GET /v1/session_requests/{request_id}`, `POST /v1/session_gc`.
  - No endpoint accepts a filesystem path.
- **Model-side support:**
  - cache-aware cross-attention in the stock causal-Wan block (prompt switches);
  - opt-in sink protection for in-place recompute writes;
  - DiT-config local windows;
  - LongLive 2.0 multi-shot fields over HTTP;
  - per-block conditioning identity;
  - session providers for pipelines without causal stages (LongCat-Video);
  - a memory-critical section around the VAE decode.
- **Workload support:**
  - opt-in request timing marks (`runtime/utils/workload_timer.py`);
  - a per-prompt text-embedding cache (`stages/text_encoding.py`);
  - workload metadata on causal KV requests.
- **Tests:** 12 unit test modules under `python/sglang/multimodal_gen/test/unit/` (`test_causal_kv_*`,
  `test_causal_wan_crossattn_cache.py`, `test_rotary_out.py`, `test_wl_*`).

Commits 16–19 still say "PROTOTYPE" in their subjects, and commit 16 names an internal development branch. Both
labels are historical: layer-overlapped restore later became the default (`overlap_restore=true`), and these commits
are part of the evaluated SGLang tree.

## Patch list

| # | Subject |
|---|---|
| 1 | Causal KV archive seam: pre-RoPE capture hook and cache restore API |
| 2 | Causal denoising: clean-boundary session hooks for KV archive backends |
| 3 | ReelCache server flags and causal_kv_session request handle |
| 4 | Tests for the causal KV archive seam |
| 5 | Causal Wan: cache-aware cross-attention in the stock block |
| 6 | Causal KV cache: opt-in sink protection for in-place recompute writes |
| 7 | Causal DMD stage: DiT-config local window and request-scoped cache geometry |
| 8 | Causal KV seam: recache capture archives only the written frames |
| 9 | Causal KV seam: resident session KV across requests |
| 10 | Causal KV seam: per-block conditioning identity |
| 11 | Causal KV sessions: controls through the scheduler and HTTP (--reelcache-enable) |
| 12 | LongLive2: multi-shot fields over HTTP; per-block session conditioning |
| 13 | Causal KV seam: session providers for pipelines without causal stages |
| 14 | Causal KV seam: memory-critical section around the VAE decode |
| 15 | Causal KV sessions: continuations take video conditioning from the archive |
| 16 | Causal KV seam: layer-overlapped restore |
| 17 | Causal KV cache: install_restored_history checks the layout in O(spans) |
| 18 | Causal KV seam: overlap test with LongLive 2.0's multi-shot layout (tests only) |
| 19 | Causal KV seam: drop the early-restore offer; overlap_restore is a bool |
| 20 | Causal KV sessions: the raw_tier control over HTTP |
| 21 | Causal KV cache: a restored layout is checked once and only its unwritten slots are zeroed |
| 22 | Rotary embedding: out= writes the rotation into a given tensor (restored KV in place) |
| 23 | Workload timer: opt-in request marks |
| 24 | Per-prompt text-embedding cache |
| 25 | Causal KV requests: workload metadata |
| 26 | Causal KV metadata: identity test |

The patches are derived from SGLang and keep its license (Apache-2.0).
