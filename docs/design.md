# How ReelCache works

ReelCache archives the clean KV of every generated chunk as an immutable, compressed node in a per-session tree,
and restores any node's history directly into the engine's KV cache:

- **Compact.** Per-layer low-rank factors at a target energy (pre-RoPE K 98 %, V 95 % by default). The bases are
  shared and augmented across chunks, and the 8-bit coefficients are coded as all-intra HEVC.
- **Branchable and durable.** Content-addressed, immutable chunk nodes form lineage trees. Resume, rewind or fork
  to any committed chunk; restarts recover from the store.
- **Fast to restore.** NVDEC decode, fused reconstruction and the engine's own RoPE write straight into the
  native cache slots. The restore runs layer by layer, overlapped with the first forward of the next chunk.
- **Out of generation's way.** Generation keeps reading its native KV (`archive_native`). Archival is asynchronous
  and ordered, can be deferred into idle time, and recent chunks stay in raw GPU/host tiers.
- **Integrated.** A backend for SGLang Diffusion's GPU worker (`--reelcache-enable`) with a session API, plus a
  standalone integration for LingBot-World-Fast.

## Data flow

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
- Optional levers cut archival's GPU and host footprint ([acceleration levers](configuration.md#acceleration-levers)).

**Baselines** for evaluation are built in:
- `raw`: the exact reference;
- `raw_offload`: raw KV behind the same LRU tiers;
- `restore_mode="recompute"` with `latents_only`: exact replay of the clean forwards from stored latents;
- `svd`: fixed-rank FP16 factors;
- `hevc`: HEVC on the normalized KV, without SVD;
- `rank_policy` `fixed_uniform` / `fixed_fraction`, and `reuse=false`: the ablation ladder.

## Supported models

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
