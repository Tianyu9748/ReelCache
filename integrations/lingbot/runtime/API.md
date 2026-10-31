# LingBot-World-Fast runtime for ReelCache (`lingbot_rt.py`)

Official code `Robbyant/lingbot-world` @ `a43bec7f8091c83e9b30b16b912f6fc906236fa6`, unmodified, imported from
`$LB_REPO` (placeholder default `/models/lingbot-world`). `lingbot_rt.py` only wraps it (hooks, no source edits).

## 1. Environment

```bash
# from the ReelCache root, inside a job; sets PATH/PYTHONPATH, caches on node-local scratch, offline HF
source integrations/lingbot/runtime/env.sh   # export LB_REPO / LB_WORK / LB_PY / LB_SCRATCH first to override
$LB_PY -B my_script.py                                                          # 1 GPU (480P)
$LB_PY -B -m torch.distributed.run --standalone --nproc_per_node=2 my_script.py # Ulysses SP=2 (720P)
```
- `LB_PY` (default: the `python3` on `PATH`): the evaluation used an overlay venv = a model environment with torch
  2.8.0+cu128, flash-attn 2.8.3 (with sm_120 kernels) and diffusers 0.39, plus `easydict==1.13` and
  `PyNvVideoCodec==2.2.3` (NVENC/NVDEC for the ReelCache archive). Nothing else. Without ReelCache, `easydict` alone is
  enough.
- `import wan` needs a visible GPU (T5EncoderModel evaluates `torch.cuda.current_device()` at import).
- Slurm: one task with N GPUs and 16 CPUs per GPU bound to them (e.g. `--gres=gpu:N -c $((16*N))
  --gres-flags=enforce-binding`). Memory: ~240 GB host RAM per process is plenty.

## 2. Paths / constants (`lingbot_rt`)
| name | value |
|---|---|
| `CKPT_DIR` | `$LB_REPO/lingbot-world-base-cam` (name must contain `cam`) |
| weights | `lingbot_world_fast/` (16 F32 shards, 74.18 GB, loaded as bf16 = 37.1 GB), `models_t5_umt5-xxl-enc-bf16.pth`, `Wan2.1_VAE.pth`, `google/umt5-xxl/` |
| manifest | `../setup/weights_manifest.json` (sha256 of all 29 files, verified against HF LFS oids) |
| `SIZES` | `480p=(480,832)` -> 60x104 latents, 1560 tokens/latent; `720p=(720,1280)` -> 90x160, 3600 tokens/latent |
| `KV_BYTES_PER_TOKEN` | 819,200 (40 layers x K,V x 5120 x bf16): 1.278 GB/latent at 480P, 2.949 GB at 720P |
| `CHUNK_LATENTS` | 3 (one chunk = 3 latents = 12 new RGB frames; chunk 0 = 9 frames) |
| `DEFAULT_SHIFT` | 10.0 (generate_fast.py default = cfg.sample_shift; the generate() signature default 5.0 is NOT what the CLI uses) |

## 3. Pipeline, camera, image
```python
import lingbot_rt as rt
pipe = rt.build_pipe()                          # WanI2VFast; SP iff WORLD_SIZE>1; no FSDP; T5/VAE as official
rt.write_static_action_path()                   # -> rt.STATIC_CAMERA_DIR (961 identity c2w poses + ex-03 intrinsics)
img, box = rt.center_crop_resize(PIL_image, 480, 832)   # exact target size => exact latent grid
rt.set_attention_window(pipe, 21)               # rolling window of 21 latents incl. current chunk (-1 = full)
video, secs = rt.generate(pipe, prompt, img, height=480, width=832, frame_num=81, seed=0)
frames = rt.video_to_uint8(video)               # [81, H, W, 3] uint8, identical to official save_video
```
- Intrinsics are given in the 480x832 frame; `generate()` rescales them to h x w itself (do not pre-scale).
- `generate()` = one prompt per call; it allocates the whole self-attn KV (`kv_size = frame_seqlen * lat_f`,
  or `* local_attn_size`) at the start and keeps it alive through the VAE decode. Noise for ALL `lat_f` latents is
  drawn up front from the seeded generator, then 3 re-noise draws per chunk; the image condition `y` is the VAE
  encoding of [image, F-1 zero frames] (depends on the horizon F).
- `rt.encode_prompt(pipe, prompt)` runs generate()'s own T5 path once (GPU, then T5 back to CPU) and fills
  `pipe._t5_cache`, so generate() never parks the 11.4 GB T5 on the GPU. Always use `offload_model=False`
  (the official offload path moves the DiT to CPU at the end and never back).
- Prompt switch inside a session (not possible in one generate() call; needs a custom chunk loop): encode the new
  prompt and pass `cross_attn_first_call=True` on the first DiT forward after the switch; that rebuilds only the
  cross-attention cache, the self-attention KV is untouched.

## 4. KV adapter (`rt.ChunkKV(pipe)`)
Wraps `pipe.model.forward` + a forward hook on every `self_attn.norm_k`; `remove()` restores everything.
A chunk = 4 denoising forwards + 1 clean forward at t=0 that (re)writes that chunk's KV rows; the adapter acts
right after the clean forward of the chunks you name.
```python
kv = rt.ChunkKV(pipe)
kv.reset(); kv.capture_chunks = {0}
rt.generate(...)                                   # RAW
K0 = kv.captured[0][("K", 7)]                      # layer 7 pre-RoPE K (norm_k output) [N, 5120] bf16, CPU
V0 = kv.captured[0][("V", 7)]                      # layer 7 V [N, 5120]; N = 3 * tokens_per_latent
assert kv.rope_check[0]["all_exact"]               # re-RoPE(K0) == cached post-RoPE K, bitwise, every layer/rank
kv.reset(); kv.inject = {0: {("K", l): Khat[l], ("V", l): Vhat[l] for l in range(40)}}  # pre-RoPE K, plain V
rt.generate(...)                                   # chunks 1.. attend to the injected chunk-0 KV
kv.export_chunks = {3}; ...; snap = kv.exports[3]  # raw per-rank cache snapshot after chunk 3 (rt.export_kv)
rt.import_kv(kv_cache_list, snap)                  # inverse (same rank/geometry); restores both end indices
```
- K is re-roped with the model's own `causal_rope_apply` at `start_frame = current_start // frame_seqlen`
  (absolute position; also correct with a rolling window, where the cache rows are `[local_end-N, local_end)`).
- SP: K is gathered over ranks along the sequence (norm_k runs on sequence shards), V along heads (cache rows);
  captured matrices are full-head on every rank. For injection every rank must hold the same full-head tensors;
  each rank writes its head slice `[r*40/sp, (r+1)*40/sp)`. `export_kv`/`import_kv` are per rank (local heads).
- Also: `kv.timeline` (every forward: chunk, t, seconds, cache end), `kv.chunk_summary()` (per-chunk wall time,
  peak allocated/reserved, device used), `kv.alloc` (bytes of the self/cross-attn caches, all ranks),
  `kv.last_latents` (rank 0: latents handed to the VAE decode).
- Identity guarantee: capture -> re-RoPE -> inject of the unmodified K/V reproduces the RAW latents and RGB bitwise
  (gated in every energy-sweep case; see `../smoke/preflight_*`).

## 5. Measured footprint (Phase 0, RTX PRO 6000 Blackwell 96 GB)
See `../smoke/PHASE0_REPORT.md` (per-chunk wall time and peak memory at 480P/720P, full history vs
`local_attn_size=21`, at several depths).

Key numbers (RTX PRO 6000): 480P 1 GPU: 81 frames in ~38 s, 3.7 -> 5.1 s/chunk, peak 72 GB; full history grows the KV by
3.83 GB/chunk (KV is allocated up front for the whole horizon); local_attn_size 21 caps it at 26.8 GB and the chunk time
at 5.2 s (peak 66.5 GB, flat after chunk 7). 720P needs 2 GPUs (SP=2) for the 81-frame history (61.9 GB KV): 71 s per
81 frames, 6.0 -> 9.7 s/chunk, window 21 flat at 10.5 s/chunk (73 GB/GPU). One GPU at 720P fits a window of 15
(88 GB, 15.8 s/chunk) or 12 (79 GB, 14.4 s/chunk). Cold model load from network storage: ~12 min (page cache: seconds).
480P geometry: use `exact_max_area(480, 832)` (the CLI's 480*832 produces 464x832).
