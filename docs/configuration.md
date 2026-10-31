# Configuration

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

## Acceleration levers

All levers are opt-in and off by default. They reduce archival's interference with generation and the exposed
restore time. The defaults reproduce the evaluated runtime bit for bit. Identity classes:
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
- **LingBot:** the same compression flags, `finalize_on_fit_device`, and its overlap options ([LingBot README](../integrations/lingbot/README.md)).

Measured on RTX PRO 6000:
- **CausVid / RAVEN:**
  - with a 48 GiB decoded tier, end-to-end next-chunk P50 dropped 13–15 %, with bit-identical frames;
  - under fully overlapped archival, RAVEN's generation slowdown fell from 32.8 % to 21.5 %;
  - CUDA graphs, green contexts and the Rust extension did not reduce interference, which is SM contention.
- **LingBot:** P50 dropped 6.5 %.
