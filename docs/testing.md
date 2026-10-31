# Tests

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

## Test assets

A few tests compare against artifacts of the original evaluation that this repository does not
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
