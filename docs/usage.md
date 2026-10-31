# Usage

## Quick start without a GPU

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

## Serving with SGLang Diffusion

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
- `--reelcache-config`: a JSON object, or the path of a JSON file, holding `ReelCacheRuntimeConfig` fields ([configuration](configuration.md)).
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
semantics are documented in [`reelcache/integrations/sglang/README.md`](../reelcache/integrations/sglang/README.md):
publication across requests, resident KV, the seam protocol, session controls, profiles, and attaching a stage.

## LingBot-World-Fast

LingBot runs the unmodified official pipeline outside SGLang. `integrations/lingbot/` contains:
- the runtime wrapper `lingbot_rt.py`, with its `API.md`;
- the `reelcache_lingbot` adapters: `LingBotModel`, `ReelCacheArchive`, and the layer-overlapped restore;
- weight setup and a portable `env.sh`;
- 9 CPU tests.

See [`integrations/lingbot/README.md`](../integrations/lingbot/README.md).
