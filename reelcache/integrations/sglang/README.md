# `reelcache.integrations.sglang` — shared causal-Wan KV bridge for SGLang Diffusion

Optional package (ReelCache core never imports it; it never imports SGLang). SGLang loads it only with
`--reelcache-enable` (branch `reelcache-integration`, `runtime/layers/kvcache/causal_kv_backend.py`) and calls
`create_sglang_backend(config=..., pipeline=..., server_args=...)` once per GPU worker.

## Representations

| `representation` | Archive backend | Commits | Restore |
|---|---|---|---|
| `svd_hevc` (default) | `LineageSessionArchive`: one Step 6 `KVRuntime` lineage per session | asynchronous `ArchivePipeline` (latest method at the energy targets, inherit/certify/augment/refit on basis versions, one all-intra node pack per range, ordered publication, bounded queues, explicit backpressure) | `LineageRestorer` (packed NVDEC, seal-time preparation, `prewarm` when a persisted session is opened) or the explicit `SoftwareLineageRestorer` (`decoder="software"`) |
| `raw` | `RawSessionArchive`: RAW node manifests, heads `sglang/sessions/<id>` | synchronous, exact | bitwise (engine RoPE) — a reference point, not a performance path |

SGLang serves requests under `torch.inference_mode`; capture, archival and restore (packed or software, fused or
native RoPE) run in the caller's mode with no workaround (the core keeps its own state in normal tensors). HEVC
inter prediction is rejected for `svd_hevc` (per-matrix packets are neither archived asynchronously nor
restorable by the packed path). Energy targets come from the profile's `energy_profile` (K98/V95 for every
model at the moment) unless the configuration names explicit fractions.

## Publication across requests (`archive_native`, report §3.4)

Generation runs on the valid native KV; archival runs behind it **and across request boundaries**:

* A session has a *published head* (`head`, durable, survives a restart) and a *generated head* (published
  head + `pending` commits, in order). Each chunk boundary only submits the captured sources; a request returns
  as soon as its chunks are generated (`summary["published"]` is false while commits are pending).
* Per-session queue: `max_unpublished` submitted, unpublished commits (plus the pipeline-wide
  `max_source_chunks`). A full queue blocks the next chunk boundary: `summary["archive"]["backpressure"]`
  (`waits`, `seconds`), per commit `backpressure_seconds`/`queue_depth`, cumulative `state.backpressure`.
  `submit_timeout` turns the wait into `CapacityError` (the request fails; submitted chunks stay valid).
* Waiting for the published frontier happens only where it is required:
  * explicit pause: a request that pauses (`pause_after_chunks`, `controller.request_pause`/`pause`) publishes
    before it returns (`config.durability="pause"`, default; `"always"` = every request, `"never"` = none);
    the handle key `"durable": true|false` overrides per request;
  * `bridge.checkpoint(id)` / `controller.checkpoint|checkpoints|pause(idle)`, fork, rewind, `close_session`,
    `bridge.close()` / `drain_all()`;
  * a resume that restores from the archive (it needs every node up to the generated head);
* a resume whose KV is still **resident** never waits (below).
* Ordered publication and failures follow the Step 6 pipeline: a failed commit fails its descendants. The
  failure surfaces once, on the session's next request (begin_request raises `StateError`), pause, checkpoint or
  close; it is acknowledged (earlier commits still publish, failed ones are retired) and the session continues
  from its last published checkpoint (`state.status == "failed"`, `generated_chunks` reset to the published
  path). A request whose own submit hits the failure fails with it.
* Commit reports of non-durable requests: `bridge.commit_reports(request_id)` (waits for publication).
* Idle, fully published sessions release their pipeline planning state when another session's request starts.
* GPU memory of background archival next to generation and VAE decode: `max_source_chunks` (captured chunks
  held), `plan_memory_bytes` (the planner's GPU working set, `ArchiveOptions.plan_memory_bytes`) and
  `defer_archival_during_decode` (off by default): SGLang's decoding stage runs the VAE decode inside the
  bridge's `memory_critical_section("vae_decode", request_id=...)` (seam registered at attach), where the pipeline
  starts no GPU fitting, a node already on the GPU finishes first and the caching allocator's unoccupied blocks
  (the fitter's per-stream pool) are released (`ArchivePipeline.defer_gpu_work`); the request summary records it
  (`memory_critical`). The policy applies to decode sections only (`vae_decode`, `decode`, `*_decode`). Since F4
  a restore section (`restore`) never takes it: it keeps the entry check, the pressure-triggered deferral, the
  fitter re-check and the OOM relief, so a restore without pressure demotes, waits for and releases nothing (F to
  F3 evacuated GPU raw copies and released the cache at every restore). LongLive 2.0 on an A100-80GB:
  `max_source_chunks=1`, `plan_memory_bytes` 4 GiB and the deferral. Archival yields to restores in flight (`yield_to` = the bridge's restorer, with or without seal-time
  preparation).

* Overlapped restore (freeze F5, `overlap_restore`, default on; Step 11's bridge, Step 5c's
  `reelcache.overlapped_restore`). `LineageSessionArchive.restore(..., overlapped=True)` starts the compressed calls as
  layer-overlapped restores and returns at once: one call per CFG stream, chained cond, then uncond.
  * The request's first forward waits per layer.
  * `lease.complete()` joins the calls, raising a failure typed, and fills the plain path's records. The bridge calls
    it before anything derived from the restored KV leaves the request.
  * Results are bitwise the plain restore.
  * An unprepared history restores plain, recorded in `lease.stats["overlap"]["fallback"]`.
  * RAW-tier ranges (Step 5c's O4, dev22j6v5) install layer by layer on their own stream and join their call group's
    readiness, so each cache layer waits once for its raw and compressed ranges. GPU raw copies are read in place (no
    staging); host raw copies come from the pinned buffers. An unprepared history still overlaps its raw ranges.
    `overlap_options`: `raw_stream`, `raw_depth` (default 1), `raw_rope_per` (`"span"`, default, or `"layer"`).
* `restore_allocates(plans)` (Step 11's cut 4). It is False only when every attended range has a GPU raw copy now and
  `raw_restore="prefer"`. The bridge then skips the pre-restore relief and the restore section. The in-place install
  (`raw_install_in_place`, True since O4) still takes the engine RoPE's transient outputs from the caching allocator,
  e.g. one device span's K (14.4 MB on CausVid).
* Decode entry guard (dev22j6, reworked in dev22j7 for F5.1 and dev22j8 for F5.2; `decode_entry_guard`, default on).
  It applies with a GPU fitter and the raw tier's pressure response, while the deferral policy is off.
  * **Need:** the larger of the reserve and a prior (`decode_reserve_prior_bytes`; None: 16% of the device), plus the
    growth a node on the GPU may still have: the fitter's peak LIVE working set (allocated bytes on the fitter's own
    streams, the node's sources excluded), scaled by the node's source bytes, plus the re-upload of a demoted raw
    copy. Learned by default (dev22j8): at the attach warm-up (`archive.warmup(profile)` runs the archive prewarm
    without the encode when the guard is active; it plans and finalizes in a node's groups (`plan_memory_bytes`); the
    allocator's alloc/free history over `warm_commit_path` gives the live high-water, its synthetic sources
    subtracted) and online (the first 3 nodes, then every 16th, sample their fitter streams' live bytes).
    `stats()["pipeline"]["fitter_pause"]`: `warmup_peak_bytes`, `online_peak_bytes`, the bound's `peak_bytes` /
    `peak_basis`, reserved bytes as diagnostics. Measured on A100-40GB (bench 4444057): CausVid and RAVEN 4.4-4.7 GiB
    live (no plan budget: all 60 matrices planned at once), a LongLive 2.0 node 0.9-1.0 GiB (4 GiB budget).
  * **Checked against the headroom:** the driver's free memory plus the wholly free cached segments. The caching
    allocator releases those itself when an allocation needs them.
    1. It fits: nothing is done.
    2. Short with a node on the GPU: one snapshot of what the fitter's streams hold live gives the node's remaining
       growth (`max(0, bound - held)`, `ArchivePipeline.fitter_memory`). It fits: nothing is done.
    3. Still short: the fitter pauses at its next step boundary for the decode (`ArchivePipeline.pause_gpu_work`,
       acknowledged within 0.1 s), or at once while it waits without allocating (host reads, the staging-slot wait,
       an in-place encode, the rate-lag dependency wait). Step boundaries also sit around the source fingerprints,
       after the version store and per matrix in the fingerprints, anchor sketches, Gram products, CholeskyQR and
       Rayleigh-Ritz loops. A paused fitter allocates nothing more, so its growth counts 0; not acknowledged, its
       measured remaining growth counts and the fitter's phase is recorded.
    4. Still short: the allocator's cached blocks go back to the driver, only when the fragmented cached bytes could
       cover the shortfall; then the pressure response runs (least recently used GPU raw copies to host, then trims).
    5. Still short: the decode waits for the node (`short_after_relief`; `unlearned_reserve` while the reserve is not
       learned, recorded with the wait). There is no unconditional first-decode deferral.

  Why dev22j6: 5b's F4 A/B (4429789 / 4429970). The first never-block 12-chunk RAVEN request of a fresh A100-40GB
  process ran out of memory in the decode with the policy off: the auto reserve (2.4 / 4.8 GiB) was below the
  decode's working set (5.2 / 6.0 GB).

  Why dev22j7: the F5 gate (full_ab57efc5a8e528a0). The dev22j6 guard counted the node's whole working set as measured
  next to the generation (14.7 GiB on LongLive 2.0, 9.0 GiB on CausVid) and compared with the driver's free memory
  (13–15 GiB with 30–35 GiB of headroom). So it released 15–19 GiB of cache at almost every decode and made decodes
  wait for nodes: TTFNC +0.36 to +3.13 s, sustained generation −5 to −8%. Counted in `stats()["decode_guard"]`; the
  pauses also in the pipeline's `stats()["fitter_pause"]`.

  Why dev22j8: the F5.1 gate and its warm-up rerun. The warm-up peak was never learned at the default attach; forced
  on, its reserved growth (16.06 GiB on LongLive 2.0, the synthetic sources and fragmentation included) put every
  in-flight LongLive 2.0 decode 1.65 GiB short; the pauses timed out (the fitter held nothing on its streams then:
  most likely waiting for an ancestor's rates, which had no step boundary); the cache releases (15–16.5 GiB each)
  moved the headroom by 0.17 GiB; the first decode always waited.
* Archival defaults since F4: `rate_lag=3`. Planning uses the encoded byte rates of the node four above, and
  never "the latest available", so it stays deterministic. Up to four encodes overlap. `encode_workers` and
  `x265_threads` are derived from the CPUs the worker may use (`encode_workers = max(2, min(4, cpus // 8))`,
  `x265_threads = max(2, min(6, cpus // encode_workers))`: 2 x 6 at 16 CPUs, 4 x 6 at 32) and recorded in the
  archive's encoder record with the affinity-mask size they came from (`cpus_available`, read when the bridge
  attaches, after SGLang's distributed init). Explicit values override them. Step 19's decomposition (A100-40GB,
  sustained generation; lag 3 vs lag 1, CausVid / RAVEN; `results/lag_cpu_decomposition_20261017.md`):
  * published nodes/s during generation: 1.34x / 1.15x at 16 CPUs (2 x 6), 1.30x / 1.29x at 32 CPUs (4 x 6);
  * generation + drain at 16 CPUs: +13 % / +15 %;
  * the fitter's dependency-wait share: 0.54 / 0.45 -> 0.00;
  * no energy misses; bytes reachable from published nodes: -0.19 % / -0.85 %;
  * lag 2 gets almost all of the gain. Four encode workers on 16 CPUs oversubscribe them and lose; the derived
    defaults prevent it;
  * RAVEN's minimum device headroom: 3.5 GiB at lag 3 vs 4.4 GiB at lag 1 (2 x 6, 16 CPUs; no OOM in 2 runs).
    Planning ahead keeps a fitter node in flight during more VAE decodes.
* Content ids depend on content only. The predictor provenance (registry pins, predictions), planner work
  statistics and encoder thread counts are run telemetry in unhashed sidecars:
  `runtime.load_record(ref, telemetry=True)` / `node_commit_report(ref)` merge them back. Two sessions that share
  the predictor registry and archive concurrently therefore produce the same node ids whenever their content is
  the same.

### Resident native KV

SGLang's causal stages keep their caches across requests and only reset cursors. At the end of a request the
bridge records every cache's `index_state()` (cursors, pinned sink, geometry, K/V `mutation_generation`) plus the
session's chunk latents (CPU) and RNG state. The session's next request continues on that native KV
(`restore_index_state`, `summary["restore"]["mode"] == "resident"`; no decode, no waiting, exact for both
representations) when the session head did not move and no K/V write happened since; otherwise
(`mode == "archive"`, `restore["resident"]` says why: another session/ordinary request wrote the caches, a
re-initialized cache, a rewind/fork/replace, a failure) it restores from the archive. `bridge.drop_resident()`
forces the archive path (tests, quality measurements); `config.resident_reuse=false` disables reuse. Caches
without the resident API (e.g. QVG packed caches) always restore from the archive. Lossy note: a hot resume is
the native continuation; a cold resume decodes the lossy archive, so the two may differ (RAW: bitwise equal).

### Content-only ids

Every manifest the bridge writes hashes content only: RAW packets keep their timings (`*_seconds`) and nodes
their request ids in the store's unhashed telemetry sidecar (`store.telemetry(ref)["bridge"|"metrics"]`); a
session root hashes its identity, conditioning, profile and the archive method (`config.content_dict()`), while
the run configuration (store path, budgets, workers) is sidecar telemetry. Identical sessions therefore produce
identical root, node, record and packet ids in any store (with the Step 7 dev4 core's content-only lineage ids);
within one RAW store an identical continuation (fork or rewind regenerating the same chunks) deduplicates to the
same nodes.

### Process-level restore warm-up (CR-12p2-11)

The first restore of a fresh worker process used to pay one-time costs inside the restore (Triton kernel variants
compiled or loaded on first use, decoder sessions, allocator growth) and could lose to recompute at one chunk. When
the backend attaches, the archive's `warmup(profile)` runs (`config.warmup`: `None` = on for CUDA bridges with the
packed NVDEC or the RAW restore; `True`/`False` to force) and reports per-component seconds in
`bridge.startup["warmup"]`. By default it is process-level only (restore accounting rule, report §3.5: work that
does not depend on the session or node restored, which restore-vs-recompute comparisons may exclude):

* `restore_kernels`: `restore_kernels.warmup(device, shapes)` (restore_kernels d34662fc+) compiles/loads every
  kernel variant the bridge's restores launch, from `restore.kernel_warmup_shapes(profile, model, path=...)`:
  lineage restores `WarmupShape(heads, head_dim, dtype=<cache dtype>, source="pictures", rope="fused", ranks=(1,
  min(channels, chunk tokens)))` (`rope="native", native_rope_dtype=<K capture dtype>` in `rope_mode="native"`); RAW
  restores `source="dense", rope="native"` with K and V split when their capture dtypes differ (RAVEN);
* `engine_rope` (native-RoPE restores only: RAW, `rope_mode="native"`): the SGLang backend factory calls
  `bridge.warm_engine_rope(stage.transformer)`, which runs the engine's `causal_rotary_cos_sin` +
  `causal_rotate_key` on one chunk of the profile geometry (`restore_kernels.warmup` uses an identity there);
* `decoder_pool`: a decoder session for every canonical node canvas (ranks up to 1024) of the profile's tile
  geometry on every worker (`PackedRestorer.warmup`, Step 5c), when the core offers it (else reported as skipped).

Opt-in extras (`archive.warmup(profile, ...)`): `sessions=True` prepares the store's persisted sessions at attach
(`prewarm_lineages`: packs prepared, bases on the GPU, decoder sessions: session-specific CPU packing);
`archive=True` adds the archive pipeline's cold start with the encoder warm-up (`ArchivePipeline.prewarm`; ~9 s and
several GB of cached memory for LongLive 2.0). By default (`archive=None`, dev22j8) the prewarm runs without the
encode whenever the decode entry guard is active, to learn the fitter's peak live working set; `archive=False`
skips it. `synthetic=True` a diagnostic end-to-end restore of a synthetic chunk archived into a
scratch store by a private pipeline (own GPU fitter: a pipeline's fitter thread adopts its fitter; fresh predictor
registry). After `archive`/`synthetic`, unused cached CUDA blocks are released (`release`: blocks cached on their
streams pushed the first real restore's canvas allocation down the allocator's slow free-and-retry path under
LongLive 2.0's memory pressure). Nothing of the warm-up stays in the session store and the shared online learner
observes nothing (ids are identical with and without warm-up). A failing component is reported, never raised. The
SGLang backend factory finally calls `freeze_long_lived_objects()` once per worker (`gc.collect()` + `gc.freeze()`,
`startup["gc_freeze"]`): later full collections skip the model/SGLang heap (they took ~0.5 s and are the likely
cause of the 0.77 s restore stall, Step 5b diagnosis).

Restore accounting (user decision, option A): the restore-vs-recompute requirement is judged on a PREPARED session
(packs prepared, bases on the GPU: `open_session` or a reopen with `config.prewarm`, or seal-time preparation of a
session archived in the same process); `summary["restore"]["total_seconds"]` then times decode, reconstruction,
RoPE, install and latents/RNG ("restore (prepared; requirement)"). Session preparation ("CPU packing": reads,
manifests, integrity checks, pack preparation, basis assembly and upload) is reported separately: a request that
reopens a persisted session itself reports `session_open` (manifest load + session prewarm) and
`total_with_session_open_seconds`; a session restored with `prewarm=False` does the packing inside the restore
("cold session incl. CPU packing", informational).

### Process exit with NVDEC (`reelcache.nvdec` exit guard)

A process that used NVDEC (decoder caps or sessions: packed restores, the `decoder_pool` warm-up, any `CanvasSession`
or `NVDECDecoder`) ends C `exit(status)` with `_exit(status)` once its Python exit is complete: libnvcuvid's own exit
handler can otherwise wait forever on one of the library's threads, even with every decoder session closed (RTX PRO
6000: Step 18 job 4429779; reproduced by Step 5c jobs 4431652 and 4431867). Python exit hooks (bridge and decoder-pool
closes), interpreter finalization and the C stdio flush (`fcloseall`) still run, and the exit status is kept (0,
`sys.exit(n)`, 1 for an uncaught exception; a crash still ends the process by its signal). Skipped: the native exit
handlers registered before that point, i.e. C/C++ static and ELF destructors of the loaded libraries (CUDA runtime,
NCCL, libnvcuvid teardown); the kernel releases the GPU contexts and NVDEC/NVENC sessions. Tools that write their
results from native exit handlers (Nsight Systems / Nsight Compute, compute-sanitizer, sanitizer leak checks, gcov)
need `REELCACHE_NVDEC_EXIT_GUARD=0`. `REELCACHE_NVDEC_EXIT_REPORT=1` prints the guard state and the NVDEC sessions still
open at exit to stderr.

## Seam protocol (SGLang side, all plain data or worker-owned tensors)

1. `begin_request(CausalKVRequest)` -> run or `None` (no `causal_kv_session` handle: ordinary generation; with
   `observe_plain_requests` such a request gets an observer run that only times its clean forwards).
   `CausalKVRequest.dispatched_at` (optional): the worker's `time.perf_counter()` when it dispatched the request
   (time-to-first-new-chunk accounting).
2. `run.restore(CausalKVRestoreTarget)` -> number of leading blocks restored (latents, RNG, attended KV; or the
   resident KV's cursors when the caches still hold it).
3. Clean forwards: `run.begin_clean_forward` (optional; CUDA-event timing) when the capture scope opens; inside
   `CausalWanSelfAttention.forward` the capture scope hands the normalized pre-RoPE `k` and `v` (`[1, T, H, D]`,
   native dtype) to `run.capture_layer`; `run.end_clean_forward` per stream.
4. `run.end_chunk(chunk, latents)` -> pause? (submits the chunk; the archive pipeline owns the source
   references until its last read). Before the pause decision the bridge calls the worker's boundary hook
   (`backend.set_boundary_hook(hook)`; SGLang's scheduler polls for in-flight session controls there).
5. Recache forwards (`CausalKVRecache`: window, written token ranges, epoch id): captured whole, archived
   only where written (whole archived chunks, never the profile's sink) by `run.end_recache(recache)`.
6. `run.finish(error=...)` -> summary (`batch.extra["causal_kv"]`): status, published head, `pending`,
   `published`, `durable`, restore report (`mode`: `none`/`resident`/`archive`, `publish_wait_seconds`), commit
   entries (published ones with ranks, energies, bytes, stage times when the request was durable), archive
   frontiers, backpressure and `publish_wait_seconds`, per-chunk `chunk_marks` and latents `chunk_digests`
   (SHA-256), `timing` (`pre_denoise_seconds`, `time_to_first_new_chunk_seconds` from `dispatched_at`), and
   `clean_forwards` (filled when read through the `request` control: the request path never waits on events).
7. Session controls (Step 14): `backend.control(op, args)` and, from the boundary hook while a request runs,
   `backend.inflight_control(op, args)` (None = run it after the request); see below.

## Session controls (Step 14)

`CausalWanKVBridge.control(op, args) -> dict` is the worker's typed, plain-JSON control surface
(`controls.py`; SGLang's `CausalKVSessionControlReq` reaches it through the scheduler). Generation actions are
ordinary video requests with the `causal_kv_session` handle (`generate`, `resume`, `fork`, `rewind`; checkpoints
are chunk counts, node references or checkpoint names); the ops generate nothing:

| op | args | result / effect |
|---|---|---|
| `capabilities` | - | model, representation, boundary (`chunk`), ops, in-flight ops, the native prompt change (or why there is none), attach warm-up |
| `list` | - | open sessions (status, heads, pending commits, resident KV) and persisted ones (`status: closed`), with named checkpoints |
| `describe` | `session_id`, `history?` | live state (open) or the persisted head (never reopens); recent requests |
| `checkpoints` | `session_id` | clean boundaries of the head path (drained first) and named checkpoints (`on_head_path`) |
| `request` | `request_id` | the bridge summary of a recent request (restore report, digests, timings, clean forwards) |
| `open` | `session_id` | reopen + prepare a persisted session (the prepared state of the restore requirement) |
| `pause` | `session_id` | idle: drain, durable state. Running: flag (`deferred: true`); stops at the next clean boundary; call again after the request to acknowledge durably |
| `checkpoint` | `session_id`, `name?`, `replace?` | drain; durable head; `name` pins it (`sglang/checkpoints/<session>/<name>`) |
| `fork` | `session_id`, `target`, `checkpoint?` | new session at a clean boundary (copy-on-write; the source never changes) |
| `rewind` | `session_id`, `checkpoint` | head back to an ancestor checkpoint (non-ancestor names: fork instead) |
| `close` | `session_id`, `mode?` (`drain`/`abort`) | unload: publish or retire pending commits, drop resident KV and pipeline state; stays persisted |
| `delete` | `session_id`, `checkpoint?`, `gc?` | delete a named checkpoint, or the session (head, names; pending retired); then GC |
| `gc` | - | drain every open session, reclaim objects no head, named checkpoint or lease reaches |

Errors are `SessionControlError(code, status)`: `invalid_argument` 400, `unknown_session` /
`unknown_checkpoint` / `unknown_request` 404, `illegal_transition` / `busy` 409, `incompatible_model` 422,
`archival_failure` / `integrity` / `internal` 500, `timeout` 504. Names match `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`;
no argument is a filesystem path. While a request runs, `capabilities`, `list`, `describe`, `request` and the
pause of the running session are served at its chunk boundaries; anything else on that session is `busy` (the
scheduler queues it behind the request).

Prompt changes are the model's native ones only (`prompt_change_capability`, by the profile's recache policy):
LongLive 1.0 `prompt_switches` (recache epoch), LongLive 2.0 `shot_prompts` / `shot_durations` (multi-shot);
CausVid and RAVEN have none (their repositories generate under one prompt): a resume/fork/rewind under another
prompt is rejected with `UnsupportedError` ("... has no native prompt change ...").

LongLive 2.0: **a session that will change prompts must start in multi-shot mode** (`shot_prompts`, even with a
single shot). Multi-shot requests run with a global sink in the cache (a different cache geometry) and carry one
conditioning id per block (shot index, scene cut, block prompt); a single-prompt session has neither, so it cannot
be continued under a new shot (rejected: different conditioning). Continue it with `shot_prompts` that repeat the
session's shots for the restored chunks and add the new shot at the next chunk boundary, e.g. a session generated
with `shot_prompts=["A"]` and paused after 2 chunks resumes with `shot_prompts=["A", "B"]`,
`shot_durations=[2, n]`.

Output of requests that restored history: only the new chunks by default (`causal_kv_session.output = "new"`;
`"full"` returns the whole clip). SGLang's decoding stage decodes `decode_context_latents` preceding latent frames
first (config default 40, or per request) to warm the causal VAE decoder and drops their frames; a context that
reaches back to the first frame reproduces the full-clip frames bitwise (`decode.exact` in the request summary).
With real weights (all four causal-Wan models, Wan2.1 and Wan2.2 VAEs) 40 context latents already give the
full-clip frames bitwise (36: one uint8 level on <0.1% of values; 16: max |d| up to 0.06); the decode cost is
linear in the decoded latents, so new-only output bounds the decode of long sessions only.

LongLive 1.0 prompt switches: a switch at chunk >= 5 keeps the sink (the upstream recache window starts after frame
0); an earlier switch rewrites the sink and is refused for session requests. Restored blocks must carry the
request's own block conditioning (rewinding/forking into another prompt's epoch is rejected).

Garbage collection marks from
every store ref (session heads, named checkpoints) and lease, so a deleted branch never takes objects another
branch or checkpoint reaches; a SIGKILLed worker leaves only unreferenced staging, which `gc` reclaims (heads move
only after a node is complete).

Diagnostics: `observe_plain_requests=true` times the clean forwards of requests without a session (nothing is
captured or archived): with `request`, the in-server recompute baseline of the restore requirement.

## Profiles

`ModelProfile` is data. Matching is by exact model path (a registered path, its normalized form, or a Hugging
Face cache snapshot of a registered repo id at the bound revision) and, if `transformers` is not empty, the
transformer class. `window_frames=None` is the request-sized `full_history` buffer (no sink);
`sink_rolling_window` is an alias of `sink_window`. Out-of-tree profiles:

```python
from reelcache_sglang_models.causal_wan.profiles import CAUSVID, bridge_profile_fields
from reelcache.integrations.sglang import register_profile
register_profile(bridge_profile_fields(CAUSVID))             # before the worker builds the bridge, or:
# --reelcache-config '{"store_root": "...", "profile_fields": <bridge_profile_fields(CAUSVID, "window_21_frames")>}'
```

## Attaching a profile stage (Step 12 phase 2 recipe)

A stage that runs its own block loop (e.g. `ProfileCausalDenoisingStage`) uses the base-stage helpers of
`CausalDMDDenoisingStage` on the branch:

```python
kv_run, resume = self._begin_causal_kv_run(batch, server_args, ctx=ctx, block_sizes=sizes, pos_start_base=0,
                                           use_cfg=False, kv_cache=kv_cache, kv_cache_neg=None, latents=latents)
with self._causal_kv_run_scope(kv_run):
    for ci, frames in enumerate(sizes):
        if ci < resume:
            continue                                   # restored: latents, RNG and attended KV are installed
        if switch_before(ci):                           # LongLive 1.0 prompt switch
            recache = self._causal_kv_recache(batch, window_start_frame=rs, window_frames=cs - rs,
                                              written_frames=plan.written_global_frames, pos_start_base=0,
                                              epoch_id=f"switch-c{ci}", details=record.cache_epoch_fields())
            with self._causal_kv_recache_scope(kv_run, recache):
                ...                                     # the clean recache forward(s) (_update_causal_context_cache)
        chunk = self._causal_kv_chunk(batch, block_index=ci, start_frame=start, num_frames=frames,
                                      pos_start_base=0, scene_cut=False) if kv_run else None
        self._causal_kv_pending = (kv_run, chunk) if kv_run else None
        out = self._denoise_and_update_causal_block(...)
        self._causal_kv_pending = None
        latents[:, :, start:start + frames] = out
        if kv_run is not None and kv_run.end_chunk(chunk, out):
            latents = latents[:, :, :start + frames]
            break
if kv_run is not None:
    batch.extra["causal_kv"] = kv_run.finish(error=None)
```

* Caches must be `CausalSelfAttentionKVCache` (or subclasses): restore replays their own cursor/eviction logic
  (`replay_layout`) and installs through `install_restored_history`. Sink-protected recompute writes are now a
  base-class option (`protect_sink_on_recompute=True`), and request-scoped sizing is the
  `_causal_kv_cache_size_for_batch` hook.
* The DiT must expose `causal_rotary_cos_sin(frames, h, w, start_frame=, device=)` and `causal_rotate_key`
  matching what its attention applies (`verify_capture_rope` checks the captured tables bitwise at the recorded
  position; a DiT with its own RoPE overrides both).
* The profile's `native_dtype` must equal the KV cache dtype; `cfg_streams` the request's streams.
* Resident reuse needs nothing from the stage beyond keeping its caches across requests (the base stage does) and
  writing K/V only through the cache methods (they renew `mutation_generation`); a stage that writes `cache.k` /
  `cache.v` directly must renew it too (or reallocate the cache). `mutation_generation` is process-unique, so a
  request-scoped (re-allocated) cache never matches a recorded state: such stages always restore from the archive
  (correct, just not the resident fast path); reusing caches when the request geometry matches enables it.
* Timeline-varying conditioning (prompt switches, shot schedules): return one id per block from
  `_causal_kv_block_conditioning(batch, server_args, block_sizes)` (`CausalKVRequest.block_conditioning`, branch
  8cbb8d1d9f) and leave the schedule fields out of the conditioning identity (extend
  `_CAUSAL_KV_IDENTITY_EXCLUDE`). The bridge stores the id per chunk node (`record["block_conditioning"]`) and per
  recache node (`record["precedes_block"]`: the block generated next, under the epoch's conditioning) and compares
  only restored blocks: a fork/rewind/resume whose restored chunks and epochs were generated under the request's
  own per-block conditioning is accepted (e.g. a fork from the pre-switch chunk node under another prompt, which
  then runs its own recache epoch); restoring an epoch of another prompt is rejected before any head moves. A
  chunk-count checkpoint names the newest node at that count, i.e. the switch's recache node when one follows
  the chunk; fork from the chunk node ref to branch before a switch.
* `summary["restore"]` reports the restored head: `head_kind` (`root`/`chunk`/`recache`), `head_epoch`,
  `head_block` (for a recache: the block it precedes). A stage resuming exactly at a prompt switch reruns its
  recache only if `head_kind != "recache"`.
* Captured dtypes: `ModelProfile.capture_dtypes={"K": "float32", "V": "bfloat16"}` when the attention-input
  pre-RoPE K is FP32 but V and the cache are BF16 (RAVEN). RAW packets keep FP32 K; restore rotates keys in their
  capture dtype with the engine's RoPE and casts on store (bitwise the engine path). The lossy path hands FP32 K
  and BF16 V to the Step 6/7 fitter (CPU-tested with the dev4 core).
* TF32: `ModelProfile.matmul_precision="fp32_verified_fp64"` (or `ReelCacheRuntimeConfig.matmul_precision`) selects
  the rank method's policy for models whose generation runs TF32 matmuls (needs the dev4+ core's
  `RankMethodConfig.matmul_precision`); global flags are never toggled.
* Recache staging: the archive pipeline stages one host canvas per range of a node before encoding it, so a
  recache node needs `max_staged_canvases` >= its ranges. `max_staged_canvases=None` (default) sizes it from the
  profile (`ModelProfile.max_recache_ranges`: the KV buffer minus the sink, in chunks, for
  `kv_recache_on_prompt_switch` profiles; at least 2); an explicit smaller value is rejected when the bridge is
  built, and a recache node with more ranges than canvases fails explicitly (`CapacityError`) instead of hanging.
* RAW recache epochs are one node per range; only the epoch's last node is a checkpoint (`checkpoints()` lists the
  others with `checkpoint: False`; fork/rewind to them raises).
* Commit reports count native bytes in the captured dtype (FP32 keys: 4 bytes per element).
