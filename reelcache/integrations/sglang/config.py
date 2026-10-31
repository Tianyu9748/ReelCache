"""Validated runtime configuration of the SGLang causal-Wan KV bridge (plain data).

``ReelCacheRuntimeConfig`` is what ``--reelcache-config`` (JSON) plus
``--reelcache-hevc-inter-prediction`` resolve to inside the GPU worker. It holds
names and numbers only (no live objects), so it is safe to log and to persist
next to an archive.
"""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass
from typing import Any, Mapping

from ...errors import ConfigurationError
from ...types import CodecSpec, EnergyTargets, energy_targets

MODES = ("archive_native",)
REPRESENTATIONS = ("raw", "svd_hevc")
FIT_DEVICES = ("cuda", "cpu")
DECODERS = ("nvdec_packed", "software")
ROPE_MODES = ("fused", "native")
PIX_FMTS = ("yuv444p", "yuv420p")
NORMALIZATIONS = ("per_channel", "global")
CANVASES = ("canonical", "own_size")
ORTHOGONALIZERS = ("batched", "householder")
DURABILITY = ("pause", "always", "never")
#: ``on_open_error``: a session head found damaged when the store is opened is quarantined, or fails the bridge.
OPEN_ERRORS = ("quarantine", "raise")
MATMUL_PRECISIONS = ("fp32", "fp32_verified_fp64")
#: Freeze D: whether restores take raw-tier copies first (``prefer``) or compressed nodes only (``off``).
RAW_RESTORES = ("prefer", "off")
#: Freeze D: ``auto`` = NVENC where the GPU has it (RTX PRO 6000), libx265 elsewhere; or explicit.
ENCODERS = ("auto", "x265", "nvenc")
SCHEMA = "reelcache.sglang.runtime_config.v1"
#: Fields that identify what an archive stores (method, codec, targets): hashed into session roots. The others
#: (paths, budgets, workers, decoders, verification, scheduling) are run configuration, kept in the root's
#: unhashed telemetry sidecar, so identical sessions produce identical ids wherever they are archived.
CONTENT_FIELDS = ("mode", "representation", "profile", "profile_fields", "energy_profile", "energy_key",
                  "energy_value", "predictor_registry_id", "reuse", "refit", "refit_rule", "rate_lag",
                  "hevc_inter_prediction",
                  "hevc_qp", "hevc_pix_fmt", "hevc_preset", "normalization", "orthogonalizer", "canvas",
                  "matmul_precision")
#: Freeze D (D-A): the refit rule. ``refit_rule`` enters a session's content identity only away from the legacy
#: ``"paper"`` rule, so paper-rule sessions keep their C3 ids (as ``RankMethodConfig`` serializes it).
REFIT_RULES = ("cumulative", "paper")
LEGACY_CONTENT = {"refit_rule": "paper"}

# -- workload v2 (W1b): session trees and the exact recompute arm ------------------------------------------------------
#: ``tree_retention``: ``head`` (F6: a session owns its head path; rewind moves along it) or ``full`` (every node the
#: session generated stays owned through ``tree/<session>/<generation_index:08d>`` refs and is a legal rewind target).
TREE_RETENTIONS = ("head", "full")
#: W1.3 ``tree_switch`` (tree mode only): how a rewind leaves a branch whose commits are still being archived. ``drain``
#: (default, W1.2) waits for every pending commit of the session; ``detach`` waits only for the target's own commit and
#: lets the old branch's pending commits publish in order to a temporary per-branch ref (never the session head).
TREE_SWITCHES = ("drain", "detach")
#: ``restore_mode``: ``archive`` (F6) or ``recompute`` (exact replay of the target path's clean forwards from stored
#: clean latents, behind the same raw tiers; ``representation`` ``latents_only``, or ``raw`` for validation).
RESTORE_MODES = ("archive", "recompute")
#: Nodes hold latents, generator and execution state, no KV objects (``restore_mode="recompute"`` only).
LATENTS_ONLY = "latents_only"
#: CONTRACT §10: the RAW archive behind the raw tiers (GPU raw -> host raw, LRU, the same budget fields) with an
#: asynchronous, ordered write-behind of its RAW objects; F6's ``raw`` stays the synchronous exact reference.
RAW_OFFLOAD = "raw_offload"
#: ``store_io``: ``buffered`` (F6) or ``nocache`` (every durable object write and every object read drops the file's
#: pages, ``posix_fadvise(DONTNEED)``, so the OS page cache never acts as an unbudgeted extra tier).
STORE_IOS = ("buffered", "nocache")
#: wl-w1b7 (symmetric raw-tier LRU, opt-in): ``raw_tier_retain`` -- a published raw entry that loses its last holder is
#: dropped (``drop``, default) or stays as an unheld LRU entry (``lru``; whole-tree delete still drops);
#: ``raw_tier_promote`` -- a restore inserts the rows its miss path produced into host raw, held by the restoring
#: session (``host``) or not (``off``, default). Only raw_offload (store reads) and the recompute arm (replays)
#: promote: the compressed arms' restore-populated host cache is their prepared tier.
RAW_TIER_RETAINS = ("drop", "lru")
RAW_TIER_PROMOTES = ("off", "host")
PROMOTION_SCOPES = {RAW_OFFLOAD: "store_read", LATENTS_ONLY: "replay"}


def available_cpus() -> int:
    """CPUs this process may run on (its affinity mask, e.g. a Slurm allocation; else the machine's count)."""
    import os
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


#: dev22j8v6 (F6 a): the job's CPUs for archival's encoder and fitter threads -- ``"job"`` (default): the job's full
#: CPU set (:func:`job_cpu_set`), whatever affinity SGLang's NCCL init left the worker; ``"inherit"``: the threads keep
#: the affinity they inherit and the derived counts use the attach-time mask (F4-F5.5).
ARCHIVAL_CPUS = ("job", "inherit")
#: dev22j8v6 (F6 b): the fitter's CUDA stream -- ``"dedicated"`` (default): its own, created with the driver API, so
#: no other stream can be it; ``"pooled"``: one of PyTorch's pooled streams (F4-F5.5), which a restore can share.
FITTER_STREAMS = ("dedicated", "pooled")

#: wl-base (workload-v2 W1c): the two baselines and the uniform-rank ablation (opt-in; see the class docstring).
#: ``svd``: fixed-rank SVD factors kept as FP16 (no HEVC, no basis sharing), restored by a P.Q GEMM; ``hevc``: the
#: per-channel-normalized KV matrix itself HEVC-coded (an all-intra node pack with a full-width identity basis).
BASELINE_REPRESENTATIONS = ("svd", "hevc")
REPRESENTATIONS = REPRESENTATIONS + BASELINE_REPRESENTATIONS
#: ``adaptive``: the paper's energy-target ranks; ``fixed_uniform`` (``svd_hevc``): ONE ``uniform_rank`` for every layer
#: and both K and V (no energy target, no sharing); ``fixed_fraction`` (``svd_hevc``; wl-base 2026-10-20): the ``svd``
#: arm's ranks (``svd_rank_fraction_{k,v}`` of the channels, defaults 1/6 and 1/4) coded with HEVC (no energy target,
#: no sharing), so the ablation step SVD only -> SVD + HEVC changes the codec alone.
RANK_POLICIES = ("adaptive", "fixed_uniform", "fixed_fraction")
#: the fixed-rank policies of ``svd_hevc`` (both need ``reuse=False``)
FIXED_RANK_POLICIES = ("fixed_uniform", "fixed_fraction")
#: ``representation="svd"`` rank fractions of the channel count (user decision 2026-10-19: K = C/6, V = C/4).
SVD_RANK_FRACTIONS = {"K": 1 / 6, "V": 1 / 4}
#: The arms' fields are content fields; they enter a session's identity only away from their defaults (F6 ids hold).
CONTENT_FIELDS = CONTENT_FIELDS + ("rank_policy", "uniform_rank", "svd_rank_fraction_k", "svd_rank_fraction_v")
LEGACY_CONTENT = {**LEGACY_CONTENT, "rank_policy": "adaptive", "uniform_rank": None, "svd_rank_fraction_k": None,
                  "svd_rank_fraction_v": None}


def fixed_rank(channels: int, fraction: float, bucket: int = 32) -> int:
    """``channels x fraction`` to the nearest multiple of ``bucket`` (ties up), within ``[min(bucket, C), C]``:
    C 1536 -> K 256 / V 384 (1/6, 1/4); C 3072 -> K 512 / V 768."""
    rank = int(math.floor(channels * fraction / bucket + 0.5)) * bucket
    return max(min(bucket, channels), min(rank, channels))


def _affinity() -> frozenset[int] | None:
    try:
        return frozenset(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


#: The calling thread's CPUs when this module was imported (a fallback of :func:`job_cpu_set`: before SGLang's NCCL
#: init narrows the worker to its GPU's NUMA node, when the import came first).
_IMPORT_AFFINITY = _affinity()


def parse_cpu_list(text: str) -> frozenset[int]:
    """A kernel CPU list (``"0-3,8,10-11"``) as a set; empty for an empty list."""
    cpus: set[int] = set()
    for part in str(text).strip().split(","):
        part = part.strip()
        if not part:
            continue
        low, _, high = part.partition("-")
        first, last = int(low), int(high) if high else int(low)
        if first < 0 or last < first:
            raise ValueError(f"bad CPU range {part!r}")
        cpus.update(range(first, last + 1))
    return frozenset(cpus)


def format_cpu_list(cpus) -> str:
    """A CPU set as a kernel CPU list (``"0-3,8,10-11"``)."""
    ranges, start, previous = [], None, None
    for cpu in sorted(cpus):
        if start is None:
            start = previous = cpu
        elif cpu == previous + 1:
            previous = cpu
        else:
            ranges.append(f"{start}-{previous}" if previous > start else f"{start}")
            start = previous = cpu
    if start is not None:
        ranges.append(f"{start}-{previous}" if previous > start else f"{start}")
    return ",".join(ranges)


def job_cpu_set(*, proc_cgroup: str = "/proc/self/cgroup",
                cgroup_root: str = "/sys/fs/cgroup") -> tuple[frozenset[int], str]:
    """The CPUs of this process's job (dev22j8v6, F6 a): ``(cpus, source)``. ``source`` ``"cgroup"``: the effective
    cpuset of this process's cgroup (v2 ``cpuset.cpus.effective``; v1 ``cpuset.effective_cpus`` / ``cpuset.cpus``),
    the nearest ancestor below the hierarchy's root that has one -- under Slurm the allocation, which a thread's
    ``sched_setaffinity`` (SGLang's NCCL init pinning the worker to its GPU's NUMA node) does not narrow; else
    ``"import"``: the affinity when this module was imported; else ``"affinity"``: the calling thread's now."""
    try:
        with open(proc_cgroup) as handle:
            lines = handle.read().splitlines()
    except OSError:
        lines = []
    candidates = []
    for line in lines:
        hierarchy, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if hierarchy == "0" and controllers == "":
            candidates.append((cgroup_root, path, ("cpuset.cpus.effective",)))
        elif "cpuset" in controllers.split(","):
            candidates.append((os.path.join(cgroup_root, "cpuset"), path, ("cpuset.effective_cpus", "cpuset.cpus")))
    for root, path, names in candidates:
        parts = [part for part in path.split("/") if part]
        for depth in range(len(parts), 0, -1):  # not the hierarchy's root: there no cpuset confines the job
            directory = os.path.join(root, *parts[:depth])
            for name in names:
                try:
                    with open(os.path.join(directory, name)) as handle:
                        cpus = parse_cpu_list(handle.read())
                except (OSError, ValueError):
                    continue
                if cpus:
                    return cpus, "cgroup"
    if _IMPORT_AFFINITY:
        return _IMPORT_AFFINITY, "import"
    current = _affinity()
    if current:
        return current, "affinity"
    return frozenset(range(max(1, os.cpu_count() or 1))), "cpu_count"


def derived_encode_workers(cpus: int | None = None) -> int:
    """F4 default parallel node encodes: one per 8 CPUs of the allocation, 2 to 4 (16 CPUs: 2; 32: 4)."""
    cpus = available_cpus() if cpus is None else cpus
    return max(2, min(4, cpus // 8))


def derived_x265_threads(encode_workers: int, cpus: int | None = None) -> int:
    """F4 default libx265 threads per encode: the allocation's CPUs split over the workers, 2 to 6 each."""
    cpus = available_cpus() if cpus is None else cpus
    return max(2, min(6, cpus // max(1, encode_workers)))


#: Step 7 genfirst: generation-first archival's fields at their defaults (dropped from ``to_dict``: off, every record
#: stays W1.4.1's).
GENFIRST_DEFAULTS = {"defer_archival_during_generation": False, "archival_defer_backlog_fraction": 0.5,
                     "archival_defer_max_age_seconds": 30.0, "archival_defer_inflight": "wait"}
GENFIRST_INFLIGHT = ("wait", "overlap")  # genfirst3: the node on the GPU when a hold opens


@dataclass(frozen=True)
class ReelCacheRuntimeConfig:
    """Bridge configuration; defaults are the paper's ``archive_native`` mode on the Step 6 runtime.

    * ``mode``: ``archive_native`` (generation keeps the valid native active
      KV; clean commits are archived for resume/rewind/fork). The only mode.
    * ``representation``: ``svd_hevc`` (the latest method at the energy
      targets: lineage commits through :class:`reelcache.archive_pipeline.ArchivePipeline`,
      one all-intra node pack per committed range referencing immutable basis
      versions, packed device restore through
      :class:`reelcache.archive_restore.LineageRestorer`) or ``raw`` (the exact
      reference: synchronous RAW packets, bitwise restore; a reference point,
      not a performance path).
    * ``energy_profile``/``energy_key``/``energy_value``: targets for normalized
      pre-RoPE K and V. Default: the model profile's built-in profile
      (provisional K98/V95 except CausVid's final K98/V95). Explicit fractions
      must be given together and are recorded as ``custom`` provisional targets.
    * ``predictor_registry_id``: the shared, model-scoped online rank learner.
    * ``reuse``/``refit``: cross-chunk inherit/certify/augment/refit on basis
      versions (paper method, default on); ``refit_rule``: ``"paper"``
      (the default again since the user decision of 2026-10-17: the per-chunk
      test; C3's rule and session ids) or ``"cumulative"`` (selectable; the
      default of freezes D to F: refit once the coefficient excess an augmented
      basis accrued since its last fresh fit reaches the refit's basis cost);
      ``rate_lag``: refit estimates use
      the coefficient rate measured that many nodes above the parent, so chunk
      ``n`` plans while ``n-1`` encodes (0 = Step 4 reference). Default 3 since
      F4. Step 19's decomposition (A100-40GB, sustained generation; lag 3 vs
      lag 1, CausVid / RAVEN): published nodes/s during generation 1.34x /
      1.15x at 16 CPUs (2 encode workers x 6 x265 threads) and 1.30x / 1.29x
      at 32 CPUs (4 x 6); generation + drain +13 % / +15 % at 16 CPUs; the
      fitter's dependency-wait share 0.54 / 0.45 -> 0.00; no energy misses;
      bytes reachable from published nodes -0.19 % / -0.85 %. Lag 2 gets
      almost all of the gain. Four encode workers on 16 CPUs oversubscribe
      them and lose; the derived encoder defaults prevent it. Planning waits
      on the encode of the node four above, so up to four encodes overlap. A
      fixed lag (never "the latest available") keeps planning deterministic.
    * ``matmul_precision``: the rank method's matmul policy (``None``: the
      model profile's, else the runtime default ``fp32``, which refuses CUDA
      planning while the process allows TF32; ``fp32_verified_fp64`` for
      profiles whose generation runs TF32 matmuls: planning proceeds and every
      acceptance measurement uses FP64 products; global flags are never toggled).
    * ``hevc_inter_prediction``: off by default (genuine all-intra streams).
      Inter coding needs per-matrix packets, which neither the asynchronous
      archive nor the packed restore supports: it is rejected explicitly.
    * Baselines and the uniform-rank ablation (wl-base, workload-v2 W1c; opt-in,
      content fields omitted from session ids at their defaults):
      ``representation="svd"`` stores fixed-rank factors as FP16 (no HEVC) in
      ``reelcache.factor_pack.v1`` node packs and restores them with a P.Q GEMM
      (fused RoPE, same install, same tiers); ranks are ``C x
      svd_rank_fraction_{k,v}`` to the nearest multiple of 32 (defaults 1/6 and
      1/4: C 1536 -> K256 / V384, C 3072 -> K512 / V768). ``representation="hevc"``
      HEVC-codes the per-channel-normalized KV matrix itself (an all-intra node
      pack with a full-width identity basis; same packing, encoder, NVDEC and
      restore kernels as ReelCache). ``rank_policy="fixed_uniform"`` (``svd_hevc``)
      fits ONE ``uniform_rank`` (a multiple of 32, clamped to the channels) for
      every layer and both K and V. ``rank_policy="fixed_fraction"``
      (``svd_hevc``) fits the ``svd`` arm's ranks (``svd_rank_fraction_{k,v}``,
      defaults 1/6 and 1/4, resolved into the session identity like ``svd``'s)
      and codes them with HEVC; it needs ``uniform_rank=None``. These fixed-rank arms have no energy target
      (achieved energies are recorded per matrix) and share no basis across
      chunks: they require ``reuse=False`` (the adaptive policy without sharing is
      ``reuse=False`` alone).
    * Archival execution: ``fit_device`` (planning device), ``encode_workers``
      parallel node encodes of ``x265_threads`` libx265 threads each. Both
      default to None since F4: derived from the CPUs this process may use,
      ``encode_workers = max(2, min(4, cpus // 8))`` and ``x265_threads =
      max(2, min(6, cpus // encode_workers))``, so the encoders never
      oversubscribe the allocation (16 CPUs: 2 x 6, the F3 configuration;
      32: 4 x 6, Step 19's A/B). The archive resolves them once, when the
      bridge attaches (after SGLang's distributed init), and records them
      with the affinity-mask size they came from (``cpus_available``). Thread
      and worker counts never change the stream bytes or any content id.
      ``archival_cpus`` (dev22j8v6): ``"job"`` (default) derives them from
      the job's full CPU set (:func:`job_cpu_set`: the cgroup cpuset, which
      SGLang's NCCL init does not narrow) and gives archival's fitter and
      encoder threads (and so their FFmpeg children) that mask; the
      generation threads keep theirs. ``"inherit"``: the attach-time mask
      (F4-F5.5). Several GPU workers in one job each derive from the whole
      set: configure the counts there. Recorded (``job_cpus``,
      ``job_cpus_source``; the pipeline's ``stats()["cpu_sets"]``).
      ``fitter_stream`` (dev22j8v6): ``"dedicated"`` (default), the fitter's
      own CUDA stream (no restore can share it and inflate the decode
      guard's ``held``), or ``"pooled"`` (F4-F5.5),
      ``max_source_chunks`` (captured chunks whose source tensors the fitter
      may still read; generation blocks beyond it), ``max_unpublished`` (the
      per-session queue of submitted, unpublished commits; a full queue
      blocks the next chunk boundary: reported backpressure),
      ``max_staged_canvases`` (host canvases staged for encoding; a node
      stages one per range, so ``None`` = automatic: enough for the profile's
      largest prompt-switch recache node, at least 2), ``submit_timeout`` (None: wait; else
      ``CapacityError``), ``canvas`` (canonical canvases or own size),
      ``validate_decode`` (FFmpeg validation decode of each pack),
      accel C (dev_accel_20261030, opt-in; defaults = the base; class (i): the same stored bytes,
      decisions and decoded-coefficient digests, :mod:`reelcache.accel_c`): ``validation_decoder``
      (``"ffmpeg"`` or ``"nvdec"``: the validation decode on the bridge's NVDEC pool), ``hash_once``
      (payload digests computed once per in-process bytes object), ``seal_prepare_after_publish``
      (seal-time preparation after the node's publication), ``nvenc_device_input`` (NVENC reads the
      device canvas), ``gil_fastpaths`` (GIL-light parsing / packing / JSON copies), ``native_ext``
      (the Rust extension ``reelcache._native`` for the remaining host hot spots),
      ``stream_priority`` (CUDA priority of the fitter stream),
      ``archival_pump_pause`` (default on: archival's FFmpeg pipe pump pauses
      chunk by chunk while a restore runs, at most ``yield_max_seconds`` = 1 s
      per restore busy period, never while a caller waits on archival),
      ``archival_cpu_priority`` (opt-in CPU priority of archival's FFmpeg
      processes: None = unchanged, ``"idle"`` = SCHED_IDLE, or a nice increment
      1..19; under sustained CPU saturation idle priority can slow archival).
    * Freeze D raw tier and encoder (``reelcache.raw_tier``; Step 7
      ``DESIGN_D_tiers.md``): ``raw_tier`` (default on: every submitted chunk's
      captured pre-RoPE K/V stays as a GPU or host raw copy until its node is
      published, then while the budgets allow; generation waits only when the
      budgets are full of unpublished chunks -- last-resort backpressure,
      recorded with its cause; False = C3 source leases and backpressure),
      ``gpu_raw_budget_bytes`` (None: ``max_source_chunks`` x the largest
      chunk, C3's held-source footprint), ``gpu_raw_reserve_bytes`` (freeze F
      pressure response: device headroom GPU raw copies leave free; None =
      auto: max(2 chunks, 5 % of the device, the largest VAE-decode working set
      observed); 0 = off; below it, copies of fitted and published chunks
      demote to host in the background at admission and synchronously at the
      decode seam), ``host_raw_budget_bytes`` (None:
      half of this worker's share of the cgroup memory headroom after the
      other host pools, at most 64 GiB; a budget below one chunk degrades to
      C3 backpressure, recorded), ``host_raw_pinned`` (page-locked host copies),
      ``raw_tier_workers`` (workers sharing the job's memory cgroup; None:
      ``LOCAL_WORLD_SIZE`` or 1), ``raw_restore`` (``prefer``: restores take
      raw copies first, bitwise the native cache; ``off``: compressed only --
      lossy-quality measurements use it), ``raw_tier_retain`` / ``raw_tier_promote``
      (wl-w1b7, :data:`RAW_TIER_RETAINS` / :data:`RAW_TIER_PROMOTES`; defaults
      ``drop`` / ``off`` = F6), ``max_archival_lag_seconds`` (None;
      else last-resort backpressure also when the estimated drain time of the
      unpublished chunks exceeds it), ``encoder`` (``auto``: NVENC where the GPU
      has it, libx265 elsewhere, chosen once and recorded; ``x265``; ``nvenc``).
    * Publication across requests: archival runs behind generation and
      across request boundaries. ``durability`` says which requests wait for
      their published frontier before returning: ``pause`` (default: an
      explicit pause, i.e. ``pause_after_chunks`` or a pause control, drains;
      a completed request returns at once), ``always`` or ``never``; the
      request handle's ``durable`` overrides it. Checkpoint, fork, rewind and
      close always drain; a restore from the archive waits for the session's
      published frontier. ``resident_reuse``: a resume whose session KV is
      still resident in the stage's caches (nothing wrote them since its last
      request) continues on that native KV without restore or waiting.
    * Workload v2 (W1b): ``tree_retention`` (``head``, default: F6; ``full``:
      every chunk node a session generates stays owned through the store ref
      ``tree/<session>/<generation_index:08d>``, set in the transaction that
      publishes it; any owned node is a rewind target; garbage collection
      keeps owned nodes until the whole tree is deleted) and ``restore_mode``
      (``archive``, default; ``recompute``: an archive restore whose attended
      ranges are not all raw-tier hits replays the target path's clean
      forwards from the stored clean latents in the stage, exactly;
      ``representation="latents_only"`` stores no KV objects and needs it).
      W1.3 ``tree_switch`` (``drain``, default: a rewind first publishes every
      pending commit of the session; ``detach``, tree mode only: it waits only
      for the target's own commit, the old branch's pending commits publish in
      order to a temporary per-branch ref, binding their tree refs).
      ``representation="raw_offload"`` (CONTRACT §10): RAW objects written
      behind the raw tiers (``raw_tier`` required) by an ordered background
      writer; a tier entry is evictable once its node is durable; restores
      read GPU raw, then host raw, then the store, per span. ``store_io``
      (``buffered``, default: F6; ``nocache``: drop each object's pages after
      its durable write and after every read).
    * Restore: ``decoder`` (``nvdec_packed`` or an explicit ``software``
      decode), ``nvdec_workers`` (>= the GPU's NVDEC engines),
      ``restore_arena_bytes``, ``basis_budget_bytes`` (device-resident basis
      versions), ``prepared_budget_bytes`` (prepared packs), ``staging_engine``
      (default on, ``nvdec_packed`` only: one Step 5c
      :class:`reelcache.restore_staging.StagingEngine` per worker on its GPU
      stages cold histories — packs not prepared, bases not resident — for
      restores and session-open prewarms: parallel reads and hash checks into
      a pinned ring, asynchronous basis uploads, no PyTorch CPU ops; restored
      KV bitwise that of the store path, which ``False`` selects;
      ``staging_threads`` / ``staging_prepare_threads``: None = min(8,
      max(2, cpus // 2)) and max(1, threads // 2) of the worker's CPU
      affinity; ``staging_pinned_bytes``: the pinned ring, 512 MiB),
      ``seal_prepare``
      (prepare every published pack for restore while still in memory),
      ``prewarm`` (prepare a persisted session's history when it is opened),
      ``warmup`` (process-level restore warm-up when the backend attaches:
      every restore-kernel variant the bridge's restores launch, the
      engine's RoPE for native-RoPE restores, decoder sessions for every
      canonical canvas when the core offers it; nothing session-specific;
      ``None`` = on for CUDA bridges with the packed NVDEC or the RAW
      restore; its cost is reported in ``bridge.startup["warmup"]``),
      ``rope_mode`` (fused RoPE of the reconstruction kernel, or the engine's
      own rotation).
      ``pipelined_restore`` (decomp_opt 2026-10-25; default off): a cold
      (staged) restore submits each pack's NVDEC jobs as soon as the staging
      engine prepared it, overlapping decoding with the staging of the
      following packs (same KV, bitwise; ``LineageRestorer(pipelined_staging)``).
    * Budgets: ``max_store_bytes``, ``max_resident_bytes`` (RAW staging).
    * Verification: ``verify_reads`` (default on: every archived object read
      for a restore is hash-checked against its content address, so a corrupt
      or missing packet/manifest fails that request with a typed integrity
      error instead of restoring wrong KV; publication, leases and GC keep
      their index-check cost), ``audit_on_open`` (default off: opening the
      store checks every session head's graph — manifests parse, references
      exist — at O(manifests) cost; on: a full hash audit of every stored
      byte), ``on_open_error`` (``quarantine``, default: a session head found
      damaged at open is quarantined — listed in the archive's ``describe()``
      and ``stats()``, typed integrity error when used, omitted from
      ``sessions()`` — while the bridge starts and every other session works;
      ``raise``: the bridge fails to start), ``verify_capture_rope`` (captured RoPE
      tables must equal the engine's recomputation at the recorded position),
      ``verify_post_codec`` (measure the post-codec error of every tile on the
      validation decode; diagnostics only).
      ``verify_bases`` (decomp_opt 2026-10-25; default on): False, with
      ``verify_reads`` off, also skips the staging engine's basis-row hashing,
      so a cold restore hashes nothing (no integrity check of restored bytes).
      accel A1 (2026-10-30; default off): ``decoded_tier_bytes`` > 0 keeps the
      archival validation decode's pictures (normative HEVC output, tile
      rectangles) of recent node packs in that many bytes of pinned host RAM
      (``reelcache.decoded_tier``; needs ``validate_decode``); restores route
      cached nodes past NVDEC (host-to-device copies, same kernels, bitwise the
      same KV): ``decoded_tier_route`` ``"deficit"`` (the oldest nodes until
      NVDEC's predicted time per layer fits the forward's per-layer compute
      ``decoded_tier_layer_ms`` = ``[a, b]``: a + b * history ms, None: the
      model's default) or ``"all"``.
    * Output of requests that restored history (resume, fork, rewind): by
      default only their new chunks are decoded and returned (``output`` of
      the session handle: ``new``; ``full`` returns the whole clip). The
      causal VAE decoder is warmed by ``decode_context_latents`` preceding
      latent frames (their frames are dropped). Default 40: from 40 context
      latents on, the new frames equal the full-clip decode bitwise (float
      and uint8; measured with the real weights of CausVid, LongLive 1.0,
      RAVEN and LongLive 2.0); shorter contexts decode faster with boundary
      errors (max |d| 0.004 at 36, 0.004-0.012 at 32, up to 0.06 at 16).
      Decode cost is bounded by 40 + new latents whatever the history length.
    * ``plan_memory_bytes``: bounds the archive planner's GPU working set
      (Step 7 ``ArchiveOptions.plan_memory_bytes``: matrices are planned and
      finalized in groups that fit; identical plans and bytes; slower). With
      ``max_source_chunks=1`` it bounds the archival memory that overlaps
      generation and VAE decode (e.g. LongLive 2.0 on an A100-80GB).
    * ``defer_archival_during_decode``: the worker's VAE decode is a
      memory-critical section for background archival (Step 7
      ``ArchivePipeline.defer_gpu_work``): no GPU fitting starts inside it, a
      node already on the GPU finishes first (the decode waits for at most
      that node) and the caching allocator's unoccupied blocks (the fitter's
      per-stream pool) are released on entry. Off by default; on for
      LongLive 2.0 on an A100-80GB, with ``max_source_chunks=1`` and
      ``plan_memory_bytes``. A memory policy: without a GPU fitter (RAW
      commits, ``fit_device="cpu"``) there is nothing to defer (no-op). It
      applies to decode sections only (``"vae_decode"``, ``"decode"``,
      ``"*_decode"``). Since F4 a restore section never takes it: a restore
      keeps the entry check, the pressure-triggered deferral, the fitter
      re-check and the OOM relief, so without pressure it demotes, waits for
      and releases nothing (F to F3 evacuated GPU raw copies at every
      restore).
    * ``defer_archival_during_generation`` (Step 7 genfirst, 2026-10-20; default
      off): generation first, then background archival. A request's
      generation -- from its restore's return (or its recompute replay's end)
      to the end of its VAE decode seam, so [t2, vae_end] (not the unheld
      post-decode tail to t3) and LongLive 1.0's first-block recache -- holds
      background archival
      (``ArchivePipeline.open_generation``): no new GPU archival work (fit,
      projection, verification, reconstruction checks) starts inside it, a
      node already on the GPU finishes first (the request's
      ``archival_defer_wait``, an exposed stall of that source) and fitting
      resumes when it ends. The restore is never held; encoding and
      publication are not affected. Starvation guard (never-block): a node
      still starts inside a hold when the unpublished raw backlog exceeds
      ``archival_defer_backlog_fraction`` (0.5) of the raw tier's budget
      (device + host), the oldest unpublished chunk is older than
      ``archival_defer_max_age_seconds`` (30 s) or a caller waits on
      archival (drain, ticket, backpressure); each such node is an override
      by reason. Records: ``archive_stats.pipeline.generation_defer`` and each
      request summary's ``archival_defer``; off, no record changes. A no-op
      without an archive pipeline (raw_offload, recompute).
      ``archival_defer_inflight`` (genfirst3): ``"wait"`` (default, as above)
      or ``"overlap"``: opening a hold does not wait for the node on the GPU;
      its remainder runs beside the request (``inflight_overlap_seconds``,
      no stall) and new fits stay held.
    * ``decode_entry_guard`` (dev22j6, reworked in dev22j7 and dev22j8; default
      on): with a GPU fitter and the raw tier's pressure response, a VAE
      decode needs the larger of the reserve and a prior
      (``decode_reserve_prior_bytes``; None: 16 % of the device, a floor for
      decodes the tier has not learned, e.g. a fresh process), plus the growth
      a node on the GPU may still have (its peak live working set, learned at
      the attach warm-up and online, scaled by the node's size). Checked
      against the device headroom: when it fits nothing is done; short with a
      node on the GPU, the node's measured remaining growth (the bound minus
      what the fitter's streams hold) may fit, else the fitter pauses at its
      next step boundary (or at once while it waits without allocating) for
      the decode (then it grows no more); still short, cached blocks go back
      to the driver (only when the fragments could cover the shortfall) and
      GPU raw copies are demoted; still short, the decode waits for the node.
      No unconditional first-decode deferral (dev22j8). Off: freeze F4's
      entry check only. ``decode_reserve_prior_bytes=0`` drops the prior.
      dev22j8v4: with a node on the GPU the headroom is strict (the driver's
      free memory plus what the allocator can release). dev22j8v5: such a
      decode also needs at least its floor: ``decode_working_set_floor_bytes``
      when set, else 1.1 x the bundled model's clean decode working set at
      its default geometry (``lineage_archive.DECODE_WORKING_SETS``), else
      1.1 x the largest clean decode working set learned in this process;
      before any, the decode waits for the node.
    * ``overlap_restore`` (freeze F5; default on): a session restore of
      compressed history runs layer-overlapped with the request's first
      forward (Step 5c's :mod:`reelcache.overlapped_restore`). The restore thread
      installs layer by layer on its own CUDA stream, in the caller's
      inference mode. The forward's attention of layer ``i`` waits for layer
      ``i`` through SGLang's ``CausalSelfAttentionKVCache.restore_pending``
      (a host wait until issued, then a stream wait). The request joins the
      restore, including its integrity gate, before anything derived from
      the restored KV leaves it. Results are bitwise those of the plain
      restore. An unprepared history (packs not prepared) restores plain
      (recorded). RAW-tier ranges and the RAW reference archive are
      unaffected. ``False`` restores plain. A request's session handle
      ``overlap_restore`` (bool) overrides it for that request.
    * ``cuda_memory_limit_bytes`` (diagnostics): caps this worker's PyTorch
      allocator (``torch.cuda.set_per_process_memory_fraction``) when the
      bridge attaches, to emulate a smaller GPU (e.g. 76 GiB of an
      A100-80GB on a 96 GB card).
    * Diagnostics: ``observe_plain_requests`` times the clean context forwards
      and chunk boundaries of requests without a session (nothing is
      captured or archived; the request generates as without the bridge):
      the in-server recompute baseline of the restore requirement, read back
      with the ``request`` session control.
    """

    store_root: str
    mode: str = "archive_native"
    representation: str = "svd_hevc"
    profile: str | None = None
    profile_fields: Mapping[str, Any] | None = None
    energy_profile: str | None = None
    energy_key: float | None = None
    energy_value: float | None = None
    predictor_registry_id: str = "sglang"
    reuse: bool = True
    refit: bool = True
    refit_rule: str = "paper"
    rate_lag: int = 3
    matmul_precision: str | None = None
    hevc_inter_prediction: bool = False
    hevc_qp: int = 8
    hevc_pix_fmt: str = "yuv444p"
    hevc_preset: str = "ultrafast"
    normalization: str = "per_channel"
    rank_policy: str = "adaptive"  # wl-base (W1c): fixed-rank arms, see the class docstring
    uniform_rank: int | None = None
    svd_rank_fraction_k: float | None = None
    svd_rank_fraction_v: float | None = None
    x265_threads: int | None = None
    encode_workers: int | None = None
    fit_device: str = "cuda"
    orthogonalizer: str = "batched"
    max_source_chunks: int = 2
    max_unpublished: int = 4
    max_staged_canvases: int | None = None
    submit_timeout: float | None = None
    canvas: str = "canonical"
    validate_decode: bool = True
    # accel C (dev_accel_20261030; opt-in, defaults = the base path; reelcache.accel_c)
    validation_decoder: str = "ffmpeg"
    hash_once: bool = False
    seal_prepare_after_publish: bool = False
    nvenc_device_input: bool = False
    gil_fastpaths: bool = False
    native_ext: bool = False
    stream_priority: int | None = None
    # accel B (2026-10-30; opt-in, defaults = the base tree; reelcache.accel_b; dropped from to_dict at their defaults):
    # B1 fitter_green_context_sms: the fitter's stream is a CUDA green context's of that many SMs (CUDA fit device,
    # fitter_stream "dedicated", no stream_priority; fitter_green_workqueue "balanced"/"device_ctx" needs CUDA 13.1+).
    # B3 fitter_cuda_graphs (+ fitter_graph_memory_bytes; None: 1 GiB): CUDA graphs of the FP64 measurement passes
    # (bitwise; a capture fails other threads' device-wide synchronizes -> unsafe next to SGLang generation).
    # B4 fused_fingerprints: one fused Triton integer kernel per source tile (identical fingerprints).
    # B5 finalize_on_fit_device: host sources finalized and fingerprinted on the CUDA fit device. B2 fitter_fp64:
    # "on" (FP64, the base), "off" (the gated FP32 passes) or "fp32:<pass>[+<pass>]" (single passes, for the gates).
    # Execution options, not content: no session id changes.
    fitter_green_context_sms: int | None = None
    fitter_green_workqueue: str | None = None
    fitter_cuda_graphs: bool = False
    fitter_graph_memory_bytes: int | None = None
    fused_fingerprints: bool = False
    finalize_on_fit_device: bool = False
    fitter_fp64: str = "on"
    archival_pump_pause: bool = True
    archival_cpu_priority: Any = None
    raw_tier: bool = True
    gpu_raw_budget_bytes: int | None = None
    gpu_raw_reserve_bytes: int | None = None
    host_raw_budget_bytes: int | None = None
    host_raw_pinned: bool = True
    raw_tier_workers: int | None = None
    raw_restore: str = "prefer"
    raw_tier_retain: str = "drop"
    raw_tier_promote: str = "off"
    # W1.4.3 (E1): the compressed archive's payload budget (bytes; None: no eviction). Ablation arms only (reuse=False);
    # a store whose archive evicted payloads reopens only with a budget (the archive refuses it otherwise)
    archive_budget_bytes: int | None = None
    max_archival_lag_seconds: float | None = None
    encoder: str = "auto"
    durability: str = "pause"
    resident_reuse: bool = True
    tree_retention: str = "head"
    tree_switch: str = "drain"
    restore_mode: str = "archive"
    store_io: str = "buffered"
    decoder: str = "nvdec_packed"
    nvdec_workers: int = 8
    restore_arena_bytes: int = 4 << 30
    basis_budget_bytes: int = 4 << 30
    prepared_budget_bytes: int = 4 << 30
    staging_engine: bool = True
    staging_threads: int | None = None
    staging_prepare_threads: int | None = None
    staging_pinned_bytes: int = 512 << 20
    seal_prepare: bool = True
    prewarm: bool = True
    overlap_restore: bool = True
    pipelined_restore: bool = False
    warmup: bool | None = None
    rope_mode: str = "fused"
    max_store_bytes: int | None = None
    max_resident_bytes: int = 16 << 30
    verify_reads: bool = True
    verify_bases: bool = True
    # accel A: decoded-coefficient tier (A1; 0 = off), its routing policy and per-layer compute model [a, b] ms
    decoded_tier_bytes: int = 0
    decoded_tier_route: str = "deficit"
    decoded_tier_layer_ms: Any = None
    audit_on_open: bool = False
    on_open_error: str = "quarantine"
    verify_capture_rope: bool = True
    verify_post_codec: bool = False
    observe_plain_requests: bool = False
    workload_telemetry: bool = False
    decode_context_latents: int = 40
    plan_memory_bytes: int | None = None
    defer_archival_during_decode: bool = False
    defer_archival_during_generation: bool = False
    archival_defer_backlog_fraction: float = 0.5
    archival_defer_max_age_seconds: float = 30.0
    archival_defer_inflight: str = "wait"
    decode_entry_guard: bool = True
    decode_reserve_prior_bytes: int | None = None
    decode_working_set_floor_bytes: int | None = None
    cuda_memory_limit_bytes: int | None = None
    archival_cpus: str = "job"
    fitter_stream: str = "dedicated"

    def __post_init__(self) -> None:
        if not isinstance(self.store_root, str) or not self.store_root.strip():
            raise ConfigurationError("store_root must name the archive directory")
        object.__setattr__(self, "store_root", os.path.abspath(os.path.expanduser(self.store_root)))
        for name, allowed in (("mode", MODES), ("representation", REPRESENTATIONS + (LATENTS_ONLY, RAW_OFFLOAD)),
                              ("fit_device", FIT_DEVICES),
                              ("decoder", DECODERS), ("rope_mode", ROPE_MODES), ("hevc_pix_fmt", PIX_FMTS),
                              ("normalization", NORMALIZATIONS), ("canvas", CANVASES),
                              ("orthogonalizer", ORTHOGONALIZERS), ("durability", DURABILITY),
                              ("on_open_error", OPEN_ERRORS)):
            if getattr(self, name) not in allowed:
                raise ConfigurationError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        for name in ("profile", "energy_profile"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ConfigurationError(f"{name} must be a nonempty string or None")
        if self.profile_fields is not None:
            if not isinstance(self.profile_fields, Mapping):
                raise ConfigurationError("profile_fields must be a JSON object (ModelProfile fields) or None")
            if self.profile is not None:
                raise ConfigurationError("give either a registered profile name or inline profile_fields")
        if (self.energy_key is None) != (self.energy_value is None):
            raise ConfigurationError("energy_key and energy_value are given together")
        for name in ("energy_key", "energy_value"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or not 0 < value <= 1):
                raise ConfigurationError(f"{name} must be a fraction in (0, 1]")
        if self.refit_rule not in REFIT_RULES:
            raise ConfigurationError(f"refit_rule must be one of {REFIT_RULES}, got {self.refit_rule!r}")
        if self.matmul_precision is not None and self.matmul_precision not in MATMUL_PRECISIONS:
            raise ConfigurationError(f"matmul_precision must be None or one of {MATMUL_PRECISIONS}")
        if self.energy_key is not None and self.energy_profile is not None:
            raise ConfigurationError("give either an energy profile or explicit energy fractions")
        if not isinstance(self.predictor_registry_id, str) or not self.predictor_registry_id:
            raise ConfigurationError("predictor_registry_id must be a nonempty string")
        for name in ("reuse", "refit", "hevc_inter_prediction", "verify_reads", "verify_capture_rope",
                     "verify_post_codec", "validate_decode", "seal_prepare", "prewarm", "resident_reuse",
                     "observe_plain_requests", "defer_archival_during_decode", "decode_entry_guard", "audit_on_open",
                     "staging_engine", "archival_pump_pause", "raw_tier", "host_raw_pinned", "overlap_restore",
                     "pipelined_restore", "verify_bases"):
            if type(getattr(self, name)) is not bool:
                raise ConfigurationError(f"{name} must be a bool")
        if self.decode_reserve_prior_bytes is not None and (type(self.decode_reserve_prior_bytes) is not int
                                                            or self.decode_reserve_prior_bytes < 0):
            raise ConfigurationError("decode_reserve_prior_bytes must be None or a nonnegative byte count")
        if self.decode_working_set_floor_bytes is not None and (type(self.decode_working_set_floor_bytes) is not int
                                                                or self.decode_working_set_floor_bytes < 1):
            raise ConfigurationError("decode_working_set_floor_bytes must be None or a positive byte count")
        if self.warmup is not None and type(self.warmup) is not bool:
            raise ConfigurationError("warmup must be a bool or None (automatic)")
        if type(self.workload_telemetry) is not bool:
            raise ConfigurationError("workload_telemetry must be a bool")
        if type(self.defer_archival_during_generation) is not bool:  # Step 7 genfirst
            raise ConfigurationError("defer_archival_during_generation must be a bool")
        for name, high in (("archival_defer_backlog_fraction", 1.0), ("archival_defer_max_age_seconds", math.inf)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or not 0 < value <= high):
                raise ConfigurationError(f"{name} must be a positive number{' at most 1' if high == 1.0 else ''}")
        if self.archival_defer_inflight not in GENFIRST_INFLIGHT:
            raise ConfigurationError(f"archival_defer_inflight must be one of {GENFIRST_INFLIGHT}")
        if self.hevc_inter_prediction and self.representation == "svd_hevc":
            raise ConfigurationError(
                "inter-coded coefficient streams are per-matrix packets: the asynchronous archive stores "
                "all-intra node packs and the packed restore rejects inter streams; "
                "--reelcache-hevc-inter-prediction is not supported by the SGLang bridge"
            )
        if self.tree_retention not in TREE_RETENTIONS:
            raise ConfigurationError(f"tree_retention must be one of {TREE_RETENTIONS}, got {self.tree_retention!r}")
        if self.tree_switch not in TREE_SWITCHES:
            raise ConfigurationError(f"tree_switch must be one of {TREE_SWITCHES}, got {self.tree_switch!r}")
        if self.tree_switch == "detach" and self.tree_retention != "full":
            raise ConfigurationError("tree_switch='detach' moves the head off branches that must stay owned: it needs "
                                     "tree_retention='full'")
        if self.restore_mode not in RESTORE_MODES:
            raise ConfigurationError(f"restore_mode must be one of {RESTORE_MODES}, got {self.restore_mode!r}")
        if self.representation == LATENTS_ONLY and self.restore_mode != "recompute":
            raise ConfigurationError("representation='latents_only' stores no KV: it needs restore_mode='recompute'")
        if self.restore_mode == "recompute" and self.representation not in (LATENTS_ONLY, "raw"):
            raise ConfigurationError("restore_mode='recompute' replays from stored clean latents: representation "
                                     f"'latents_only' (or 'raw' for validation), not {self.representation!r}")
        if self.store_io not in STORE_IOS:
            raise ConfigurationError(f"store_io must be one of {STORE_IOS}, got {self.store_io!r}")
        if self.representation == RAW_OFFLOAD and not self.raw_tier:
            raise ConfigurationError("representation='raw_offload' writes RAW objects behind the raw tiers: it needs "
                                     "raw_tier=true")
        self._check_fixed_rank_arms()
        if not self.verify_bases and self.verify_reads:
            raise ConfigurationError("verify_bases=False skips basis hashing only when reads are unverified "
                                     "(verify_reads=False)")
        # accel A: decoded-coefficient tier (A1)
        if type(self.decoded_tier_bytes) is not int or self.decoded_tier_bytes < 0:
            raise ConfigurationError("decoded_tier_bytes must be a nonnegative byte count (0: off)")
        if self.decoded_tier_route not in ("deficit", "all"):
            raise ConfigurationError("decoded_tier_route must be 'deficit' or 'all'")
        if self.decoded_tier_layer_ms is not None:
            pair = self.decoded_tier_layer_ms
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                           for v in pair)):
                raise ConfigurationError("decoded_tier_layer_ms must be None or [a, b]: nonnegative milliseconds")
            object.__setattr__(self, "decoded_tier_layer_ms", tuple(float(v) for v in pair))
        if self.decoded_tier_bytes and not self.validate_decode:
            raise ConfigurationError("the decoded tier is filled from the archival validation decode "
                                     "(validate_decode=true)")
        if self.verify_post_codec and not self.validate_decode:
            raise ConfigurationError("verify_post_codec measures on the validation decode (validate_decode=true)")
        # accel C: the opt-in host-footprint levers (reelcache.accel_c)
        from ...accel_c import VALIDATION_DECODERS
        if self.validation_decoder not in VALIDATION_DECODERS:
            raise ConfigurationError(f"validation_decoder must be one of {VALIDATION_DECODERS}, got "
                                     f"{self.validation_decoder!r}")
        for name in ("hash_once", "seal_prepare_after_publish", "nvenc_device_input", "gil_fastpaths", "native_ext"):
            if type(getattr(self, name)) is not bool:
                raise ConfigurationError(f"{name} must be a bool")
        if self.validation_decoder == "nvdec" and not self.validate_decode:
            raise ConfigurationError("validation_decoder='nvdec' selects the decoder of the validation decode: it "
                                     "needs validate_decode=true")
        for name, low, high in (("hevc_qp", 0, 51), ("x265_threads", 1, 64), ("nvdec_workers", 1, 64),
                                ("encode_workers", 1, 64), ("rate_lag", 0, 8), ("max_source_chunks", 1, 64),
                                ("max_unpublished", 1, 256), ("max_staged_canvases", 1, 64),
                                ("decode_context_latents", 1, 4096), ("staging_threads", 1, 64),
                                ("staging_prepare_threads", 1, 64)):
            value = getattr(self, name)
            if name in ("max_staged_canvases", "staging_threads", "staging_prepare_threads",
                        "x265_threads", "encode_workers") and value is None:
                continue
            if type(value) is not int or not low <= value <= high:
                raise ConfigurationError(f"{name} must be an integer in [{low}, {high}]")
        if self.max_unpublished < self.max_source_chunks:
            raise ConfigurationError("max_unpublished must admit every source-holding chunk")
        for name in ("restore_arena_bytes", "max_resident_bytes", "basis_budget_bytes", "prepared_budget_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ConfigurationError(f"{name} must be a positive integer")
        if type(self.staging_pinned_bytes) is not int or self.staging_pinned_bytes < 1 << 20:
            raise ConfigurationError("staging_pinned_bytes must be an integer of at least 1 MiB")
        if self.max_store_bytes is not None and (type(self.max_store_bytes) is not int or self.max_store_bytes <= 0):
            raise ConfigurationError("max_store_bytes must be a positive integer or None")
        if self.submit_timeout is not None and (isinstance(self.submit_timeout, bool)
                                                or not isinstance(self.submit_timeout, (int, float))
                                                or not math.isfinite(self.submit_timeout)
                                                or self.submit_timeout < 0):
            raise ConfigurationError("submit_timeout must be None or a nonnegative number of seconds")
        for name in ("plan_memory_bytes", "cuda_memory_limit_bytes"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ConfigurationError(f"{name} must be a positive integer or None")
        if self.stream_priority is not None and type(self.stream_priority) is not int:
            raise ConfigurationError("stream_priority must be an integer CUDA stream priority or None")
        # accel B (2026-10-30): the fitter options' values (reelcache.accel_b.FitterAccel) and their combinations
        from ...accel_b import FitterAccel
        accel = FitterAccel.from_config(self)
        if accel is not None and accel.needs_cuda and self.fit_device != "cuda":
            raise ConfigurationError("the accel B fitter options (fitter_green_context_sms, fitter_cuda_graphs, "
                                     "finalize_on_fit_device, fitter_fp64) need fit_device='cuda'")
        if self.fitter_green_context_sms is not None and (self.fitter_stream != "dedicated"
                                                          or self.stream_priority is not None):
            raise ConfigurationError("fitter_green_context_sms replaces the dedicated fitter stream: it needs "
                                     "fitter_stream='dedicated' and no stream_priority")
        if self.archival_cpus not in ARCHIVAL_CPUS:
            raise ConfigurationError(f"archival_cpus must be one of {ARCHIVAL_CPUS}")
        if self.fitter_stream not in FITTER_STREAMS:
            raise ConfigurationError(f"fitter_stream must be one of {FITTER_STREAMS}")
        if not (self.archival_cpu_priority is None or self.archival_cpu_priority == "idle"
                or (type(self.archival_cpu_priority) is int and 1 <= self.archival_cpu_priority <= 19)):
            raise ConfigurationError("archival_cpu_priority must be None, 'idle' or a nice increment in [1, 19]")
        if not isinstance(self.hevc_preset, str) or not self.hevc_preset:
            raise ConfigurationError("hevc_preset must be a nonempty string")
        for name in ("gpu_raw_budget_bytes", "gpu_raw_reserve_bytes", "host_raw_budget_bytes"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ConfigurationError(f"{name} must be a nonnegative integer or None")
        if self.raw_tier_workers is not None and (type(self.raw_tier_workers) is not int
                                                  or not 1 <= self.raw_tier_workers <= 64):
            raise ConfigurationError("raw_tier_workers must be an integer in [1, 64] or None")
        if self.raw_restore not in RAW_RESTORES:
            raise ConfigurationError(f"raw_restore must be one of {RAW_RESTORES}, got {self.raw_restore!r}")
        if self.raw_tier_retain not in RAW_TIER_RETAINS:
            raise ConfigurationError(f"raw_tier_retain must be one of {RAW_TIER_RETAINS}, got {self.raw_tier_retain!r}")
        if self.raw_tier_promote not in RAW_TIER_PROMOTES:
            raise ConfigurationError(f"raw_tier_promote must be one of {RAW_TIER_PROMOTES}, got "
                                     f"{self.raw_tier_promote!r}")
        if (self.raw_tier_retain, self.raw_tier_promote) != ("drop", "off") and (
                not self.raw_tier or self.representation == "raw"):
            raise ConfigurationError("raw_tier_retain / raw_tier_promote act on the raw tier: they need raw_tier=true "
                                     "and a representation that has one (not 'raw', the synchronous exact reference)")
        if self.archive_budget_bytes is not None:  # W1.4.3 (E1): LRU eviction of an ablation arm's archive
            if type(self.archive_budget_bytes) is not int or self.archive_budget_bytes <= 0:
                raise ConfigurationError("archive_budget_bytes must be None or a positive integer")
            if self.representation not in ("svd_hevc", "svd", "hevc"):
                raise ConfigurationError("archive_budget_bytes evicts a compressed archive's node payloads: it needs "
                                         f"representation svd_hevc, svd or hevc, not {self.representation!r}")
            if self.reuse:
                raise ConfigurationError("archive_budget_bytes needs reuse=False: with basis sharing a node's payload "
                                         "is part of its descendants', so the ReelCache arm never evicts")
        if self.raw_tier_promote == "host" and self.representation not in PROMOTION_SCOPES and (
                self.archive_budget_bytes is None):
            raise ConfigurationError(
                f"raw_tier_promote='host' promotes only raw_offload store reads and recompute replays "
                f"({sorted(PROMOTION_SCOPES)}), and the replays of an evicting ablation arm (archive_budget_bytes); "
                f"representation {self.representation!r} is a compressed arm, whose "
                "restore-populated host cache is the prepared tier (prepared_budget_bytes)")
        if self.max_archival_lag_seconds is not None and (
                isinstance(self.max_archival_lag_seconds, bool)
                or not isinstance(self.max_archival_lag_seconds, (int, float))
                or not math.isfinite(self.max_archival_lag_seconds) or self.max_archival_lag_seconds <= 0):
            raise ConfigurationError("max_archival_lag_seconds must be None or a positive number of seconds")
        if self.encoder not in ENCODERS:
            raise ConfigurationError(f"encoder must be one of {ENCODERS}, got {self.encoder!r}")

    # -- derived -------------------------------------------------------------------------------
    def targets(self, default_profile: str) -> EnergyTargets:
        if self.energy_key is not None:
            return EnergyTargets("custom", float(self.energy_key), float(self.energy_value), True,
                                 "explicit --reelcache-config energy fractions")
        return energy_targets(self.energy_profile or default_profile)

    def _check_fixed_rank_arms(self) -> None:
        """wl-base (W1c): the baselines' and the uniform-rank ablation's fields (illegal combinations raise)."""
        if self.rank_policy not in RANK_POLICIES:
            raise ConfigurationError(f"rank_policy must be one of {RANK_POLICIES}, got {self.rank_policy!r}")
        if self.rank_policy == "fixed_uniform":
            if self.representation != "svd_hevc":
                raise ConfigurationError("rank_policy='fixed_uniform' is the svd_hevc uniform-rank ablation")
            if (type(self.uniform_rank) is not int or self.uniform_rank < 32 or self.uniform_rank % 32
                    or self.uniform_rank > 65536):
                raise ConfigurationError("uniform_rank must be a positive multiple of 32 (a rank bucket), at most "
                                         "65536; it is clamped to the channel count")
        elif self.uniform_rank is not None:
            raise ConfigurationError("uniform_rank needs rank_policy='fixed_uniform'")
        if self.rank_policy == "fixed_fraction" and self.representation != "svd_hevc":
            raise ConfigurationError("rank_policy='fixed_fraction' is the svd_hevc ablation at the svd arm's ranks "
                                     "(SVD only is representation='svd')")
        fractions = ("svd_rank_fraction_k", "svd_rank_fraction_v")
        if self.representation == "svd" or self.rank_policy == "fixed_fraction":
            for name, target in zip(fractions, ("K", "V")):
                value = getattr(self, name)
                if value is None:  # resolved, so the session identity states the ranks' fractions
                    object.__setattr__(self, name, SVD_RANK_FRACTIONS[target])
                elif (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                      or not 0 < value <= 1):
                    raise ConfigurationError(f"{name} must be a fraction in (0, 1]")
                else:
                    object.__setattr__(self, name, float(value))
        elif any(getattr(self, name) is not None for name in fractions):
            raise ConfigurationError("svd_rank_fraction_k / svd_rank_fraction_v apply to representation='svd' or "
                                     "rank_policy='fixed_fraction' only")
        fixed = self.representation in BASELINE_REPRESENTATIONS or self.rank_policy in FIXED_RANK_POLICIES
        if fixed and self.reuse:
            raise ConfigurationError(f"representation={self.representation!r} rank_policy={self.rank_policy!r} is a "
                                     "fixed-rank arm without basis sharing: set reuse=False")
        if self.representation in BASELINE_REPRESENTATIONS and self.hevc_inter_prediction:
            raise ConfigurationError(f"representation={self.representation!r} stores no inter-coded HEVC stream")

    def fixed_ranks(self, channels: Mapping[str, int]) -> dict[str, int] | None:
        """wl-base (W1c): the fixed rank of K and V for these channel counts, or None (adaptive ranks; the
        ``hevc`` identity basis, whose width is the channel count). ``svd`` and ``fixed_fraction``: :func:`fixed_rank`
        of the fractions (the same ranks); ``fixed_uniform``: ``uniform_rank`` clamped to the channels (the planner
        also clamps to a chunk's tokens)."""
        if self.representation == "svd" or self.rank_policy == "fixed_fraction":
            return {"K": fixed_rank(int(channels["K"]), self.svd_rank_fraction_k),
                    "V": fixed_rank(int(channels["V"]), self.svd_rank_fraction_v)}
        if self.rank_policy == "fixed_uniform":
            return {target: min(self.uniform_rank, int(channels[target])) for target in ("K", "V")}
        return None

    def codec_template(self) -> CodecSpec:
        if self.representation == "raw":
            return CodecSpec()
        # rank=1 is a template placeholder; RankMethodConfig.spec_for replaces it per matrix.
        return CodecSpec(kind="svd" if self.representation == "svd" else "svd_hevc", rank=1,
                         normalization=self.normalization, pix_fmt=self.hevc_pix_fmt,
                         qp=self.hevc_qp, preset=self.hevc_preset, inter_prediction=self.hevc_inter_prediction)

    @property
    def packing(self) -> str:
        """Node packing: one all-intra stream per committed range (the only packing of the bridge)."""
        return "node"

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        if self.profile_fields is not None:
            data["profile_fields"] = dict(self.profile_fields)
        if data.get("tree_switch") == "drain":  # W1.3: the default stays out (drain-mode records equal W1.2's)
            del data["tree_switch"]
        for name in ("pipelined_restore", "verify_bases", "decoded_tier_bytes", "decoded_tier_route",
                     "decoded_tier_layer_ms"):  # decomp_opt + accel A: the defaults stay out (W1.4.1's records)
            if data.get(name) == type(self).__dataclass_fields__[name].default:
                del data[name]
        for name, default in (("raw_tier_retain", "drop"), ("raw_tier_promote", "off"),
                              ("archive_budget_bytes", None)):  # wl-w1b7 / W1.4.3: likewise
            if data.get(name) == default:
                del data[name]
        for name, default in GENFIRST_DEFAULTS.items():  # Step 7 genfirst: the defaults stay out (W1.4.1's records)
            if data.get(name) == default:
                del data[name]
        for name in ("fitter_green_context_sms", "fitter_green_workqueue", "fitter_cuda_graphs",
                     "fitter_graph_memory_bytes", "fused_fingerprints", "finalize_on_fit_device", "fitter_fp64",
                     "validation_decoder", "hash_once", "seal_prepare_after_publish", "nvenc_device_input",
                     "gil_fastpaths", "native_ext"):  # accel B + C (composition): the defaults stay out (the base's records)
            if data.get(name) == type(self).__dataclass_fields__[name].default:
                del data[name]
        return {"schema": SCHEMA, **data}

    def archival_cpu_count(self) -> int:
        """The CPUs the derived encoder counts split (dev22j8v6): the job's (``archival_cpus="job"``) or the calling
        thread's mask (``"inherit"``, F4-F5.5)."""
        return len(job_cpu_set()[0]) if self.archival_cpus == "job" else available_cpus()

    def resolved_encode_workers(self) -> int:
        """``encode_workers``, or (None) ``max(2, min(4, cpus // 8))`` for :meth:`archival_cpu_count`."""
        if self.encode_workers is not None:
            return self.encode_workers
        return derived_encode_workers(self.archival_cpu_count())

    def resolved_x265_threads(self) -> int:
        """``x265_threads``, or (None) ``max(2, min(6, cpus // encode_workers))`` for :meth:`archival_cpu_count`."""
        if self.x265_threads is not None:
            return self.x265_threads
        return derived_x265_threads(self.resolved_encode_workers(), self.archival_cpu_count())

    def content_dict(self) -> dict[str, Any]:
        """The archive-identifying subset of :meth:`to_dict` (see :data:`CONTENT_FIELDS`)."""
        data = self.to_dict()
        return {"schema": SCHEMA, **{name: data[name] for name in CONTENT_FIELDS
                                     if name not in LEGACY_CONTENT or data[name] != LEGACY_CONTENT[name]}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReelCacheRuntimeConfig:
        if not isinstance(data, Mapping):
            raise ConfigurationError("ReelCache runtime configuration must be a JSON object")
        values = dict(data)
        schema = values.pop("schema", SCHEMA)
        if schema != SCHEMA:
            raise ConfigurationError(f"unsupported ReelCache runtime configuration schema {schema!r}")
        known = {item.name for item in dataclasses.fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ConfigurationError(f"unknown ReelCache runtime configuration keys: {unknown}")
        if "store_root" not in values:
            raise ConfigurationError("ReelCache runtime configuration requires store_root")
        return cls(**values)


__all__ = ["ARCHIVAL_CPUS", "CANVASES", "CONTENT_FIELDS", "DECODERS", "DURABILITY", "FIT_DEVICES", "FITTER_STREAMS",
           "MATMUL_PRECISIONS", "MODES", "ReelCacheRuntimeConfig", "OPEN_ERRORS", "PROMOTION_SCOPES", "RAW_TIER_PROMOTES",
           "RAW_TIER_RETAINS", "REPRESENTATIONS", "ROPE_MODES"]
