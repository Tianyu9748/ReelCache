"""Lossy session archive of the SGLang bridge on the Step 6 runtime (``representation="svd_hevc"``).

Each session is a :class:`~reelcache.manager.KVRuntime` lineage (head
``lineages/<session>``): the root's initial cache epoch records the session
identity; every committed clean block is a ``chunk`` node of all CFG streams
(one all-intra node pack referencing immutable basis versions, latest method
at the energy targets, inherit/certify/augment/refit across chunks); a cache
epoch that re-materializes earlier chunks (LongLive 1.0 prompt switch) is one
``recache`` node whose ``kv_recache`` epoch names those chunks. Bridge data
the runtime does not model (block index, RoPE start, scene cut, shot, native
cache cursors, clean latents, generator states) is the node's execution state.

Archival is asynchronous (:class:`~reelcache.archive_pipeline.ArchivePipeline`):
the commit at a chunk boundary only submits the captured source tensors
(generated frontier); fitting, encoding and in-order publication (published
frontier) run behind generation, across request boundaries, with bounded
queues (``max_unpublished`` per session) and explicit backpressure. A pause,
checkpoint, fork, rewind, close or archive restore drains the session
(:meth:`LineageSessionArchive.drain`); an idle, fully published session's
pipeline state is released (:meth:`LineageSessionArchive.release`).
Restore is the packed device path (:class:`~reelcache.archive_restore.LineageRestorer`;
``SoftwareLineageRestorer`` on request), with seal-time preparation of every
published pack and ``prewarm`` when a persisted session is opened.

Freeze D (``raw_tier``, default on): the pipeline keeps every submitted chunk's
captured pre-RoPE K/V in a :class:`~reelcache.raw_tier.RawChunkTier` (device
copies under ``gpu_raw_budget_bytes``, host copies under
``host_raw_budget_bytes``) until its node is published, and afterwards while
the budgets allow (least recently used first). A restore takes each attended
range from its best tier: a raw copy (GPU, else host; installed through the
RAW path with the engine's own RoPE: bitwise the native cache) or the
compressed node (prepared or cold). When a published entry's last raw copy is
evicted its node pack is prepared in host memory (the restorer's prepared
LRU), so the next restore of it is a prepared one. ``encoder="auto"`` (default)
encodes with NVENC on GPUs that have it and libx265 elsewhere, chosen once at
construction and recorded.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import gc
import inspect
import os
import shutil
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from . import config as sglang_config
from ...archive_pipeline import ArchivePipeline
from ...codecs import HEVCProvider
from ...errors import CapacityError, ConfigurationError, IntegrityError, StateError, UnsupportedError
from ...lineage import LINEAGE_PREFIX
from ...lineage_commit import ArchiveOptions
from ...manager import KVRuntime
from ...node_packets import CANVAS_RANK_LADDER
from ...predictor_registry import RankPredictorRegistry
from ...rank_method import RankMethodConfig
from ...raw_tier import RawChunkTier
from ...serialization import deserialize_tensors
from ...types import EnergyTargets, MatrixKey, ModelIdentity, entry_parts
from .commit import matrix_contexts
from .nodes import (LATENTS_KEY, RECORD_KEY, BridgeNode, CoverageRange, annotate, chunk_node, is_detached_branch,
                    recache_node, record_tensor, root_node, split_record, tensor_record)
from .restore import (AttentionKVLease, DenseInstall, LineageInstall, check_restorable, kernel_warmup_shapes,
                      release_cached_blocks, restore_dense, warm_restore_kernels, warmup_component)
from .workload_telemetry import active_telemetry, compressed_tier_label, raw_tier_label, tier_segment

SESSION_DETAILS_KEY = "sglang_session"


def staging_threads(config: Any) -> tuple[int, int]:
    """The staging engine's ``(threads, prepare_threads)``: configured, else min(8, max(2, cpus // 2)) and
    max(1, threads // 2) of the worker's CPU affinity (Step 5c's cold rows: 8 / 4 on 16-CPU workers)."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    threads = config.staging_threads or min(8, max(2, cpus // 2))
    return threads, config.staging_prepare_threads or max(1, threads // 2)


#: Freeze F: cap of a pressure-triggered section's wait for the fitter to leave GPU work (seconds); on timeout the
#: section runs without deferral (recorded).
PRESSURE_DEFER_TIMEOUT = 30.0
#: dev22j6: decode sections that defer archival while the reserve is not learned yet (the process's first decodes;
#: a section that sets no new allocator peak learns nothing, so the guard gave up after this many). dev22j8: no longer
#: used -- an unlearned decode takes the same checks (the prior as its floor) and waits only when still short after
#: the pause and the relief (``unlearned_reserve``); kept for importers.
UNLEARNED_DECODE_DEFERRALS = 2
#: dev22j6: the decode guard's prior (a share of the device) when ``decode_reserve_prior_bytes`` is None: a floor for
#: the decode target, so a decode the tier has not learned yet (a fresh process) gets room (RAVEN 12-chunk
#: never-block decodes on an A100-40GB: 5.15-5.95 GB; prior 6.3 GiB).
DECODE_RESERVE_PRIOR_FRACTION = 0.16
#: dev22j8v5 (F5.4 finding 1): the clean decode working set -- the VAE decode, bounded by ``decode_context_latents``
#: plus the new latents -- of the bundled models at their default geometry (post-patch token grid), the largest
#: measured on 5b's F5-F5.4 records with no node in flight or the fitter paused: CausVid 6.88, RAVEN 6.81, LongLive 2.0
#: 14.42 GiB. With a node on the GPU a decode needs at least ``DECODE_FLOOR_MARGIN`` x this (the decode floor). Not
#: scaled to other geometries, which learn theirs. dev22j8v6 (F6 c): LongLive 1.0 6.80 GiB (F5.4 4446362, request 5,
#: no node in flight, both lag arms).
DECODE_WORKING_SETS = {"causvid": (7390852608, (30, 52)), "raven": (7308185088, (30, 52)),
                       "longlive1": (7300996608, (30, 52)), "longlive2": (15486119936, (22, 40))}
DECODE_FLOOR_MARGIN = 1.1


def decode_reserve_prior_bytes(config: Any, memory: Mapping[str, int]) -> int:
    """The decode guard's prior: ``decode_reserve_prior_bytes`` when configured, else
    ``DECODE_RESERVE_PRIOR_FRACTION`` of the device's total memory (dev22j6)."""
    configured = getattr(config, "decode_reserve_prior_bytes", None)
    if configured is not None:
        return int(configured)
    return int(DECODE_RESERVE_PRIOR_FRACTION * int(memory.get("total", 0) or 0))


def _decode_guard_state(enabled: bool) -> dict[str, Any]:
    """The decode entry guard's counters (``stats()["decode_guard"]``). dev22j7 adds ``fits`` (nothing done),
    ``pauses`` / ``pause_acks`` / ``pause_ack_timeouts`` (the fitter paused at a step boundary) and ``fits_paused``,
    ``fits_after_relief``; ``inflight_deferrals`` now counts waits for a node that only its growth made necessary
    (cause ``short_after_relief``). dev22j8 adds ``fits_measured`` (the node's measured remaining growth fits: no
    pause), ``pause_landed`` (acknowledged while the fitter waited without allocating) and ``cache_release_skips``
    (short, but the fragmented cached bytes could not cover the shortfall); ``unlearned_deferrals`` now counts
    unlearned decodes still short after the pause and the relief. dev22j8v4 (F5.2's RAVEN decode OOM): checks with a
    node on the GPU decide on the strict headroom (``strict_checks``; ``strict_overrides``: the old formula would have
    let the decode run, the strict one did not), whose snapshots cost ``snapshot_seconds`` (sum and max);
    ``strict_release_skips``: short on the strict headroom, the cache release is skipped (it would only turn
    releasable bytes, already counted, into free ones). dev22j8v5 (F5.4 findings): ``free_fits`` (the driver's free
    memory alone covered a decode next to a node: no snapshot), ``floor_raised`` (the decode floor raised the target),
    ``floor_waits`` (no floor yet: the decode waited for the node), ``pauses_first`` (short on the formula: paused
    before the snapshot), ``snapshot_gcs`` (garbage collections that ran during snapshots)."""
    return {"enabled": enabled, "checks": 0, "cache_releases": 0, "reliefs": 0, "unlearned_deferrals": 0,
            "inflight_checks": 0, "inflight_relieved": 0, "inflight_deferrals": 0, "short_deferrals": 0,
            "fits": 0, "pauses": 0, "pause_acks": 0, "pause_ack_timeouts": 0, "fits_paused": 0,
            "fits_after_relief": 0, "fits_measured": 0, "pause_landed": 0, "cache_release_skips": 0,
            "strict_checks": 0, "strict_overrides": 0, "strict_release_skips": 0, "snapshot_seconds": 0.0,
            "snapshot_seconds_max": 0.0, "free_fits": 0, "floor_raised": 0, "floor_waits": 0, "pauses_first": 0,
            "snapshot_gcs": 0}
#: Host bytes reserved per pipeline canvas staging slot when the raw tier derives its host budget (R5).
CANVAS_RESERVE_BYTES = 256 << 20
ENCODERS = ("auto", "x265", "nvenc")


def select_encoder(config: Any, device: torch.device) -> tuple[Any, dict[str, Any]]:
    """``(NVENCEncoder or None, record)``: ``encoder="auto"`` opens NVENC on a CUDA device that has it (RTX PRO
    6000), libx265 elsewhere (A100, H200, CPU); ``"nvenc"`` requires it, ``"x265"`` never probes. Probed once; the
    choice holds for the process (a later NVENC failure is a typed archival failure, never a silent fallback)."""
    requested = getattr(config, "encoder", "x265")
    record: dict[str, Any] = {"requested": requested, "selected": "libx265", "reason": None, "probe_seconds": 0.0,
                              "nvenc_device_bytes": None, "device": str(device)}
    if config.representation not in ("svd_hevc", "hevc"):  # wl-base (W1c): the hevc baseline encodes like svd_hevc
        record.update(selected=None, reason=f"{config.representation} representation: no HEVC encode")
        return None, record
    if requested == "x265":
        record["reason"] = "encoder='x265'"
        return None, record
    if device.type != "cuda":
        if requested == "nvenc":
            raise ConfigurationError("encoder='nvenc' needs the bridge on a CUDA device")
        record["reason"] = "the bridge device is not CUDA"
        return None, record
    started = time.perf_counter()
    free_before = torch.cuda.mem_get_info(device)[0]
    try:
        from ...nvenc import NVENCEncoder
        encoder = NVENCEncoder(qp=config.hevc_qp, preset="p1", tuning="hq", gpu_id=device.index,
                               pix_fmts=(config.hevc_pix_fmt,))
    except Exception as exc:  # noqa: BLE001 - "auto": no NVENC on this device (recorded); "nvenc": raised
        if requested == "nvenc":
            raise UnsupportedError(f"encoder='nvenc': {type(exc).__name__}: {exc}") from exc
        record.update(reason=f"NVENC unavailable: {type(exc).__name__}: {str(exc).strip()[:300]}",
                      probe_seconds=time.perf_counter() - started)
        return None, record
    record.update(selected="nvenc", reason=f"NVENC available on {encoder.implementation.get('gpu')}",
                  probe_seconds=time.perf_counter() - started,
                  # Device memory outside PyTorch (sessions, surfaces) after the qualification encode (R5).
                  nvenc_device_bytes=max(0, free_before - torch.cuda.mem_get_info(device)[0]),
                  preset=encoder.settings.label, qp=encoder.settings.qp,
                  driver=encoder.implementation.get("driver_version"),
                  library=encoder.implementation.get("library_version"))
    return encoder, record


def fixed_rank_fields(config: Any, model: ModelIdentity) -> dict[str, Any]:
    """wl-base (W1c): the :class:`RankMethodConfig` fields of the fixed-rank arms for this model's channel counts --
    ``hevc``: the full-width identity basis; ``svd`` / ``rank_policy="fixed_uniform"``: ``fixed_rank_k`` /
    ``fixed_rank_v`` (``ReelCacheRuntimeConfig.fixed_ranks``); ``{}`` for the adaptive method (configs without these
    fields included)."""
    if getattr(config, "representation", None) == "hevc":
        return {"fixed_basis": "identity"}
    ranks_of = getattr(config, "fixed_ranks", None)
    ranks = ranks_of({target: model.channels_for(target) for target in ("K", "V")}) if callable(ranks_of) else None
    return {} if ranks is None else {"fixed_rank_k": ranks["K"], "fixed_rank_v": ranks["V"]}


def factor_warmup_shapes(profile: Any, model: ModelIdentity, method: RankMethodConfig, *,
                         rope_mode: str = "fused") -> list[Any]:
    """wl-base (W1c): ``restore_kernels.WarmupShape`` of the SVD-only arm's restores (FP16 coefficient sources, the
    method's fixed ranks; K with fused RoPE, or the engine's RoPE from pre-RoPE K in its capture dtype)."""
    from ... import restore_kernels as rk

    cache = getattr(torch, profile.native_dtype)
    ranks = sorted(rank for rank in (method.fixed_rank_k, method.fixed_rank_v) if rank is not None) or [1, 1024]
    common = {"dtype": cache, "source": "float", "ranks": (ranks[0], ranks[-1]), "source_dtype": torch.float16}
    if rope_mode == "native":
        return [rk.WarmupShape(profile.num_heads, profile.head_dim, rope="native",
                               native_rope_dtype=getattr(torch, model.dtype_for("K")), **common)]
    return [rk.WarmupShape(profile.num_heads, profile.head_dim, rope="fused", **common)]



#: F4: the sections ``defer_archival_during_decode`` defers unconditionally (the VAE decode seams); see
#: :func:`policy_defers`.
POLICY_DEFER_REASONS = frozenset({"vae_decode", "decode"})


def policy_defers(reason: str) -> bool:
    """Whether the ``defer_archival_during_decode`` policy applies to a ``memory_critical(reason)`` section: decode
    sections only (``"vae_decode"``, ``"decode"`` or any ``"*_decode"`` label), never a restore."""
    return isinstance(reason, str) and (reason in POLICY_DEFER_REASONS or reason.endswith("_decode"))

def _watch_section(archive: Any, tier: Any, reason: str):
    """The pipeline's watched-section context of a ``memory_critical`` section (dev22j4), or None: a GPU fitter, a
    raw tier with the pressure response on and a pipeline that watches sections are needed."""
    pipeline = getattr(archive, "pipeline", None)
    if (getattr(archive, "fitter", None) is None or tier is None or not tier.pressure_enabled
            or not callable(getattr(pipeline, "watch_section", None))):
        return None
    device = getattr(archive, "device", None)
    return pipeline.watch_section(reason, lambda need: tier.headroom_check(need, device=device))


def _record_rechecks(report: dict[str, Any], tier: Any, reason: str, events: Any) -> None:
    """The section's fitter re-check deferrals: in its report (``fitter_rechecks``, ``deferred_archival``,
    ``deferral_reason``, ``archival_deferred_seconds``) and with the raw tier's pressure events."""
    if not events:
        return
    copies = [{key: value for key, value in event.items() if not key.startswith("_")} for event in list(events)]
    waited = sum(float(event.get("waited_seconds") or 0.0) for event in copies)
    report["fitter_rechecks"] = copies
    report["deferred_archival"] = True
    report.setdefault("deferral_reason", "fitter_recheck")
    report["archival_deferred_seconds"] = float(report.get("archival_deferred_seconds") or 0.0) + waited
    try:
        tier.record_recheck_deferrals(reason, copies)
    except Exception:  # noqa: BLE001 - recording only
        pass

def _guard_make_room(archive, reason: str, tier, device, memory, target: int, state, record, own_fragments: int = 0,
                     strict: bool = False):
    """The decode guard short of ``target`` (dev22j7, dev22j8): the caching allocator's unoccupied blocks back to the
    driver -- only when the fragmented cached bytes (inactive split; wholly free segments are already in the headroom
    and the allocator frees them on demand) could cover the shortfall, not counting ``own_fragments`` (a running
    node's own free blocks: releasing them only moves its remaining growth into the headroom) -- then, still short of
    the headroom, the pressure response for the target. Returns the memory measured after and the relief event (None
    when no relief ran). A function, not a method: archive stand-ins bind only ``_decode_entry_guard`` and
    ``_release_cached_blocks``. dev22j8v4, ``strict`` (a node on the GPU; the caller is short on the strict headroom):
    no cache release -- it would only turn releasable bytes, which the strict headroom already counts, into free ones --
    and the pressure response runs."""
    from ...raw_tier import device_headroom

    if strict:
        state["strict_release_skips"] += 1
        record["shortfall"] = target - int(record.get("headroom", 0))
        event = archive.relieve_pressure(f"{reason}:decode_guard", needed_bytes=target)
        state["reliefs"] += 1
        record["actions"].append({"action": "relief", "event": event})
        memory = tier._memory(device) or memory
        return memory, event
    shortfall = target - device_headroom(memory)
    fragments = max(0, int(memory.get("inactive_split", 0) or 0) - int(own_fragments))
    record.update(shortfall=shortfall, fragments=fragments)
    if fragments >= shortfall:
        started = time.perf_counter()
        released = archive._release_cached_blocks()
        if released is not None:  # cached blocks of any stream's pool back to the driver
            state["cache_releases"] += 1
            record["actions"].append({"action": "release_cached", "bytes": released,
                                      "seconds": time.perf_counter() - started})
            memory = tier._memory(device) or memory
    else:
        state["cache_release_skips"] += 1
    event = None
    if device_headroom(memory) < target:  # still short: GPU raw copies to host, registered trims
        event = archive.relieve_pressure(f"{reason}:decode_guard", needed_bytes=target)
        state["reliefs"] += 1
        record["actions"].append({"action": "relief", "event": event})
        memory = tier._memory(device) or memory
    return memory, event


def _fitter_memory(pipeline, segments: list | None = None) -> dict[str, int] | None:
    """``ArchivePipeline.fitter_memory()`` (dev22j8: what the fitter's own streams hold now), None for pipelines
    without it (stand-ins, older cores) or when it cannot be measured (off CUDA). ``segments`` (dev22j8v4): the
    guard's snapshot, reused (pipelines that take none measure their own)."""
    measure = getattr(pipeline, "fitter_memory", None)
    if not callable(measure):
        return None
    try:
        if segments is not None:
            try:
                return measure(segments=segments)
            except TypeError:  # a stand-in or an older core: its own measurement
                pass
        return measure()
    except Exception:  # noqa: BLE001 - measurement only: the bound counts
        return None


def _strict_measure(tier, device, memory, record, state, suffix: str = ""):
    """dev22j8v4: one allocator snapshot -> the strict headroom (:func:`reelcache.raw_tier.strict_device_headroom`),
    recorded as ``strict_*{suffix}`` with ``snapshot_seconds``. Returns ``(strict, segments)``; ``(None, None)`` when
    the tier has no snapshot (CPU stand-ins: the formula decides). A failed snapshot counts nothing releasable."""
    from ...raw_tier import strict_device_headroom

    segments_of = getattr(tier, "_segments", None)
    if not callable(segments_of):
        return None, None
    collections = sum(item.get("collections", 0) for item in gc.get_stats())
    started = time.perf_counter()
    try:
        segments = segments_of(device)
        error = None
    except Exception as exc:  # noqa: BLE001 - conservative: the driver's free memory only
        segments, error = [], f"{type(exc).__name__}: {exc}"[:200]
    seconds = time.perf_counter() - started
    if segments is None:
        return None, None
    # dev22j8v5 (F5.4 finding 2): garbage collections that ran meanwhile (any thread's) and the snapshots taken
    gcs = sum(item.get("collections", 0) for item in gc.get_stats()) - collections
    record["snapshot_gc"] = int(record.get("snapshot_gc") or 0) + gcs
    record["snapshot_count"] = int(record.get("snapshot_count") or 0) + 1
    state["snapshot_gcs"] += gcs
    strict = strict_device_headroom(memory, segments)
    record.update({f"strict_headroom{suffix}": strict["headroom"], f"strict_free{suffix}": strict["free"],
                   f"strict_releasable{suffix}": strict["releasable"],
                   f"strict_unreleasable{suffix}": strict["unreleasable"],
                   f"strict_fragments{suffix}": strict["fragments"], f"strict_segments{suffix}": strict["segments"]})
    if error is not None:
        record[f"strict_error{suffix}"] = error
    record["snapshot_seconds"] = float(record.get("snapshot_seconds") or 0.0) + seconds
    state["snapshot_seconds"] += seconds
    state["snapshot_seconds_max"] = max(state["snapshot_seconds_max"], seconds)
    return strict, segments


def _decode_floor(archive, tier) -> tuple[int | None, str]:
    """dev22j8v5 (F5.4 finding 1): the least a decode next to a node on the GPU needs -- ``(bytes, source)``:
    ``decode_working_set_floor_bytes`` when configured (``config``); else ``DECODE_FLOOR_MARGIN`` x the profile's clean
    decode working set (:data:`DECODE_WORKING_SETS`) while the profile and archival's geometry are its default one
    (``profile``; never scaled to another geometry); else that margin x the largest clean working set this process
    measured (``learned``); else ``(None, "unlearned-wait")``: the decode waits for the node (F5.1)."""
    configured = getattr(getattr(archive, "config", None), "decode_working_set_floor_bytes", None)
    if configured is not None:
        return int(configured), "config"
    profile = getattr(archive, "_decode_profile", None)
    known = DECODE_WORKING_SETS.get(profile[0]) if profile else None
    grid = getattr(archive, "_capture_grid", None)
    if (known is not None and profile[1] is not None and tuple(profile[1]) == tuple(known[1])
            and grid is not None and tuple(grid) == tuple(known[1])):
        return int(DECODE_FLOOR_MARGIN * known[0]), "profile"
    learned = getattr(tier, "learned_clean_working_set_bytes", None)
    learned = int(learned()) if callable(learned) else 0
    if learned > 0:
        return int(DECODE_FLOOR_MARGIN * learned), "learned"
    return None, "unlearned-wait"


def _node_on_gpu(archive) -> bool:
    """Whether a node (or the warm-up) is on the GPU now (dev22j8v5: a section with one running unpaused does not
    measure the decode's clean working set). Unknown counts as on the GPU."""
    if getattr(archive, "fitter", None) is None:
        return False
    pipeline = getattr(archive, "pipeline", None)
    try:
        bound_of = getattr(pipeline, "inflight_growth_bound", None)
        if callable(bound_of):
            return bool(bound_of()["inflight"])
        inflight = getattr(pipeline, "inflight_gpu_bytes", None)
        return bool(callable(inflight) and int(inflight()) > 0)
    except Exception:  # noqa: BLE001 - unknown: not clean
        return True


def _open_guard_pause(pipeline, record, state, base: int, growth: int):
    """Pause the node at its next step boundary (or where it waits without allocating): a paused fitter allocates
    nothing more, its growth counts 0; not acknowledged, its (measured) remaining growth still counts. Returns
    ``(pause, acked, growth)``."""
    pause = _GuardPause(pipeline)
    info = pause.open()
    record["pause"] = dict(info)
    state["pauses"] += 1
    acked = bool(info["acked"])
    state["pause_acks" if acked else "pause_ack_timeouts"] += 1
    if not acked:  # dev22j8v6 (F5.5 gate): the step that held the fitter, in the per-decode record itself
        record.update(ack_timeout_phase=info.get("phase"), ack_timeout_phase_seconds=info.get("phase_seconds"),
                      ack_timeout_boundary_seconds=info.get("boundary_seconds"))
    if acked and info.get("landed") not in (None, "idle", "boundary"):
        state["pause_landed"] += 1
    if acked:
        growth = 0
        record.update(growth=0, target=base)
    return pause, acked, growth


class _GuardPause:
    """The decode guard's pause of the fitter for one section (dev22j7): ``ArchivePipeline.pause_gpu_work`` entered
    by :meth:`LineageSessionArchive._decode_entry_guard` and closed by :meth:`LineageSessionArchive.memory_critical`
    when the section exits (or by the guard when the section must wait for the node instead)."""

    def __init__(self, pipeline):
        self._context = pipeline.pause_gpu_work()
        self._open = False

    def open(self) -> dict[str, Any]:
        info = self._context.__enter__()
        self._open = True
        return info

    def close(self) -> None:
        if self._open:
            self._open = False
            self._context.__exit__(None, None, None)


class LineageSessionArchive:
    """Step 6 lineages + asynchronous archival + packed device restore for one bridge."""

    name = "lineage"
    asynchronous = True
    #: W1.4.3 (E1): the archive budget's eviction (:class:`.archive_eviction.ArchiveEviction`; None: no budget)
    eviction = None
    #: Freeze F5 (Step 11 cut 4; Step 5c's O4): the RAW install reads GPU raw copies in place (``DenseInstall``'s fast
    #: path: no staging copy, no residency staging), so an all-GPU-raw restore allocates nothing but the engine RoPE's
    #: transient outputs (:meth:`restore_allocates`).
    raw_install_in_place = True

    def __init__(self, root: str, model: ModelIdentity, *, config: Any, targets: EnergyTargets, device: Any,
                 matmul_precision: str | None = None, max_staged_canvases: int = 2):
        os.makedirs(root, exist_ok=True)
        self.config, self.model, self.root = config, model, root
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        # Freeze D (D-D): the encoder is chosen once for the process lifetime and recorded (R8).
        self.encoder, self.encoder_record = select_encoder(config, self.device)
        # F4: x265 threads per encode, derived from this process's CPUs unless configured (recorded).
        x265_threads = (config.resolved_x265_threads() if hasattr(config, "resolved_x265_threads")
                        else config.x265_threads)
        self.x265_threads = x265_threads
        encode_workers = (config.resolved_encode_workers() if hasattr(config, "resolved_encode_workers")
                          else config.encode_workers)
        self.encode_workers = encode_workers
        if getattr(config, "max_staged_canvases", None) is None and max_staged_canvases < encode_workers:
            # F4: automatic staging holds one canvas per parallel encode (each encode reads its staged canvas).
            max_staged_canvases = encode_workers
        # cpus_available: the affinity-mask size read here, at bridge attach (after SGLang's distributed init).
        # dev22j8v6 (F6 a): with archival_cpus="job" the derived counts split the job's CPU set, which the
        # pipeline's fitter and encoder threads get as their mask (``stats()["pipeline"]["cpu_sets"]``).
        self.archival_cpus = getattr(config, "archival_cpus", "inherit")
        job_cpus, job_source = sglang_config.job_cpu_set()
        attach_cpus = sglang_config._affinity()
        self.archival_cpu_set = job_cpus if self.archival_cpus == "job" else None
        self.encoder_record = {**self.encoder_record, "x265_threads": x265_threads, "encode_workers": encode_workers,
                               "x265_threads_configured": getattr(config, "x265_threads", None),
                               "encode_workers_configured": getattr(config, "encode_workers", None),
                               "cpus_available": sglang_config.available_cpus(),
                               "attach_cpus": sglang_config.format_cpu_list(attach_cpus) if attach_cpus else None,
                               "archival_cpus": self.archival_cpus, "job_cpus": sglang_config.format_cpu_list(job_cpus),
                               "job_cpus_count": len(job_cpus), "job_cpus_source": job_source}
        self.provider = HEVCProvider(x265_threads=x265_threads, packing="node")
        self.runtime = KVRuntime(root, model, encoder=self.provider.encode_tensor, decoder=self.provider.decode_tensor,
                                 max_store_bytes=config.max_store_bytes, max_resident_bytes=config.max_resident_bytes,
                                 verify_reads=config.verify_reads, verify_references=False,
                                 audit_on_open=config.audit_on_open, on_open_error=config.on_open_error)
        self.store = self.runtime.store
        if getattr(config, "archive_budget_bytes", None) is None and self.store.evicted_ids():
            # W1.4.3 (E1): an ablation arm's archive evicted payloads of this store; without a budget nothing knows
            # them (their reads would fail): refuse the reopen, actionably
            evicted = len(self.store.evicted_ids())
            self.runtime.close()
            raise ConfigurationError(
                f"the store at {root} holds {evicted} payload objects an archive budget evicted (an ablation arm's "
                "archive): reopen it with archive_budget_bytes set -- any positive budget; nodes whose payload was "
                "evicted are restored by replay")
        precision = {} if matmul_precision is None else {"matmul_precision": matmul_precision}
        codec = (config.codec_template() if self.encoder is None
                 else self.encoder.spec(1, pix_fmt=config.hevc_pix_fmt, normalization=config.normalization))
        try:
            self.method = RankMethodConfig(targets, codec, reuse=config.reuse, refit=config.refit,
                                           refit_rule=getattr(config, "refit_rule", "paper"),
                                           rate_lag=config.rate_lag, **precision, **fixed_rank_fields(config, model))
        except TypeError as exc:
            raise ConfigurationError(f"matmul_precision={matmul_precision!r} needs a ReelCache core whose "
                                     "RankMethodConfig has a matmul precision policy") from exc
        self.registry = RankPredictorRegistry(config.predictor_registry_id, anchor_rank=self.method.anchor_rank)
        bound = {} if config.plan_memory_bytes is None else {"plan_memory_bytes": config.plan_memory_bytes}
        try:
            self.options = ArchiveOptions(packing="node",
                                          canvas_ladder=CANVAS_RANK_LADDER if config.canvas == "canonical" else None,
                                          encode_threads=x265_threads, validate_decode=config.validate_decode,
                                          measure_post_codec=config.verify_post_codec,
                                          orthogonalizer=config.orthogonalizer, **bound)
        except TypeError as exc:
            raise ConfigurationError("plan_memory_bytes needs a ReelCache core whose ArchiveOptions bounds the "
                                     "planner's working set") from exc
        self.pool = self.fitter = self.staging_engine = None
        self.eigh_solver = None
        uuid = None
        if self.device.type == "cuda":
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            self.device = torch.device("cuda", index)
            uuid = "GPU-" + str(torch.cuda.get_device_properties(index).uuid).removeprefix("GPU-")
        fit_device = torch.device("cpu")
        if config.fit_device == "cuda":
            if self.device.type != "cuda":
                raise ConfigurationError("fit_device='cuda' needs the bridge on a CUDA device")
            from ...gpu_fitting import GPUFitter

            self._fitter_options = {"gpu_id": self.device.index, "gpu_uuid": uuid, "method": "gram_batched",
                                    "eigh_solver": "auto"}
            self.fitter = GPUFitter(**self._fitter_options)
            fit_device = self.device
        if config.representation == "svd":
            # wl-base (W1c): FP16 factor packs (no HEVC, no NVDEC): the P.Q GEMM restore on the bridge device, with the
            # LineageRestorer's prepared-pack and device basis caches (decoder settings do not apply). W1.2: on CUDA
            # its cold histories are staged by the same engine as the compressed arms' (CONTRACT §10 reader check)
            from ...factor_packs import FactorLineageRestorer

            if config.staging_engine and self.device.type == "cuda":
                from ...restore_staging import StagingEngine

                threads, prepare_threads = staging_threads(config)
                self.staging_engine = StagingEngine(self.device, threads=threads, prepare_threads=prepare_threads,
                                                    pinned_bytes=config.staging_pinned_bytes)
            self.restorer = FactorLineageRestorer(self.runtime, self.device, arena_bytes=config.restore_arena_bytes,
                                                  basis_budget_bytes=config.basis_budget_bytes,
                                                  prepared_budget_bytes=config.prepared_budget_bytes,
                                                  staging_engine=self.staging_engine)
        elif config.decoder == "nvdec_packed":
            if self.device.type != "cuda":
                raise ConfigurationError("decoder='nvdec_packed' restores on a CUDA device; use decoder='software'")
            from ...archive_restore import LineageRestorer
            from ...decoder_pool import DecoderPool

            self.pool = DecoderPool(gpu_id=self.device.index, gpu_uuid=uuid, workers=config.nvdec_workers,
                                    max_sessions_per_worker=16)
            if config.staging_engine:
                # One cold-session staging engine per worker (CR-17-4; Step 5c's recommended configuration), shared
                # by every restorer of the worker; the restorers do not own it (closed after them).
                from ...restore_staging import StagingEngine

                threads, prepare_threads = staging_threads(config)
                self.staging_engine = StagingEngine(self.device, threads=threads, prepare_threads=prepare_threads,
                                                    pinned_bytes=config.staging_pinned_bytes,
                                                    verify_bases=config.verify_bases)
            tier = {}
            if getattr(config, "decoded_tier_bytes", 0):  # accel A1: normative decoded pictures routed past NVDEC
                from ...decoded_tier import DecodedTier

                tier = {"decoded_tier": DecodedTier(config.decoded_tier_bytes, device=self.device,
                                                    route=config.decoded_tier_route, model=model,
                                                    layer_ms=config.decoded_tier_layer_ms)}
            self.restorer = LineageRestorer(self.runtime, self.pool, arena_bytes=config.restore_arena_bytes,
                                            basis_budget_bytes=config.basis_budget_bytes,
                                            prepared_budget_bytes=config.prepared_budget_bytes,
                                            staging_engine=self.staging_engine,
                                            pipelined_staging=config.pipelined_restore, **tier)
        else:
            from ...archive_restore import SoftwareLineageRestorer

            self.restorer = SoftwareLineageRestorer(self.runtime)
        seal = self.restorer if (config.seal_prepare and (self.pool is not None or config.representation == "svd")) \
            else None
        # Archival yields to restores in flight (Step 7): the pipeline yields to the seal-time restorer's activity
        # by default; without seal-time preparation it is given the bridge's restorer explicitly, so restores
        # always get the yield. Older cores (no yield_to) keep their behavior.
        yields = ({"yield_to": self.restorer}
                  if "yield_to" in inspect.signature(ArchivePipeline.__init__).parameters else {})
        # Archival FFmpeg (Step 19 follow-up): the pipe pump pauses while restores run (default on); CPU priority
        # is opt-in. Older cores keep their behavior unless an option needs the newer pipeline.
        parameters = inspect.signature(ArchivePipeline.__init__).parameters
        if "pause_ffmpeg" in parameters:
            yields["pause_ffmpeg"] = bool(getattr(config, "archival_pump_pause", True))
        if getattr(config, "archival_cpu_priority", None) is not None:
            if "cpu_priority" not in parameters:
                raise ConfigurationError("archival_cpu_priority needs a ReelCache core whose ArchivePipeline has "
                                         "cpu_priority")
            yields["cpu_priority"] = config.archival_cpu_priority
        self.raw_tier = None
        if getattr(config, "raw_tier", False) and config.representation in ("svd_hevc", "svd", "hevc"):  # + wl-base
            # Host pools that share the headroom (R5): the staging engine's pinned ring, the prepared-pack cache and
            # the pipeline's canvas staging (bounded per canvas here; recorded).
            reserved = {"staging_pinned_bytes": (config.staging_pinned_bytes
                                                 if self.staging_engine is not None else 0),
                        "prepared_budget_bytes": (config.prepared_budget_bytes
                                                  if self.pool is not None or config.representation == "svd" else 0),
                        "canvas_staging_bytes": max_staged_canvases * CANVAS_RESERVE_BYTES,
                        # accel A1: the decoded tier's pinned pictures share the headroom too (only when configured)
                        **({"decoded_tier_bytes": config.decoded_tier_bytes}
                           if getattr(self.restorer, "decoded_tier", None) is not None else {})}
            self.raw_tier = RawChunkTier(
                gpu_budget_bytes=config.gpu_raw_budget_bytes, gpu_budget_chunks=config.max_source_chunks,
                gpu_reserve_bytes=getattr(config, "gpu_raw_reserve_bytes", None),
                host_budget_bytes=config.host_raw_budget_bytes, host_reserved_bytes=sum(reserved.values()),
                workers=config.raw_tier_workers, pinned=config.host_raw_pinned,
                max_lag_seconds=config.max_archival_lag_seconds,
                on_host_drop=self._prepare_on_drop if callable(getattr(self.restorer, "ensure_prepared", None))
                else None,
                **({"retain": config.raw_tier_retain} if getattr(config, "raw_tier_retain", "drop") != "drop"
                   else {}),
                # W1.4.3 (E1): an evicting ablation arm promotes its replayed chunks (config: archive budget only)
                **({"promote": config.raw_tier_promote} if getattr(config, "raw_tier_promote", "off") != "off"
                   else {}))
            self.raw_tier.host_derivation["reserved"] = reserved
            yields["raw_tier"] = self.raw_tier
        self.eviction = None
        if getattr(config, "archive_budget_bytes", None) is not None:  # W1.4.3 (E1): reuse=False arms only (config)
            from .archive_eviction import ArchiveEviction

            self.eviction = ArchiveEviction(self.runtime.store, config.archive_budget_bytes,
                                            on_evict=self._payload_evicted)
            yields["archive_budget"] = self.eviction
        from ...accel_b import FitterAccel
        fitter_accel = FitterAccel.from_config(config)  # accel B (2026-10-30, opt-in): None at the defaults
        if fitter_accel is not None:
            yields["fitter_accel"] = fitter_accel
        from ...accel_c import AccelOptions
        accel = AccelOptions.from_config(config)  # accel C (opt-in; inactive: the base pipeline, no arguments)
        if accel.active:
            yields.update(package_accel=accel, validation_pool=self.pool)
        self.pipeline = ArchivePipeline(
            self.runtime, method=self.method, registry=self.registry, options=self.options, device=fit_device,
            fitter=self.fitter, encode_workers=encode_workers, max_source_chunks=config.max_source_chunks,
            max_staged_canvases=max_staged_canvases, max_unpublished=config.max_unpublished,
            submit_timeout=config.submit_timeout, restorer=seal, stream_priority=config.stream_priority,
            encoder=self.encoder, cpu_set=self.archival_cpu_set,
            fitter_stream=getattr(config, "fitter_stream", "dedicated"), **yields)
        if config.defer_archival_during_decode and not callable(getattr(self.pipeline, "defer_gpu_work", None)):
            raise ConfigurationError("defer_archival_during_decode needs a ReelCache core whose ArchivePipeline has "
                                     "defer_gpu_work (Step 7 dev16 or later)")
        self.max_staged_canvases = max_staged_canvases
        self.fit_device = fit_device
        self._records: dict[str, BridgeNode] = {}
        self._states: dict[str, dict[str, torch.Tensor]] = {}
        self.stats_ = {"restores": 0, "prewarms": 0, "prewarm_seconds": 0.0, "drains": 0, "drain_seconds": 0.0}
        # dev22j6: the decode entry guard (memory_critical): its switch and counters.
        self.decode_guard_ = _decode_guard_state(bool(getattr(config, "decode_entry_guard", True)))
        # dev22j8v5: the decode floor's inputs -- the profile (name, default token grid) recorded at the attach
        # warm-up and the token grid of the last captured chunk (the geometry archival is working on)
        self._decode_profile: tuple[str | None, tuple[int, int] | None] | None = None
        self._capture_grid: tuple[int, int] | None = None
        if self.eviction is not None:  # W1.4.3 (E1): a reopened archive's published nodes (evicted ones as such)
            self.eviction.adopt(self._archived_nodes())

    # -- W1.4.3 (E1): archive-budget eviction (ablation arms) -------------------------------------------------------
    def _archived_nodes(self) -> list[str]:
        """Every node manifest the store's refs reach (lineage heads, tree refs, checkpoints), parents before children:
        by depth (the root first), then by ref -- the LRU order a reopened archive starts from (the order of the run
        that wrote them is not persisted; runs do not reopen mid-run)."""
        store, parents = self.store, {}
        pending = sorted(set(store.list_refs("").values()))
        while pending:
            ref = pending.pop()
            if ref in parents or not ref.startswith("manifest:"):
                continue
            manifest = store.get_manifest(ref)
            if manifest.get("type") != "node":
                continue
            parents[ref] = manifest["node"].get("parent")
            if parents[ref]:
                pending.append(parents[ref])
        depth: dict[str, int] = {}
        for ref in parents:
            chain = []
            while ref in parents and ref not in depth:
                chain.append(ref)
                ref = parents[ref]
            known = depth.get(ref, -1)
            for item in reversed(chain):
                known += 1
                depth[item] = known
        return sorted(parents, key=lambda ref: (depth[ref], ref))

    def _payload_evicted(self, node_ref: str, packets: tuple, versions: tuple) -> None:
        """An evicted node's prepared packs and device-resident basis versions go with its payload."""
        prepared = getattr(self.restorer, "prepared", None)
        if prepared is not None and packets:
            prepared.forget(packets)
        bases = getattr(self.restorer, "bases", None)
        if bases is not None and versions:
            bases.forget(versions)
        tier = getattr(self.restorer, "decoded_tier", None)
        if tier is not None and packets:  # accel A1: an evicted node's decoded pictures go with its payload
            tier.forget_packets(packets)
        self._records.pop(node_ref, None)

    def hold_available(self, node_refs: Any) -> tuple[set, Any]:
        """The nodes of ``node_refs`` a restore can read now -- a payload the archive still holds (prepared or cold),
        or a usable raw-tier copy (GPU or host) of an evicted one -- and a hold that keeps them so until
        :meth:`release_available`: the archived nodes pinned against eviction (no LRU touch), the evicted ones' raw
        copies pinned in the raw tier (no drop, no demotion). A restore's choice between its archive and a replay
        stays true until the restore pinned what it reads."""
        refs = tuple(dict.fromkeys(node_refs))
        live = self.eviction.pin_live(refs)
        evicted = set(refs) - set(live)  # evicted when pinned (nothing pinned is evicted after)
        raw = []
        if evicted and self.raw_tier is not None and self.config.raw_restore == "prefer":
            raw = self.raw_tier.pin_copies(evicted)
        return set(live) | {entry.node_ref for entry in raw}, (live, raw)

    def release_available(self, hold: Any) -> None:
        if hold is None:
            return
        live, raw = hold
        self.eviction.unpin(live)
        if raw:
            self.raw_tier.unpin_copies(raw)

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "method": self.method.to_dict(), "options": self.options.to_dict(),
                "fit_device": "cpu" if self.fitter is None else str(self.device),
                "eigh_solver": None if self.fitter is None else getattr(self.fitter, "eigh_label", None),
                "restorer": type(self.restorer).__name__, "seal_prepare": self.pipeline.restorer is not None,
                "nvdec_workers": None if self.pool is None else self.config.nvdec_workers,
                "staging_engine": self.staging_description(), "quarantined": self.quarantined(),
                "encoder": dict(self.encoder_record), "raw_tier": self.raw_tier_description()}

    def staging_description(self) -> dict[str, Any]:
        """Whether cold histories are staged (CR-17-4), with the engine's device, threads and pinned ring bytes."""
        engine = self.staging_engine
        if engine is None:
            reason = ("staging_engine=False: cold restores use the store path" if self.pool is not None
                      or (getattr(self.config, "representation", None) == "svd" and self.device.type == "cuda")
                      else "svd factor packs off CUDA: cold restores use the store path"  # wl-base
                      if getattr(self.config, "representation", None) == "svd"
                      else "software decoder: no packed device restore")
            return {"enabled": False, "reason": reason}
        return {"enabled": True, "device": str(engine.device), "threads": engine.threads,
                "prepare_threads": engine.prepare_threads, "pinned_bytes": engine.ring.capacity}

    # -- sessions ---------------------------------------------------------------------------
    def head(self, session_id: str) -> str | None:
        return self.runtime.list_lineages().get(session_id)

    def sessions(self) -> dict[str, str]:
        """Session heads, without the quarantined ones (:meth:`quarantined`; their ``head`` stays, so they can be
        deleted or replaced, and using them raises a typed integrity error). W1.3: never a detached branch's lineage."""
        quarantined = self.quarantined()
        return {name: ref for name, ref in self.runtime.list_lineages().items()
                if name not in quarantined and not is_detached_branch(name)}

    def quarantined(self) -> dict[str, str]:
        """``{session_id: error}`` of sessions whose head was found damaged when the store was opened."""
        return {name[len(LINEAGE_PREFIX):]: error for name, error in self.runtime.quarantined.items()
                if name.startswith(LINEAGE_PREFIX) and not is_detached_branch(name)}

    def create_session(self, session_id: str, *, record: Mapping[str, Any], streams: Sequence[str],
                       replace: bool = False, telemetry: Mapping[str, Any] | None = None) -> str:
        if self.head(session_id) is not None:
            if not replace:
                raise StateError(f"causal KV session {session_id!r} already exists")
            self.runtime.delete_lineage(session_id)
        content, run = split_record(record)
        self.raw_release(session_id)
        root_ref = self.runtime.create_lineage(session_id, epoch_id="e0", details={SESSION_DETAILS_KEY: content},
                                               streams=tuple(sorted(streams)))
        self.annotate(root_ref, {**(run or {}), **dict(telemetry or {})})
        return root_ref

    def annotate(self, ref: str, telemetry: Mapping[str, Any] | None) -> None:
        """Attach bridge run telemetry (request ids, configuration) to a node's unhashed sidecar."""
        annotate(self.store, ref, telemetry)

    def fork(self, node_ref: str, session_id: str) -> None:
        self.runtime.fork_lineage(node_ref, session_id)
        self.raw_hold(session_id, node_ref)

    def rewind(self, session_id: str, node_ref: str) -> None:
        self.runtime.rewind_lineage(session_id, node_ref)
        self.raw_hold(session_id, node_ref)

    def delete(self, session_id: str) -> None:
        if self.head(session_id) is not None:
            self.runtime.delete_lineage(session_id)
        self.raw_release(session_id, drop=True)
        self.raw_forget(session_id)

    # -- raw tier: holders and controls (freeze D) ------------------------------------------------------------------
    def raw_hold(self, session_id: str, head_ref: str) -> dict[str, int] | None:
        """Atomically make ``session_id`` hold exactly the raw entries of ``head_ref``'s path (plus its unpublished
        commits): called by :meth:`fork` and :meth:`rewind`; the bridge calls it after any other head move. Entries
        no session holds are dropped once published (no memory ratchet)."""
        if self.raw_tier is None:
            return None
        refs = [node.ref for node in self.path(head_ref)[1:]]
        return self.raw_tier.hold(session_id, refs, exclusive=True)

    def raw_release(self, session_id: str, *, drop: bool = False) -> dict[str, int] | None:
        """Detach ``session_id`` from every raw entry (session close, delete, replace; called by :meth:`delete` and
        :meth:`create_session`). The bridge calls it when a session closes. ``drop`` (whole-tree delete, wl-w1b7):
        the entries it leaves without holders drop whatever ``raw_tier_retain``."""
        if self.raw_tier is None:
            return None
        return self.raw_tier.release_holder(session_id, drop=True if drop else None)

    release_session = raw_release

    def raw_forget(self, session_id: str) -> int:
        """wl-w1b7 (whole-tree delete): drop the unheld raw entries the session's holders held last."""
        return 0 if self.raw_tier is None else self.raw_tier.drop_unheld(session_id)

    def _raw_entries(self, session_id: str | None, *, published: bool = False) -> list:
        if self.raw_tier is None:
            raise StateError("the raw tier is off (raw_tier=False)")
        if session_id is None:
            entries = self.raw_tier.entries()
        else:
            head = self.head(session_id)
            refs = [] if head is None else [node.ref for node in self.path(head)[1:]]
            entries = self.raw_tier.entries(node_refs=refs) + [
                entry for entry in self.raw_tier.entries(holder=session_id) if entry.node_ref is None]
        return [entry for entry in entries if entry.node_ref is not None] if published else entries

    def demote_raw(self, session_id: str | None = None, *, wait: bool = True,
                   timeout: float | None = None) -> dict[str, Any]:
        """Move the session's (or every) GPU raw copy to host raw; waits for the copies. Step 19 control."""
        return self.raw_tier.demote(self._raw_entries(session_id), wait=wait, timeout=timeout)

    def drop_raw(self, session_id: str | None = None, *, drain: bool = True,
                 timeout: float | None = None) -> dict[str, Any]:
        """Drop both raw tiers of the session's (or every) published entries, leaving the compressed archive only;
        ``drain`` first publishes the session's pending commits (raw copies of unpublished commits never drop)."""
        if self.raw_tier is None:
            raise StateError("the raw tier is off (raw_tier=False)")
        if drain:
            sessions = [session_id] if session_id is not None else list(self.pipeline._lineages)
            for name in sessions:
                self.drain(name, timeout=timeout)
        return self.raw_tier.drop(self._raw_entries(session_id, published=True))

    def place_raw(self, session_id: str, placement: Mapping[Any, str], *, wait: bool = True,
                  timeout: float | None = None) -> dict[str, Any]:
        """Explicit, deterministic placement of the session's published chunks: ``{chunk_index or node_ref:
        "gpu_raw" | "host_raw" | "compressed"}`` (promotion to GPU happens only here, within the GPU budget)."""
        if self.raw_tier is None:
            raise StateError("the raw tier is off (raw_tier=False)")
        head = self.head(session_id)
        if head is None:
            raise StateError(f"unknown session {session_id!r}")
        nodes = self.path(head)[1:]
        by_ref = {entry.node_ref: entry for entry in self.raw_tier.entries(node_refs=[node.ref for node in nodes])}
        groups: dict[str, list] = {"gpu_raw": [], "host_raw": [], "compressed": []}
        missing = []
        for key, tier in placement.items():
            if tier not in groups:
                raise ConfigurationError(f"placement tiers are {tuple(groups)}, got {tier!r}")
            refs = ([node.ref for node in nodes if node.kind == "chunk" and node.chunk_index == key]
                    if isinstance(key, int) else [key])
            entries = [by_ref[ref] for ref in refs if ref in by_ref]
            if not entries and tier != "compressed":
                missing.append(key)
            groups[tier].extend(entries)
        if missing:
            raise StateError(f"no raw copy left for {missing}: placement cannot recreate dropped raw data")
        report = {"gpu_raw": self.raw_tier.promote(groups["gpu_raw"], timeout=timeout) if groups["gpu_raw"] else None,
                  "host_raw": (self.raw_tier.demote(groups["host_raw"], wait=wait, timeout=timeout)
                               if groups["host_raw"] else None),
                  "compressed": self.raw_tier.drop(groups["compressed"]) if groups["compressed"] else None}
        return report

    def reset_raw_peaks(self) -> None:
        if self.raw_tier is not None:
            self.raw_tier.reset_peaks()

    def shrink_raw_buffers(self, keep_bytes: int = 0) -> dict[str, Any]:
        """Explicit idle shrink of the raw tier's pinned host-block pool (D2): unpins and frees cached blocks beyond
        ``keep_bytes``. It holds the CUDA driver lock (~1.5 s per GB): call it between requests. Recorded in
        ``raw_tier.host_buffers.shrink_events``."""
        if self.raw_tier is None:
            raise StateError("the raw tier is off (raw_tier=False)")
        return self.raw_tier.shrink_buffers(keep_bytes)

    def raw_tier_description(self) -> dict[str, Any]:
        if self.raw_tier is None:
            return {"enabled": False, "reason": ("raw_tier=False (C3 source leases and backpressure)"
                                                 if self.config.representation in ("svd_hevc", "svd", "hevc")
                                                 else "raw representation")}
        stats = self.raw_tier.stats()
        return {"enabled": stats["enabled"], "disabled_reason": stats["disabled_reason"], "budgets": stats["budgets"],
                "gpu_raw_budget_bytes": stats["gpu_raw_budget_bytes"],
                "host_raw_budget_bytes": stats["host_raw_budget_bytes"],
                "raw_restore": self.config.raw_restore, "pinned": self.config.host_raw_pinned}

    def commit_encoder(self, ticket: Any) -> str | None:
        """The encoder of a published commit (``nvenc`` or ``libx265``; None while unpublished)."""
        return getattr(ticket, "encoder", None)

    def _node_packets(self, node_ref: str) -> list[str]:
        with self.runtime._lock, self.store.transaction():
            node = self.runtime.load_node(node_ref)
            if node.node_kind == "recache":
                epoch = self.runtime.load_epoch(node.epoch_ref)
                return [item.packet_ref for item in epoch.ranges if item.packet_ref is not None]
            return [] if node.packet_ref is None else [node.packet_ref]

    def _prepare_on_drop(self, node_ref: str) -> None:
        """The raw tier dropped a published entry's last copy: keep its node pack prepared in host memory (R6)."""
        token = None
        if self.eviction is not None:  # W1.4.3 (E1): an evicted node has nothing left to prepare; else pinned
            token = self.eviction.pin_available((node_ref,), touch=False)
            if token is None:
                return
        try:
            self.restorer.ensure_prepared(self._node_packets(node_ref))
        finally:
            if token is not None:
                self.eviction.unpin(token)

    # -- nodes -------------------------------------------------------------------------------
    def execution_state(self, node: BridgeNode | str) -> dict[str, torch.Tensor]:
        ref = node if isinstance(node, str) else node.ref
        state = self._states.get(ref)
        if state is None:
            chunk = self.runtime.load_node(ref)
            state = {} if chunk.execution_state_ref is None else deserialize_tensors(
                self.store.get(chunk.execution_state_ref))
            if len(self._states) > 64:
                self._states.clear()
            self._states[ref] = state
        return state

    def _bridge_record(self, ref: str, chunk: Any) -> dict[str, Any]:
        if chunk.execution_state_ref is None:
            raise IntegrityError(f"{ref}: bridge nodes carry their execution state")
        state = self.execution_state(ref)
        if RECORD_KEY not in state:
            raise IntegrityError(f"{ref}: execution state without a bridge record")
        return tensor_record(state[RECORD_KEY])

    def load(self, ref: str, path_entry: tuple | None = None, root_ref: str | None = None) -> BridgeNode:
        cached = self._records.get(ref)
        if cached is not None:
            return cached
        if path_entry is None:
            path_entry = self.runtime.node_path(ref)[-1]
        _, chunk, epoch = path_entry
        if chunk.node_kind == "root":
            details = dict(epoch.details)
            if SESSION_DETAILS_KEY not in details:
                raise IntegrityError(f"{ref} is a lineage root without an SGLang session record")
            node = root_node(ref, details[SESSION_DETAILS_KEY])
        elif chunk.node_kind == "chunk":
            record = self._bridge_record(ref, chunk)
            node = chunk_node(ref, chunk.parent_ref, root_ref or record.get("root_ref", ""), record)
            if (node.token_start, node.token_end) != (chunk.token_start, chunk.token_end):
                raise IntegrityError(f"{ref}: bridge record and lineage node disagree on the token range")
        elif chunk.node_kind == "recache":
            record = self._bridge_record(ref, chunk)
            ropes = {int(item["token_start"]): item for item in record["ranges"]}
            ranges = []
            for item in epoch.ranges:
                meta = ropes.get(int(item.token_start))
                if meta is None or int(meta["num_tokens"]) != item.token_end - item.token_start:
                    raise IntegrityError(f"{ref}: recache epoch range {item.token_start} has no bridge record")
                ranges.append(CoverageRange(item.token_start, item.token_end - item.token_start,
                                            int(meta["chunk_index"]), float(meta["rope_start_frame"])))
            node = recache_node(ref, chunk.parent_ref, root_ref or "", record, tuple(ranges), chunk.frontier)
        else:
            raise IntegrityError(f"{ref}: unsupported lineage node kind {chunk.node_kind!r} for the SGLang bridge")
        self._records[ref] = node
        return node

    def path(self, ref: str) -> list[BridgeNode]:
        entries = self.runtime.node_path(ref)
        root_ref = entries[0][0]
        nodes = [self.load(item[0], item, root_ref) for item in entries]
        nodes = [node if node.root_ref == root_ref else BridgeNode(**{**node.__dict__, "root_ref": root_ref})
                 for node in nodes]
        if nodes[0].kind != "root" or any(node.kind not in ("chunk", "recache") for node in nodes[1:]):
            raise IntegrityError("a bridge node path must be one root followed by chunk and recache nodes")
        return nodes

    def root_record(self, ref: str) -> Mapping[str, Any]:
        return self.path(ref)[0].record

    def chunk_latents(self, node: BridgeNode, device: Any = "cpu") -> torch.Tensor:
        return self.execution_state(node)[LATENTS_KEY].to(device)

    # -- commits -----------------------------------------------------------------------------
    def commit_chunk(self, session_id: str, *, node_id: str, parent_ref: str, root_ref: str,
                     matrices: Mapping[Any, torch.Tensor], contexts: Mapping[Any, Any], streams: Sequence[str],
                     record: Mapping[str, Any], execution_state: Mapping[str, torch.Tensor],
                     ready_event: Any = None, release=None, timeout: Any = ..., lossy_history: bool = False) -> Any:
        state = {**dict(execution_state), RECORD_KEY: record_tensor({**split_record(record)[0],
                                                                      "root_ref": root_ref})}
        for context in contexts.values():  # dev22j8v5: the geometry archival works on (the decode floor's check)
            grid = (getattr(context, "height", None), getattr(context, "width", None))
            if all(type(value) is int for value in grid):
                self._capture_grid = grid
            break
        with self._capture_exactness(session_id, node_id, lossy_history):
            return self.pipeline.submit(session_id, node_id, matrices, contexts, kind="chunk", execution_state=state,
                                        ready_event=ready_event, release=release, timeout=timeout)

    def _capture_exactness(self, session_id: str, node_id: str, lossy_history: bool) -> Any:
        """W1.4.3 (review C1): a commit captured on a history the session installed (in part) from the lossy archive
        (the bridge passes ``lossy_history`` on an evicting arm) adopts an inexact raw entry."""
        if lossy_history and self.raw_tier is not None:
            return self.raw_tier.inexact_capture(session_id, node_id)
        return contextlib.nullcontext()

    def commit_recache(self, session_id: str, *, node_id: str, parent_ref: str, root_ref: str,
                       ranges: Sequence[tuple[Mapping[str, Any], Mapping[Any, torch.Tensor], Mapping[Any, Any]]],
                       streams: Sequence[str], record: Mapping[str, Any],
                       execution_state: Mapping[str, torch.Tensor] | None, epoch_id: str,
                       details: Mapping[str, Any] | None = None, ready_event: Any = None, release=None,
                       timeout: Any = ..., lossy_history: bool = False) -> Any:
        if len(ranges) > self.max_staged_canvases:
            # The pipeline stages one host canvas per range before the node encodes: more ranges than canvases
            # would wait forever for a release only this node's own encode makes.
            raise CapacityError(f"a recache node of {len(ranges)} ranges needs max_staged_canvases >= {len(ranges)} "
                                f"(configured {self.max_staged_canvases})")
        groups = [(item["chunk_id"], matrices, contexts) for item, matrices, contexts in ranges]
        meta = [{key: item[key] for key in ("chunk_id", "chunk_index", "token_start", "num_tokens",
                                            "rope_start_frame")} for item, _, _ in ranges]
        state = {**dict(execution_state or {}),
                 RECORD_KEY: record_tensor({**split_record(record)[0], "ranges": meta, "root_ref": root_ref})}
        with self._capture_exactness(session_id, node_id, lossy_history):
            return self.pipeline.submit(session_id, node_id, {}, {}, kind="recache", ranges=groups,
                                        epoch_id=epoch_id, details=dict(details or {}), execution_state=state,
                                        ready_event=ready_event, release=release, timeout=timeout)

    def drain(self, session_id: str, *, through: str | None = None, timeout: float | None = None) -> str | None:
        """Wait until the session's submitted commits (or those through node ``through``) are published.

        Returns the published head; raises the session's archival failure (``ArchiveFailure``) or
        ``PendingError`` on timeout. The pipeline keeps the session's planning state (the next chunk
        continues the lineage without re-reading it)."""
        started = time.perf_counter()
        head = self.pipeline.drain(session_id, through=through, timeout=timeout)
        self.stats_["drains"] += 1
        self.stats_["drain_seconds"] += time.perf_counter() - started
        return head

    def release(self, session_id: str, timeout: float | None = None) -> str | None:
        """Drop the session's pipeline state once its unpublished commits retired; returns the published head.

        Acknowledges an archival failure (the failed commit and its descendants are retired; earlier commits
        still publish first). Needed before the head moves outside the pipeline (rewind, replace, delete)."""
        return self.pipeline.reset(session_id, timeout=timeout)

    def abandon(self, session_id: str) -> str | None:
        """Retire every unpublished commit of the session (discarded session); returns the published head."""
        return self.pipeline.cancel(session_id)

    def frontiers(self, session_id: str) -> dict[str, Any]:
        return self.pipeline.frontiers(session_id)

    def live(self, session_id: str) -> bool:
        """Whether the pipeline holds planning state for the session (submitted since the last release)."""
        return session_id in self.pipeline._lineages

    # -- W1.3: detached branches (``tree_switch="detach"``) ------------------------------------------------------------
    def detach(self, session_id: str, branch: str) -> dict[str, Any]:
        """Hand the session's unpublished commits to the detached lineage ``branch`` (their ref starts at their
        published head, the cut; they keep publishing in order there with their publication refs, never moving the
        session head). Strong exception guarantee (``ArchivePipeline.detach``). Returns ``{"branch", "moved",
        "cut"}`` (``moved`` []: nothing in flight)."""
        detach = getattr(self.pipeline, "detach", None)
        return {"branch": branch, "moved": [], "cut": None} if detach is None else detach(session_id, branch)

    def detached_head(self, branch: str) -> str | None:
        """A detached lineage's head (its last published node, or its cut), None once closed."""
        return self.runtime.list_lineages().get(branch)

    def abandon_detached(self, branch: str) -> dict[str, Any]:
        """Retire every unpublished commit of a detached lineage (idempotent; waits for work in progress, which never
        publishes). ``quiesced``: nothing of the branch is in flight."""
        if branch in self.pipeline._lineages:
            self.pipeline.cancel(branch)
        return {"branch": branch, "quiesced": branch not in self.pipeline._lineages}

    def close_detached(self, branch: str, timeout: float | None = 0, keep: Any = None) -> dict[str, Any]:
        """Once every commit of the detached branch retired (published or failed): drop its pipeline state and its
        publication bindings, root its last published node under ``keep(head)`` when that names a ref (F2: a node no
        tree ref covers; set before the lineage is deleted), delete its lineage and release its raw-tier holder.
        Idempotent (``closed``); ``PendingError`` while commits are in flight."""
        if branch in self.pipeline._lineages:
            self.pipeline.reset(branch, timeout=timeout)
        head = self.runtime.list_lineages().get(branch)
        kept = None
        if head is not None:
            kept = keep(head) if callable(keep) else keep
            if kept is not None:
                self.store.set_ref(kept, head)
            self.runtime.delete_lineage(branch)
        self.runtime.unbind_publication_refs(branch)
        raw = self.raw_tier.release_holder(branch) if self.raw_tier is not None else None
        return {"branch": branch, "head": head, "kept": kept, "raw_tier": raw, "closed": True}

    def stale_detached(self) -> dict[str, str]:
        """At open: ``{branch: head}`` of the detached lineages a previous process left (a detached branch never
        outlives its process); the controller closes each, keeping what no tree ref covers (F2)."""
        return {name: ref for name, ref in self.runtime.list_lineages().items() if is_detached_branch(name)}

    def backpressure(self) -> tuple[int, float]:
        """Pipeline-wide (submits that waited for a bound, seconds waited) so far."""
        stats = self.pipeline.stats_
        return int(stats["backpressure_waits"]), float(stats["backpressure_seconds"])

    def commit_report(self, ticket: Any) -> dict[str, Any]:
        """Published node, its commit report and per-matrix decisions (after the ticket resolved)."""
        ref = ticket.result(0)
        node = self.runtime.load_node(ref)
        commit = dict(self.runtime.node_commit_report(ref))
        rows, native, payload = [], 0, set()
        records = list(node.matrix_records)
        if node.node_kind == "recache":
            epoch = self.runtime.load_epoch(node.epoch_ref)
            records = [item for piece in epoch.ranges for item in piece.records]
        for entry, record_ref in records:
            record = self.runtime.load_record(record_ref)
            key, stream = entry_parts(entry)
            energy = record.energy
            actual = record.decision.get("actual_bytes", {})
            rows.append({
                "key": f"{key.layer}:{key.target}", "stream": stream, "decision": record.decision.get("decision"),
                "rank_view": record.rank_view, "target": energy.target, "observed_rank": energy.observed_rank,
                "bucket_rank": energy.bucket_rank, "ideal_subspace_energy": energy.ideal_subspace_energy,
                "stored_factor_energy": energy.stored_factor_energy,
                "post_codec_relative_error": energy.post_codec_relative_error,
                "incremental_bytes": actual.get("incremental_bytes"),
                "new_basis_block_bytes": actual.get("new_basis_block_bytes"),
                "inherited_view_rows": actual.get("inherited_view_rows"),
            })
            # Native bytes in the captured dtype (FP32 keys of RAVEN count 4 bytes per element).
            native += record.context.token_count * record.context.channels * torch.empty(
                (), dtype=getattr(torch, record.context.dtype)).element_size()
            payload.add(record.packet_ref)
        incremental = sum(row["incremental_bytes"] or 0 for row in rows)
        timeline = [item for item in self.pipeline.stats_["timeline"] if item.get("node_id") == ticket.node_id
                    and item.get("branch") == getattr(ticket, "publication_branch_id", getattr(ticket, "branch_id",
                                                                                                None))]
        return {
            "representation": getattr(self.config, "representation", "svd_hevc"), "node": ref,
            "kind": node.node_kind, "matrices": rows,
            "native_bytes": native, "incremental_bytes": incremental,
            "compression_ratio": native / incremental if incremental else None,
            "packets": len(payload), "decisions": commit.get("decisions"), "method": commit.get("method"),
            "options": commit.get("options"), "registry": commit.get("registry"),
            "stage_seconds": commit.get("stage_seconds"),
            "seconds_before_publication": commit.get("seconds_before_publication"),
            "timeline": timeline[-1] if timeline else None,
        }

    # -- process-level warm-up (backend attach) -------------------------------------------------
    def warmup(self, profile: Any, *, sessions: bool = False, archive: bool | None = None, synthetic: bool = False,
               layers: int = 4, seed: int = 0) -> dict[str, Any]:
        """Move the restore path's process-level cold start off the first restore (CR-12p2-11; backend attach).

        The default is process-level only (restore accounting rule, report §3.5: session-independent work that
        may be excluded from restore time). Components (each timed; a failure is reported, never raised):

        * ``restore_kernels``: ``restore_kernels.warmup`` compiles/loads every Triton variant the packed restore
          launches (``kernel_warmup_shapes``: cache dtype, fused or native RoPE, ranks 1 .. min(channels, chunk
          tokens)); in ``rope_mode="native"`` the engine's own RoPE kernel is not warmed (an identity stands in);
        * ``decoder_pool``: a decoder session for every canonical node canvas of the profile's tile geometry on
          every worker (``PackedRestorer.warmup``, Step 5c), when the core has it;
        * ``sessions`` (opt-in: session-specific CPU packing for every persisted session at attach; the attach
          warm-up itself stays process-level): the store's persisted sessions get their packs prepared, bases made
          resident and decoder sessions created (``LineageRestorer.prewarm_lineages``). A prepared session (this,
          ``open_session`` / reopen with ``config.prewarm``, or seal-time preparation) is the state the
          restore-vs-recompute requirement is judged in;
        * ``archive`` (``ArchivePipeline.prewarm``, nothing stored or learned): ``True`` (opt-in) runs it with the
          encoder warm-up (for LongLive 2.0 ~9 s and several GB of cached GPU memory); ``None`` (default, dev22j8)
          runs it without the encode (``encode=False``) when the decode entry guard is active (GPU fitter, raw tier
          with the pressure response, ``decode_entry_guard`` on, ``defer_archival_during_decode`` off): the fitter's
          peak working set it measures bounds a node's growth at the decode guard
          (``ArchivePipeline.inflight_growth_bound``; ``stats()["pipeline"]["fitter_pause"]``); ``False`` never;
        * ``synthetic_restore`` (opt-in diagnostic): one synthetic chunk of the profile's default geometry,
          ``layers`` layers deep (several spectral decays), archived into a scratch store by a private pipeline with
          its own fitter and a fresh predictor registry and restored three times (slot 0, one frame's offset, slot
          0 again) by a restorer on the shared decoder pool; the scratch store is deleted;
        * ``release`` (after the opt-in components only): the CUDA caching allocator's unused blocks are returned
          (``release_cached_blocks``); the kernel warm-up's scratch is tiny and cached on the current stream.

        Returns the report (``bridge.startup["warmup"]``) with the total ``seconds``."""
        report: dict[str, Any] = {"components": {}}
        started = time.perf_counter()
        if self.staging_engine is not None:  # pinned ring and worker threads (process-level, CR-17-4)
            warmup_component(report, "staging_engine", self.staging_engine.warmup)
        prewarm_lineages = getattr(self.restorer, "prewarm_lineages", None)
        if sessions and prewarm_lineages is not None and self.runtime.list_lineages():
            warmup_component(report, "sessions", prewarm_lineages)
        grid = profile.default_grid()
        self._decode_profile = (getattr(profile, "name", None), grid)  # dev22j8v5: the decode floor's profile
        if self.pool is not None:
            warm_restore_kernels(report, self.device, lambda: kernel_warmup_shapes(
                profile, self.model, path="pictures", rope_mode=self.config.rope_mode, grid=grid))
            pool_warmup = getattr(getattr(self.restorer, "restorer", None), "warmup", None)
            if pool_warmup is None:
                report["components"]["decoder_pool"] = {"skipped": "this core's PackedRestorer has no warmup() "
                                                                   "(Step 5c canonical canvases)"}
            elif grid is None:
                report["components"]["decoder_pool"] = {"skipped": "the profile defaults give no token grid"}
            else:
                geometries = sorted({(profile.chunk_frames, grid[0], grid[1], self.config.hevc_pix_fmt,
                                      self.model.channels_for(target)) for target in ("K", "V")})
                top = min(1024, max(geometry[-1] for geometry in geometries))
                if getattr(self.config, "representation", None) == "hevc":  # wl-base (W1c): full-width tiles
                    top = max(geometry[-1] for geometry in geometries)
                warmup_component(report, "decoder_pool", lambda: pool_warmup(geometries, max_rank=top))
        elif getattr(self.config, "representation", None) == "svd" and self.device.type == "cuda":  # wl-base
            warm_restore_kernels(report, self.device, lambda: factor_warmup_shapes(
                profile, self.model, self.method, rope_mode=self.config.rope_mode))
        prewarm_archive = self._decode_guard_active() if archive is None else bool(archive)
        if prewarm_archive and grid is not None and self.device.type == "cuda":
            contexts = matrix_contexts(self.model, token_start=0, frames=profile.chunk_frames, grid=grid,
                                       streams=("cond",))
            encode = archive is True  # the default (guard) warm-up measures the fitter only: attach stays cheap
            warmup_component(report, "archive", lambda: self.pipeline.prewarm(contexts, encode=encode))
        if synthetic:
            if grid is None:
                report["components"]["synthetic_restore"] = {"skipped": "the profile defaults give no token grid"}
            else:
                warmup_component(report, "synthetic_restore",
                                 lambda: self._synthetic_restore(profile, grid, layers, seed))
        if {"archive", "synthetic_restore"} & set(report["components"]):
            release_cached_blocks(report, self.device)  # their pipelines/arenas cached GB on their own streams
        report["seconds"] = time.perf_counter() - started
        return report

    def _decode_guard_active(self) -> bool:
        """Whether decode sections take the decode entry guard (dev22j8: the attach warm-up then measures the fitter's
        peak working set): a GPU fitter, a raw tier with the pressure response, ``decode_entry_guard`` on and the
        decode deferral policy off."""
        tier = getattr(self, "raw_tier", None)
        state = getattr(self, "decode_guard_", None) or {}
        return (getattr(self, "fitter", None) is not None and tier is not None and tier.enabled
                and tier.pressure_enabled and bool(state.get("enabled", True))
                and not getattr(self.config, "defer_archival_during_decode", False))

    def _scratch_fitter(self):
        """A private GPU fitter for the warm-up pipeline: a pipeline's fitter thread adopts its fitter
        (``GPUFitter.adopt_current_thread``), so sharing the session pipeline's fitter would leave it owned by the
        closed warm-up thread."""
        if self.fitter is None:
            return None
        from ...gpu_fitting import GPUFitter

        return GPUFitter(**self._fitter_options)

    def _synthetic_restore(self, profile: Any, grid: tuple[int, int], layers: int, seed: int) -> dict[str, Any]:
        from ... import restore_kernels as rk

        try:
            model = dataclasses.replace(self.model, layers=max(1, min(int(layers), self.model.layers)))
        except ConfigurationError:  # per-matrix identities keep every layer
            model = self.model
        frames, streams, device = profile.chunk_frames, ("cond",), self.device
        tokens = frames * grid[0] * grid[1]
        scratch = os.path.join(self.root, f".warmup-{os.getpid()}-{id(self):x}")
        shutil.rmtree(scratch, ignore_errors=True)
        timings: dict[str, float] = {}
        clock = time.perf_counter()
        runtime = KVRuntime(scratch, model, encoder=self.provider.encode_tensor, decoder=self.provider.decode_tensor)
        restorer = pipeline = None
        try:
            if getattr(self.config, "representation", None) == "svd":  # wl-base (W1c): factor packs by GEMM
                from ...factor_packs import FactorLineageRestorer

                restorer = FactorLineageRestorer(runtime, self.device, arena_bytes=self.config.restore_arena_bytes,
                                                 basis_budget_bytes=self.config.basis_budget_bytes,
                                                 prepared_budget_bytes=self.config.prepared_budget_bytes,
                                                 staging_engine=self.staging_engine)
            elif self.pool is not None:
                from ...archive_restore import LineageRestorer

                restorer = LineageRestorer(runtime, self.pool, arena_bytes=self.config.restore_arena_bytes,
                                           basis_budget_bytes=self.config.basis_budget_bytes,
                                           prepared_budget_bytes=self.config.prepared_budget_bytes,
                                           staging_engine=self.staging_engine)
            else:
                from ...archive_restore import SoftwareLineageRestorer

                restorer = SoftwareLineageRestorer(runtime)
            pipeline = ArchivePipeline(
                runtime, method=self.method, registry=RankPredictorRegistry("reelcache-sglang-warmup",
                                                                            anchor_rank=self.method.anchor_rank),
                options=self.options, device=self.fit_device, fitter=self._scratch_fitter(),
                encode_workers=self.encode_workers, max_source_chunks=1, max_unpublished=1,
                max_staged_canvases=self.max_staged_canvases,
                restorer=restorer if (self.pool is not None and self.config.seal_prepare) else None)
            runtime.create_lineage("warmup", epoch_id="e0", streams=streams)
            # Power-law spectra (sigma_i ~ i^-a, a in 0.7..1.0): K98 / V95 land on low to mid ladder ranks, like
            # real KV chunks, and differ between matrices (kernel specializations by rank).
            generator = torch.Generator(device=device).manual_seed(seed)
            matrices = {}
            for layer in range(model.layers):
                for target in ("K", "V"):
                    channels = model.channels_for(target)
                    components = min(tokens, channels, 1024)
                    decay = 0.7 + 0.1 * ((2 * layer + (target == "V")) % 4)
                    spectrum = torch.arange(1, components + 1, dtype=torch.float32, device=device).pow(-decay)
                    left = torch.randn(tokens, components, generator=generator, device=device) * spectrum
                    right = torch.randn(components, channels, generator=generator, device=device)
                    matrices[(MatrixKey(layer, target), "cond")] = (left @ right).to(
                        getattr(torch, model.dtype_for(target)))
            contexts = matrix_contexts(model, token_start=0, frames=frames, grid=grid, streams=streams)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings["sources"] = time.perf_counter() - clock
            clock = time.perf_counter()
            pipeline.submit("warmup", "c000000", matrices, contexts, kind="chunk", timeout=None)
            node_ref = pipeline.drain("warmup")
            pipeline.reset("warmup")
            matrices.clear()
            timings["archive"] = time.perf_counter() - clock
            # Restores land at slot 0 and at one frame's offset: real slots are multiples of a frame, so both
            # integer-divisibility classes of the slot offset (kernel specializations) are exercised.
            frame_tokens = grid[0] * grid[1]
            buffers = {entry: torch.zeros(1, tokens + frame_tokens, profile.num_heads, profile.head_dim,
                                          device=device, dtype=getattr(torch, profile.native_dtype))
                       for entry in contexts}
            angles = (torch.arange(tokens, dtype=torch.float32, device=device)[:, None]
                      * torch.linspace(1e-4, 1.0, profile.head_dim // 2, device=device)[None, :])
            positions = rk.RopePositions(rk.RopeTable.from_cos_sin(angles.cos(), angles.sin()), 0)
            fused = self.config.rope_mode != "native"  # native mode rotates outside the restore kernels

            def entry_of(entry):
                return entry if isinstance(entry, tuple) else (entry, "cond")

            offset = [0]

            def destination(node_id, entry, tile):
                return rk.NativeView(buffers[entry_of(entry)], "BTHD", token_offset=offset[0])

            def rope(node_id, entry, tile):
                return positions if fused and entry_of(entry)[0].target == "K" else None

            for label, offset[0] in (("first", 0), ("shifted", frame_tokens), ("second", 0)):
                clock = time.perf_counter()  # "second": the warm cost of the same restore
                restored = restorer.restore(node_ref, destination, rope, ranges=[(0, tokens)])
                for handle in restored.pop("handles", []):
                    handle.release()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timings[f"restore_{label}"] = time.perf_counter() - clock
            stored = sum(os.path.getsize(os.path.join(folder, name))
                         for folder, _, names in os.walk(scratch) for name in names)
        finally:
            for close in (None if pipeline is None else pipeline.close,
                          None if restorer is None else getattr(restorer, "close", None), runtime.close):
                if close is not None:
                    try:
                        close()
                    except Exception:  # noqa: BLE001 - scratch resources; the directory is removed below
                        pass
            shutil.rmtree(scratch, ignore_errors=True)
        return {"tokens": tokens, "grid": list(grid), "layers": model.layers, "matrices": 2 * model.layers,
                "rope": "fused" if fused else "none", "stored_bytes": stored, "seconds": timings}

    # -- restore -------------------------------------------------------------------------------
    def prewarm(self, head_ref: str) -> float:
        """Load-time preparation of a session's history (packs prepared, bases resident, NVDEC sessions)."""
        prewarm = getattr(self.restorer, "prewarm", None)
        if prewarm is None or not self.runtime.load_node(head_ref).frontier:
            return 0.0
        token = None
        if self.eviction is not None:  # W1.4.3 (E1): a history with an evicted node is not prepared (its restore
            token = self.eviction.pin_available(  # replays); else pinned while it is read (not a restore: no touch)
                (node.ref for node in self.path(head_ref)), touch=False)
            if token is None:
                return 0.0
        started = time.perf_counter()
        try:
            prewarm(self.restorer.history(head_ref))
        finally:
            if token is not None:
                self.eviction.unpin(token)
        seconds = time.perf_counter() - started
        self.stats_["prewarms"] += 1
        self.stats_["prewarm_seconds"] += seconds
        return seconds

    def restore(self, *, head_ref: str, plans: Mapping[str, Any], caches: Mapping[str, Any], rope_cos_sin,
                rotate_key, rope_mode: str, num_layers: int, grid: tuple[int, int], overlapped: bool = False,
                overlap_options: Mapping[str, Any] | None = None) -> AttentionKVLease:
        """Install the attended history: ranges with a raw copy (freeze D tiers) through the RAW path with the
        engine's RoPE (bitwise the native cache), the others from their compressed nodes. ``lease.stats["tiers"]``
        records the tier of every attended range; ``lease.stats["verification"]`` the payload verification.

        ``overlapped`` (freeze F5, ``ReelCacheRuntimeConfig.overlap_restore``; Step 11's bridge): the compressed calls
        start as layer-overlapped restores (:func:`reelcache.overlapped_restore.start_overlapped_restore`) and return
        at once. ``lease.readiness`` holds ``(stream or None for every stream, LayerReadiness)`` per started call;
        ``lease.complete()`` joins them (raising a failure, typed) and fills the records the plain path fills before
        returning (restorer reports, verification, seconds); call it before anything derived from the restored KV
        leaves the request, then close the lease. Step 5c's O3 runs the restore thread in the caller's
        ``torch.inference_mode`` / grad mode (SGLang's caches are inference tensors), on its own CUDA stream, which
        waits for the caller's pending work (the install's zeroing, the RoPE tables: ``wait_event``). A history whose
        packs are not prepared falls back to the plain call (``lease.stats["overlap"]["fallback"]``).
        ``overlap_options``: ``stream`` (the restore stream; default a new one), ``require_prepared`` (default True),
        ``raw_stream`` (the RAW install's stream; default a new one), ``raw_depth`` (RAW layers in flight, default 1),
        ``raw_rope_per`` (``"span"``, default: one engine RoPE call per span, fewest bytes beside the forward; or
        ``"layer"``).
        ``lease.stats["overlap"]`` also times the synchronous start (``labels_seconds``, ``start_call_seconds``,
        ``raw_setup_seconds``).

        Overlapped RAW ranges (Step 5c's O4): ranges served by the raw tier (GPU raw: device-to-device copy;
        host raw: host-to-device copy from the pinned buffers) install layer by layer on their own stream, each layer
        with the engine's RoPE, through the same per-layer calls as the synchronous RAW path
        (:class:`~.restore.DenseInstall`, fast path: GPU raw read in place, no staging, no per-layer host wait), so they
        write the same bits. They join the readiness of their call group:
        one call for a single stream, one per CFG stream otherwise. So every cache layer waits ONCE, on a readiness
        that covers its raw and its compressed ranges (sources ``"raw"`` and ``"restore"``); an all-raw history
        returns a raw-only readiness, and an unprepared history overlaps its raw ranges while its compressed ranges
        run plain. ``lease.stats["tiers"]["raw_seconds"]`` is then the RAW setup on the caller's thread (the engine's
        span tables) and ``raw_install_seconds`` the background install (call to its last layer completed)."""
        check_restorable(plans)
        started = time.perf_counter()
        leases, raw = self._raw_sources(plans)
        try:
            lossy_plans = {stream: dataclasses.replace(plan, spans=tuple(span for span in plan.spans
                                                                         if (stream, span.token_start) not in raw))
                           for stream, plan in plans.items()}
            install = LineageInstall(plans=lossy_plans, caches=caches, rope_cos_sin=rope_cos_sin,
                                     rotate_key=rotate_key, rope_mode=rope_mode)
            tables_seconds = time.perf_counter() - started
            decoder = "nvdec_packed" if self.pool is not None else "software"
            lease = AttentionKVLease(store_lease=None, plans=plans, description={
                "nodes": sorted({span.node_ref for plan in plans.values() for span in plan.spans}),
                "representation": getattr(self.config, "representation", "svd_hevc"),  # wl-base (W1c): svd by GEMM
                "decoder": "factor_gemm" if getattr(self.config, "representation", None) == "svd" else decoder,
                "rope_mode": rope_mode,
                "restorer": type(self.restorer).__name__, "raw_tier": self.raw_tier is not None})
        except BaseException:
            for item in leases.values():
                item.release()
            raise
        for item in leases.values():
            lease.track(finalizer=item.release)
        if self.eviction is not None:  # W1.4.3 (E1): what this restore reads from the archive stays until it closes
            lossy_refs = {span.node_ref for plan in lossy_plans.values() for span in plan.spans}
            token = self.eviction.pin_available(lossy_refs)
            if token is None:
                evicted = sorted(self.eviction.evicted(lossy_refs))
                lease.close()
                raise StateError(f"archive restore of {head_ref}: nodes {evicted} were evicted by the archive budget "
                                 "(the bridge restores such a history from an available ancestor and a replay)")
            lease.track(finalizer=functools.partial(self.eviction.unpin, token))
        dtype = getattr(torch, self.model.dtype_for("K"))  # keys rotate in their capture dtype
        placements = {stream: plan.placement() for stream, plan in lossy_plans.items()}
        calls = [(None, None)] if len({tuple(item) for item in placements.values()}) <= 1 else [
            (stream, [(MatrixKey(layer, target), stream) for layer in range(num_layers) for target in ("K", "V")])
            for stream in lossy_plans]
        if overlapped and len(lossy_plans) > 1:
            # one overlapped call per CFG stream, in the forward's order (cond, then uncond), chained
            # (``after=``): every cond layer is released without waiting for the uncond work of that layer
            calls = [(stream, [(MatrixKey(layer, target), stream) for layer in range(num_layers)
                               for target in ("K", "V")]) for stream in lossy_plans]
        # Runs in the caller's mode: under SGLang's torch.inference_mode the restorer keeps its own state in normal
        # tensors and writes the (inference) native cache inside inference mode, fused or native RoPE alike.
        reports = []
        raw_seconds = 0.0
        options = dict(overlap_options or {})
        overlap = {"requested": bool(overlapped), "calls": 0, "fallback": None}
        lease.readiness = []
        lease.stats["overlap"] = overlap
        try:
            labels_started = time.perf_counter()
            labels = self._compressed_labels(head_ref, install)
            overlap["labels_seconds"] = time.perf_counter() - labels_started
            raw_installs: dict[str | None, Any] = {}
            if raw:
                raw_started = time.perf_counter()
                raw_plans = {stream: dataclasses.replace(plan, spans=tuple(
                    span for span in plan.spans if (stream, span.token_start) in raw)) for stream, plan in plans.items()}
                # W1.4.3 (E1): an evicting arm's replay promotions hold installed-form rows (post-RoPE K, as the
                # replay installed them): copied as they are, never rotated again (restore_dense's ``rotated``)
                installed = {ref for ref, item in leases.items() if getattr(item.entry, "form", None) == "installed"}

                def source(span, stream, key):
                    return self._raw_matrix(leases, span, stream, key)

                if overlapped:
                    if installed:  # copied on the caller's stream before the overlapped calls (no overlap for them)
                        restore_dense(lease, plans={stream: dataclasses.replace(plan, spans=tuple(
                            span for span in plan.spans if span.node_ref in installed))
                            for stream, plan in raw_plans.items()}, caches=caches, rope_cos_sin=rope_cos_sin,
                            rotate_key=rotate_key, num_layers=num_layers, device=self.device,
                            residency=self.runtime.residency, grid=grid, source=source, key_dtype=dtype,
                            rotated=lambda stream, span: True)
                        raw_plans = {stream: dataclasses.replace(plan, spans=tuple(
                            span for span in plan.spans if span.node_ref not in installed))
                            for stream, plan in raw_plans.items()}
                    # Step 5c's O4: per call group, the raw ranges join the group's readiness (installed layer
                    # by layer on their own stream); the engine's span tables are computed here, on this thread
                    for name, _ in calls:
                        group = {stream: plan for stream, plan in raw_plans.items()
                                 if plan.spans and (name is None or stream == name)}
                        if group:
                            raw_installs[name] = DenseInstall(
                                plans=group, caches=caches, rope_cos_sin=rope_cos_sin, rotate_key=rotate_key,
                                device=self.device, residency=self.runtime.residency, grid=grid, source=source,
                                key_dtype=dtype, non_blocking=True, rope_per=options.get("raw_rope_per", "span"))
                    overlap["raw_setup_seconds"] = raw_seconds = time.perf_counter() - raw_started
                else:
                    with tier_segment(raw_tier_label(raw.values())):  # workload telemetry (no-op without it)
                        restore_dense(lease, plans=raw_plans, caches=caches, rope_cos_sin=rope_cos_sin,
                                      rotate_key=rotate_key, num_layers=num_layers, device=self.device,
                                      residency=self.runtime.residency, grid=grid, source=source, key_dtype=dtype,
                                      **({"rotated": lambda stream, span: span.node_ref in installed}
                                         if installed else {}))
                    raw_seconds = time.perf_counter() - raw_started
            lease.stats["seconds"]["rope_tables"] = tables_seconds
            for stream, keys in calls:
                ranges = install.ranges(stream)
                raw_install = raw_installs.get(stream)
                if not ranges and raw_install is None:
                    continue
                if overlapped:
                    call_started = time.perf_counter()
                    after = lease.readiness[-1][1] if lease.readiness else None
                    readiness, compressed = self._start_overlapped(
                        head_ref, install, ranges=ranges, keys=keys, dtype=dtype, num_layers=num_layers,
                        options=options, overlap=overlap, after=after, raw=raw_install)
                    overlap["start_call_seconds"] = (overlap.get("start_call_seconds") or 0.0) + (
                        time.perf_counter() - call_started)
                    if readiness is not None:
                        lease.readiness.append((stream, readiness))
                    if compressed or not ranges:
                        continue
                with tier_segment(compressed_tier_label(labels.get(start, "compressed_cold") for start, _ in ranges)):
                    report = self.restorer.restore(head_ref, install.destination, install.rope, ranges=ranges,
                                                   keys=keys, **install.options(dtype))
                for handle in report.pop("handles", []):
                    lease.track(handle=handle)
                reports.append({key: value for key, value in report.items() if key != "history"} | {
                    "history_entries": len(report.get("history", [])), "stream": stream})
                lease.stats["jobs"] += int(report.get("entries", 0))
        except BaseException:
            for _, readiness in lease.readiness:  # a started call writes the caches until it returns
                with contextlib.suppress(BaseException):
                    readiness.join()
            lease.close()
            raise
        lease.stats["tiers"] = self._tier_records(plans, raw, labels, raw_seconds, num_layers)
        if self.eviction is not None:  # W1.4.3 (E1): a raw-served range's copy -- captured or a replay promotion
            for item in lease.stats["tiers"]["per_chunk"]:
                held = leases.get(item["node"])
                if held is not None and (item["stream"], item["token_start"]) in raw:
                    # ``exact``: False for a replay promoted from a history installed from the lossy archive
                    item.update(origin=held.entry.origin, form=held.entry.form,
                                exact=bool(getattr(held.entry, "exact", True)))

        def complete() -> AttentionKVLease:
            """Join the overlapped calls (their failure re-raised, typed), then the plain path's records."""
            if lease.stats.get("restorer") is None:
                raw_reports = []
                for stream, readiness in lease.readiness:
                    report = readiness.join()
                    raw_report = report.pop("raw_install", None)
                    if raw_report is not None:  # O4: the call group's RAW ranges
                        raw_reports.append(raw_report | {"stream": stream, "timeline": readiness.timeline()})
                    if "restore" not in getattr(readiness, "sources", ("restore",)):
                        continue  # a raw-only readiness: no restorer report
                    reports.append({key: value for key, value in report.items() if key != "history"} | {
                        "history_entries": 0, "stream": stream, "overlap_timeline": readiness.timeline()})
                    lease.stats["jobs"] += int(report.get("entries", 0))
                if raw_reports:  # the records restore_dense fills on the synchronous path
                    lease.stats["jobs"] += sum(int(item["jobs"]) for item in raw_reports)
                    lease.stats["spans"] = sum(int(item["spans"]) for item in raw_reports)
                    lease.stats["seconds"].update(
                        decode=sum(item["decode_seconds"] for item in raw_reports),
                        reconstruct=sum(item["launch_seconds"] + item["release_seconds"] for item in raw_reports))
                    lease.stats["tiers"]["raw_install_seconds"] = max(item["complete_seconds_from_call"]
                                                                      for item in raw_reports)
                    overlap["raw_install"] = raw_reports
                lease.stats["restorer"] = reports
                lease.stats["verification"] = self._verification_record(reports)
                lease.stats["seconds"]["restore"] = time.perf_counter() - started
            return lease

        lease.complete = complete
        if not lease.readiness:
            complete()
        self.stats_["restores"] += 1
        return lease

    def _start_overlapped(self, head_ref: str, install: Any, *, ranges, keys, dtype, num_layers: int,
                          options: Mapping[str, Any], overlap: dict[str, Any], after: Any = None,
                          raw: Any = None) -> tuple[Any, bool]:
        """One overlapped call group (see :meth:`restore`): its compressed ranges (Step 5c's O3) and its
        RAW ranges (``raw``, a :class:`~.restore.DenseInstall`; 5c's O4) in ONE readiness. Returns the readiness
        (None: nothing started) and whether the compressed ranges started overlapped; if not (unprepared history:
        ``overlap["fallback"]``), the caller runs them plain while the RAW ranges still overlap."""
        from ...overlapped_restore import start_overlapped_restore

        prepared = None
        if self.device.type == "cuda":
            prepared = torch.cuda.Event()
            prepared.record(torch.cuda.current_stream(self.device))  # the install's zeroing, the RoPE tables
        common = {"layers": num_layers, "wait_event": prepared, "after": after, "device": self.device,
                  "raw": None if raw is None else raw.layer, "raw_stream": options.get("raw_stream"),
                  "raw_depth": int(options.get("raw_depth", 1)), "telemetry": active_telemetry()}
        readiness, compressed = None, False
        if ranges:
            try:
                readiness = start_overlapped_restore(
                    self.restorer, head_ref, install.destination, install.rope, ranges=ranges, keys=keys,
                    stream=options.get("stream"), require_prepared=bool(options.get("require_prepared", True)),
                    **common, **install.options(dtype))
                compressed = True
            except StateError as exc:  # not prepared (or no prepared-pack cache): the plain call
                overlap["fallback"] = f"{type(exc).__name__}: {exc}"
        if readiness is None and raw is not None:
            readiness = start_overlapped_restore(None, **common)  # the RAW ranges alone
        if readiness is not None:
            overlap["calls"] += 1
            if raw is not None:
                overlap["raw_calls"] = overlap.get("raw_calls", 0) + 1
        return readiness, compressed

    def restore_allocates(self, plans: Mapping[str, Any]) -> bool:
        """Whether :meth:`restore` of ``plans`` would allocate device memory beyond the request's caches (Step 11 cut
        4: the bridge then runs the pre-restore relief and the restore section). False only when the RAW install reads
        device copies in place (:attr:`raw_install_in_place`) and every attended range has a GPU raw copy now
        (``raw_restore="prefer"``). True for host raw (host-to-device copies into per-layer buffers), compressed ranges
        (decode canvases) and mixed histories. An in-place install still takes the engine RoPE's outputs from the
        caching allocator, reused in stream order: one device span's K in the overlapped install (CausVid: 14.4 MB),
        one layer's gathered K and its rotation in the synchronous one (2 x 14.4 MB per attended chunk). A copy demoted
        between this query and the restore is read from host through such a per-layer buffer."""
        if not self.raw_install_in_place or self.raw_tier is None or self.config.raw_restore != "prefer":
            return True
        refs = {span.node_ref for plan in plans.values() for span in plan.spans}
        if not refs:
            return True
        entries = {entry.node_ref: entry for entry in self.raw_tier.entries(node_refs=refs)}
        return not all(ref in entries and entries[ref].on_device for ref in refs)

    def _raw_sources(self, plans: Mapping[str, Any]) -> tuple[dict[str, Any], dict[tuple[str, int], str]]:
        """Raw leases per node and the tier of every attended range served raw (``raw_restore="prefer"``)."""
        if self.raw_tier is None or self.config.raw_restore != "prefer":
            return {}, {}
        leases, raw = {}, {}
        for stream, plan in plans.items():
            for span in plan.spans:
                lease = leases.get(span.node_ref)
                if lease is None and span.node_ref not in leases:
                    lease = leases[span.node_ref] = self.raw_tier.acquire(span.node_ref)
                if lease is None:
                    continue
                start = span.token_start - span.node_offset
                group = lease.entry.group_for(start)
                if group is None or not group.rows:
                    continue  # this range of the node is not in the entry (never expected): compressed
                stream_bytes = sum(meta[5] for key, meta in group.meta.items()
                                   if not isinstance(key, MatrixKey) and key[1] == stream) or sum(
                    meta[5] for key, meta in group.meta.items() if isinstance(key, MatrixKey))
                raw[(stream, span.token_start)] = (lease.tier, stream_bytes * span.length // group.rows)
        held = {ref: lease for ref, lease in leases.items() if lease is not None}
        used = {plan_span.node_ref for stream, plan in plans.items() for plan_span in plan.spans
                if (stream, plan_span.token_start) in raw}
        for ref in set(held) - used:
            held.pop(ref).release()
        return held, raw

    @staticmethod
    def _raw_matrix(leases: Mapping[str, Any], span: Any, stream: str, key: MatrixKey) -> torch.Tensor:
        lease = leases[span.node_ref]
        start = span.token_start - span.node_offset
        tensor = lease.matrix(start, (key, stream))
        if tensor is None:
            tensor = lease.matrix(start, key)
        if tensor is None:
            raise IntegrityError(f"raw tier entry of {span.node_ref} lacks {key} of stream {stream!r}")
        return tensor

    def _compressed_labels(self, head_ref: str, install: Any) -> dict[int, str]:
        """``compressed_prepared`` (node pack in the restorer's prepared cache) or ``compressed_cold`` per range
        start, looked up before the restore prepares anything."""
        ranges = install.ranges()
        prepared = getattr(self.restorer, "prepared", None)
        if not ranges or prepared is None or not callable(getattr(self.restorer, "history", None)):
            return {start: "compressed_cold" for start, _ in ranges}
        labels = {}
        try:
            for item in self.restorer.history(head_ref, ranges=ranges):
                labels[int(item["token_start"])] = ("compressed_prepared" if item.get("packet") is not None
                                                    and prepared.contains(item["packet"]) else "compressed_cold")
        except Exception:  # noqa: BLE001 - records only; the restore reports its own errors
            return {start: "compressed_cold" for start, _ in ranges}
        return labels

    def _tier_records(self, plans, raw, labels, raw_seconds, num_layers) -> dict[str, Any]:
        per_chunk, chunks, nbytes = [], {}, {}
        for stream, plan in plans.items():
            for span in plan.spans:
                served = raw.get((stream, span.token_start))
                tier = served[0] if served is not None else labels.get(span.token_start, "compressed_cold")
                item = {"stream": stream, "chunk": span.chunk_index, "node": span.node_ref,
                        "token_start": span.token_start, "tokens": span.length, "tier": tier}
                if served is not None:
                    item["bytes"] = served[1]
                    nbytes[tier] = nbytes.get(tier, 0) + served[1]
                chunks[tier] = chunks.get(tier, 0) + 1
                per_chunk.append(item)
        return {"per_chunk": per_chunk, "chunks": chunks, "bytes": nbytes, "raw_seconds": raw_seconds,
                "policy": None if self.raw_tier is None else self.config.raw_restore}

    def verification_mode(self) -> str:
        """The CONFIGURED payload verification of compressed restores: the staging engine's
        (``async`` | ``sync`` | ``off``, Step 5c D-E) when it reports one, else ``sync`` with ``verify_reads`` (hashed
        before use) or ``off``."""
        mode = getattr(self.staging_engine, "verification_mode", None)
        if callable(mode):
            return mode(self.store)
        return "sync" if self.config.verify_reads else "off"

    def _verification_record(self, reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """``restore.lease.verification``: the CONFIGURED mode (also for all-raw restores, which read no payload),
        ``verified_payloads`` (payload objects of the compressed ranges verified; 0 when none was read) and
        ``payload_wait_seconds``, summed over the restorer reports (Step 5c's ``verification_report``); a differing
        reported mode is kept as ``reported_mode``."""
        mode = self.verification_mode()
        record: dict[str, Any] = {"mode": mode, "verified_payloads": 0, "payload_wait_seconds": 0.0}
        for report in reports:
            verification = report.get("verification")
            if isinstance(verification, Mapping):
                if verification.get("mode") not in (None, mode):
                    record["reported_mode"] = verification["mode"]
                record["verified_payloads"] += int(verification.get("verified_payloads") or 0)
                record["payload_wait_seconds"] += float(verification.get("payload_wait_seconds") or 0.0)
            elif "payload_wait_seconds" in report:
                record["payload_wait_seconds"] += float(report.get("payload_wait_seconds") or 0.0)
        return record

    def storage(self, ref: str) -> dict[str, Any]:
        return self.runtime.node_storage(ref)

    def stats(self) -> dict[str, Any]:
        pipeline = {key: value for key, value in self.pipeline.stats().items() if key != "timeline"}
        return {"backend": self.name, **self.stats_, "pipeline": pipeline, "restorer": self.restorer.stats(),
                "encoder": dict(self.encoder_record),
                "staging_engine": None if self.staging_engine is None else self.staging_engine.stats(),
                "registry_sequence": self.registry.sequence, "quarantined": self.quarantined(),
                "damaged_objects": len(self.store.damaged), "decode_guard": dict(self.decode_guard_),
                **({"archive": self.eviction.stats()} if self.eviction is not None else {})}  # W1.4.3 (E1)

    @contextlib.contextmanager
    def memory_critical(self, reason: str):
        """A memory-critical section of the worker (the VAE decode).

        * Pressure response (freeze F; raw tier on): on entry, when the device headroom is below the reserve, the
          least recently used GPU raw copies of fitted and published chunks are demoted until it is covered, waiting
          for their spills (``report["raw_pressure"]``). The section's working set is learned for the auto reserve.
          On ``torch.cuda.OutOfMemoryError`` inside the section every GPU raw copy is demoted, the CUDA cache emptied
          and the error re-raised (``report["oom_relief"]``; no retry here: a decode may have advanced its caches).
        * With ``defer_archival_during_decode``, in a decode section only (:func:`policy_defers`: ``"vae_decode"``,
          ``"decode"``, ``"*_decode"``): the pipeline starts no GPU fitting inside it, a node already on the GPU
          finishes first (the caller waits for it), GPU raw copies are evacuated (not a placement) and the caching
          allocator's unoccupied blocks are released on entry (``defer_gpu_work``). A restore section (``"restore"``)
          never takes this path (F4): it keeps the entry check, the pressure-triggered deferral below, the fitter
          re-check and the OOM relief, so a restore without pressure demotes, waits for and releases nothing.
        * Decode entry guard (dev22j6, reworked in dev22j7 and dev22j8; ``decode_entry_guard``, default on; GPU
          fitter, raw tier with the pressure response, the policy above off). The decode needs ``base`` = the larger
          of the reserve and a prior (``decode_reserve_prior_bytes``, else ``DECODE_RESERVE_PRIOR_FRACTION`` of the
          device), plus the growth a node on the GPU may still have (``ArchivePipeline.inflight_growth_bound``: the
          fitter's peak working set, learned at the attach warm-up and online on the fitter's own streams, scaled by
          the node's size). It is checked against the device headroom (the driver's free memory plus the allocator's
          wholly free cached segments, which the caching allocator releases itself when an allocation needs them):

          1. it fits: nothing is done (no cache release, relief, pause or wait);
          2. short with a node on the GPU (dev22j8): one snapshot of what the fitter's streams hold gives the node's
             remaining growth (``max(0, bound - held)``); when that fits, nothing is done either;
          3. still short: the fitter pauses at its next step boundary for the section, or at once while it waits
             without allocating (``ArchivePipeline.pause_gpu_work``); once paused it allocates nothing more, so the
             growth counts 0 (the measured remaining growth still counts when the pause is not acknowledged within
             ``PAUSE_ACK_TIMEOUT``);
          4. still short: the caching allocator's unoccupied blocks go back to the driver -- only when the fragmented
             cached bytes could cover the shortfall (dev22j8) -- then the pressure response runs for the target
             (least recently used GPU raw copies to host, trims);
          5. still short: the section waits for the node and defers archival
             (``report["deferral_reason"] = "short_after_relief"``, or ``"unlearned_reserve"`` while the reserve is
             not learned, with its wait; dev22j8v6: next to a node, where the decode floor is known by then, always
             ``"short_after_relief"``, with ``fits_if_acked`` after an unacknowledged pause and the step that held
             the fitter in ``ack_timeout_phase``).

          dev22j8: no unconditional first-decode deferral; an unlearned decode takes the same checks with the prior
          as its floor. ``report["decode_guard"]`` records the check and its actions; ``stats()["decode_guard"]``
          counts them. dev22j8v4: with a node on the GPU the headroom is the strict one (the driver's free memory
          plus what the allocator can release; one ``torch.cuda.memory_snapshot``). dev22j8v5 (F5.4 findings), with a
          node on the GPU: ``base`` is at least the decode floor (``DECODE_FLOOR_MARGIN`` x the decode's clean working
          set: configured, the profile's at its default geometry, else learned; with none, the decode waits for the
          node, cause ``"unlearned_reserve"``); the driver's free memory alone covering the target decides without a
          snapshot; short on the formula, the pause comes before the snapshot. The working-set mark is taken after
          any wait; a clean section (no node ran unpaused through it) also teaches the learned floor.
        * Pressure-triggered deferral (freeze F, dev22j2): without that setting, a section whose entry check found
          the headroom below the target defers archival GPU work the same way for this section only (no planner
          re-upload of spilled chunks, no fitter working set; the fitter's cached blocks are released), with the
          entry wait for a node already on the GPU capped at ``PRESSURE_DEFER_TIMEOUT`` (on timeout the section runs
          without deferral, recorded). No pressure, no deferral. Recorded in the report and the pressure event:
          ``deferred_archival``, ``deferral_wait_seconds`` (the section waiting for the fitter) and
          ``archival_deferred_seconds`` (archival paused).

        ``reason`` labels the section ("vae_decode", "restore", ...): the report, the pressure event's ``where`` and
        the trims' reason. Inside a deferred section the pipeline's ``drain`` / ``prewarm`` raise ``StateError`` and
        ``submit`` raises ``CapacityError`` rather than wait for deferred fits: wrap a restore after its drain.

        Yields a report, completed on exit. A pipeline that cannot defer (closed) never fails the section: recorded.
        """
        report: dict[str, Any] = {"reason": reason, "deferred": False}
        tier = self.raw_tier if self.raw_tier is not None and self.raw_tier.enabled else None
        # Fitter re-check (dev22j4): watched from before the entry check, so GPU work the fitter starts any time inside
        # the section is re-checked against the reserve plus its own bytes and waits for the section (capped).
        watch = _watch_section(self, tier, reason)
        rechecks = None if watch is None else watch.__enter__()
        token = None
        pressure = None
        guard = None
        paused = None
        node_at_mark = True

        def mark() -> None:
            # dev22j8v5: the working-set mark is taken after any wait for a node (a node finishing during the wait
            # frees its memory, which would hide the decode's own need), right before the section runs
            nonlocal token, node_at_mark
            if tier is None or token is not None:
                return
            try:
                node_at_mark = _node_on_gpu(self)
                token = tier.section_begin()
            except Exception as exc:  # noqa: BLE001 - recording only
                report["section_mark_error"] = f"{type(exc).__name__}: {exc}"

        try:
            try:
                pressure = self.relieve_pressure(reason)
                if pressure is not None:
                    report["raw_pressure"] = pressure
                guarded = getattr(self, "_decode_entry_guard", None)  # stand-ins without it: no guard
                if guarded is not None:
                    guard = guarded(reason, tier, pressure, report)
                    if isinstance(guard, _GuardPause):
                        paused, guard = guard, None
            except Exception as exc:  # noqa: BLE001 - the response is best effort; the section always runs
                report["raw_pressure_error"] = f"{type(exc).__name__}: {exc}"
            try:
                if (self.config.defer_archival_during_decode and self.fitter is not None
                        and policy_defers(reason)):
                    yield from self._deferred_section(report, tier, mark=mark)
                elif (pressure is not None and pressure.get("pressure") and self.fitter is not None
                      and callable(getattr(self.pipeline, "defer_gpu_work", None))):
                    yield from self._pressure_deferred_section(report, tier, pressure, mark=mark)
                elif guard is not None and callable(getattr(self.pipeline, "defer_gpu_work", None)):
                    yield from self._pressure_deferred_section(report, tier, guard, mark=mark)
                elif paused is not None:  # dev22j7: the fitter stays paused at its step boundary for the section
                    mark()
                    entered = time.perf_counter()
                    try:
                        yield report
                    finally:
                        report["decode_guard"]["pause"]["section_seconds"] = time.perf_counter() - entered
                else:
                    mark()
                    yield report
            except torch.cuda.OutOfMemoryError:
                if tier is not None and tier.pressure_enabled:
                    report["oom_relief"] = self._oom_relief(tier)
                raise
            finally:
                if tier is not None:
                    try:
                        # dev22j8v5: clean -- a decode section no node ran unpaused through (it waited for the node,
                        # or the node stayed paused, or none was on the GPU): its working set is the decode's own need
                        acked = paused is not None and bool(report["decode_guard"]["pause"].get("acked"))
                        clean = policy_defers(reason) and (report.get("deferred") is True or acked
                                                           or (not node_at_mark and not _node_on_gpu(self)))
                        working = tier.section_end(token, clean=clean)
                        if working is not None:
                            report["working_set_bytes"] = working
                            report["working_set_clean"] = clean
                    except Exception:  # noqa: BLE001 - recording only
                        pass
        finally:
            if paused is not None:
                paused.close()
            if watch is not None:
                watch.__exit__(None, None, None)  # closed: a fitter held back by the re-check starts
                _record_rechecks(report, tier, reason, rechecks)

    def _deferred_section(self, report: dict[str, Any], tier, *, mark: Callable[[], None] | None = None):
        before = self.pipeline.stats().get("deferral_released_bytes", 0)
        started = time.perf_counter()
        if tier is not None:
            # GPU raw copies leave the device before the section (their host copies keep I1); the release of the
            # caching allocator's blocks below then frees them. Not a placement: the host copies stay LRU-evictable
            # (D2; an explicit placement would exempt every entry from eviction and fill the host tier).
            try:
                report["raw_demoted"] = tier.demote(tier.entries(), wait=True, timeout=30.0, place=False)
            except (CapacityError, StateError) as exc:
                report["raw_demote_error"] = f"{type(exc).__name__}: {exc}"
        with contextlib.ExitStack() as stack:
            try:
                stack.enter_context(self.pipeline.defer_gpu_work(wait=True, release_cached_memory=True))
                report["deferred"] = True
            except (CapacityError, StateError) as exc:
                report["error"] = f"{type(exc).__name__}: {exc}"
            entered = time.perf_counter()
            report["wait_seconds"] = entered - started
            if mark is not None:
                mark()  # dev22j8v5: the working-set mark after the wait
            yield report
            report["section_seconds"] = time.perf_counter() - entered
        if report["deferred"]:
            report["released_bytes"] = self.pipeline.stats().get("deferral_released_bytes", 0) - before

    def _decode_entry_guard(self, reason: str, tier, pressure: dict | None, report: dict[str, Any]):
        """dev22j6, reworked in dev22j7 and dev22j8: make room for a decode section (see :meth:`memory_critical`).
        Returns None (run the section as is), a :class:`_GuardPause` (the fitter stays paused for the section) or the
        event to defer archival on (the section waits for the node). Runs after the reserve check and before the
        section's working-set mark."""
        from ...raw_tier import device_headroom

        state = getattr(self, "decode_guard_", None)
        if state is None:
            state = self.decode_guard_ = _decode_guard_state(True)
        for key, value in _decode_guard_state(True).items():  # counters of an older state (stand-ins, reloads)
            state.setdefault(key, value)
        if (not state.get("enabled", True) or not policy_defers(reason) or tier is None
                or not tier.pressure_enabled or self.fitter is None or self.config.defer_archival_during_decode
                or (pressure is not None and pressure.get("pressure"))):
            return None  # off, not a decode, nothing to protect, the policy defers, or the reserve check defers
        with tier._cond:
            device = tier._gpu_device() or tier._device_hint
        memory = tier._memory(device) if device is not None else None
        if memory is None:
            return None
        state["checks"] += 1
        unlearned = not tier.reserve_configured() and tier.learned_working_set_bytes() == 0
        reserve = tier.reserve_bytes()
        prior = decode_reserve_prior_bytes(self.config, memory)
        base = max(reserve, prior)
        bound_of = getattr(self.pipeline, "inflight_growth_bound", None)
        if callable(bound_of):
            bound = dict(bound_of())
        else:  # an older pipeline: its node's working set
            inflight = getattr(self.pipeline, "inflight_gpu_bytes", None)
            nbytes = int(inflight()) if callable(inflight) else 0
            bound = {"inflight": nbytes > 0, "bytes": nbytes, "basis": "working_set" if nbytes else None}
        growth = int(bound["bytes"]) if bound["inflight"] else 0
        formula = device_headroom(memory)
        record = {"reserve": reserve, "prior": prior, "base": base, "inflight": bool(bound["inflight"]),
                  "inflight_bytes": growth, "bound_basis": bound.get("basis"), "bound_scale": bound.get("scale"),
                  "growth": growth, "target": base + growth, "free": int(memory.get("free", 0)),
                  "headroom": formula, "formula_headroom": formula, "decided_by": "formula",
                  "unlearned": unlearned, "actions": []}
        report["decode_guard"] = record
        strict = segments = None
        pause = None
        acked = False
        measured = None
        try:
            if bound["inflight"]:
                state["inflight_checks"] += 1
                # dev22j8v5 (F5.4 finding 1): the decode floor -- the least a decode next to a node needs, its own
                # clean working set (the prior and a lagging learned reserve can be below it); with none known yet,
                # the decode waits for the node (F5.1)
                floor, source = _decode_floor(self, tier)
                record.update(target_before_floor=base + growth, floor_bytes=floor, floor_source=source)
                if floor is None:
                    state["floor_waits"] += 1
                    record["short"] = True
                    report["deferral_reason"] = "unlearned_reserve"
                    event = pressure if pressure is not None else {"event": "pressure", "where": reason,
                                                                   "time": time.time()}
                    event["deferral_reason"] = "unlearned_reserve"
                    return event
                if floor > base:
                    state["floor_raised"] += 1
                    base = floor
                    record.update(base=base, target=base + growth)
                if int(memory.get("free", 0)) >= base + growth:
                    # (i) the driver's free memory alone covers the decode and the node's bound (a lower bound on the
                    # strict headroom): no snapshot
                    state["fits"] += 1
                    state["free_fits"] += 1
                    record.update(short=False, fits="as_is", decided_by="free", headroom=int(memory.get("free", 0)))
                    return None
                if formula < base + growth and callable(getattr(self.pipeline, "pause_gpu_work", None)):
                    # (ii) short on the formula, so on the strict headroom too: pause first, snapshot after the
                    # acknowledgement (a paused fitter allocates nothing under the allocator's lock)
                    state["pauses_first"] += 1
                    pause, acked, growth = _open_guard_pause(self.pipeline, record, state, base, growth)
                # dev22j8v4 (F5.2's RAVEN decode OOM): with a node on the GPU every verdict uses the strict headroom --
                # the driver's free memory plus what the allocator can actually release. The formula counts free
                # fragments inside used segments, which expandable segments hide from ``inactive_split`` and nobody
                # can reclaim (a paused node's segments are such). Without a node, F5.1's formula decides.
                strict, segments = _strict_measure(tier, device, memory, record, state)
                if strict is not None:
                    state["strict_checks"] += 1
                    record.update(headroom=strict["headroom"], decided_by="strict")
            headroom = record["headroom"]
            if pause is None:
                if headroom >= base + growth:
                    state["fits"] += 1
                    record.update(short=False, fits="as_is")
                    return None  # room for the decode and the node's growth bound: the F4 path
                if strict is not None and formula >= base + growth:
                    state["strict_overrides"] += 1  # the formula alone would have let the decode run as is
                    record["strict_override"] = True
            if bound["inflight"]:
                if not acked:
                    # dev22j8: the snapshot of what the fitter's streams hold gives the node's remaining growth (its
                    # bound minus what it holds); when that fits, no pause (or an unacknowledged one) is needed
                    measured = _fitter_memory(self.pipeline, segments)
                    if measured is not None:
                        growth = max(0, int(bound["bytes"]) - int(measured["held"]))
                        record.update(held=int(measured["held"]), fitter_reserved=int(measured["reserved"]),
                                      remaining=growth, growth=growth, target=base + growth)
                        if headroom >= base + growth and pause is None:
                            state["fits_measured"] += 1
                            record.update(short=False, fits="measured")
                            return None
                if pause is None and callable(getattr(self.pipeline, "pause_gpu_work", None)):
                    pause, acked, growth = _open_guard_pause(self.pipeline, record, state, base, growth)
                if pause is not None and headroom >= base + growth:
                    state["fits_paused"] += 1
                    state["inflight_relieved"] += 1
                    record.update(short=False, fits="paused")
                    return pause
            relief = None
            if headroom < base + growth:
                running = measured if (measured is not None and not acked) else None
                memory, relief = _guard_make_room(self, reason, tier, device, memory, base + growth, state, record,
                                                  own_fragments=0 if running is None else int(running["fragments"]),
                                                  strict=strict is not None)
                if record["actions"] and strict is not None:  # dev22j8v4: the strict headroom after the actions
                    after, segments = _strict_measure(tier, device, memory, record, state, suffix="_after")
                    if after is None:  # the snapshot went away: the driver's free memory only
                        after = {"headroom": int(memory.get("free", 0))}
                    strict = after
                if running is not None and record["actions"]:
                    # a release or a trim may also have returned the running node's free blocks: measure again
                    again = _fitter_memory(self.pipeline, segments)
                    if again is not None:
                        growth = max(0, int(bound["bytes"]) - int(again["held"]))
                        record.update(held_after=int(again["held"]), remaining_after=growth, growth=growth,
                                      target=base + growth)
            record["free_after"] = int(memory.get("free", 0))
            record["formula_headroom_after"] = device_headroom(memory)
            record["headroom_after"] = (record["formula_headroom_after"] if strict is None
                                        else int(strict["headroom"]))
            short = record["headroom_after"] < base + growth
            record["short"] = short
            if not short:
                state["fits_after_relief"] += 1
                if bound["inflight"]:
                    state["inflight_relieved"] += 1
                record["fits"] = "after_relief"
                return pause
            if pause is not None and not acked:
                # dev22j8v6: an acknowledged pause would have let the decode run (only the node's growth was short)
                record["fits_if_acked"] = record["headroom_after"] >= base
            if unlearned and not bound["inflight"]:  # dev22j8: no unconditional first-decode deferral
                state["unlearned_deferrals"] += 1
                cause = "unlearned_reserve"
            else:
                # dev22j8v6 (F5.5 gate): next to a node the decode floor is known by now (else it waited at once,
                # ``unlearned_reserve``), whether or not the reserve has learned a working set
                state["short_deferrals"] += 1
                if record["headroom_after"] >= base:
                    state["inflight_deferrals"] += 1  # only the node's growth made the wait necessary
                cause = "short_after_relief"
            if pause is not None:  # the section waits for the node: it may not stay paused
                pause.close()
                pause = None
            report["deferral_reason"] = cause
            # deferred on the guard's relief when it ran, else on the entry check's event (as dev22j6)
            event = relief if relief is not None else pressure
            if event is None:
                event = {"event": "pressure", "where": reason, "time": time.time()}
            event["deferral_reason"] = cause
            return event
        except BaseException:
            if pause is not None:
                pause.close()
            raise

    def _release_cached_blocks(self) -> int | None:
        """Return the caching allocator's unoccupied blocks (every stream's pool) to the driver: the bytes released,
        None without CUDA (dev22j6)."""
        device = getattr(self, "device", None)
        if device is None or getattr(device, "type", None) != "cuda":
            return None
        before = torch.cuda.memory_reserved(device)
        torch.cuda.empty_cache()
        return int(before - torch.cuda.memory_reserved(device))

    def _pressure_deferred_section(self, report: dict[str, Any], tier, pressure: dict, *,
                                   mark: Callable[[], None] | None = None):
        """The section under entry pressure: archival GPU work is deferred until it exits (see
        :meth:`memory_critical`); the entry wait is capped, a timeout leaves the section undeferred (recorded)."""
        stats = self.pipeline.stats()
        before = (stats.get("deferred_seconds", 0.0), stats.get("deferral_released_bytes", 0))
        started = time.perf_counter()
        with contextlib.ExitStack() as stack:
            try:
                stack.enter_context(self.pipeline.defer_gpu_work(wait=True, release_cached_memory=True,
                                                                 timeout=PRESSURE_DEFER_TIMEOUT))
                report["deferred"] = report["deferred_archival"] = True
            except (CapacityError, StateError) as exc:
                report["deferred_archival"] = False
                report["deferral_error"] = f"{type(exc).__name__}: {exc}"
            entered = time.perf_counter()
            report["deferral_wait_seconds"] = entered - started
            if mark is not None:
                mark()  # dev22j8v5: the working-set mark after the wait
            yield report
            report["section_seconds"] = time.perf_counter() - entered
        fields = {"deferred_archival": report["deferred_archival"],
                  "deferral_wait_seconds": report["deferral_wait_seconds"]}
        if report["deferred_archival"]:
            stats = self.pipeline.stats()
            fields["archival_deferred_seconds"] = stats.get("deferred_seconds", 0.0) - before[0]
            report["archival_deferred_seconds"] = fields["archival_deferred_seconds"]
            report["released_bytes"] = stats.get("deferral_released_bytes", 0) - before[1]
        else:
            fields["deferral_error"] = report.get("deferral_error")
        if tier is not None:
            tier.annotate_pressure_event(pressure, **fields)
        else:
            pressure.update(fields)

    @staticmethod
    def _oom_relief(tier) -> dict[str, Any]:
        """Layer 3: demote every GPU raw copy (waiting for the spills), trim every registered device cache and
        return the freed segments to the driver (``torch.cuda.empty_cache``, run by the trims)."""
        try:
            relief = tier.relieve_all("oom", timeout=30.0) or {}
        except Exception as exc:  # noqa: BLE001 - the OOM is re-raised either way
            relief = {"error": f"{type(exc).__name__}: {exc}"}
        if "trims" not in relief and torch.cuda.is_available():
            started = time.perf_counter()
            torch.cuda.empty_cache()
            relief["empty_cache_seconds"] = time.perf_counter() - started
        return relief

    def relieve_device_memory(self, reason: str = "oom") -> dict[str, Any] | None:
        """Layer 3 for callers with their own OOM handling (e.g. the bridge around a restartable step): demote every
        GPU raw copy, trim the registered device caches, empty the CUDA cache; returns the report (None without a
        raw tier)."""
        if self.raw_tier is None or not self.raw_tier.enabled:
            return None
        return self._oom_relief(self.raw_tier)

    def relieve_pressure(self, where: str = "request_start", *, needed_bytes: int | None = None) -> dict | None:
        """Pressure check before a memory-hungry step (freeze F): the decode seam (:meth:`memory_critical`) and the
        bridge at request start, before the denoise loop allocates its KV cache (``needed_bytes``: what it is about
        to allocate). Below the target (max of the tier's reserve and ``needed_bytes``) GPU raw copies demote first
        (least recently used fitted/published chunks), then the registered device caches are trimmed in order
        (:mod:`reelcache.device_pressure`) and the CUDA cache emptied. Without GPU raw copies only the trims run. Every
        event (bytes freed per component, latency) is recorded in ``stats()["pressure"]``. None on CPU workers or
        with the response off (``gpu_raw_reserve_bytes=0``)."""
        from ...device_pressure import trim_device_caches
        from ...raw_tier import cuda_device_memory, device_headroom

        tier = self.raw_tier if self.raw_tier is not None and self.raw_tier.enabled else None
        if tier is not None:
            if not tier.pressure_enabled:
                return None
            event = tier.pressure_check(where, wait=True, timeout=30.0, needed_bytes=needed_bytes)
            if event is not None:
                return event
            device = tier._device_hint
            measure = (lambda: tier._memory(device)) if device is not None else None
            reserve = tier.reserve_bytes()
        else:
            measure, reserve, device = None, 0, None
        if measure is None or (device is not None and tier._memory(device) is None):
            device = getattr(self, "device", None)
            if device is None or device.type != "cuda":
                return None
            measure = lambda: cuda_device_memory(device)  # noqa: E731
        started = time.perf_counter()
        target = max(reserve, int(needed_bytes or 0))
        headroom = device_headroom(measure())
        event: dict[str, Any] = {"event": "pressure", "where": where, "time": time.time(), "headroom": headroom,
                                 "reserve": reserve, "needed_bytes": needed_bytes, "target": target,
                                 "gpu_bytes_before": 0, "pressure": headroom < target}
        if headroom < target:
            event["trims"] = trim_device_caches(where, device=device, needed_bytes=target,
                                                headroom=lambda: device_headroom(measure()))
            event["headroom_after"] = device_headroom(measure())
        event["seconds"] = time.perf_counter() - started
        if tier is not None:
            tier.record_pressure_event(event)
        return event

    def collect_garbage(self) -> dict[str, Any]:
        """Reclaim store objects no ref (lineage head, named checkpoint) or lease reaches; drops node caches and the
        restorer's cached packs and device-resident bases of the reclaimed objects (CR-17-3).
        The caller drains every lineage first (no commit in flight)."""
        collected = dict(self.runtime.collect_garbage())
        self._records.clear()
        self._states.clear()
        forget = getattr(self.restorer, "forget_unstored", None)
        if callable(forget):
            collected["restorer_forgotten"] = forget()
        if self.eviction is not None:  # W1.4.3 (E1): collected (deleted) nodes leave the budget's accounting
            collected["eviction_pruned"] = self.eviction.prune()
        return collected

    def close(self) -> None:
        errors = []
        try:
            self.pipeline.close(drain=True)
        except BaseException as exc:  # noqa: BLE001 - every resource is still released below
            errors.append(exc)
        for closer in (getattr(self.restorer, "close", None),
                       None if self.staging_engine is None else self.staging_engine.close,
                       None if self.pool is None else self.pool.close, self.runtime.close):
            if closer is None:
                continue
            try:
                closer()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise errors[0]


__all__ = ["LineageSessionArchive", "SESSION_DETAILS_KEY", "staging_threads"]
