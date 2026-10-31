"""LongCat-Video session provider for SGLang's ReelCache session controls (Step 14 CR-14-3).

SGLang (branch ``reelcache-integration``, 321536df3e) attaches a pipeline-declared session provider when
``--reelcache-enable`` is given and the pipeline has no causal denoising stage:
``LongCatVideoContinuationPipeline.reelcache_session_provider = PROVIDER_SPEC``, and
``reelcache.integrations.sglang.create_sglang_backend`` calls :func:`create_provider` (``config`` = the
``--reelcache-config`` dict). The returned :class:`LongCatSessionProvider` serves the worker interface of Step 14:
``control(op, args)`` (between requests), ``inflight_control(op, args)`` (from the boundary hook while a request
runs), ``set_boundary_hook(hook)`` and ``close()``; ops, arguments, results and errors are those of
``reelcache.integrations.sglang.controls`` (same JSON, same :class:`SessionControlError` codes) with the SEGMENT as the
boundary unit.

Generation is an ordinary video request carrying the ``causal_kv_session`` handle; the port's stages ask
:meth:`LongCatSessionProvider.begin_request` for the request's plan and report every segment boundary:

* ``generate`` -- a new session from the request's conditioning video (``replace`` drops an existing one);
* ``resume`` -- continue from the session head; ``fork`` (``from_session``, ``checkpoint``) -- a new session at a
  boundary of the source (copy-on-write) and continue from there; ``rewind`` (``checkpoint``) -- move the head back
  to an ancestor boundary and continue from there. The boundary conditioning (13 frames + generator state) always
  comes from the session archive, never from the request; a request that also carries conditioning frames, a start
  segment, a generator state or a cache backend is refused;
* the request's ``prompt`` applies to its new segments: a new prompt is a native new continuation segment. The
  conditioning cache depends only on the conditioning latents, so a segment regenerated from an archived boundary
  (rewind/fork) restores its cache from the segment's ``longcat_segment`` epoch when that epoch is published, and
  builds it natively otherwise (recorded per segment: never a silent substitute);
* ``pause_after_chunks`` = the session's segment count at which the request stops; ``num_segments`` (a declared
  ``/v1/videos`` field) = the new segments requested; an in-flight ``pause`` stops at the next segment boundary;
* ``output``: ``new`` only (the session archives boundaries and caches, not every decoded frame);
  ``decode_context_latents`` does not apply (each segment always decodes with its 4 conditioning latents);
* ``overlap_restore`` (bool, freeze F5): this request's svd_hevc restores run layer-overlapped with the segment's first
  denoising step, or plain; absent, the config's ``overlap_restore`` (default on) applies.

Persistence (restart safety): sessions, boundaries and named checkpoints are records in the ReelCache store of the cache
archive (:mod:`.session_archive`); the conditioning-cache epochs are lineage nodes of the same store
(:mod:`.reelcache_backend`). A boundary is committed (frames blob + boundary manifest, then a compare-and-swap of the
session ref) before the request continues past it; cache epochs publish atomically through the archive pipeline.
After a kill, a new provider on the same store lists every session at its last committed boundary and re-indexes the
published epochs (``reload_index``); nothing half-written is ever referenced.

Live epochs (CR-14-6): the conditioning-cache epochs recorded by the live boundaries (every boundary a session head or
a named checkpoint reaches). Garbage collection keeps exactly their archives (RAW reference branches of other epochs
are deleted; svd_hevc nodes no lineage reaches are collected) and re-indexes exactly them, in both representations; so
does a restart. A RAW branch carries no epoch identity of its own: the live boundaries supply it. Deleting a session
keeps the svd_hevc epochs other sessions still record but only its lineages reach (they restored them): a holder
lineage at such a node (``ReelCacheConditioningCacheBackend.hold_epochs``), released by a later collection once no live
boundary records it.

Durability (``durability`` / the handle's ``durable``): boundaries are durable as soon as they are reported; the
cache epochs of a request publish behind generation (``never``), when the request ends by a pause (``pause``, the
default) or always (``always``); ``checkpoint``, ``fork``, ``pause`` of an idle session, ``close`` and ``gc`` drain.

Integrity (CR-17-1; config ``verify_reads``, ``audit_on_open``, ``on_open_error``; defaults
``reelcache_backend.INTEGRITY_DEFAULTS``): every archived object a request reads (boundary frames, cache packets, RAW
packets) is hash-verified; a damaged one fails the request with a typed ``integrity`` error before any output, the
damaged epoch is quarantined and the session's archive is unchanged. Opening the store checks heads at index cost (no
hash audit of every stored byte). Damage found at attach -- a cache lineage or RAW branch the runtime quarantined, a
lineage path, session record, boundary chain or named checkpoint that does not read back -- does not stop the worker:
the damaged session is listed with ``status: "quarantined"`` (``list`` and ``describe`` report it, ``delete``
removes it, every other op fails with ``integrity``); epochs a damaged archive may hold are quarantined (a request that
needs one fails with ``integrity``; never a silent rebuild); every other session works. A fork or rewind request that
fails with a damaged archive before its first new boundary is rolled back. While a session record is damaged, garbage
collection keeps every RAW branch and holder lineage (it cannot tell which ones the damaged session reaches) and the
store keeps damaged graphs. ``on_open_error: "raise"`` restores the old behavior (the first damaged record fails the
attach); ``verify_reads: false`` the old unverified reads (a corrupt packet may then restore into wrong output).
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
import torch

from . import reelcache_backend as mb
from . import raw_tiers as rt
from .conditioning_cache import ConditioningCacheEpoch, register_cache_backend, unregister_cache_backend
from .session_archive import (
    CHECKPOINT_PREFIX,
    NAME_PATTERN,
    SESSION_PREFIX,
    ArchiveScan,
    Boundary,
    SessionArchive,
    SessionArchiveError,
    check_session_id,
)

PROVIDER_SPEC = "reelcache_sglang_models.longcat_video.session_provider:create_provider"
CACHE_BACKEND_NAME = "reelcache-longcat-sessions"
SESSION_RUN_KEY = "longcat_session_run"
INFLIGHT_OPS = ("capabilities", "list", "describe", "request", "pause", "archive_stats")
GENERATION_ACTIONS = ("generate", "resume", "fork", "rewind")
_HANDLE_KEYS = {"session_id", "action", "pause_after_chunks", "from_session", "checkpoint", "replace", "durable",
                "output", "decode_context_latents", "overlap_restore"}
_SEGMENT_FIELDS = ("cache_source", "cache_build_seconds", "cache_restore_seconds", "cond_frames_sha256",
                   "final_latents_sha256", "frames_uint8_sha256", "generator_state", "latent_prep_seconds",
                   "time_to_step0_end_seconds", "decode_seconds", "latents_source", "planning_deferral_wait_seconds",
                   "cache_restore_overlapped", "cache_restore_issue_seconds", "cache_restore_exposed_seconds")
PROMPT_CHANGE = {
    "supported": True, "mode": "new_segment", "request_fields": ["prompt"],
    "reason": "LongCat-Video continues a video segment by segment from the last 13 decoded frames; a resume, fork or "
              "rewind under a new prompt generates its new segments under that prompt from the archived boundary "
              "(the conditioning cache depends only on the conditioning frames, so an archived boundary's cache is "
              "restored, not rebuilt)",
}

_ACTIVE: list = []
_LOG = logging.getLogger(__name__)

# Eager GPU planning that overlaps a segment's denoising (schedule "eager" without defer_planning), LongCat 720p, measured
# (artifacts/sglang_integration/sglang/step13_longcat/gpu_planner): 34.3 GiB allocated / 37.9 GiB reserved above the
# loaded model on the RTX PRO 6000 with an 8 GiB planning bound (denoising working set + one node's 10.6 GiB of sources
# + the bounded planner); on the A100-80GB (79.25 GiB, 37.4 GiB resident) the fitter ran out of memory and the
# session's lineage failed. Required headroom: a fixed part (with ~6 GiB for the two streams' allocator pools) plus a
# quarter of the planning bound (the bounded working set: Step 7, 1.9 GiB at 8 GiB, 5.5 GiB at 24 GiB).
EAGER_OVERLAP_FIXED_BYTES = 41 << 30


def eager_overlap(*, defer_planning: bool, plan_device: str, plan_memory_bytes: int | None, resident_bytes: int,
                  total_bytes: int) -> dict:
    """The startup memory check of eager GPU planning overlapping denoising (``applies`` only to a GPU planner without
    ``defer_planning``): ``fits`` when resident + :data:`EAGER_OVERLAP_FIXED_BYTES` + ``plan_memory_bytes``/4 is
    within the device."""
    record = {"applies": plan_device == "cuda" and not defer_planning,
              "resident_gib": resident_bytes / (1 << 30), "total_gib": total_bytes / (1 << 30)}
    if not record["applies"]:
        record["fits"] = True
        return record
    need = EAGER_OVERLAP_FIXED_BYTES + (plan_memory_bytes or 0) // 4
    record["headroom_gib"] = need / (1 << 30)
    record["fits"] = resident_bytes + need <= total_bytes
    return record


def check_eager_overlap(*, schedule: str, defer_planning: bool, plan_device: str, plan_memory_bytes: int | None,
                        resident_bytes: int, total_bytes: int) -> dict:
    """Refuse (``ConfigurationError``) eager GPU planning that overlaps denoising on a device without the measured
    headroom; ``defer_planning`` or the deferred schedule fit there. Returns the check's record."""
    from reelcache.errors import ConfigurationError

    record = eager_overlap(defer_planning=defer_planning, plan_device=plan_device, plan_memory_bytes=plan_memory_bytes,
                           resident_bytes=resident_bytes, total_bytes=total_bytes)
    record["applies"] = record["applies"] and schedule == "eager"
    if record["applies"] and not record["fits"]:
        raise ConfigurationError(
            f"schedule 'eager' without defer_planning overlaps GPU planning with denoising: it needs "
            f"{record['headroom_gib']:.1f} GiB above the {record['resident_gib']:.1f} GiB resident, but the device has "
            f"{record['total_gib']:.1f} GiB (LongCat 720p measured: the A100-80GB runs out of memory). Use "
            f"defer_planning: true or schedule 'deferred'")
    return record


def resolve_schedule(requested: str, *, defer_planning: bool, plan_device: str, plan_memory_bytes: int | None,
                     resident_bytes: int, total_bytes: int, representation: str = "svd_hevc",
                     raw_tier: bool = False) -> dict:
    """The archival schedule a worker runs (logged and reported at attach).

    ``auto`` (the default; coordinator decision 2026-10-16, reversible): ``eager`` for a GPU planner when it fits --
    with ``defer_planning`` always, otherwise when the eager-overlap memory check passes (RTX PRO 6000: durable pause
    ~0.06 s for one segment, +0.9 % step time) -- else ``deferred`` (A100-80GB, or a CPU planner). ``eager`` and
    ``deferred`` are taken as given (``eager`` without ``defer_planning`` is refused where the check fails).

    ``defer_planning`` (the effective value; CR-17-6): forced on for a GPU planner wherever the eager-overlap check
    does not fit, whatever the schedule -- a deferred schedule submits a request's epochs when it ends, and their GPU
    planning would otherwise overlap the next request's denoising (A100-80GB 4410238: the second interleaved session
    ran out of memory). With it, GPU planning never overlaps any request's memory-critical sections (denoising, VAE
    decoding; one gated planner per worker) and the planner's cached device memory is released when a section opens
    (``release_planner_memory``)."""
    measured = {"defer_planning": defer_planning, "plan_device": plan_device, "plan_memory_bytes": plan_memory_bytes,
                "resident_bytes": resident_bytes, "total_bytes": total_bytes}
    gate = eager_overlap(**{**measured, "defer_planning": False})
    forced = representation == "svd_hevc" and gate["applies"] and not gate["fits"]
    gating = {"defer_planning": bool(defer_planning or forced), "release_planner_memory": bool(forced),
              "defer_planning_reason": (
                  f"forced: GPU planner and the eager-overlap memory check does not fit "
                  f"({gate['resident_gib']:.1f} GiB resident + {gate['headroom_gib']:.1f} GiB > "
                  f"{gate['total_gib']:.1f} GiB): GPU planning never overlaps any request's denoising or decoding"
                  if forced else ("set explicitly" if defer_planning else "not needed (no GPU planner, or the "
                                                                          "eager-overlap check fits)"))}
    if requested != "auto":
        record = check_eager_overlap(schedule=requested, **measured)
        reason = f"schedule '{requested}' set explicitly"
        return {"requested": requested, "chosen": requested, "reason": reason, "eager_overlap": record, **gating}
    record = eager_overlap(**measured)
    if raw_tier and representation == "svd_hevc":
        how = ("with GPU planning gated out of every request's denoising and decoding (forced defer_planning)"
               if gating["defer_planning"] and forced else "behind the following segments' generation")
        return {"requested": requested, "chosen": "eager",
                "reason": f"auto (raw_tier, freeze D): every epoch is submitted for archival as soon as its cache is "
                          f"built, {how}; its raw copies serve restores meanwhile",
                "eager_overlap": record, **gating}
    if representation != "svd_hevc":
        chosen, reason = "deferred", (f"auto: representation '{representation}' plans nothing (its epochs are sealed "
                                      "when a request pauses or ends)")
    elif plan_device != "cuda":
        chosen, reason = "deferred", ("auto: the archive plans on the CPU (eager is chosen only for a GPU planner); "
                                      "a request's epochs are archived when it pauses or ends")
    elif defer_planning:
        chosen, reason = "eager", ("auto: GPU planner with defer_planning -- each epoch is planned between the "
                                   "memory-critical sections (no overlap with denoising or decoding)")
    elif record["fits"]:
        chosen, reason = "eager", (f"auto: GPU planner and the eager-overlap memory check fits ({record['resident_gib']:.1f}"
                                   f" GiB resident + {record['headroom_gib']:.1f} GiB <= {record['total_gib']:.1f} GiB): "
                                   "each epoch is planned while the next segment generates")
    else:
        chosen, reason = "deferred", (f"auto: GPU planner but the eager-overlap memory check does not fit "
                                      f"({record['resident_gib']:.1f} GiB resident + {record['headroom_gib']:.1f} GiB > "
                                      f"{record['total_gib']:.1f} GiB): a request's epochs are planned when it pauses "
                                      "or ends, never during any request's denoising or decoding (defer_planning)")
    return {"requested": requested, "chosen": chosen, "reason": reason, "eager_overlap": record, **gating}


def memory_critical_section(backend: Any = None):
    """The memory-critical section the port's stages open around denoising and VAE decoding (CR-17-6).

    The request's conditioning-cache backend's ``generation_section`` when it has one (session requests), and the
    worker's session provider's when that is another backend (one gated planner per worker: a plain request, or one
    naming another backend, never overlaps any session's archival planning either); entered in that order when the
    returned context manager is entered. None when there is neither."""
    sections = []
    section = getattr(backend, "generation_section", None) if backend is not None else None
    if section is not None:
        sections.append(section)
    provider = active_provider()
    if provider is not None and provider.cache_backend is not backend:
        sections.append(provider.cache_backend.generation_section)
    if not sections:
        return None
    if len(sections) == 1:
        return sections[0]()
    return _nested_sections(sections)


@contextlib.contextmanager
def _nested_sections(sections: list):
    with contextlib.ExitStack() as stack:
        for section in sections:
            stack.enter_context(section())
        yield


def active_provider() -> "LongCatSessionProvider | None":
    """The worker's provider (set by :func:`create_provider`), or None when sessions are not enabled."""
    return _ACTIVE[0] if _ACTIVE else None


def _control_error(code: str, message: str, **details):
    from reelcache.integrations.sglang.controls import SessionControlError

    return SessionControlError(code, message, **details)


def _jsonable(value):
    from reelcache.integrations.sglang.controls import jsonable

    return jsonable(value)


def _error_code(error: BaseException) -> str:
    """The typed control-error code of a request failure (``integrity`` for damaged archives)."""
    from reelcache.integrations.sglang.controls import error_of

    return error_of(error).code


# ----------------------------------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class LongCatSessionConfig:
    """``--reelcache-config`` of a LongCat worker (validated; unknown keys are refused).

    ``store_root`` (required): the ReelCache store of the session archive. ``representation``: ``svd_hevc`` (the latest
    method at the LongCat targets, default) or ``raw`` (exact reference). ``plan_device``: archival planning device,
    ``auto`` (default: the GPU when the worker has one, else the CPU), ``cuda`` or ``cpu``. ``plan_memory_bytes``:
    the planning/finalization working-set bound (``ArchiveOptions(plan_memory_bytes=...)``; default 8 GiB: a LongCat
    node then peaks at ~1.9 GiB above its sources, with the node of unbounded planning -- Step 7); GPU planning next
    to the served model requires a bound. ``schedule``: ``auto`` (default: ``eager`` for a GPU planner whose
    eager-overlap memory check fits -- or with ``defer_planning`` --, else ``deferred``; the choice and its reason are
    logged and reported at attach: :func:`resolve_schedule`), ``deferred`` (a request's epochs are planned when it
    ends, or when the capture slots are full) or ``eager`` (each epoch is submitted when captured, so it is planned
    while the following segments generate; a GPU planner then needs the bound, and without ``defer_planning`` the
    worker refuses to start on a device without the measured headroom: :func:`check_eager_overlap`).
    ``defer_planning``: GPU
    planning never overlaps the stage's denoising and VAE decode (``ArchivePipeline.defer_gpu_work``; it runs at
    segment boundaries and after the request), for every request of the worker (sessions and plain requests: one
    gated planner); forced on, with the planner's cached memory released at each section, wherever the
    eager-overlap check does not fit (CR-17-6: :func:`resolve_schedule`). A GPU planner receives the pinned host copies of the captured epochs
    and uploads one node's sources (11.35 GB) when it plans that node. ``durability``: ``pause`` | ``always`` |
    ``never`` (module
    docstring). ``prewarm``: ``open`` prepares the session's archived epochs (packs + bases: the prepared state).
    ``warmup``: process-level restore warm-up at attach (None: on CUDA). ``capture_slots``: pinned capture slots (a
    full set drains before the next capture; None: the freeze-D host budget, or 4 without the raw tier). ``energy_key``/``energy_value``: explicit targets (provisional; both or
    neither; default the K98/V95 profile). Integrity (CR-17-1; module docstring): ``verify_reads`` (default true:
    hash-verify every archived object read for use), ``audit_on_open`` (default false: check heads at index cost, no
    hash audit of every stored byte) and ``on_open_error`` (``quarantine``, default: a damaged head or session record
    is quarantined while everything else works; ``raise``: the attach fails). Cold-session staging (CR-17-4; CUDA
    workers): ``staging_engine`` (default true: the packed restorer stages cold histories with one Step 5c
    ``StagingEngine`` on the worker GPU -- bitwise the restore of the store path, which ``false`` selects),
    ``staging_threads`` / ``staging_prepare_threads`` (None: min(8, max(2, cpus // 2)) of the worker's CPU affinity and
    max(1, threads // 2)) and ``staging_pinned_bytes`` (the pinned ring, default 512 MiB); reported at attach and in
    ``capabilities.staging_engine`` (the SGLang bridge's names). ``planning_gate`` (CUDA workers; g7):
    ``group`` (default: the GPU planner's matrix groups are the gated units and a node's host sources go to the
    device one group at a time, so a memory-critical section waits for one group, not a node; identical plans)
    or ``node`` (g6: a section waits for the planner's whole node); reported at attach and in
    ``capabilities.archival.planning_gate``.
    Freeze D raw tiers (``raw_tiers.py``; Step 7 ``DESIGN_D_tiers`` names; :func:`plan_host_slots`): ``raw_tier``
    (default true: every epoch's raw copies serve restores -- ``gpu_raw`` / ``host_raw`` -- and ``schedule: auto``
    submits each epoch for archival when its cache is built), ``gpu_raw_budget_bytes`` (None: auto, whole epochs
    above the measured generation need: A100-80GB 0, RTX PRO 6000 1), ``host_raw_budget_bytes`` (None: auto, half of
    this worker's share of the cgroup memory headroom -- divided by ``raw_tier_workers``, default ``LOCAL_WORLD_SIZE``
    or 1 -- after its fixed host consumers, capped at 64 GiB; 0: no host tier), or ``capture_slots`` host slots
    instead; an explicit host budget above the worker's share is refused at attach. No minimum above the safe
    headroom: a budget below one epoch (10.57 GiB at 720p) disables the tier, and captures then use C3 back-pressure
    in the slots the share holds (none: conditioning-cache archival is off; recorded). ``host_raw_pinned``,
    ``raw_restore`` (``prefer`` | ``off`` = compressed only), ``encoder`` (``auto`` | ``x265`` | ``nvenc``: NVENC when
    the GPU has it, probed once at attach; explicit ``nvenc`` without it is refused). ``raw_tier: false`` restores C3
    exactly (``capture_slots`` default 4, unchecked).
    ``overlap_restore`` (freeze F5, d5; default true; the SGLang bridge's name): an svd_hevc restore of a session
    request runs layer-overlapped with the segment's first denoising step: the cache returns at once, a background
    thread restores it layer by layer (``LayerReadiness``), the DiT's first step waits per layer and the stage joins
    the restore after that step (a failure, typed, fails the request before any output). Bitwise the plain restore;
    the tier (prepared / cold) is labelled when the restore completes. Raw tiers (``gpu_raw`` / ``host_raw``), the
    RAW representation and context-parallel requests (``cp_tiers``) are plain. ``false`` = plain (d4). Precedence:
    a request's ``causal_kv_session.overlap_restore`` (bool), then an explicit
    ``ReelCacheConditioningCacheBackend.set_overlap`` (tools, harnesses, tests: the d4 mechanism), then this setting.
    Outside a request a restore is plain unless ``set_overlap`` was set. While it is effective the GPU raw budget's
    generation need gains ``raw_tiers.OVERLAP_RESTORE_NEED_BYTES``.
    """

    store_root: str
    representation: str = "svd_hevc"
    plan_device: str = "auto"
    plan_memory_bytes: int | None = 8 << 30
    schedule: str = "auto"
    defer_planning: bool = False
    durability: str = "pause"
    prewarm: bool = True
    overlap_restore: bool = True
    warmup: bool | None = None
    capture_slots: int | None = None
    energy_key: float | None = None
    energy_value: float | None = None
    restore_arena_bytes: int = 4 << 30
    basis_budget_bytes: int = 2 << 30
    prepared_budget_bytes: int = 4 << 30
    history: int = 256
    verify_reads: bool = mb.INTEGRITY_DEFAULTS["verify_reads"]
    audit_on_open: bool = mb.INTEGRITY_DEFAULTS["audit_on_open"]
    on_open_error: str = mb.INTEGRITY_DEFAULTS["on_open_error"]
    staging_engine: bool = mb.STAGING_DEFAULTS["staging_engine"]
    staging_threads: int | None = mb.STAGING_DEFAULTS["staging_threads"]
    staging_prepare_threads: int | None = mb.STAGING_DEFAULTS["staging_prepare_threads"]
    staging_pinned_bytes: int = mb.STAGING_DEFAULTS["staging_pinned_bytes"]
    planning_gate: str = "group"
    raw_tier: bool = True
    gpu_raw_budget_bytes: int | None = None
    host_raw_budget_bytes: int | None = None
    host_raw_pinned: bool = True
    raw_restore: str = "prefer"
    encoder: str = "auto"
    raw_tier_workers: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LongCatSessionConfig":
        from reelcache.errors import ConfigurationError

        if not isinstance(data, Mapping):
            raise ConfigurationError("the LongCat session provider config must be a mapping")
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ConfigurationError(f"unknown LongCat session provider config keys {sorted(unknown)}; "
                                     f"known: {sorted(known)}")
        if not isinstance(data.get("store_root"), str) or not data["store_root"]:
            raise ConfigurationError("store_root (a directory for the session archive) is required")
        config = cls(**dict(data))
        if config.representation not in ("svd_hevc", "raw"):
            raise ConfigurationError("representation must be 'svd_hevc' or 'raw'")
        if config.plan_device not in ("auto", "cpu", "cuda"):
            raise ConfigurationError("plan_device must be 'auto', 'cpu' or 'cuda'")
        if config.plan_memory_bytes is not None and (type(config.plan_memory_bytes) is not int
                                                     or config.plan_memory_bytes < 1):
            raise ConfigurationError("plan_memory_bytes must be None or a positive byte count")
        if config.plan_device == "cuda" and config.plan_memory_bytes is None:
            raise ConfigurationError("GPU planning next to the served model needs plan_memory_bytes (unbounded "
                                     "LongCat planning needs ~35 GiB above the sources)")
        if config.schedule not in ("auto", "deferred", "eager"):
            raise ConfigurationError("schedule must be 'auto', 'deferred' or 'eager'")
        if type(config.defer_planning) is not bool:
            raise ConfigurationError("defer_planning must be a bool")
        if config.defer_planning and config.plan_device == "cpu":
            raise ConfigurationError("defer_planning applies to GPU planning (plan_device 'cuda' or 'auto')")
        if config.durability not in ("pause", "always", "never"):
            raise ConfigurationError("durability must be 'pause', 'always' or 'never'")
        for name in ("prewarm", "verify_reads", "audit_on_open", "staging_engine", "raw_tier", "host_raw_pinned",
                     "overlap_restore"):
            if type(getattr(config, name)) is not bool:
                raise ConfigurationError(f"{name} must be a bool")
        for name in ("staging_threads", "staging_prepare_threads"):
            value = getattr(config, name)
            if value is not None and (type(value) is not int or not 1 <= value <= 64):
                raise ConfigurationError(f"{name} must be None or an integer in [1, 64]")
        if type(config.staging_pinned_bytes) is not int or config.staging_pinned_bytes < 1 << 20:
            raise ConfigurationError("staging_pinned_bytes must be an integer of at least 1 MiB")
        if config.planning_gate not in ("group", "node"):
            raise ConfigurationError("planning_gate must be 'group' or 'node'")
        if config.raw_restore not in mb.RAW_RESTORE_MODES:
            raise ConfigurationError(f"raw_restore must be one of {mb.RAW_RESTORE_MODES}")
        if config.encoder not in mb.ENCODERS:
            raise ConfigurationError(f"encoder must be one of {mb.ENCODERS}")
        for name in ("gpu_raw_budget_bytes", "host_raw_budget_bytes"):
            value = getattr(config, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ConfigurationError(f"{name} must be None or a nonnegative byte count")
        if config.raw_tier_workers is not None and (type(config.raw_tier_workers) is not int
                                                    or not 1 <= config.raw_tier_workers <= 64):
            raise ConfigurationError("raw_tier_workers must be None or an integer in [1, 64]")
        if config.capture_slots is not None and (type(config.capture_slots) is not int or config.capture_slots < 1):
            raise ConfigurationError("capture_slots must be None or a positive int")
        if config.capture_slots is not None and config.host_raw_budget_bytes is not None:
            raise ConfigurationError("give capture_slots or host_raw_budget_bytes, not both")
        if config.on_open_error not in mb.OPEN_ERROR_POLICIES:
            raise ConfigurationError(f"on_open_error must be one of {mb.OPEN_ERROR_POLICIES}")
        if config.warmup is not None and type(config.warmup) is not bool:
            raise ConfigurationError("warmup must be a bool or None")
        for name in ("history", "restore_arena_bytes", "basis_budget_bytes", "prepared_budget_bytes"):
            if type(getattr(config, name)) is not int or getattr(config, name) < 1:
                raise ConfigurationError(f"{name} must be a positive int")
        if (config.energy_key is None) != (config.energy_value is None):
            raise ConfigurationError("energy_key and energy_value are given together")
        return config

    def method(self):
        if self.energy_key is None:
            return mb.default_method()
        return mb.diagnostic_method(float(self.energy_key), float(self.energy_value))

    def c3_slots(self) -> int:
        """Host capture slots without the raw tier (C3: ``capture_slots``, default 4)."""
        return 4 if self.capture_slots is None else self.capture_slots

    def integrity(self) -> dict:
        """The store's integrity settings (``reelcache_backend.open_runtime`` keywords)."""
        return {"verify_reads": self.verify_reads, "audit_on_open": self.audit_on_open,
                "on_open_error": self.on_open_error}

    def staging(self) -> dict:
        """The restorer's cold-session staging settings (``reelcache_backend.lineage_restorer`` keywords)."""
        return {"staging_engine": self.staging_engine, "staging_threads": self.staging_threads,
                "staging_prepare_threads": self.staging_prepare_threads,
                "staging_pinned_bytes": self.staging_pinned_bytes}

    def resolved_plan_device(self) -> str:
        """``auto`` -> ``cuda`` when the worker has a CUDA device, else ``cpu`` (explicit errors for impossible
        combinations)."""
        from reelcache.errors import ConfigurationError

        device = self.plan_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            if not torch.cuda.is_available():
                raise ConfigurationError("plan_device 'cuda' needs a CUDA device")
            if self.plan_memory_bytes is None:
                raise ConfigurationError("GPU planning next to the served model needs plan_memory_bytes")
        if self.defer_planning and device != "cuda":
            raise ConfigurationError("defer_planning applies to GPU planning; this worker plans on the CPU")
        return device


def _completion_tier(note: Mapping[str, Any], joined: Any, last: Any) -> dict:
    """A segment's cache note with the tier of an overlapped restore resolved (d5): it is labelled when the restore
    completes, so the note taken when it was issued says ``tier_pending``; the stage's join record
    (``overlapped_restore``) has it, else the backend's completed ``last_restore_tier``. Unresolved: recorded as
    ``tier_pending``."""
    note = dict(note)
    if not note.pop("tier_pending", False):
        return note
    for source in (joined, last):
        if isinstance(source, Mapping) and source.get("tier") and not source.get("pending"):
            note.update(tier=source["tier"], tier_bytes=source.get("bytes"), tier_seconds=source.get("seconds"))
            return note
    note["tier_pending"] = True
    return note


# ----------------------------------------------------------------------------------------------- session state
@dataclass
class _OpenSession:
    session_id: str
    record_ref: str
    created: float
    source: dict | None
    head: Boundary | None  # None only while a generate request has not stored boundary 0 yet
    status: str = "idle"  # idle | running
    pause_requested: bool = False
    archive_failures: list = field(default_factory=list)
    history: collections.deque = field(default_factory=lambda: collections.deque(maxlen=256))
    requests: collections.deque = field(default_factory=lambda: collections.deque(maxlen=64))

    def event(self, name: str, **fields) -> None:
        self.history.append({"event": name, "time": time.time(), **fields})


@dataclass
class RequestPlan:
    """What the port's stages run for one session request."""

    action: str
    session_id: str
    start_segment: int
    num_segments: int
    frames: np.ndarray | None  # boundary conditioning frames (None: generate, from the request)
    generator_state: str | None  # None: generate, from the request's seed
    geometry: dict | None


class _SessionCacheBackend:
    """The port's conditioning-cache backend for session requests: restore published epochs, build otherwise; a
    quarantined or damaged archive fails the request with a typed IntegrityError (never a silent rebuild). An svd_hevc
    restore within a request overlaps the first denoising step when the request's choice says so
    (:meth:`LongCatSessionProvider.overlap_for`; d5); outside a request only when ``set_overlap`` was set."""

    def __init__(self, provider: "LongCatSessionProvider"):
        self.provider = provider

    def restore(self, epoch: ConditioningCacheEpoch):
        provider = self.provider
        backend, representation = provider.backend, provider.config.representation
        run = provider._run
        if backend.raw_tier and backend.raw_restore == "prefer":
            cache = backend.restore_raw_tier(epoch)  # freeze D: gpu_raw / host_raw, bitwise the native build
            if cache is not None:
                if run is not None:
                    run.note_cache(epoch, "restore", backend.last_restore_tier["tier"], tier=backend.last_restore_tier)
                return cache
        state = backend.archive_state(epoch, representation)
        if state == "quarantined":
            if run is not None:
                run.note_cache(epoch, "quarantined", representation)
            backend.lookup(epoch, representation)  # raises the typed IntegrityError with the recorded damage
        if state != "published":
            if run is not None:
                run.note_cache(epoch, "build", "not archived" if state is None else state)
            return None
        backend.arm_restore(representation)
        try:
            cache = backend.restore(epoch, overlapped=provider.overlap_for(run))
        finally:
            backend.arm_restore(None)
        if run is not None:
            run.note_cache(epoch, "restore", representation, tier=backend.last_restore_tier)
        return cache

    def generation_section(self):
        """The stage's denoising / decode section (GPU planning deferred when configured)."""
        return self.provider.backend.generation_section()

    def on_cache_built(self, cache) -> None:
        backend = self.provider.backend
        if backend.archive and backend.free_capture_slots() < 1:
            started = time.perf_counter()
            if backend.raw_tier:
                # freeze D last resort: every host slot holds an unpublished epoch; wait for the oldest to publish
                # (outside sections: capture precedes the denoising section)
                cause = backend.make_room()["cause"] or "host_full"
            else:
                backend.flush(wait=True)  # C3 backpressure: captures are never dropped
                cause = "c3_bounds"
            run = self.provider._run
            if run is not None:
                run.summary["backpressure_seconds"] += time.perf_counter() - started
                causes = run.summary.setdefault("backpressure_causes", {})
                causes[cause] = causes.get(cause, 0) + 1
        backend.on_cache_built(cache)

    def cache_released(self, epoch: ConditioningCacheEpoch) -> None:
        """The stage dropped the segment's cache (after denoising): retained GPU raw follows the budget."""
        self.provider.backend.cache_released(epoch)


def request_fields(batch: Any) -> dict:
    """Plain request fields a session request is checked against (from an SGLang ``Req``)."""
    def single(value):
        return value[0] if isinstance(value, (list, tuple)) and len(value) == 1 else value

    inputs = getattr(batch, "condition_inputs", None) or {}
    embeds = inputs.get("prompt_embeds")
    embeds_sha = None
    if embeds is not None:
        from .conditioning_cache import tensor_sha256

        embeds_sha = tensor_sha256(embeds)
    return {"request_id": getattr(batch, "request_id", None), "prompt": single(getattr(batch, "prompt", None)),
            "prompt_embeds_sha256": embeds_sha,
            "seed": single(getattr(batch, "seed", None)), "num_segments": getattr(batch, "num_segments", 1),
            "num_frames": getattr(batch, "num_frames", None), "session_id": getattr(batch, "session_id", None),
            "start_segment_index": getattr(batch, "start_segment_index", 0),
            "generator_state": getattr(batch, "generator_state", None),
            "conditioning_cache_backend": getattr(batch, "conditioning_cache_backend", None),
            "has_client_frames": bool(single(getattr(batch, "video_path", None)) is not None
                                      or inputs.get("cond_frames") is not None),
            "injected": sorted(key for key in ("initial_latents", "cond_latents") if inputs.get(key) is not None)}


class SessionRun:
    """One session request between :meth:`LongCatSessionProvider.begin_request` and :meth:`finish`."""

    def __init__(self, provider: "LongCatSessionProvider", session: _OpenSession, plan: RequestPlan, *,
                 request_id: str, stop_at: int, durable: bool | None, prompt: str | None,
                 prompt_embeds_sha256: str | None = None):
        self.provider, self.session, self.plan = provider, session, plan
        self.request_id, self.stop_at, self.durable, self.prompt = request_id, stop_at, durable, prompt
        self.prompt_embeds_sha256 = prompt_embeds_sha256
        self.geometry = plan.geometry
        self.finished = False
        self.paused_by: str | None = None
        self.ends_by_pause = False
        self.rollback: dict | None = None  # fork/rewind undo when a damaged archive fails it before its first boundary
        self.overlap_restore: bool | None = None  # the handle's override (None: the config's overlap_restore)
        self._cache_notes: dict[int, dict] = {}
        self.summary = {"request_id": request_id, "session_id": session.session_id, "action": plan.action,
                        "start_segment": plan.start_segment, "requested_segments": plan.num_segments,
                        "stop_at": stop_at, "segments": [], "dispatched_at": time.time(), "backpressure_seconds": 0.0,
                        "boundary_commit_seconds": 0.0, "hook_seconds": 0.0}

    # -- called by the stages --------------------------------------------------------------------------------
    def bind_request(self, *, frames: np.ndarray, generator_state: str, geometry: Mapping[str, Any]) -> None:
        """After the input stage built the request: ``generate`` stores boundary 0 (the request's conditioning
        frames and the seed's generator state) and creates the session; other actions check the geometry."""
        geometry = dict(geometry)
        if self.plan.action == "generate":
            provider, sid = self.provider, self.session.session_id
            boundary = provider.archive.put_boundary(segments=0, frames=frames, generator_state=generator_state,
                                                     geometry=geometry, parent=None, segment=None, session_origin=sid)
            record = provider.archive.put_session(sid, head=boundary.ref, expected=None, created=self.session.created)
            self.session.record_ref, self.session.head = record["ref"], boundary
            self.session.event("created", head=boundary.ref)
        elif geometry != self.geometry:
            raise _control_error("invalid_argument", f"the request's geometry {geometry} differs from the session "
                                                     f"boundary's {self.geometry} (same resolution and frame counts)")
        self.geometry = geometry

    def note_cache(self, epoch: ConditioningCacheEpoch, source: str, detail: str,
                   tier: Mapping[str, Any] | None = None) -> None:
        note = {"provider_cache": source, "provider_cache_detail": detail}
        if tier and tier.get("pending"):
            note["tier_pending"] = True  # an overlapped restore: labelled when it completes (end_segment)
        elif tier:
            note.update(tier=tier.get("tier"), tier_bytes=tier.get("bytes"), tier_seconds=tier.get("seconds"))
        self._cache_notes[epoch.segment_index] = note

    def end_segment(self, *, segment_index: int, record: Mapping[str, Any], next_frames: np.ndarray,
                    generator_state: str, prompt: str | None = None) -> bool:
        """Commit boundary ``segment_index + 1`` and apply in-flight controls; True = stop the request here."""
        provider, session = self.provider, self.session
        if segment_index != session.head.segments:
            raise _control_error("internal", f"segment {segment_index} reported at boundary {session.head.segments}")
        started = time.perf_counter()
        prompt = self.prompt if prompt is None else prompt
        epoch = dict(record.get("epoch") or {})
        identity = (prompt or "") + ("" if self.prompt_embeds_sha256 is None else "|embeds:" + self.prompt_embeds_sha256)
        note = _completion_tier(self._cache_notes.pop(segment_index, {}), record.get("overlapped_restore"),
                                provider.backend.last_restore_tier)
        segment = {"index": segment_index, "prompt": prompt,
                   "prompt_sha256": hashlib.sha256(identity.encode()).hexdigest(),
                   "prompt_embeds_sha256": self.prompt_embeds_sha256,
                   "request_id": self.request_id, "epoch": epoch,
                   "epoch_digest": str(epoch.get("cond_latents_sha256", ""))[:16],
                   **{key: record.get(key) for key in _SEGMENT_FIELDS if record.get(key) is not None},
                   **note}
        if epoch:
            state = provider.backend.archive_state(ConditioningCacheEpoch.from_dict(epoch),
                                                   provider.config.representation)
            segment["cache_archive"] = state or "not archived"
        boundary = provider.archive.put_boundary(segments=segment_index + 1, frames=next_frames,
                                                 generator_state=generator_state, geometry=self.geometry,
                                                 parent=session.head.ref, segment=segment,
                                                 session_origin=session.session_id)
        try:
            record_out = provider.archive.put_session(session.session_id, head=boundary.ref,
                                                      expected=session.record_ref, created=session.created,
                                                      source=session.source, prompt_sha256=segment["prompt_sha256"])
        except SessionArchiveError as exc:
            raise _control_error("illegal_transition", str(exc)) from exc
        session.record_ref, session.head = record_out["ref"], boundary
        session.event("segment", segments=boundary.segments, head=boundary.ref, request_id=self.request_id)
        self.summary["boundary_commit_seconds"] += time.perf_counter() - started
        self.summary["segments"].append({"index": segment_index, "boundary": boundary.ref,
                                         **{key: segment.get(key) for key in (
                                             "cache_source", "provider_cache", "provider_cache_detail",
                                             "cache_archive", "cache_build_seconds", "cache_restore_seconds",
                                             "time_to_step0_end_seconds", "epoch_digest", "final_latents_sha256",
                                             "planning_deferral_wait_seconds", "tier", "tier_bytes",
                                             "tier_seconds", "cache_restore_overlapped",
                                             "cache_restore_issue_seconds", "cache_restore_exposed_seconds")},
                                         **({"tier_pending": True} if segment.get("tier_pending") else {})})
        hook = provider._hook
        if hook is not None:
            started = time.perf_counter()
            provider._in_hook = True
            try:
                hook()
            finally:
                provider._in_hook = False
            self.summary["hook_seconds"] += time.perf_counter() - started
        if session.pause_requested:
            self.paused_by = "control"
            return True
        if boundary.segments >= self.stop_at:
            if self.ends_by_pause:
                self.paused_by = "pause_after_chunks"
            return True
        return False

    def finish(self, error: BaseException | None = None) -> dict:
        """End the request: the session is idle again; archival per durability (module docstring)."""
        if self.finished:
            return self.summary
        self.finished = True
        provider, session = self.provider, self.session
        if provider.backend.raw_tier:
            self.summary["gpu_raw_released"] = provider.backend.release_resident()
        session.status = "idle"
        session.pause_requested = False
        provider._run = None
        summary = self.summary
        if session.head is None:  # a generate request that failed before storing boundary 0: nothing persisted
            provider._open.pop(session.session_id, None)
            summary.update(finished_at=time.time(), generated_segments=0, head=None, paused_by=None, durable=False,
                           archival="none", error=None if error is None else f"{type(error).__name__}: {error}")
            if error is not None:
                summary["error_code"] = _error_code(error)
            provider._remember(summary)
            return summary
        if error is not None and self.rollback is not None and not summary["segments"] and mb.is_damage(error):
            summary["rolled_back"] = provider._roll_back(self.rollback)
        summary.update(finished_at=time.time(), generated_segments=session.head.segments, head=session.head.ref,
                       paused_by=self.paused_by, error=None if error is None else f"{type(error).__name__}: {error}")
        if error is not None:
            summary["error_code"] = _error_code(error)
        policy = provider.config.durability
        durable = self.durable if self.durable is not None else (
            policy == "always" or (policy == "pause" and self.paused_by is not None))
        summary["durable"] = durable
        started = time.perf_counter()
        try:
            provider.backend.flush(wait=durable)
            summary["archival"] = "published" if durable else "behind generation"
        except Exception as exc:  # noqa: BLE001 - recorded; surfaced to durable requests and in describe/pause
            session.archive_failures.append({"request_id": self.request_id, "error": f"{type(exc).__name__}: {exc}"})
            summary["archival"] = f"failed: {type(exc).__name__}: {exc}"
            if durable and error is None:
                provider._remember(summary)
                raise _control_error("archival_failure", f"archival of session {session.session_id!r} failed: {exc}",
                                     request_id=self.request_id) from exc
        summary["archival_wait_seconds"] = time.perf_counter() - started
        if provider.backend.raw_tier:
            summary["prepare_on_evict"] = provider.backend.prepare_evicted()  # idle point (freeze D)
        session.event("request_finished", request_id=self.request_id, segments=session.head.segments,
                      paused_by=self.paused_by, error=summary["error"])
        provider._remember(summary)
        return summary


# ----------------------------------------------------------------------------------------------- provider
class LongCatSessionProvider:
    """Step 14 worker backend of the LongCat port (module docstring)."""

    observes_plain_requests = False

    def __init__(self, config: LongCatSessionConfig, *, backend: mb.ReelCacheConditioningCacheBackend,
                 owned: tuple = (), startup: Mapping[str, Any] | None = None):
        self.config = config
        self.backend = backend
        self.runtime = backend.runtime
        self.archive = SessionArchive(self.runtime.store)
        self._owned = tuple(owned)
        self._hook: Callable[[], None] | None = None
        self._in_hook = False
        self._open: dict[str, _OpenSession] = {}
        self._run: SessionRun | None = None
        self._requests: collections.OrderedDict[str, dict] = collections.OrderedDict()
        self._closed = False
        self.startup = dict(startup or {})
        # damaged session records / named checkpoints of the last scan: ref name -> error (CR-17-1 quarantine)
        self._damaged: dict[str, str] = {}
        self._owners: dict[tuple, set] = {}
        started = time.perf_counter()
        epochs = self._live_epochs()
        self.startup["reindexed"] = backend.reload_index(live_epochs=epochs, owners=self._owners)
        self.startup["reindexed_by_representation"] = dict(backend.last_reload or {})
        self.startup["reindex_seconds"] = time.perf_counter() - started
        self.startup["sessions"] = len(self.archive.session_ids())
        self.startup["integrity"] = self._integrity()
        if self._damaged or any(self.startup["integrity"]["damage"].values()):
            _LOG.warning("LongCat session provider: damaged archive quarantined at attach (the other sessions work): "
                         "%s", self.startup["integrity"]["damage"])
        self.overlap_restore = self._overlap_record()
        self.startup["overlap_restore"] = dict(self.overlap_restore)
        self.cache_backend = _SessionCacheBackend(self)
        register_cache_backend(CACHE_BACKEND_NAME, self.cache_backend, replace=True)
        self._ops = {
            "capabilities": (self._capabilities, set()), "list": (self._list, set()),
            "describe": (self._describe, {"session_id", "history"}), "checkpoints": (self._checkpoints, {"session_id"}),
            "request": (self._request, {"request_id"}), "open": (self._open_op, {"session_id"}),
            "pause": (self._pause, {"session_id"}), "checkpoint": (self._checkpoint, {"session_id", "name", "replace"}),
            "fork": (self._fork_op, {"session_id", "target", "checkpoint"}),
            "rewind": (self._rewind_op, {"session_id", "checkpoint"}), "close": (self._close_op, {"session_id", "mode"}),
            "delete": (self._delete_op, {"session_id", "checkpoint", "gc"}), "gc": (self._gc_op, set()),
            "archive_stats": (self._archive_stats, {"session_id"}),
            "drop_raw": (self._drop_raw, {"session_id", "drain"}), "demote_raw": (self._demote_raw, {"session_id"}),
            "reset_raw_peaks": (self._reset_raw_peaks, set()),
        }

    def _overlap_record(self) -> dict:
        """The session overlap (``overlap_restore``, d5): configured, effective, and why."""
        configured = bool(self.config.overlap_restore)
        if self.config.representation != "svd_hevc":
            return {"configured": configured, "effective": False,
                    "reason": "representation raw: RAW restores are plain"}
        if not configured:
            return {"configured": False, "effective": False,
                    "reason": "overlap_restore false: session restores are plain (a session handle's "
                              "overlap_restore=true overrides it per request)"}
        return {"configured": True, "effective": True,
                "reason": "svd_hevc session restores run layer-overlapped with the segment's first denoising step "
                          "(a session handle's overlap_restore overrides it per request); raw tiers, restores outside "
                          "a request and context-parallel requests (cp_tiers) are plain"}

    def overlap_for(self, run: "SessionRun | None") -> bool | None:
        """The ``overlapped`` argument of an svd_hevc restore. Within a session request: the handle's
        ``overlap_restore``; else None when the backend's overlap was set explicitly (``set_overlap``: tools,
        harnesses, tests -- the d4 mechanism); else the config's (effective). Outside a request: None -- the explicit
        setting, else plain (tools and timing harnesses call ``cache_backend.restore`` with nothing waiting per layer
        or joining it)."""
        if run is None:
            return None
        if run.overlap_restore is not None:
            return bool(run.overlap_restore) and self.config.representation == "svd_hevc"
        if getattr(self.backend, "overlap", None) is not None:
            return None
        return bool(self.overlap_restore["effective"])

    def overlap_effective(self, run: "SessionRun | None") -> bool:
        """Whether this request's svd_hevc restores overlap (R ``overlap_restore``)."""
        if self.config.representation != "svd_hevc":
            return False
        choice = self.overlap_for(run)
        return bool(getattr(self.backend, "overlap", None)) if choice is None else choice

    # -- worker interface ------------------------------------------------------------------------------------
    @property
    def ops(self) -> tuple[str, ...]:
        return tuple(self._ops)

    def memory_critical_section(self, reason: str, request_id: str | None = None):
        """SGLang's memory-critical seam (``causal_kv_memory_critical_section``, branch bb6f7c5116): GPU planning is
        deferred there when ``defer_planning`` is on (the port's own stages use the same section around denoising
        and decoding)."""
        return self.backend.generation_section()

    def set_boundary_hook(self, hook: Callable[[], None] | None) -> None:
        if hook is not None and not callable(hook):
            raise TypeError("the boundary hook must be callable")
        self._hook = hook

    def control(self, op: Any, args: Any = None) -> dict:
        """A session control between requests (typed errors: :class:`SessionControlError`)."""
        if self._run is not None and not self._in_hook:
            # the scheduler serves controls only between requests: a run still open here was abandoned
            self._run.finish(error=RuntimeError("the request ended before its session run finished"))
        return self._dispatch(op, args)

    def inflight_control(self, op: Any, args: Any = None) -> dict | None:
        """A control received at a segment boundary of a running request: read-only ops and the pause of the
        running session are served now; None = run it after the request."""
        if op not in INFLIGHT_OPS:
            return None
        if op == "pause":
            if not isinstance(args, Mapping):
                return self._dispatch(op, args)
            session = self._open.get(args.get("session_id")) if isinstance(args.get("session_id"), str) else None
            if session is None or session.status != "running":
                return None
        return self._dispatch(op, args)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors = []
        try:
            self.backend.flush(wait=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        unregister_cache_backend(CACHE_BACKEND_NAME)
        if self in _ACTIVE:
            _ACTIVE.remove(self)
        for item in self._owned:
            try:
                item.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise errors[0]

    # -- dispatch ----------------------------------------------------------------------------------------------
    def _dispatch(self, op: Any, args: Any) -> dict:
        from reelcache.errors import IntegrityError
        from reelcache.integrations.sglang.controls import SessionControlError, error_of

        if not isinstance(op, str) or op not in self._ops:
            raise _control_error("invalid_argument", f"unknown session control op {op!r}; one of {sorted(self._ops)}")
        args = {} if args is None else args
        if not isinstance(args, Mapping):
            raise _control_error("invalid_argument", "control arguments must be a JSON object")
        handler, allowed = self._ops[op]
        unknown = set(args) - allowed
        if unknown:
            raise _control_error("invalid_argument", f"unknown arguments {sorted(unknown)}")
        started = time.perf_counter()
        try:
            result = handler(args)
        except SessionControlError:
            raise
        except SessionArchiveError as exc:
            if isinstance(exc, IntegrityError):  # a damaged stored record (SessionArchiveIntegrityError)
                raise error_of(exc) from exc
            raise _control_error("illegal_transition", str(exc)) from exc
        except BaseException as exc:  # noqa: BLE001 - every failure leaves as a typed control error
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise error_of(exc) from exc
        return _jsonable({"op": op, **result, "control_seconds": time.perf_counter() - started})

    # -- helpers ------------------------------------------------------------------------------------------------
    @staticmethod
    def _sid(args: Mapping[str, Any], key: str = "session_id") -> str:
        try:
            return check_session_id(args.get(key))
        except SessionArchiveError as exc:
            raise _control_error("invalid_argument", f"{key}: {exc}") from None

    @staticmethod
    def _name(value: Any, label: str) -> str:
        if not isinstance(value, str) or NAME_PATTERN.fullmatch(value) is None:
            raise _control_error("invalid_argument", f"{label} must match {NAME_PATTERN.pattern} (names, never "
                                                     f"paths); got {value!r}")
        return value

    def _exists(self, sid: str) -> bool:
        return sid in self._open or SESSION_PREFIX + sid in self._damaged or self.archive.session(sid) is not None

    # -- integrity (CR-17-1) -------------------------------------------------------------------------------------
    def _quarantined(self, sid: str) -> str | None:
        """Why session ``sid`` is quarantined (its record or boundary chain is damaged), else None."""
        return self._damaged.get(SESSION_PREFIX + sid)

    def _guard(self, sid: str, op: str) -> None:
        reason = self._quarantined(sid)
        if reason is not None:
            raise _control_error("integrity", f"LongCat session {sid!r} is quarantined: its archive is damaged "
                                              f"({reason}); {op} is refused (list, describe and delete apply)",
                                 session_id=sid, quarantined=reason)

    def _session_damage(self, sid: str) -> dict:
        """Damage that concerns one session: its record/chain, named checkpoints, cache lineages, epochs (the
        quarantined epochs its boundaries or named checkpoints record; read only when some epoch is quarantined)."""
        prefix = f"{CHECKPOINT_PREFIX}{sid}/"
        lineages = set(self.backend.session_branches(sid))
        record = self._quarantined(sid)
        epochs = []
        if self.backend.quarantined_epochs():
            keys = {key for key, owners in self._owners.items() if sid in owners}
            if record is None:
                try:
                    keys |= {mb.epoch_key(epoch) for epoch in self._segment_epochs(self._head(sid)).values()}
                except Exception:  # noqa: BLE001 - a session being created, or damaged since the scan
                    pass
            epochs = self.backend.quarantined_epochs(keys)
        damage = {"record": record,
                  "checkpoints": {name[len(prefix):]: error for name, error in self._damaged.items()
                                  if name.startswith(prefix)},
                  "lineages": {branch: error for branch, error in self.backend.damaged_lineages.items()
                               if branch in lineages},
                  "epochs": epochs}
        return {key: value for key, value in damage.items() if value}

    def _staging(self) -> dict:
        """Cold-session staging of the restorer (CR-17-4): the attach report plus the engine's counters now."""
        restorer = self.backend.restorer
        engine = getattr(restorer, "staging_engine", None)
        report = self.startup.get("staging_engine")
        if report is None:
            report = mb.staging_description(engine, reason=None if hasattr(restorer, "staging_engine") else (
                f"{type(restorer).__name__}: no packed device restore to stage"))
        report = dict(report)
        if engine is not None and callable(getattr(engine, "stats", None)):
            stats = engine.stats()
            report["stats"] = {key: stats.get(key) for key in ("histories", "packs", "tips", "tip_bytes",
                                                                "payload_bytes", "seconds")}
        return report

    def _integrity(self) -> dict:
        """Integrity settings and the damage known now (startup report and ``capabilities``)."""
        report = self.backend.damage_report()
        return {**self.backend.integrity,
                "damage": {"sessions": {name[len(SESSION_PREFIX):]: error for name, error in self._damaged.items()
                                        if name.startswith(SESSION_PREFIX)},
                           "checkpoints": {name: error for name, error in self._damaged.items()
                                           if not name.startswith(SESSION_PREFIX)},
                           "runtime_quarantined": report["runtime_quarantined"],
                           "lineages": report["damaged_lineages"], "raw_branches": report["damaged_raw_branches"],
                           "epochs": report["quarantined_epochs"]}}

    def _session(self, sid: str, *, load: bool = True) -> _OpenSession:
        """The open session (loaded from the archive without preparing anything when only persisted)."""
        self._guard(sid, "loading it")
        session = self._open.get(sid)
        if session is not None:
            return session
        record = self.archive.session(sid)
        if record is None:
            raise _control_error("unknown_session", f"unknown LongCat session {sid!r}")
        if not load:
            raise _control_error("unknown_session", f"LongCat session {sid!r} is not open")
        session = _OpenSession(session_id=sid, record_ref=record["ref"], created=record["created"],
                               source=record.get("source"), head=self.archive.boundary(record["head"]))
        session.history = collections.deque(maxlen=self.config.history)
        session.event("loaded", head=session.head.ref, segments=session.head.segments)
        self._open[sid] = session
        return session

    def _require_idle(self, sid: str, op: str) -> _OpenSession | None:
        self._guard(sid, op)
        session = self._open.get(sid)
        if session is None and self.archive.session(sid) is None:
            raise _control_error("unknown_session", f"unknown LongCat session {sid!r}")
        if session is not None and session.status == "running":
            raise _control_error("busy", f"session {sid!r} is running a request; {op} waits for it to return "
                                         "(pause it first)")
        return session

    def _head(self, sid: str) -> Boundary:
        self._guard(sid, "reading its head")
        session = self._open.get(sid)
        if session is not None:
            if session.head is None:
                raise _control_error("busy", f"session {sid!r} is being created by a running request")
            return session.head
        record = self.archive.session(sid)
        if record is None:
            raise _control_error("unknown_session", f"unknown LongCat session {sid!r}")
        return self.archive.boundary(record["head"])

    def _resolve(self, sid: str, checkpoint: Any, *, ancestor: bool) -> Boundary:
        """A boundary of ``sid``: None = head; int = segment count on the head path; a name; or a boundary node on
        the head path (or named). ``ancestor``: must lie on the head path (rewind)."""
        head = self._head(sid)
        if checkpoint is None:
            return head
        path = self.archive.path(head.ref)
        if type(checkpoint) is int:
            if checkpoint < 0 or checkpoint > head.segments:
                raise _control_error("unknown_checkpoint", f"session {sid!r} has {head.segments} segments; no "
                                                           f"boundary {checkpoint}")
            return path[checkpoint]
        if isinstance(checkpoint, str) and checkpoint.startswith("manifest:"):
            on_path = {item.ref: item for item in path}
            named = set(self.archive.named(sid).values())
            if checkpoint in on_path:
                return on_path[checkpoint]
            if checkpoint in named and not ancestor:
                return self.archive.boundary(checkpoint)
            raise _control_error("unknown_checkpoint", f"{checkpoint} is not a boundary on the head path of "
                                                       f"{sid!r}" + ("" if ancestor else " nor a named checkpoint"))
        if isinstance(checkpoint, str) and NAME_PATTERN.fullmatch(checkpoint) is not None:
            named = self.archive.named(sid)
            if checkpoint not in named:
                raise _control_error("unknown_checkpoint", f"session {sid!r} has no checkpoint named {checkpoint!r}")
            damaged = self._damaged.get(f"{CHECKPOINT_PREFIX}{sid}/{checkpoint}")
            if damaged is not None:
                raise _control_error("integrity", f"checkpoint {checkpoint!r} of {sid!r} is quarantined: its boundary "
                                                  f"chain is damaged ({damaged})", session_id=sid,
                                     checkpoint=checkpoint, quarantined=damaged)
            boundary = self.archive.boundary(named[checkpoint])
            if ancestor and boundary.ref not in {item.ref for item in path}:
                raise _control_error("illegal_transition", f"checkpoint {checkpoint!r} is not an ancestor of the head "
                                                           f"of {sid!r} (rewind needs an ancestor; fork it instead)")
            return boundary
        raise _control_error("invalid_argument", "checkpoint must be a segment count, a boundary node reference or a "
                                                 f"checkpoint name; got {checkpoint!r}")

    def _pending(self, sid: str) -> list[dict]:
        return self.backend.pending_epochs(sid)

    def _summary(self, sid: str) -> dict:
        reason = self._quarantined(sid)
        if reason is not None:
            return {"session_id": sid, "open": False, "status": "quarantined", "head": None,
                    "generated_segments": None, "generated_chunks": None, "pending": [], "published": None,
                    "pause_requested": False, "archive_failures": [], "resident": False,
                    "named_checkpoints": self.archive.named(sid), "quarantined": reason,
                    "damage": self._session_damage(sid)}
        session = self._open.get(sid)
        record = None if session is not None else self.archive.session(sid)
        if session is None and record is None:
            raise _control_error("unknown_session", f"unknown LongCat session {sid!r}")
        head = session.head if session is not None else self.archive.boundary(record["head"])
        pending = self._pending(sid)
        segments = 0 if head is None else head.segments
        entry = {"session_id": sid, "open": session is not None,
                 "status": "closed" if session is None else session.status, "head": None if head is None else head.ref,
                 "generated_segments": segments, "generated_chunks": segments,
                 "pending": [item["epoch_digest"] for item in pending], "published": not pending,
                 "pause_requested": bool(session is not None and session.pause_requested),
                 "archive_failures": [] if session is None else list(session.archive_failures),
                 "resident": False, "named_checkpoints": self.archive.named(sid)}
        damage = self._session_damage(sid)
        if damage:
            entry["damage"] = damage
        return entry

    def _drain(self) -> float:
        started = time.perf_counter()
        self.backend.flush(wait=True)
        return time.perf_counter() - started

    def _remember(self, summary: dict) -> None:
        self._requests[summary["request_id"]] = summary
        self._requests.move_to_end(summary["request_id"])
        while len(self._requests) > 64:
            self._requests.popitem(last=False)
        session = self._open.get(summary["session_id"])
        if session is not None and summary["request_id"] not in session.requests:
            session.requests.append(summary["request_id"])

    def _live_epochs(self) -> list[ConditioningCacheEpoch]:
        """The conditioning-cache epochs the live boundaries record (every boundary a session head or a named
        checkpoint reaches): what the sessions can still restore. Garbage collection keeps exactly their archives;
        a restart or a collection re-indexes exactly them (both representations).

        The session archive is scanned record by record (CR-17-1): with ``on_open_error="quarantine"`` a damaged
        session record, boundary chain or named checkpoint is recorded in ``_damaged`` (the session is quarantined)
        and the boundaries read before the damage still count; ``"raise"`` fails on the first. ``_owners``: epoch
        key -> the sessions whose head path or named checkpoints record it."""
        scan = self.archive.scan(strict=self.backend.integrity["on_open_error"] != "quarantine")
        damaged = dict(scan.damaged)
        epochs: dict[tuple, ConditioningCacheEpoch] = {}
        owners: dict[tuple, set] = {}
        for name, refs in scan.paths.items():
            sid = ArchiveScan.session_of(name)
            for ref in refs:
                data = (scan.boundaries[ref].segment or {}).get("epoch")
                if not data:
                    continue
                try:
                    epoch = ConditioningCacheEpoch.from_dict(data)
                except Exception as exc:  # noqa: BLE001 - a malformed stored epoch record is damage of this head
                    if self.backend.integrity["on_open_error"] != "quarantine":
                        raise
                    damaged.setdefault(name, f"{type(exc).__name__}: boundary {ref} records a malformed cache "
                                             f"epoch ({exc})")
                    continue
                key = mb.epoch_key(epoch)
                epochs.setdefault(key, epoch)
                owners.setdefault(key, set()).add(sid)
        self._damaged, self._owners = damaged, owners
        return list(epochs.values())

    def _segment_epochs(self, head: Boundary) -> dict[int, ConditioningCacheEpoch]:
        epochs = {}
        for boundary in self.archive.path(head.ref)[1:]:
            epoch = (boundary.segment or {}).get("epoch")
            if epoch:
                epochs[boundary.segments - 1] = ConditioningCacheEpoch.from_dict(epoch)
        return epochs

    # -- ops -----------------------------------------------------------------------------------------------------
    def _capabilities(self, args: Mapping[str, Any]) -> dict:
        warm = self.startup.get("warmup") or {}
        return {"model": mb.MODEL_NAME, "representation": self.config.representation, "boundary": "segment",
                "boundary_frames": 13, "segment_new_frames": 80, "first_segment_frames": 93,
                "streams": ["cond"], "ops": sorted(self._ops), "inflight_ops": list(INFLIGHT_OPS),
                "generation_actions": list(GENERATION_ACTIONS), "prompt_change": dict(PROMPT_CHANGE),
                "outputs": ["new"], "durability": self.config.durability, "prewarm": self.config.prewarm,
                "archival": {"plan_device": str(getattr(self.backend.pipeline, "device", None)),
                             "plan_memory_bytes": getattr(getattr(self.backend.pipeline, "options", None),
                                                          "plan_memory_bytes", None),
                             "schedule": self.backend.schedule, "schedule_requested": self.config.schedule,
                             "schedule_reason": (self.startup.get("schedule") or {}).get("reason"),
                             "defer_planning": self.backend.defer_planning,
                             "defer_planning_reason": (self.startup.get("schedule") or {}).get(
                                 "defer_planning_reason"),
                             "release_planner_memory": self.backend.release_planner_memory,
                             "planning_gate": self.startup.get("planning_gate"),
                             "deferred_sources": self.backend.deferred_sources,
                             "raw_tier": self.startup.get("raw_tier"),
                             "encoder": self.startup.get("encoder")},
                "restore_policy": "restore published epochs, build the rest (recorded per segment)",
                "observe_plain_requests": False, "realtime": False, "resident_reuse": False,
                "warmup": {key: warm.get(key) for key in ("seconds", "max_rank", "level", "staging_engine")
                           if key in warm},
                "integrity": self._integrity(), "staging_engine": self._staging(),
                "overlap_restore": dict(self.overlap_restore)}

    def _list(self, args: Mapping[str, Any]) -> dict:
        from reelcache.errors import IntegrityError

        sessions = []
        for sid in sorted(set(self.archive.session_ids()) | set(self._open)):
            try:
                sessions.append(self._summary(sid))
            except IntegrityError as exc:  # damaged since the last scan: quarantined from now on
                if self.backend.integrity["on_open_error"] != "quarantine" or not mb.is_damage(exc):
                    raise
                self._damaged[SESSION_PREFIX + sid] = f"{type(exc).__name__}: {exc}"
                sessions.append(self._summary(sid))
            except Exception:  # noqa: BLE001 - deleted meanwhile
                continue
        return {"sessions": sessions, "resident": None}

    def _describe(self, args: Mapping[str, Any]) -> dict:
        sid = self._sid(args)
        history = args.get("history", 50)
        if type(history) is not int or history < 0:
            raise _control_error("invalid_argument", "history must be a nonnegative int")
        entry = self._summary(sid)
        if entry["head"] is None:  # being created by a running request, or quarantined (status, reason, damage)
            return {**entry, "segments": [], "history": []}
        head = self._head(sid)
        segments = []
        for boundary in self.archive.path(head.ref)[1:]:
            segment = dict(boundary.segment or {})
            prompt = segment.get("prompt")
            segment["prompt"] = None if prompt is None else (prompt if len(prompt) <= 200 else prompt[:200] + "...")
            epoch = segment.pop("epoch", None)
            segment["cache_epoch_now"] = None if not epoch else (self.backend.archive_state(
                ConditioningCacheEpoch.from_dict(epoch), self.config.representation) or "not archived")
            segment["boundary"] = boundary.ref
            segments.append(segment)
        session = self._open.get(sid)
        record = self.archive.session(sid)
        entry.update({"segments": segments, "geometry": head.geometry,
                      "source": None if record is None else record.get("source"),
                      "created": None if record is None else record.get("created"),
                      "updated": None if record is None else record.get("updated")})
        if session is not None:
            events = list(session.history)
            entry["history"] = events[-history:] if history else []
            entry["requests"] = list(session.requests)
        return entry

    def _checkpoints(self, args: Mapping[str, Any]) -> dict:
        sid = self._sid(args)
        self._require_idle(sid, "checkpoints")
        self._drain()
        path = self.archive.path(self._head(sid).ref)
        boundaries = [{"segments": item.segments, "chunks": item.segments, "node": item.ref, "kind": "boundary",
                       "checkpoint": True} for item in path]
        on_path = {item.ref: item.segments for item in path}
        named = {}
        for name, ref in self.archive.named(sid).items():
            damaged = self._damaged.get(f"{CHECKPOINT_PREFIX}{sid}/{name}")
            if damaged is not None and ref not in on_path:
                named[name] = {"node": ref, "on_head_path": False, "segments": None, "quarantined": damaged}
                continue
            named[name] = {"node": ref, "on_head_path": ref in on_path,
                           "segments": on_path.get(ref, self.archive.boundary(ref).segments)}
        return {"session_id": sid, "checkpoints": boundaries, "named": named}

    def _request(self, args: Mapping[str, Any]) -> dict:
        request_id = args.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise _control_error("invalid_argument", "request_id must be a nonempty string")
        summary = self._requests.get(request_id)
        if summary is None and self._run is not None and self._run.request_id == request_id:
            summary = self._run.summary
        if summary is None:
            raise _control_error("unknown_request", f"no recent LongCat session request {request_id!r}")
        return {"request_id": request_id, "summary": summary}

    def _open_op(self, args: Mapping[str, Any]) -> dict:
        sid = self._sid(args)
        was_open = sid in self._open
        started = time.perf_counter()
        session = self._session(sid)
        prewarm = None
        if not was_open and self.config.prewarm and self.config.representation == "svd_hevc":
            epochs = [epoch for epoch in self._segment_epochs(session.head).values()
                      if self.backend.archive_state(epoch, "svd_hevc") == "published"]
            if epochs:
                report = self.backend.prepare_epochs(epochs)
                prewarm = {"seconds": report["seconds"], "epochs": report["epochs"], "level": report["level"],
                           "quarantined": report.get("quarantined", [])}
            session.event("reopened", prewarm=prewarm)
        described = self._describe({"session_id": sid, "history": 0})
        described.pop("history", None)
        return {**described, "was_open": was_open, "open_seconds": time.perf_counter() - started,
                "prewarm_seconds": None if prewarm is None else prewarm["seconds"], "prepared": prewarm}

    def _pause(self, args: Mapping[str, Any]) -> dict:
        sid = self._sid(args)
        self._guard(sid, "pause")
        session = self._open.get(sid)
        if session is None:
            if self.archive.session(sid) is None:
                raise _control_error("unknown_session", f"unknown LongCat session {sid!r}")
            return {**self._summary(sid), "deferred": False, "wait_seconds": 0.0}
        if session.status == "running":
            session.pause_requested = True
            session.event("pause_requested")
            segments = 0 if session.head is None else session.head.segments
            return {"session_id": sid, "status": "pause_requested", "deferred": True,
                    "generated_segments": segments, "generated_chunks": segments}
        wait = self._drain()
        return {**self._summary(sid), "deferred": False, "wait_seconds": wait}

    def _checkpoint(self, args: Mapping[str, Any]) -> dict:
        sid = self._sid(args)
        name = None if args.get("name") is None else self._name(args.get("name"), "checkpoint name")
        replace = args.get("replace", False)
        if type(replace) is not bool:
            raise _control_error("invalid_argument", "replace must be a bool")
        self._require_idle(sid, "checkpoint")
        wait = self._drain()
        head = self._head(sid)
        result = {"session_id": sid, "head": head.ref, "generated_segments": head.segments,
                  "generated_chunks": head.segments, "published": not self._pending(sid), "wait_seconds": wait}
        if name is not None:
            previous = self.archive.pin(sid, name, head.ref, replace=replace)
            result.update(name=name, replaced=previous if previous not in (None, head.ref) else None)
        return result

    def _fork(self, source: str, target: str, checkpoint: Any) -> Boundary:
        self._require_idle(source, "fork")
        if self._exists(target):
            raise _control_error("illegal_transition", f"fork target {target!r} already exists")
        boundary = self._resolve(source, checkpoint, ancestor=False)
        self._drain()
        self._fork_cache_lineage(target, boundary)
        record = self.archive.put_session(target, head=boundary.ref, expected=None,
                                          source={"from_session": source, "checkpoint": checkpoint,
                                                  "boundary": boundary.ref, "segments": boundary.segments})
        session = _OpenSession(session_id=target, record_ref=record["ref"], created=record["created"],
                               source=record["source"], head=boundary)
        session.history = collections.deque(maxlen=self.config.history)
        session.event("forked", source=source, head=boundary.ref)
        self._open[target] = session
        return boundary

    def _fork_cache_lineage(self, target: str, boundary: Boundary) -> None:
        """The target's cache lineage starts as a fork at the boundary's last segment node (basis sharing, and the
        source's history stays reachable for the target when the source is deleted)."""
        epoch = (boundary.segment or {}).get("epoch")
        if not epoch or self.config.representation != "svd_hevc":
            return
        node = self.backend.node_ref(ConditioningCacheEpoch.from_dict(epoch))
        if node is None:
            return
        branch = mb.lineage_branch(target)
        if branch in self.runtime.list_lineages():
            raise _control_error("illegal_transition", f"a cache lineage of {target!r} already exists ({branch})")
        self.runtime.fork_lineage(node, branch)
        self.backend.forget_session(target)

    def _fork_op(self, args: Mapping[str, Any]) -> dict:
        source = self._sid(args)
        target = self._name(args.get("target"), "fork target")
        checkpoint = args.get("checkpoint")
        boundary = self._fork(source, target, checkpoint)
        return {**self._summary(target), "source": source, "checkpoint": checkpoint, "node": boundary.ref}

    def _rewind(self, sid: str, checkpoint: Any) -> tuple[Boundary, Boundary, str]:
        """Move the head back; returns (previous head, new head, previous session record ref)."""
        self._require_idle(sid, "rewind")
        boundary = self._resolve(sid, checkpoint, ancestor=True)
        session = self._session(sid)
        before, previous = session.head, session.record_ref
        record = self.archive.put_session(sid, head=boundary.ref, expected=session.record_ref, created=session.created,
                                          source=session.source)
        session.record_ref, session.head = record["ref"], boundary
        session.event("rewound", previous_head=before.ref, head=boundary.ref)
        return before, boundary, previous

    def _rewind_op(self, args: Mapping[str, Any]) -> dict:
        sid = self._sid(args)
        if "checkpoint" not in args:
            raise _control_error("invalid_argument", "checkpoint is required")
        before, boundary, _ = self._rewind(sid, args["checkpoint"])
        return {**self._summary(sid), "previous_head": before.ref, "checkpoint": args["checkpoint"],
                "node": boundary.ref}

    def _roll_back(self, rollback: Mapping[str, Any]) -> dict:
        """Undo the fork or rewind of a generation request that hit a damaged archive before its first new boundary
        (CR-17-1: the sessions' published state stays as it was): a rewound session gets its previous record back,
        a fork target is removed (its cache lineage too). Returns what was done (never raises)."""
        sid, action = rollback["session_id"], rollback["action"]
        session = self._open.get(sid)
        store = self.runtime.store
        try:
            if session is None:
                return {"action": action, "skipped": "the session is not open"}
            if action == "rewind":
                store.set_ref(SESSION_PREFIX + sid, rollback["record_ref"], expected=session.record_ref)
                session.record_ref, session.head = rollback["record_ref"], rollback["head"]
                session.event("rewind_rolled_back", head=session.head.ref)
                return {"action": action, "head": session.head.ref}
            if self.backend.pending_epochs(sid):
                return {"action": action, "skipped": "the fork target's archival is in flight"}
            store.delete_ref(SESSION_PREFIX + sid, expected=session.record_ref)
            for branch in self.backend.session_branches(sid):
                if self.backend.pipeline is not None:
                    self.backend.pipeline.reset(branch, timeout=self.backend.timeout)
                self.runtime.delete_lineage(branch)
            self.backend.forget_session(sid)
            self._open.pop(sid, None)
            return {"action": action, "removed": sid}
        except Exception as exc:  # noqa: BLE001 - reported; the request's own error is the one raised
            return {"action": action, "failed": f"{type(exc).__name__}: {exc}"}

    def _close_session(self, sid: str, *, abort: bool) -> dict:
        retired = self.backend.discard_pending(sid) if abort else 0
        wait = self._drain()
        self._open.pop(sid, None)
        return {"session_id": sid, "retired": retired, "wait_seconds": wait}

    def _close_op(self, args: Mapping[str, Any]) -> dict:
        sid = self._sid(args)
        mode = args.get("mode", "drain")
        if mode not in ("drain", "abort"):
            raise _control_error("invalid_argument", "mode must be 'drain' or 'abort'")
        self._require_idle(sid, "close")
        report = self._close_session(sid, abort=mode == "abort")
        return {**report, "mode": mode, "status": "closed", "head": self._head(sid).ref}

    def _delete_session(self, sid: str) -> dict:
        if self._quarantined(sid) is not None:
            return self._delete_quarantined(sid)
        self._require_idle(sid, "delete")
        report = self._close_session(sid, abort=True)
        deleted = self.archive.delete_session(sid)
        lineages = self.backend.session_branches(sid)
        # epochs other sessions still record but only this session's lineages reach (they restored them) stay
        held = self.backend.hold_epochs(self._live_epochs(), dropping=lineages)
        for branch in lineages:
            if self.backend.pipeline is not None:  # the archive pipeline releases its hold on the lineage
                self.backend.pipeline.reset(branch, timeout=self.backend.timeout)
            self.runtime.delete_lineage(branch)
        self.backend.forget_session(sid)
        return {**deleted, "retired": report["retired"], "deleted_lineages": lineages, "held_epochs": held}

    def _delete_quarantined(self, sid: str) -> dict:
        """Delete a quarantined session without reading its damaged records: its session ref, named checkpoints and
        cache lineages go; the store keeps damaged graphs until no ref reaches them (CR-17-1)."""
        reason = self._quarantined(sid)
        report = self._close_session(sid, abort=True)
        store = self.runtime.store
        names = sorted(self.archive.named(sid))
        for name in names:
            store.delete_ref(f"{CHECKPOINT_PREFIX}{sid}/{name}")
        store.delete_ref(SESSION_PREFIX + sid)
        lineages = self.backend.session_branches(sid)
        held = self.backend.hold_epochs(self._live_epochs(), dropping=lineages)
        for branch in lineages:
            if self.backend.pipeline is not None:
                self.backend.pipeline.reset(branch, timeout=self.backend.timeout)
            self.runtime.delete_lineage(branch)
        self.backend.forget_session(sid)
        return {"session_id": sid, "head": None, "deleted_checkpoints": names, "retired": report["retired"],
                "deleted_lineages": lineages, "held_epochs": held, "quarantined": reason}

    def _delete_op(self, args: Mapping[str, Any]) -> dict:
        sid = self._sid(args)
        name = None if args.get("checkpoint") is None else self._name(args.get("checkpoint"), "checkpoint name")
        collect = args.get("gc", True)
        if type(collect) is not bool:
            raise _control_error("invalid_argument", "gc must be a bool")
        if name is not None:
            if not self._exists(sid):
                raise _control_error("unknown_session", f"unknown LongCat session {sid!r}")
            try:
                node = self.archive.unpin(sid, name)
            except KeyError:
                raise _control_error("unknown_checkpoint", f"session {sid!r} has no checkpoint named {name!r}") from None
            result = {"session_id": sid, "deleted_checkpoint": name, "node": node}
        else:
            result = self._delete_session(sid)
        if collect:
            result["gc"] = self._collect_garbage()
        return result

    def _gc_op(self, args: Mapping[str, Any]) -> dict:
        return {"gc": self._collect_garbage()}

    # -- freeze D raw tiers (Step 7 §9 containers: A = archive_stats, B = drop_raw / demote_raw / reset_raw_peaks) --
    def _archive_stats(self, args: Mapping[str, Any]) -> dict:
        sid = args.get("session_id")
        if sid is not None and not isinstance(sid, str):
            raise _control_error("invalid_argument", "session_id must be a string")
        return {"pipeline": self.backend.raw_tier_stats(), "entries": self.backend.raw_entries(sid),
                "planning_gate": self.backend.planning_gate_stats()}

    def _drop_raw(self, args: Mapping[str, Any]) -> dict:
        sid, drain = args.get("session_id"), args.get("drain", True)
        if not isinstance(sid, str) or type(drain) is not bool:
            raise _control_error("invalid_argument", "drop_raw takes session_id (string) and drain (bool)")
        return {"drop_raw": self.backend.drop_raw(sid, drain=drain)}

    def _demote_raw(self, args: Mapping[str, Any]) -> dict:
        sid = args.get("session_id")
        if not isinstance(sid, str):
            raise _control_error("invalid_argument", "demote_raw takes session_id (string)")
        return {"demote_raw": self.backend.demote_raw(sid)}

    def _reset_raw_peaks(self, args: Mapping[str, Any]) -> dict:
        self.backend.reset_raw_peaks()
        return {"reset_raw_peaks": True}

    def _collect_garbage(self) -> dict:
        """Drain; keep exactly the archives of the live epochs (:meth:`_live_epochs`): drop RAW reference branches of
        other epochs, release holder lineages no live epoch needs (and hold live epochs only a released holder
        reached); collect unreachable store objects; re-index the live epochs (both representations)."""
        if self._run is not None:
            raise _control_error("busy", "garbage collection waits for the running request")
        wait = self._drain()
        epochs = self._live_epochs()
        # a damaged session record or named checkpoint may reach RAW branches and holders the scan could not see:
        # keep them all while one is quarantined (CR-17-1); the store keeps damaged graphs
        conservative = bool(self._damaged)
        raw_deleted, released = [], []
        if not conservative:
            live = {mb.raw_branch(epoch) for epoch in epochs}
            for name in self.runtime.store.list_refs("branches/" + mb.RAW_BRANCH_PREFIX):
                branch = name[len("branches/"):]
                if branch not in live:
                    self.runtime.delete_branch(branch)
                    raw_deleted.append(branch)
            released = self.backend.release_holders(epochs)
        held = self.backend.hold_epochs(epochs)
        collected = self.runtime.collect_garbage()
        reindexed = self.backend.refresh_index(live_epochs=epochs, owners=self._owners)
        return {**collected, "raw_branches_deleted": len(raw_deleted), "holders_released": len(released),
                "holders_created": len(held), "live_epochs": len(epochs), "reindexed": reindexed,
                "reindexed_by_representation": dict(self.backend.last_reload or {}), "drain_seconds": wait,
                "conservative": ({"quarantined": sorted(self._damaged)} if conservative else None)}

    # -- generation ----------------------------------------------------------------------------------------------
    def begin_request(self, request: Mapping[str, Any], handle: Mapping[str, Any]) -> SessionRun:
        """The plan of one session request (``request``: :func:`request_fields`; ``handle``: the validated
        ``causal_kv_session``). Raises :class:`SessionControlError` for refused requests, a typed IntegrityError for
        a damaged archive (a fork or rewind it already applied is rolled back); a refused request with a request id
        is remembered for the ``request`` op (``error``, ``error_code``)."""
        if self.backend.raw_tier:
            self.backend.release_resident()  # admission: no segment runs; GPU raw beyond the budget leaves
        try:
            return self._begin_request(request, handle)
        except Exception as exc:
            request_id = request.get("request_id") if isinstance(request, Mapping) else None
            if isinstance(request_id, str) and request_id:
                sid = handle.get("session_id") if isinstance(handle, Mapping) else None
                self._remember({"request_id": request_id, "session_id": sid if isinstance(sid, str) else None,
                                "action": handle.get("action") if isinstance(handle, Mapping) else None,
                                "refused": True, "segments": [], "finished_at": time.time(),
                                "error": f"{type(exc).__name__}: {exc}", "error_code": _error_code(exc)})
            raise

    def _begin_request(self, request: Mapping[str, Any], handle: Mapping[str, Any]) -> SessionRun:
        from reelcache.errors import IntegrityError

        rollback = None
        if self._run is not None:
            self._run.finish(error=RuntimeError("a new request started before the previous session run finished"))
        if not isinstance(handle, Mapping):
            raise _control_error("invalid_argument", "causal_kv_session must be a JSON object")
        unknown = set(handle) - _HANDLE_KEYS
        if unknown:
            raise _control_error("invalid_argument", f"unknown causal_kv_session keys {sorted(unknown)}")
        sid = self._sid(handle)
        action = handle.get("action", "generate")
        if action not in GENERATION_ACTIONS:
            raise _control_error("invalid_argument", f"action must be one of {GENERATION_ACTIONS}")
        if handle.get("output") not in (None, "new"):
            raise _control_error("incompatible_model", "LongCat sessions return the new segments only (output='new'): "
                                                       "the archive keeps boundaries and caches, not decoded frames")
        if handle.get("decode_context_latents") is not None:
            raise _control_error("incompatible_model", "decode_context_latents does not apply to LongCat-Video: each "
                                                       "segment is decoded with its own 4 conditioning latents")
        for key in ("generator_state", "conditioning_cache_backend"):
            if request.get(key) is not None:
                raise _control_error("invalid_argument", f"{key} is set by the session archive, not by the request")
        if request.get("start_segment_index") not in (None, 0):
            raise _control_error("invalid_argument", "start_segment_index is set by the session archive")
        if request.get("session_id") not in (None, sid):
            raise _control_error("invalid_argument", "session_id differs from causal_kv_session.session_id")
        if request.get("injected"):
            raise _control_error("invalid_argument", f"injected {request['injected']} are not accepted with sessions")
        num_segments = request.get("num_segments", 1)
        if type(num_segments) is not int or num_segments < 1:
            raise _control_error("invalid_argument", "num_segments must be a positive int")
        pause_after = handle.get("pause_after_chunks")
        if pause_after is not None and (type(pause_after) is not int or pause_after < 1):
            raise _control_error("invalid_argument", "pause_after_chunks must be a positive int")
        durable = handle.get("durable")
        if durable is not None and type(durable) is not bool:
            raise _control_error("invalid_argument", "durable must be a bool")
        overlap = handle.get("overlap_restore")
        if overlap is not None and type(overlap) is not bool:
            raise _control_error("invalid_argument", "overlap_restore must be a bool")
        replace = handle.get("replace", False)
        if type(replace) is not bool or (replace and action != "generate"):
            raise _control_error("invalid_argument", "replace is a bool for generate only")
        checkpoint = handle.get("checkpoint")
        if action == "generate":
            if checkpoint is not None:
                raise _control_error("invalid_argument", "generate starts a new session (no checkpoint)")
            if not request.get("has_client_frames"):
                raise _control_error("invalid_argument", "generate needs the conditioning video of segment 0")
            if self._exists(sid):
                if not replace:
                    raise _control_error("illegal_transition", f"session {sid!r} exists (replace=true drops it)")
                self._delete_session(sid)
            session = _OpenSession(session_id=sid, record_ref="", created=time.time(), source=None, head=None)
            session.history = collections.deque(maxlen=self.config.history)
            plan = RequestPlan(action, sid, 0, num_segments, None, None, None)
        else:
            if request.get("has_client_frames"):
                raise _control_error("invalid_argument", f"{action} takes the boundary conditioning from the session "
                                                         "archive, never from the request (drop the video input)")
            # the boundary's frames are read (and verified) before a fork or rewind changes any session
            if action == "fork":
                source = self._sid(handle, "from_session")
                self._name(sid, "fork target")
                self._require_idle(source, "fork")
                if self._exists(sid):
                    raise _control_error("illegal_transition", f"fork target {sid!r} already exists")
                frames = self.archive.frames(self._resolve(source, checkpoint, ancestor=False))
                head = self._fork(source, sid, checkpoint)
                rollback = {"action": "fork", "session_id": sid, "record_ref": self._open[sid].record_ref}
            elif action == "rewind":
                if checkpoint is None:
                    raise _control_error("invalid_argument", "rewind needs a checkpoint")
                self._require_idle(sid, "rewind")
                frames = self.archive.frames(self._resolve(sid, checkpoint, ancestor=True))
                before, head, previous = self._rewind(sid, checkpoint)
                rollback = {"action": "rewind", "session_id": sid, "record_ref": previous, "head": before}
            else:
                self._require_idle(sid, "resume")
                head = self._head(sid)
                if checkpoint is not None and self._resolve(sid, checkpoint, ancestor=True).ref != head.ref:
                    raise _control_error("illegal_transition", "resume continues from the session head; rewind to an "
                                                               "earlier checkpoint first (action 'rewind')")
                frames = self.archive.frames(head)
                rollback = None
            try:
                session = self._session(sid)
                plan = RequestPlan(action, sid, head.segments, num_segments, frames, head.generator_state,
                                   dict(head.geometry))
                self.backend.seed_timeline(sid, self._segment_epochs(head))
            except IntegrityError as exc:
                if rollback is not None and mb.is_damage(exc):
                    self._roll_back(rollback)
                raise
        stop_at = plan.start_segment + plan.num_segments
        if pause_after is not None:
            if pause_after <= plan.start_segment:
                raise _control_error("invalid_argument", f"pause_after_chunks={pause_after} is not after the session's "
                                                         f"{plan.start_segment} segments")
            stop_at = min(stop_at, pause_after)
        plan.num_segments = stop_at - plan.start_segment
        request_id = request.get("request_id") or f"longcat-{uuid.uuid4().hex[:16]}"
        run = SessionRun(self, session, plan, request_id=request_id, stop_at=stop_at, durable=durable,
                         prompt=request.get("prompt"), prompt_embeds_sha256=request.get("prompt_embeds_sha256"))
        run.rollback = rollback
        run.overlap_restore = overlap
        run.summary["overlap_restore"] = self.overlap_effective(run)
        run.summary["pause_after"] = pause_after
        run.ends_by_pause = pause_after is not None and stop_at == pause_after
        if action == "generate":
            self._open[sid] = session
        session.status = "running"
        session.pause_requested = False
        session.event("request_started", request_id=request_id, action=action, start_segment=plan.start_segment,
                      stop_at=stop_at)
        self._run = run
        return run


# ----------------------------------------------------------------------------------------------- factory
def host_fixed_bytes(cfg: LongCatSessionConfig, *, cuda: bool, encoder: Mapping[str, Any] | None = None) -> dict:
    """This worker's fixed host consumers next to the raw tier (review amendment 2): the staging engine's pinned ring,
    the archive pipeline's two staged canvases (measured 1.72 GiB each at 720p), its fingerprint / finalize tile
    workspaces (d5: ``raw_tiers.FINALIZE_WORKSPACE_BYTES``, Step 7 dev22i), the restorer's prepared cache and the
    host memory the NVENC probe took."""
    parts = {"staging_ring": cfg.staging_pinned_bytes if cuda and cfg.staging_engine else 0,
             "canvas_pool": 2 * rt.CANVAS_STAGING_BYTES if cfg.representation == "svd_hevc" else 0,
             "finalize_workspace": rt.FINALIZE_WORKSPACE_BYTES if cfg.representation == "svd_hevc" else 0,
             "prepared_cache": cfg.prepared_budget_bytes if cuda else 0,
             "nvenc_probe": int((encoder or {}).get("probe_host_bytes") or 0)}
    return {"parts": parts, "bytes": sum(parts.values())}


def plan_host_slots(cfg: LongCatSessionConfig, *, epoch_bytes: int, fixed_bytes: int,
                    memory: Mapping[str, Any] | None = None) -> dict:
    """Host capture slots and the raw tier of one worker (freeze D; Step 7 R5; review amendment 2).

    ``raw_tier: false``: C3 exactly (``capture_slots``, default 4; the budget is recorded, not applied). Otherwise the
    budget of :func:`.raw_tiers.host_raw_budget` (auto, ``host_raw_budget_bytes`` or ``capture_slots``): an explicit
    budget above this worker's physical share is refused (``ConfigurationError``); a budget of at least one epoch
    enables the tier with that many slots; below one epoch (auto, or explicit such as 0) the tier is disabled and
    captures use C3 back-pressure in ``min(capture_slots or 4, physical slots)`` slots -- with none,
    conditioning-cache archival is off. Never a slot above the safe headroom."""
    from reelcache.errors import ConfigurationError

    gib = float(1 << 30)
    workers = rt.local_workers(cfg.raw_tier_workers)
    if not cfg.raw_tier:
        record = rt.host_raw_budget(epoch_bytes=epoch_bytes, workers=workers, fixed_bytes=fixed_bytes, memory=memory,
                                    explicit_slots=cfg.c3_slots())
        record.update(raw_tier=False, capture_slots=cfg.c3_slots(), archival=True,
                      reason="raw_tier false: C3 capture slots (capture_slots, default 4; not budgeted)")
        return record
    record = rt.host_raw_budget(epoch_bytes=epoch_bytes, workers=workers, fixed_bytes=fixed_bytes, memory=memory,
                                explicit_bytes=cfg.host_raw_budget_bytes, explicit_slots=cfg.capture_slots)
    if record["source"] != "auto" and not record["fits"]:
        memory_record = record["memory"]
        raise ConfigurationError(
            f"{record['source']} asks for {record['budget_bytes'] / gib:.1f} GiB of host raw tier, but this worker's "
            f"share of the memory headroom after its fixed host consumers is {record['physical_bytes'] / gib:.1f} GiB "
            f"({memory_record.get('source')}: headroom {(memory_record.get('headroom_bytes') or 0) / gib:.1f} GiB / "
            f"{workers} worker(s) - {fixed_bytes / gib:.1f} GiB fixed); lower it or raise the job's memory")
    if record["slots"] >= 1:
        record.update(raw_tier=True, capture_slots=record["slots"], archival=True,
                      reason=f"{record['source']}: {record['slots']} host slot(s) of {epoch_bytes / gib:.2f} GiB")
        return record
    slots = min(cfg.c3_slots(), record["physical_slots"])
    record.update(raw_tier=False, capture_slots=slots, archival=slots > 0, reason=(
        f"host raw budget {record['budget_bytes'] / gib:.1f} GiB ({record['source']}) is below one epoch "
        f"({epoch_bytes / gib:.2f} GiB): raw tier disabled; " + (
            f"captures use C3 back-pressure in {slots} slot(s) of the worker's physical share" if slots else
            "the worker's share holds no capture slot: conditioning-cache archival is off (resumes rebuild the "
            "cache)")))
    return record


def plan_gpu_raw(cfg: LongCatSessionConfig, *, enabled: bool, total_bytes: int, resident_bytes: int,
                 epoch_bytes: int, reserved_bytes: int = 0) -> dict:
    """GPU raw budget of a CUDA worker with the tier enabled (:func:`.raw_tiers.gpu_raw_budget`; explicit
    ``gpu_raw_budget_bytes`` taken as given), else 0. The generation need gains the overlapped restore's step-0
    working set (``raw_tiers.OVERLAP_RESTORE_NEED_BYTES``) while ``overlap_restore`` is effective (d5)."""
    if not enabled or total_bytes <= 0:
        return {"source": "disabled", "budget_bytes": 0, "epochs": 0, "epoch_bytes": epoch_bytes,
                "reason": "raw tier disabled" if not enabled else "no CUDA device", "fits": True}
    overlap = bool(cfg.overlap_restore) and cfg.representation == "svd_hevc"
    need = rt.SECTION_NEED_BYTES + (rt.OVERLAP_RESTORE_NEED_BYTES if overlap else 0)
    record = rt.gpu_raw_budget(total_bytes=total_bytes, resident_bytes=resident_bytes, epoch_bytes=epoch_bytes,
                               reserved_bytes=reserved_bytes, need_bytes=need, explicit_bytes=cfg.gpu_raw_budget_bytes)
    record["overlap_restore_need_bytes"] = rt.OVERLAP_RESTORE_NEED_BYTES if overlap else 0
    record["reason"] = (f"{record['source']}: {record['epochs']} epoch(s) above the resident model, the measured "
                        f"generation need{' (with the overlapped restore)' if overlap else ''}, the margin and the "
                        f"NVENC probe")
    return record


def create_provider(*, config: Mapping[str, Any], pipeline: Any = None, server_args: Any = None
                    ) -> LongCatSessionProvider:
    """SGLang's provider factory (``reelcache_session_provider``): the worker's archive stack + provider."""
    cfg = LongCatSessionConfig.from_dict(config)
    model = mb.model_identity()
    startup: dict = {"config": {key: getattr(cfg, key) for key in cfg.__dataclass_fields__}}
    owned: tuple = ()
    plan_device = cfg.resolved_plan_device()
    startup["plan_device"] = plan_device
    cuda = torch.cuda.is_available()
    device_index = torch.cuda.current_device() if cuda else None
    resident_bytes = torch.cuda.memory_allocated(device_index) if cuda else 0
    total_bytes = torch.cuda.get_device_properties(device_index).total_memory if cuda else 0
    epoch_bytes = mb.epoch_nbytes(model)
    _LOG.info("LongCat session provider: store integrity verify_reads=%s audit_on_open=%s on_open_error=%s",
              cfg.verify_reads, cfg.audit_on_open, cfg.on_open_error)
    stack = None
    try:
        if cuda:
            stack = mb.gpu_archive_stack(cfg.store_root, model, device_index=device_index, plan_device=plan_device,
                                         method=cfg.method(), arena_bytes=cfg.restore_arena_bytes,
                                         basis_budget_bytes=cfg.basis_budget_bytes,
                                         prepared_budget_bytes=cfg.prepared_budget_bytes,
                                         plan_memory_bytes=cfg.plan_memory_bytes, **cfg.integrity(), **cfg.staging(),
                                         gated_planner=cfg.planning_gate == "group", encoder=cfg.encoder)
            encoder = dict(stack.config["encoder"])
        else:
            encoder = mb.select_encoder(cfg.encoder, None)[1]
        startup["encoder"] = encoder
        fixed = host_fixed_bytes(cfg, cuda=cuda, encoder=encoder)
        host = plan_host_slots(cfg, epoch_bytes=epoch_bytes, fixed_bytes=fixed["bytes"])
        gpu = plan_gpu_raw(cfg, enabled=host["raw_tier"], total_bytes=total_bytes, resident_bytes=resident_bytes,
                           epoch_bytes=epoch_bytes, reserved_bytes=int(encoder.get("probe_device_bytes") or 0))
    except BaseException:
        if stack is not None:
            stack.close()
        raise
    startup["raw_tier"] = {"requested": cfg.raw_tier, "enabled": host["raw_tier"], "reason": host["reason"],
                           "archival": host["archival"], "raw_restore": cfg.raw_restore,
                           "host_raw_pinned": cfg.host_raw_pinned, "epoch_bytes": epoch_bytes,
                           "capture_slots": host["capture_slots"], "fixed_host": fixed, "host": host, "gpu": gpu}
    _LOG.info("LongCat session provider: encoder %s (requested %r): %s", encoder.get("selected"), cfg.encoder,
              encoder.get("reason"))
    (_LOG.info if host["archival"] else _LOG.warning)(
        "LongCat session provider: raw tier %s (requested %s): %s; host %d slot(s), budget %.1f GiB of a %.1f GiB "
        "worker share (%s), fixed host consumers %.1f GiB; GPU raw budget %.1f GiB (%s); raw_restore %s",
        "on" if host["raw_tier"] else "off", cfg.raw_tier, host["reason"], host["capture_slots"],
        host["budget_bytes"] / (1 << 30), host["worker_headroom_bytes"] / (1 << 30), host["memory"].get("source"),
        fixed["bytes"] / (1 << 30), gpu["budget_bytes"] / (1 << 30), gpu["reason"], cfg.raw_restore)
    decision = resolve_schedule(
        cfg.schedule, representation=cfg.representation, defer_planning=cfg.defer_planning, plan_device=plan_device,
        plan_memory_bytes=cfg.plan_memory_bytes, resident_bytes=resident_bytes, total_bytes=total_bytes,
        raw_tier=host["raw_tier"])
    startup["schedule"] = decision
    startup["eager_overlap"] = decision["eager_overlap"]
    schedule = decision["chosen"]
    _LOG.info("LongCat session provider: archival schedule %r (requested %r, plan device %s, defer_planning %s): %s",
              schedule, cfg.schedule, plan_device, decision["defer_planning"], decision["reason"])
    _LOG.info("LongCat session provider: defer_planning %s (requested %s): %s; planner memory released when a "
              "memory-critical section opens: %s", decision["defer_planning"], cfg.defer_planning,
              decision["defer_planning_reason"], decision["release_planner_memory"])
    archive = (cfg.representation,) if host["archival"] else ()
    tiers = {"capture_slots": max(1, host["capture_slots"]), "raw_tier": host["raw_tier"],
             "gpu_raw_budget_bytes": gpu["budget_bytes"], "raw_restore": cfg.raw_restore,
             "pin_memory": cfg.host_raw_pinned and cuda}
    if cuda:
        try:
            backend = stack.backend(archive=archive, schedule=schedule, deferred_sources="host", **tiers)
        except BaseException:
            stack.close()
            raise
        backend.set_defer_planning(decision["defer_planning"],
                                   release_cached_memory=decision["release_planner_memory"])
        owned = (stack,)
        startup["planning_gate"] = stack.config["planning_gate"]
        _LOG.info("LongCat session provider: planning gate %s (requested %r): a memory-critical section waits for "
                  "the planner's current %s", startup["planning_gate"], cfg.planning_gate,
                  "matrix group" if startup["planning_gate"]["granularity"] == "group" else "node")
        startup["staging_engine"] = stack.config["staging_engine"]
        _LOG.info("LongCat session provider: cold-session staging engine %s", startup["staging_engine"])
        if cfg.warmup is not False:
            started = time.perf_counter()
            report = backend.prewarm_process()
            startup["warmup"] = {**report, "seconds": time.perf_counter() - started}
    else:
        from reelcache.archive_pipeline import ArchivePipeline
        from reelcache.archive_restore import SoftwareLineageRestorer
        from reelcache.lineage_commit import ArchiveOptions
        from reelcache.predictor_registry import RankPredictorRegistry

        runtime = mb.open_runtime(cfg.store_root, model, **cfg.integrity())
        archive_pipeline = None
        if cfg.representation == "svd_hevc":
            archive_pipeline = ArchivePipeline(runtime, method=cfg.method(), registry=RankPredictorRegistry(
                "longcat_sessions"), options=ArchiveOptions(encode_threads=1, plan_memory_bytes=cfg.plan_memory_bytes),
                device="cpu", encode_workers=1)
        backend = mb.ReelCacheConditioningCacheBackend(runtime, pipeline=archive_pipeline,
                                                    restorer=SoftwareLineageRestorer(runtime), archive=archive,
                                                    schedule=schedule, **tiers)
        backend.encoder_info = dict(encoder)
        owned = tuple(item for item in (_Closer(archive_pipeline, runtime),) if item is not None)
        startup["warmup"] = {"skipped": "no CUDA device (software restorer)"}
        startup["staging_engine"] = mb.staging_description(None, reason="software restorer (no CUDA): no packed "
                                                                          "device restore to stage")
    try:
        provider = LongCatSessionProvider(cfg, backend=backend, owned=owned, startup=startup)
    except BaseException:
        for item in owned:  # e.g. on_open_error "raise" on a damaged store: release the runtime before failing
            try:
                item.close()
            except Exception:  # noqa: BLE001 - the attach error is the one to report
                pass
        raise
    _ACTIVE[:] = [provider]
    if pipeline is not None:
        setattr(pipeline, "longcat_session_provider", provider)
    return provider


class _Closer:
    """Closes a CPU archive pipeline, then its runtime."""

    def __init__(self, pipeline, runtime):
        self.pipeline, self.runtime = pipeline, runtime

    def close(self) -> None:
        try:
            if self.pipeline is not None and not self.pipeline._closed:
                self.pipeline.close()
        finally:
            if not self.runtime._closed:
                self.runtime.close()


__all__ = ["CACHE_BACKEND_NAME", "EAGER_OVERLAP_FIXED_BYTES", "GENERATION_ACTIONS", "INFLIGHT_OPS", "LongCatSessionConfig",
           "LongCatSessionProvider", "PROVIDER_SPEC", "PROMPT_CHANGE", "RequestPlan", "SESSION_RUN_KEY", "SessionRun",
           "active_provider", "check_eager_overlap", "create_provider", "eager_overlap", "host_fixed_bytes",
           "plan_gpu_raw", "plan_host_slots", "request_fields", "resolve_schedule"]
