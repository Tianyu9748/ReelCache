"""Session controls of the SGLang bridge (Step 14): the worker's typed, plain-JSON control surface.

One entry point per worker, ``CausalWanKVBridge.control(op, args) -> dict``: the SGLang server sends
``CausalKVSessionControlReq(op, args)`` to the GPU worker, which calls it; Python callers inside the worker use
the same entry point. Arguments and results are plain JSON. No argument is a filesystem path: sessions and
checkpoints are names (``[A-Za-z0-9][A-Za-z0-9._-]{0,127}``) or node references the archive returned. Failures
raise :class:`SessionControlError` (``code`` + HTTP ``status``):

=====================  ======  ==========================================================================
code                   status  meaning
=====================  ======  ==========================================================================
``invalid_argument``   400     unknown op, missing or malformed argument
``unknown_session``    404     no open or persisted session of that id
``unknown_checkpoint`` 404     no such named checkpoint / chunk boundary / node on the session
``unknown_request``    404     no recent request of that id
``illegal_transition`` 409     e.g. fork target exists, rewind to a non-ancestor, named checkpoint exists
``busy``               409     the session (or, for ``gc``, the worker) runs a request
``incompatible_model`` 422     the model has no native form of the request (e.g. no prompt change)
``archival_failure``   500     the session's archival failed (surfaced once; it continues from its last
                               published checkpoint)
``integrity``          500     archive integrity check failed
``timeout``            504     the published frontier did not arrive in time
``internal``           500     anything else
=====================  ======  ==========================================================================

Generation actions are ordinary video requests carrying the ``causal_kv_session`` handle (``generate``,
``resume``, ``fork``, ``rewind``) and, for a prompt change, the model's native prompt-change fields
(:func:`prompt_change_capability`). The ops below generate nothing:

* ``capabilities`` ``{}`` -> model, representation, boundary unit, ops, generation actions, the native
  prompt-change form (or why the model has none) and the attach warm-up report.
* ``list`` ``{}`` -> ``{"sessions": [...]}``: open sessions (status, heads, pending commits, resident KV) and
  persisted ones (``status: "closed"``), each with its named checkpoints.
* ``describe`` ``{session_id, history?: int = 50}`` -> the session (open: live state, pending commits, recent
  events and request ids; persisted: from its head, never reopened).
* ``checkpoints`` ``{session_id}`` -> clean boundaries of the head path (drained first when open) and the named
  checkpoints.
* ``request`` ``{request_id}`` -> the bridge summary of a recent request (restore report, chunk marks and
  digests, first-new-chunk timings, archive frontiers).
* ``open`` ``{session_id}`` -> reopen a persisted session and prepare its history for restore (the prepared
  state of the restore requirement).
* ``pause`` ``{session_id}`` -> idle session: drain and return its durable state. Running session (in-flight
  pause, e.g. from the scheduler's chunk-boundary poll): flag it to stop at the next clean boundary and return
  ``{"status": "pause_requested", "deferred": true}``; the caller acknowledges the pause by calling ``pause``
  again once the request returned (the SGLang scheduler does so before it replies).
* ``checkpoint`` ``{session_id, name?, replace?: bool}`` -> drain (the published frontier reaches the generated
  head) and return the durable head; ``name`` pins it as a named checkpoint (garbage collection keeps it).
* ``fork`` ``{session_id, target, checkpoint?}`` -> new session ``target`` at a clean boundary of the source
  (chunk count, node reference or name; default its head); copy-on-write: the source is never written.
* ``rewind`` ``{session_id, checkpoint}`` -> move the head back to an ancestor checkpoint; the abandoned
  descendants stay reachable only through other sessions or named checkpoints.
* ``close`` ``{session_id, mode?: "drain" | "abort"}`` -> unload the session: publish (``drain``) or retire
  (``abort``) its unpublished commits, drop its resident native KV and pipeline state. It stays persisted.
* ``delete`` ``{session_id, checkpoint?: name, gc?: bool = true}`` -> delete one named checkpoint, or the whole
  session (head, named checkpoints; unpublished commits retired), then collect garbage: objects no other
  session or named checkpoint reaches are reclaimed (never objects of another branch).
* ``gc`` ``{}`` -> drain every open session, then collect unreachable objects of the store.
* ``raw_tier`` ``{session_id?, action?: "stats" | "place" | "demote" | "drop" | "reset_peaks", placement?, wait?}``
  -> freeze D raw-chunk tier (Step 7 ``reelcache.raw_tier``): ``stats`` (default) the tier's budgets, bytes, entries
  and peaks (plus the session's held entries); ``place`` ``placement = {chunk index or node ref: "gpu_raw" |
  "host_raw" | "compressed"}`` (the only promotion to GPU; deterministic mixed tiers); ``demote`` GPU raw -> host
  raw (session or all); ``drop`` drains, then drops both raw tiers (compressed only); ``reset_peaks``. A
  ``invalid_argument`` error without a raw tier (``raw_tier=False`` or RAW representation).

Concurrency: ops serialize with requests on the worker (the SGLang scheduler is single-threaded); while a request
runs, only ``capabilities``, ``list``, ``describe``, ``request`` and the in-flight ``pause`` are served (from
the chunk-boundary hook); every other op on the running session fails with ``busy``.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping

from ...errors import (CapacityError, ConfigurationError, IntegrityError, ReelCacheError, PendingError, StateError,
                       UnsupportedError)
from .session import NAME_PATTERN

ERROR_STATUS = {"invalid_argument": 400, "unknown_session": 404, "unknown_checkpoint": 404,
                "unknown_request": 404, "illegal_transition": 409, "busy": 409, "incompatible_model": 422,
                "archival_failure": 500, "integrity": 500, "timeout": 504, "internal": 500}
#: Ops a caller may run while a session request is in flight (served from the chunk-boundary hook).
INFLIGHT_OPS = ("capabilities", "list", "describe", "request", "pause")
#: Native prompt change of a model, by the bridge profile's recache policy (report §3.7).
PROMPT_CHANGE = {
    "none": {
        "supported": False, "mode": None, "request_fields": [],
        "reason": "the original repository generates a video under one prompt (no prompt switch); a request "
                  "that resumes, forks or rewinds the session under another prompt is rejected",
    },
    "kv_recache_on_prompt_switch": {
        "supported": True, "mode": "recache_epoch", "request_fields": ["prompt_switches"],
        "reason": "LongLive 1.0: a prompt switch before chunk n recaches the local window under the new prompt "
                  "(the sink keeps epoch e0) and publishes a cache epoch; resume/fork/rewind requests carry the "
                  "whole switch schedule; restored blocks must match it, later blocks may differ",
    },
    "optional_scene_cut_clean_recache": {
        "supported": True, "mode": "multi_shot", "request_fields": ["shot_prompts", "shot_durations",
                                                                    "chunks_per_shot", "scene_cut_prefix"],
        "reason": "LongLive 2.0: native multi-shot; a new shot starts at a chunk boundary (scene-cut prefix, "
                  "pinned shot sink, multi-shot RoPE offset); requests carry the whole shot list; restored "
                  "blocks must match it, later blocks may differ",
    },
}
#: Workload v2 (freeze W1a; user decision ``cross_attention_only``): CausVid and RAVEN (bridge profiles with recache
#: ``none``, by base profile name: ``causvid``, ``causvid:<visibility>``, ``raven``) change prompts through LongLive
#: 1.0's ``prompt_switches``. The profiles' fingerprints are unchanged, so the bridge cannot read this from their data.
CROSS_ATTENTION_SWITCH_PROFILES = ("causvid", "raven")
CROSS_ATTENTION_SWITCH = {
    "supported": True, "mode": "cross_attention_only", "request_fields": ["prompt_switches"],
    "reason": "CausVid / RAVEN (workload v2): a prompt switch before chunk n resets the cross-attention caches, so "
              "blocks n.. cross-attend to the new prompt; archived and self-attention KV are never recomputed (no "
              "recache; any chunk n > 0). A session that switches prompts carries prompt_switches (possibly empty) "
              "with the whole schedule on every request (per-block conditioning); restored blocks must match it, "
              "later blocks may differ",
}


class SessionControlError(ReelCacheError):
    """A failed session control: ``code`` (see the module table), HTTP ``status`` and JSON ``details``."""

    def __init__(self, code: str, message: str, **details: Any):
        if code not in ERROR_STATUS:
            raise ValueError(f"unknown session control error code {code!r}")
        super().__init__(message)
        self.code, self.status, self.details = code, ERROR_STATUS[code], details

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "status": self.status, "message": str(self), **self.details}


def prompt_change_capability(profile: Any) -> dict[str, Any]:
    """The model's native prompt change (``PROMPT_CHANGE`` by the profile's recache policy; CausVid / RAVEN:
    ``CROSS_ATTENTION_SWITCH``)."""
    entry = PROMPT_CHANGE.get(getattr(profile, "recache", None))
    if entry is PROMPT_CHANGE["none"] and str(getattr(profile, "name", "")).split(":", 1)[0] in \
            CROSS_ATTENTION_SWITCH_PROFILES:
        entry = CROSS_ATTENTION_SWITCH
    if entry is None:
        return {"supported": False, "mode": None, "request_fields": [],
                "reason": f"no prompt-change semantics are known for recache policy {profile.recache!r}"}
    return {**entry, "request_fields": list(entry["request_fields"])}


def _gpu_memory(bridge: Any) -> dict[str, Any] | None:
    """The worker's PyTorch allocator counters (bytes) on the bridge's CUDA device, with its cap if any."""
    device = getattr(bridge, "device", None)
    if device is None or getattr(device, "type", None) != "cuda":
        return None
    import torch

    return {"allocated": torch.cuda.memory_allocated(device), "reserved": torch.cuda.memory_reserved(device),
            "max_allocated": torch.cuda.max_memory_allocated(device),
            "max_reserved": torch.cuda.max_memory_reserved(device),
            "cap": (bridge.startup or {}).get("cuda_memory_cap")}


def _quarantined(archive: Any) -> dict[str, str]:
    """Sessions whose head was found damaged when the store was opened (CR-17-1; {} for archives without quarantine)."""
    quarantined = getattr(archive, "quarantined", None)
    return dict(quarantined()) if callable(quarantined) else {}


#: Raw-tier stats keys a control reports (bytes per tier now and at peak, entries, host buffers, waits; freeze E: the
#: host-block pool; freeze F: the device-pressure response with its recent events); the tier's event log stays in
#: ``archive.stats()``.
RAW_TIER_KEYS = ("enabled", "disabled_reason", "budgets", "gpu_bytes", "host_bytes", "unpublished_bytes",
                 "reserved_bytes", "unpublished_entries", "entries", "gpu_peak_bytes", "host_peak_bytes",
                 "unpublished_peak_bytes", "host_buffers", "restores", "restore_misses", "spills",
                 "evictions_gpu", "evictions_host", "last_resort_backpressure_seconds",
                 "last_resort_backpressure_waits", "last_resort_causes", "host_pool_bytes", "host_pool_peak_bytes",
                 "pressure")


def _raw_tier(archive: Any) -> Any:
    tier = getattr(archive, "raw_tier", None)
    return tier if tier is not None and callable(getattr(tier, "stats", None)) else None


def _retained_caches(archive: Any) -> dict[str, Any]:
    """Device caches the archive keeps across restores (freeze F, coordinator R-D2b): ``retained_caches`` holds only
    device byte fields (a verifier sums them and subtracts the sum from the allocated memory at teardown) -- the
    restorer's canvas arena (decoded canvases, in use or free for reuse), its resident bases, and prepared packs
    (a host cache: 0 device bytes). ``retained_host_caches`` and ``retained_caches_detail`` carry host bytes, peaks
    and budgets (never summed)."""
    device = {"canvas_arena_bytes": 0, "resident_bases_bytes": 0, "prepared_packs_bytes": 0}
    host: dict[str, Any] = {"prepared_packs_bytes": 0}
    detail: dict[str, Any] = {}
    restorer = getattr(archive, "restorer", None)
    packed = getattr(restorer, "restorer", None)
    arena = getattr(packed, "arena", None)
    try:
        if arena is not None and callable(getattr(arena, "report", None)):
            report = arena.report()
            device["canvas_arena_bytes"] = int(report.get("used_bytes") or 0)
            detail["canvas_arena"] = {key: report.get(key) for key in ("budget_bytes", "peak_bytes", "free",
                                                                       "retiring", "allocations", "evictions")}
        bases = getattr(restorer, "bases", None)
        if bases is not None and callable(getattr(bases, "stats", None)):
            stats = bases.stats()
            device["resident_bases_bytes"] = int(stats.get("resident_bytes") or 0)
            detail["resident_bases"] = {key: stats.get(key) for key in ("versions", "pinned", "budget_bytes",
                                                                         "peak_resident_bytes")}
        prepared = getattr(restorer, "prepared", None)
        if prepared is not None and callable(getattr(prepared, "stats", None)):
            stats = prepared.stats()
            host["prepared_packs_bytes"] = int(stats.get("bytes") or 0)
            detail["prepared_packs"] = {key: stats.get(key) for key in ("packets", "budget_bytes")}
    except Exception as exc:  # noqa: BLE001 - records only
        detail["error"] = f"{type(exc).__name__}: {exc}"
    return {"retained_caches": device, "retained_host_caches": host, "retained_caches_detail": detail}


def _raw_tier_stats(archive: Any) -> dict[str, Any]:
    """The worker's raw-tier state (freeze D): description and current / peak bytes; ``enabled: False`` (with the
    reason) without one."""
    describe = getattr(archive, "raw_tier_description", None)
    description = dict(describe()) if callable(describe) else {"enabled": False,
                                                               "reason": "archive without a raw tier"}
    tier = _raw_tier(archive)
    if tier is None:
        return description
    stats = tier.stats()
    return {**description, **{key: stats[key] for key in RAW_TIER_KEYS if key in stats}}


def _raw_tier_session(archive: Any, session_id: str) -> dict[str, Any] | None:
    """The raw-tier entries a session holds: count, bytes with a GPU copy / a complete host copy, unpublished."""
    tier = _raw_tier(archive)
    if tier is None or not callable(getattr(tier, "entries", None)):
        return None
    entries = [entry.describe() for entry in tier.entries(holder=session_id)]
    return {"entries": len(entries),
            "gpu_bytes": sum(item["bytes"] for item in entries if item.get("device")),
            "host_bytes": sum(item["bytes"] for item in entries if item.get("host") == "complete"),
            "unpublished_bytes": sum(item["bytes"] for item in entries
                                     if item.get("state") in ("submitted", "planning", "fitted")),
            "shared_entries": sum(1 for item in entries if len(item.get("holders") or ()) > 1)}


def _placement(value: Any) -> dict[Any, str]:
    """``{chunk index or node ref: tier}`` from JSON (object keys are strings: decimal keys are chunk indices)."""
    if not isinstance(value, Mapping) or not value:
        raise SessionControlError("invalid_argument", "placement must be a nonempty object {chunk or node: tier}")
    placement: dict[Any, str] = {}
    for key, tier in value.items():
        if isinstance(key, str) and key.isdigit():
            key = int(key)
        elif not (type(key) is int and key >= 0) and not (isinstance(key, str) and key.startswith("manifest:")):
            raise SessionControlError("invalid_argument", f"placement keys are chunk indices or node references; "
                                                          f"got {key!r}")
        if tier not in ("gpu_raw", "host_raw", "compressed"):
            raise SessionControlError("invalid_argument", f"placement tiers are gpu_raw, host_raw or compressed; "
                                                          f"got {tier!r}")
        placement[key] = tier
    return placement


def error_of(exc: BaseException) -> SessionControlError:
    """Map a bridge/controller exception to a typed control error."""
    if isinstance(exc, SessionControlError):
        return exc
    cause = exc.__cause__
    message = str(exc) or type(exc).__name__
    if isinstance(exc, PendingError):
        if "timed out" in message:
            return SessionControlError("timeout", message)
        return SessionControlError("busy", message)
    if isinstance(exc, StateError) and cause is not None and "archival failed" in message:
        return SessionControlError("archival_failure", message, cause=repr(cause))
    if isinstance(exc, UnsupportedError):
        return SessionControlError("incompatible_model", message)
    if isinstance(exc, IntegrityError):
        return SessionControlError("integrity", message)
    if isinstance(exc, (ConfigurationError, CapacityError)):
        return SessionControlError("invalid_argument", message)
    if isinstance(exc, StateError):
        return SessionControlError("illegal_transition", message)
    if type(exc).__name__ == "ArchiveFailure":
        return SessionControlError("archival_failure", message)
    return SessionControlError("internal", f"{type(exc).__name__}: {message}")


def jsonable(value: Any) -> Any:
    """Plain-JSON copy of a control result (mappings, sequences, numbers, strings; anything else as its repr;
    non-finite floats as strings)."""
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    return repr(value)


def _name(args: Mapping[str, Any], key: str, *, required: bool = True, label: str | None = None) -> str | None:
    value = args.get(key)
    if value is None:
        if required:
            raise SessionControlError("invalid_argument", f"{label or key} is required")
        return None
    if not isinstance(value, str) or NAME_PATTERN.fullmatch(value) is None:
        raise SessionControlError("invalid_argument",
                                  f"{label or key} must match {NAME_PATTERN.pattern} (names, never paths); "
                                  f"got {value!r}")
    return value


def _session_ref(args: Mapping[str, Any], key: str = "session_id") -> str:
    """A session id: any id without '/' (ids created before Step 14 may fall outside the name pattern)."""
    value = args.get(key)
    if not isinstance(value, str) or not value.strip() or "/" in value or "\x00" in value or len(value) > 256:
        raise SessionControlError("invalid_argument", f"{key} must be a nonempty session id without '/'")
    return value


def _checkpoint_arg(value: Any) -> Any:
    if value is None or (type(value) is int and value >= 0):
        return value
    if isinstance(value, str) and (NAME_PATTERN.fullmatch(value) is not None or value.startswith("manifest:")):
        return value
    raise SessionControlError("invalid_argument", "checkpoint must be a chunk count, a node reference or a "
                                                  f"checkpoint name; got {value!r}")


def _known(args: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = set(args) - allowed
    if unknown:
        raise SessionControlError("invalid_argument", f"unknown arguments {sorted(unknown)}")


class SessionControls:
    """The ops of the module docstring over one :class:`~.bridge.CausalWanKVBridge`."""

    def __init__(self, bridge: Any):
        self.bridge = bridge
        self._ops: dict[str, tuple[Callable[[Mapping[str, Any]], dict[str, Any]], set[str]]] = {
            "capabilities": (self.capabilities, set()),
            "list": (self.list_sessions, set()),
            "describe": (self.describe, {"session_id", "history"}),
            "checkpoints": (self.checkpoints, {"session_id"}),
            "request": (self.request, {"request_id"}),
            "open": (self.open, {"session_id"}),
            "pause": (self.pause, {"session_id"}),
            "checkpoint": (self.checkpoint, {"session_id", "name", "replace"}),
            "fork": (self.fork, {"session_id", "target", "checkpoint"}),
            "rewind": (self.rewind, {"session_id", "checkpoint"}),
            "close": (self.close, {"session_id", "mode"}),
            "delete": (self.delete, {"session_id", "checkpoint", "gc"}),
            "gc": (self.gc, set()),
            "raw_tier": (self.raw_tier, {"session_id", "action", "placement", "wait"}),
        }

    @property
    def ops(self) -> tuple[str, ...]:
        return tuple(self._ops)

    def dispatch(self, op: Any, args: Any = None) -> dict[str, Any]:
        """Run one op; raises :class:`SessionControlError` only. Results are plain JSON."""
        if not isinstance(op, str) or op not in self._ops:
            raise SessionControlError("invalid_argument", f"unknown session control op {op!r}; one of "
                                                          f"{sorted(self._ops)}")
        if args is None:
            args = {}
        if not isinstance(args, Mapping):
            raise SessionControlError("invalid_argument", "control arguments must be a JSON object")
        handler, allowed = self._ops[op]
        _known(args, allowed)
        started = time.perf_counter()
        try:
            result = handler(args)
        except BaseException as exc:  # noqa: BLE001 - every failure leaves as a typed control error
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise error_of(exc) from exc
        return jsonable({"op": op, **result, "control_seconds": time.perf_counter() - started})

    def dispatch_inflight(self, op: Any, args: Any = None) -> dict[str, Any] | None:
        """A control received at a chunk boundary of a running request: served now if it cannot block generation
        (``INFLIGHT_OPS``; ``pause`` only for a running session: it sets the flag), else None (queue it)."""
        if op not in INFLIGHT_OPS:
            return None
        if op == "pause":
            if not isinstance(args, Mapping):
                return self.dispatch(op, args)  # raises the argument error now
            session_id = args.get("session_id")
            if not isinstance(session_id, str) or not self._running(session_id):
                return None  # an idle session's pause drains: after the running request
        return self.dispatch(op, args)

    # -- helpers ---------------------------------------------------------------------------------------------
    @property
    def _controller(self):
        return self.bridge.controller

    def _running(self, session_id: str) -> bool:
        state = self._controller.state(session_id)
        return state is not None and state.status == "running"

    def _require(self, session_id: str) -> None:
        if self._controller.state(session_id) is None and self.bridge.archive.head(session_id) is None:
            raise SessionControlError("unknown_session", f"unknown causal KV session {session_id!r}")

    def _require_idle(self, session_id: str, op: str) -> None:
        self._require(session_id)
        if self._running(session_id):
            raise SessionControlError("busy", f"session {session_id!r} is running a request; {op} waits for it "
                                              "to return (pause it first)")

    def _check_checkpoint(self, session_id: str, checkpoint: Any) -> None:
        """Unknown names and chunk counts beyond the session fail as ``unknown_checkpoint`` up front."""
        if isinstance(checkpoint, str) and NAME_PATTERN.fullmatch(checkpoint) is not None:
            if checkpoint not in self._controller.named_checkpoints(session_id):
                raise SessionControlError("unknown_checkpoint",
                                          f"session {session_id!r} has no checkpoint named {checkpoint!r}")
        elif type(checkpoint) is int:
            state = self._controller.state(session_id)
            chunks = (state.generated_chunks if state is not None
                      else self._controller.persisted(session_id)["generated_chunks"])
            if checkpoint > chunks:
                raise SessionControlError("unknown_checkpoint", f"session {session_id!r} has {chunks} chunks; no "
                                                                f"checkpoint after {checkpoint}")

    def _summary(self, session_id: str) -> dict[str, Any]:
        state = self._controller.state(session_id)
        resident = self.bridge.resident
        named = self._controller.named_checkpoints(session_id)
        if state is not None:
            with state.lock:
                entry = {"session_id": session_id, "open": True, "status": state.status, "head": state.head_ref,
                         "generated_chunks": state.generated_chunks,
                         "pending": [item["node_id"] for item in state.pending], "published": not state.pending,
                         "pause_requested": state.pause_requested, "archive_failures": state.archive_failures}
        else:
            persisted = self._controller.persisted(session_id)
            if persisted is None:
                raise SessionControlError("unknown_session", f"unknown causal KV session {session_id!r}")
            entry = {"open": False, **{key: persisted[key] for key in ("session_id", "status", "head",
                                                                       "generated_chunks", "pending",
                                                                       "published")}}
        entry["resident"] = resident is not None and resident.get("session_id") == session_id
        entry["named_checkpoints"] = named
        raw = _raw_tier_session(self.bridge.archive, session_id)
        if raw is not None:
            entry["raw_tier"] = raw
        return entry

    # -- ops -------------------------------------------------------------------------------------------------
    def capabilities(self, args: Mapping[str, Any]) -> dict[str, Any]:
        bridge = self.bridge
        profile, config = bridge.profile, bridge.config
        warmup = bridge.startup.get("warmup") or {}
        return {"model": profile.name, "representation": config.representation, "boundary": "chunk",
                "chunk_frames": profile.chunk_frames, "streams": list(profile.cfg_streams),
                "visibility": profile.visibility, "ops": sorted(self._ops), "inflight_ops": list(INFLIGHT_OPS),
                "generation_actions": ["generate", "resume", "fork", "rewind"],
                "prompt_change": prompt_change_capability(profile), "durability": config.durability,
                "resident_reuse": config.resident_reuse, "prewarm": config.prewarm,
                "decode_context_latents": config.decode_context_latents,
                "archive_memory": {"max_source_chunks": config.max_source_chunks,
                                   "plan_memory_bytes": config.plan_memory_bytes,
                                   "defer_archival_during_decode": config.defer_archival_during_decode,
                                   "gpu_raw_reserve_bytes": getattr(config, "gpu_raw_reserve_bytes", None)},
                "observe_plain_requests": config.observe_plain_requests, "realtime": False,
                "integrity": {"verify_reads": config.verify_reads,
                              "audit_on_open": getattr(config, "audit_on_open", None),
                              "on_open_error": getattr(config, "on_open_error", None),
                              "quarantined": _quarantined(bridge.archive)},
                "warmup": {key: warmup.get(key) for key in ("seconds", "components", "skipped") if key in warmup},
                "gpu_memory": _gpu_memory(bridge), "raw_tier": _raw_tier_stats(bridge.archive),
                "encoder": dict(getattr(bridge.archive, "encoder_record", None) or {}) or None,
                # F3: memory-critical section records no request summary could take (kept, never dropped)
                "unattributed_memory_critical": list(getattr(bridge, "unattributed_sections", None) or []),
                # F5: the layer-overlapped restore. ``default``: the config's ``overlap_restore`` (on when the config
                # has no such field), overridable per request by the session handle's ``overlap_restore``;
                # ``available``: the archive restores overlapped (lineage archives; not the RAW reference)
                "overlap_restore": {"default": getattr(bridge, "overlap_restore_default", False),
                                    "available": bool(getattr(bridge, "_overlap_capable", False)),
                                    "config_field": hasattr(config, "overlap_restore")},
                # F5 (cut 3): restore plans cached by head, stream, next chunk and cache geometry
                "plan_cache": bridge.plans.describe() if hasattr(bridge, "plans") else None,
                **_retained_caches(bridge.archive)}

    def list_sessions(self, args: Mapping[str, Any]) -> dict[str, Any]:
        controller, archive = self._controller, self.bridge.archive
        names = set(controller.sessions()) | set(archive.sessions())
        sessions = []
        for session_id in sorted(names):
            try:
                sessions.append(self._summary(session_id))
            except SessionControlError:
                continue  # deleted meanwhile
        return {"sessions": sessions, "resident": self.bridge.resident}

    def describe(self, args: Mapping[str, Any]) -> dict[str, Any]:
        session_id = _session_ref(args)
        history = args.get("history", 50)
        if type(history) is not int or history < 0:
            raise SessionControlError("invalid_argument", "history must be a nonnegative int")
        entry = self._summary(session_id)
        state = self._controller.state(session_id)
        if state is not None:
            described = state.describe()
            events = described.pop("history")
            entry.update({key: value for key, value in described.items() if key not in entry})
            entry["history"] = events[-history:] if history else []
            entry["requests"] = self.bridge.session_requests(session_id)
        else:
            entry.update({key: value for key, value in self._controller.persisted(session_id).items()
                          if key not in entry})
        return entry

    def checkpoints(self, args: Mapping[str, Any]) -> dict[str, Any]:
        session_id = _session_ref(args)
        self._require_idle(session_id, "checkpoints")
        controller = self._controller
        if controller.state(session_id) is not None:
            boundaries = controller.checkpoints(session_id)
        else:
            from .session import partial_epoch_nodes

            path = self.bridge.archive.path(self.bridge.archive.head(session_id))
            partial = partial_epoch_nodes(path)
            boundaries, chunks = [], 0
            for node in path:
                chunks += node.kind == "chunk"
                boundaries.append({"chunks": chunks, "node": node.ref, "kind": node.kind,
                                   "checkpoint": node.ref not in partial})
        named = controller.named_checkpoints(session_id)
        on_path = {item["node"]: item["chunks"] for item in boundaries}
        return {"session_id": session_id, "checkpoints": boundaries,
                "named": {name: {"node": node, "on_head_path": node in on_path, "chunks": on_path.get(node)}
                          for name, node in named.items()}}

    def request(self, args: Mapping[str, Any]) -> dict[str, Any]:
        request_id = args.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise SessionControlError("invalid_argument", "request_id must be a nonempty string")
        summary = self.bridge.request_summary(request_id)
        if summary is None:
            raise SessionControlError("unknown_request", f"no recent causal KV request {request_id!r}")
        return {"request_id": request_id, "summary": summary}

    def open(self, args: Mapping[str, Any]) -> dict[str, Any]:
        session_id = _session_ref(args)
        self._require(session_id)
        was_open = self._controller.state(session_id) is not None
        started = time.perf_counter()
        described = self._controller.open(session_id)
        described.pop("history", None)
        state = self._controller.state(session_id)
        reopened = [] if state is None else [item for item in state.history if item["event"] == "reopened"]
        return {**described, "was_open": was_open, "open_seconds": time.perf_counter() - started,
                "prewarm_seconds": None if was_open or not reopened else reopened[-1].get("prewarm_seconds")}

    def pause(self, args: Mapping[str, Any]) -> dict[str, Any]:
        session_id = _session_ref(args)
        self._require(session_id)
        controller = self._controller
        state = controller.state(session_id)
        if state is None:  # closed: everything it holds is published
            return {**self._summary(session_id), "deferred": False, "wait_seconds": 0.0}
        if state.status == "running":
            controller.request_pause(session_id)
            return {"session_id": session_id, "status": "pause_requested", "deferred": True,
                    "generated_chunks": state.generated_chunks}
        result = controller.checkpoint(session_id)
        return {**self._summary(session_id), "deferred": False, "wait_seconds": result["wait_seconds"]}

    def checkpoint(self, args: Mapping[str, Any]) -> dict[str, Any]:
        session_id = _session_ref(args)
        name = _name(args, "name", required=False, label="checkpoint name")
        replace = args.get("replace", False)
        if type(replace) is not bool:
            raise SessionControlError("invalid_argument", "replace must be a bool")
        self._require_idle(session_id, "checkpoint")
        controller = self._controller
        if controller.state(session_id) is not None:
            result = controller.checkpoint(session_id)
        else:
            persisted = controller.persisted(session_id)
            result = {"session_id": session_id, "head": persisted["head"],
                      "generated_chunks": persisted["generated_chunks"], "published": True, "wait_seconds": 0.0}
        if name is not None:
            if NAME_PATTERN.fullmatch(session_id) is None:
                raise SessionControlError("invalid_argument", f"session {session_id!r} cannot hold named "
                                                              f"checkpoints (ids must match {NAME_PATTERN.pattern})")
            try:
                previous = controller.pin(session_id, name, result["head"], replace=replace)
            except StateError as exc:
                raise SessionControlError("illegal_transition", str(exc)) from exc
            result = {**result, "name": name, "replaced": previous if previous != result["head"] else None}
        return result

    def fork(self, args: Mapping[str, Any]) -> dict[str, Any]:
        source = _session_ref(args)
        target = _name(args, "target", label="fork target")
        checkpoint = _checkpoint_arg(args.get("checkpoint"))
        self._require_idle(source, "fork")
        if self._controller.state(target) is not None or self.bridge.archive.head(target) is not None:
            raise SessionControlError("illegal_transition", f"fork target {target!r} already exists")
        self._check_checkpoint(source, checkpoint)
        state = self._controller.fork(source, target, checkpoint=checkpoint)
        return {**self._summary(target), "source": source, "checkpoint": checkpoint, "node": state.head_ref}

    def rewind(self, args: Mapping[str, Any]) -> dict[str, Any]:
        session_id = _session_ref(args)
        if "checkpoint" not in args:
            raise SessionControlError("invalid_argument", "checkpoint is required")
        checkpoint = _checkpoint_arg(args["checkpoint"])
        self._require_idle(session_id, "rewind")
        self._check_checkpoint(session_id, checkpoint)
        before = self._controller.get(session_id).head_ref
        state = self._controller.rewind(session_id, checkpoint)
        return {**self._summary(session_id), "previous_head": before, "checkpoint": checkpoint,
                "node": state.head_ref}

    def close(self, args: Mapping[str, Any]) -> dict[str, Any]:
        session_id = _session_ref(args)
        mode = args.get("mode", "drain")
        if mode not in ("drain", "abort"):
            raise SessionControlError("invalid_argument", "mode must be 'drain' or 'abort'")
        self._require_idle(session_id, "close")
        report = self._controller.close_session(session_id, abort=mode == "abort")
        return {**report, "mode": mode, "status": "closed"}

    def delete(self, args: Mapping[str, Any]) -> dict[str, Any]:
        session_id = _session_ref(args)
        name = _name(args, "checkpoint", required=False, label="checkpoint name")
        collect = args.get("gc", True)
        if type(collect) is not bool:
            raise SessionControlError("invalid_argument", "gc must be a bool")
        controller = self._controller
        if name is not None:
            self._require(session_id)
            try:
                node = controller.unpin(session_id, name)
            except KeyError:
                raise SessionControlError("unknown_checkpoint",
                                          f"session {session_id!r} has no checkpoint named {name!r}") from None
            result: dict[str, Any] = {"session_id": session_id, "deleted_checkpoint": name, "node": node}
        else:
            self._require_idle(session_id, "delete")
            result = controller.close_session(session_id, delete=True)
        if collect:
            result["gc"] = self.bridge.collect_garbage()
        return result

    def gc(self, args: Mapping[str, Any]) -> dict[str, Any]:
        return {"gc": self.bridge.collect_garbage()}

    def raw_tier(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """Freeze D raw-chunk tier: stats and the Step 19 placement controls (module docstring)."""
        action = args.get("action", "stats")
        if action not in ("stats", "place", "demote", "drop", "reset_peaks"):
            raise SessionControlError("invalid_argument", "action must be stats, place, demote, drop or reset_peaks")
        wait = args.get("wait", True)
        if type(wait) is not bool:
            raise SessionControlError("invalid_argument", "wait must be a bool")
        archive = self.bridge.archive
        session_id = _session_ref(args) if args.get("session_id") is not None else None
        if session_id is not None:
            self._require(session_id)
        if action != "stats" and _raw_tier(archive) is None:
            raise SessionControlError("invalid_argument", "this worker has no raw tier (raw_tier=False or a RAW "
                                                          "representation)")
        result: dict[str, Any] = {"action": action, "session_id": session_id}
        if action == "place":
            if session_id is None:
                raise SessionControlError("invalid_argument", "place needs session_id")
            placement = _placement(args.get("placement"))
            self._require_idle(session_id, "raw_tier place")
            result["report"] = archive.place_raw(session_id, placement, wait=wait)
        elif action == "demote":
            if session_id is not None:
                self._require_idle(session_id, "raw_tier demote")
            result["report"] = archive.demote_raw(session_id, wait=wait)
        elif action == "drop":
            if session_id is not None:
                self._require_idle(session_id, "raw_tier drop")
            result["report"] = archive.drop_raw(session_id, drain=True)
        elif action == "reset_peaks":
            archive.reset_raw_peaks()
        elif "placement" in args:
            raise SessionControlError("invalid_argument", "placement is an argument of action place")
        result["raw_tier"] = _raw_tier_stats(archive)
        result["gpu_memory"] = _gpu_memory(self.bridge)
        result.update(_retained_caches(archive))
        if session_id is not None:
            result["session"] = _raw_tier_session(archive, session_id)
        return result


__all__ = ["ERROR_STATUS", "INFLIGHT_OPS", "PROMPT_CHANGE", "SessionControlError", "SessionControls", "error_of",
           "jsonable", "prompt_change_capability"]
