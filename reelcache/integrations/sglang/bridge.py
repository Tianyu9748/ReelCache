"""``CausalWanKVBridge``: the shared causal-Wan archive backend for SGLang (``--reelcache-enable``).

Implements SGLang's ``CausalKVBackend`` protocol (see SGLang
``runtime/layers/kvcache/causal_kv_backend.py``) without importing SGLang:

* capture: normalized pre-RoPE K and V of every layer at the clean context
  forward of each chunk (and each CFG stream), handed over inside the
  self-attention module before RoPE; clean recache forwards (cache epochs that
  re-materialize earlier chunks) are captured whole but only the written
  chunk ranges are archived, never the sink;
* commit (``archive_native``): generation keeps using the valid native active
  KV. Each complete clean chunk is *submitted* at its boundary (generated
  frontier); with ``svd_hevc`` the Step 6
  :class:`~reelcache.archive_pipeline.ArchivePipeline` fits, encodes and
  publishes it behind generation, in order (published frontier), across
  request boundaries: a request returns as soon as its chunks are generated.
  The per-session queue is bounded (``max_unpublished``; a full queue blocks
  the next chunk boundary and is reported as backpressure). Only an explicit
  pause (``durability``/``durable``), checkpoint, fork, rewind, close or a
  restore from the archive waits for the required published frontier; an
  archival failure surfaces on the session's next request, pause, checkpoint
  or close. The RAW reference publishes synchronously (exact, not a
  performance path);
* resident KV: the stage keeps its caches across requests and only resets
  their cursors. At the end of a request the bridge records the caches'
  index state; the session's next request continues on that native KV
  (``restore_index_state``) when no K/V write happened since (SGLang's
  ``mutation_generation``) and the session head did not move: no restore, no
  waiting for publication;
* visibility: the attended history of the next chunk is computed by replaying
  SGLang's own cache logic (sink, global sink, pinned shot sinks, window);
* restore/install (otherwise): attended archived ranges (the newest version
  on the path, cache epochs included) are reconstructed with the engine's own
  RoPE at their recorded positions straight into the reset native cache
  (``install_restored_history``); then the checkpoint's latents and RNG state
  are restored so the stage skips the already generated chunks. Persisted
  sessions are prepared for restore when opened (``prewarm``).
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import inspect
import math
import os
import sys
import threading
import time
import traceback
import weakref
import itertools
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from ...errors import ConfigurationError, IntegrityError, PendingError, StateError, UnsupportedError
from ...types import ModelIdentity, canonical_digest
from .archive import RawSessionArchive, generator_state, raw_tier_for
from .capture import CleanChunkBuilder
from .commit import matrix_contexts, slice_matrices, stream_matrices
from .config import ReelCacheRuntimeConfig
from .controls import SessionControls, prompt_change_capability
from .generation_defer import closes_on_failure, closes_open_holds, holds_decode_seam, opens_after
from .lineage_archive import LineageSessionArchive
from .nodes import GENERATOR_PREFIX, LATENTS_KEY, partial_recache_node, replay_steps, split_record
from .profiles import ModelProfile
from .restore import (GC_PAUSES, RopeTableCache, configured_verification_mode, decoder_pool_record,
                      decoder_pool_snapshot, gc_pause_record, install_gc_pauses, resident_lease_record, warm_engine_rope,
                      warm_rope_tables)
from .session import SGLangSessionController, SGLangSessionState, planned_new_chunks, timeline_of
from .visibility import build_visibility_plan, cache_appends, check_cache_state
from .workload_telemetry import WorkloadTelemetry, request_scope

#: Recent request summaries kept for the ``request`` control (oldest dropped first).
MAX_REQUEST_SUMMARIES = 64
#: Compact per-request entries kept in ``bridge.requests``.
MAX_REQUEST_ENTRIES = 256
#: Memory-critical section records that could not be filed with a request summary (kept, never dropped; F3).
MAX_UNATTRIBUTED_SECTIONS = 64
#: Freeze F5: a session restore of compressed history is layer-overlapped with the request's first forward (Step 5c's
#: ``reelcache.overlapped_restore``; the first forward's attention of layer i waits for layer i through SGLang's
#: ``CausalSelfAttentionKVCache.restore_pending``). Default: ``ReelCacheRuntimeConfig.overlap_restore`` (True when the
#: config has no such field); a request's session handle ``overlap_restore`` (bool) overrides it.
OVERLAP_RESTORE_DEFAULT = True
NO_ALLOCATION_SKIP = ("the restore allocates nothing beyond the request's caches and the RoPE scratch the headroom "
                      "covers (every attended range a GPU raw copy read in place)")


class _ForwardTimer:
    """Clean-forward durations: CUDA events on the current stream (no synchronization until ``resolve``), or the
    host clock on the CPU."""

    def __init__(self, device: Any):
        self.cuda = device is not None and torch.device(device).type == "cuda" and torch.cuda.is_available()
        self._open: dict[tuple, Any] = {}
        self._records: list[tuple] = []

    @staticmethod
    def _key(capture: Any) -> tuple:
        return (getattr(capture, "kind", "clean_commit"), int(capture.chunk.index), str(capture.stream))

    def _mark(self) -> Any:
        if self.cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def begin(self, capture: Any) -> None:
        self._open[self._key(capture)] = self._mark()

    def end(self, capture: Any) -> None:
        start = self._open.pop(self._key(capture), None)
        if start is not None:
            self._records.append((self._key(capture), start, self._mark()))

    def resolve(self) -> list[dict[str, Any]]:
        result = []
        for (kind, chunk, stream), start, end in self._records:
            if self.cuda:
                end.synchronize()
                seconds = start.elapsed_time(end) / 1000.0
            else:
                seconds = end - start
            result.append({"kind": kind, "chunk": chunk, "stream": stream, "seconds": seconds})
        return result


def _latents_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.int16)
    return hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()


def _request_timing(dispatched_at: Any, started: float, chunk_marks: list[dict[str, Any]]) -> dict[str, Any]:
    """First-new-chunk timings of a request (``dispatched_at``: the worker's ``time.perf_counter()`` when it
    dispatched the request, when the engine reports it)."""
    first = chunk_marks[0]["boundary_seconds"] if chunk_marks else None
    timing: dict[str, Any] = {"first_new_chunk_from_begin_seconds": first, "pre_denoise_seconds": None,
                              "time_to_first_new_chunk_seconds": None}
    if isinstance(dispatched_at, (int, float)) and not isinstance(dispatched_at, bool):
        timing["pre_denoise_seconds"] = started - float(dispatched_at)
        if first is not None:
            timing["time_to_first_new_chunk_seconds"] = started - float(dispatched_at) + first
    return timing


def _generators(value: Any) -> list:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _cache_cursors(cache: Any) -> dict[str, int]:
    global_end, local_end = cache._read_indices()
    return {"global_end": int(global_end), "local_end": int(local_end), "pinned_start": int(cache.pinned_start),
            "pinned_len": int(cache.pinned_len)}


def conditioning_digest(conditioning: Any) -> str:
    return canonical_digest(plain(conditioning))


def plain(value: Any) -> Any:
    """JSON-shaped copy (mappings -> dicts, sequences -> lists): stored records compare with live requests."""
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def chunk_id(index: int) -> str:
    return f"c{index:06d}"


def commit_encoder(archive: Any, ticket: Any) -> str | None:
    """The encoder of one commit (freeze D, ``commits[i].encoder``): ``nvenc`` / ``libx265`` once published; None
    while unpublished and for archives without a codec (RAW). Records only: never raises."""
    encoder = getattr(archive, "commit_encoder", None)
    if not callable(encoder):
        return None
    try:
        return encoder(ticket)
    except Exception:  # noqa: BLE001 - records only
        return None


def error_summary(error: BaseException) -> dict[str, Any]:
    """Plain data of a failure kept in records: type, message, repr and the formatted traceback (text). Never the
    exception object itself: its traceback frames can pin device memory (F2 census 4423272: a stored failed-commit
    exception pinned the commit's staged sources, one chunk of K/V)."""
    try:
        text = "".join(traceback.format_exception(error))[-8000:]
    except Exception:  # noqa: BLE001 - records only
        text = None
    return {"type": type(error).__name__, "message": str(error)[:2000], "repr": repr(error)[:2000], "traceback": text}


class ResolvedTicket:
    """A published commit's ticket as plain data: the fields ``commit_report`` / ``commit_encoder`` read (node, ids,
    admission, encoder, a synchronous archive's report), without the ticket's event, waiter or failure."""

    def __init__(self, fields: Mapping[str, Any]):
        self.__dict__.update(fields)
        self.error = None

    def done(self) -> bool:
        return True

    def result(self, timeout: float | None = None) -> str:
        return self.node_ref


def _ticket_fields(ticket: Any) -> dict[str, Any]:
    try:
        fields = {key: value for key, value in vars(ticket).items() if key not in ("error", "_event", "_waiting")}
    except TypeError:  # a ticket without __dict__
        fields = {key: getattr(ticket, key, None) for key in ("node_id", "branch_id", "sequence", "token_end",
                                                             "admission", "encoder", "report")}
    fields["node_ref"] = ticket.result(0)
    return fields


def resolved_commit(archive: Any, entry: Mapping[str, Any]) -> dict[str, Any] | None:
    """The plain-data form of one resolved commit for the bridge's request history (``commit_reports``): the entry's
    own fields, the published ticket's fields or the failure (``error`` repr, ``failure`` summary) and the encoder; no
    live ticket and no exception object. None while the commit is pending (the live ticket stays until it resolves)."""
    if entry.get("plain"):
        return dict(entry)
    ticket = entry["ticket"]
    if not ticket.done():
        return None
    error = getattr(ticket, "error", None)
    return {"plain": True, "base": {key: value for key, value in entry.items() if key != "ticket"},
            "ticket_fields": None if error is not None else _ticket_fields(ticket),
            "error": None if error is None else repr(error), "failure": None if error is None else error_summary(error),
            "encoder": commit_encoder(archive, ticket)}


def block_conditioning(request: Any) -> tuple[str, ...] | None:
    """The request's per-block conditioning identity (SGLang ``CausalKVRequest.block_conditioning``), validated."""
    blocks = getattr(request, "block_conditioning", None)
    if blocks is None:
        return None
    if (not isinstance(blocks, (tuple, list)) or len(blocks) != len(request.block_sizes)
            or any(not isinstance(item, str) or not item for item in blocks)):
        raise ConfigurationError("block_conditioning names one nonempty string per block of the request")
    return tuple(blocks)


def _itemsize(dtype: Any) -> int:
    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, str(dtype).removeprefix("torch."))
    return torch.empty((), dtype=dtype).element_size()


def request_kv_geometry(request: Any, capture_dtypes: Mapping[str, Any], generated_chunks: int,
                        pause_after: Any = None) -> dict[str, Any]:
    """Device KV bytes of a session request from its native cache geometry (freeze F request-start pressure check).

    * ``native_bytes``: the stage's self-attention caches the request runs on, layers x 2 (K, V) x streams x
      ``cache_tokens`` x heads x head_dim x the cache dtype (batch 1). ``cache_tokens`` is the request's cache size:
      full history (CausVid, RAVEN) holds every frame of the request, i.e. the history plus the new chunks; sink +
      window (LongLive) holds the window. Both streams with CFG.
    * SGLang's stage allocates (or reuses) these caches *before* it opens the backend run: ``forward`` sizes them at
      its top and calls ``_begin_causal_kv_run`` afterwards (``causal_denoising.py``; ``longlive2.py`` likewise, for
      both streams), and the request's geometry is read from them. So at ``begin_request`` they are already on the
      device (``native_allocated``) whenever the request carries ``model.cache_tokens``.
    * ``chunk_capture_bytes``: the KV the denoise loop still allocates: the captured pre-RoPE K and V of its first new
      chunk (every layer and stream, capture dtypes), which the archive adopts as that chunk's GPU raw entry.
    * ``needed_bytes``: what is not on the device yet: ``chunk_capture_bytes``, plus ``native_bytes`` for a stage
      that has not allocated its caches (a request without ``model.cache_tokens``: every frame of the request)."""
    model = dict(getattr(request, "model", None) or {})
    streams = len(request.streams)
    layers, heads, head_dim = int(request.num_layers), int(request.num_heads), int(request.head_dim)
    sizes = [int(size) for size in request.block_sizes]
    cache_tokens = model.get("cache_tokens")
    allocated = cache_tokens is not None
    if not allocated:
        cache_tokens = sum(sizes) * int(request.tokens_per_frame)
    native = layers * 2 * streams * int(cache_tokens) * heads * head_dim * _itemsize(request.dtype)
    new_chunks = max(0, len(sizes) - int(generated_chunks))
    if new_chunks:  # W1.3 (record only): the chunks the request plans up to its pause point (``planned_new_chunks``)
        new_chunks = planned_new_chunks(len(sizes), int(generated_chunks), pause_after)
    chunk_tokens = sizes[generated_chunks] * int(request.tokens_per_frame) if new_chunks else 0
    per_token = heads * head_dim * sum(_itemsize(capture_dtypes[target]) for target in ("K", "V"))
    capture = layers * streams * chunk_tokens * per_token
    return {"cache_tokens": int(cache_tokens), "streams": list(request.streams), "layers": layers, "heads": heads,
            "head_dim": head_dim, "dtype": str(request.dtype), "native_bytes": native, "native_allocated": allocated,
            "history_chunks": int(generated_chunks), "new_chunks": new_chunks, "chunk_tokens": chunk_tokens,
            "chunk_capture_bytes": capture, "needed_bytes": capture + (0 if allocated else native)}


#: Device bytes of decoded canvases plus device bases per byte of attended raw K/V a compressed restore reads: 5b's
#: census at CausVid h=14 (A100-40GB) held 1.28 GiB of canvases and 0.11 GiB of bases for ~11 GiB of raw history
#: (about 0.12); 0.3 keeps a 2.5x margin.
RESTORE_DEVICE_FRACTION = 0.3


def restore_working_set(request: Any, capture_dtypes: Mapping[str, Any], plans: Mapping[str, Any], *,
                        arena_bytes: int | None = None, basis_bytes: int | None = None) -> dict[str, Any]:
    """Device bytes an archive restore of ``plans`` may allocate (freeze F pre-restore relief), from geometry:

    * decoded canvases and device bases of the attended ranges: ``RESTORE_DEVICE_FRACTION`` of their raw K/V bytes
      (every layer, both targets, every stream), at most the restorer's canvas-arena plus basis budgets (it never
      exceeds them: a history whose canvases do not fit restores in bounded layer-ahead mode);
    * reconstruct / install temporaries: one chunk of raw K/V (every layer and stream; the RAW install of a host_raw
      range uploads per matrix).

    Ranges the raw tier serves need no canvases; before the restore their tier is unknown, so every attended range
    counts as compressed (conservative)."""
    per_token = (int(request.num_layers) * int(request.num_heads) * int(request.head_dim)
                 * sum(_itemsize(capture_dtypes[target]) for target in ("K", "V")))
    attended = sum(int(span.length) for plan in plans.values() for span in plan.spans)
    raw_bytes = attended * per_token
    canvases = int(math.ceil(RESTORE_DEVICE_FRACTION * raw_bytes))
    caps = [value for value in (arena_bytes, basis_bytes) if value is not None]
    cap = sum(caps) if len(caps) == 2 else None
    if cap is not None:
        canvases = min(canvases, cap)
    chunk_tokens = max(int(size) for size in request.block_sizes) * int(request.tokens_per_frame)
    install = chunk_tokens * per_token * len(plans)
    return {"attended_tokens": attended, "attended_raw_bytes": raw_bytes, "fraction": RESTORE_DEVICE_FRACTION,
            "canvases_and_bases_bytes": canvases, "cap_bytes": cap, "install_bytes": install,
            "bytes": canvases + install}


def _pressure_freed(event: Mapping[str, Any]) -> dict[str, int]:
    """Device bytes one pressure event freed, per component: ``gpu_raw`` (demoted GPU raw copies) and every
    registered trim by name (:mod:`reelcache.device_pressure`)."""
    freed: dict[str, int] = {}
    demoted = int(event.get("freed_bytes") or 0)
    if demoted:
        freed["gpu_raw"] = demoted
    trims = event.get("trims")
    for record in (trims.get("trims") or []) if isinstance(trims, Mapping) else []:
        if record.get("bytes"):
            freed[record["name"]] = freed.get(record["name"], 0) + int(record["bytes"])
    return freed


def _pressure_latency(event: Mapping[str, Any]) -> float:
    """Wall seconds of one pressure event: the bridge's own measurement around the call when present, else the
    event's ``seconds`` (a raw-tier event times its demotion and records its trims, run after it, separately)."""
    if event.get("check_seconds") is not None:
        return float(event["check_seconds"])
    seconds = float(event.get("seconds") or 0.0)
    trims = event.get("trims")
    if "gpu_bytes_after" in event and isinstance(trims, Mapping):
        seconds += float(trims.get("seconds") or 0.0)
    return seconds


def add_pressure_event(record: dict[str, Any], source: str, event: Any) -> None:
    """Append one pressure event (``source``: ``request_start``, the decode seam's reason or ``oom_relief``) to a
    run's ``pressure`` record and update its totals; None (no device, response off) is skipped."""
    if not isinstance(event, Mapping):
        return
    if "error" in event and "where" not in event:
        record["errors"].append({"source": source, "error": event["error"]})
        return
    item = plain(dict(event))
    item["source"] = source
    item["freed_by_component"] = _pressure_freed(event)
    item["latency_seconds"] = _pressure_latency(event)
    for name, nbytes in item["freed_by_component"].items():
        record["freed_by_component"][name] = record["freed_by_component"].get(name, 0) + nbytes
    record["freed_bytes"] = sum(record["freed_by_component"].values())
    record["pressured"] += 1 if item.get("pressure") else 0
    record["seconds"] += item["latency_seconds"]
    # Step 7 dev22j2: a section entered under pressure defers archival GPU work while it runs (annotated on its event);
    # dev22j4: a section's fitter re-check deferrals (:func:`recheck_event`)
    if item.get("deferred_archival"):
        record["deferred_sections"] += 1
        record["archival_deferred_seconds"] += float(item.get("archival_deferred_seconds") or 0.0)
    record["deferral_wait_seconds"] += float(item.get("deferral_wait_seconds") or 0.0)
    record["fitter_recheck_deferrals"] += int(item.get("fitter_recheck_deferrals") or 0)
    record["events"].append(item)


#: The causes of a decode entry guard's archival deferral (Step 7 dev22j6, ``report["deferral_reason"]``).
DECODE_GUARD_DEFERRALS = ("unlearned_reserve", "inflight_node", "short_after_relief")


def _guard_reliefs(guard: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [item["event"] for item in guard.get("actions") or ()
            if isinstance(item, Mapping) and item.get("action") == "relief" and isinstance(item.get("event"), Mapping)]


def _guard_deferral(report: Mapping[str, Any]) -> tuple[str | None, Mapping[str, Any] | None]:
    """The decode entry guard's deferral cause and the recorded event the archive deferred on (the guard's relief or
    the section's entry check, annotated with the cause and the deferral); the event is None when the guard deferred
    on an event the report does not hold (no entry-check event)."""
    cause = report.get("deferral_reason")
    guard = report.get("decode_guard")
    if cause not in DECODE_GUARD_DEFERRALS or not isinstance(guard, Mapping):
        return None, None
    for event in (*_guard_reliefs(guard), report.get("raw_pressure")):
        if isinstance(event, Mapping) and event.get("deferral_reason") == cause:
            return cause, event
    return cause, None


def decode_guard_summary(report: Mapping[str, Any]) -> dict[str, Any] | None:
    """One decode section's entry guard (Step 7 dev22j6, ``report["decode_guard"]``), summed: the target (the larger
    of the reserve and the prior, plus the bytes of a node on the GPU), the driver's free memory before and after, the
    cached-block releases and their bytes, the reliefs and what they freed (``gpu_raw``: demoted GPU raw copies), the
    deferral cause (``unlearned_reserve`` / ``inflight_node`` / ``short_after_relief``; None: not deferred by the
    guard), whether the deferral took effect and its waits (``deferral_wait_seconds``: the section waiting for a node
    on the GPU; ``archival_deferred_seconds``: archival paused). None when the guard did not check (off, not a decode,
    nothing to protect, the reserve check deferred, or an archive without it)."""
    guard = report.get("decode_guard")
    if not isinstance(guard, Mapping):
        return None
    actions = [item for item in guard.get("actions") or () if isinstance(item, Mapping)]
    releases = [item for item in actions if item.get("action") == "release_cached"]
    freed: dict[str, int] = {}
    reliefs = _guard_reliefs(guard)
    for event in reliefs:
        for name, nbytes in _pressure_freed(event).items():
            freed[name] = freed.get(name, 0) + nbytes
    cause, event = _guard_deferral(report)
    deferral = (event if event is not None else report) if cause else {}
    return {"target": guard.get("target"), "reserve": guard.get("reserve"), "prior": guard.get("prior"),
            "inflight_bytes": guard.get("inflight_bytes"), "unlearned": guard.get("unlearned"),
            "free": guard.get("free"), "free_after": guard.get("free_after"), "headroom": guard.get("headroom"),
            "headroom_after": guard.get("headroom_after"), "short": guard.get("short"),
            "cache_releases": len(releases),
            "cache_released_bytes": sum(int(item.get("bytes") or 0) for item in releases),
            "cache_release_seconds": sum(float(item.get("seconds") or 0.0) for item in releases),
            "reliefs": len(reliefs), "relief_freed_by_component": freed, "demoted_bytes": freed.get("gpu_raw", 0),
            "relief_seconds": sum(_pressure_latency(item) for item in reliefs),
            "deferral_reason": cause, "deferred": bool(deferral.get("deferred_archival")),
            "deferral_wait_seconds": deferral.get("deferral_wait_seconds"),
            "archival_deferred_seconds": deferral.get("archival_deferred_seconds"),
            "deferral_error": deferral.get("deferral_error")}


def decode_guard_events(reason: str, report: Mapping[str, Any],
                        guard: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """A decode guard's pressure events for the request's record (source ``<reason>:decode_guard``): its reliefs and,
    when it deferred on an event the report does not hold, the deferral itself (``decode_guard_deferral``, no latency).
    A deferral on the entry check's event is counted with that event, never twice."""
    source = f"{reason}:decode_guard"
    events = [(source, event) for event in _guard_reliefs(report["decode_guard"])]
    cause, event = _guard_deferral(report)
    if cause is not None and event is None:
        events.append((source, {"event": "decode_guard_deferral", "where": reason, "pressure": False,
                                "deferral_reason": cause, "deferred_archival": guard["deferred"],
                                "deferral_wait_seconds": guard["deferral_wait_seconds"],
                                "archival_deferred_seconds": guard["archival_deferred_seconds"],
                                "deferral_error": guard["deferral_error"]}))
    return events


def add_decode_guard(record: dict[str, Any], guard: Mapping[str, Any]) -> None:
    """Count one decode guard summary (:func:`decode_guard_summary`) into a run's ``pressure`` record
    (``decode_guard``: checks, cache releases and bytes, reliefs, demoted bytes, deferrals per cause)."""
    totals = record.setdefault("decode_guard", {"checks": 0, "cache_releases": 0, "cache_released_bytes": 0,
                                                "cache_release_seconds": 0.0, "reliefs": 0, "demoted_bytes": 0,
                                                "short": 0, "deferrals": {}})
    totals["checks"] += 1
    for key in ("cache_releases", "cache_released_bytes", "reliefs", "demoted_bytes"):
        totals[key] += int(guard.get(key) or 0)
    totals["cache_release_seconds"] += float(guard.get("cache_release_seconds") or 0.0)
    totals["short"] += 1 if guard.get("short") else 0
    if guard.get("deferral_reason"):
        totals["deferrals"][guard["deferral_reason"]] = totals["deferrals"].get(guard["deferral_reason"], 0) + 1


def recheck_event(report: Mapping[str, Any]) -> dict[str, Any] | None:
    """A memory-critical section's fitter re-check deferrals (Step 7 dev22j4: archival GPU work that would not fit above
    the reserve waited for the section to close, capped; the report's ``fitter_rechecks``) as one pressure event of
    the section, counted as the raw tier counts them: one deferred section, the waits as archival deferred seconds.
    Archival waited, not the request: no latency, no pressure check. None without any."""
    rechecks = [dict(item) for item in report.get("fitter_rechecks") or () if isinstance(item, Mapping)]
    if not rechecks:
        return None
    return {"event": "fitter_recheck", "where": report.get("reason"), "deferred_archival": True,
            "deferral_reason": "fitter_recheck", "fitter_recheck_deferrals": len(rechecks),
            "archival_deferred_seconds": sum(float(item.get("waited_seconds") or 0.0) for item in rechecks),
            "fitter_rechecks": rechecks}


def pressure_record(kv: Mapping[str, Any] | None, events: list) -> dict[str, Any]:
    """A run's ``pressure`` record (freeze F): its request-start check, pre-restore relief, restore-section and
    decode-seam checks and OOM reliefs (the events, as the archive returns them, with ``source``), the bytes freed per
    component (``gpu_raw`` demotions and each registered trim), the latency (``seconds``) and the sections that
    deferred archival (``deferred_sections``, ``deferral_wait_seconds``, ``archival_deferred_seconds``: under entry
    pressure, Step 7 dev22j2; ``fitter_recheck_deferrals``: the fitter re-check, dev22j4). ``events``: ``(source,
    event)`` pairs. ``kv``: the request's KV geometry (:func:`request_kv_geometry`)."""
    record = {"kv": None if kv is None else dict(kv), "events": [], "pressured": 0, "freed_bytes": 0,
              "freed_by_component": {}, "seconds": 0.0, "deferred_sections": 0, "deferral_wait_seconds": 0.0,
              "archival_deferred_seconds": 0.0, "fitter_recheck_deferrals": 0, "errors": []}
    for source, event in events:
        add_pressure_event(record, source, event)
    return record


def rope_scratch_bytes(plans: Mapping[str, Any], *, num_heads: int, head_dim: int,
                       key_dtype: torch.dtype) -> dict[str, int]:
    """Device scratch of an in-place RAW install (Step 5c's O4 fast path): no staging and no canvases, but the engine
    RoPE's output, taken from the caching allocator and reused in stream order. ``span``: the layer-overlapped install
    (``rope_per="span"``), one device span's K at a time (CausVid: one chunk, 14.4 MB); ``bound``: the synchronous
    install (``rope_per="layer"``: every span's K gathered, then rotated, per layer: 2 x the layer's K, 230 MB at
    CausVid h8), which bounds both paths and an overlapped restore that falls back to the plain call."""
    size = num_heads * head_dim * torch.tensor([], dtype=key_dtype).element_size()
    lengths = [span.length for plan in plans.values() for span in plan.spans]
    return {"span": max(lengths, default=0) * size, "bound": 2 * sum(lengths) * size}


def scratch_headroom(device: Any, need_bytes: int) -> dict[str, Any]:
    """Whether ``need_bytes`` of new device allocations fit now: the driver's free memory first (one query), else the
    conservative headroom of the pressure checks (driver free plus wholly free cached segments,
    ``raw_tier.device_headroom``). Not checked off CUDA (covered: no device memory to protect)."""
    started = time.perf_counter()
    device = torch.device(device)
    if device.type != "cuda":
        return {"checked": False, "covered": True, "seconds": 0.0}
    from ...raw_tier import cuda_device_memory, device_headroom

    free, _ = torch.cuda.mem_get_info(device)
    headroom = None if free >= need_bytes else device_headroom(cuda_device_memory(device))
    return {"checked": True, "free": int(free), "headroom": headroom,
            "covered": free >= need_bytes or (headroom is not None and headroom >= need_bytes),
            "seconds": time.perf_counter() - started}


def overlap_seam(caches: Mapping[str, Any]) -> bool:
    """Whether every cache waits for ``restore_pending`` before its first K/V access: SGLang's layer-overlap seam
    (``CausalSelfAttentionKVCache.await_restored_history``). Without it a forward would read history the restore has
    not written yet, so the restore runs plain."""
    return all(callable(getattr(cache, "await_restored_history", None)) for items in caches.values() for cache in items)


#: What a cache's ``replay_layout`` depends on besides the appends (with the class and the K buffer's shape).
CACHE_GEOMETRY_FIELDS = ("cache_size", "sink_tokens", "global_sink_tokens", "attention_window_size", "allow_growth",
                         "protect_sink_on_recompute")
PLAN_CACHE_ENTRIES = 64


def cache_geometry(cache: Any) -> tuple:
    k = getattr(cache, "k", None)
    return (type(cache).__module__, type(cache).__qualname__, None if k is None else tuple(k.shape),
            *(getattr(cache, name, None) for name in CACHE_GEOMETRY_FIELDS))


class PlanCache:
    """Freeze F5 (cut 3): restore plans by (head node, stream, next chunk's token range, tokens per frame, cache
    geometry). A plan (``replay_layout`` + ``check_cache_state`` + ``build_visibility_plan``: 4 / 13 / 25 ms at 1 / 4 /
    8 CausVid chunks in P4) is a pure function of these: archived nodes are immutable and plans frozen, so any head or
    layout change is a different key (no stale entry can be found). Filled by restores and, ahead of the next resume,
    when a session pauses durably; bounded LRU."""

    def __init__(self, capacity: int = PLAN_CACHE_ENTRIES):
        self.capacity = capacity
        self._plans: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "prepared": 0, "evicted": 0}

    @staticmethod
    def key(head_ref: str, stream: str, cache: Any, next_chunk: tuple[int, int], tokens_per_frame: int) -> tuple:
        return (head_ref, stream, tuple(next_chunk), int(tokens_per_frame), cache_geometry(cache))

    def get(self, key: tuple) -> Any:
        with self._lock:
            plan = self._plans.get(key)
            if plan is None:
                self.stats["misses"] += 1
            else:
                self.stats["hits"] += 1
                self._plans.move_to_end(key)
            return plan

    def put(self, key: tuple, plan: Any, *, prepared: bool = False) -> None:
        with self._lock:
            self._plans[key] = plan
            self._plans.move_to_end(key)
            self.stats["prepared"] += int(prepared)
            while len(self._plans) > self.capacity:
                self._plans.popitem(last=False)
                self.stats["evicted"] += 1

    def describe(self) -> dict[str, Any]:
        with self._lock:
            return {"entries": len(self._plans), "capacity": self.capacity, **self.stats}

    def clear(self) -> None:
        with self._lock:
            self._plans.clear()


def build_plans(caches: Mapping[str, Any], streams: Any, nodes: list, next_chunk: tuple[int, int],
                tokens_per_frame: int, cache: PlanCache | None, *,
                prepared: bool = False) -> tuple[dict[str, Any], dict[str, bool]]:
    """The restore plan of every stream (from ``cache`` when it has them) and whether each was a cache hit."""
    plans, hits, appends = {}, {}, None
    for stream in streams:
        key = None if cache is None else PlanCache.key(nodes[-1].ref, stream, caches[stream][0], next_chunk,
                                                       tokens_per_frame)
        plan = None if key is None else cache.get(key)
        hits[stream] = plan is not None
        if plan is None:
            if appends is None:
                appends = cache_appends(nodes)
            layout = caches[stream][0].replay_layout(appends, next_chunk=next_chunk)
            check_cache_state(layout, nodes[-1].cache_state, stream)
            plan = build_visibility_plan(stream, layout, nodes, tokens_per_frame=tokens_per_frame,
                                         next_chunk=next_chunk)
            if key is not None:
                cache.put(key, plan, prepared=prepared)
        plans[stream] = plan
    return plans, hits


def _event_done(event: Any) -> bool:
    """Whether a CUDA event completed (a query that fails counts as not completed: it is still waited for)."""
    try:
        return bool(event.query())
    except Exception:  # noqa: BLE001
        return False


def promotion_fences_needed(profile: Any) -> bool:
    """wl-w1b7: whether a promotion's device-to-host copy of restored slots must fence the next forward. Rolling
    sink / window caches (LongLive 1.0 ``sink_window``, LongLive 2.0 ``global_sink_pinned_window``) may roll or
    rewrite restored slots on the next append; full-history caches (CausVid, RAVEN as shipped) append to fresh slots
    and never need one."""
    return getattr(profile, "visibility", "full_history") != "full_history"


class _PromotionFence:
    """wl-w1b7: ``CausalSelfAttentionKVCache.restore_pending`` of a cache layer whose restored slots the next append
    may overwrite (rolling sink / window caches) while the promotion's device-to-host copy of them runs. The first
    access (``await_restored_history``) runs any earlier pending wait, then orders the current stream after that
    layer's copy (a device-side wait, no host block). With workload telemetry its GPU time is a ``promotion_fence``
    span and the same wait an exposed stall of source ``promotion_fence``."""

    __slots__ = ("event", "previous", "run", "layer", "counted", "boundary")

    def __init__(self, event: Any, previous: Any, run: Any, layer: int, *, boundary: bool = False):
        self.event, self.previous, self.run, self.layer = event, previous, run, layer
        self.counted = False
        # W1.4.1 (C1): a boundary fence -- left by a finished request on every layer for the NEXT writer of the stage
        # caches (``run`` None: the next writer's telemetry is not known here)
        self.boundary = boundary

    def wait(self) -> None:
        previous, self.previous = self.previous, None
        if previous is not None:
            previous.wait()
        event, self.event = self.event, None
        run = self.run
        promotion = (getattr(run, "restore_report", None) or {}).get("promotion")
        if isinstance(promotion, dict) and not self.counted:  # the restore record counts the fences passed, once each
            self.counted = True
            promotion["fence_waits"] = promotion.get("fence_waits", 0) + 1
        if event is None:
            return
        stream = torch.cuda.current_stream()
        telemetry = getattr(run, "workload_telemetry", None)
        span = None if telemetry is None else telemetry.begin("promotion_fence", stream=stream)
        token = None if telemetry is None else telemetry.stall_begin(stream, ready=True, source="promotion_fence")
        stream.wait_event(event)
        if telemetry is not None:
            telemetry.stall_end(token, layer=self.layer)
            telemetry.end(span)


class _PromotionTrigger:
    """W1.4.1 (N1): ``restore_pending`` of every cache layer after a recompute replay whose promotion is PREPARED but
    not yet issued. Its first access -- the first new chunk's denoise forward, after the stage's pre-generation
    device synchronize, or a recache before it (whose writes must not reach the slots first) -- issues the
    device-to-host copies, once for every layer, so they overlap generation; a rolling cache then waits on this
    layer's copy before its forward may overwrite restored slots (an exposed ``promotion_fence`` stall)."""

    __slots__ = ("run", "previous", "stream", "layer", "boundary")

    def __init__(self, run: Any, previous: Any, stream: str, layer: int):
        self.run, self.previous, self.stream, self.layer = run, previous, stream, layer
        self.boundary = False

    def wait(self) -> None:
        previous, self.previous = self.previous, None
        if previous is not None:
            previous.wait()
        run, self.run = self.run, None
        if run is None:
            return
        fences = run._issue_deferred_promotion()
        if (self.stream, self.layer) in fences:  # a rolling cache: this layer's copy before its forward writes
            _PromotionFence(fences[(self.stream, self.layer)], None, run, self.layer).wait()


class _ReplayGuard:
    """W1b recompute arm: ``CausalSelfAttentionKVCache.restore_pending`` of a cache whose history the stage must replay
    (``run.begin_replay`` lifts it). A stage that reads or writes the cache before replaying fails here, typed, instead
    of generating on an empty history."""

    __slots__ = ("session_id",)

    def __init__(self, session_id: str):
        self.session_id = session_id

    def wait(self) -> None:
        raise UnsupportedError(
            f"session {self.session_id!r}: restore_mode='recompute' needs a stage that replays the restored history "
            "(run.begin_replay / end_replay) before its first forward; this stage does not")


def json_details(details: Mapping[str, Any] | None) -> dict[str, Any]:
    """A JSON copy of a stage's replay details (records only)."""
    return {} if not details else plain(dict(details))


class _LayerWait:
    """``CausalSelfAttentionKVCache.restore_pending`` of one (stream, layer) cache (freeze F5 seam): SGLang calls
    :meth:`wait` once, at the top of the cache's ``update_and_get_attention_kv``, before any read or write."""

    __slots__ = ("overlap", "readiness", "stream", "layer")

    def __init__(self, overlap: _Overlap, readiness: Any, stream: str, layer: int):
        self.overlap, self.readiness, self.stream, self.layer = overlap, readiness, stream, layer

    def wait(self) -> None:
        self.overlap.wait_layer(self)


class _Overlap:
    """One layer-overlapped restore in flight (freeze F5): the archive lease (``lease.readiness`` per call), the
    caches it fills, the restore's memory-critical section (kept open until :meth:`join`) and the consumer's waits.

    :meth:`join` runs before anything derived from the restored KV leaves the request (the first clean forward,
    commits, the request's end): it joins the restore (its failure re-raised, typed), completes the lease records,
    closes the lease and the section and clears every cache's pending wait."""

    def __init__(self, *, mode: str, lease: Any, caches: Mapping[str, Any], device: Any, started: float,
                 pressure_events: list):
        self.mode, self.lease, self.device, self.started = mode, lease, torch.device(device), started
        self.caches = {stream: list(items) for stream, items in caches.items()}
        self.stack = contextlib.ExitStack()
        self.section: Mapping[str, Any] | None = None
        self.pressure_events = pressure_events  # the restore section's events (the run's list once adopted)
        self.waits: list[dict[str, Any]] = []
        self.joined = False
        self.error: dict[str, Any] | None = None  # plain data (error_summary), never the exception
        self.join_seconds: float | None = None
        self.joined_at: float | None = None
        self.rope_tables: Any = None
        self.pool_before: Any = None
        self.relief: dict[str, Any] | None = None
        self.dispatch_to_start_seconds: float | None = None
        self.setup: dict[str, float] = {}  # the synchronous start, step by step (seconds)
        self.aborted: str | None = None

    def arm(self) -> None:
        """Leave each restored cache's per-layer wait (``restore_pending``) for the forward."""
        for name, readiness in self.lease.readiness:
            for stream in ([name] if name is not None else list(self.caches)):
                for layer, cache in enumerate(self.caches[stream]):
                    cache.restore_pending = _LayerWait(self, readiness, stream, layer)

    def wait_layer(self, item: _LayerWait) -> None:
        clock = time.perf_counter()
        ready = item.readiness.ready(item.layer)
        stream = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        item.readiness.wait(item.layer, stream)
        self.waits.append({"stream": item.stream, "layer": item.layer, "ready_at_wait": ready,
                           "host_wait_seconds": time.perf_counter() - clock, "at_seconds": clock - self.started})

    def disarm(self) -> None:
        for caches in self.caches.values():
            for cache in caches:
                pending = getattr(cache, "restore_pending", None)
                if isinstance(pending, _LayerWait) and pending.overlap is self:
                    cache.restore_pending = None

    def join(self) -> None:
        if self.joined:
            return
        self.joined = True
        clock = time.perf_counter()
        try:
            self.lease.complete()  # joins every overlapped call; raises a failure, typed
            self.lease.wait()  # plain calls of a mixed restore: the current stream waits for their handles
        except BaseException as exc:
            self.error = error_summary(exc)
            raise
        finally:
            self.disarm()
            try:
                self.lease.close()
            finally:
                self.joined_at = time.perf_counter()
                self.join_seconds = self.joined_at - clock
                with contextlib.suppress(Exception):
                    self.stack.close()  # the restore section (its pressure events go to ``pressure_events``)

    def abort(self, reason: str) -> None:
        """Leave no restore work in flight without waiting for the restore to finish (Step 5c's O3 ``quiesce``:
        cancel at the next layer, join the thread, synchronize its stream); failures are not raised (the request
        handles its own error). Then as :meth:`join`: clear the waits, close the lease and the section."""
        if self.joined:
            return
        self.joined, self.aborted = True, reason
        clock = time.perf_counter()
        try:
            for _, readiness in getattr(self.lease, "readiness", None) or ():
                with contextlib.suppress(Exception):
                    readiness.quiesce()
                if readiness.error is not None and not getattr(readiness, "cancelled", False):
                    self.error = self.error or error_summary(readiness.error)  # the restore's own failure
        finally:
            self.disarm()
            with contextlib.suppress(Exception):
                self.lease.close()
            self.joined_at = time.perf_counter()
            self.join_seconds = self.joined_at - clock
            with contextlib.suppress(Exception):
                self.stack.close()

    def record(self) -> dict[str, Any]:
        overlap = dict(self.lease.stats.get("overlap") or {})
        live = list(getattr(self.lease, "readiness", None) or ())  # until the lease closes; then its records
        waits = self.waits
        return {"mode": self.mode, "calls": overlap.get("calls"), "fallback": overlap.get("fallback"),
                "dispatch_to_start_seconds": self.dispatch_to_start_seconds,
                "layers_waited": len(waits),
                "waited_before_ready": sum(1 for item in waits if not item["ready_at_wait"]),
                "host_wait_seconds": sum(item["host_wait_seconds"] for item in waits),
                "max_host_wait_seconds": max((item["host_wait_seconds"] for item in waits), default=0.0),
                "first_wait_at_seconds": waits[0]["at_seconds"] if waits else None,
                "last_wait_at_seconds": waits[-1]["at_seconds"] if waits else None,
                "joined_at_seconds": None if self.joined_at is None else self.joined_at - self.started,
                "join_seconds": self.join_seconds, "error": None if self.error is None else self.error["repr"],
                "aborted": self.aborted, "setup_seconds": dict(self.setup),
                "archive_start": {key: overlap.get(key) for key in ("labels_seconds", "start_call_seconds")},
                "waits": waits,
                "timelines": [readiness.timeline() for _, readiness in live] if live else list(
                    overlap.get("timelines") or []),
                "producer": [dict(readiness.info) for _, readiness in live] if live else list(
                    overlap.get("producer") or [])}


@dataclass
class ResidentKV:
    """A session's native KV left resident in the stage's caches at the end of its last request.

    ``states`` are the caches' index states (cursors, pinned sink, geometry, K/V mutation generation) per
    stream and layer; ``latents`` (CPU) and ``generators`` are the request state a resumed request needs
    for its already generated chunks. Valid while the session's generated head is unchanged (``revision``)
    and nothing wrote the caches since."""

    session_id: str
    revision: int
    chunks: int
    streams: tuple[str, ...]
    states: dict[str, list]
    latents: dict[int, torch.Tensor]
    generators: dict[str, torch.Tensor]
    request_id: Any
    created: float

    def describe(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "chunks": self.chunks, "revision": self.revision,
                "streams": list(self.streams), "request_id": self.request_id, "created": self.created}


class CausalWanKVBridge:
    """Worker-owned backend; one session request at a time (serialized with the worker)."""

    def __init__(self, config: ReelCacheRuntimeConfig, profile: ModelProfile, *, device: Any,
                 model_path: str | None = None):
        if not isinstance(config, ReelCacheRuntimeConfig) or not isinstance(profile, ModelProfile):
            raise ConfigurationError("the bridge needs a ReelCacheRuntimeConfig and a ModelProfile")
        self.config, self.profile = config, profile
        self.device = torch.device(device)
        memory_cap = None
        if config.cuda_memory_limit_bytes is not None and self.device.type == "cuda":
            # Diagnostics: emulate a smaller GPU for the whole worker (generation, archival, decode).
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            total = torch.cuda.get_device_properties(index).total_memory
            fraction = min(1.0, config.cuda_memory_limit_bytes / total)
            torch.cuda.set_per_process_memory_fraction(fraction, index)
            memory_cap = {"limit_bytes": config.cuda_memory_limit_bytes, "device_total_bytes": total,
                          "fraction": fraction}
        fingerprint = canonical_digest({"profile": profile.to_dict(), "model_path": model_path})
        dtype = profile.native_dtype if profile.capture_dtypes is None else dict(profile.capture_dtypes)
        self.model = ModelIdentity(profile.name, fingerprint, profile.num_layers, profile.channels,
                                   "sglang_causal_wan_bthd", dtype)
        self.capture_dtypes = {target: getattr(torch, profile.capture_dtype(target)) for target in ("K", "V")}
        self.targets = config.targets(profile.energy_profile)
        root = os.path.join(config.store_root, f"{profile.name}-{fingerprint[:16]}", config.representation)
        if config.representation in ("raw", "latents_only", "raw_offload"):
            # W1b: "latents_only" (the recompute arm) stores no KV objects; its raw tier (the lineage archive's
            # budget fields) sits in front of the replay. CONTRACT §10: "raw_offload" writes RAW objects behind the
            # same raw tier (ordered write-behind); ``store_io`` is the store's page-cache policy
            latents_only = config.representation == "latents_only"
            offload = config.representation == "raw_offload"
            self.archive = RawSessionArchive(root, self.model, device=self.device,
                                             max_store_bytes=config.max_store_bytes,
                                             max_resident_bytes=config.max_resident_bytes,
                                             verify_reads=config.verify_reads, audit_on_open=config.audit_on_open,
                                             on_open_error=config.on_open_error,
                                             **({"kv_objects": False,
                                                 "raw_tier": raw_tier_for(config) if config.raw_tier else None}
                                                if latents_only else {}),
                                             **({"raw_tier": raw_tier_for(config), "write_behind": True}
                                                if offload else {}),
                                             **({"store_io": config.store_io}
                                                if getattr(config, "store_io", "buffered") != "buffered" else {}),
                                             # W1.1: the tier-backed archives share the lineage archive's
                                             # device-memory seams (decode / restore sections, reliefs)
                                             **({"config": config} if latents_only or offload else {}))
        else:
            needed = profile.max_recache_ranges
            staged = config.max_staged_canvases
            if staged is None:
                staged = max(2, needed)
            elif staged < needed:
                raise ConfigurationError(
                    f"max_staged_canvases={staged} cannot archive the {profile.name} prompt-switch recache epochs "
                    f"(up to {needed} chunk ranges per node, one staged canvas each); use >= {needed} or None")
            self.archive = LineageSessionArchive(root, self.model, config=config, targets=self.targets,
                                                 device=self.device,
                                                 matmul_precision=config.matmul_precision or profile.matmul_precision,
                                                 max_staged_canvases=staged)
            if getattr(config, "store_io", "buffered") != "buffered":  # CONTRACT §10: also pages read at open
                self.archive.store.set_io_policy(config.store_io, drop_existing=True)
        self.controller = SGLangSessionController(self)
        self.controls = SessionControls(self)
        # Process-level RoPE span tables of restores (CR-17-2): built at attach (warm_engine_rope), shared by requests.
        self.rope_tables = RopeTableCache()
        self.gc_pauses = install_gc_pauses()  # records only: one gc.callbacks hook per process
        self._lock = threading.RLock()
        self._active: _BridgeRun | _ObserverRun | None = None
        self._closed = False
        self._resident: ResidentKV | None = None
        # W1.4.1 (C1): aggregate completion events of replay promotions whose device-to-host copies may still read
        # the stage caches (every later writer waits for them, :meth:`_await_promotion_copies`)
        self._promotion_pending: list = []
        self._resident_lock = threading.Lock()  # innermost lock: never held while taking another one
        self._request_tickets: OrderedDict[Any, list] = OrderedDict()
        self._summaries: OrderedDict[Any, dict[str, Any]] = OrderedDict()
        # Freeze F3 (5b): a request without an SGLang request id gets an internal key; the decode seam of a request
        # without an id files under the request that just finished on this worker (one at a time), else into the
        # unattributed records. Records are never dropped.
        self._request_keys = itertools.count(1)
        self._awaiting_decode: Any = None
        self.unattributed_sections: deque = deque(maxlen=MAX_UNATTRIBUTED_SECTIONS)
        # Freeze F5: layer-overlapped restores (lineage archives whose restore takes ``overlapped``), default on
        self._overlap_capable = "overlapped" in inspect.signature(self.archive.restore).parameters
        self.overlap_restore_default = bool(getattr(config, "overlap_restore", OVERLAP_RESTORE_DEFAULT))
        self.overlap_require_prepared = True
        self._boundary_hook: Any = None
        self.plans = PlanCache()  # freeze F5 (cut 3): restore plans by head, stream, next chunk and cache geometry
        self.requests: list[dict[str, Any]] = []
        describe = getattr(self.archive, "describe", None)
        self.startup = {"model": self.model.to_dict(), "profile": profile.to_dict(), "config": config.to_dict(),
                        "energy_targets": self.targets.to_dict(), "store_root": root, "device": str(self.device),
                        "archive": describe() if callable(describe) else {"backend": self.archive.name},
                        "cuda_memory_cap": memory_cap}
        warm = config.warmup
        if warm is None:  # automatic: CUDA bridges whose restores launch the Triton kernels
            warm = self.device.type == "cuda" and (config.representation in ("raw", "raw_offload", "latents_only")
                                                   or config.decoder == "nvdec_packed")
        # Process-level warm-up at attach (the first restore of a worker must beat recompute too); never raises.
        self.startup["warmup"] = self.archive.warmup(profile) if warm else {"skipped": "warmup disabled"}
        ref = weakref.ref(self)
        atexit.register(lambda: (ref() is not None and ref()._close_at_exit()))

    # -- freeze F5: layer-overlapped restore ------------------------------------------------------------------------
    def overlap_requested(self, session: Mapping[str, Any] | None) -> bool:
        """Whether a request's restore is layer-overlapped: its session handle's ``overlap_restore`` (bool) when
        given, else the config default (``overlap_restore``, on); never for archives without overlapped restores
        (the RAW reference archive)."""
        flag = (session or {}).get("overlap_restore")
        if flag is None:
            flag = self.overlap_restore_default
        elif not isinstance(flag, bool):  # SGLang validates the handle; direct callers too
            raise ValueError("causal_kv_session.overlap_restore must be a bool")
        return flag and self._overlap_capable

    # -- SGLang backend protocol ------------------------------------------------------------------------------
    def request_key(self, request_id: Any) -> Any:
        """The key a request's records are filed under: its SGLang request id, or an internal ``bridge-<n>`` key when
        it has none (the records then carry ``request_id_missing``)."""
        return request_id if request_id is not None else f"bridge-{next(self._request_keys):06d}"

    @closes_open_holds("next_request")
    def begin_request(self, request: Any) -> _BridgeRun | _ObserverRun | None:
        session = request.session
        self._awaiting_decode = None  # a new request: the previous one's decode seam has run
        if session is None:
            if not self.config.observe_plain_requests:
                return None
            with self._lock:
                if self._closed:
                    raise StateError("the ReelCache bridge is closed")
                if self._active is not None:
                    raise PendingError("one causal KV request runs at a time per worker")
                observer = _ObserverRun(self, request)
                self._active = observer
                return observer
        begun_clock = time.perf_counter()
        with self._lock:
            if self._closed:
                raise StateError("the ReelCache bridge is closed")
            if self._active is not None:
                raise PendingError("one causal KV session request runs at a time per worker")
            self.profile.check_request(request)
            identity = plain({"profile": self.profile.name, "model": dict(request.model),
                              "tokens_per_frame": request.tokens_per_frame, "grid": list(request.grid),
                              "streams": list(request.streams), "dtype": request.dtype,
                              "pos_start_base": request.pos_start_base})
            digest = conditioning_digest(request.conditioning)
            blocks = block_conditioning(request)
            action, session_id = session["action"], session["session_id"]
            controller = self.controller
            begun = time.time()
            closed = [name for name in (session_id, session.get("from_session"))
                      if name and controller.get(name, required=False) is None]

            def same_blocks(session_name, timeline):
                # Timeline-varying conditioning (prompt switches): every restored block (and every restored
                # recache epoch, by the block it precedes) must have been generated under the request's own
                # conditioning of that block; later blocks may differ (a fork under another future prompt).
                for entry in timeline:
                    block = entry["block"]
                    if block is None or block >= len(request.block_sizes):
                        continue
                    expected = None if blocks is None else blocks[block]
                    if entry["conditioning"] != expected:
                        raise UnsupportedError(
                            f"session {session_name!r}: restored {entry['kind']} of block {block} was generated "
                            f"under block conditioning {entry['conditioning']!r}, the request has {expected!r}")

            def compatible(state, timeline=True):
                # Checked before any fork/rewind so a rejected request never moves a head.
                if state.conditioning_digest != digest:
                    change = prompt_change_capability(self.profile)
                    hint = (f"{self.profile.name} changes prompts natively through the request fields "
                            f"{change['request_fields']} (per-block conditioning); the other conditioning (seed, "
                            "sampling, geometry) stays the session's" if change["supported"] else
                            f"{self.profile.name} has no native prompt change: {change['reason']}")
                    raise UnsupportedError(
                        f"session {state.session_id!r} was generated with different conditioning (prompt, seed or "
                        f"sampling); {hint}"
                    )
                if plain(state.identity) != identity:
                    raise UnsupportedError(
                        f"session {state.session_id!r} geometry {dict(state.identity)} != request {identity}")
                if state.status == "running":
                    raise PendingError(f"session {state.session_id!r} is already running")
                if timeline:
                    with state.lock:
                        entries = list(state.timeline)
                    same_blocks(state.session_id, entries)
                return state

            # workload telemetry (W1.1): the request's, before its controller step (fork / rewind drain spans)
            workload = WorkloadTelemetry(self.device) if getattr(self.config, "workload_telemetry", False) else None
            checked_clock = time.perf_counter()
            if action == "generate":
                # Hashed root content: identity, conditioning and the archive method; run configuration (paths,
                # budgets, workers) and the request id go to the root's unhashed telemetry sidecar.
                record = {"conditioning_digest": digest, "conditioning": plain(request.conditioning),
                          "identity": identity, "profile": self.profile.to_dict(),
                          "method": self.config.content_dict(), "energy_targets": self.targets.to_dict()}
                state = controller.create(session_id, record=record, streams=request.streams,
                                          conditioning_digest=digest, identity=identity,
                                          replace=bool(session.get("replace")),
                                          telemetry={"config": self.config.to_dict(),
                                                     "request_id": request.request_id})
            elif action == "resume":
                if session.get("checkpoint") is not None:
                    raise ConfigurationError("resume continues from the session head; use rewind for a checkpoint")
                with request_scope(workload):  # W1.2: a closed session's reopen (session_open span)
                    state = controller.get(session_id)
                # An archival failure of an earlier request surfaces here (acknowledged: the session continues
                # from its last published checkpoint on the next request).
                controller.reconcile(state)
                state = compatible(state)
            elif action == "fork":
                with request_scope(workload):  # W1.2: the source's reopen (session_open), then its drain
                    compatible(controller.get(session["from_session"]), timeline=False)
                    state = controller.fork(session["from_session"], session_id, checkpoint=session.get("checkpoint"),
                                            check=lambda entries: same_blocks(session["from_session"], entries))
            else:
                with request_scope(workload):  # W1.2: the session's reopen (session_open), then its switch drain
                    compatible(controller.get(session_id), timeline=False)
                    state = controller.rewind(session_id, session["checkpoint"],
                                              check=lambda entries: same_blocks(session_id, entries))
            controlled_clock = time.perf_counter()
            with state.lock:
                if state.status == "running":
                    raise PendingError(f"session {session_id!r} is already running")
                state.status = "running"
                state.pause_requested = False
                state.log("request", action=action, request_id=request.request_id,
                          pause_after_chunks=session.get("pause_after_chunks"), pending=len(state.pending))
            controller.release_idle(keep={state.session_id})  # bounded pipeline state; never raises
            run = _BridgeRun(self, request, state, session, workload=workload)
            # freeze F5 (cut 7): the bridge's part of the pre-denoise time, by step (seconds)
            run.begin_record = {"action": action, "checks_seconds": checked_clock - begun_clock,
                                "controller_seconds": controlled_clock - checked_clock,
                                "release_idle_seconds": run.started - controlled_clock,
                                "before_run_seconds": run.started - begun_clock}
            # Freeze F (Step 7 dev22j): relieve device pressure left by earlier requests before this request's
            # restore and denoise loop allocate: GPU raw copies demote first, then the registered device caches are
            # trimmed (decoder canvases, the restorer's resident bases; never this request's state). SGLang has sized
            # the request's native caches already, so the check asks for the archive's reserve and the KV the loop
            # still allocates (``request_kv_geometry``). Recorded in the run's summary (``pressure``).
            relieve = getattr(self.archive, "relieve_pressure", None)
            if callable(relieve):
                clock = time.perf_counter()
                try:
                    run.kv = self._request_kv_geometry(request, state)
                    event = relieve("request_start", needed_bytes=run.kv["needed_bytes"])
                    if event is not None:
                        run.pressure = {**event, "check_seconds": time.perf_counter() - clock}
                except Exception as exc:  # noqa: BLE001 - best effort: the request proceeds
                    run.pressure = {"error": f"{type(exc).__name__}: {exc}",
                                    "check_seconds": time.perf_counter() - clock}
                run.pressure_events.append(("request_start", run.pressure))
            # Persisted sessions this request reopened (manifest load + optional session prewarm, on the request
            # path): restore reports add them, so a cold resume is never reported without its per-session staging.
            for name in closed:
                opened = controller.get(name, required=False)
                events = [] if opened is None else [item for item in opened.history
                                                     if item["event"] == "reopened" and item["time"] >= begun]
                if events:
                    run.session_open.append({"session": name, "seconds": events[-1].get("seconds"),
                                             "prewarm_seconds": events[-1].get("prewarm_seconds")})
            self._active = run
            return run

    def _request_kv_geometry(self, request: Any, state: Any) -> dict[str, Any]:
        """The request's KV geometry and bytes (:func:`request_kv_geometry`) after the session's generated chunks."""
        with state.lock:
            done = state.generated_chunks
        return request_kv_geometry(request, self.capture_dtypes, done,
                                   (getattr(request, "session", None) or {}).get("pause_after_chunks"))

    def _request_kv_bytes(self, request: Any, state: Any) -> int | None:
        """The device bytes this request still allocates for KV once its run opens (``needed_bytes`` of
        :func:`request_kv_geometry`): its first new chunk's capture; its native caches too when SGLang has not
        allocated them yet."""
        return self._request_kv_geometry(request, state)["needed_bytes"]

    # -- lifecycle ---------------------------------------------------------------------------------------------
    def _release(self, run: _BridgeRun) -> None:
        with self._lock:
            if self._active is run:
                self._active = None

    @property
    def active_request(self) -> bool:
        return self._active is not None

    # -- W1.4.1 (C1): replay promotions vs reuse of the stage caches -------------------------------------------------
    def _note_promotion_copies(self, done: Any) -> None:
        """A replay promotion's aggregate device-to-host completion event: until it completed, every later writer of
        the stage caches waits for it (:meth:`_await_promotion_copies`, the boundary fences ``finish()`` leaves)."""
        with self._lock:
            self._promotion_pending.append(done)

    def _await_promotion_copies(self, target: Any, telemetry: Any = None) -> None:
        """W1.4.1 (C1): at the start of a request that writes the stage caches (before its restore installs, replays
        or forwards): its current stream waits, device-side, for every replay-promotion copy that may still read them;
        an exposed wait is an exposed stall of source ``promotion_fence``. Completed events are dropped; the boundary
        fences an earlier ``finish()`` left on these caches are then lifted (this stream already waits)."""
        with self._lock:
            pending = [event for event in self._promotion_pending if not _event_done(event)]
            self._promotion_pending = pending
        device = getattr(target, "device", None)
        if pending and device is not None and torch.device(device).type == "cuda":
            stream = torch.cuda.current_stream(torch.device(device))
            for event in pending:
                token = (None if telemetry is None
                         else telemetry.stall_begin(stream, ready=True, source="promotion_fence"))
                stream.wait_event(event)
                if telemetry is not None:
                    telemetry.stall_end(token)
        for caches in (getattr(target, "caches", None) or {}).values():
            for cache in caches:
                fence = getattr(cache, "restore_pending", None)
                if isinstance(fence, _PromotionFence) and fence.boundary:
                    cache.restore_pending = fence.previous


    # -- resident native KV --------------------------------------------------------------------------------------
    @property
    def resident(self) -> dict[str, Any] | None:
        with self._resident_lock:
            return None if self._resident is None else self._resident.describe()

    def _set_resident(self, claim: ResidentKV | None) -> None:
        with self._resident_lock:
            self._resident = claim

    def drop_resident(self, session_id: str | None = None) -> bool:
        """Forget the resident native KV (of ``session_id`` only, when given): its next request restores
        from the archive. Returns whether a claim was dropped."""
        with self._resident_lock:
            claim = self._resident
            if claim is None or (session_id is not None and claim.session_id != session_id):
                return False
            self._resident = None
            return True

    def _take_resident(self, state: SGLangSessionState, target: Any, streams: tuple[str, ...]):
        """The session's resident claim when the target caches still hold it (consumed), else ``(None, why)``."""
        with state.lock:
            current = (state.revision, state.generated_chunks)
        with self._resident_lock:
            claim = self._resident
            if not self.config.resident_reuse:
                return None, "disabled"
            if claim is None:
                return None, "none"
            if claim.session_id != state.session_id:
                return None, f"held by session {claim.session_id!r}"
            self._resident = None  # consumed either way: this request writes the caches
        if (claim.revision, claim.chunks) != current or claim.streams != tuple(streams):
            return None, "session head moved"
        for stream in streams:
            caches = target.caches[stream]
            saved = claim.states[stream]
            if len(caches) != len(saved) or not all(
                    callable(getattr(cache, "holds_index_state", None)) and cache.holds_index_state(item)
                    for cache, item in zip(caches, saved)):
                return None, "cache K/V changed (another request or a re-initialized cache)"
        return claim, "resident"

    # -- sessions ------------------------------------------------------------------------------------------------
    def warm_engine_rope(self, transformer: Any) -> None:
        """Process-level warm-up of the engine's RoPE once the pipeline is built (the SGLang backend factory calls it):
        the restore's span tables for every representation (CR-17-2: the first restore builds none), and the engine's
        RoPE kernel wherever a restore rotates with it: RAW (``rope_mode="native"``) and, freeze F6 (cut 9), raw-tier
        ranges of a compressed archive (``raw_restore`` on with a raw tier). Recorded in ``startup["warmup"]``."""
        report = self.startup.get("warmup")
        if not isinstance(report, dict) or "components" not in report:
            return  # warm-up disabled
        started = time.perf_counter()
        warm_rope_tables(report, transformer, self.profile, self.device, self.rope_tables)
        raw_tier_ranges = (getattr(self.config, "raw_restore", "off") != "off"
                           and getattr(self.archive, "raw_tier", None) is not None)
        if (self.config.representation in ("raw", "raw_offload", "latents_only") or self.config.rope_mode == "native"
                or raw_tier_ranges):
            warm_engine_rope(report, transformer, self.profile, self.model, self.device)
        report["seconds"] = report.get("seconds", 0.0) + time.perf_counter() - started

    def open_session(self, session_id: str) -> dict[str, Any]:
        """Open a persisted session ahead of its next request (load-time restore preparation)."""
        return self.controller.open(session_id)

    def checkpoint(self, session_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        """Wait until everything the session generated so far is published (durable head)."""
        return self.controller.checkpoint(session_id, timeout=timeout)

    def drain_all(self, *, timeout: float | None = None) -> dict[str, float]:
        """Checkpoint every open session; raises the first archival failure after trying all of them."""
        waited, failure = {}, None
        for session_id in list(self.controller.sessions()):
            state = self.controller.get(session_id, required=False)
            if state is None or state.status == "running":
                continue
            try:
                waited[session_id] = self.controller.drain(state, timeout=timeout)
            except BaseException as exc:  # noqa: BLE001 - raised after the other sessions
                failure = failure or exc
            if state.detached:  # W1.3 (F6): and its detached branches, closed (garbage collection needs them drained)
                try:
                    waited[session_id] = waited.get(session_id, 0.0) + self.controller.drain_detached(
                        state, timeout=timeout)
                except BaseException as exc:  # noqa: BLE001 - raised after the other sessions
                    failure = failure or exc
        if failure is not None:
            raise failure
        return waited

    def commit_reports(self, request_id: Any, *, wait: bool = True, timeout: float | None = None) -> list:
        """Per-commit reports of a recent request (waits for publication with ``wait``). Resolved commits are kept as
        plain data (:func:`resolved_commit`); the history holds a live ticket only while its commit is pending."""
        entries = self._request_tickets.get(request_id)
        if entries is None:
            raise StateError(f"no recent session request {request_id!r}")
        reports = []
        for entry in entries:
            if entry.get("plain"):
                item = dict(entry["base"])
                try:
                    if entry["error"] is None:
                        resolved = ResolvedTicket(entry["ticket_fields"])
                        item["node"] = resolved.result(0)
                        item["published"] = True
                        item.update(self.archive.commit_report(resolved))
                    else:
                        item["published"] = False
                        item["error"] = entry["error"]
                except BaseException as exc:  # noqa: BLE001 - as for a live ticket
                    item["published"] = False
                    item["error"] = repr(exc)
                item["encoder"] = entry["encoder"]
                reports.append(item)
                continue
            ticket = entry["ticket"]
            item = {key: value for key, value in entry.items() if key != "ticket"}
            try:
                if wait:
                    ticket.result(timeout)
                if ticket.done() and getattr(ticket, "error", None) is None:
                    item["node"] = ticket.result(0)
                    item["published"] = True
                    item.update(self.archive.commit_report(ticket))
                else:
                    item["published"] = False
                    if ticket.done():
                        item["error"] = repr(ticket.error)
            except BaseException as exc:  # noqa: BLE001 - an unpublished (failed/cancelled) commit
                item["published"] = False
                item["error"] = repr(exc)
            item["encoder"] = commit_encoder(self.archive, ticket)
            reports.append(item)
        self._settle_request_history()
        return reports

    def _settle_request_history(self) -> None:
        """Replace the resolved commits of the recent-request history by plain data (records only; never raises):
        a stored ticket of a failed commit carries its exception and traceback, whose frames can pin device memory."""
        for request_id, entries in list(self._request_tickets.items()):
            try:
                self._request_tickets[request_id] = [resolved_commit(self.archive, entry) or entry
                                                     for entry in entries]
            except Exception:  # noqa: BLE001 - records only
                continue

    # -- session controls (Step 14) -------------------------------------------------------------------------------
    def control(self, op: str, args: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The worker's typed session control surface (:mod:`.controls`); raises ``SessionControlError`` only."""
        return self.controls.dispatch(op, args)

    def inflight_control(self, op: str, args: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        """A control received while a request runs (the scheduler's chunk-boundary poll): served now when it
        cannot block generation (read-only ops; pause of the running session), else None (run it later)."""
        return self.controls.dispatch_inflight(op, args)

    def set_boundary_hook(self, hook: Any) -> None:
        """``hook()`` runs at every clean chunk boundary of a session request, after the chunk is submitted and
        before the pause decision (SGLang's scheduler polls for in-flight controls there). None removes it."""
        if hook is not None and not callable(hook):
            raise ConfigurationError("the boundary hook must be callable or None")
        self._boundary_hook = hook

    def _at_boundary(self, run: _BridgeRun) -> None:
        hook = self._boundary_hook
        if hook is None:
            return
        started = time.perf_counter()
        try:
            hook()
        except Exception as exc:  # noqa: BLE001 - a control-plane poll never fails generation
            with run.state.lock:
                run.state.log("boundary_hook_failed", error=repr(exc))
        finally:
            run.hook_seconds += time.perf_counter() - started

    @property
    def observes_plain_requests(self) -> bool:
        """SGLang seam flag: open an observer run for requests without a session (``observe_plain_requests``)."""
        return bool(self.config.observe_plain_requests)

    def _remember(self, request_id: Any, session_id: str | None, summary: dict[str, Any], *,
                  timer: _ForwardTimer | None = None, workload: Any = None) -> None:
        key = request_id if request_id is not None else self.request_key(None)
        self._summaries[key] = {"session_id": session_id, "summary": summary, "timer": timer}
        if workload is not None:  # workload telemetry: resolved by ``request_summary`` (spans may follow: decode)
            self._summaries[key]["workload"] = workload
        self._summaries.move_to_end(key)
        self._awaiting_decode = key  # its decode seam follows on this worker (filed there when it has no id)
        while len(self._summaries) > MAX_REQUEST_SUMMARIES:
            self._summaries.popitem(last=False)
        if len(self.requests) > MAX_REQUEST_ENTRIES:
            del self.requests[:len(self.requests) - MAX_REQUEST_ENTRIES]

    def request_summary(self, request_id: Any) -> dict[str, Any] | None:
        """The finish summary of a recent request (session or observed plain request), or None. Its clean-forward
        durations are resolved here (CUDA events; the request path never waits for them)."""
        entry = self._summaries.get(request_id)
        if entry is None:
            return None
        timer = entry.get("timer")
        if timer is not None:
            entry["summary"]["clean_forwards"] = timer.resolve()
            entry["timer"] = None
        workload = entry.get("workload")
        if workload is not None:  # workload telemetry: every span recorded so far (the decode seam's included)
            entry["summary"]["workload"] = workload.resolve()
        return entry["summary"]

    def workload_record(self, request_id: Any) -> dict[str, Any] | None:
        """The ``workload`` telemetry record of a recent request (CONTRACT §3), resolved now: call it once the request's
        outputs are complete (after its RGB reached the host). None without ``workload_telemetry`` or such request."""
        summary = self.request_summary(request_id)
        return None if summary is None else summary.get("workload")

    @holds_decode_seam
    def memory_critical_section(self, reason: str, *, request_id: Any = None):
        """SGLang seam (optional): a memory-critical section of the worker, e.g. the VAE decode. With
        ``defer_archival_during_decode`` background archival starts no GPU fitting inside it and releases the
        fitter's cached memory on entry (Step 7 ``defer_gpu_work``). With a raw tier (freeze F) the archive's section
        also runs the device-pressure response on entry (GPU raw demotion, registered trims) and relieves an
        out-of-memory error inside it before re-raising it. The section is recorded in the summary of ``request_id``
        (``memory_critical``, and its pressure events in ``pressure``), also when it raises. A no-op context
        otherwise."""
        section = getattr(self.archive, "memory_critical", None)
        # Freeze F (Step 7 dev22j): with a raw tier the archive's seam also runs the device-pressure response (GPU raw
        # demotion, registered trims, OOM relief), so it is entered whenever a raw tier exists.
        if section is None or not (self.config.defer_archival_during_decode
                                   or getattr(self.archive, "raw_tier", None) is not None):
            return contextlib.nullcontext()
        return self._memory_critical(section, reason, request_id)

    @contextlib.contextmanager
    def _restore_section(self, run: _BridgeRun):
        """The archive's memory-critical section around an archive restore (freeze F, reason ``restore``): its entry
        check, OOM relief and (Step 7 dev22j2) pressure-triggered archival deferral. Entered like the decode seam:
        with a raw tier or the deferral policy (C3 configurations restore as before). Its report, also when the restore
        raises, joins the run's pressure events; yields the report (None when not entered). The caller wraps only the
        install and the restore: inside a deferred section the pipeline's drain / prewarm raise and submit refuses, so
        the publication drain runs before it and no chunk is submitted inside."""
        section = getattr(self.archive, "memory_critical", None)
        if section is None or not (self.config.defer_archival_during_decode
                                   or getattr(self.archive, "raw_tier", None) is not None):
            yield None
            return
        report, entry_seconds = None, None
        clock = time.perf_counter()
        try:
            with section("restore") as report:
                entry_seconds = time.perf_counter() - clock
                yield report
        finally:
            if isinstance(report, Mapping):
                report["entry_seconds"] = entry_seconds
                run.pressure_events.append(("restore_section", report.get("raw_pressure")))
                if report.get("raw_pressure_error"):
                    run.pressure_events.append(("restore_section", {"error": report["raw_pressure_error"]}))
                run.pressure_events.append(("restore_section", recheck_event(report)))
                run.pressure_events.append(("oom_relief", report.get("oom_relief")))

    @contextlib.contextmanager
    def _memory_critical(self, section, reason: str, request_id: Any):
        report, entry_seconds = None, None
        entry = self._summaries.get(request_id if request_id is not None else self._awaiting_decode)
        workload = None if entry is None else entry.get("workload")  # workload telemetry: the entry guard's wait
        guard = None if workload is None else workload.begin("guard_wait")
        clock = time.perf_counter()
        try:
            with section(reason) as report:
                entry_seconds = time.perf_counter() - clock
                if guard is not None:
                    workload.end(guard)
                yield report
        finally:
            if guard is not None:
                workload.end(guard)  # no-op once ended
            if isinstance(report, Mapping):
                self._file_section(reason, request_id, report, entry_seconds)

    def _file_section(self, reason: str, request_id: Any, report: Mapping[str, Any], entry_seconds: Any) -> None:
        """File a memory-critical section's record (F3, never dropped): with the summary of ``request_id``; without
        an id, with the request that just finished on this worker (its first seam; ``request_id_missing``);
        otherwise in ``unattributed_sections`` (``summary_missing``)."""
        record = {**report, "entry_seconds": entry_seconds}
        # Step 7 dev22j6: the decode entry guard's record (a plain copy) and its summary, per decode section
        guard = decode_guard_summary(report)
        if guard is not None:
            record["decode_guard"] = plain(report["decode_guard"])
            record["decode_guard_summary"] = guard
        key = request_id
        if request_id is None:
            record["request_id_missing"] = True
            key, self._awaiting_decode = self._awaiting_decode, None
        elif request_id == self._awaiting_decode:
            self._awaiting_decode = None
        entry = self._summaries.get(key) if key is not None else None
        if entry is None:
            self.unattributed_sections.append({**record, "request_id": request_id, "summary_missing": True,
                                               "time": time.time()})
            return
        if request_id is None:
            record["filed_under"] = key
        summary = entry["summary"]
        summary.setdefault("memory_critical", []).append(record)
        events = [(reason, report.get("raw_pressure")), (reason, recheck_event(report)),
                  ("oom_relief", report.get("oom_relief"))]
        if report.get("raw_pressure_error"):
            events.append((reason, {"error": report["raw_pressure_error"]}))
        if guard is not None:
            events.extend(decode_guard_events(reason, report, guard))
        if any(isinstance(event, Mapping) for _, event in events) or guard is not None:
            pressure = summary.get("pressure")
            if not isinstance(pressure, dict):
                pressure = summary["pressure"] = pressure_record(None, [])
            for source, event in events:
                add_pressure_event(pressure, source, event)
            if guard is not None:
                add_decode_guard(pressure, guard)

    def session_requests(self, session_id: str) -> list[dict[str, Any]]:
        """Recent requests of a session (oldest first): ids and outcome."""
        result = []
        for request_id, entry in self._summaries.items():
            if entry["session_id"] != session_id:
                continue
            summary = entry["summary"]
            result.append({"request_id": request_id, "action": summary.get("action"),
                           "status": summary.get("status"), "resumed_from_chunks": summary.get("resumed_from_chunks"),
                           "generated_chunks": summary.get("generated_chunks"),
                           "restore_mode": (summary.get("restore") or {}).get("mode")})
        return result

    def collect_garbage(self) -> dict[str, Any]:
        """Publish every open session's pending commits, then reclaim the store objects that no session head,
        named checkpoint or lease reaches. Refuses while a request runs (it would race the request's commits)."""
        started = time.perf_counter()
        with self._lock:
            if self._active is not None:
                raise PendingError("garbage collection waits until the running request returns")
            failure = None
            try:
                drained = self.drain_all()
            except BaseException as exc:  # noqa: BLE001 - acknowledged: its session continues from its checkpoint
                drained, failure = None, repr(exc)
            open_branches = self.controller.detached_sessions()  # W1.3 (F6): never around detached work in flight
            if open_branches:
                raise PendingError(f"garbage collection waits for the detached branches of sessions {open_branches} "
                                   f"({failure})")
            collected = self.archive.collect_garbage()
        return {**collected, "drained": drained, "drain_failure": failure,
                "seconds": time.perf_counter() - started}

    def describe(self) -> dict[str, Any]:
        described = {**self.startup, "sessions": self.controller.sessions(), "archive_stats": self.archive.stats(),
                     "resident": self.resident, "requests": list(self.requests),
                     "unattributed_memory_critical": list(self.unattributed_sections)}
        store = getattr(self.archive, "store", None)
        if getattr(store, "io_policy", "buffered") != "buffered":  # CONTRACT §10: every arm's policy and bytes
            described["store_io"] = store.io_stats()
        if self.controller.switch_mode == "detach":  # W1.3: detached branches and commits, backlog, fallbacks (F7)
            described["tree_switch"] = self.controller.switch_stats()
        retain = getattr(self.config, "raw_tier_retain", "drop")
        promote = getattr(self.config, "raw_tier_promote", "off")
        if (retain, promote) != ("drop", "off"):  # wl-w1b7: the raw-tier LRU policy (absent at the defaults)
            from .config import PROMOTION_SCOPES

            # W1.4.3 (E1): an evicting ablation arm promotes its replays
            scope = PROMOTION_SCOPES.get(self.config.representation,
                                         "replay" if getattr(self.config, "archive_budget_bytes", None) else None)
            described["raw_tier_policy"] = {
                "retain": retain, "promote": promote,
                "promotion_scope": scope if promote == "host" else None,
                "promotion_fences": (scope == "replay" and promotion_fences_needed(self.profile))
                if promote == "host" else None}
        return described

    @closes_open_holds("close")
    def close(self) -> None:
        """Publish pending archival and retire restore resources and the runtime; refuses while a request runs."""
        with self._lock:
            if self._closed:
                return
            if self._active is not None:
                raise PendingError("cannot close the ReelCache bridge while a session request is running")
            self._closed = True
        self.drop_resident()
        with self._lock:  # W1.4.1 (C1): no promotion copy reads the stage caches past close (host sync: shutdown)
            pending, self._promotion_pending = self._promotion_pending, []
        for event in pending:
            with contextlib.suppress(Exception):
                event.synchronize()
        self.archive.close()

    def _close_at_exit(self) -> None:
        try:
            if self._active is None:
                self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass


class _BridgeRun:
    """Per-request run (SGLang ``CausalKVRun``): restore, capture, commit, recache and pause decisions."""

    def __init__(self, bridge: CausalWanKVBridge, request: Any, state: SGLangSessionState, session: dict,
                 workload: Any = None):
        self.bridge, self.request, self.state = bridge, request, state
        # F3: the key this run's records are filed under (its request id, or an internal key without one)
        self.key = bridge.request_key(request.request_id)
        self.session = dict(session)
        self.pause_after = session.get("pause_after_chunks")
        self.resume_blocks = 0
        self.target = None
        self.builder: CleanChunkBuilder | None = None
        self.tickets: list[dict[str, Any]] = []
        self.restore_report: dict[str, Any] | None = None
        self.paused = False
        self.finished = False
        self.summary: dict[str, Any] | None = None
        self.started = time.perf_counter()
        self.session_open: list[dict[str, Any]] = []
        self.released_sources = 0
        self.chunk_marks: list[dict[str, Any]] = []
        self.dispatched_at = getattr(request, "dispatched_at", None)
        self.forward_timer = _ForwardTimer(bridge.device)
        # workload telemetry (W1a; config ``workload_telemetry``, default off: None, no event, F6 records); W1.1:
        # ``begin_request`` creates it before the controller step (drain spans)
        self.workload_telemetry = workload
        if workload is None and getattr(bridge.config, "workload_telemetry", False):
            self.workload_telemetry = WorkloadTelemetry(bridge.device)
        self.chunk_digests: dict[int, str] = {}
        self.hook_seconds = 0.0
        # Freeze F: the request-start pressure event (None: no check, no device or response off) and the request's
        # KV geometry (``request_kv_geometry``); the summary's ``pressure`` record.
        self.pressure: dict[str, Any] | None = None
        self.kv: dict[str, Any] | None = None
        # every pressure event of the run as (source, event): request start, the pre-restore relief, the restore
        # section's entry check and OOM relief (decode-seam events join the summary when the decode runs)
        self.pressure_events: list[tuple[str, Any]] = []
        # freeze F5: whether this request's restore is layer-overlapped, and its overlapped restore until joined
        self.overlap_requested = bridge.overlap_requested(session)
        self.overlap_unsupported: str | None = None  # requested, but the stage's caches have no seam: plain
        self.begin_record: dict[str, Any] = {}  # freeze F5 (cut 7): set by ``begin_request``
        self.overlap: _Overlap | None = None
        sizes = list(request.block_sizes)
        self._block_starts = [sum(sizes[:index]) for index in range(len(sizes))]
        # Archived chunk ranges of this session (generated head: restored + submitted chunks): recache validation.
        self.chunks: dict[int, dict[str, Any]] = {}
        # CPU latents of every chunk on the generated head and the RNG state after the newest one: what the
        # session's next request needs when it continues on resident KV.
        self.latents_cpu: dict[int, torch.Tensor] = {}
        self.generator_state: dict[str, torch.Tensor] | None = None
        self.blocks = block_conditioning(request)

    def _block(self, index: int) -> str | None:
        return None if self.blocks is None or index >= len(self.blocks) else self.blocks[index]

    def _head_report(self) -> dict[str, Any]:
        """Kind of the restored head (``chunk`` or a ``recache`` epoch and the block it precedes): a stage that
        resumes exactly at a prompt switch reruns its recache only if the switch's epoch was not restored."""
        with self.state.lock:
            last = self.state.timeline[-1] if self.state.timeline else None
        if last is None:
            return {"head_kind": "root", "head_epoch": None, "head_block": None}
        return {"head_kind": last["kind"], "head_epoch": last["epoch"], "head_block": last["block"]}

    def _raw_restore_policy(self) -> str | None:
        """The archive's raw-restore policy (freeze D ``raw_restore``) while it has a raw tier, else None."""
        if getattr(self.bridge.archive, "raw_tier", None) is None:
            return None
        return getattr(self.bridge.config, "raw_restore", None)

    # -- restore -----------------------------------------------------------------------------------------------
    def _prepare_next_plans(self) -> dict[str, Any]:
        """Freeze F5 (cut 3): plan the restore of this session's next resume now (its head is published), so a later
        restore of the same head into caches of this geometry finds its plan. Best effort; recorded."""
        clock = time.perf_counter()
        state, request = self.state, self.request
        done = state.generated_chunks
        if done < 1 or done >= len(request.block_sizes):
            return {"skipped": "no next block in this request's layout", "seconds": 0.0}
        try:
            nodes = self.bridge.archive.path(state.head_ref)[1:]
            _, hits = build_plans(self.target.caches, request.streams, nodes, self._block_token_range(done),
                                  request.tokens_per_frame, self.bridge.plans, prepared=True)
            return {"head": state.head_ref, "chunks": done, "already_cached": hits,
                    "seconds": time.perf_counter() - clock}
        except Exception as exc:  # noqa: BLE001 - an optimization: the restore plans on a miss
            return {"error": f"{type(exc).__name__}: {exc}", "seconds": time.perf_counter() - clock}

    def _block_token_range(self, index: int) -> tuple[int, int]:
        request = self.request
        return ((request.pos_start_base + self._block_starts[index]) * request.tokens_per_frame,
                request.block_sizes[index] * request.tokens_per_frame)

    def _check_chunk(self, index: int, start_frame: int, token_start: int, num_tokens: int) -> None:
        if (start_frame, (token_start, num_tokens)) != (self._block_starts[index], self._block_token_range(index)):
            raise IntegrityError(f"archived chunk {index} does not match the request's block layout")

    def _install_latents(self, index: int, stored: torch.Tensor, num_frames: int) -> None:
        start = self._block_starts[index]
        region = self.target.latents[:, :, start:start + num_frames]
        if tuple(stored.shape) != tuple(region.shape) or stored.dtype != region.dtype:
            raise IntegrityError(f"archived latents of chunk {index} are {tuple(stored.shape)} {stored.dtype}, "
                                 f"the request expects {tuple(region.shape)} {region.dtype}")
        if region.is_cuda and stored.device.type == "cpu":
            # Freeze F5 (cut 6): from pinned memory, without a host wait; the forward runs after it on this stream
            # (the caching host allocator keeps the pinned block until the copy has run)
            region.copy_(stored.pin_memory(), non_blocking=True)
        else:
            region.copy_(stored)

    def _set_generators(self, saved: Mapping[str, torch.Tensor]) -> None:
        for index, generator in enumerate(_generators(self.target.generator)):
            value = saved.get(f"{GENERATOR_PREFIX}{index}")
            if value is None:
                raise IntegrityError("the checkpoint has no generator state for this request")
            generator.set_state(value)
        self.generator_state = {key: value for key, value in saved.items() if key.startswith(GENERATOR_PREFIX)}

    @opens_after("restore")
    def restore(self, target: Any) -> int:
        request, state = self.request, self.state
        self.target = target
        # W1.4.1 (C1): before this request writes the stage caches, wait for replay-promotion copies still reading them
        self.bridge._await_promotion_copies(target, self.workload_telemetry)
        for stream in request.streams:
            caches = target.caches.get(stream)
            if caches is None or len(caches) != request.num_layers:
                raise IntegrityError(f"stream {stream!r} has no complete per-layer cache list")
        with state.lock:
            done = state.generated_chunks
        if done >= len(request.block_sizes):
            raise ConfigurationError(
                f"session {state.session_id!r} already has {done} chunks; the request covers only "
                f"{len(request.block_sizes)} (ask for more frames to continue)"
            )
        if done == 0:
            for stream in request.streams:
                for cache in target.caches[stream]:
                    if cache._read_indices() != (0, 0):
                        raise IntegrityError("a new session must start from reset causal caches")
            self.bridge.drop_resident()  # this request writes the caches
            self.restore_report = {"mode": "none", "restored_chunks": 0}
            state.lossy_history = False  # W1.4.3 (review C1): a history computed natively from the start
            return 0
        claim, reason = self.bridge._take_resident(state, target, tuple(request.streams))
        if claim is not None:
            return self._restore_resident(claim, done)
        self.bridge.drop_resident()  # the restore and this request write the caches
        return self._restore_archive(done, reason)

    def _restore_resident(self, claim: ResidentKV, done: int) -> int:
        """Continue on the session's native KV still resident in the stage caches: cursors, latents, RNG."""
        request, state, target = self.request, self.state, self.target
        started = time.perf_counter()
        exec_span = None if self.workload_telemetry is None else self.workload_telemetry.begin("exec_state")
        with state.lock:
            table = {index: dict(item) for index, item in state.chunk_table.items() if index < done}
            pending = [item["node_id"] for item in state.pending]
        if sorted(table) != list(range(done)) or sorted(claim.latents) != list(range(done)):
            raise IntegrityError("the resident session state does not cover its generated chunks")
        for index, item in table.items():
            self._check_chunk(index, self._block_starts[index], item["token_start"], item["num_tokens"])
        for stream in request.streams:
            for cache, saved in zip(target.caches[stream], claim.states[stream]):
                cache.restore_index_state(saved)
        for index in range(done):
            self._install_latents(index, claim.latents[index], table[index]["num_frames"])
        self._set_generators(claim.generators)
        self.chunks = table
        self.latents_cpu = dict(claim.latents)
        self.resume_blocks = done
        seconds = time.perf_counter() - started
        if exec_span is not None:
            self.workload_telemetry.end(exec_span)
        self.restore_report = {
            "mode": "resident", "restored_chunks": done, "head": state.head_ref, **self._head_report(),
            "pending_at_resume": pending, "publish_wait_seconds": 0.0, "restore_seconds": seconds,
            "total_seconds": seconds, "resident_from_request": claim.request_id,
            # freeze D records (after the timing): every chunk's tier, in the archive restore's lease shape
            "lease": resident_lease_record(request.streams, table, policy=self._raw_restore_policy(),
                                           verification_mode=configured_verification_mode(self.bridge.config)),
        }
        with state.lock:
            state.log("resumed_resident", chunks=done, pending=len(pending))
        return done

    def _restore_archive(self, done: int, resident: str) -> int:
        """Restore from the archive: wait for the session's published frontier, then install attended history.
        With a raw tier (freeze D) the archive serves each attended range from its best tier (gpu_raw > host_raw >
        compressed); ``restore.lease`` carries its ``tiers`` and ``verification`` records (``lease.stats``)."""
        request, archive, state, target = self.request, self.bridge.archive, self.state, self.target
        started = time.perf_counter()
        gc_before = GC_PAUSES.snapshot()
        workload = self.workload_telemetry  # workload telemetry spans (None: off)
        waiting = None if workload is None else workload.begin("publish_wait")
        # A restore that needs unpublished nodes waits for them (raises the session's archival failure).
        wait_seconds = self.bridge.controller.drain(state)
        if workload is not None:
            workload.end(waiting)
        path = archive.path(state.head_ref)
        if path[0].ref != state.root_ref:
            raise IntegrityError("session head does not descend from its root")
        nodes = path[1:]
        chunks = [node for node in nodes if node.kind == "chunk"]
        if len(chunks) != done:
            raise IntegrityError(f"session {state.session_id!r} has {len(chunks)} published chunks after its "
                                 f"drain, {done} generated")
        for index, node in enumerate(chunks):
            if node.chunk_index != index:
                raise IntegrityError(f"archived chunk {index} does not match the request's block layout")
            self._check_chunk(index, node.start_frame, node.token_start, node.num_tokens)
            if tuple(node.streams) != tuple(request.streams):
                raise IntegrityError("archived chunk streams differ from the request's CFG streams")
            self.chunks[index] = {"token_start": node.token_start, "num_tokens": node.num_tokens,
                                  "rope_start_frame": node.rope_start_frame, "num_frames": node.num_frames}
        restore_started = time.perf_counter()
        plans, plan_hits = build_plans(target.caches, request.streams, nodes, self._block_token_range(done),
                                       request.tokens_per_frame, self.bridge.plans)
        plan_seconds = time.perf_counter() - restore_started
        if workload is not None:  # restore_plan (host-only: CPU planning; perf_counter is CLOCK_MONOTONIC); W1.1:
            # before the recompute arm's branch (both arms plan)
            workload.add_host_span("restore_plan", int(restore_started * 1e9),
                                   int((restore_started + plan_seconds) * 1e9))
        if getattr(self.bridge.config, "restore_mode", "archive") == "recompute":  # W1b: the exact recompute arm
            return self._restore_recompute(done, resident, nodes, chunks, plans, plan_hits,
                                           {"started": started, "wait": wait_seconds, "plan": plan_seconds,
                                            "gc_before": gc_before})
        if getattr(archive, "eviction", None) is not None:  # W1.4.3 (E1): an evicting ablation arm
            # the path's archived nodes and its evicted nodes' usable raw copies are held (no eviction, no drop)
            # until the restore pinned what it reads (released below; at the latest by finish): this choice holds
            available, self._eviction_hold = archive.hold_available(node.ref for node in nodes)
            attended = {span.node_ref for plan in plans.values() for span in plan.spans}
            missing = attended - available
            if missing:
                return self._restore_evicted(done, resident, nodes, chunks, plans, plan_hits, missing, available,
                                             {"started": started, "wait": wait_seconds, "plan": plan_seconds,
                                              "gc_before": gc_before})
        # Freeze F (coordinator 07:40): relieve device pressure for the restore's working set and the request's next
        # chunk before anything is installed (after the drain: newly published GPU raw copies are demotable), then run
        # the restore inside the archive's memory-critical section ("restore": OOM relief; pressure-triggered
        # archival deferral from Step 7 dev22j2). The lease is closed inside the section, so its exit never waits on
        # copies the restore still holds.
        if workload is not None:  # the synchronous restore: restore_sync, the archive's tier work restore_sync.<tier>
            # (one partitioned window)
            segments = workload.segments("restore_sync")
        pool = getattr(archive, "pool", None)
        # Freeze F5: a layer-overlapped restore (default on): the first denoising forward waits per layer and the run
        # joins after its first clean forward (``_join_overlap``), before anything derived leaves the request
        overlapped = self.overlap_requested and overlap_seam({stream: target.caches[stream]
                                                              for stream in request.streams})
        if self.overlap_requested and not overlapped:
            self.overlap_unsupported = ("the stage's caches have no restore_pending seam (await_restored_history): "
                                        "restored plain")
        # Freeze F5 (cut 4): a restore that allocates nothing beyond the caches SGLang allocated already runs without
        # the pre-restore relief and the restore section (0.5-0.7 ms in P4). The archive answers for its sources
        # (every attended range a GPU raw copy read in place); the in-place install still takes the engine RoPE's
        # output from the caching allocator, which the bridge counts as non-allocating only when the device's
        # headroom covers its bound now (:meth:`_restore_scratch`)
        archive_allocates = self._restore_allocates(plans)
        scratch = None if archive_allocates else self._restore_scratch(plans)
        allocates = archive_allocates or not scratch["covered"]
        relief = (self._relieve_before_restore(plans) if allocates else
                  {"estimate": None, "needed_bytes": None, "seconds": 0.0, "skipped": NO_ALLOCATION_SKIP})
        overlap = None
        if overlapped:
            overlap = self._start_overlap(plans, section=allocates)
        if overlap is not None:
            if not overlap.joined:
                self.overlap = overlap  # before anything below can raise: finish() joins it
            lease, section, rope_tables = overlap.lease, overlap.section, overlap.rope_tables
            pool_before = overlap.pool_before
        else:
            with (self.bridge._restore_section(self) if allocates else contextlib.nullcontext()) as section:
                for stream in request.streams:
                    for cache in target.caches[stream]:
                        cache.install_restored_history(plans[stream].layout, lambda span, k_slots, v_slots: None)
                # The engine's span tables through the process cache (CR-17-2): the same tables, built at attach or
                # once.
                rope_tables = self.bridge.rope_tables.bind(target.rope_cos_sin, grid=tuple(request.grid),
                                                           device=target.device)
                # records only: the decoder pool's state at submit and its counters' change over the restore
                pool_before = decoder_pool_snapshot(pool)
                lease = archive.restore(head_ref=state.head_ref, plans=plans, caches=target.caches,
                                        rope_cos_sin=rope_tables, rotate_key=target.rotate_key,
                                        rope_mode="native" if self.bridge.config.representation == "raw"
                                        else self.bridge.config.rope_mode,
                                        num_layers=request.num_layers, grid=tuple(request.grid),
                                        **self._promotion_kwargs())
                try:
                    lease.wait()
                finally:
                    lease.close()  # host-waits for every reconstruction before returning control to generation
                if target.device.type == "cuda":
                    torch.cuda.current_stream(target.device).synchronize()
        kv_seconds = time.perf_counter() - restore_started
        self._release_eviction_hold()  # W1.4.3 (E1): the restore pinned what it reads
        if getattr(archive, "eviction", None) is not None:  # W1.4.3 (review C1): what the session captures next
            state.lossy_history = not self._exact_raw(lease)
        if workload is not None:
            segments.close()
        pool_after = decoder_pool_snapshot(pool)
        state_started = time.perf_counter()
        exec_span = None if workload is None else workload.begin("exec_state")
        for index, node in enumerate(chunks):
            stored = archive.chunk_latents(node, device="cpu")
            self._install_latents(index, stored, node.num_frames)
            self.latents_cpu[index] = stored
        self._set_generators(archive.execution_state(nodes[-1]))
        if workload is not None:
            workload.end(exec_span)
        with state.lock:
            state.chunk_table = {index: dict(item) for index, item in self.chunks.items()}
            state.timeline = timeline_of(nodes)
        self.resume_blocks = done
        gc_after = GC_PAUSES.snapshot()
        self.restore_report = {
            "mode": "archive", "resident": resident, "restored_chunks": done, "head": state.head_ref,
            **self._head_report(),
            # W1.4.3 (E1): an evicting arm's restore that found every attended node available
            **({"archive_evicted_hits": 0} if getattr(archive, "eviction", None) is not None else {}),
            "publish_wait_seconds": wait_seconds, "plan_seconds": plan_seconds, "restore_seconds": kv_seconds,
            # freeze F5: plan-cache hits per stream (cut 3); whether the restore allocates device memory (cut 4: when
            # it does not, no relief and no section), the archive's answer and the RoPE scratch check
            "plan_cache_hits": plan_hits, "allocates": allocates, "archive_allocates": archive_allocates,
            "scratch": scratch,
            "execution_state_seconds": time.perf_counter() - state_started,
            "total_seconds": time.perf_counter() - started,
            # Persisted sessions reopened by this request (manifest load + session prewarm if configured).
            "session_open": list(self.session_open),
            "total_with_session_open_seconds": time.perf_counter() - started + sum(
                item["seconds"] or 0.0 for item in self.session_open),
            "plans": {stream: plan.to_dict() for stream, plan in plans.items()},
            "lease": {**lease.description, **lease.stats, "closed": lease.closed, "open_handles": lease.open_handles},
            "rope_tables": rope_tables.report(),
            "decoder_pool": decoder_pool_record(pool_before, pool_after),
            "gc": gc_pause_record(gc_before, gc_after),  # garbage collections inside the restore's window
            # freeze F: the pre-restore relief (working-set estimate, needed bytes, its seconds) and the restore's
            # memory-critical section (entry check, learned working set, entry seconds); None when not entered
            "pressure": {**relief, "section": None if section is None else dict(section)},
            # freeze F5: whether the restore was asked to overlap (config default or the handle) and its record (None:
            # plain); completed when the run joins it
            "overlap_requested": self.overlap_requested, "overlap_unsupported": self.overlap_unsupported,
            "overlap": None if overlap is None else overlap.record(),
        }
        if "promotion" in lease.stats:  # wl-w1b7: raw_offload store reads promoted into host raw
            self.restore_report["promotion"] = lease.stats["promotion"]
        if self.bridge.controller.tree_mode:  # W1b: the head switch of a rewind (its drain, from / to)
            self.restore_report["switch"] = self._switch_record()
        return done

    # -- wl-w1b7: restore promotion (raw_tier_promote="host"; raw_offload and the recompute arm only) -----------------
    def _promotion_kwargs(self) -> dict[str, Any]:
        """The archive restore's ``promote_holder`` when raw_offload promotes its store reads (with workload
        telemetry also ``promote_span``: each copy's host interval, a ``promotion_copy`` span), else nothing."""
        config = self.bridge.config
        if getattr(config, "raw_tier_promote", "off") == "host" and config.representation == "raw_offload":
            kwargs: dict[str, Any] = {"promote_holder": self.state.session_id}
            telemetry = getattr(self, "workload_telemetry", None)
            if telemetry is not None:
                kwargs["promote_span"] = lambda start_ns, end_ns: telemetry.add_host_span(
                    "promotion_copy", start_ns, end_ns)
            return kwargs
        return {}

    def _promotion_boundary(self, error: BaseException | None) -> None:
        """W1.4.1 (C1): this request's replay-promotion copies may still read the stage caches when it ends. On
        success every layer of its caches gets a boundary fence (``restore_pending``): whatever writes them next --
        another session's restore, a plain request, a resident continuation -- first waits for the copies on its own
        stream (device-side; a session request waits explicitly and records the stall,
        :meth:`CausalWanKVBridge._await_promotion_copies`). On failure or abort the copies are waited for on the
        host; if that wait fails, the boundary fences stay (the tier quarantines the copies' blocks)."""
        target = self.target
        if getattr(self, "_deferred_promotion", None) is not None:  # N1: nothing accessed the caches after the replay
            if error is None and target is not None:
                try:  # their slots still hold the replay: issue now; the boundary below holds back later writers
                    self._issue_deferred_promotion()
                except Exception:  # noqa: BLE001 - an optimization: the request stands
                    self._drop_deferred_promotion()
            else:
                self._drop_deferred_promotion()
        self._lift_triggers()  # a layer no forward touched keeps no trigger past its request
        events = getattr(self, "_promotion_done", None)
        if not events:
            return
        self._promotion_done = []
        if error is not None:
            try:
                for event in events:
                    event.synchronize()
                return
            except Exception:  # noqa: BLE001 - the device-side fences below still order later writers
                pass
        if target is None:
            return
        for event in events:
            for caches in target.caches.values():
                for layer, cache in enumerate(caches):
                    cache.restore_pending = _PromotionFence(event, cache.restore_pending, None, layer, boundary=True)

    def _prepare_replay_promotion(self, context: Mapping[str, Any]) -> dict[str, Any]:
        """After a recompute replay: PREPARE the promotion of every node the replay installed in full (every stream,
        its whole range) into host raw in installed form, held by this session -- admission and pinned destination
        blocks, now (prep). W1.4.1 (N1): the device-to-host copies are issued by the first access after the replay
        (a :class:`_PromotionTrigger` on every cache layer): the first new chunk's denoise forward, after the stage's
        pre-generation device synchronize, so they overlap generation (or a recache before it, whose writes must not
        reach the slots first). Returns the restore record's ``promotion`` (completed when issued)."""
        request, target, archive = self.request, self.target, self.bridge.archive
        tier = getattr(archive, "raw_tier", None)
        started = time.perf_counter()
        report: dict[str, Any] = {"scope": "replay", "nodes": 0, "bytes": 0, "host_bytes": 0, "skips": {},
                                  "skip_bytes": {}, "fenced_layers": 0, "fence_waits": 0, "issued": False,
                                  "exposed_wait_seconds": 0.0,
                                  "rolling_cache": promotion_fences_needed(self.bridge.profile)}
        if tier is None or tier.promotion_mode != "host":
            report["skips"]["disabled"] = 1
            return report
        from ... import restore_kernels as rk
        from ...types import MatrixKey

        streams = tuple(request.streams)

        def token_bytes(stream: str) -> int:  # K and V of one token over every layer
            return sum(cache.k[0, 0].numel() * cache.k.element_size() + cache.v[0, 0].numel() * cache.v.element_size()
                       for cache in target.caches[stream])

        by_node: dict[str, dict[str, list]] = {}
        only = context.get("only")  # W1.4.3 (E1): only the replayed nodes (an archive-installed ancestor is lossy)
        for stream, plan in context["plans"].items():
            for span in plan.spans:
                if only is None or span.node_ref in only:
                    by_node.setdefault(span.node_ref, {}).setdefault(stream, []).append(span)
        candidates = []
        for ref, per_stream in by_node.items():
            starts = {stream: sorted(span.token_start for span in spans) for stream, spans in per_stream.items()}
            if (set(per_stream) != set(streams) or len({tuple(item) for item in starts.values()}) != 1
                    or any(span.node_offset or span.length != span.range_tokens
                           for spans in per_stream.values() for span in spans)):
                nbytes = sum(span.length * token_bytes(stream) for stream, spans in per_stream.items()
                             for span in spans)
                report["skips"]["partial_range"] = report["skips"].get("partial_range", 0) + 1
                report["skip_bytes"]["partial_range"] = report["skip_bytes"].get("partial_range", 0) + nbytes
                tier.count_promotion_skip("partial_range", nbytes)
                continue
            groups = []
            for first in sorted(per_stream[streams[0]], key=lambda span: span.token_start):
                matrices = {}
                for stream in streams:
                    span = next(item for item in per_stream[stream] if item.token_start == first.token_start)
                    for layer in range(request.num_layers):
                        cache = target.caches[stream][layer]
                        for name, buffer in (("K", cache.k), ("V", cache.v)):
                            matrices[(MatrixKey(layer, name), stream)] = rk.NativeView(
                                buffer, "BTHD", token_offset=span.slot_start).slot(span.length)
                groups.append((f"c{first.chunk_index:06d}", first.token_start, matrices))
            candidates.append({"node_ref": ref, "groups": groups})
        if candidates:
            prepared = tier.promotion_prepare_installed(candidates, holder=self.state.session_id, origin="replay",
                                                        device=target.device,
                                                        **({"exact": False} if context.get("exact") is False else {}))
            self._deferred_promotion = prepared
            for stream in streams:
                for layer, cache in enumerate(target.caches[stream]):
                    cache.restore_pending = _PromotionTrigger(self, cache.restore_pending, stream, layer)
        report["prepare_seconds"] = time.perf_counter() - started
        return report

    def _issue_deferred_promotion(self) -> Mapping[tuple, Any]:
        """W1.4.1 (N1): issue the prepared replay promotion, once (the first :class:`_PromotionTrigger` to run);
        returns the per-``(stream, layer)`` fence events of a rolling cache (``{}`` for full history)."""
        prepared, self._deferred_promotion = getattr(self, "_deferred_promotion", None), None
        if prepared is None:
            return getattr(self, "_promotion_fences", None) or {}
        tier, target = self.bridge.archive.raw_tier, self.target
        report = (self.restore_report or {}).get("promotion")
        report = report if isinstance(report, dict) else {}
        started = time.perf_counter()
        after = None
        if target.device.type == "cuda":
            after = torch.cuda.Event()
            after.record(torch.cuda.current_stream(target.device))
        out = tier.promotion_issue_installed(prepared, after_event=after)
        # I1: the safety work first, before telemetry or any record. C1: the aggregate completion event holds back
        # every later writer of these caches (the next request's restore; plain requests and resident continuations
        # through the boundary fences finish() leaves); rolling caches fence each layer of this request's forwards.
        start, done = out.get("copy_events") or (None, None)
        if done is not None:
            self._promotion_done = [*(getattr(self, "_promotion_done", None) or ()), done]
            self.bridge._note_promotion_copies(done)
        fences: dict[tuple, Any] = {}
        if out["promoted"] and report.get("rolling_cache"):
            events = out["layer_events"]
            fences = {(stream, layer): events.get(layer) for stream in self.request.streams
                      for layer in range(len(target.caches[stream]))}
        self._promotion_fences = fences
        report.update(issued=True, nodes=len(out["promoted"]), bytes=out["bytes"], host_bytes=out["host_bytes"],
                      fenced_layers=len(fences))
        for reason, count in out["skips"].items():
            report.setdefault("skips", {})[reason] = report.get("skips", {}).get(reason, 0) + count
            report.setdefault("skip_bytes", {})[reason] = (report.get("skip_bytes", {}).get(reason, 0)
                                                           + out["skip_bytes"].get(reason, 0))
        if "error" in out:
            report["error"] = out["error"]
        report["issue_seconds"] = time.perf_counter() - started
        telemetry = getattr(self, "workload_telemetry", None)
        if telemetry is not None and out["promoted"] and out.get("issue_ns") is not None:
            # the device-to-host copies, GPU-timed on the tier's copy stream (hidden side work)
            telemetry.add_event_span("promotion_copy", start, done, *out["issue_ns"], stream="promotion")
        return fences

    def _drop_deferred_promotion(self) -> None:
        """A prepared promotion that will not be issued (its request failed): its blocks go back, its triggers go."""
        prepared, self._deferred_promotion = getattr(self, "_deferred_promotion", None), None
        if prepared is not None:
            tier = getattr(self.bridge.archive, "raw_tier", None)
            if tier is not None:
                tier.promotion_cancel_installed(prepared)
        self._lift_triggers()

    def _lift_triggers(self) -> None:
        target = self.target
        for caches in (getattr(target, "caches", None) or {}).values():
            for cache in caches:
                pending = getattr(cache, "restore_pending", None)
                if isinstance(pending, _PromotionTrigger) and pending.run is self:
                    cache.restore_pending = pending.previous

    def before_device_sync(self, where: str) -> None:
        """W1.4.1 (N1): the stage is about to synchronize the whole device (``where``: its call site). A replay-
        promotion copy still running would be absorbed there, outside every span and stall: it is waited for
        explicitly first (host), timed -- an exposed stall of source ``promotion_wait`` (counted once in
        ``kv_rebuild_exposed``; a ``promotion_wait`` span for information) and ``restore.promotion.
        exposed_wait_seconds`` -- then the stage's own synchronize finds nothing of it."""
        events = [event for event in getattr(self, "_promotion_done", None) or () if not _event_done(event)]
        if not events:
            return
        telemetry = self.workload_telemetry
        stream = (torch.cuda.current_stream(self.target.device)
                  if self.target is not None and self.target.device.type == "cuda" else None)
        span = None if telemetry is None else telemetry.begin("promotion_wait", stream=stream)
        token = None if telemetry is None else telemetry.stall_begin(stream, ready=False, source="promotion_wait")
        started = time.perf_counter()
        for event in events:
            event.synchronize()
        waited = time.perf_counter() - started
        if telemetry is not None:
            telemetry.stall_end(token)
            telemetry.end(span)
        promotion = (self.restore_report or {}).get("promotion")
        if isinstance(promotion, dict):
            promotion["exposed_wait_seconds"] = promotion.get("exposed_wait_seconds", 0.0) + waited
            promotion.setdefault("exposed_waits", []).append({"where": where, "seconds": waited})

    # -- workload v2 (W1b): the exact recompute arm ------------------------------------------------------------------
    def _switch_record(self) -> dict[str, Any] | None:
        """The head switch this request's rewind made (drain seconds, from / to, ancestor or owned), else None."""
        if self.session.get("action") != "rewind":
            return None
        with self.state.lock:
            return None if self.state.last_switch is None else dict(self.state.last_switch)

    def _restore_recompute(self, done: int, resident: str, nodes: list, chunks: list, plans: Mapping[str, Any],
                           plan_hits: Mapping[str, bool], timing: Mapping[str, Any]) -> int:
        """``restore_mode="recompute"``: when every attended range has a raw-tier copy (GPU, else host; the archive's
        LRU) the RAW install from those copies (bitwise the native cache, mode ``archive``, source ``raw_tier``);
        otherwise the stage replays the path's clean forwards from the stored clean latents, in native order and under
        each block's historical conditioning (mode ``recompute``: :attr:`replay_plan`, :meth:`begin_replay`,
        :meth:`end_replay`). The caches stay reset and guarded until the stage begins the replay, so a stage without it
        fails at its first attention access instead of generating on an empty history. Latents and RNG are restored as
        for any restore. W1.4.1 (N2): on a miss, the deepest ancestor whose visible set the tiers hold is installed
        first (:meth:`_cached_ancestor`) and the stage replays only the steps after it (``replay_plan["start_step"]``;
        ``restore.recompute`` records the split)."""
        request, archive, state, target = self.request, self.bridge.archive, self.state, self.target
        workload = self.workload_telemetry  # W1.1: the recompute arm's synchronous restore window (None: off)
        segments = None if workload is None else workload.segments("restore_sync")
        acquire = getattr(archive, "acquire_tier", None)
        tiered = callable(acquire) and getattr(self.bridge.config, "raw_restore", "prefer") != "off"
        leases = acquire(plans) if tiered else None
        restore_started = time.perf_counter()
        lease, rope_tables, replay, anchor, search = None, None, None, None, None
        if leases is not None:
            for stream in request.streams:
                for cache in target.caches[stream]:
                    cache.install_restored_history(plans[stream].layout, lambda span, k_slots, v_slots: None)
            rope_tables = self.bridge.rope_tables.bind(target.rope_cos_sin, grid=tuple(request.grid),
                                                       device=target.device)
            lease = archive.restore(head_ref=state.head_ref, plans=plans, caches=target.caches,
                                    rope_cos_sin=rope_tables, rotate_key=target.rotate_key, rope_mode="native",
                                    num_layers=request.num_layers, grid=tuple(request.grid), tier_leases=leases)
            try:
                lease.wait()
            finally:
                lease.close()
            if target.device.type == "cuda":
                torch.cuda.current_stream(target.device).synchronize()
        else:
            caches = {stream: target.caches[stream] for stream in request.streams}
            if not overlap_seam(caches):
                raise UnsupportedError("restore_mode='recompute' replays the history in the stage; the caches lack "
                                       "SGLang's restore_pending seam (await_restored_history) that refuses a stage "
                                       "which does not replay")
            for stream, items in caches.items():
                for cache in items:
                    if cache._read_indices() != (0, 0) or getattr(cache, "pinned_len", 0):
                        raise IntegrityError("a recompute replay needs reset causal caches")
            # W1.4.1 (N2): the deepest ancestor whose visible set the raw tiers hold is installed like a raw restore of
            # it; only the rest of the path is replayed (none cached: the full replay)
            anchor, search = self._cached_ancestor(nodes) if tiered else (None, None)
            if anchor is not None:
                index, anchor_plans, anchor_leases, _ = anchor
                for stream in request.streams:
                    for cache in target.caches[stream]:
                        cache.install_restored_history(anchor_plans[stream].layout,
                                                       lambda span, k_slots, v_slots: None)
                rope_tables = self.bridge.rope_tables.bind(target.rope_cos_sin, grid=tuple(request.grid),
                                                           device=target.device)
                lease = archive.restore(head_ref=nodes[index].ref, plans=anchor_plans, caches=target.caches,
                                        rope_cos_sin=rope_tables, rotate_key=target.rotate_key, rope_mode="native",
                                        num_layers=request.num_layers, grid=tuple(request.grid),
                                        tier_leases=anchor_leases)
                try:
                    lease.wait()
                finally:
                    lease.close()
                if target.device.type == "cuda":
                    torch.cuda.current_stream(target.device).synchronize()
            for stream, items in caches.items():
                for cache in items:
                    cache.restore_pending = _ReplayGuard(state.session_id)
            steps = replay_steps(nodes)
            start = 0 if anchor is None else len(replay_steps(nodes[:anchor[0] + 1]))
            replay = {"head": state.head_ref, "chunks": done, "streams": list(request.streams),
                      "next_chunk": list(self._block_token_range(done)), "steps": steps, "start_step": start}
            self._replay_layouts = {stream: plans[stream].layout for stream in request.streams}
            if getattr(self.bridge.config, "raw_tier_promote", "off") == "host":  # wl-w1b7: promoted at end_replay
                self._replay_promotion = {"plans": plans, "nodes": nodes}
        kv_seconds = time.perf_counter() - restore_started
        if segments is not None:  # W1.1: the window closes where kv_seconds is taken; the execution state follows
            segments.close()
        exec_span = None if workload is None else workload.begin("exec_state")
        state_started = time.perf_counter()
        for index, node in enumerate(chunks):
            stored = archive.chunk_latents(node, device="cpu")
            self._install_latents(index, stored, node.num_frames)
            self.latents_cpu[index] = stored
        self._set_generators(archive.execution_state(nodes[-1]))
        with state.lock:
            state.chunk_table = {index: dict(item) for index, item in self.chunks.items()}
            state.timeline = timeline_of(nodes)
        self.resume_blocks = done
        self.replay_plan = replay
        if workload is not None:
            workload.end(exec_span)
        started = timing["started"]
        ran = [] if replay is None else replay["steps"][replay["start_step"]:]  # N2: the steps the stage replays
        installed = plans if replay is None else None if anchor is None else anchor[1]
        self.restore_report = {
            "mode": "archive" if replay is None else "recompute", "restore_mode": "recompute",
            "source": "raw_tier" if replay is None else "replay", "resident": resident, "restored_chunks": done,
            "head": state.head_ref, **self._head_report(), "publish_wait_seconds": timing["wait"],
            "plan_seconds": timing["plan"], "restore_seconds": kv_seconds, "plan_cache_hits": dict(plan_hits),
            "execution_state_seconds": time.perf_counter() - state_started,
            "total_seconds": time.perf_counter() - started, "session_open": list(self.session_open),
            "total_with_session_open_seconds": time.perf_counter() - started + sum(
                item["seconds"] or 0.0 for item in self.session_open),
            "plans": {stream: plan.to_dict() for stream, plan in plans.items()},
            "lease": None if lease is None else {**lease.description, **lease.stats, "closed": lease.closed,
                                                 "open_handles": lease.open_handles},
            "rope_tables": None if rope_tables is None else rope_tables.report(),
            "gc": gc_pause_record(timing["gc_before"], GC_PAUSES.snapshot()),
            "replay": None if replay is None else {
                "steps": len(ran), "chunk_forwards": sum(s["kind"] == "chunk" for s in ran),
                "recache_forwards": sum(s["kind"] == "recache" for s in ran),
                "epochs": [s["epoch"] for s in ran if s["kind"] == "recache"], "begun": False,
                "completed": False},
            # W1.4.1 (N2): chunks of the installed state (the target's ``done``: an install; 0: a full replay), the
            # nodes installed from the raw tiers and the path nodes replayed (chunks, recache range nodes)
            "recompute": {
                "cached_ancestor_depth": done if replay is None else 0 if anchor is None
                else anchor[3],
                "installed_nodes": 0 if installed is None else len({span.node_ref for plan in installed.values()
                                                                    for span in plan.spans}),
                "replayed_nodes": sum(1 if s["kind"] == "chunk" else len(s["nodes"]) for s in ran),
                "anchor": state.head_ref if replay is None else None if anchor is None else nodes[anchor[0]].ref,
                "search": search},
            "recompute_replay_seconds": None, "switch": self._switch_record(),
        }
        return done

    @staticmethod
    def _exact_raw(lease: Any) -> bool:
        """W1.4.3 (review C1): whether every range ``lease`` installed came from a raw tier and is exact (captured on a
        clean history, or replayed on an exact anchor): the installed history is then bitwise a clean replay."""
        per_chunk = ((lease.stats.get("tiers") or {}).get("per_chunk")) or []
        return all(item.get("tier") in ("gpu_raw", "host_raw") and item.get("exact", True) for item in per_chunk)

    def _restore_evicted(self, done: int, resident: str, nodes: list, chunks: list, plans: Mapping[str, Any],
                         plan_hits: Mapping[str, bool], missing: set, available: set,
                         timing: Mapping[str, Any]) -> int:
        """W1.4.3 (E1): an evicting ablation arm's restore whose attended history misses nodes the archive budget
        evicted (and no raw tier holds): the deepest ancestor whose visible set is available -- GPU / host raw,
        prepared, or still archived -- is restored through the archive's normal path (lossy where the archive
        serves it), then the stage replays the rest of the path from the stored clean latents (as the recompute arm;
        ``replay_plan["start_step"]``); no ancestor: a full replay. The replayed nodes are promoted into host raw with
        ``raw_tier_promote="host"`` -- not re-compressed (a limitation, uniform across the ablation arms).

        What a miss equals (coordinator decision, review C1: the intended behaviour of a compressed cache with
        eviction, as raw_offload's selective recompute): the replay is exact GIVEN the installed anchor -- the same
        forwards on the same installed snapshot. It is bitwise a clean full replay ONLY when the anchor is exact raw
        (``restore.recompute.anchor_source = "raw"``: every range from a raw tier and exact -- captured on a clean
        history, or replayed on an exact anchor) or there is none (``"none"``: a full replay); with ``"compressed"``
        (some range from the lossy archive, or derived from it: a replay or a capture on a history installed from it)
        it equals a replay from the installed (lossy) snapshot, and its promoted rows are marked inexact, as are the
        session's captures until its next restore (``state.lossy_history``). Records ``restore.recompute``
        (``anchor_source`` included) and ``restore.archive_evicted_hits``."""
        request, archive, state, target = self.request, self.bridge.archive, self.state, self.target
        workload = self.workload_telemetry
        segments = None if workload is None else workload.segments("restore_sync")
        archive.eviction.count_miss(len(missing))
        restore_started = time.perf_counter()
        caches = {stream: target.caches[stream] for stream in request.streams}
        if not overlap_seam(caches):
            raise UnsupportedError("an evicted history is replayed in the stage; the caches lack SGLang's "
                                   "restore_pending seam (await_restored_history)")
        for items in caches.values():
            for cache in items:
                if cache._read_indices() != (0, 0) or getattr(cache, "pinned_len", 0):
                    raise IntegrityError("a replay of an evicted history needs reset causal caches")

        def confirmed(candidate: Mapping[str, Any]) -> Any:  # ``available`` is held (_restore_archive)
            attended = {span.node_ref for plan in candidate.values() for span in plan.spans}
            return True if attended <= available else None

        anchor, search = self._deepest_ancestor(nodes, lambda refs: set(refs) & available, confirmed)
        lease, rope_tables, pressure = None, None, None
        if anchor is not None:
            index, anchor_plans, _, _ = anchor
            # as a normal archive restore (freeze F): the pre-restore relief and the memory-critical section when the
            # restore allocates (a lossy decode does)
            allocates = self._restore_allocates(anchor_plans) or not self._restore_scratch(anchor_plans)["covered"]
            relief = (self._relieve_before_restore(anchor_plans) if allocates else
                      {"estimate": None, "needed_bytes": None, "seconds": 0.0, "skipped": NO_ALLOCATION_SKIP})
            with (self.bridge._restore_section(self) if allocates else contextlib.nullcontext()) as section:
                for stream in request.streams:
                    for cache in target.caches[stream]:
                        cache.install_restored_history(anchor_plans[stream].layout,
                                                       lambda span, k_slots, v_slots: None)
                rope_tables = self.bridge.rope_tables.bind(target.rope_cos_sin, grid=tuple(request.grid),
                                                           device=target.device)
                lease = archive.restore(head_ref=nodes[index].ref, plans=anchor_plans, caches=target.caches,
                                        rope_cos_sin=rope_tables, rotate_key=target.rotate_key,
                                        rope_mode=self.bridge.config.rope_mode, num_layers=request.num_layers,
                                        grid=tuple(request.grid))
                try:
                    lease.wait()
                finally:
                    lease.close()
                if target.device.type == "cuda":
                    torch.cuda.current_stream(target.device).synchronize()
            pressure = {**relief, "section": None if section is None else dict(section)}
        self._release_eviction_hold()  # the anchor is installed; the replay reads stored latents only
        anchor_source = "none"  # review C1: what the miss equals (docstring)
        if lease is not None:
            anchor_source = "raw" if self._exact_raw(lease) else "compressed"
        state.lossy_history = anchor_source == "compressed"  # what the session captures next (review C1)
        for items in caches.values():
            for cache in items:
                cache.restore_pending = _ReplayGuard(state.session_id)
        steps = replay_steps(nodes)
        start = 0 if anchor is None else len(replay_steps(nodes[:anchor[0] + 1]))
        replay = {"head": state.head_ref, "chunks": done, "streams": list(request.streams),
                  "next_chunk": list(self._block_token_range(done)), "steps": steps, "start_step": start}
        self._replay_layouts = {stream: plans[stream].layout for stream in request.streams}
        ran = steps[start:]
        replayed = {ref for step in ran for ref in ([step["node"]] if step["kind"] == "chunk" else step["nodes"])}
        if getattr(self.bridge.config, "raw_tier_promote", "off") == "host":  # only the replayed nodes
            self._replay_promotion = {"plans": plans, "nodes": nodes, "only": replayed,
                                      "exact": anchor_source != "compressed"}
        kv_seconds = time.perf_counter() - restore_started
        if segments is not None:
            segments.close()
        exec_span = None if workload is None else workload.begin("exec_state")
        state_started = time.perf_counter()
        for position, node in enumerate(chunks):
            stored = archive.chunk_latents(node, device="cpu")
            self._install_latents(position, stored, node.num_frames)
            self.latents_cpu[position] = stored
        self._set_generators(archive.execution_state(nodes[-1]))
        with state.lock:
            state.chunk_table = {position: dict(item) for position, item in self.chunks.items()}
            state.timeline = timeline_of(nodes)
        self.resume_blocks = done
        self.replay_plan = replay
        if workload is not None:
            workload.end(exec_span)
        started = timing["started"]
        self.restore_report = {
            "mode": "recompute", "restore_mode": "archive", "source": "replay", "resident": resident,
            "restored_chunks": done, "head": state.head_ref, **self._head_report(),
            "publish_wait_seconds": timing["wait"], "plan_seconds": timing["plan"], "restore_seconds": kv_seconds,
            "plan_cache_hits": dict(plan_hits), "execution_state_seconds": time.perf_counter() - state_started,
            "total_seconds": time.perf_counter() - started, "session_open": list(self.session_open),
            "total_with_session_open_seconds": time.perf_counter() - started + sum(
                item["seconds"] or 0.0 for item in self.session_open),
            "plans": {stream: plan.to_dict() for stream, plan in plans.items()},
            "lease": None if lease is None else {**lease.description, **lease.stats, "closed": lease.closed,
                                                 "open_handles": lease.open_handles},
            "rope_tables": None if rope_tables is None else rope_tables.report(),
            "gc": gc_pause_record(timing["gc_before"], GC_PAUSES.snapshot()),
            # the anchor restore's relief and memory-critical section (None: a full replay, nothing restored)
            "pressure": pressure,
            "replay": {"steps": len(ran), "chunk_forwards": sum(s["kind"] == "chunk" for s in ran),
                       "recache_forwards": sum(s["kind"] == "recache" for s in ran),
                       "epochs": [s["epoch"] for s in ran if s["kind"] == "recache"], "begun": False,
                       "completed": False},
            # the attended nodes the archive budget had evicted (none of them in a raw tier)
            "archive_evicted_hits": len(missing),
            "recompute": {
                "cached_ancestor_depth": 0 if anchor is None else anchor[3],
                "installed_nodes": 0 if anchor is None else len({span.node_ref for plan in anchor[1].values()
                                                                 for span in plan.spans}),
                "replayed_nodes": sum(1 if s["kind"] == "chunk" else len(s["nodes"]) for s in ran),
                "anchor": None if anchor is None else nodes[anchor[0]].ref, "search": search,
                # raw: bitwise a clean full replay; compressed: a replay from the installed snapshot; none: full replay
                "anchor_source": anchor_source},
            "recompute_replay_seconds": None, "switch": self._switch_record(),
        }
        return done

    def _release_eviction_hold(self) -> None:
        """W1.4.3 (E1): release the restore's availability hold (:meth:`LineageSessionArchive.hold_available`)."""
        hold, self._eviction_hold = getattr(self, "_eviction_hold", None), None
        if hold is not None:
            self.bridge.archive.release_available(hold)

    def _cached_ancestor(self, nodes: list) -> tuple[tuple[int, dict[str, Any], dict[str, Any], int] | None, dict]:
        """W1.4.1 (N2): the deepest strict ancestor ``nodes[i]`` of the restored head -- a chunk node or the last range
        node of a recache epoch (a state generation had; a partial range never is) -- whose visible set -- what a RAW
        restore of it installs for its next block (full history: every chunk; sink / rolling window: the sink and
        window chunks; each range in its version valid there) -- the raw tiers hold entirely: ``((i, plans, tier
        leases, chunks), search record)`` (leases pinned; ``chunks``: the chunks before its next block), or ``(None,
        search record)``. Candidate plans go through the bridge's plan cache (they are the plans a restore of that
        ancestor uses). A node attended by candidate ``i`` is attended by every candidate from its own position up
        to ``i`` (sink and pinned tokens stay, a window holding a token at ``i`` holds it at every shallower
        candidate that has it, newer recache versions only appear deeper), so a candidate missing nodes skips to
        below the shallowest of them; an ancestor evicted between the check and the acquisition falls back to a
        shallower one."""
        archive = self.bridge.archive
        cached_refs = getattr(archive, "tier_cached_refs", None)
        if not callable(cached_refs) or not callable(getattr(archive, "tier_covers", None)):
            return None, {"plans_built": 0, "seconds": 0.0}
        return self._deepest_ancestor(nodes, cached_refs, lambda plans: archive.acquire_tier(plans)
                                      if archive.tier_covers(plans) else None)

    def _deepest_ancestor(self, nodes: list, available: Callable[[Any], set],
                          acquire: Callable[[dict], Any]) -> tuple[tuple[int, dict, Any, int] | None, dict]:
        """The search of :meth:`_cached_ancestor` over an availability predicate (W1.4.3 E1: also an evicting
        archive's): ``available(refs)`` -> the refs a restore can read now; ``acquire(plans)`` -> something not None
        (e.g. pinned tier leases) once a candidate's whole visible set is confirmed."""
        request, target = self.request, self.target
        search = {"plans_built": 0, "seconds": 0.0}
        started = time.perf_counter()
        cached = available(node.ref for node in nodes)
        position = {node.ref: index for index, node in enumerate(nodes)}
        chunks = [0] * len(nodes)  # chunk nodes in nodes[:i + 1]: the block a restore of nodes[i] continues with
        for index, node in enumerate(nodes):
            chunks[index] = (chunks[index - 1] if index else 0) + (node.kind == "chunk")
        index, found = len(nodes) - 2, None
        while index >= 0 and found is None:
            node = nodes[index]
            state = node.kind == "chunk" or (node.kind == "recache" and not partial_recache_node(node))
            if not state or node.ref not in cached:  # its next block always attends its own (newest) range
                index -= 1
                continue
            plans, _ = build_plans(target.caches, request.streams, nodes[:index + 1],
                                   self._block_token_range(chunks[index]), request.tokens_per_frame,
                                   self.bridge.plans)
            search["plans_built"] += 1
            missing = {span.node_ref for plan in plans.values() for span in plan.spans} - cached
            if missing:
                index = min(position[ref] for ref in missing) - 1
                continue
            acquired = acquire(plans)
            if acquired is not None:
                found = (index, plans, acquired, chunks[index])
            index -= 1
        search["seconds"] = time.perf_counter() - started
        return found, search

    def begin_replay(self) -> dict[str, Any]:
        """The stage starts the recompute replay of :attr:`replay_plan` (W1b): the caches' guards are lifted and the
        plan returned (``steps`` in native order; W1.4.1 N2: the first ``start_step`` of them are installed already
        and skipped). The stage runs every other step's clean forward without capture, then calls
        :meth:`end_replay`."""
        plan = getattr(self, "replay_plan", None)
        if plan is None:
            raise StateError("no recompute replay is pending for this request")
        for stream in self.request.streams:
            for cache in self.target.caches[stream]:
                if isinstance(getattr(cache, "restore_pending", None), _ReplayGuard):
                    cache.restore_pending = None
        self._replay_started = time.perf_counter()
        workload = self.workload_telemetry  # W1.1: recompute_replay = the stage's replay forwards, until end_replay
        self._replay_span = None if workload is None else workload.begin("recompute_replay")
        self.restore_report["replay"]["begun"] = True
        return plan

    @opens_after("replay")
    def end_replay(self, *, seconds: float | None = None, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The replay ran: every cache's cursors (and pinned shot sink) must be the ones the RAW install of the same
        node sets (the visibility plan's layout), else ``IntegrityError``. Records ``recompute_replay_seconds`` (the
        stage's synchronized time when given, else host time since :meth:`begin_replay`)."""
        plan = getattr(self, "replay_plan", None)
        if plan is None or not self.restore_report["replay"]["begun"]:
            raise StateError("end_replay without a begun recompute replay")
        if self.workload_telemetry is not None:  # W1.1: the replay's forwards are issued (the stage synchronized)
            self.workload_telemetry.end(getattr(self, "_replay_span", None))
        for stream in self.request.streams:
            layout = self._replay_layouts[stream]
            expected = (layout.global_end_index, layout.local_end_index, layout.pinned_start, layout.pinned_len)
            for layer, cache in enumerate(self.target.caches[stream]):
                global_end, local_end = cache._read_indices()
                actual = (global_end, local_end, cache.pinned_start, cache.pinned_len)
                if actual != expected:
                    raise IntegrityError(f"replayed cache cursors {actual} of stream {stream!r} layer {layer} differ "
                                         f"from the restore plan's {expected}")
        wall = time.perf_counter() - self._replay_started
        self.replay_plan = None
        report = self.restore_report
        report["recompute_replay_seconds"] = float(seconds) if seconds is not None else wall
        report["replay"].update(completed=True, host_seconds=wall, stage=json_details(details))
        report["total_with_replay_seconds"] = report["total_seconds"] + report["recompute_replay_seconds"]
        promotion, self._replay_promotion = getattr(self, "_replay_promotion", None), None
        if promotion is not None:  # wl-w1b7 (raw_tier_promote="host"): the replayed nodes into host raw
            try:  # W1.4.1 (N1): prepared now, issued by the first access after the replay (_PromotionTrigger)
                report["promotion"] = self._prepare_replay_promotion(promotion)
            except Exception as exc:  # noqa: BLE001 - an optimization: the replay stands
                # W1.4.1 (I1): swallowed only once nothing it may have issued still runs (a failed wait fails it)
                self._drop_deferred_promotion()
                for event in getattr(self, "_promotion_done", None) or ():
                    event.synchronize()
                report["promotion"] = {"scope": "replay", "error": f"{type(exc).__name__}: {exc}"}
        with self.state.lock:
            self.state.log("replayed", chunks=plan["chunks"], steps=len(plan["steps"]) - plan.get("start_step", 0),
                           seconds=report["recompute_replay_seconds"])
        return report

    # -- freeze F5: layer-overlapped restore -----------------------------------------------------------------------
    def _start_overlap(self, plans: Mapping[str, Any], *, section: bool = True) -> _Overlap:
        """The run's layer-overlapped restore. An unprepared history runs the plain call inside the archive (joined at
        once; ``overlap.fallback``). ``section=False``: no restore section (every attended range is GPU raw)."""
        request, target, bridge = self.request, self.target, self.bridge
        overlap = _Overlap(mode="first_forward", lease=None,
                           caches={stream: target.caches[stream] for stream in request.streams},
                           device=target.device, started=time.perf_counter(), pressure_events=self.pressure_events)
        setup, clock = overlap.setup, time.perf_counter()

        def lap(name: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            setup[name] = now - clock
            clock = now

        try:
            if section:
                overlap.section = overlap.stack.enter_context(bridge._restore_section(self))
            lap("section_entry")
            for stream in request.streams:
                for cache in target.caches[stream]:
                    cache.install_restored_history(plans[stream].layout, lambda span, k_slots, v_slots: None)
            lap("install_cursors")
            overlap.rope_tables = bridge.rope_tables.bind(target.rope_cos_sin, grid=tuple(request.grid),
                                                          device=target.device)
            overlap.pool_before = decoder_pool_snapshot(getattr(bridge.archive, "pool", None))
            lap("rope_tables_and_pool_snapshot")
            overlap.started = time.perf_counter()
            overlap.lease = bridge.archive.restore(
                head_ref=self.state.head_ref, plans=plans, caches=target.caches, rope_cos_sin=overlap.rope_tables,
                rotate_key=target.rotate_key, rope_mode=bridge.config.rope_mode, num_layers=request.num_layers,
                grid=tuple(request.grid), overlapped=True,
                overlap_options={"require_prepared": bridge.overlap_require_prepared})
            lap("archive_restore_start")
        except BaseException:
            if not overlap.stack.__exit__(*sys.exc_info()):  # the section sees the error (OOM relief)
                raise
        if self.dispatched_at is not None:
            overlap.dispatch_to_start_seconds = overlap.started - self.dispatched_at
        if overlap.lease.readiness:
            overlap.arm()
        else:
            overlap.join()
            if target.device.type == "cuda":
                torch.cuda.current_stream(target.device).synchronize()
        return overlap

    def _join_overlap(self, *, raise_failure: bool = True, abort: str | None = None) -> BaseException | None:
        """Join the request's layer-overlapped restore (before anything derived from the restored KV leaves the
        request) and complete its records; a failure is raised (typed) or, with ``raise_failure=False``, returned.
        ``abort`` (a failed request): stop it without waiting for it (O3 ``quiesce``)."""
        overlap, self.overlap = self.overlap, None
        if overlap is None:
            return None
        failure = None
        if abort is not None:
            overlap.abort(abort)
        else:
            workload = self.workload_telemetry  # the join is a readiness wait too (an event pair around it)
            stall = None if workload is None else workload.stall_begin(ready=all(
                readiness.done for _, readiness in getattr(overlap.lease, "readiness", None) or ()), source="join")
            try:
                overlap.join()
            except BaseException as exc:  # noqa: BLE001 - raised below or returned
                failure = exc
            if workload is not None:
                workload.stall_end(stall)
        report = self.restore_report
        if report is not None:
            lease = overlap.lease
            report["lease"] = {**lease.description, **lease.stats, "closed": lease.closed,
                               "open_handles": lease.open_handles}
            report["decoder_pool"] = decoder_pool_record(overlap.pool_before,
                                                         decoder_pool_snapshot(getattr(self.bridge.archive, "pool",
                                                                                       None)))
            if isinstance(report.get("pressure"), dict):
                report["pressure"]["section"] = None if overlap.section is None else dict(overlap.section)
            report["overlap"] = overlap.record()
        if failure is not None and raise_failure:
            raise failure
        return failure

    def _restore_allocates(self, plans: Mapping[str, Any]) -> bool:
        """Whether the archive restore of ``plans`` allocates device memory beyond the request's caches: the archive
        answers (``restore_allocates``: False when every attended range is a GPU raw copy read in place, Step 5c's O4
        fast path); without the query, or when it fails, a restore allocates (relief and section as in F-F4)."""
        query = getattr(self.bridge.archive, "restore_allocates", None)
        if not callable(query):
            return True
        try:
            return bool(query(plans))
        except Exception:  # noqa: BLE001 - keep the relief and the section
            return True

    def _restore_scratch(self, plans: Mapping[str, Any]) -> dict[str, Any]:
        """Cut 4's scratch rule for a restore whose sources need no allocation (coordinator 19:4x): the in-place
        install still takes each engine RoPE output from the caching allocator, bounded and reused in stream order
        (:func:`rope_scratch_bytes`). It counts as non-allocating when the device's headroom covers the bound now
        (:func:`scratch_headroom`; the synchronous install's 2 x one layer's K, which also covers the overlapped
        install's one span and a fallback to the plain call). ``covered``: whether the restore may skip the relief and
        the section; a failing check keeps them."""
        request = self.request
        scratch = rope_scratch_bytes(plans, num_heads=request.num_heads, head_dim=request.head_dim,
                                     key_dtype=self.bridge.capture_dtypes["K"])
        try:
            check = scratch_headroom(self.target.device, scratch["bound"])
        except Exception as exc:  # noqa: BLE001 - keep the relief and the section
            check = {"checked": False, "covered": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"span_bytes": scratch["span"], "bytes": scratch["bound"], **check}

    def _relieve_before_restore(self, plans: Mapping[str, Any]) -> dict[str, Any]:
        """Freeze F pre-restore relief: ``archive.relieve_pressure("restore", needed_bytes=...)`` with the restore's
        working set (:func:`restore_working_set`) plus the KV the request still allocates (``needed_bytes`` of
        :func:`request_kv_geometry`: its first new chunk's capture; the native caches are on the device already).
        Best effort; recorded in the run's pressure events and returned for the restore report."""
        relieve = getattr(self.bridge.archive, "relieve_pressure", None)
        if not callable(relieve):
            return {"estimate": None, "needed_bytes": None, "seconds": 0.0}
        clock = time.perf_counter()
        config = self.bridge.config
        estimate, needed = None, None
        try:
            estimate = restore_working_set(self.request, self.bridge.capture_dtypes, plans,
                                           arena_bytes=getattr(config, "restore_arena_bytes", None),
                                           basis_bytes=getattr(config, "basis_budget_bytes", None))
            needed = estimate["bytes"] + int((self.kv or {}).get("needed_bytes") or 0)
            event = relieve("restore", needed_bytes=needed)
            if event is not None:
                self.pressure_events.append(("restore", {**event, "check_seconds": time.perf_counter() - clock}))
        except Exception as exc:  # noqa: BLE001 - best effort: the restore proceeds
            self.pressure_events.append(("restore", {"error": f"{type(exc).__name__}: {exc}",
                                                     "check_seconds": time.perf_counter() - clock}))
        return {"estimate": estimate, "needed_bytes": needed, "seconds": time.perf_counter() - clock}

    # -- capture -----------------------------------------------------------------------------------------------
    def stream_of(self, kv_cache: Any) -> str:
        for stream, caches in self.target.caches.items():
            if caches is kv_cache:
                return stream
        raise IntegrityError("the clean forward writes a cache that does not belong to this session request")

    def capture_layer(self, capture: Any, layer_index: int, key: torch.Tensor, value: torch.Tensor,
                      freqs_cis: Any) -> None:
        builder = self.builder
        kind = getattr(capture, "kind", "clean_commit")
        if builder is None:
            request = self.request
            builder = self.builder = CleanChunkBuilder(
                capture.chunk, streams=tuple(request.streams), num_layers=request.num_layers,
                num_heads=request.num_heads, head_dim=request.head_dim, dtype=self.bridge.capture_dtypes,
                rope=self.target.rope_cos_sin if self.bridge.config.verify_capture_rope else None, kind=kind,
            )
        elif builder.chunk != capture.chunk or builder.kind != kind:
            raise IntegrityError("a new clean forward was captured before the previous one was committed")
        builder.add(capture.stream, layer_index, key, value, freqs_cis)

    def begin_clean_forward(self, capture: Any) -> None:
        """SGLang seam (optional): a clean forward starts (timed; restore-vs-recompute accounting)."""
        self.forward_timer.begin(capture)
        if self.workload_telemetry is not None:  # context_forward / recache span
            self.workload_telemetry.forward_begin(capture)

    def end_clean_forward(self, capture: Any) -> None:
        if self.workload_telemetry is not None:  # ends before the join: its wait is a stall, not forward time
            self.workload_telemetry.forward_end(capture)
        # Freeze F5: a layer-overlapped restore is joined once the first clean forward ran, before its capture is
        # committed (``end_chunk`` / ``end_recache``). A recache forward that is the first to read the restored history
        # (a resume at a LongLive 1.0 prompt switch) therefore still overlaps the restore. Its per-layer waits release
        # the last layer only after the restore's integrity gate.
        self._join_overlap()
        self.forward_timer.end(capture)
        if self.builder is None or self.builder.chunk != capture.chunk:
            raise IntegrityError("clean forward ended without captured layers")
        self.builder.end_stream(capture.stream)

    def _ready_event(self) -> Any:
        if self.target is None or self.target.device.type != "cuda":
            return None
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.target.device))
        return event

    def _released(self, event: Any) -> None:
        self.released_sources += 1

    def _submit(self, kind: str, *, chunk: int | None = None, timeline: Mapping[str, Any] | None = None,
                **arguments) -> dict[str, Any]:
        """Hand one complete clean commit to the archive (returns once it is queued: generated frontier)."""
        archive, state = self.bridge.archive, self.state
        method = archive.commit_chunk if kind == "chunk" else archive.commit_recache
        if getattr(self, "replay_plan", None) is not None:  # W1b: a begun replay must end (cursor check) first
            raise IntegrityError("the stage committed before completing the recompute replay (run.end_replay)")
        if self.bridge.controller.tree_mode:  # W1b: the chunk's generation index; its ownership ref publishes with it
            self.bridge.controller.prepare_commit(state, kind=kind, node_id=arguments["node_id"], chunk=chunk,
                                                  request=self.request, resume_blocks=self.resume_blocks,
                                                  block_prompts=getattr(self, "block_prompts", None),
                                                  pause_after=self.pause_after)  # W1.2: the run's pause point
        waits, waited = archive.backpressure()
        started = time.perf_counter()
        if getattr(archive, "eviction", None) is not None and state.lossy_history:  # W1.4.3 (review C1)
            arguments["lossy_history"] = True  # the capture's raw entry is inexact (no clean replay)
        ticket = method(state.session_id, parent_ref=state.head_ref, root_ref=state.root_ref,
                        streams=tuple(self.request.streams), ready_event=self._ready_event(),
                        release=self._released, **arguments)
        seconds = time.perf_counter() - started
        after_waits, after_waited = archive.backpressure()
        entry = {"kind": kind, "ticket": ticket, "node_id": arguments["node_id"], "submit_seconds": seconds,
                 "backpressure_waits": after_waits - waits, "backpressure_seconds": after_waited - waited}
        if chunk is not None:
            entry["chunk"] = chunk
        # RAW tickets are already published: tracking moves the head (the next commit's parent) at once.
        self.bridge.controller.track(state, node_id=arguments["node_id"], kind=kind, ticket=ticket, chunk=chunk,
                                     request_id=self.request.request_id,
                                     telemetry=split_record(arguments["record"])[1], timeline=timeline)
        with state.lock:
            state.backpressure["waits"] += entry["backpressure_waits"]
            state.backpressure["seconds"] += entry["backpressure_seconds"]
            entry["queue_depth"] = len(state.pending)
        self.tickets.append(entry)
        return entry

    # -- chunk boundary -------------------------------------------------------------------------------------------
    def end_chunk(self, chunk: Any, latents: torch.Tensor) -> bool:
        self._join_overlap()
        capturing = None if self.workload_telemetry is None else self.workload_telemetry.chunk_boundary(chunk)
        builder, request, state, bridge = self.builder, self.request, self.state, self.bridge
        if builder is None or builder.chunk != chunk or builder.kind != "clean_commit" or not builder.complete:
            raise IntegrityError(f"chunk {chunk.index} ended without a complete clean capture of {request.streams}")
        if (chunk.token_start, chunk.num_tokens) != self._block_token_range(chunk.index):
            raise IntegrityError(f"chunk {chunk.index} does not match the request's block layout")
        streams = tuple(request.streams)
        matrices = stream_matrices(builder, streams)
        contexts = matrix_contexts(bridge.model, token_start=chunk.token_start, frames=chunk.num_frames,
                                   grid=tuple(request.grid), streams=streams)
        cache_state = {stream: _cache_cursors(self.target.caches[stream][0]) for stream in streams}
        record = {
            "chunk_index": chunk.index, "start_frame": chunk.start_frame, "num_frames": chunk.num_frames,
            "token_start": chunk.token_start, "num_tokens": chunk.num_tokens,
            "rope_start_frame": float(chunk.rope_start_frame), "scene_cut": bool(chunk.scene_cut),
            "shot_index": int(chunk.shot_index), "streams": list(streams), "request_id": request.request_id,
            "grid": list(request.grid), "tokens_per_frame": request.tokens_per_frame,
            "rope_check": builder.rope_check, "captured_bytes": builder.source_bytes(),
            "mutation_check": builder.mutation_check, "key_space": "normalized_pre_rope",
            "captured_dtype": request.dtype if bridge.profile.capture_dtypes is None
            else dict(bridge.profile.capture_dtypes), "cache_state": cache_state,
            "representation": bridge.config.representation,
        }
        if self._block(chunk.index) is not None:
            record["block_conditioning"] = self._block(chunk.index)
        # The request latents after the stage's write-back are the authoritative clean latents of the chunk.
        written = self.target.latents[:, :, chunk.start_frame:chunk.start_frame + chunk.num_frames]
        generators = generator_state(_generators(self.target.generator))
        execution_state = {LATENTS_KEY: written.detach().to("cpu", copy=True), **generators}
        boundary = time.perf_counter() - self.started
        entry = self._submit("chunk", chunk=chunk.index, node_id=chunk_id(chunk.index), matrices=matrices,
                             contexts=contexts, record=record, execution_state=execution_state,
                             timeline={"kind": "chunk", "block": chunk.index, "epoch": None,
                                       "conditioning": self._block(chunk.index)})
        self.chunk_marks.append({"chunk": chunk.index, "boundary_seconds": boundary,
                                 "submitted_seconds": time.perf_counter() - self.started})
        builder.release()
        self.builder = None
        meta = {"token_start": chunk.token_start, "num_tokens": chunk.num_tokens,
                "rope_start_frame": float(chunk.rope_start_frame), "num_frames": chunk.num_frames}
        self.chunks[chunk.index] = meta
        self.latents_cpu[chunk.index] = execution_state[LATENTS_KEY]
        self.chunk_digests[chunk.index] = _latents_digest(execution_state[LATENTS_KEY])
        self.generator_state = generators
        with state.lock:
            state.generated_chunks += 1
            state.chunk_table[chunk.index] = dict(meta)
            state.log("submitted", chunk=chunk.index, node_id=chunk_id(chunk.index),
                      queue_depth=entry["queue_depth"], backpressure_seconds=entry["backpressure_seconds"])
        if capturing is not None:
            self.workload_telemetry.end(capturing)
        # In-flight controls (e.g. a pause of this session) are applied here, before the pause decision.
        bridge._at_boundary(self)
        with state.lock:
            pause = state.pause_requested or (self.pause_after is not None
                                              and state.generated_chunks >= self.pause_after)
        remaining = chunk.index + 1 < len(request.block_sizes)
        if pause and remaining:
            self.paused = True
            return True
        return False

    # -- recache (cache epoch) ------------------------------------------------------------------------------------
    def end_recache(self, recache: Any) -> dict[str, Any]:
        """Commit the KV a clean recache forward wrote back (SGLang ``CausalKVRecache``) as a cache epoch.

        Only ``recache.written`` is archived: each written range must be a union of whole archived chunks of
        this session inside the recache window and must not touch the profile's sink frames (the sink keeps
        its original KV, so it is never re-archived). Restore resolves each attended range to the newest
        version on the path, so ancestors and other branches keep their earlier KV.
        """
        self._join_overlap()
        capturing = None if self.workload_telemetry is None else self.workload_telemetry.recache_commit()
        builder, request, bridge = self.builder, self.request, self.bridge
        window = recache.window
        if builder is None or builder.chunk != window or builder.kind != "kv_recache" or not builder.complete:
            raise IntegrityError("recache ended without a complete clean capture of every stream")
        epoch = recache.epoch_id
        if not isinstance(epoch, str) or not epoch or "/" in epoch:
            raise ConfigurationError("a recache needs an epoch id without '/'")
        tpf = request.tokens_per_frame
        sink_end = (request.pos_start_base + bridge.profile.sink_frames) * tpf
        streams = tuple(request.streams)
        matrices = stream_matrices(builder, streams)
        ranges, written_meta = [], []
        for token_start, num_tokens in recache.written:
            token_end = token_start + num_tokens
            if not window.token_start <= token_start < token_end <= window.token_start + window.num_tokens:
                raise IntegrityError(f"written range {(token_start, num_tokens)} is outside the recache window")
            if bridge.profile.sink_frames and token_start < sink_end:
                raise IntegrityError(f"written range {(token_start, num_tokens)} touches the sink "
                                     f"[{request.pos_start_base * tpf}, {sink_end}); the sink is never re-archived")
            covered = sorted((index, item) for index, item in self.chunks.items()
                             if token_start <= item["token_start"] < token_end)
            if not covered or covered[0][1]["token_start"] != token_start or sum(
                    item["num_tokens"] for _, item in covered) != num_tokens or any(
                    item["token_start"] + item["num_tokens"] > token_end for _, item in covered):
                raise IntegrityError(f"written range {(token_start, num_tokens)} is not a union of whole archived "
                                     "chunks of this session")
            for index, item in covered:
                offset = item["token_start"] - window.token_start
                if offset % tpf:
                    raise IntegrityError("recache ranges cover whole latent frames")
                rope_start = float(window.rope_start_frame) + offset // tpf
                range_record = {"chunk_id": chunk_id(index), "chunk_index": index,
                                "token_start": item["token_start"], "num_tokens": item["num_tokens"],
                                "rope_start_frame": rope_start, "frontier": self._frontier()}
                contexts = matrix_contexts(bridge.model, token_start=item["token_start"], frames=item["num_frames"],
                                           grid=tuple(request.grid), streams=streams)
                ranges.append((range_record, slice_matrices(matrices, offset, item["num_tokens"]), contexts))
                written_meta.append({key: range_record[key] for key in ("chunk_index", "token_start", "num_tokens",
                                                                        "rope_start_frame")})
        cache_state = {stream: _cache_cursors(self.target.caches[stream][0]) for stream in streams}
        with self.state.lock:
            precedes = self.state.generated_chunks  # the block generated next, under the recache's conditioning
        record = {"epoch": epoch, "cause": recache.cause, "details": dict(recache.details or {}),
                  "precedes_block": precedes,
                  "window": {"token_start": window.token_start, "num_tokens": window.num_tokens,
                             "rope_start_frame": float(window.rope_start_frame)},
                  "written": [list(item) for item in recache.written], "streams": list(streams),
                  "request_id": request.request_id, "rope_check": builder.rope_check, "cache_state": cache_state,
                  "key_space": "normalized_pre_rope", "representation": bridge.config.representation}
        if self._block(precedes) is not None:
            record["block_conditioning"] = self._block(precedes)
        generators = generator_state(_generators(self.target.generator))
        entry = self._submit("recache", node_id=f"recache-{epoch}", ranges=ranges, record=record,
                             execution_state=generators, epoch_id=epoch,
                             timeline={"kind": "recache", "block": precedes, "epoch": epoch,
                                       "conditioning": self._block(precedes)},
                             details={"cause": recache.cause, **dict(recache.details or {}),
                                      "written": [list(item) for item in recache.written]})
        entry.update(recache=epoch, ranges=written_meta)
        self.generator_state = generators
        builder.release()
        self.builder = None
        with self.state.lock:
            self.state.log("recache_submitted", epoch=epoch, ranges=written_meta)
        if capturing is not None:
            self.workload_telemetry.end(capturing)
        return {"epoch": epoch, "ranges": written_meta}

    def _frontier(self) -> int:
        return max((item["token_start"] + item["num_tokens"] for item in self.chunks.values()), default=0)

    # -- end of request ---------------------------------------------------------------------------------------------
    def _decode_plan(self) -> dict[str, Any]:
        """Which latents SGLang's decoding stage turns into frames (``output`` of the session handle): a request
        that restored history returns only its new chunks by default, decoded after ``decode_context_latents``
        context latents (``exact`` when they reach back to the first frame: the full-clip frames bitwise)."""
        output = self.session.get("output") or "new"
        if output == "full" or self.resume_blocks <= 0 or self.resume_blocks >= len(self._block_starts):
            return {"output": "full"}
        first_new = self._block_starts[self.resume_blocks]
        wanted = self.session.get("decode_context_latents") or self.bridge.config.decode_context_latents
        context = min(int(wanted), first_new)
        return {"output": "new", "start_latent": first_new - context, "context_latents": context,
                "first_new_latent": first_new, "exact": context == first_new}

    def _durable(self) -> bool:
        requested = self.session.get("durable")
        if requested is not None:
            return bool(requested)
        policy = self.bridge.config.durability
        return policy == "always" or (policy == "pause" and self.paused)

    def _resident_claim(self) -> ResidentKV | None:
        """The session's native KV as this request leaves it in the stage caches (None if unsupported)."""
        request, state, target = self.request, self.state, self.target
        if target is None or self.generator_state is None:
            return None
        states = {}
        for stream in request.streams:
            caches = target.caches[stream]
            if not all(callable(getattr(cache, "index_state", None)) for cache in caches):
                return None
            states[stream] = [cache.index_state() for cache in caches]
        with state.lock:
            chunks, revision = state.generated_chunks, state.revision
        if any(index not in self.latents_cpu for index in range(chunks)):
            return None
        return ResidentKV(state.session_id, revision, chunks, tuple(request.streams), states,
                          {index: self.latents_cpu[index] for index in range(chunks)}, dict(self.generator_state),
                          request.request_id, time.time())

    def _commit_entries(self, published_only: bool) -> list[dict[str, Any]]:
        commits, archive = [], self.bridge.archive
        for entry in self.tickets:
            ticket = entry["ticket"]
            item = {key: value for key, value in entry.items() if key != "ticket"}
            if ticket.done() and getattr(ticket, "error", None) is None:
                item["node"] = ticket.result(0)
                item["published"] = True
                if not published_only:
                    try:
                        item.update(archive.commit_report(ticket))
                    except BaseException as exc:  # noqa: BLE001 - diagnostics only
                        item["report_error"] = repr(exc)
            else:
                item["published"] = False
                if ticket.done():
                    item["error"] = repr(ticket.error)
            item["encoder"] = commit_encoder(archive, ticket)
            commits.append(item)
        return commits

    @closes_on_failure
    def finish(self, *, error: BaseException | None) -> dict[str, Any]:
        if self.finished:
            return self.summary
        self.finished = True
        # Freeze F5: a restore still in flight is joined before the request ends; its failure fails
        # a request that would otherwise complete
        overlap_failure = self._join_overlap(raise_failure=False,
                                             abort=None if error is None else f"request failed: {error!r}")
        raise_overlap = error is None and overlap_failure is not None
        if raise_overlap:
            error = overlap_failure
        self._promotion_boundary(error)  # W1.4.1 (C1): while self.target still names the stage caches
        self._release_eviction_hold()  # W1.4.3 (E1): a restore that failed before releasing it
        workload = self.workload_telemetry
        if workload is not None:  # a span a failure left open (e.g. the restore window) ends here
            workload.close_open()
        if self.builder is not None:
            self.builder.release()
            self.builder = None
        state, archive, bridge = self.state, self.bridge.archive, self.bridge
        controller = bridge.controller
        durable = error is None and self._durable()
        wait_seconds = 0.0
        archive_error = None
        if error is None:
            try:
                claim = self._resident_claim()
            except BaseException:  # noqa: BLE001 - resident reuse is an optimization; restore stays exact
                claim = None
            bridge._set_resident(claim)
            if durable:
                draining = None if workload is None else workload.begin("drain")
                try:
                    wait_seconds = controller.drain(state)
                except BaseException as exc:  # noqa: BLE001 - raised below (the failure surfaces at the pause)
                    archive_error = exc
                if workload is not None:
                    workload.end(draining)
            else:
                controller.reconcile(state, surface=False)
        else:
            bridge.drop_resident()  # the caches hold a partly generated chunk
            # A request that failed on the session's archival failure surfaces it now (acknowledged); otherwise
            # the commits it submitted keep publishing behind the failure.
            pending_failure = controller.failure(state)
            if pending_failure is not None:
                archive_error = controller.acknowledge(state, pending_failure)
            else:
                controller.reconcile(state, surface=False)
        unsurfaced = controller.failure(state)
        with state.lock:
            failure = error or archive_error
            if failure is not None:
                state.status = "failed"
                state.log("failed", error=repr(failure), head=state.head_ref)
            else:
                state.status = "paused" if self.paused else "completed"
                state.log(state.status, head=state.head_ref, chunks=state.generated_chunks,
                          pending=len(state.pending), durable=durable)
            state.pause_requested = False
            pending = [item["node_id"] for item in state.pending]
        # Freeze F3: the history keeps plain data for resolved commits (a live ticket only while pending), so a
        # failed commit's exception and traceback frames never outlive the request through the bridge.
        bridge._request_tickets[self.key] = list(self.tickets)
        while len(bridge._request_tickets) > 64:
            bridge._request_tickets.popitem(last=False)
        bridge._settle_request_history()
        backpressure = {"waits": sum(entry["backpressure_waits"] for entry in self.tickets),
                        "seconds": sum(entry["backpressure_seconds"] for entry in self.tickets)}
        plan_prepared = (self._prepare_next_plans() if self.paused and not pending and error is None
                         and archive_error is None and self.target is not None else None)
        self.summary = {
            "session_id": state.session_id, "action": self.session["action"], "status": state.status,
            "paused": self.paused, "head": state.head_ref, "generated_chunks": state.generated_chunks,
            "published": not pending, "pending": pending, "durable": durable,
            "resumed_from_chunks": self.resume_blocks, "restore": self.restore_report,
            "commits": self._commit_entries(published_only=archive.asynchronous and not durable),
            "chunk_marks": self.chunk_marks,
            "archive": {"backend": archive.name, "asynchronous": archive.asynchronous, "durable": durable,
                        "publish_wait_seconds": wait_seconds, "drain_seconds": wait_seconds,
                        "released_sources": self.released_sources, "backpressure": backpressure,
                        "frontiers": archive.frontiers(state.session_id) if archive.asynchronous else None,
                        "unsurfaced_failure": None if unsurfaced is None else repr(unsurfaced)},
            "error": None if error is None else repr(error),
            "archive_error": None if archive_error is None else repr(archive_error),
            "seconds": time.perf_counter() - self.started,
            "timing": _request_timing(self.dispatched_at, self.started, self.chunk_marks),
            "decode": self._decode_plan(),
            "clean_forwards": None,  # resolved lazily by ``bridge.request_summary`` (no host sync here)
            "chunk_digests": {str(index): digest for index, digest in sorted(self.chunk_digests.items())},
            "boundary_hook_seconds": self.hook_seconds,
            # freeze F5: the next resume's restore plan, prepared at this durable pause (cut 3); the bridge's own part
            # of the pre-denoise time (cut 7: session controller and checks before the run starts)
            "plan_prepared": plan_prepared, "begin": dict(self.begin_record),
            # freeze F: request-start check here; the decode seam's checks and OOM reliefs are added when they run
            "pressure": pressure_record(self.kv, self.pressure_events),
            "request_id": self.request.request_id,
            # F3: the key of this request's records (``request_summary`` / ``commit_reports``); internal without an id
            "request_key": self.key, "request_id_missing": self.request.request_id is None,
        }
        if bridge.controller.switch_mode == "detach":  # W1.3: what a rewind's switch did (drain: no field, F7)
            self.summary.update(bridge.controller.switch_fields(self.session.get("action"), state))
        self.bridge.requests.append({key: self.summary[key] for key in
                                     ("session_id", "action", "status", "head", "generated_chunks", "published",
                                      "error", "archive_error")} | {"request_id": self.request.request_id,
                                                                   "request_key": self.key,
                                                                   "restore_mode": (self.restore_report or {}).get(
                                                                       "mode")})
        if workload is not None:  # resolved after the request's outputs (``request_summary`` / ``workload_record``)
            from .workload_telemetry import bytes_per_token

            request = self.request
            workload.set_restore(self.restore_report, representation=bridge.config.representation,
                                 per_token=bytes_per_token(num_layers=request.num_layers, num_heads=request.num_heads,
                                                           head_dim=request.head_dim,
                                                           capture_dtypes=bridge.capture_dtypes))
            self.summary["workload"] = None
        self.bridge._remember(self.key, state.session_id, self.summary, timer=self.forward_timer, workload=workload)
        self.target = None
        self.bridge._release(self)
        if archive_error is not None and error is None:
            raise archive_error  # the acknowledged failure: the head is the last published checkpoint
        if raise_overlap:
            raise overlap_failure
        return self.summary


class _ObserverRun:
    """A plain (session-less) request observed by the bridge (``observe_plain_requests``): its clean forwards and
    chunk boundaries are timed (the in-server recompute baseline of the restore requirement); nothing is captured,
    archived or restored, and the request generates exactly as without the bridge."""

    def __init__(self, bridge: CausalWanKVBridge, request: Any):
        self.bridge, self.request = bridge, request
        self.key = bridge.request_key(request.request_id)
        self.resume_blocks = 0
        self.target = None
        self.restore_report = None
        self.started = time.perf_counter()
        self.dispatched_at = getattr(request, "dispatched_at", None)
        self.forward_timer = _ForwardTimer(bridge.device)
        self.chunk_marks: list[dict[str, Any]] = []
        self.recaches = 0
        self.finished = False
        self.summary: dict[str, Any] | None = None

    def restore(self, target: Any) -> int:
        self.target = target
        self.bridge._await_promotion_copies(target)  # W1.4.1 (C1): it writes the stage caches
        self.bridge.drop_resident()  # this request writes the stage caches
        return 0

    def stream_of(self, kv_cache: Any) -> str:
        for stream, caches in self.target.caches.items():
            if caches is kv_cache:
                return stream
        raise IntegrityError("the clean forward writes a cache that does not belong to this request")

    def capture_layer(self, capture: Any, layer_index: int, key: torch.Tensor, value: torch.Tensor,
                      freqs_cis: Any) -> None:
        return None

    def begin_clean_forward(self, capture: Any) -> None:
        self.forward_timer.begin(capture)

    def end_clean_forward(self, capture: Any) -> None:
        self.forward_timer.end(capture)

    def end_chunk(self, chunk: Any, latents: torch.Tensor) -> bool:
        self.chunk_marks.append({"chunk": chunk.index, "boundary_seconds": time.perf_counter() - self.started})
        return False

    def end_recache(self, recache: Any) -> dict[str, Any]:
        self.recaches += 1
        return {"observed": True, "epoch": getattr(recache, "epoch_id", None)}

    def finish(self, *, error: BaseException | None) -> dict[str, Any]:
        if self.finished:
            return self.summary
        self.finished = True
        self.summary = {
            "observer": True, "request_id": self.request.request_id, "request_key": self.key,
            "request_id_missing": self.request.request_id is None,
            "status": "failed" if error is not None else "completed",
            "error": None if error is None else repr(error), "chunks": len(self.chunk_marks),
            "recaches": self.recaches, "chunk_marks": self.chunk_marks,
            "timing": _request_timing(self.dispatched_at, self.started, self.chunk_marks),
            "clean_forwards": None, "seconds": time.perf_counter() - self.started,
        }
        self.bridge._remember(self.key, None, self.summary, timer=self.forward_timer)
        self.target = None
        self.bridge._release(self)
        return self.summary


__all__ = ["CausalWanKVBridge", "ResidentKV", "chunk_id", "conditioning_digest"]
