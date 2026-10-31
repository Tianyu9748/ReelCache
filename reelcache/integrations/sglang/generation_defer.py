"""Generation-first archival, the bridge side (Step 7 genfirst; config ``defer_archival_during_generation``, default
off).

The user's rule (2026-10-20): focus on generation first, then do the background work, so archival never interferes
with a request's generation. A request's generation -- from the return of its restore (or the end of its recompute
replay) to the end of its VAE decode seam -- is a **generation hold** of the archive pipeline
(:meth:`reelcache.archive_pipeline.ArchivePipeline.open_generation`): no new GPU archival work (fit, projection,
verification, reconstruction checks) starts inside it; a node already on the GPU finishes first, and the wait for it is
the request's ``archival_defer_wait`` (an exposed stall of that source, counted once, with workload telemetry; no span:
the CONTRACT's closed span list is unchanged); fitting resumes when the hold closes. With ``archival_defer_inflight=
"overlap"`` (genfirst3) the hold opens without that wait: the node's remainder runs beside the request, recorded as
``inflight_overlaps`` / ``inflight_overlap_seconds`` in the request's ``archival_defer`` (no stall, no span). The
hold spans [restore return, ``vae_decode`` exit] ⊇ [t2, vae_end] (coordinator 2026-10-20 01:36): the post-decode tail
from vae_end to t3 (SGLang's post-decoding and the uint8 host copy; the genfirst GPU A/B measured CausVid 0.375 s and
LongLive 2.0 0.97 s, the same with and without holds) is not held. The hold also covers LongLive 1.0's first-block
prompt-switch recache, request-critical GPU work that waits on no archival (coordinator 2026-10-20 00:25). The restore
itself is never held: its publish waits and drains need archival. The starvation guard (backlog, age, a waiting
caller) and the pipeline-wide telemetry are the pipeline's (``stats()["generation_defer"]``); each request's summary
carries ``archival_defer``.

Off (the default) none of this acts: the decorators below return the methods' own results unchanged, no hold or record
exists and every record is W1.4.1's. Only a bridge whose archive runs an ArchivePipeline holds anything (raw_offload
and the recompute arm have no fitter: a no-op).

Hooks (one-line decorators in :mod:`.bridge`): :func:`opens_after` on ``_BridgeRun.restore`` (a pending recompute
replay opens at its end instead) and ``end_replay``; :func:`closes_on_failure` on ``_BridgeRun.finish`` (it also
attaches the record); :func:`holds_decode_seam` on ``memory_critical_section`` (the decode seam closes the request's
hold at its exit); :func:`closes_open_holds` on ``begin_request`` and ``close`` (fallbacks: a request without a decode
seam, a failure). A worker's first request keeps its hold past ``finish`` until its decode seam, so its own chunk's
fit cannot start between its finish and its decode and make the seam wait for it (genfirst3; genfirst2 paid one fit
there, CausVid 0.63 s, LongLive 2.0 2.58 s). When no seam comes before the next request (or ``close``), the worker has
none (a CPU harness): that first hold closes there, and every later hold closes at its request's ``finish``.
"""

from __future__ import annotations

import functools
import threading
import time
from typing import Any, Callable

#: The decode seams that end a request's generation (as ``lineage_archive.policy_defers``: the VAE decode).
def _decode_reason(reason: Any) -> bool:
    return isinstance(reason, str) and (reason == "decode" or reason.endswith("_decode"))


_LOCK = threading.Lock()


class _BridgeHolds:
    """A bridge's open holds (request key -> hold) and per-request records (attached to the summaries)."""

    def __init__(self):
        self.open: dict[Any, dict[str, Any]] = {}
        self.records: dict[Any, dict[str, Any]] = {}
        self.seen_decode = False
        self.probed = False  # the worker's first request has finished (genfirst3: held past finish, the seam probe)
        self.decode_depth: dict[Any, int] = {}  # open decode seams per request key (genfirst3, review #7: nesting)


class _Reserved:
    """A key whose hold is being opened (its opener may wait for the node on the GPU): a concurrent opener of the key
    joins it -- returns once that open completed, so it never starts ahead of the wait (genfirst4, review I2); a close
    meanwhile cancels it, and the opener then closes the hold it gets (genfirst3, review #7)."""

    __slots__ = ("closed_by", "done")

    def __init__(self):
        self.closed_by: str | None = None
        self.done = threading.Event()


def enabled(bridge: Any) -> bool:
    return bool(getattr(getattr(bridge, "config", None), "defer_archival_during_generation", False))


def _state(bridge: Any) -> _BridgeHolds:
    state = bridge.__dict__.get("_genfirst")
    if state is None:
        with _LOCK:
            state = bridge.__dict__.get("_genfirst")
            if state is None:
                state = bridge.__dict__["_genfirst"] = _BridgeHolds()
    return state


def _pipeline(bridge: Any):
    """The bridge's ArchivePipeline with its generation holds enabled, or None (option off, no pipeline)."""
    if not enabled(bridge):
        return None
    pipeline = getattr(getattr(bridge, "archive", None), "pipeline", None)
    enable = getattr(pipeline, "enable_generation_holds", None)
    if enable is None:
        return None
    config = bridge.config
    enable(backlog_fraction=config.archival_defer_backlog_fraction,
           max_age_seconds=config.archival_defer_max_age_seconds,
           inflight=getattr(config, "archival_defer_inflight", "wait"))
    return pipeline


def _record() -> dict[str, Any]:
    return {"holds": 0, "inflight_waits": 0, "wait_seconds": 0.0, "held_seconds": 0.0,
            "inflight_overlaps": 0, "inflight_overlap_seconds": 0.0,
            "overrides": {"backlog": 0, "age": 0, "waiter": 0}, "opened_at": [], "closed_by": None}


def open_hold(bridge: Any, key: Any, where: str, telemetry: Any = None) -> None:
    """Open the generation hold of request ``key`` (idempotent while open or being opened: the key is reserved
    before the wait). The wait for a node on the GPU is the request's ``archival_defer_wait``: recorded only when
    there was such a node."""
    pipeline = _pipeline(bridge)
    if pipeline is None:
        return
    state = _state(bridge)
    reserved = _Reserved()
    with _LOCK:
        current = state.open.get(key)
        if current is None:
            state.open[key] = reserved
    if current is not None:
        if isinstance(current, _Reserved):
            current.done.wait()  # join the open in progress (its wait for the node on the GPU)
        return
    # the stall only (CONTRACT §3 closed span list unchanged): its token is appended at stall_end, so it is dropped
    # unless a node was on the GPU
    stall = None if telemetry is None else telemetry.stall_begin(None, ready=False, source="archival_defer_wait")
    try:
        opened = pipeline.open_generation(where)
    except BaseException:
        with _LOCK:
            if state.open.get(key) is reserved:
                del state.open[key]
        reserved.done.set()
        raise
    with _LOCK:
        mine = state.open.get(key) is reserved
        if opened is None or not mine:
            if mine:
                del state.open[key]
            record = None
        else:
            record = state.records.setdefault(key, _record())
            record["holds"] += 1
            record["opened_at"].append(where)
            if opened["waited"]:  # "wait": it waited for the node on the GPU (archival_defer_wait)
                record["inflight_waits"] += 1
                record["wait_seconds"] += opened["wait_seconds"]
            state.open[key] = {"pipeline": pipeline, "token": opened["token"], "started": time.perf_counter(),
                               "record": record}
    reserved.done.set()  # joiners of the key proceed: the open is complete (or cancelled)
    if opened is not None and not mine:  # closed while it was being opened: give the hold back at once
        pipeline.close_generation(opened["token"])
        return
    if opened is not None and opened["waited"] and telemetry is not None:  # a real wait: an exposed stall
        telemetry.stall_end(stall)


def close_hold(bridge: Any, key: Any, closed_by: str) -> None:
    """Close the generation hold of request ``key`` (if open; one being opened is cancelled and its opener closes
    it): fitting resumes; the overrides the guard granted while this hold was open join the request's record."""
    state = bridge.__dict__.get("_genfirst")
    if state is None:
        return
    with _LOCK:
        hold = state.open.pop(key, None)
        if isinstance(hold, _Reserved):
            hold.closed_by = closed_by
            return
    if hold is None:
        return
    report: dict[str, Any] = {}
    hold["pipeline"].close_generation(hold["token"], report)
    record = hold["record"]
    with _LOCK:
        record["held_seconds"] += time.perf_counter() - hold["started"]
        for reason, count in (report.get("overrides") or {}).items():
            record["overrides"][reason] = record["overrides"].get(reason, 0) + count
        if report.get("inflight_overlap"):  # "overlap": the node on the GPU at open ran beside the request
            record["inflight_overlaps"] += 1
            record["inflight_overlap_seconds"] += report["inflight_overlap_seconds"]
        record["closed_by"] = closed_by


def close_all(bridge: Any, closed_by: str) -> None:
    state = bridge.__dict__.get("_genfirst")
    if state is None:
        return
    with _LOCK:
        keys = list(state.open)
    for key in keys:
        close_hold(bridge, key, closed_by)


# -- the decorators (one line each in bridge.py) ----------------------------------------------------------------------
def opens_after(where: str) -> Callable:
    """``_BridgeRun.restore`` / ``end_replay``: once the method returned, the request's generation begins (a restore
    with a recompute replay still pending opens at the replay's end instead)."""
    def decorate(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(run, *args, **kwargs):
            result = method(run, *args, **kwargs)
            if enabled(run.bridge) and getattr(run, "replay_plan", None) is None:
                open_hold(run.bridge, run.key, where, getattr(run, "workload_telemetry", None))
            return result
        return wrapper
    return decorate


def closes_on_failure(method: Callable) -> Callable:
    """``_BridgeRun.finish``: a failed request's hold closes; so does every hold of a worker without decode seams (its
    first request finished and no seam followed before the next one). The worker's first request keeps its hold: its
    decode seam (or the next request, or ``close``) ends it. The request's ``archival_defer`` record joins its summary
    (updated until its decode seam ends)."""
    @functools.wraps(method)
    def wrapper(run, *, error):
        bridge = run.bridge
        try:
            summary = method(run, error=error)
        except BaseException:
            if enabled(bridge):  # genfirst3 (review #5): a raising finish still closes the hold and files its record
                close_hold(bridge, run.key, "failure")
                with _LOCK:
                    state = bridge.__dict__.get("_genfirst")
                    record = None if state is None else state.records.pop(run.key, None)
                # the run's own summary first (genfirst4, review M4); the remembered one only as a fallback
                summary = getattr(run, "summary", None)
                if not isinstance(summary, dict):
                    summary = (getattr(bridge, "_summaries", {}).get(run.key) or {}).get("summary")
                if record is not None and isinstance(summary, dict):
                    summary.setdefault("archival_defer", record)
            raise
        if not enabled(bridge):
            return summary
        state = _state(bridge)
        if error is not None:
            close_hold(bridge, run.key, "failure")
        elif not state.seen_decode:
            if state.probed:  # no seam followed the worker's first request: a worker without decode seams
                close_hold(bridge, run.key, "finish")
            state.probed = True  # else the worker's first request: held until its seam (or the next request)
        with _LOCK:  # the same record object: its decode seam keeps updating it
            record = state.records.get(run.key)
            if record is None:
                record = state.records[run.key] = _record()
        if isinstance(summary, dict):
            summary["archival_defer"] = record
        return summary
    return wrapper


def holds_decode_seam(method: Callable) -> Callable:
    """``memory_critical_section``: the VAE decode seam is part of its request's generation -- held (opening a hold
    when none is open, e.g. a plain request) -- and its exit closes the request's hold (at vae_end; t3 follows after
    the unheld post-decode tail)."""
    @functools.wraps(method)
    def wrapper(bridge, reason, *, request_id=None):
        inner = method(bridge, reason, request_id=request_id)
        if not enabled(bridge) or not _decode_reason(reason) or _pipeline(bridge) is None:
            return inner
        key = request_id if request_id is not None else getattr(bridge, "_awaiting_decode", None)
        return _DecodeHold(bridge, key, inner)
    return wrapper


class _DecodeHold:
    """The decode seam inside its request's generation hold (a context manager around the seam's own)."""

    def __init__(self, bridge: Any, key: Any, inner: Any):
        self.bridge, self.key, self.inner = bridge, key, inner

    def __enter__(self):
        state = _state(self.bridge)
        state.seen_decode = True
        with _LOCK:  # nested seams of one request: the outermost one owns the hold's end (genfirst3, review #7)
            depth = state.decode_depth[self.key] = state.decode_depth.get(self.key, 0) + 1
        if depth == 1:
            entry = getattr(self.bridge, "_summaries", {}).get(self.key) or {}
            open_hold(self.bridge, self.key, "decode", entry.get("workload"))
        try:
            return self.inner.__enter__()
        except BaseException:
            self._leave()
            raise

    def __exit__(self, *exc):
        try:
            return self.inner.__exit__(*exc)
        finally:
            self._leave()

    def _leave(self) -> None:
        state = _state(self.bridge)
        with _LOCK:
            depth = state.decode_depth.get(self.key, 1) - 1
            if depth:
                state.decode_depth[self.key] = depth
                return
            state.decode_depth.pop(self.key, None)
        close_hold(self.bridge, self.key, "decode")
        with _LOCK:
            record = state.records.pop(self.key, None)  # the request's generation is over
        summary = (getattr(self.bridge, "_summaries", {}).get(self.key) or {}).get("summary")
        if record is not None and isinstance(summary, dict):
            summary.setdefault("archival_defer", record)


def closes_open_holds(where: str) -> Callable:
    """``begin_request`` / ``close``: any hold still open (a request whose decode seam never came) closes once the
    method accepted the call -- a refused call (a request is running) leaves the running request's hold alone
    (genfirst3, review #6). The method runs first: a hold still open meanwhile never blocks it (its drains wait on
    archival: the ``waiter`` rule)."""
    def decorate(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(bridge, *args, **kwargs):
            result = method(bridge, *args, **kwargs)
            state = bridge.__dict__.get("_genfirst")
            if state is not None:  # the new run opens its own hold later (its restore): every open one is stale
                close_all(bridge, where)
                with _LOCK:
                    state.records.clear()  # attached to their summaries already (finish) or never had a request
                    state.decode_depth.clear()
            return result
        return wrapper
    return decorate


__all__ = ["close_all", "close_hold", "closes_on_failure", "closes_open_holds", "enabled", "holds_decode_seam",
           "open_hold", "opens_after"]
