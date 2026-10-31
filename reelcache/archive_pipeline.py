"""Ordered asynchronous archival of complete clean chunks (``archive_native`` execution).

Native active KV keeps serving generation; archival runs behind it with two
frontiers per lineage: the **generated frontier** (the newest submitted clean
chunk) and the **published frontier** (the lineage head in the store). A
submitted chunk is not a checkpoint until it is published.

Stages and ownership (see :mod:`reelcache.lineage_commit`):

* the **fitter thread** (one per pipeline; the owner of an optional
  :class:`~reelcache.gpu_fitting.GPUFitter`) waits on the producer's capture
  event, plans chunk ``n`` from chunk ``n - 1``'s planned bases (c3 tests c2's
  extension before c2 is published) and the registry pinned with the
  lineage's unpublished observations, finalizes it on a side stream, starts
  one D2H canvas copy into bounded host staging and then releases the source
  leases with the event after which nothing reads them;
* **encode workers** (a :class:`~reelcache.execution.BoundedExecutor`) run
  libx265 node encodes in parallel (``encode_workers`` x ``encode_threads``
  bounded by the CPU budget; thread counts never change bytes) and store
  packets/records as staging objects;
* the **publisher** publishes stored chunks strictly in order with a
  compare-and-swap of the lineage head, then trains the registry.

Every queue is bounded: source leases (``max_source_chunks``), host canvas
staging (``max_staged_canvases``), the encode executor and the unpublished
window (``max_unpublished``). A full queue applies explicit backpressure to
:meth:`ArchivePipeline.submit` (block up to ``submit_timeout``, then
``CapacityError``). A failure of chunk ``n`` fails every later chunk of the
lineage (never a success-shaped partial checkpoint), retires their leases and
staging, and is raised by ``submit``/``drain``/``ticket.result`` until
:meth:`ArchivePipeline.reset` acknowledges it. :meth:`drain` waits for the
published frontier (pause/checkpoint); :meth:`cancel` retires all unpublished
work of a lineage; :meth:`close` drains or cancels everything.

Archival yields to restores in flight (Step 7, after Step 5b's stall
diagnosis): with a restore activity (``yield_to``; by default the seal-time
``restorer``'s), the fitter checks it before every node, every routine step of
planning/finalization and its other yield points, and encode workers before
every encode, store and publication; while a restore runs they wait (without
the GIL) at most ``yield_max_seconds`` per busy period. Yielding changes timing
only. Yield points are stage boundaries, plus, with ``pause_ffmpeg``, every
pipe chunk of the pipeline's FFmpeg processes (x265 encodes, validation
decodes): a running encode then stalls on its pipes for the restore (no CPU,
no GIL hand-offs). While a caller waits on archival (a drain, a ticket,
backpressure, a deferral) nobody yields and waiting workers resume within
50 ms, so a restore that waits for its own dependencies is never held by the
pause. ``cpu_priority`` (opt-in) runs those FFmpeg processes at background CPU
priority (``"idle"``: SCHED_IDLE, or a nice increment).

With a raw tier (:class:`reelcache.raw_tier.RawChunkTier`, freeze D) the
pipeline owns no bounded source leases: ``submit`` hands the captured tensors
to the tier (no copy), which keeps a complete device or host copy of every
unpublished chunk, demotes device copies beyond its GPU budget to host, and
re-stages a spilled source on its capture device for the planner, so planning
and finalization never depend on where the raw copy was. ``submit`` waits only
when device and host budgets are full of unpublished chunks or the optional
lag bound is exceeded (last-resort backpressure, recorded with its cause on
the ticket and in the tier's records). Without a raw tier the bounds above
apply unchanged.

A caller's memory-critical section (e.g. a VAE decode next to archival on one
GPU) defers GPU fitting: inside :meth:`ArchivePipeline.defer_gpu_work` no node
(or prewarm) starts planning; a node already on the GPU finishes first, and
the CUDA caching allocator's unoccupied blocks (the fitter's pool) can be
released on entry. Queued nodes keep their order and bounds; encode workers
and ordered publication continue.
"""

from __future__ import annotations

import bisect
import contextlib
import functools
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import torch

from . import _hevc
from . import accel_b as _accel_b
from . import accel_c as _accel_c
from .errors import (CapacityError, ConfigurationError, IntegrityError, PendingError, StateError,
                     UnsupportedError)
from .execution import BoundedExecutor
from .deferred_reads import step_hook, workspace_stats
from .lineage_commit import ArchiveOptions, CommitJob, advance_shape, frame_free, raisable, warm_commit_path
from .rank_method import RankMethodConfig
from .types import entry_parts


#: Fitter re-check (dev22j4): the longest a node's GPU work waits behind an open watched memory-critical section
#: (then it starts anyway: archival never deadlocks behind a section).
RECHECK_DEFER_TIMEOUT = 30.0
#: Recent fitter re-check deferrals kept in ``stats()["fitter_recheck"]["events"]``.
RECHECK_EVENTS = 32
#: Fitter pause (dev22j7, :meth:`ArchivePipeline.pause_gpu_work`): how long the pausing caller waits for the fitter to
#: reach its next step boundary, and the longest the fitter stays paused at one boundary (archival never deadlocks
#: behind a pause).
PAUSE_ACK_TIMEOUT = 0.1
PAUSE_MAX_SECONDS = 30.0
#: dev22j8: the fitter thread measures its own streams' allocator segments (its peak working set, see
#: :meth:`ArchivePipeline.inflight_growth_bound`) at the end of the first ``PEAK_MEASURE_FIRST`` nodes, then of every
#: ``PEAK_MEASURE_EVERY``-th node; and the pause acknowledgement timeouts kept with the fitter's phase.
PEAK_MEASURE_FIRST = 3
PEAK_MEASURE_EVERY = 16
PAUSE_EVENTS = 32
#: dev22j8 (coordinator 13:1x: live, not reserved, bytes): a measured node samples its fitter streams' live bytes at
#: its step boundaries and blocked waits at most once per ``PEAK_SAMPLE_INTERVAL`` seconds; the attach warm-up replays
#: the allocator's alloc/free history (at most ``PEAK_TRACE_ENTRIES`` events).
PEAK_SAMPLE_INTERVAL = 0.05
PEAK_TRACE_ENTRIES = 1 << 20
#: dev22j8v6 (F6 b): the fitter's CUDA stream -- ``"dedicated"`` (:func:`_dedicated_stream`; the SGLang bridge's
#: default) or ``"pooled"`` (``torch.cuda.Stream()``, this class's default and F4-F5.5: a restore's pooled stream can be
#: the fitter's). ``cpu_set``: the CPUs given to the fitter and encoder threads (None: inherited).
FITTER_STREAMS = ("dedicated", "pooled")
#: dev22j8v6 (F5.5 gate, RAVEN d32 4446697): the fitter's unpausable spans -- from a step boundary to the next boundary,
#: blocked wait or idle, when a pause can land -- per phase, counted in these upper bounds (seconds; the last bucket is
#: longer): ``stats()["fitter_pause"]["step_spans"]``. An ack-timeout event also gets ``boundary_after_seconds``, how
#: long its step ran on after the timeout (what a longer acknowledgement window would have had to wait).
SPAN_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)


class _Watch:
    """A memory-critical section the fitter re-checks before it starts GPU work (dev22j4)."""

    __slots__ = ("reason", "recheck", "events")

    def __init__(self, reason: str, recheck: Callable[[int], Mapping[str, Any] | None]):
        self.reason, self.recheck, self.events = reason, recheck, []


class ArchiveFailure(StateError):
    """An archived chunk (or an ancestor it depends on) failed or was cancelled."""

    def __init__(self, message: str, *, branch_id: str, node_id: str, cause: BaseException | None = None):
        super().__init__(message)
        self.branch_id, self.node_id, self.cause = branch_id, node_id, cause


class DetachRolledBack(Exception):
    """W1.3 (R1): an archive's ``detach`` failed before its commit point and rolled back completely (nothing moved:
    the caller may drain instead). Any other exception from ``detach`` carries no such guarantee."""


class ArchiveTicket:
    """Completion of one submitted chunk: its published node reference or its failure."""

    def __init__(self, branch_id: str, node_id: str, sequence: int, token_end: int):
        self.branch_id, self.node_id, self.sequence, self.token_end = branch_id, node_id, sequence, token_end
        # W1.3: the lineage the commit publishes to (``branch_id`` until a detach hands it to a detached branch)
        self.publication_branch_id = branch_id
        self._event = threading.Event()
        self.node_ref: str | None = None
        self.error: BaseException | None = None
        # The pipeline's "a caller waits on archival" bracket (its workers stop yielding to restores meanwhile).
        self._waiting: Callable[[], Any] | None = None
        # Records: how the submit was admitted (raw tier, wait and its cause) and the encoder of the published node.
        self.admission: dict[str, Any] | None = None
        self.encoder: str | None = None

    def done(self) -> bool:
        return self._event.is_set()

    def result(self, timeout: float | None = None) -> str:
        # Only a caller that really waits enters the "a caller waits on archival" bracket (a poll with timeout 0 must
        # not lift a fitter re-check hold, dev22j4).
        waits = self._waiting is not None and not self._event.is_set() and (timeout is None or timeout > 0)
        with (self._waiting() if waits else contextlib.nullcontext()):
            published = self._event.wait(timeout)
        if not published:
            raise PendingError(f"chunk {self.node_id} is not published yet")
        if self.error is not None:
            raise raisable(self.error)  # a copy: the caller's frames never attach to the kept error
        return self.node_ref

    def _resolve(self, node_ref: str | None = None, error: BaseException | None = None) -> None:
        if not self._event.is_set():
            self.node_ref, self.error = node_ref, frame_free(error)
            self._event.set()


@dataclass
class _Entry:
    ticket: ArchiveTicket
    lineage: _Lineage
    kind: str
    node_id: str
    groups: list
    epoch_id: str | None
    details: Mapping[str, Any] | None
    execution_state: Mapping[str, torch.Tensor] | None
    ready_event: Any
    release: Callable[[Any], None] | None
    submitted: float
    job: CommitJob | None = None
    stage: str = "queued"
    busy: bool = False
    # Inside the publisher's compare-and-swap of the lineage head: its outcome, not a concurrent failure or
    # cancellation, decides whether the chunk is published.
    publishing: bool = False
    sources_held: bool = True
    staging: list = field(default_factory=list)
    # (range index, buffer) of this node's staged canvases whose ranges are not encoded yet, oldest first.
    staged_ranges: list = field(default_factory=list)
    next_range: int = 0
    times: dict = field(default_factory=dict)
    retired: bool = False
    # The raw tier's entry of this commit (freeze D), or None (C3 source leases).
    raw: Any = None


@dataclass
class _Lineage:
    branch_id: str
    published_ref: str
    published_frontier: int
    generated_frontier: int
    shape: Any
    entries: deque = field(default_factory=deque)
    last_job: CommitJob | None = None
    error: BaseException | None = None
    cancelled: bool = False
    sequence: int = 0
    failed_objects: list = field(default_factory=list)
    publish_lock: threading.Lock = field(default_factory=threading.Lock)
    published_state: Any = None
    published_ids: set = field(default_factory=set)
    # W1.3 (``tree_switch="detach"``): the branch this lineage was detached from (its pending commits publish in order
    # to this lineage's own ref, never to that branch's head), else None; ``detaching``: a detach is handing this
    # lineage over (the publisher starts no further transaction until it finished or rolled back).
    detached_from: str | None = None
    detaching: bool = False


#: Generation-first archival (Step 7 genfirst): the starvation guard's defaults (config ``archival_defer_*``).
GENERATION_BACKLOG_FRACTION = 0.5
GENERATION_MAX_AGE_SECONDS = 30.0
#: Why the fitter started a node inside a generation hold: the unpublished raw backlog passed its fraction of the raw
#: budget, the oldest unpublished chunk passed the age bound, or a caller waits on archival (never-block).
GENERATION_OVERRIDES = ("backlog", "age", "waiter")
#: What opening a hold does about the node already on the GPU (genfirst3, coordinator 2026-10-20 02:12): ``wait``
#: (W1.4.2) -- it finishes first, the wait is the request's ``archival_defer_wait``; ``overlap`` -- the request starts
#: at once and that node's remainder runs beside it (recorded as overlap seconds, no stall); new fits stay held.
GENERATION_INFLIGHT = ("wait", "overlap")


class _GenerationHolds:
    """Generation holds of an :class:`ArchivePipeline` (all state under the pipeline's condition).

    While a hold is open the fitter starts no node (fit, projection, verification, reconstruction checks), except
    when the starvation guard lets it through (:meth:`override`): the unpublished raw backlog above
    ``backlog_fraction`` of the raw tier's budget (device + host; no rule without a raw tier), the oldest unpublished
    chunk older than ``max_age_seconds``, or a caller waiting on archival (a drain, a ticket, a backpressured
    ``submit``: never-block). Each node started so counts one override by reason, pipeline-wide and for every hold
    open then (genfirst3, review #7). The age is that of the oldest chunk still waiting for the fitter (queued, not
    failed or cancelled; review #8), measured from its ``submit`` call (its capture at the chunk boundary, before
    any admission wait; review #9): the hold delays only fitting. Encode workers, publication, the raw tier's copies
    and an attach-time warm-up are never held."""

    def __init__(self, backlog_fraction: float, max_age_seconds: float, inflight: str = "wait"):
        self.backlog_fraction, self.max_age, self.inflight = backlog_fraction, max_age_seconds, inflight
        # token -> (label, opened at, its overrides, its overlap: [the node in flight at open, that node's end])
        self.tokens: dict[int, tuple[str, float, dict[str, int], list]] = {}
        self._next = 0
        self.held_since: float | None = None
        self.lags: deque = deque(maxlen=4096)
        self.counters = {"holds": 0, "held_seconds": 0.0, "waits": 0, "wait_seconds": 0.0, "inflight_waits": 0,
                         "overrides": {reason: 0 for reason in GENERATION_OVERRIDES}, "backlog_peak_bytes": 0,
                         "inflight_overlaps": 0, "inflight_overlap_seconds": 0.0}

    def open(self, label: str, now: float, overlapping: Any = None) -> int:
        self._next += 1
        self.tokens[self._next] = (label, now, {reason: 0 for reason in GENERATION_OVERRIDES}, [overlapping, None])
        if overlapping is not None:
            self.counters["inflight_overlaps"] += 1
        if self.held_since is None:
            self.held_since = now
        self.counters["holds"] += 1
        return self._next

    def close(self, token: Any, now: float) -> dict[str, Any] | None:
        """The closed token's report (None: not open): its override counts and, in ``overlap`` mode, how long the node
        in flight when it opened ran inside it."""
        info = self.tokens.pop(token, None)
        if info is None:
            return None
        if not self.tokens and self.held_since is not None:
            self.counters["held_seconds"] += now - self.held_since
            self.held_since = None
        label, opened, overrides, (item, ended) = info
        overlap = 0.0 if item is None else min(ended if ended is not None else now, now) - opened
        self.counters["inflight_overlap_seconds"] += overlap
        return {"overrides": overrides, "inflight_overlap_seconds": overlap, "inflight_overlap": item is not None}

    def gpu_done(self, item: Any, now: float) -> None:
        """The fitter left ``item``'s GPU work (finished or handed back): the overlap of holds opened beside it ends."""
        for _, _, _, overlap in self.tokens.values():
            if overlap[0] is item and overlap[1] is None:
                overlap[1] = now

    def waited(self, seconds: float, inflight: bool) -> None:
        self.counters["waits"] += 1
        self.counters["wait_seconds"] += seconds
        self.counters["inflight_waits"] += int(bool(inflight))

    @staticmethod
    def backlog(pipeline: "ArchivePipeline") -> tuple[int, int]:
        """``(unpublished raw bytes, raw budget = device + host)`` of the pipeline's raw tier (caller holds the
        condition the tier shares); ``(0, 0)`` without an enabled raw tier: then no backlog rule applies -- the C3
        bounds' backpressure is a waiting ``submit``, the ``waiter`` rule."""
        tier = pipeline.raw_tier
        if tier is None or not tier.enabled:
            return 0, 0
        return int(tier._unpublished_bytes), int(tier.gpu_budget() + tier.host_budget())

    def observe_backlog(self, pipeline: "ArchivePipeline") -> None:
        nbytes = self.backlog(pipeline)[0]
        if nbytes > self.counters["backlog_peak_bytes"]:
            self.counters["backlog_peak_bytes"] = nbytes

    def oldest(self, pipeline: "ArchivePipeline") -> float | None:
        """Capture time of the oldest chunk waiting for the fitter (queued; a failed or cancelled one, or one of a
        failed lineage, never ages the guard)."""
        captured = [getattr(entry, "captured_at", None) or entry.times.get("submitted") for entry in pipeline._queue
                    if not isinstance(entry, _Warmup) and entry.stage not in ("failed", "cancelled")
                    and not entry.retired and entry.lineage.error is None]
        captured = [value for value in captured if value is not None]
        return min(captured) if captured else None

    def override(self, pipeline: "ArchivePipeline") -> str | None:
        """The starvation guard: why the fitter may start a node inside a hold now, else None."""
        if pipeline._waiters:
            return "waiter"
        nbytes, budget = self.backlog(pipeline)
        if nbytes > self.counters["backlog_peak_bytes"]:
            self.counters["backlog_peak_bytes"] = nbytes
        if budget > 0 and nbytes > self.backlog_fraction * budget:
            return "backlog"
        oldest = self.oldest(pipeline)
        if oldest is not None and time.perf_counter() - oldest > self.max_age:
            return "age"
        return None

    def blocking(self, pipeline: "ArchivePipeline") -> bool:
        """Whether the open holds keep the fitter from starting the queue's head now. A warm-up (attach time) is never
        held: a prewarm caller waits for it outside every archival wait."""
        if not self.tokens or not pipeline._queue or isinstance(pipeline._queue[0], _Warmup):
            return False
        return self.override(pipeline) is None

    def admission(self, pipeline: "ArchivePipeline") -> tuple[bool, str | None]:
        """ONE decision for the queue's head (caller holds the condition, the head still queued: the age rule reads
        the queue): ``(blocked, reason)``. The same reason that lets the head through an open hold is the one counted
        for it -- the waiter rule reads ``_waiters`` once here (genfirst4, review I3). ``(False, None)`` without an open
        hold or for a warm-up."""
        if not self.tokens or not pipeline._queue or isinstance(pipeline._queue[0], _Warmup):
            return False, None
        reason = self.override(pipeline)
        return reason is None, reason

    def started(self, entry: Any, reason: str | None) -> None:
        """The fitter starts GPU work on ``entry`` (a node, not a warm-up, nor a failed or cancelled one): inside an
        open hold the guard let it through for ``reason`` -- one override, pipeline-wide and for each open hold."""
        if reason is not None and self.tokens and not isinstance(entry, _Warmup):
            self.counters["overrides"][reason] += 1
            for _, _, overrides, _ in self.tokens.values():
                overrides[reason] += 1

    def poll(self, pipeline: "ArchivePipeline") -> float | None:
        """The fitter's wait bound while holds are open: until the oldest unpublished chunk reaches the age bound."""
        if not self.tokens:
            return None
        oldest = self.oldest(pipeline)
        if oldest is None:
            return None
        return max(0.01, min(1.0, oldest + self.max_age - time.perf_counter()))

    def summary(self, pipeline: "ArchivePipeline") -> dict[str, Any]:
        lags = sorted(self.lags)

        def quantile(q: float) -> float | None:
            if not lags:
                return None
            return lags[min(len(lags) - 1, int(q * (len(lags) - 1) + 0.5))]
        held = self.counters["held_seconds"] + (time.perf_counter() - self.held_since
                                                if self.held_since is not None else 0.0)
        return {**self.counters, "overrides": dict(self.counters["overrides"]), "held_seconds": held,
                "open": len(self.tokens), "backlog_fraction": self.backlog_fraction, "inflight": self.inflight,
                "max_age_seconds": self.max_age, "publications": len(lags),
                "publication_lag_p50": quantile(0.5), "publication_lag_p90": quantile(0.9),
                "guard": self.guard(pipeline)}

    def guard(self, pipeline: "ArchivePipeline") -> dict[str, Any]:
        """Which bound limits the deferred backlog (genfirst3): the backlog rule needs an enabled raw tier; without
        one (none, or disabled: e.g. a zero host raw budget with svd_hevc) the rule is inert and the C3 bounds plus
        the ``waiter`` rule (a backpressured ``submit`` waits on archival) bound it."""
        tier = pipeline.raw_tier
        nbytes, budget = self.backlog(pipeline)
        if tier is None:
            rule = "inert: no raw tier"
        elif not tier.enabled:
            rule = "inert: raw tier disabled"
        else:
            rule = "raw_tier"
        return {"backlog_rule": rule,
                "raw_tier_disabled_reason": None if tier is None else tier.disabled_reason,
                "raw_budget_bytes": budget, "backlog_threshold_bytes": int(self.backlog_fraction * budget) if budget
                else None, "unpublished_raw_bytes": nbytes, "max_source_chunks": pipeline.max_source_chunks,
                "max_unpublished": pipeline.max_unpublished, "sources_held": pipeline._sources_held}


class _Warmup:
    """A fitter-thread warm-up request (see :meth:`ArchivePipeline.prewarm`)."""

    def __init__(self, contexts: Mapping, max_frames: int | None, seed: int):
        self.contexts, self.max_frames, self.seed = contexts, max_frames, seed
        self.done = threading.Event()
        self.result: dict[str, Any] | None = None
        self.canvas = None
        self.error: BaseException | None = None


class _HostStaging:
    """Bounded pool of (pinned) host canvas buffers."""

    def __init__(self, count: int, pin: bool):
        self.count, self.pin = count, pin
        self._free: list[torch.Tensor] = []
        self._allocated = 0
        self._in_use = 0
        self._cond = threading.Condition()
        self.peak_in_use = self.peak_bytes = 0
        self.wait_seconds = 0.0

    def acquire(self, nbytes: int, timeout: float | None = None) -> torch.Tensor:
        started = time.perf_counter()
        with self._cond:
            while self._in_use >= self.count:
                if not self._cond.wait(timeout):
                    raise CapacityError("host canvas staging stayed exhausted")
            fitting = [index for index, buffer in enumerate(self._free) if buffer.numel() >= nbytes]
            if fitting:
                # Never compare tensors with ==/list.remove: select by position.
                index = min(fitting, key=lambda position: self._free[position].numel())
                buffer = self._free.pop(index)
            else:
                if self._free:
                    self._free.pop()
                buffer = torch.empty(nbytes, dtype=torch.uint8, pin_memory=self.pin)
            self._in_use += 1
            self.peak_in_use = max(self.peak_in_use, self._in_use)
            self._bytes_in_use = getattr(self, "_bytes_in_use", 0) + buffer.numel()
            self.peak_bytes = max(self.peak_bytes, self._bytes_in_use)
            self.wait_seconds += time.perf_counter() - started
            return buffer

    def release(self, buffer: torch.Tensor) -> None:
        with self._cond:
            self._free.append(buffer)
            self._in_use -= 1
            self._bytes_in_use -= buffer.numel()
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._free.clear()
            self._allocated = 0


class _RestoreYield:
    """Archival workers' gate on a :class:`~reelcache.archive_restore.RestoreActivity`.

    A worker waits while a restore runs, at most ``max_seconds`` per busy period (``activity.epoch``): after a
    timeout it does not wait again until the next busy period, so archival always progresses. While ``bypass()`` is
    true -- a caller is blocked on archival progress (a drain, a ticket, backpressure, a deferral), e.g. a restore
    that waits for its own dependencies inside its bracket -- nobody waits, and waiting workers resume within
    :attr:`SLICE` seconds: the pause never holds work that someone is waiting for. :meth:`shared` hooks (an FFmpeg
    pipe pump: its helper threads and its worker) share one expiry, so one worker pauses at most ``max_seconds`` per
    busy period whichever of its threads waits.
    """

    #: Granularity at which a waiting worker notices a released pause (``bypass``).
    SLICE = 0.05

    def __init__(self, activity, max_seconds: float, bypass: Callable[[], bool] | None = None):
        self.activity, self.max_seconds, self.bypass = activity, max_seconds, bypass
        self._local = threading.local()
        self._lock = threading.Lock()
        self.stats_ = {"yields": 0, "yield_seconds": 0.0, "yield_timeouts": 0, "yield_releases": 0}

    def __call__(self) -> None:
        self._wait(self._local)

    def shared(self) -> Callable[[], None]:
        """A hook whose per-busy-period expiry is shared by every thread that calls it."""
        state = _SharedExpiry()
        return lambda: self._wait(state)

    def _released(self) -> bool:
        return self.bypass is not None and self.bypass()

    def _wait(self, state) -> None:
        activity = self.activity
        if not activity.active() or self._released():
            return
        epoch = activity.epoch
        if getattr(state, "expired", None) == epoch:
            return
        started = time.perf_counter()
        deadline = started + self.max_seconds
        idle = released = False
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            if activity.wait_idle(min(self.SLICE, remaining)):
                idle = True
                break
            if self._released():
                released = True
                break
        waited = time.perf_counter() - started
        if not (idle or released):
            state.expired = epoch
        with self._lock:
            self.stats_["yields"] += 1
            self.stats_["yield_seconds"] += waited
            self.stats_["yield_timeouts"] += 0 if (idle or released) else 1
            self.stats_["yield_releases"] += 1 if released else 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.stats_)


class _SharedExpiry:
    expired = None


def _allocator_segments() -> list:
    """The caching allocator's segments (``torch.cuda.memory_snapshot``), without the history's trace entries where
    the running torch can leave them out (dev22j8)."""
    try:
        return torch.cuda.memory_snapshot(include_traces=False)
    except TypeError:  # torch < 2.10: no include_traces (traces are only built while history is recorded)
        return torch.cuda.memory_snapshot()


class _LivePeak:
    """The live (allocated) high-water of fitter work on its own streams over a block with no other GPU work (dev22j8;
    the attach warm-up): replayed from the caching allocator's alloc/free history (``alloc`` adds, ``free_completed``
    subtracts) when no one else records it -- exact; else the allocator's global peak when the block raised it (only
    the warm-up allocated meanwhile). ``bytes`` stays None when neither applies; ``method`` names the one used."""

    def __init__(self, device: torch.device, streams: set):
        self.device, self.streams = device, streams
        self.bytes: int | None = None
        self.method: str | None = None
        self._history = False

    def __enter__(self):
        try:
            if not torch._C._cuda_isHistoryEnabled():
                torch.cuda.memory._record_memory_history(enabled="all", context=None, stacks="python",
                                                         max_entries=PEAK_TRACE_ENTRIES, clear_history=True)
                self._history = True
        except Exception:  # noqa: BLE001 - measurement only: the global peak is the fallback
            self._history = False
        stats = torch.cuda.memory_stats(self.device)
        self._current = int(stats.get("allocated_bytes.all.current", 0))
        self._peak = int(stats.get("allocated_bytes.all.peak", 0))
        return self

    def __exit__(self, *exc) -> bool:
        try:
            entries = None
            if self._history:
                index = self.device.index if self.device.index is not None else torch.cuda.current_device()
                traces = torch.cuda.memory._snapshot().get("device_traces") or []
                entries = traces[index] if index < len(traces) else None
            if entries is not None and len(entries) < PEAK_TRACE_ENTRIES:  # a wrapped ring buffer lost events
                live = high = 0
                for entry in entries:
                    if entry.get("stream") not in self.streams:
                        continue
                    action = entry.get("action")
                    if action == "alloc":
                        live += int(entry.get("size", 0))
                        high = max(high, live)
                    elif action == "free_completed":
                        live -= int(entry.get("size", 0))
                self.bytes, self.method = high, "history"
            else:
                peak = int(torch.cuda.memory_stats(self.device).get("allocated_bytes.all.peak", 0))
                if peak > self._peak:
                    self.bytes, self.method = peak - self._current, "global_peak"
        except Exception:  # noqa: BLE001 - measurement only
            pass
        finally:
            if self._history:
                try:
                    torch.cuda.memory._record_memory_history(enabled=None)
                except Exception:  # noqa: BLE001
                    pass
        return False


_DEDICATED_STREAMS: dict = {}
_DEDICATED_LOCK = threading.Lock()


def _dedicated_stream(device: torch.device, priority: int | None):
    """The fitter's own CUDA stream (dev22j8v6, F6 b; prototyped in dev22j9a): created once per device and priority
    with the driver API (``cuStreamCreateWithPriority``, non-blocking like PyTorch's pooled streams) and wrapped as
    ``torch.cuda.ExternalStream``; never destroyed (the caching allocator may still record events on it).
    ``torch.cuda.Stream()`` hands out pooled streams round-robin, so a restore could share a pooled fitter stream and
    the decode guard's ``held`` (the fitter streams' live bytes) would count the restore's tensors."""
    import ctypes

    index = device.index if device.index is not None else torch.cuda.current_device()
    least, greatest = torch.cuda.Stream.priority_range()
    level = 0 if priority is None else max(min(int(priority), least), greatest)
    key = (index, level)
    with _DEDICATED_LOCK:
        stream = _DEDICATED_STREAMS.get(key)
        if stream is None:
            with torch.cuda.device(index):
                torch.cuda.current_stream(index).query()  # the device's primary context is current on this thread
                handle = ctypes.c_void_p()
                code = ctypes.CDLL("libcuda.so.1").cuStreamCreateWithPriority(ctypes.byref(handle), ctypes.c_uint(1),
                                                                              ctypes.c_int(level))
                if code != 0 or not handle.value:
                    raise RuntimeError(f"cuStreamCreateWithPriority failed (CUresult {code})")
                stream = torch.cuda.ExternalStream(handle.value, device=torch.device("cuda", index))
            _DEDICATED_STREAMS[key] = stream
    return stream


def _cpu_list(cpus) -> str:
    """A CPU set as a kernel CPU list (``"0-3,8"``)."""
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


class ArchivePipeline:
    """Ordered asynchronous latest-method archival for one runtime (see module docstring).

    ``device`` is the planning device (``"cpu"`` or ``"cuda:N"``); a CUDA
    pipeline plans on its own side stream. ``fitter`` optionally supplies the
    configured :class:`~reelcache.gpu_fitting.GPUFitter` (eigensolver choice); the
    fitter thread adopts it. ``hooks(stage, node_id)`` is a test seam called
    at the start of every stage (it may sleep or raise).

    ``encoder`` optionally selects an explicit :class:`reelcache.nvenc.NVENCEncoder`
    (opt-in; NVENC sessions open only on RTX PRO 6000 here): the method's
    codec must then be that encoder's NVENC specification
    (:func:`reelcache.nvenc.nvenc_spec`), node streams are encoded by NVENC one at
    a time (encode workers still run validation, hashing and storage in
    parallel), and packets record the encoder identity. Without ``encoder``
    an NVENC specification is rejected; libx265 stays the default.

    ``yield_to`` selects the restores archival yields to: ``"auto"`` (default)
    the ``restorer``'s :class:`~reelcache.archive_restore.RestoreActivity` when a
    restorer is given, ``None`` never, or a ``RestoreActivity`` / an object
    with an ``activity`` (any lineage restorer). ``yield_max_seconds`` bounds
    each worker's wait per restore busy period (one expiry per worker thread and
    its FFmpeg pump threads). ``pause_ffmpeg`` (default False; the SGLang bridge
    turns it on) also pauses the FFmpeg pipe pump chunk by chunk while a restore
    runs (``reelcache._hevc.ffmpeg_pump_pause``). ``cpu_priority`` (None, ``"idle"``
    or a nice increment 1..19) lowers the CPU priority of the FFmpeg processes
    the pipeline starts (``reelcache._hevc.ffmpeg_priority``). Both change timing
    only.

    ``raw_tier`` (a :class:`reelcache.raw_tier.RawChunkTier`, freeze D) keeps the
    raw copies of submitted chunks under its budgets instead of bounded source
    leases (see the module docstring); ``max_source_chunks`` and
    ``max_unpublished`` then apply only while the tier is disabled or degraded.
    """

    def __init__(self, runtime, *, method: RankMethodConfig, registry, options: ArchiveOptions | None = None,
                 device: Any = "cpu", fitter=None, encode_workers: int = 2, max_source_chunks: int = 2,
                 max_staged_canvases: int = 2, max_unpublished: int = 4, submit_timeout: float | None = None,
                 pin_staging: bool | None = None, hooks: Callable[[str, str], None] | None = None,
                 restorer=None, stream_priority: int | None = None, encoder=None, yield_to: Any = "auto",
                 yield_max_seconds: float = 1.0, cpu_priority: Any = None, pause_ffmpeg: bool = False,
                 raw_tier=None, cpu_set: Any = None, fitter_stream: str = "pooled", archive_budget: Any = None,
                 fitter_accel: Any = None, package_accel: Any = None, validation_pool: Any = None):
        options = ArchiveOptions() if options is None else options
        # accel C (dev_accel_20261030): opt-in host-footprint levers of the archival threads (None: the base pipeline)
        if package_accel is not None and not isinstance(package_accel, _accel_c.AccelOptions):
            raise ConfigurationError("package_accel must be a reelcache.accel_c.AccelOptions or None")
        self.package_accel = package_accel if package_accel is not None and package_accel.active else None
        self._validation_pool = validation_pool
        self._sealing = threading.local()  # accel C (seal_prepare_after_publish): seals this thread published
        if not isinstance(method, RankMethodConfig) or not isinstance(options, ArchiveOptions):
            raise ConfigurationError("an archive pipeline needs a RankMethodConfig and ArchiveOptions")
        if options.packing != "node":
            raise UnsupportedError("asynchronous archival stores node packs; per-matrix packets need the CPU source")
        from .nvenc import is_nvenc_preset
        if encoder is not None:
            from .nvenc import NVENCEncoder
            if not isinstance(encoder, NVENCEncoder):
                raise ConfigurationError("encoder must be an explicitly constructed reelcache.nvenc.NVENCEncoder")
            if not is_nvenc_preset(method.codec.preset):
                raise ConfigurationError("an NVENC encoder needs the method codec to be its NVENC specification "
                                         "(reelcache.nvenc.nvenc_spec); libx265 specifications stay on libx265")
            encoder.validate_spec(method.spec_for(1))
        elif is_nvenc_preset(method.codec.preset):
            raise UnsupportedError("the method codec is an NVENC specification: pass encoder=NVENCEncoder(...) "
                                   "(libx265 never substitutes for NVENC)")
        self.encoder = encoder
        self._encoder_lock = threading.Lock()
        self.encoder_identity = _encoder_identity(encoder, method)
        self.cpu_priority = _hevc.check_cpu_priority(cpu_priority)
        if type(pause_ffmpeg) is not bool:
            raise ConfigurationError("pause_ffmpeg must be boolean")
        self.pause_ffmpeg = pause_ffmpeg
        self._pump_hooks = threading.local()  # per worker thread: its pumps' shared yield hook
        # Callers blocked on archival progress (drain, tickets, backpressure, deferrals): workers do not yield then.
        self._waiters = 0
        self._waiters_lock = threading.Lock()
        for name, value in (("encode_workers", encode_workers), ("max_source_chunks", max_source_chunks),
                            ("max_staged_canvases", max_staged_canvases), ("max_unpublished", max_unpublished)):
            if type(value) is not int or value < 1:
                raise ConfigurationError(f"{name} must be a positive integer")
        if max_unpublished < max_source_chunks:
            raise ConfigurationError("the unpublished window must admit every source-holding chunk")
        if submit_timeout is not None and (isinstance(submit_timeout, bool) or submit_timeout < 0):
            raise ConfigurationError("submit_timeout must be None or nonnegative")
        self.runtime, self.method, self.registry, self.options = runtime, method, registry, options
        # W1.4.3 (E1): an archive budget (:class:`~reelcache.integrations.sglang.archive_eviction.ArchiveEviction`):
        # commits reserve their writes, publications become its live nodes (None: no budget, the F6 pipeline)
        self.archive_budget = archive_budget
        self.device = torch.device(device)
        self.fitter, self.hooks = fitter, hooks
        if stream_priority is not None and type(stream_priority) is not int:
            raise ConfigurationError("stream_priority must be an integer CUDA stream priority or None")
        # CUDA priority of the fitter's side stream (lower is higher priority); None: default priority.
        self.stream_priority = stream_priority
        # dev22j8v6: F6 b, the fitter's stream (the bridge passes "dedicated"); F6 a, the fitter and encoder threads' CPUs
        if fitter_stream not in FITTER_STREAMS:
            raise ConfigurationError(f"fitter_stream must be one of {FITTER_STREAMS}")
        self.fitter_stream_mode = fitter_stream
        self._fitter_stream_note = None
        self._init_accel(fitter_accel)
        if cpu_set is not None:
            cpu_set = frozenset(cpu_set)
            if not cpu_set or any(type(cpu) is not int or cpu < 0 for cpu in cpu_set):
                raise ConfigurationError("cpu_set must be None or a nonempty set of CPU indices")
        self.cpu_set = cpu_set
        self._cpu_local = threading.local()
        # Seal-time preparation: stored packs are prepared for device restore while still in memory.
        self.restorer = restorer
        self._yield = self._restore_yield(yield_to, yield_max_seconds)
        self.max_source_chunks, self.max_unpublished = max_source_chunks, max_unpublished
        self.submit_timeout = submit_timeout
        self._staging = _HostStaging(max_staged_canvases,
                                     self.device.type == "cuda" if pin_staging is None else pin_staging)
        self._executor = BoundedExecutor(max_workers=encode_workers, max_pending=max(encode_workers,
                                                                                     max_staged_canvases),
                                         max_staging_bytes=1 << 62)
        self._cond = threading.Condition()
        if raw_tier is not None:
            from .raw_tier import RawChunkTier
            if not isinstance(raw_tier, RawChunkTier):
                raise ConfigurationError("raw_tier must be a reelcache.raw_tier.RawChunkTier")
            raw_tier.attach(self._cond)
        self.raw_tier = raw_tier
        self._lineages: dict[str, _Lineage] = {}
        # W1.3 (R5): origin branch -> its detached lineages still in the pipeline (empty unless a detach happened)
        self._detached_by_origin: dict[str, set[str]] = {}
        self._queue: deque[_Entry] = deque()
        self._closed = False
        self._sources_held = 0
        # Memory-critical sections (defer_gpu_work) per thread, and whether the fitter is inside GPU work.
        self._deferrals: dict[int, int] = {}
        # Generation-first archival (Step 7 genfirst; None: off, the default -- nothing below acts)
        self._generation: _GenerationHolds | None = None
        self._gpu_busy = False
        self._gpu_item = None  # the node (or warm-up) whose GPU work is under way (dev22j6: inflight_gpu_bytes)
        self._dispatch_id = 0  # genfirst4: counts the fitter's dispatches while generation holds are on
        self._fitter_stream = None
        # Watched memory-critical sections (dev22j4): the fitter re-checks the device headroom before GPU work.
        self._watches: list[_Watch] = []
        self._fitter_working_set = 0
        # Fitter pause at step boundaries (dev22j7): open requests, whether the fitter is paused at a boundary now,
        # and its peak working set measured at the warm-up (no other GPU work then) with the warm-up's source bytes.
        self._pause_requests = 0
        self._pause_generation = 0  # one per pause request; a pause that outlasted its cap expires for its generation
        self._pause_expired = -1
        self._fitter_paused = False
        self._fitter_peak_bytes = 0
        self._fitter_peak_source_bytes = 0
        # dev22j8: the peak learned by default -- at the attach warm-up and online on the fitter's own streams, each
        # kept with its source bytes (the bound takes the larger, ``_fitter_peak_*``, and its basis); waits of the
        # fitter that allocate nothing (a pause landed meanwhile); the fitter's phase and its last step boundary.
        # Peaks are LIVE (allocated) bytes without the node's sources; reserved bytes are diagnostics only.
        self._warmup_peak = (0, 0)
        self._online_peak = (0, 0)
        self._peak_basis = None
        self._fitter_nodes = 0
        self._node_sample = None
        self._fitter_blocked = 0
        self._fitter_phase = ("idle", time.perf_counter())
        self._fitter_boundary = time.perf_counter()
        self._pause_events: deque = deque(maxlen=PAUSE_EVENTS)
        # dev22j8v6: the fitter's open unpausable span, its spans per phase, and the ack-timeout events still in theirs
        self._span_start: float | None = None
        self._step_spans: dict[str, dict] = {}
        self._timeouts_open: list = []
        self.stats_ = {"submitted": 0, "published": 0, "failed": 0, "cancelled": 0, "backpressure_waits": 0,
                       "backpressure_seconds": 0.0, "peak_unpublished": 0, "peak_sources_held": 0,
                       "fitter_busy_seconds": 0.0, "streamed_ranges": 0, "deferrals": 0,
                       "deferral_wait_seconds": 0.0, "deferred_seconds": 0.0, "deferral_released_bytes": 0,
                       "deferral_rejections": 0, "timeline": [], "planner_seconds": 0.0,
                       "fits": {"nodes": 0, "total": 0, **{name: 0 for name in _DECISIONS}},
                       "ranks": {"matrices": 0, "sum": 0, "mean": None}, "encoder_commits": {},
                       "adopt": {"calls": 0, "failures": 0, "last_failure": None},
                       "fitter_recheck": {"checks": 0, "deferrals": 0, "deferred_seconds": 0.0, "timeouts": 0,
                                          "bypassed": 0, "errors": 0, "working_set_bytes": 0},
                       "fitter_pause": {"requests": 0, "acked": 0, "ack_timeouts": 0, "ack_seconds": 0.0,
                                        "pauses": 0, "paused_seconds": 0.0, "timeouts": 0, "bypassed": 0,
                                        "landed": 0, "ack_timeout_phases": {}, "online_measurements": 0,
                                        "online_samples": 0, "online_sample_seconds": 0.0, "online_last_bytes": 0,
                                        "online_reserved_bytes": 0, "warmup_reserved_bytes": 0,
                                        "warmup_method": None, "warmup_baseline_bytes": 0},
                       # dev22j8v6 (F6 a): the CPU masks given to the fitter and encoder threads (None: inherited)
                       "cpu_sets": {"requested": None if cpu_set is None else _cpu_list(cpu_set), "fitter": None,
                                    "encoders": {}, "errors": 0, "last_error": None}}
        self._recheck_events: deque = deque(maxlen=RECHECK_EVENTS)
        self._lags: deque = deque(maxlen=256)
        self._lag_totals = {"count": 0, "sum": 0.0, "max": 0.0}
        self._thread = threading.Thread(target=self._fitter_loop, name="reelcache-archive-fitter", daemon=True)
        self._thread.start()

    # -- producer API ------------------------------------------------------------
    def submit(self, branch_id: str, node_id: str, matrices: Mapping, contexts: Mapping, *, kind: str = "chunk",
               ranges: Sequence[tuple[str, Mapping, Mapping]] | None = None, epoch_id: str | None = None,
               details: Mapping[str, Any] | None = None, execution_state: Mapping[str, torch.Tensor] | None = None,
               ready_event: Any = None, release: Callable[[Any], None] | None = None,
               timeout: float | None = ...) -> ArchiveTicket:
        """Enqueue one complete clean commit; returns at once unless a bound applies backpressure.

        ``matrices`` (device or CPU) stay owned by the caller until
        ``release(event)`` is called; the caller must make its writing stream
        wait on ``event`` (when not None) before overwriting them. ``ready_event``
        is recorded by the producer after writing the sources.
        """
        timeout = self.submit_timeout if timeout is ... else timeout
        if kind not in ("chunk", "recache", "segment"):
            raise ConfigurationError("commit kind must be chunk, recache or segment")
        groups = list(ranges) if ranges is not None else [(node_id, matrices, contexts)]
        if kind == "recache" and ranges is None:
            raise ConfigurationError("a recache commit lists its ranges")
        if release is not None and not callable(release):
            raise ConfigurationError("release must be callable")
        deadline = None if timeout is None else time.monotonic() + timeout
        started = time.perf_counter()
        waited = False
        nbytes = _commit_bytes(groups) if self.raw_tier is not None else 0
        cause = waited_for = None
        with self._cond:
            while True:
                if self._closed:
                    raise StateError("archive pipeline is closed")
                lineage = self._lineage(branch_id)
                if lineage.error is not None:
                    raise raisable(lineage.error)
                # The raw tier (freeze D) admits by bytes; disabled or degraded, the C3 bounds apply.
                tier = self.raw_tier if self.raw_tier is not None and self.raw_tier.usable(nbytes) else None
                if tier is None:
                    if (self._unpublished_of(lineage) < self.max_unpublished
                            and self._sources_held < self.max_source_chunks):
                        break
                    cause = "c3_bounds"
                    blocked = self._blocked_by_deferral(lineage)
                else:
                    cause = tier.admission(nbytes)
                    if cause is None:
                        break
                    # Room for unpublished chunks comes from publication, which needs fits; spills do not.
                    blocked = cause != "gpu_spill" and not self._gpu_busy
                if threading.get_ident() in self._deferrals and blocked:
                    self.stats_["deferral_rejections"] += 1
                    raise CapacityError("archive pipeline is full and this thread defers GPU work: the queued fits "
                                        "that would make room cannot start before its defer_gpu_work section exits")
                waited, waited_for = True, cause
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise CapacityError("archive pipeline queue is full (explicit backpressure)")
                with self._waiting():
                    self._cond.wait(remaining)
            seconds = time.perf_counter() - started
            if waited:
                self.stats_["backpressure_waits"] += 1
                self.stats_["backpressure_seconds"] += seconds
                if tier is not None and waited_for not in (None, "c3_bounds"):
                    tier.record_wait(waited_for, seconds, branch=branch_id, node_id=node_id)
            # A reservation holds the entry's bytes while it is validated; it is rolled back before any lineage
            # state advances when validation fails (R2).
            reservation = tier.reserve(nbytes) if tier is not None else None
            try:
                checked = []
                for chunk_id, group_matrices, group_contexts in groups:
                    checked.append((chunk_id, group_matrices,
                                    self.runtime._check_group(chunk_id, group_matrices, group_contexts,
                                                              lineage.shape.streams)))
                if kind != "recache" and len(checked) != 1:
                    raise ConfigurationError("chunk and segment commits carry exactly one range")
                self.runtime._validate_commit_shape(kind, node_id, checked, lineage.shape)
                raw = None
                if tier is not None:
                    raw = tier.adopt(reservation, branch=branch_id, node_id=node_id, ready_event=ready_event,
                                     release=release, groups=[(chunk_id, _token_start(contexts_), matrices_)
                                                              for chunk_id, matrices_, contexts_ in checked])
            except BaseException:
                if reservation is not None:
                    tier.unreserve(reservation)
                raise
            lineage.shape = advance_shape(lineage.shape, kind, node_id,
                                          [(chunk_id, contexts_) for chunk_id, _, contexts_ in checked], epoch_id)
            lineage.sequence += 1
            ticket = ArchiveTicket(branch_id, node_id, lineage.sequence, lineage.shape.frontier)
            ticket._waiting = self._waiting
            ticket.admission = {"raw_tier": raw is not None, "waited": waited,
                                "seconds": seconds if waited else 0.0, "cause": waited_for if waited else None,
                                "bytes": nbytes if raw is not None else None}
            if raw is not None:
                # The tier holds the sources (and calls ``release`` when it gives them up); the entry keeps contexts.
                entry = _Entry(ticket, lineage, kind, node_id, [(chunk_id, {}, contexts_)
                                                                for chunk_id, _, contexts_ in checked],
                               epoch_id, details, execution_state, ready_event, None, time.perf_counter())
                entry.raw = raw
            else:
                entry = _Entry(ticket, lineage, kind, node_id, checked, epoch_id, details, execution_state,
                               ready_event, release, time.perf_counter())
            entry.times["submitted"] = time.perf_counter()
            entry.captured_at = started  # genfirst3 (#9): the age guard's clock, from the chunk's capture
            lineage.entries.append(entry)
            lineage.generated_frontier = lineage.shape.frontier
            self._sources_held += 1
            self.stats_["submitted"] += 1
            self.stats_["peak_unpublished"] = max(self.stats_["peak_unpublished"], len(lineage.entries))
            self.stats_["peak_sources_held"] = max(self.stats_["peak_sources_held"], self._sources_held)
            self._queue.append(entry)
            self._cond.notify_all()
            return ticket

    # -- memory-critical sections ------------------------------------------------------
    @contextlib.contextmanager
    def defer_gpu_work(self, *, wait: bool = True, release_cached_memory: bool = False,
                       timeout: float | None = None):
        """A caller's memory-critical section (e.g. a VAE decode): no GPU fitting starts while it is open.

        On entry (``wait=True``) the call blocks until the fitter thread is outside GPU work: a node being planned
        and finalized (or a prewarm) finishes first -- at most one node; ``timeout`` bounds this wait
        (``CapacityError``, the section is not entered). ``release_cached_memory=True`` then synchronizes the
        fitter's stream and releases the CUDA caching allocator's unoccupied blocks (``torch.cuda.empty_cache()``,
        process-wide), so the fitter's cached pool is free for the section. While any section is open the fitter
        takes no queued node or prewarm; nodes keep their order, ``submit`` keeps its bounds and backpressure,
        encode workers and ordered publication continue. A thread inside a section must not wait for work only the
        fitter can do: its ``submit`` raises ``CapacityError`` instead of blocking when only deferred fits could
        make room, and its ``drain``, ``prewarm`` and ``close(drain=True)`` raise ``StateError`` when they would
        wait for deferred fits. Sections nest and overlap across threads; fitting resumes when the last one exits.
        ``stats()``: ``deferrals``, ``deferral_wait_seconds``, ``deferred_seconds``, ``deferral_released_bytes``,
        ``deferral_rejections``, ``deferred`` (a section is open).
        """
        if type(wait) is not bool or type(release_cached_memory) is not bool:
            raise ConfigurationError("wait and release_cached_memory must be boolean")
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                                    or timeout < 0):
            raise ConfigurationError("timeout must be None or a nonnegative number of seconds")
        me = threading.get_ident()
        if me == self._thread.ident:
            raise StateError("the fitter thread cannot defer its own GPU work")
        started = time.perf_counter()
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            if self._closed:
                raise StateError("archive pipeline is closed")
            self._deferrals[me] = self._deferrals.get(me, 0) + 1
            try:
                while wait and self._gpu_busy:
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise CapacityError("the archive fitter did not leave its GPU work within the deferral "
                                            "timeout")
                    with self._waiting():
                        self._cond.wait(remaining)
            except BaseException:
                self._leave_deferral(me)
                raise
            self.stats_["deferrals"] += 1
            self.stats_["deferral_wait_seconds"] += time.perf_counter() - started
        entered = time.perf_counter()
        try:
            if release_cached_memory and self.device.type == "cuda":
                if self._fitter_stream is not None:
                    self._fitter_stream.synchronize()
                before = torch.cuda.memory_reserved(self.device)
                torch.cuda.empty_cache()
                released = max(0, before - torch.cuda.memory_reserved(self.device))
                with self._cond:
                    self.stats_["deferral_released_bytes"] += released
            yield self
        finally:
            with self._cond:
                self.stats_["deferred_seconds"] += time.perf_counter() - entered
                self._leave_deferral(me)

    @contextlib.contextmanager
    def pause_gpu_work(self, *, ack_timeout: float | None = None):
        """Pause archival GPU work for the caller's section without waiting for a node to finish (dev22j7, the decode
        entry guard).

        The section holds queued nodes back like :meth:`defer_gpu_work` with ``wait=False`` (same rules for
        ``submit``, ``drain`` and ``prewarm`` inside it), and the node on the GPU, if any, pauses at its next step
        boundary: every lockstep routine step of its planner and finalization and every ``yield_point`` (the step
        hook the restore yield uses; no lock is held there). The fitter allocates device memory only from its own
        thread, so a paused fitter's allocation stays what it is: it no longer grows during the section, and what it
        holds is already in the driver's free memory. The caller waits for that boundary at most ``ack_timeout``
        seconds (default ``PAUSE_ACK_TIMEOUT``). Yields ``{"acked": bool, "ack_seconds": float, "inflight": bool,
        "landed": str | None}``; ``acked`` is also True when no node is on the GPU (``landed`` ``"idle"``). dev22j8:
        the pause also lands while the fitter waits without allocating (``deferred_reads.blocked_wait``: a host read's
        device synchronization, the staging-slot wait, an in-place encode, the rate-lag dependency wait; ``landed`` is
        that wait's label, else ``"boundary"``): the end of such a wait is a step boundary, where it holds. An
        acknowledgement timeout records the fitter's phase (``phase``, ``phase_seconds``, ``boundary_seconds``: since
        its last step boundary), also in ``stats()["fitter_pause"]`` (``ack_timeout_phases``,
        ``ack_timeout_events``). A paused fitter resumes when the last pause exits, when a
        caller waits for archival (``drain``, a ticket), when the pipeline closes, or after ``PAUSE_MAX_SECONDS``
        (then the open pauses expire: the fitter pauses again only for a new request).
        ``stats()["fitter_pause"]`` counts requests, acknowledgements, ack timeouts and paused seconds.
        """
        timeout = PAUSE_ACK_TIMEOUT if ack_timeout is None else ack_timeout
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0:
            raise ConfigurationError("ack_timeout must be a nonnegative number of seconds")
        with self.defer_gpu_work(wait=False):
            started = time.perf_counter()
            deadline = time.monotonic() + timeout
            with self._cond:
                self._pause_requests += 1
                self._pause_generation += 1
                self.stats_["fitter_pause"]["requests"] += 1
                inflight = self._gpu_busy
                self._cond.notify_all()
                try:
                    while (self._gpu_busy and not self._fitter_paused and not self._fitter_blocked
                           and not self._closed):
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._cond.wait(remaining)
                    landed = self._landed()
                    acked = landed is not None
                    waited = time.perf_counter() - started
                    pause = self.stats_["fitter_pause"]
                    pause["acked" if acked else "ack_timeouts"] += 1
                    pause["ack_seconds"] += waited
                    info = {"acked": acked, "ack_seconds": waited, "inflight": inflight, "landed": landed}
                    if acked and landed not in ("idle", "boundary"):
                        pause["landed"] += 1
                    if not acked:  # locate the step that held the fitter past the timeout
                        info.update(self._phase_record())
                        phases = pause["ack_timeout_phases"]
                        phases[info["phase"]] = phases.get(info["phase"], 0) + 1
                        event = {"time": time.time(), **info}
                        self._pause_events.append(event)
                        if self._span_start is not None:  # completed when the step ends (dev22j8v6)
                            self._timeouts_open.append((event, time.perf_counter()))
                except BaseException:
                    self._pause_requests -= 1
                    self._cond.notify_all()
                    raise
            try:
                yield info
            finally:
                with self._cond:
                    self._pause_requests -= 1
                    self._cond.notify_all()

    def _landed(self) -> str | None:
        """Where a requested pause landed (caller holds the condition): ``"idle"`` (no node on the GPU),
        ``"boundary"`` (paused at a step boundary), a blocked wait's label (dev22j8), or None (not yet)."""
        if not self._gpu_busy:
            return "idle"
        if self._fitter_paused:
            return "boundary"
        if self._fitter_blocked:
            return self._fitter_phase[0]
        return None

    def _phase_record(self) -> dict[str, Any]:
        """The fitter's phase now (dev22j8): its label, seconds in it and seconds since its last step boundary."""
        label, since = self._fitter_phase
        now = time.perf_counter()
        return {"phase": label, "phase_seconds": now - since, "boundary_seconds": now - self._fitter_boundary}

    def _set_phase(self, label: str) -> None:
        """The fitter thread's phase hook (dev22j8, ``deferred_reads.step_phase``; records only). dev22j8v6: going idle
        ends the fitter's unpausable span; leaving idle starts one."""
        now = time.perf_counter()
        if label == "idle":
            self._end_span(now, "idle")
        elif self._span_start is None:
            self._span_start = now
        self._fitter_phase = (label, now)

    def _end_span(self, now: float, end: str) -> None:
        """dev22j8v6: close the fitter's open unpausable span (since its last step boundary) under its phase
        (``step_spans``) and give the ack-timeout events it held ``boundary_after_seconds`` (how long the step ran on
        after them), ``step_seconds`` and ``step_end`` (``"boundary"``, ``"blocked"`` or ``"idle"``)."""
        start, self._span_start = self._span_start, None
        if start is None:
            return
        span = now - start
        with self._cond:
            label = self._fitter_phase[0]
            record = self._step_spans.get(label)
            if record is None:
                record = self._step_spans[label] = {"n": 0, "seconds": 0.0, "max": 0.0,
                                                    "buckets": [0] * (len(SPAN_BUCKETS) + 1)}
            record["n"] += 1
            record["seconds"] += span
            record["max"] = max(record["max"], span)
            record["buckets"][bisect.bisect_left(SPAN_BUCKETS, span)] += 1
            for event, when in self._timeouts_open:
                event.update(boundary_after_seconds=max(0.0, now - when), step_seconds=span, step_end=end)
            self._timeouts_open.clear()

    @contextlib.contextmanager
    def _blocked(self, label: str):
        """A wait of the fitter thread that allocates no device memory (dev22j8, ``deferred_reads.blocked_wait``): a
        pause requested meanwhile counts as landed (it is acknowledged at once), and the wait's end is a step boundary,
        where the landed pause holds the fitter before it allocates again."""
        self._sample_node()  # a measured node's live set right before the wait
        self._end_span(time.perf_counter(), "blocked")  # a pause lands at once inside the wait
        with self._cond:
            outer = self._fitter_phase
            self._fitter_blocked += 1
            self._fitter_phase = (label, time.perf_counter())
            if self._pause_requests:
                self._cond.notify_all()  # a pausing caller waits for the acknowledgement
        try:
            yield
        finally:
            with self._cond:
                self._fitter_blocked -= 1
                self._fitter_phase = outer
            self._step_boundary()

    def _step_boundary(self) -> None:
        """The fitter thread's step hook: the restore yield, then a requested pause (dev22j7). dev22j8v6: it ends the
        current unpausable span and starts the next one after the yield and the pause."""
        self._fitter_boundary = time.perf_counter()
        self._end_span(self._fitter_boundary, "boundary")
        if self._node_sample is not None:
            self._sample_node()
        if self._yield is not None:
            self._yield()
        if self._pause_requests and not self._closed:  # an unlocked read: the common case costs one attribute test
            self._pause_at_boundary()
        if self._fitter_phase[0] != "idle":
            self._span_start = time.perf_counter()

    def _pause_at_boundary(self) -> None:
        with self._cond:
            if not self._pause_requests or self._closed or self._pause_expired == self._pause_generation:
                return  # nothing asked, closing, or these pauses already held the fitter for PAUSE_MAX_SECONDS
            pause = self.stats_["fitter_pause"]
            if self._waiters:  # someone waits for archival progress: never hold it
                pause["bypassed"] += 1
                return
            started = time.perf_counter()
            deadline = started + PAUSE_MAX_SECONDS
            self._fitter_paused = True
            self._cond.notify_all()
            try:
                while self._pause_requests and not self._closed:
                    if self._waiters:
                        pause["bypassed"] += 1
                        break
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        pause["timeouts"] += 1
                        self._pause_expired = self._pause_generation
                        break
                    self._cond.wait(remaining)
            finally:
                self._fitter_paused = False
                pause["pauses"] += 1
                pause["paused_seconds"] += time.perf_counter() - started
                self._cond.notify_all()

    @contextlib.contextmanager
    def watch_section(self, reason: str, recheck: Callable[[int], Mapping[str, Any] | None]):
        """A memory-critical section the fitter watches without deferring (dev22j4; ``LineageSessionArchive.
        memory_critical`` opens one around every section).

        While one is open, before a queued node's (or a warm-up's) GPU work starts the fitter calls
        ``recheck(need_bytes)``. ``need_bytes`` is the work's device bytes:

        * the re-upload of a demoted raw copy;
        * plus the fitter working set: the largest measured on the device, else ``options.plan_memory_bytes``, else
          the node's source bytes.

        A result with ``short`` true defers that work until every watched section has closed. The defer is capped at
        ``RECHECK_DEFER_TIMEOUT``, and lifted at once while a caller waits for archival (``drain``, a ticket's
        ``result``), so archival never deadlocks behind a section.

        Each deferral is an event (``deferred_archival``, ``reason: "fitter_recheck"``, the node, the bytes, the wait).
        It goes to ``stats()["fitter_recheck"]`` and to the list this context yields.
        """
        watch = _Watch(reason, recheck)
        with self._cond:
            self._watches.append(watch)
        try:
            yield watch.events
        finally:
            with self._cond:
                self._watches.remove(watch)
                now = time.perf_counter()
                for event in watch.events:
                    if "waited_seconds" not in event:
                        event["waited_seconds"] = now - event["_started"]
                self._cond.notify_all()

    def inflight_gpu_bytes(self) -> int:
        """Device bytes the node (or warm-up) now on the GPU may still allocate (dev22j6, the decode entry guard): the
        fitter working set (the largest measured, else ``options.plan_memory_bytes``), else the node's source bytes;
        0 with nothing on the GPU. Conservative: what the node has already allocated is not subtracted."""
        with self._cond:
            item = self._gpu_item
            if not self._gpu_busy or item is None:
                return 0
            working = self._fitter_working_set or self.options.plan_memory_bytes or 0
            if isinstance(item, _Warmup):
                return int(working)
            try:
                return int(working or _commit_bytes(getattr(item, "groups", None) or ()))
            except Exception:  # noqa: BLE001 - an estimate only
                return int(working)

    def inflight_growth_bound(self) -> dict[str, Any]:
        """An upper bound of the device bytes the node (or warm-up) now on the GPU may still allocate (dev22j7, the
        decode entry guard): ``{"inflight": bool, "bytes": int, "basis": str | None}`` (with a measured peak also
        ``"scale"`` and ``"upload"``). dev22j8: the fitter's peak LIVE working set (allocated bytes on its own streams,
        without the node's sources) times max(1, the node's source bytes over the measurement's), from the larger of
        two measurements -- ``basis`` ``"warmup"``, the attach warm-up's live high-water (``warm_commit_path``; its
        synthetic sources excluded), or ``"online"``, the live high-water sampled on a measured node (see
        :meth:`fitter_memory`) -- plus the re-upload of the node's raw copy while it is not on the device (staged on
        the fitter's stream, then the tier's). Without either measurement ``"working_set"``: :meth:`inflight_gpu_bytes`
        (conservative: the node's working set as measured next to the generation's allocations). 0 bytes with nothing
        on the GPU."""
        with self._cond:
            item = self._gpu_item
            if not self._gpu_busy or item is None:
                return {"inflight": False, "bytes": 0, "basis": None}
            peak, reference = self._fitter_peak_bytes, self._fitter_peak_source_bytes
            if peak > 0:
                scale, upload = 1.0, 0
                if not isinstance(item, _Warmup):
                    try:
                        if reference > 0:
                            scale = max(1.0, _source_bytes(item) / reference)
                        raw = getattr(item, "raw", None)
                        if raw is not None and not raw.on_device and not raw.dropped:
                            upload = int(raw.nbytes)
                    except Exception:  # noqa: BLE001 - an estimate only
                        scale, upload = 1.0, 0
                return {"inflight": True, "bytes": int(peak * scale) + upload, "basis": self._peak_basis or "warmup",
                        "scale": scale, "upload": upload}
        return {"inflight": True, "bytes": self.inflight_gpu_bytes(), "basis": "working_set"}

    def fitter_memory(self, segments: list | None = None) -> dict[str, int] | None:
        """What the fitter's own streams (the pipeline's planning stream and the GPU fitter's side stream) hold on this
        device now, from one ``torch.cuda.memory_snapshot`` (dev22j8): ``held``, their live (allocated) bytes without
        the raw tier's device copies (a demoted copy re-staged on the fitter's stream becomes the tier's); diagnostics
        ``allocated`` (every live block), ``tier``, ``reserved`` (every segment), ``fragments`` (the free bytes of
        their segments in use: inactive split, outside the conservative device headroom) and ``segments``. The decode
        guard's remaining growth is ``max(0, bound - held)``. None off CUDA or before the fitter thread made its
        stream. ``segments``: a snapshot the caller already took (dev22j8v4: the decode guard's strict headroom)."""
        if self.device.type != "cuda":
            return None
        streams = {stream.cuda_stream for stream in (self._fitter_stream, getattr(self.fitter, "_stream", None))
                   if stream is not None}
        if not streams:
            return None
        try:
            segments = _allocator_segments() if segments is None else segments
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            owned = self._tier_pointers()
        except Exception:  # noqa: BLE001 - measurement only
            return None
        result = {"held": 0, "allocated": 0, "tier": 0, "reserved": 0, "fragments": 0, "segments": 0}
        for segment in segments:
            if segment.get("stream") not in streams or segment.get("device", index) != index:
                continue
            size, allocated = int(segment.get("total_size", 0)), int(segment.get("allocated_size", 0))
            result["segments"] += 1
            result["reserved"] += size
            result["allocated"] += allocated
            if allocated > 0:
                result["fragments"] += size - allocated
                if owned:
                    result["tier"] += sum(int(block.get("size", 0)) for block in segment.get("blocks") or ()
                                          if block.get("state") == "active_allocated"
                                          and block.get("address") in owned)
        result["held"] = max(0, result["allocated"] - result["tier"])
        return result

    def _tier_pointers(self) -> set:
        """Storage addresses of the raw tier's device copies (dev22j8: excluded from the fitter's holdings)."""
        tier = self.raw_tier
        if tier is None:
            return set()
        pointers = set()
        with tier._cond:
            for entry in tier._entries.values():
                if entry.on_device and not entry.dropped:
                    for group in entry.groups:
                        for tensor in (group.device or {}).values():
                            if tensor.device.type == "cuda":
                                pointers.add(tensor.untyped_storage().data_ptr())
        return pointers

    def _gpu_work_bytes(self, item: Any) -> int:
        """Device bytes a queued node's (or warm-up's) GPU work needs (caller holds the condition)."""
        working = self._fitter_working_set or self.options.plan_memory_bytes or 0
        if isinstance(item, _Warmup):
            return int(working)
        raw = item.raw
        upload = raw.nbytes if raw is not None and not raw.on_device and not raw.dropped else 0
        return int(upload + (working or _commit_bytes(item.groups)))

    def _recheck_verdict(self, item: Any) -> dict[str, Any] | None:
        """The newest watched section's verdict on ``item``'s GPU work (caller holds the condition); None when it
        fits or nothing is watched."""
        if not self._watches or self._closed or getattr(item, "stage", None) in ("failed", "cancelled"):
            return None
        recheck = self.stats_["fitter_recheck"]
        recheck["checks"] += 1
        watch = self._watches[-1]
        need = self._gpu_work_bytes(item)
        try:
            verdict = watch.recheck(need)
        except Exception as exc:  # noqa: BLE001 - the re-check is best effort: the work starts
            recheck["errors"] += 1
            self._recheck_events.append({"event": "recheck_error", "section": watch.reason,
                                         "error": f"{type(exc).__name__}: {exc}"})
            return None
        if not verdict or not verdict.get("short"):
            return None
        return {"event": "deferred_archival", "deferred_archival": True, "reason": "fitter_recheck",
                "section": watch.reason, "node_id": getattr(item, "node_id", "prewarm"), "need_bytes": need,
                **{key: verdict.get(key) for key in ("headroom", "reserve", "target")}}

    def _recheck_wait(self, event: dict[str, Any]) -> bool:
        """Hold the queue head back while watched sections stay open (caller holds the condition; the wait releases
        it). Returns True when the sections closed (the caller re-evaluates the queue). Returns False when it timed
        out, a caller waits for archival, or the pipeline closed: the work then starts."""
        started = event["_started"] = time.perf_counter()
        deadline = started + RECHECK_DEFER_TIMEOUT
        watches = list(self._watches)
        for watch in watches:
            watch.events.append(event)
        recheck = self.stats_["fitter_recheck"]
        recheck["deferrals"] += 1
        outcome = "sections_closed"
        while self._watches and not self._closed:
            if self._waiters:
                outcome = "caller_waits"
                recheck["bypassed"] += 1
                break
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                outcome = "timed_out"
                recheck["timeouts"] += 1
                break
            self._cond.wait(remaining)
        waited = time.perf_counter() - started
        event.setdefault("waited_seconds", waited)
        event["archival_deferred_seconds"] = event["waited_seconds"]
        event["outcome"] = outcome
        recheck["deferred_seconds"] += waited
        self._recheck_events.append({key: value for key, value in event.items() if not key.startswith("_")})
        return outcome == "sections_closed"

    # -- generation-first archival (Step 7 genfirst; config ``defer_archival_during_generation``) ----------------------
    def enable_generation_holds(self, *, backlog_fraction: float = GENERATION_BACKLOG_FRACTION,
                                max_age_seconds: float = GENERATION_MAX_AGE_SECONDS, inflight: str = "wait") -> None:
        """Turn generation holds on (idempotent; off by default: nothing of them exists). A request's generation --
        from its restore's return to the end of its VAE decode -- opens a hold (:meth:`open_generation`); while any is
        open the fitter starts no node, unless the starvation guard lets one through (see :class:`_GenerationHolds`).
        ``inflight`` (:data:`GENERATION_INFLIGHT`): whether opening waits for the node already on the GPU."""
        for name, value, low in (("backlog_fraction", backlog_fraction, 0.0), ("max_age_seconds", max_age_seconds, 0.0)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > low or value != value:
                raise ConfigurationError(f"{name} must be a positive number")
        if inflight not in GENERATION_INFLIGHT:
            raise ConfigurationError(f"inflight must be one of {GENERATION_INFLIGHT}")
        with self._cond:
            if self._generation is None:
                self._generation = _GenerationHolds(float(backlog_fraction), float(max_age_seconds), inflight)

    def open_generation(self, label: str = "generation") -> dict[str, Any] | None:
        """Open a generation hold (the caller's request generates now): no new GPU archival work starts until it is
        closed. A node already on the GPU (``inflight``): with ``inflight="wait"`` it finishes first -- the wait (that
        node only) is the request's ``archival_defer_wait``; with ``"overlap"`` the hold opens at once and the node's
        remainder runs beside the request (``waited`` False; its overlap is reported when the hold closes). Returns
        ``{"token", "wait_seconds", "inflight", "waited"}``, or None when holds are off or the pipeline is closed. Never
        waits for anything else: ``submit`` keeps its bounds and never raises because of a hold, and the starvation
        guard lets fitting through while a caller waits on archival."""
        holds = self._generation
        if holds is None:
            return None
        if self._thread.ident is not None and threading.get_ident() == self._thread.ident:
            raise StateError("the fitter thread cannot hold its own GPU work")
        started = time.perf_counter()
        with self._cond:
            if self._closed:
                return None
            inflight, dispatch = self._gpu_busy, self._dispatch_id
            overlapping = self._gpu_item if inflight and holds.inflight == "overlap" else None
            # opened once the condition is held: the overlap excludes the lock's entry latency (genfirst4, M5)
            token = holds.open(label, time.perf_counter(), overlapping)
            # "wait": the dispatch on the GPU when the hold opened finishes first (a plain wait: no waiter); a successor
            # the guard lets through meanwhile -- or the same node handed back and restarted -- is a new dispatch,
            # not waited for (genfirst3 review #4; genfirst4 review I1).
            while overlapping is None and self._gpu_busy and self._dispatch_id == dispatch and not self._closed:
                self._cond.wait()
            waited = time.perf_counter() - started
            holds.waited(waited, inflight and overlapping is None)
            holds.observe_backlog(self)
        return {"token": token, "wait_seconds": waited, "inflight": inflight,
                "waited": overlapping is None and inflight}

    def close_generation(self, token: Any, report: dict[str, Any] | None = None) -> bool:
        """Close a hold (idempotent: False when it was closed already); fitting resumes with the last one.
        ``report`` (a dict) receives this hold's ``overrides`` (counted while it was open; genfirst3, review #7) and
        ``inflight_overlap_seconds`` / ``inflight_overlap`` (``overlap`` mode)."""
        holds = self._generation
        if holds is None or token is None:
            return False
        with self._cond:
            closed = holds.close(token, time.perf_counter())
            if closed is not None:
                holds.observe_backlog(self)
                self._cond.notify_all()
                if report is not None:
                    report.update(closed)
        return closed is not None

    def generation_overrides(self) -> dict[str, int] | None:
        """The starvation guard's override counts by reason so far (None: holds off)."""
        with self._cond:
            return None if self._generation is None else dict(self._generation.counters["overrides"])

    def _generation_blocks(self) -> bool:
        """Whether open generation holds keep the fitter from starting a node now (caller holds the condition)."""
        holds = self._generation
        return holds is not None and holds.blocking(self)

    def _leave_deferral(self, thread_id: int) -> None:
        count = self._deferrals[thread_id] - 1
        if count:
            self._deferrals[thread_id] = count
        else:
            del self._deferrals[thread_id]
        self._cond.notify_all()

    def _blocked_by_deferral(self, lineage: _Lineage) -> bool:
        """Whether a full pipeline can only make room through fits that the open sections defer (under the
        condition): no node is on the GPU, and either every source lease belongs to a queued node or the lineage's
        oldest unpublished node is still queued (publication is in order)."""
        if self._gpu_busy:
            return False
        if self._sources_held >= self.max_source_chunks:
            return True
        return not any(entry.stage in ("finalized", "stored") for entry in lineage.entries)

    def _deferring_thread(self) -> bool:
        return threading.get_ident() in self._deferrals

    def _restore_yield(self, yield_to, max_seconds) -> _RestoreYield | None:
        if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or not max_seconds > 0:
            raise ConfigurationError("yield_max_seconds must be a positive number of seconds")
        if isinstance(yield_to, str):
            if yield_to != "auto":
                raise ConfigurationError("yield_to must be 'auto', None, a RestoreActivity or a restorer")
            yield_to = getattr(self.restorer, "activity", None)
        elif yield_to is not None and not hasattr(yield_to, "wait_idle"):
            if not hasattr(getattr(yield_to, "activity", None), "wait_idle"):
                raise ConfigurationError("yield_to must be 'auto', None, a RestoreActivity or a restorer")
            yield_to = yield_to.activity
        return None if yield_to is None else _RestoreYield(yield_to, float(max_seconds),
                                                           bypass=lambda: self._waiters > 0)

    def _lineage(self, branch_id: str) -> _Lineage:
        lineage = self._lineages.get(branch_id)
        if lineage is not None:
            return lineage
        runtime = self.runtime
        with runtime._lock:
            runtime._open()
            if branch_id in runtime._lineage_pending:
                raise PendingError(f"lineage has a commit in flight: {branch_id}")
            with runtime.store.transaction():
                head_ref = runtime.lineage_head(branch_id)
                path = runtime.node_path(head_ref)
                # Shape and bindings only: basis rows are assembled on the fitter thread.
                shape = runtime._parent_state(branch_id, head_ref, path, self.method.rate_lag, rows=False)
            runtime._lineage_pending.add(branch_id)
        lineage = _Lineage(branch_id, head_ref, shape.frontier, shape.frontier, shape, published_state=shape)
        self._lineages[branch_id] = lineage
        return lineage

    def _unpublished_of(self, lineage: _Lineage) -> int:
        """W1.3: the unpublished commits a lineage counts against ``max_unpublished`` (caller holds the condition):
        its own plus those of every lineage detached from it, which keep publishing (the same in-flight bound).
        Constant time while nothing was detached from it (drain mode: always, R5)."""
        count = len(lineage.entries)
        if self._detached_by_origin.get(lineage.branch_id):
            count += self._detached_unpublished(lineage.branch_id)
        return count

    def _detached_unpublished(self, origin: str) -> int:
        """W1.3: the unpublished commits of the lineages detached from ``origin`` (caller holds the condition)."""
        return sum(len(self._lineages[branch].entries) for branch in self._detached_by_origin.get(origin, ())
                   if branch in self._lineages)

    def detach(self, branch_id: str, new_branch_id: str) -> dict[str, Any]:
        """W1.3 (``tree_switch="detach"``): hand ``branch_id``'s unpublished commits to a new lineage
        ``new_branch_id`` whose ref starts at their published head (the cut); they keep publishing in order there (with
        their publication refs), never moving ``branch_id``'s head, which the caller may then move (rewind).

        Waits for at most one publication in progress (``detaching``: the publisher starts no further transaction; it
        waits and resumes under the outcome). Strong exception guarantee: an exception before the commit point (the new
        lineage's ref created) is raised as :class:`DetachRolledBack` once everything is unchanged (the created ref
        rolled back), so the caller may drain instead -- the lineage's own failure too, which the drain then surfaces;
        after the commit point the hand-off is in memory and starts nothing (no thread: R1), so it cannot fail (a
        raw-tier bookkeeping error disables the tier, recorded, as at publication). Returns ``{"branch", "moved" (node
        ids; [] when nothing is in flight), "cut"}``. Test seam: ``detach_hook(step)``."""
        runtime, hook = self.runtime, getattr(self, "detach_hook", None)
        moved, cut, lineage = [], None, None
        with self._cond:
            lineage = self._lineages.get(branch_id)
            if lineage is None:
                return {"branch": new_branch_id, "moved": [], "cut": None}
            lineage.detaching = True
            try:
                created = None
                try:
                    while any(entry.publishing for entry in lineage.entries):
                        self._cond.wait(0.05)
                    if lineage.error is not None:
                        raise raisable(lineage.error)
                    cut = lineage.published_ref
                    if not lineage.entries:
                        return {"branch": new_branch_id, "moved": [], "cut": cut}
                    if new_branch_id in self._lineages:
                        raise StateError(f"lineage {new_branch_id} is already in the pipeline")
                    moved = [entry.node_id for entry in lineage.entries]
                    if hook is not None:
                        hook("validated")
                    with runtime._lock:
                        runtime._open()
                        name = runtime._lineage_name(new_branch_id)
                        with runtime.store.transaction():
                            runtime.store.set_ref(name, cut, expected=None)
                        created = name
                        if hook is not None:
                            hook("created")
                except BaseException as exc:
                    if created is not None:  # roll back; if that fails the guarantee is lost: not DetachRolledBack
                        with runtime._lock, runtime.store.transaction():
                            runtime.store.delete_ref(created, expected=cut)
                    raise DetachRolledBack(f"{type(exc).__name__}: {exc}") from exc
                # -- commit point: the hand-off below is in memory and cannot fail ------------------------------------
                with runtime._lock:
                    runtime._lineage_pending.discard(branch_id)
                    runtime._lineage_pending.add(new_branch_id)
                    runtime.rebind_publication_refs(branch_id, new_branch_id, moved)
                del self._lineages[branch_id]
                self._lineages[new_branch_id] = lineage
                lineage.branch_id, lineage.detached_from = new_branch_id, branch_id
                self._detached_by_origin.setdefault(branch_id, set()).add(new_branch_id)
                for entry in lineage.entries:
                    entry.ticket.publication_branch_id = new_branch_id
                self.stats_["detached"] = self.stats_.get("detached", 0) + len(moved)
                if self.raw_tier is not None:  # the moved commits' raw entries belong to the detached branch now
                    try:
                        if hook is not None:
                            hook("transfer")
                        self.raw_tier.transfer([entry.raw for entry in lineage.entries if entry.raw is not None],
                                               branch_id, new_branch_id)
                    except BaseException as exc:  # noqa: BLE001 - after the commit point: recorded (R2)
                        self.stats_["detach_bookkeeping_errors"] = self.stats_.get("detach_bookkeeping_errors", 0) + 1
                        try:
                            self.raw_tier.disable(f"bookkeeping error at detach: {type(exc).__name__}: {exc}")
                        except BaseException:  # noqa: BLE001 - the count above records it; never raised here
                            pass
            finally:  # the waiting publisher resumes under the outcome (detached branch or rolled back)
                lineage.detaching = False
                self._cond.notify_all()
        return {"branch": new_branch_id, "moved": moved, "cut": cut}

    def frontiers(self, branch_id: str) -> dict[str, Any]:
        with self._cond:
            lineage = self._lineages.get(branch_id)
            if lineage is None:
                head = self.runtime.lineage_head(branch_id)
                frontier = self.runtime.load_node(head).frontier
                return {"generated_frontier": frontier, "published_frontier": frontier, "published_ref": head,
                        "pending": 0, "error": None}
            return {"generated_frontier": lineage.generated_frontier,
                    "published_frontier": lineage.published_frontier, "published_ref": lineage.published_ref,
                    "pending": len(lineage.entries), "error": None if lineage.error is None else str(lineage.error)}

    def drain(self, branch_id: str, *, through: str | None = None, timeout: float | None = None) -> str:
        """Wait until every submitted chunk (or through ``through``) is published; returns the head."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            lineage = self._lineages.get(branch_id)
            if lineage is None:
                return self.runtime.lineage_head(branch_id)
            if through is not None and through not in {entry.node_id for entry in lineage.entries} | lineage.published_ids:
                raise ConfigurationError(f"{through} was not submitted to this lineage")
            while True:
                if lineage.error is not None:
                    raise raisable(lineage.error)
                pending = [entry.node_id for entry in lineage.entries]
                if not pending or (through is not None and through not in pending):
                    return lineage.published_ref
                if self._deferring_thread():
                    waited_for = list(lineage.entries)
                    if through is not None:
                        waited_for = waited_for[:pending.index(through) + 1]
                    if any(entry.stage == "queued" for entry in waited_for):
                        raise StateError("drain inside defer_gpu_work would wait for deferred fits; drain after the "
                                         "section exits")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise PendingError("archive drain timed out; the published frontier trails generation")
                with self._waiting():
                    self._cond.wait(remaining)

    def cancel(self, branch_id: str, *, timeout: float | None = None) -> str:
        """Retire every unpublished chunk of a lineage (leases, staging, staged objects); returns the head."""
        with self._cond:
            lineage = self._lineages.get(branch_id)
            if lineage is None:
                return self.runtime.lineage_head(branch_id)
            lineage.cancelled = True
            if lineage.error is None:
                lineage.error = ArchiveFailure(f"archival of lineage {branch_id} was cancelled",
                                               branch_id=branch_id, node_id="*")
            self._fail_from(lineage, None, lineage.error, cancelled=True)
            self._cond.notify_all()
        return self._wait_retired(branch_id, timeout)

    def reset(self, branch_id: str, *, timeout: float | None = None) -> str:
        """Acknowledge a failure/cancellation once every unpublished chunk is retired; returns the head."""
        return self._wait_retired(branch_id, timeout)

    def _wait_retired(self, branch_id: str, timeout: float | None) -> str:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            lineage = self._lineages.get(branch_id)
            if lineage is None:
                return self.runtime.lineage_head(branch_id)
            while any(not entry.retired for entry in lineage.entries):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise PendingError("archive work of the lineage is still retiring")
                with self._waiting():
                    self._cond.wait(remaining)
            objects, lineage.failed_objects = lineage.failed_objects, []
            head = lineage.published_ref
            del self._lineages[branch_id]
            if lineage.detached_from is not None:  # W1.3 (R5)
                detached = self._detached_by_origin.get(lineage.detached_from)
                if detached is not None:
                    detached.discard(branch_id)
                    if not detached:
                        del self._detached_by_origin[lineage.detached_from]
            with self.runtime._lock:
                self.runtime._lineage_pending.discard(branch_id)
        if objects:
            self.runtime.store.collect_garbage(candidates=objects)
        return head

    def close(self, *, drain: bool = True, timeout: float | None = None) -> None:
        """Drain (default) or cancel every lineage, then stop the fitter and encode workers."""
        if drain and self._deferring_thread():
            raise StateError("close(drain=True) inside defer_gpu_work would wait for deferred fits")
        errors = []
        for branch_id in list(self._lineages):
            try:
                if drain:
                    self.drain(branch_id, timeout=timeout)
                    self._wait_retired(branch_id, timeout)
                else:
                    self.cancel(branch_id, timeout=timeout)
            except BaseException as exc:  # noqa: BLE001 - every lineage is still retired below
                errors.append(exc)
                try:
                    self.cancel(branch_id, timeout=timeout)
                except BaseException as nested:  # noqa: BLE001
                    errors.append(nested)
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._thread.join(timeout)
        self._close_accel()
        self._executor.close(raise_errors=False)
        self._staging.close()
        if self.raw_tier is not None:
            self.raw_tier.close()
        if errors:
            raise errors[0]

    def prewarm(self, contexts: Mapping, *, encode: bool = True, max_frames: int | None = None, seed: int = 0,
                timeout: float | None = None) -> dict[str, Any]:
        """Pay the archive path's cold start before the first chunk (call at session start; blocks).

        On the fitter thread and its side stream, plans and finalizes two
        synthetic chunks of ``contexts``'s geometry (fresh, then active-basis
        paths; nothing is stored or learned) so cuBLAS/cuSOLVER handles,
        workspaces, lazily loaded kernels and allocator pools exist; then
        pre-allocates the pinned canvas staging and, with ``encode``, encodes
        the warm canvas once on an encode worker (FFmpeg/x265 binary or the NVENC
        session). Returns per-phase seconds, also kept in ``stats()["prewarm"]``.
        """
        if type(encode) is not bool:
            raise ConfigurationError("encode must be boolean")
        started = time.perf_counter()
        warm = _Warmup(dict(contexts), max_frames, seed)
        with self._cond:
            if self._closed:
                raise StateError("archive pipeline is closed")
            if self._deferring_thread():
                raise StateError("prewarm inside defer_gpu_work would wait for deferred GPU work")
            self._queue.append(warm)
            self._cond.notify_all()
        if not warm.done.wait(timeout):
            raise TimeoutError("archive pipeline warm-up did not finish")
        if warm.error is not None:
            raise raisable(warm.error)
        report = warm.result
        canvas, warm.canvas = warm.canvas, None
        if encode and canvas is not None:
            clock = time.perf_counter()
            self._executor.submit(self._encode_prewarm, canvas, threads=self.options.encode_threads,
                                  validate_decode=self.options.validate_decode, encoder=self.encoder,
                                  encoder_lock=self._encoder_lock, block=True,
                                  label="archive:prewarm-encode").result(timeout)
            report["encode_seconds"] = time.perf_counter() - clock
        report["total_seconds"] = time.perf_counter() - started
        with self._cond:
            self.stats_.setdefault("prewarm", []).append(dict(report))
        return report

    def stats(self) -> dict[str, Any]:
        with self._cond:
            stats = dict(self.stats_)
            stats["timeline"] = list(self.stats_["timeline"])
            stats["staging"] = {"peak_in_use": self._staging.peak_in_use, "peak_bytes": self._staging.peak_bytes,
                                "wait_seconds": self._staging.wait_seconds, "pinned": self._staging.pin}
            stats["executor"] = {key: value for key, value in self._executor.stats().items() if key != "recent_jobs"}
            stats["restore_yield"] = None if self._yield is None else self._yield.stats()
            stats["cpu_priority"] = self.cpu_priority
            stats["fits"] = dict(self.stats_["fits"])
            stats["ranks"] = dict(self.stats_["ranks"])
            stats["encoder"] = {**self.encoder_identity, "commits": dict(self.stats_["encoder_commits"])}
            stats["commit_to_publish"] = self._lag_summary()
            seal = getattr(self.restorer, "seal_stats", None)
            stats["adopt"] = {**self.stats_["adopt"], **(seal() if callable(seal) else {})}
            if self.package_accel is not None:  # accel C (records only)
                stats["accel_c"] = {"options": self.package_accel.to_dict(), **_accel_c.stats()}
            stats["raw_tier"] = None if self.raw_tier is None else self.raw_tier.stats()
            # Host tile workspaces of fingerprints / finalization / host projections (freeze E): process-wide bytes,
            # transient (released with each call or range); not part of any budget, recorded here (R5 spirit).
            stats["host_workspace"] = workspace_stats()
            stats["deferred"] = bool(self._deferrals)
            if self._generation is not None:  # genfirst (off: no such key, the records stay W1.4.1's)
                stats["generation_defer"] = self._generation.summary(self)
            stats["cpu_sets"] = {**self.stats_["cpu_sets"], "encoders": dict(self.stats_["cpu_sets"]["encoders"])}
            stream = self._fitter_stream
            stats["fitter_stream"] = {"mode": self.fitter_stream_mode, "note": self._fitter_stream_note,
                                      "dedicated": (isinstance(stream, torch.cuda.ExternalStream)
                                                    if stream is not None else None)}
            stats["fitter_recheck"] = {**self.stats_["fitter_recheck"], "watched_sections": len(self._watches),
                                       "events": [dict(event) for event in self._recheck_events]}
            stats["fitter_pause"] = {**self.stats_["fitter_pause"], "paused": self._fitter_paused,
                                     "open": self._pause_requests, "peak_bytes": self._fitter_peak_bytes,
                                     "peak_source_bytes": self._fitter_peak_source_bytes,
                                     "peak_basis": self._peak_basis,
                                     "warmup_peak_bytes": self._warmup_peak[0],
                                     "warmup_source_bytes": self._warmup_peak[1],
                                     "online_peak_bytes": self._online_peak[0],
                                     "online_source_bytes": self._online_peak[1], "nodes": self._fitter_nodes,
                                     "measuring": self._node_sample is not None,
                                     "blocked": bool(self._fitter_blocked), "phase": self._fitter_phase[0],
                                     "ack_timeout_phases": dict(self.stats_["fitter_pause"]["ack_timeout_phases"]),
                                     "ack_timeout_events": [dict(event) for event in self._pause_events],
                                     "step_span_buckets": list(SPAN_BUCKETS),
                                     "step_spans": {label: {**record, "buckets": list(record["buckets"])}
                                                    for label, record in self._step_spans.items()}}
            stats["lineages"] = {name: {"pending": len(item.entries), "generated_frontier": item.generated_frontier,
                                        "published_frontier": item.published_frontier}
                                 for name, item in self._lineages.items()}
            if self.fitter_accel is not None:  # accel B (opt-in; off: no such key)
                stats["accel_b"] = {**self.fitter_accel.to_dict(), "counters": _accel_b.stats(),
                                    "cuda_graphs": self._graph_stats()}
            return stats

    # -- accel B (2026-10-30, opt-in; reelcache.accel_b) -------------------------------------------------------------
    def _init_accel(self, fitter_accel: Any) -> None:
        """Validate the track B options (None: nothing changes) and prepare the green context and graph trims."""
        if fitter_accel is not None and not isinstance(fitter_accel, _accel_b.FitterAccel):
            raise ConfigurationError("fitter_accel must be a reelcache.accel_b.FitterAccel or None")
        self.fitter_accel = None if fitter_accel is None or fitter_accel.default else fitter_accel
        self._graph_caches: list = []
        self._graph_trim = None
        if self.fitter_accel is None:
            return
        if self.fitter_accel.needs_cuda and self.device.type != "cuda":
            raise ConfigurationError("the accel B fitter options (green context, CUDA graphs, finalize on the fit "
                                     "device, FP32 passes) need a CUDA fit device")
        if self.fitter_accel.green_sms is not None:
            if self.fitter_stream_mode != "dedicated":
                raise ConfigurationError("fitter_green_context_sms replaces the dedicated fitter stream: it needs "
                                         "fitter_stream='dedicated'")
            if self.stream_priority is not None:
                raise ConfigurationError("a green context's stream has the default priority: fitter_green_context_sms "
                                         "excludes stream_priority")
            _accel_b.green_stream(self.device, self.fitter_accel.green_sms, self.fitter_accel.green_workqueue)  # refused here
        if self.fitter_accel.cuda_graphs:
            from .device_pressure import register_device_trim
            self._graph_trim = register_device_trim("fitter_graphs", self.trim_graphs, device=self.device)

    def _accel_scope(self):
        if self.fitter_accel is None:
            return contextlib.nullcontext()
        from .graph_capture import graph_scope
        stack = contextlib.ExitStack()
        stack.enter_context(_accel_b.scope(self.fitter_accel, fit_device=self.device))
        stack.enter_context(graph_scope(self._thread_graph_cache("fitter")))
        return stack

    def _thread_graph_cache(self, name: str):
        """dev22l (B3): this GPU thread's CUDA-graph cache (None unless ``cuda_graphs``)."""
        if self.fitter_accel is None or not self.fitter_accel.cuda_graphs:
            return None
        from .graph_capture import GraphCache
        cache = GraphCache(self.device, budget_bytes=self.fitter_accel.graph_memory_bytes, name=name)
        with self._cond:
            self._graph_caches.append(cache)
        return cache

    def trim_graphs(self, reason: str = "trim") -> int:
        """Device trim (registered as ``fitter_graphs``): drop the fitter's captured graphs, pools and static buffers;
        they are recaptured on demand. Returns the bytes released (pools are an estimate)."""
        with self._cond:
            caches = list(self._graph_caches)
        return sum(cache.trim(reason) for cache in caches)

    def _graph_stats(self) -> dict[str, Any] | None:
        if self.fitter_accel is None or not self.fitter_accel.cuda_graphs:
            return None
        caches = list(self._graph_caches)
        per = [cache.stats() for cache in caches]
        total = {name: sum(item[name] for item in per) for name in (
            "calls", "replays", "eager_calls", "captures", "capture_failures", "capture_seconds", "budget_refusals",
            "evictions", "invalidations", "trims", "trimmed_bytes", "graphs", "pool_bytes", "buffer_bytes")}
        failures = [item["last_capture_failure"] for item in per if item.get("last_capture_failure")]
        return {"enabled": True, **total, "memory_bytes": total["pool_bytes"] + total["buffer_bytes"],
                "budget_bytes_per_thread": per[0]["budget_bytes"] if per else self.fitter_accel.graph_memory_bytes,
                "last_capture_failure": failures[-1] if failures else None}

    def _close_accel(self) -> None:
        if self._graph_trim is not None:
            self._graph_trim.unregister()
            self._graph_trim = None
        for cache in list(self._graph_caches):
            cache.close()

    def _record_publication(self, entry: _Entry, job: CommitJob | None) -> None:
        """Per-commit records at publication (caller holds the condition): commit -> publish seconds, planner
        seconds, per-matrix decisions and stored ranks, the encoder."""
        lag = entry.times["published"] - entry.submitted
        self._lags.append({"branch": entry.lineage.branch_id, "node_id": entry.node_id, "seconds": lag})
        if self._generation is not None:  # genfirst: the run's publication lags (p50 / p90)
            self._generation.lags.append(lag)
        totals = self._lag_totals
        totals["count"] += 1
        totals["sum"] += lag
        totals["max"] = max(totals["max"], lag)
        if "plan_start" in entry.times and "finalize_end" in entry.times:
            self.stats_["planner_seconds"] += entry.times["finalize_end"] - entry.times["plan_start"]
        name = self.encoder_identity["encoder"]
        self.stats_["encoder_commits"][name] = self.stats_["encoder_commits"].get(name, 0) + 1
        fits, ranks = self.stats_["fits"], self.stats_["ranks"]
        fits["nodes"] += 1
        for item in getattr(job, "ranges", ()) or ():
            for plan in getattr(item, "plans", {}).values():
                decision = getattr(plan, "decision", None)
                if decision is not None:
                    fits[decision] = fits.get(decision, 0) + 1
                    if decision in _NEW_BASIS:
                        fits["total"] += 1
                rank = getattr(plan, "rank_view", None)
                if rank is not None:
                    ranks["matrices"] += 1
                    ranks["sum"] += int(rank)
        ranks["mean"] = ranks["sum"] / ranks["matrices"] if ranks["matrices"] else None

    def _lag_summary(self) -> dict[str, Any]:
        recent = [item["seconds"] for item in self._lags]
        ordered = sorted(recent)
        totals = self._lag_totals

        def quantile(q):
            return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else None

        return {"count": totals["count"], "mean": totals["sum"] / totals["count"] if totals["count"] else None,
                "max": totals["max"] if totals["count"] else None, "last": recent[-1] if recent else None,
                "p50": quantile(0.5), "p95": quantile(0.95), "recent": [dict(item) for item in self._lags]}

    # -- failure handling ------------------------------------------------------------
    def _fail_from(self, lineage: _Lineage, entry: _Entry | None, error: BaseException, *, cancelled=False) -> None:
        """Fail ``entry`` and every later unpublished entry (caller holds the condition). The error is kept
        frame-free (lineage, tickets, jobs): its traceback would pin the failed commit's staged sources."""
        frame_free(error)
        if lineage.error is None:
            lineage.error = error
        failing = False
        for item in list(lineage.entries):
            if entry is None or item is entry:
                failing = True
            if not failing or item.stage == "published" or item.publishing:
                continue
            if item.stage not in ("failed", "cancelled"):
                cause = error if item is entry or entry is None else ArchiveFailure(
                    f"chunk {item.node_id} cannot publish: ancestor {entry.node_id} failed",
                    branch_id=lineage.branch_id, node_id=item.node_id, cause=error)
                item.stage = "cancelled" if cancelled else "failed"
                item.ticket._resolve(error=cause)
                if item.job is not None:
                    item.job.fail(cause)
                self.stats_["cancelled" if cancelled else "failed"] += 1
            if not item.busy:
                self._retire(item)
        if self.archive_budget is not None:  # W1.4.3 (E1): a failed commit waiting for budget room stops at once
            self.archive_budget.wake()

    def _retire(self, entry: _Entry) -> None:
        """Release leases, staging and staged objects of a failed entry not running anywhere."""
        if entry.retired:
            return
        entry.retired = True
        self._release_sources(entry, None)
        if entry.raw is not None:
            # Failed or cancelled: the raw copies go once no worker reads them (R7).
            self.raw_tier.retire(entry.raw, entry.stage)
        for buffer in entry.staging:
            self._staging.release(buffer)
        entry.staging.clear()
        if entry.job is not None:
            entry.job.drop_sources()  # no worker runs it any more; it may live on as the lineage's last_job
            entry.lineage.failed_objects.extend(entry.job.staged_objects)
            if self.archive_budget is not None:  # W1.4.3 (E1): an unpublished commit's reservation goes
                self.archive_budget.release(entry.job)
            try:
                entry.job.release_staging(collect=False)
            except BaseException:  # noqa: BLE001 - the lineage error already reports the failure
                pass
        self._cond.notify_all()

    def _release_sources(self, entry: _Entry, event: Any) -> None:
        if not entry.sources_held:
            return
        entry.sources_held = False
        self._sources_held -= 1
        entry.groups = [(chunk_id, {}, contexts) for chunk_id, _, contexts in entry.groups]
        if entry.raw is not None:
            self.raw_tier.fitted(entry.raw, event)
        elif entry.release is not None:
            entry.release(event)
        entry.times["sources_released"] = time.perf_counter()
        self._cond.notify_all()

    # -- fitter thread ------------------------------------------------------------------
    def _budget_published(self, job: Any, ref: str) -> None:
        """W1.4.3 (E1): a published node's payload joins the archive budget's live nodes (its reservation goes). The
        commit is durable: a bookkeeping error keeps its reservation counted and the node is registered before the
        next write is admitted (review round 2) -- never released while its payload stays in the store."""
        try:
            self.archive_budget.published(job, ref)
        except BaseException as exc:  # noqa: BLE001 - recorded; conservative accounting, reconciled later
            self.archive_budget.unreconciled(job, ref, exc)

    @staticmethod
    def _without_basis(parent):
        """W1.4.3 (E1): an evicting arm (no basis reuse) plans every commit as one with no active basis -- empty
        bindings (the refit memory kept), no rows -- so no commit depends on another node's payload, which the budget
        may evict at any time (in memory or not): every matrix is fitted fresh (recorded ``no_active_basis`` instead of
        ``reuse_disabled``) and a zero matrix fits its own one-row basis instead of reusing the parent's version."""
        bindings = {entry: replace(binding, version_ref=None, total_rank=0, group_id=None)
                    for entry, binding in parent.bindings.items()}
        return replace(parent, bindings=bindings, rows={entry: None for entry in bindings})

    def _hook(self, stage: str, entry: _Entry) -> None:
        if self.hooks is not None:
            self.hooks(stage, entry.node_id)

    def _archival_thread(self, role: str) -> None:
        """dev22j8v6 (F6 a): give the calling archival thread (``role`` ``"fitter"`` or ``"encoder"``) the pipeline's
        CPU set, once per thread, and record the mask it got. SGLang's NCCL init pins the worker to its GPU's NUMA node
        and threads inherit their creator's mask; the FFmpeg children these threads start inherit theirs."""
        if self.package_accel is not None:  # accel C: this pipeline's levers on its archival threads
            _accel_c.enter_thread(self.package_accel, pool=self._validation_pool)
        if self.cpu_set is None or getattr(self._cpu_local, "applied", False):
            return
        self._cpu_local.applied = True
        import os
        try:
            os.sched_setaffinity(0, self.cpu_set)
            applied, error = _cpu_list(os.sched_getaffinity(0)), None
        except (AttributeError, OSError, ValueError) as exc:  # the inherited mask stays
            applied, error = None, f"{type(exc).__name__}: {exc}"
        with self._cond:
            record = self.stats_["cpu_sets"]
            if error is not None:
                record["errors"] += 1
                record["last_error"] = error
            elif role == "fitter":
                record["fitter"] = applied
            else:
                record["encoders"][applied] = record["encoders"].get(applied, 0) + 1

    def _fitter_loop(self) -> None:
        self._archival_thread("fitter")
        with step_hook(self._step_boundary, blocked=self._blocked, phase=self._set_phase), self._accel_scope():
            self._fitter_rounds()

    def _fitter_rounds(self) -> None:
        if self.fitter is not None:
            self.fitter.adopt_current_thread()
        stream = None
        if self.device.type == "cuda":
            if self.fitter_accel is not None and self.fitter_accel.green_sms is not None:  # accel B1 (opt-in): green-context stream
                stream = _accel_b.green_stream(self.device, self.fitter_accel.green_sms, self.fitter_accel.green_workqueue)
                self._fitter_stream_note = f"green context: {self.fitter_accel.green_sms} SMs"
            if stream is None and self.fitter_stream_mode == "dedicated":
                try:
                    stream = _dedicated_stream(self.device, self.stream_priority)
                except Exception as exc:  # noqa: BLE001 - recorded: a pooled stream instead
                    self._fitter_stream_note = f"dedicated stream failed ({type(exc).__name__}: {exc}): pooled"
            if stream is None:
                stream = (torch.cuda.Stream(self.device) if self.stream_priority is None
                          else torch.cuda.Stream(self.device, priority=self.stream_priority))
        self._fitter_stream = stream
        while True:
            with self._cond:
                # Open memory-critical sections (defer_gpu_work) hold queued GPU work back; so do generation holds
                # (genfirst; off: None), unless the starvation guard lets a node through (re-checked by age).
                while (not self._queue or self._deferrals or self._generation_blocks()) and not self._closed:
                    self._cond.wait(None if self._generation is None else self._generation.poll(self))
                if not self._queue:
                    return
                # Watched sections (dev22j4): GPU work that would not fit above the reserve waits for them (capped).
                verdict = None if self._waiters else self._recheck_verdict(self._queue[0])
                if verdict is not None and self._recheck_wait(verdict):
                    continue
                if verdict is not None and self._generation is not None and not self._closed and self._deferrals:
                    continue  # genfirst3 (review #3): a section opened during the re-check wait
                reason = None
                if self._generation is not None and not self._closed:
                    # genfirst4 (review I3): one decision -- it permits (or holds back) the head and is what is counted
                    blocked, reason = self._generation.admission(self)
                    if blocked:
                        continue
                entry = self._queue.popleft()
                if isinstance(entry, _Warmup):
                    warm = entry
                else:
                    warm = None
                    lineage = entry.lineage
                    if entry.stage in ("failed", "cancelled"):
                        if not entry.retired:
                            self._retire(entry)
                        continue
                    if self._generation is not None:  # genfirst: a node let through a hold counts its override
                        self._generation.started(entry, reason)
                    entry.busy = True
                    entry.stage = "planning"
                self._gpu_busy = True
                self._gpu_item = entry
                if self._generation is not None:  # genfirst4 (review I1): each dispatch, a restart of one entry too
                    self._dispatch_id += 1
            if warm is not None:
                try:
                    self._set_phase("warmup")
                    self._warm(warm, stream)
                finally:
                    self._set_phase("idle")
                    with self._cond:
                        self._gpu_busy = False
                        self._gpu_item = None
                        if self._generation is not None:
                            self._generation.gpu_done(warm, time.perf_counter())
                        self._cond.notify_all()
                continue
            if self._yield is not None:
                self._yield()
                with self._cond:
                    if ((self._deferrals or self._generation_blocks()
                         or (not self._waiters and self._recheck_verdict(entry) is not None))
                            and not self._closed and entry.stage == "planning"):
                        # A memory-critical section opened while this node waited for a restore: hand it back.
                        entry.stage, entry.busy = "queued", False
                        self._queue.appendleft(entry)
                        self._gpu_busy = False
                        self._gpu_item = None
                        if self._generation is not None:
                            self._generation.gpu_done(entry, time.perf_counter())
                        self._cond.notify_all()
                        continue
            started = time.perf_counter()
            measured = self._memory_mark()
            node_bytes = _source_bytes(entry)
            self._begin_node()
            try:
                with self._ffmpeg_context():  # ranges encoded in place when staging is full
                    self._fit(entry, stream)
                failed = None
            except BaseException as exc:  # noqa: BLE001 - surfaced through the lineage
                failed = frame_free(exc)  # its traceback held _fit's frame: the staged sources (R-D2b)
            working = self._working_set_since(measured)
            self._end_node(node_bytes, failed is None)
            self._set_phase("idle")
            with self._cond:
                if working:
                    self._fitter_working_set = max(self._fitter_working_set, working)
                    self.stats_["fitter_recheck"]["working_set_bytes"] = self._fitter_working_set
                self._gpu_busy = False
                self._gpu_item = None
                if self._generation is not None:
                    self._generation.gpu_done(entry, time.perf_counter())
                self._cond.notify_all()
                self.stats_["fitter_busy_seconds"] += time.perf_counter() - started
                entry.busy = False
                if failed is not None:
                    self._fail_from(lineage, entry, ArchiveFailure(
                        f"archival of chunk {entry.node_id} failed: {type(failed).__name__}: {failed}",
                        branch_id=lineage.branch_id, node_id=entry.node_id, cause=failed))
                    self._retire(entry)
                    continue
                if entry.stage in ("failed", "cancelled"):
                    self._retire(entry)
                    continue
                entry.stage = "finalized"
                entry.busy = True
            try:
                self._executor.submit(self._encode_task, entry, block=True, label=f"archive:{entry.node_id}")
            except BaseException as exc:  # noqa: BLE001
                frame_free(exc)
                with self._cond:
                    entry.busy = False
                    self._fail_from(lineage, entry, ArchiveFailure(
                        f"archival of chunk {entry.node_id} could not be encoded: {exc}",
                        branch_id=lineage.branch_id, node_id=entry.node_id, cause=exc))
                    self._retire(entry)

    def _reserved_bytes(self) -> int | None:
        """The caching allocator's reserved bytes on this pipeline's CUDA device (None off CUDA)."""
        if self.device.type != "cuda":
            return None
        try:
            return int(torch.cuda.memory_stats(self.device).get("reserved_bytes.all.current", 0))
        except Exception:  # noqa: BLE001 - measurement only
            return None

    def _learn_fitter_peak(self, reserved_before: int | None, report: Mapping[str, Any], live: Any = None,
                           baseline: int = 0) -> None:
        """The fitter's peak working set from a warm-up (dev22j7; dev22j8: LIVE bytes). ``live`` (a :class:`_LivePeak`)
        measured the live high-water of the fitter's streams over ``warm_commit_path``, which runs before any request
        (no other GPU work), above ``baseline``: what those streams already held (dev22j8v3; the peak is absolute, as
        the online samples and the guard's ``held`` are). The warm-up's synthetic sources (``report["source_bytes"]``)
        are subtracted: in production the sources sit in the raw tier, they are no fitter growth. Kept with those
        source bytes, the scale of :meth:`inflight_growth_bound`. The allocator's reserved growth over the warm-up
        (dev22j7's measure: it also counts fragmentation and the sources) is kept as a diagnostic
        (``warmup_reserved_bytes``)."""
        after = self._reserved_bytes()
        sources = int(report.get("source_bytes") or 0)
        with self._cond:
            pause = self.stats_["fitter_pause"]
            if reserved_before is not None and after is not None:
                pause["warmup_reserved_bytes"] = max(pause["warmup_reserved_bytes"], after - reserved_before)
            if live is None or live.bytes is None:
                return
            pause["warmup_baseline_bytes"] = int(baseline)
            peak = max(0, int(baseline) + int(live.bytes) - sources)
            pause["warmup_method"] = live.method
            if peak > self._warmup_peak[0]:
                self._warmup_peak = (peak, sources)
            self._keep_peak(peak, sources, "warmup")

    def _begin_node(self) -> None:
        """dev22j8: the first ``PEAK_MEASURE_FIRST`` nodes, then every ``PEAK_MEASURE_EVERY``-th, are measured -- their
        fitter streams' live bytes sampled at step boundaries and blocked waits (:meth:`_sample_node`)."""
        self._fitter_nodes += 1
        count = self._fitter_nodes
        if count <= PEAK_MEASURE_FIRST or count % PEAK_MEASURE_EVERY == 0:
            self._node_sample = {"peak": 0, "reserved": 0, "samples": 0, "seconds": 0.0, "next": 0.0}

    def _sample_node(self) -> None:
        """One sample of a measured node (fitter thread; at most every ``PEAK_SAMPLE_INTERVAL`` seconds): what its
        streams hold live (:meth:`fitter_memory`: the raw tier's copies excluded)."""
        sample = self._node_sample
        if sample is None:
            return
        now = time.perf_counter()
        if now < sample["next"]:
            return
        memory = self.fitter_memory()
        done = time.perf_counter()
        sample["next"] = done + PEAK_SAMPLE_INTERVAL
        sample["seconds"] += done - now
        if memory is not None:
            sample["samples"] += 1
            sample["peak"] = max(sample["peak"], int(memory["held"]))
            sample["reserved"] = max(sample["reserved"], int(memory["reserved"]))

    def _end_node(self, source_bytes: int, succeeded: bool) -> None:
        """The online peak (dev22j8): a measured node's sampled live high-water, kept (the largest) with its source
        bytes; the largest fitter-stream reserved bytes seen is a diagnostic."""
        sample, self._node_sample = self._node_sample, None
        if sample is None:
            return
        with self._cond:
            pause = self.stats_["fitter_pause"]
            pause["online_samples"] += sample["samples"]
            pause["online_sample_seconds"] += sample["seconds"]
            if not succeeded or not sample["samples"]:
                return
            pause["online_measurements"] += 1
            pause["online_last_bytes"] = sample["peak"]
            pause["online_reserved_bytes"] = max(pause["online_reserved_bytes"], sample["reserved"])
            if sample["peak"] > self._online_peak[0]:
                self._online_peak = (sample["peak"], int(source_bytes))
            self._keep_peak(sample["peak"], int(source_bytes), "online")

    def _keep_peak(self, peak: int, source_bytes: int, basis: str) -> None:
        """The growth bound's measurement (caller holds the condition): the largest peak, with its source bytes."""
        if peak > self._fitter_peak_bytes:
            self._fitter_peak_bytes, self._fitter_peak_source_bytes = int(peak), int(source_bytes)
            self._peak_basis = basis

    def _memory_mark(self) -> tuple[int, int] | None:
        """The allocator's allocated and peak bytes before a node's GPU work (CUDA; never resets the peak)."""
        if self.device.type != "cuda":
            return None
        try:
            stats = torch.cuda.memory_stats(self.device)
        except Exception:  # noqa: BLE001 - measurement only
            return None
        return int(stats.get("allocated_bytes.all.current", 0)), int(stats.get("allocated_bytes.all.peak", 0))

    def _working_set_since(self, mark: tuple[int, int] | None) -> int:
        """The node's GPU working set when it set a new allocator peak: the peak minus the allocation at its start.
        This is conservative: concurrent allocations of other threads count too. Otherwise 0 (nothing learned)."""
        if mark is None:
            return 0
        try:
            peak = int(torch.cuda.memory_stats(self.device).get("allocated_bytes.all.peak", 0))
        except Exception:  # noqa: BLE001 - measurement only
            return 0
        return peak - mark[0] if peak > mark[1] else 0

    def _warm(self, warm: _Warmup, stream) -> None:
        try:
            context = torch.cuda.device(self.device) if stream is not None else _Null()
            reserved = self._reserved_bytes()
            live = None
            baseline = 0
            if stream is not None:
                # dev22j8v3: what the fitter's streams already hold (a pooled CUDA stream may carry an earlier user's
                # live blocks, e.g. a cuBLAS workspace): the peak is absolute, like ``fitter_memory()["held"]``
                baseline = int((self.fitter_memory() or {}).get("held", 0))
                live = _LivePeak(self.device, {item.cuda_stream for item in (stream, getattr(self.fitter, "_stream",
                                                                                                None))
                                               if item is not None})
            with context, (live if live is not None else _Null()):
                report = warm_commit_path(self.method, self.registry, warm.contexts, options=self.options,
                                          device=self.device, fitter=self.fitter, stream=stream,
                                          max_frames=warm.max_frames, seed=warm.seed)
            self._learn_fitter_peak(reserved, report, live, baseline)
            if live is not None:
                report["fitter_live_peak"] = {"bytes": live.bytes, "method": live.method}
            warm.canvas = report.pop("canvas")
            clock = time.perf_counter()
            staged = []
            if warm.canvas is not None and self._staging.pin:
                with self._staging._cond:
                    idle = self._staging._in_use == 0
                if idle:
                    for _ in range(self._staging.count):
                        staged.append(self._staging.acquire(warm.canvas.nbytes, timeout=0))
                    for buffer in staged:
                        self._staging.release(buffer)
            report["pinned_staging_buffers"] = len(staged)
            report["staging_seconds"] = time.perf_counter() - clock
            warm.result = report
        except BaseException as exc:  # noqa: BLE001 - re-raised by prewarm()
            warm.error = frame_free(exc)  # its traceback would pin the synthetic sources
        finally:
            warm.done.set()

    def _fit(self, entry: _Entry, stream) -> None:
        lineage = entry.lineage
        if entry.raw is not None:
            # The planner reads every source on its capture device, wherever the raw copy is (A1): a demoted copy
            # is uploaded on the planner's stream.
            self._set_phase("stage_raw")
            context = (torch.cuda.device(self.device) if stream is not None else _Null())
            with context, (torch.cuda.stream(stream) if stream is not None else _Null()):
                staged = self.raw_tier.stage_for_plan(entry.raw, stream)
            for group, tensors in zip(entry.raw.groups, staged):
                for key, tensor in tensors.items():
                    if tensor.device != group.meta[key][0]:
                        raise IntegrityError(f"raw source {key} of {entry.node_id} was staged on {tensor.device}, "
                                             f"not its capture device {group.meta[key][0]}")
            entry.groups = [(chunk_id, tensors, contexts)
                            for (chunk_id, _, contexts), tensors in zip(entry.groups, staged)]
        with self._cond:
            if lineage.last_job is not None:
                parent = lineage.last_job.child_state()
            else:
                parent = lineage.published_state
        if self.archive_budget is not None and not self.method.reuse:  # W1.4.3 (E1)
            parent = self._without_basis(parent)
        if parent.rows is None:
            self._set_phase("parent_rows")
            parent = self.runtime.assemble_parent_rows(parent)
            with self._cond:
                if lineage.published_state is not None and lineage.published_state.rows is None:
                    lineage.published_state = parent
        job = CommitJob(self.runtime, entry.kind, entry.node_id, parent, entry.groups, method=self.method,
                        options=self.options, epoch_id=entry.epoch_id, details=entry.details,
                        execution_state=entry.execution_state)
        if self.archive_budget is not None:  # W1.4.3 (E1)
            job.budget_order = (id(lineage), entry.ticket.sequence)  # its lineage's publication order (deadlocks)
            job.archive_write = functools.partial(self.archive_budget.writing, job)
        entry.job = job
        with self._cond:
            lineage.last_job = job
        self._set_phase("plan")
        self._hook("plan", entry)
        entry.times["plan_start"] = time.perf_counter()
        context = (torch.cuda.device(self.device) if stream is not None else _Null())
        with context, (torch.cuda.stream(stream) if stream is not None else _Null()):
            if stream is not None:
                if entry.ready_event is not None:
                    stream.wait_event(entry.ready_event)
                for _, matrices, _ in entry.groups:
                    for matrix in matrices.values():
                        if matrix.device.type == "cuda":
                            matrix.record_stream(stream)
            if self.fitter is not None:
                job.plan(self.registry, fitter=self.fitter)
            else:
                job.plan(self.registry, device=self.device)
            entry.times["plan_end"] = time.perf_counter()
            self._check_live(entry)
            self._set_phase("finalize")
            self._hook("finalize", entry)

            def staging(nbytes):
                # A node may have more ranges (e.g. a recache epoch) than staging slots: its ranges stream through
                # them. When the pool is exhausted while this node holds staged canvases, the fitter encodes the
                # oldest one in place (ranges encode independently: same bytes) and frees its slot; otherwise it
                # waits for other nodes' encodes as usual. Never a wait on slots only this node could free.
                # dev22j8: both waits allocate no device memory (a pause lands; their ends are step boundaries).
                while True:
                    try:
                        with self._blocked("staging_wait"):
                            buffer = self._staging.acquire(nbytes, timeout=0 if entry.staged_ranges else None)
                        break
                    except CapacityError:
                        if not entry.staged_ranges:
                            raise
                        index, held = entry.staged_ranges.pop(0)
                        with self._blocked("inplace_encode"):
                            job.encode_range(index, encoder=self.encoder, encoder_lock=self._encoder_lock)
                        with self._cond:
                            entry.staging = [item for item in entry.staging if item is not held]
                            self._staging.release(held)
                            self.stats_["streamed_ranges"] += 1
                with self._cond:
                    entry.staging.append(buffer)
                entry.staged_ranges.append((entry.next_range, buffer))
                entry.next_range += 1
                return buffer

            # accel B5 (opt-in): host sources are finalized on the CUDA fit device
            fit_device = ({"device": self.device} if self.fitter_accel is not None and self.fitter_accel.finalize_on_fit_device
                          and self.device.type == "cuda" else {})
            job.finalize(staging=staging, stream=stream, **fit_device)
        # No host wait here: the producer's stream waits on this event before it overwrites the sources,
        # and the encode worker waits on the canvas copy's own event.
        event = job.sources_released_event
        entry.times["finalize_end"] = time.perf_counter()
        with self._cond:
            self._release_sources(entry, event)

    def _seal_adopt(self, packet_ref: str, encoded) -> None:
        """Seal-time preparation of a stored pack (``restorer.adopt``). It only speeds up later restores, so a failure
        other than an integrity error is recorded (``stats()["adopt"]``) and never fails the commit (freeze D, R2)."""
        with self._cond:
            self.stats_["adopt"]["calls"] += 1
        try:
            self.restorer.adopt(packet_ref, encoded)
        except IntegrityError:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded; the node is durable and restorable
            with self._cond:
                self.stats_["adopt"]["failures"] += 1
                self.stats_["adopt"]["last_failure"] = f"{type(exc).__name__}: {str(exc).strip()[:300]}"

    def _check_live(self, entry: _Entry) -> None:
        with self._cond:
            if entry.stage in ("failed", "cancelled"):
                raise ArchiveFailure(f"chunk {entry.node_id} was retired while it ran",
                                     branch_id=entry.lineage.branch_id, node_id=entry.node_id)

    # -- encode workers and the ordered publisher ---------------------------------------
    @contextlib.contextmanager
    def _ffmpeg_context(self):
        """This pipeline's FFmpeg settings for the calling worker: CPU priority and, with ``pause_ffmpeg``, the pump's
        restore yield (one shared expiry per worker thread and its pump threads)."""
        pause = None
        if self.pause_ffmpeg and self._yield is not None:
            pause = getattr(self._pump_hooks, "hook", None)
            if pause is None:
                pause = self._pump_hooks.hook = self._yield.shared()
        with _hevc.ffmpeg_priority(self.cpu_priority), _hevc.ffmpeg_pump_pause(pause):
            yield

    @contextlib.contextmanager
    def _waiting(self):
        """A caller blocks on archival progress: archival workers stop yielding to restores until it returns (a
        restore that waits for archival inside its bracket is never held by its own pause)."""
        with self._waiters_lock:
            self._waiters += 1
        if self._watches or self._pause_requests or self._generation is not None:
            # a fitter held back by a re-check (dev22j4), paused (dev22j7) or held by a generation hold (genfirst3,
            # review #10: the waiter rule lets it through now, not at its next age poll) continues now
            with self._cond:
                self._cond.notify_all()
        try:
            yield
        finally:
            with self._waiters_lock:
                self._waiters -= 1

    def _encode_task(self, entry: _Entry) -> None:
        self._archival_thread("encoder")
        with self._ffmpeg_context():
            self._encode_node(entry)

    def _encode_prewarm(self, canvas, **kwargs):
        from .node_packets import encode_node_pack
        self._archival_thread("encoder")
        return encode_node_pack(canvas, **kwargs)

    def _encode_node(self, entry: _Entry) -> None:
        lineage = entry.lineage
        failed = None
        try:
            self._check_live(entry)
            if self._yield is not None:
                self._yield()
            self._hook("encode", entry)
            entry.times["encode_start"] = time.perf_counter()
            entry.job.encode(encoder=self.encoder, encoder_lock=self._encoder_lock)
            entry.times["encode_end"] = time.perf_counter()
            with self._cond:
                for buffer in entry.staging:
                    self._staging.release(buffer)
                entry.staging.clear()
            self._check_live(entry)
            if self._yield is not None:
                self._yield()
            self._hook("store", entry)
            accel = self.package_accel
            if self.restorer is not None and accel is not None and accel.seal_prepare_after_publish:
                entry.sealed = []  # accel C: seal-time preparation after this node's publication (_publish_ready)
                entry.job.store(adopt=lambda ref, encoded, sealed=entry.sealed: sealed.append((ref, encoded)))
            else:
                entry.job.store(adopt=None if self.restorer is None else self._seal_adopt)
            entry.times["store_end"] = time.perf_counter()
        except BaseException as exc:  # noqa: BLE001 - surfaced through the lineage
            failed = frame_free(exc)
        with self._cond:
            entry.busy = False
            if failed is not None:
                if entry.stage not in ("failed", "cancelled"):
                    self._fail_from(lineage, entry, ArchiveFailure(
                        f"archival of chunk {entry.node_id} failed: {type(failed).__name__}: {failed}",
                        branch_id=lineage.branch_id, node_id=entry.node_id, cause=failed))
                self._retire(entry)
                return
            if entry.stage in ("failed", "cancelled"):
                self._retire(entry)
                return
            entry.stage = "stored"
        if self._yield is not None:
            self._yield()
        self._publish_ready(lineage)
        if self.package_accel is not None:
            self._seal_published()

    def _seal_published(self) -> None:
        """accel C (``seal_prepare_after_publish``): the seal-time preparation of the nodes this thread just published
        (each node's packs, in publication order). It only warms restore caches: any failure is recorded."""
        pending = getattr(self._sealing, "pending", None)
        while pending:
            packet_ref, encoded = pending.pop(0)
            started = time.perf_counter()
            try:
                self._seal_adopt(packet_ref, encoded)
            except Exception as exc:  # noqa: BLE001 - the node is published and durable: record only
                _accel_c._count("seal_after_publish_failures")
                with self._cond:
                    self.stats_["adopt"]["failures"] += 1
                    self.stats_["adopt"]["last_failure"] = f"after publish: {type(exc).__name__}: {str(exc)[:300]}"
            _accel_c._count("seal_after_publish")
            _accel_c._count("seal_after_publish_seconds", time.perf_counter() - started)

    def _publish_ready(self, lineage: _Lineage) -> None:
        with lineage.publish_lock:
            while True:
                with self._cond:
                    # W1.3 (F5, R1): a detach goes first -- it waits for at most this one transaction -- and the
                    # publisher resumes under its outcome (the detached branch, or the rolled-back one)
                    while lineage.detaching:
                        self._cond.wait()
                    # Ancestors of a failed chunk still publish; the failed chunk and its descendants never do.
                    if not lineage.entries:
                        return
                    entry = lineage.entries[0]
                    if entry.stage != "stored":
                        return
                    entry.busy = True
                    entry.publishing = True
                    expected = lineage.published_ref
                    branch = lineage.branch_id  # W1.3: a detached lineage publishes to its own ref
                try:
                    self._hook("publish", entry)
                    ref = self.runtime.publish_job(entry.job, expected, branch_id=branch)
                    if self.archive_budget is not None:  # W1.4.3 (E1): after the durable commit; never fails it
                        self._budget_published(entry.job, ref)
                    applied = self.registry.observe(entry.job.observations)
                    entry.job.release_staging(collect=False)
                    failed = None
                except BaseException as exc:  # noqa: BLE001
                    failed = frame_free(exc)
                with self._cond:
                    entry.busy = False
                    entry.publishing = False
                    if failed is not None:
                        self._fail_from(lineage, entry, ArchiveFailure(
                            f"publication of chunk {entry.node_id} failed: {type(failed).__name__}: {failed}",
                            branch_id=lineage.branch_id, node_id=entry.node_id, cause=failed))
                        self._retire(entry)
                        return
                    entry.times["published"] = time.perf_counter()
                    entry.stage = "published"
                    entry.retired = True
                    lineage.entries.popleft()
                    job = entry.job
                    job.observed = True
                    lineage.published_ref = ref
                    lineage.published_frontier = job.node.frontier
                    lineage.published_ids.add(entry.node_id)
                    if entry.raw is not None:
                        # After the store's compare-and-swap: bookkeeping must not fail the durable commit (R2).
                        try:
                            self.raw_tier.published(entry.raw, ref)
                        except BaseException as exc:  # noqa: BLE001 - recorded; the commit is durable
                            self.raw_tier.disable(f"bookkeeping error at publication: {type(exc).__name__}: {exc}")
                    try:
                        self._record_publication(entry, entry.job)
                    except BaseException:  # noqa: BLE001 - records never fail a publication
                        pass
                    entry.ticket.encoder = self.encoder_identity["encoder"]
                    entry.ticket._resolve(node_ref=ref)
                    sealed = getattr(entry, "sealed", None)
                    if sealed:  # accel C: run by this thread after _publish_ready (_seal_published)
                        if getattr(self._sealing, "pending", None) is None:
                            self._sealing.pending = []
                        self._sealing.pending.extend(sealed)
                        entry.sealed = None
                    self.stats_["published"] += 1
                    self.stats_["timeline"].append({
                        "branch": lineage.branch_id, "node_id": entry.node_id,
                        **{name: value - entry.submitted for name, value in entry.times.items()},
                        "stage_seconds": dict(job.stage_seconds), "observations_applied": sum(applied),
                    })
                    # Bound memory: a published job keeps only what its children still read.
                    job.measured_rates()
                    if lineage.last_job is job:
                        state = job.child_state()
                        lineage.published_state = replace(
                            state, head_ref=ref, job=None, pending_observations=(),
                            bindings=dict(job.published_bindings),
                            rate_sources=tuple(source.measured_rates() if isinstance(source, CommitJob) else source
                                               for source in state.rate_sources))
                        lineage.last_job = None
                    job.parent = None
                    entry.job = None
                    self._cond.notify_all()


_DECISIONS = ("fresh", "share", "augment", "refit", "zero_input", "diagnostic_full_width")
#: Decisions that fit a new basis (the refit ablation's "fits").
_NEW_BASIS = ("fresh", "refit", "diagnostic_full_width")


def _commit_bytes(groups) -> int:
    return sum(tensor.numel() * tensor.element_size() for _, matrices, _ in groups for tensor in matrices.values())


def _source_bytes(item: Any) -> int:
    """A queued or running node's source bytes (dev22j8): its raw tier entry's bytes (the entry's groups hold no
    tensors while the tier keeps them), else its groups' tensors."""
    raw = getattr(item, "raw", None)
    if raw is not None:
        return int(raw.nbytes)
    return _commit_bytes(getattr(item, "groups", None) or ())


def _token_start(contexts: Mapping) -> int | None:
    for context in contexts.values():
        return getattr(context, "token_start", None)
    return None


def _encoder_identity(encoder, method: RankMethodConfig) -> dict[str, Any]:
    """The resolved encoder of every node this pipeline publishes (records; packets carry the full identity)."""
    codec = method.codec
    if codec.kind == "svd":  # wl-base (W1c): FP16 factor packs are stored without an encoder
        return {"encoder": "none (fp16 factors)", "preset": None, "qp": None, "driver": None}
    if encoder is None:
        return {"encoder": "libx265", "preset": codec.preset, "qp": codec.qp, "driver": None}
    implementation = getattr(encoder, "implementation", {}) or {}
    settings = getattr(encoder, "settings", None)
    return {"encoder": getattr(encoder, "name", "nvenc"), "preset": codec.preset,
            "qp": getattr(settings, "qp", codec.qp), "driver": implementation.get("driver_version"),
            "gpu": implementation.get("gpu")}


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class RawArchive:
    """Exact RAW archival of complete clean chunks: the lifecycle identity reference (not compressed).

    The producer interface of :class:`ArchivePipeline` (``submit``, ``drain``,
    ``frontiers``, ``cancel``, ``reset``, ``close``, ``stats``,
    ``max_source_chunks``) over
    :meth:`reelcache.lineage.LineageRuntimeMixin.commit_raw_node`: each submitted
    chunk's captured pre-RoPE K and V (and execution state) are committed
    bit for bit before ``submit`` returns (a synchronous host copy after the
    producer's ``ready_event``), so tickets are already published and
    ``drain`` never waits. Restores go through
    :class:`reelcache.archive_restore.RawLineageRestorer` with the engine's own
    RoPE. RAW-identity tests use it; it is not an archival performance path.
    """

    def __init__(self, runtime, *, max_source_chunks: int = 2):
        if type(max_source_chunks) is not int or max_source_chunks < 1:
            raise ConfigurationError("max_source_chunks must be a positive integer")
        self.runtime, self.max_source_chunks = runtime, max_source_chunks
        self._sequence: dict[str, int] = {}
        self._closed = False
        self._lock = threading.Lock()
        self.stats_ = {"submitted": 0, "published": 0, "commit_seconds": 0.0, "payload_bytes": 0}

    def submit(self, branch_id: str, node_id: str, matrices: Mapping, contexts: Mapping, *, kind: str = "chunk",
               ranges=None, epoch_id=None, details=None, execution_state=None, ready_event=None, release=None,
               timeout=...) -> ArchiveTicket:
        if self._closed:
            raise StateError("RAW archive is closed")
        if kind != "chunk" or ranges is not None:
            raise UnsupportedError("the RAW identity archive commits ordinary chunks only")
        if ready_event is not None:
            ready_event.synchronize()
        started = time.perf_counter()
        with self._lock:
            commit = self.runtime.commit_raw_node(branch_id, node_id, matrices, contexts,
                                                  execution_state=execution_state)
            sequence = self._sequence.get(branch_id, 0) + 1
            self._sequence[branch_id] = sequence
            self.stats_["submitted"] += 1
            self.stats_["published"] += 1
            self.stats_["commit_seconds"] += time.perf_counter() - started
            self.stats_["payload_bytes"] += commit.report["payload_bytes"]
        if release is not None:
            release(None)
        ticket = ArchiveTicket(branch_id, node_id, sequence, commit.node.frontier)
        ticket._resolve(node_ref=commit.node_ref)
        return ticket

    def drain(self, branch_id: str, *, through: str | None = None, timeout: float | None = None) -> str:
        return self.runtime.lineage_head(branch_id)

    def frontiers(self, branch_id: str) -> dict[str, Any]:
        head = self.runtime.lineage_head(branch_id)
        frontier = self.runtime.load_node(head).frontier
        return {"generated_frontier": frontier, "published_frontier": frontier, "published_ref": head,
                "pending": 0, "error": None}

    def cancel(self, branch_id: str, *, timeout: float | None = None) -> str:
        return self.runtime.lineage_head(branch_id)

    def reset(self, branch_id: str, *, timeout: float | None = None) -> str:
        return self.runtime.lineage_head(branch_id)

    def close(self, *, drain: bool = True, timeout: float | None = None) -> None:
        self._closed = True

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {**self.stats_, "representation": "raw", "timeline": []}
