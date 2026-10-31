"""Workload-v2 telemetry of the SGLang bridge (CONTRACT §3, "ReelCache bridge record"; freeze W1a, agent wl-w1a).

Opt-in: :class:`~.config.ReelCacheRuntimeConfig` ``workload_telemetry`` (default False). Off, no object of this module
exists for a request, no CUDA event is created and every record is F6's. On, each session request
(``_BridgeRun.workload``) collects, without any new synchronization on the request path:

* **spans**: ``{"name", "phase", "host_start_ns", "host_end_ns", "gpu_ms", "stream"}``. Host times are
  ``time.monotonic_ns()`` (CLOCK_MONOTONIC: comparable across the processes of a node; ``time.perf_counter`` is the
  same clock on Linux). ``gpu_ms`` is the elapsed time of a CUDA timing-event pair recorded on the span's stream
  (``stream``: ``compute`` = the request's stream, ``restore`` / ``raw`` = the overlapped restore's compressed / RAW
  install streams), i.e. from the completion of the work queued before the span to the completion of the span's own
  work; None for host-only spans (``stream`` ``host``) and off CUDA. Names are the closed list :data:`SPAN_NAMES`.
* **exposed stalls**: a CUDA event pair on the CONSUMING stream around every readiness wait (the per-layer waits of
  a layer-overlapped restore, :meth:`reelcache.overlapped_restore.LayerReadiness.wait`, and the run's join of it): the
  first event right before the (host) wait, the second right after the stream's device waits were enqueued.
  elapsed = max(0, wait - work already queued) = the GPU idle time the wait caused. ``kind``: ``host_wait`` when the
  layer was not yet issued (the host blocked), else ``device_wait``. Off CUDA the host wait time stands in (CPU
  timers, ``clock: "host"``).
* **phase** (``prep`` | ``generation``): ``generation`` once the first denoise forward of the first NEW block has
  started. The ReelCache profile stages say so (:meth:`WorkloadTelemetry.mark_generation`, right before that forward);
  otherwise the bridge infers it from the first call SGLang makes for a new block: a readiness wait outside a
  recache forward, the clean forward of a chunk (``context_forward``) or its boundary (``end_chunk``). A wait inside
  a LongLive 1.0 recache forward (a resume at a prompt switch) is ``prep``.
* **hidden restore work**: the overlapped restore's source streams record their active window (first restore work
  to last layer issued, then completed); ``hidden_rebuild_gpu_ms`` = max(0, union of the windows - exposed stall).
* ``prompt_transition`` (``kind`` ``none|switch|recache|shot``, ``gpu_ms`` / ``host_ms`` summed over the
  ``prompt_transition`` spans), ``tiers`` (the restore's tier records, as ``restore.lease.tiers``) and ``bytes``.

Span nesting (so sums never double count):

* ``restore_sync`` and ``restore_sync.<tier>`` PARTITION the synchronous (critical-path) restore: the bridge opens a
  :class:`Segments` window; the archive switches it to ``restore_sync.<tier>`` around its tier work and back. Their
  sum is the synchronous restore (``kv_rebuild_exposed`` sums ``restore_sync*``). The recompute arm's restore (its
  raw-tier install, or the replay's guard setup) is such a window too. Tier vocabulary (W1.1): a raw_offload install
  is named by its slowest range (:func:`offload_tier_label`): ``nvme`` = store reads under ``store_io="nocache"``
  (pages dropped: cold by construction), ``page_cache`` = wl-w1b's ``store`` label, store reads under
  ``store_io="buffered"`` (served from the page cache when resident; not measured), then ``host_raw`` / ``gpu_raw``;
  the restore records (``tiers``) keep wl-w1b's labels. A RAW reference archive's store reads stay ``restore_sync``.
* ``drain`` (W1.1): a fork's source drain and a rewind's (tree mode: switch) drain, in the controller step of
  ``begin_request`` (:func:`request_scope` / :func:`drain_span`), before the run's own restore: never inside
  ``publish_wait`` or ``restore_sync*``. The rest of the controller step (head switch, fork bookkeeping) is not a
  span (``begin.controller_seconds`` keeps the whole step).
* ``recompute_replay`` (W1.1): the recompute arm's replay, from ``run.begin_replay`` to ``run.end_replay`` (the
  stage's uncaptured clean forwards), after the restore window closed and before the first new denoise: ``prep``.
* ``session_open`` (W1.2; added to the closed list): a closed session's reopen in the controller step of
  ``begin_request`` -- manifest load, the tree registry, the prewarm of its prepared packs / resident bases
  (``config.prewarm``) -- for the resumed session or a fork / rewind's session, before its ``drain``: ``prep``,
  never inside another summed span (``kv_rebuild_exposed`` sums it: KV-state preparation). ``restore.session_open``
  records the same interval. An explicit ``open`` control outside a request records nothing.
* ``decode`` / ``install`` are NESTED step breakdowns of RAW installs (:class:`~.restore.DenseInstall`, per layer:
  host-to-device copies, then RoPE straight into the slot plus the V copies); ``rope`` is fused into ``install``
  (F6 rotates into the slot) and ``reconstruct`` into the compressed restorer's fused kernel, whose step breakdown is
  its own report (``restore.lease.restorer[*].breakdown``); neither is emitted. Overlapped installs record them on
  their own stream (``raw``): hidden work, never part of the critical path.
* ``recache`` (the LongLive 1.0 recache forward) and the recache epoch's ``archival_capture_sync`` are NESTED inside
  the stage's ``prompt_transition`` span.
* ``promotion_copy`` / ``promotion_fence`` (wl-w1b7 / W1.4, ``raw_tier_promote="host"``; added to the closed list):
  ``promotion_copy`` is a restore promotion's copy into pinned host raw, HIDDEN side work that is never part of the
  critical path. The recompute arm's device-to-host copies after its replay are GPU-timed on the tier's copy stream
  (``stream`` ``promotion``). raw_offload's host-to-host copy of its store reads runs on the tier's promotion worker
  and is host-timed (``stream`` ``host``); it is recorded when the copy ends, so a record resolved before that has
  none (``raw_tier.promotion.host_copy_seconds`` counts every copy). ``promotion_fence`` is the next forward's
  per-layer wait for that layer's promotion copy on rolling (sink / window) caches, NESTED in that forward; the same
  wait is in ``stalls`` (``source`` ``promotion_fence``) and so in ``exposed_stall``: never add both.
* ``promotion_wait`` (W1.4.1; added to the closed list): before a device-wide synchronize of the stage (its pre- /
  post-chunk and recache syncs), a replay-promotion copy still running is waited for explicitly on the host; the
  wait is an exposed stall (``source`` ``promotion_wait``, kind ``host_wait``; counted once, in ``exposed_stall``)
  and, for information, a ``promotion_wait`` span. Stall sources (closed): ``layer``, ``join``, ``promotion_fence``,
  ``promotion_wait``, ``archival_defer_wait``.
* ``archival_defer_wait`` (Step 7 genfirst, ``defer_archival_during_generation``; added to the closed source list):
  opening a request's generation hold (its restore returned, or its VAE decode seam opens one) waits for the archive
  node already on the GPU; that wait is an exposed stall (``source`` ``archival_defer_wait``, kind ``host_wait``;
  counted once, in ``exposed_stall``; no span). Nothing is recorded without such a node.

Records are resolved (:meth:`WorkloadTelemetry.resolve`) only after the request's outputs are complete: the bridge's
``request_summary`` / ``workload_record`` fill ``summary["workload"]`` (None until then), as the clean-forward
timings already are. ``elapsed_time`` of completed events then adds no synchronization.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from typing import Any, Mapping

import torch

SCHEMA = "reelcache.sglang.workload_telemetry.v1"
#: Restore tiers of ``restore_sync.<tier>`` (CONTRACT §3).
RESTORE_TIERS = ("gpu_raw", "host_raw", "page_cache", "nvme", "compressed_prepared", "compressed_cold")
#: The closed span-name list of CONTRACT §3.
SPAN_NAMES = frozenset({
    "restore_plan", "restore_sync", *(f"restore_sync.{tier}" for tier in RESTORE_TIERS), "decode", "reconstruct",
    "rope", "install", "exec_state", "publish_wait", "drain", "recompute_replay", "session_open",
    "prompt_transition", "recache",
    "context_forward", "archival_capture_sync", "guard_wait", "promotion_copy", "promotion_fence",
    "promotion_wait"})
PHASES = ("prep", "generation")
STALL_KINDS = ("device_wait", "host_wait")
PROMPT_TRANSITIONS = ("none", "switch", "recache", "shot")
#: The most significant transition of a request wins (a LongLive 1.0 switch both switches and recaches).
_TRANSITION_RANK = {"none": 0, "switch": 1, "shot": 2, "recache": 3}

_ACTIVE: contextvars.ContextVar = contextvars.ContextVar("reelcache_workload_restore_segments", default=None)
#: W1.1: the request whose controller step (``begin_request``: fork / rewind drains) runs in this context.
_REQUEST: contextvars.ContextVar = contextvars.ContextVar("reelcache_workload_request", default=None)
#: W1.1: restore-record tier labels -> the telemetry vocabulary (raw_offload's buffered store reads, wl-w1b ``store``).
TIER_ALIASES = {"store": "page_cache"}
#: W1.1: a mixed raw_offload install is named by its slowest (copy-bound) tier.
_TIER_ORDER = ("nvme", "page_cache", "host_raw", "gpu_raw")
#: W1.1: tiers whose ranges reach the device by a host-to-device copy (``bytes.h2d``).
_H2D_TIERS = ("host_raw", "nvme", "store", "page_cache")


def _check(name: str) -> None:
    if name not in SPAN_NAMES:
        raise ValueError(f"unknown workload span name {name!r} (closed list: {sorted(SPAN_NAMES)})")


class _Event:
    """A CUDA timing event recorded on ``stream`` (None: the current stream)."""

    __slots__ = ()

    @staticmethod
    def record(stream: Any) -> Any:
        event = torch.cuda.Event(enable_timing=True)
        event.record(stream)
        return event


def _elapsed_ms(start: Any, end: Any) -> float | None:
    if start is None or end is None:
        return None
    try:
        end.synchronize()  # resolved after the request: completed already, so no wait
        return float(start.elapsed_time(end))
    except RuntimeError:  # an event that never completed (a failed request): no GPU time
        return None


class WorkloadTelemetry:
    """One session request's workload telemetry (see the module docstring)."""

    def __init__(self, device: Any):
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.clock = "cuda_events" if self.cuda else "host"
        self._lock = threading.Lock()
        self._spans: list[list] = []  # [name, phase, start_ns, end_ns, event0, event1, stream]
        self._stalls: list[list] = []  # [kind, phase, start_ns, end_ns, event0, event1, layer, source]
        self._windows: list[list] = []  # [source, start_ns, end_ns, event0, event1]
        self.compute_stream: Any = None
        self.generation_started_ns: int | None = None
        self.generation_rule: str | None = None
        self._recache_open = 0
        self._transition = "none"
        self._forwards: dict[tuple, list] = {}
        self._open_segments: list = []
        self._last_shot: int | None = None
        self._window_cache: tuple[int, float] | None = None
        self.tiers: Any = None
        self.bytes = {"read": 0, "decoded": 0, "h2d": 0}
        self.events = 0  # CUDA events created (diagnostics: telemetry off creates none)

    # -- clocks -----------------------------------------------------------------------------------------------
    def _record(self, stream: Any = None) -> Any:
        if not self.cuda:
            return None
        self.events += 1
        return _Event.record(stream)

    def stream_label(self, stream: Any = None) -> str:
        """``compute`` for the request's stream (the bridge thread's current stream when the restore started), else
        ``raw`` (a side stream: the overlapped RAW install)."""
        if not self.cuda:
            return "host"
        if stream is None:
            stream = torch.cuda.current_stream(self.device)
        compute = self.compute_stream
        return "compute" if compute is None or stream == compute else "raw"

    @property
    def phase(self) -> str:
        return "prep" if self.generation_started_ns is None else "generation"

    # -- phase --------------------------------------------------------------------------------------------------
    def mark_generation(self, rule: str = "stage") -> None:
        """The first denoise forward of the first new block starts now (first call wins)."""
        if self.generation_started_ns is None:
            self.generation_started_ns = time.monotonic_ns()
            self.generation_rule = rule

    def capture_opened(self, kind: str) -> None:
        if kind == "kv_recache":
            self._recache_open += 1
        else:
            self.mark_generation("clean_forward")

    def capture_closed(self, kind: str) -> None:
        if kind == "kv_recache":
            self._recache_open = max(0, self._recache_open - 1)

    def prompt_transition(self, kind: str) -> None:
        """A prompt transition of this request (``switch``, ``recache``, ``shot``); the most significant is kept."""
        if kind not in PROMPT_TRANSITIONS:
            raise ValueError(f"prompt transition must be one of {PROMPT_TRANSITIONS}")
        if _TRANSITION_RANK[kind] > _TRANSITION_RANK[self._transition]:
            self._transition = kind

    # -- spans --------------------------------------------------------------------------------------------------
    def begin(self, name: str, *, stream: Any = None, gpu: bool = True) -> list:
        """Open a span (end it with :meth:`end`); ``gpu=False``: a host-only span."""
        _check(name)
        start = time.monotonic_ns()
        event = self._record(stream) if gpu else None
        return [name, self.phase, start, None, event, None, self.stream_label(stream) if event is not None else "host",
                stream]

    def end(self, token: list | None) -> None:
        if token is None or token[3] is not None:
            return
        if token[4] is not None:
            token[5] = self._record(token[7])
        token[3] = time.monotonic_ns()
        with self._lock:
            self._spans.append(token[:7])

    def next(self, token: list | None, name: str) -> list | None:
        """End ``token``'s span and open ``name`` at the same instant (one shared boundary event); None stays None."""
        _check(name)
        if token is None or token[3] is not None:
            return None
        boundary = self._record(token[7]) if token[4] is not None else None
        now = time.monotonic_ns()
        token[3], token[5] = now, boundary
        with self._lock:
            self._spans.append(token[:7])
        return [name, self.phase, now, None, boundary, None, token[6], token[7]]

    @contextlib.contextmanager
    def span(self, name: str, *, stream: Any = None, gpu: bool = True):
        token = self.begin(name, stream=stream, gpu=gpu)
        try:
            yield token
        finally:
            self.end(token)

    def add_host_span(self, name: str, start_ns: int, end_ns: int, phase: str | None = None) -> None:
        _check(name)
        with self._lock:
            self._spans.append([name, phase or self.phase, int(start_ns), int(end_ns), None, None, "host"])

    def add_event_span(self, name: str, start_event: Any, end_event: Any, start_ns: int, end_ns: int, *,
                       stream: str, phase: str | None = None) -> None:
        """A span timed by CUDA timing events another component recorded (wl-w1b7: the raw tier's copy stream);
        ``start_ns`` / ``end_ns``: the host interval that issued its work; ``stream``: its label."""
        _check(name)
        cuda = self.cuda and start_event is not None and end_event is not None
        with self._lock:
            self._spans.append([name, phase or self.phase, int(start_ns), int(end_ns),
                                start_event if cuda else None, end_event if cuda else None,
                                stream if cuda else "host"])

    def segments(self, name: str = "restore_sync") -> Segments:
        """Open a partitioned window (:class:`Segments`) and make it the active one of this thread's context (the
        archive's tier switches find it there); close it with :meth:`Segments.close`."""
        return Segments(self, name)

    # -- the bridge's seams ---------------------------------------------------------------------------------------
    @staticmethod
    def _forward_key(capture: Any) -> tuple:
        return (getattr(capture, "kind", "clean_commit"), int(capture.chunk.index), str(capture.stream))

    def forward_begin(self, capture: Any) -> None:
        """A clean forward starts (``begin_clean_forward``): ``context_forward`` (a chunk's clean forward: generation
        has started) or ``recache`` (a recache forward)."""
        kind = getattr(capture, "kind", "clean_commit")
        self.capture_opened(kind)
        self._forwards[self._forward_key(capture)] = self.begin("recache" if kind == "kv_recache"
                                                                else "context_forward")

    def forward_end(self, capture: Any) -> None:
        token = self._forwards.pop(self._forward_key(capture), None)
        if token is not None:
            self.end(token)
            self.capture_closed(getattr(capture, "kind", "clean_commit"))

    def chunk_boundary(self, chunk: Any) -> list:
        """A new chunk's boundary (``end_chunk``): generation has started; a scene cut after the first chunk, or a
        shot change within the request, is a ``shot`` transition. Returns the open ``archival_capture_sync`` span."""
        self.mark_generation("end_chunk")
        shot = int(getattr(chunk, "shot_index", 0) or 0)
        if int(chunk.index) > 0 and (bool(getattr(chunk, "scene_cut", False))
                                     or (self._last_shot is not None and shot != self._last_shot)):
            self.prompt_transition("shot")
        self._last_shot = shot
        return self.begin("archival_capture_sync")

    def recache_commit(self) -> list:
        """A recache epoch is committed (``end_recache``): a ``recache`` transition; the open capture span."""
        self.prompt_transition("recache")
        return self.begin("archival_capture_sync")

    def close_open(self) -> None:
        """End what a failure left open (the restore window, a clean forward)."""
        for segments in list(self._open_segments):
            segments.close()
        for key in list(self._forwards):
            self.end(self._forwards.pop(key))
        self._recache_open = 0

    # -- stalls -------------------------------------------------------------------------------------------------
    def stall_begin(self, stream: Any = None, *, ready: bool = False, source: str = "layer") -> list:
        """Before a readiness wait of the consuming ``stream`` (None: current). ``ready``: the layer was issued
        already (no host block). ``source`` ``layer``: a per-layer wait of a forward that reads restored history (the
        first one outside a recache forward starts the generation phase); ``join``: the run's join of the restore."""
        if source == "layer" and self._recache_open == 0:
            self.mark_generation("readiness_wait")
        return [None, self.phase, time.monotonic_ns(), None, self._record(stream), None, None, source, bool(ready),
                stream]

    def stall_end(self, token: list | None, *, layer: int | None = None) -> None:
        if token is None or token[3] is not None:
            return
        if token[4] is not None:
            token[5] = self._record(token[9])
        token[3] = time.monotonic_ns()
        token[0] = "device_wait" if token[8] else "host_wait"
        token[6] = layer
        with self._lock:
            self._stalls.append(token[:8])

    # -- overlapped restore source windows ---------------------------------------------------------------------
    def window_begin(self, source: str, stream: Any = None) -> list:
        return [source, time.monotonic_ns(), None, self._record(stream), None, stream]

    def window_end(self, token: list | None) -> None:
        if token is None or token[2] is not None:
            return
        if token[3] is not None:
            token[4] = self._record(token[5])
        token[2] = time.monotonic_ns()
        with self._lock:
            self._windows.append(token[:5])

    # -- request facts ------------------------------------------------------------------------------------------
    def set_restore(self, report: Mapping[str, Any] | None, *, per_token: int = 0,
                    representation: str | None = None) -> None:
        """The restore's tiers (``restore.lease.tiers`` plus its ``mode``) and bytes (raw-equivalent attended KV,
        :func:`restore_bytes`; ``per_token``: :func:`bytes_per_token`)."""
        if not isinstance(report, Mapping):
            return
        lease = report.get("lease") or {}
        tiers = lease.get("tiers") if isinstance(lease, Mapping) else None
        self.tiers = {"mode": report.get("mode"), **(dict(tiers) if isinstance(tiers, Mapping) else {})}
        self.bytes = restore_bytes(report, per_token, representation=representation)

    # -- resolution ---------------------------------------------------------------------------------------------
    @staticmethod
    def _settle(entry: list, first: int) -> float | None:
        """The elapsed time of ``entry``'s event pair at ``first`` / ``first + 1``, resolved once: the events are
        dropped and the value kept in their place (``first`` None, ``first + 1`` the value)."""
        start, end = entry[first], entry[first + 1]
        if start is None:
            return end if isinstance(end, float) else None
        value = _elapsed_ms(start, end)
        entry[first], entry[first + 1] = None, value
        return value

    def resolve(self) -> dict[str, Any]:
        """The record (CONTRACT §3). Call only after the request's outputs are complete; again after later spans
        (the decode seam): resolved pairs keep their value, their events are released."""
        with self._lock:
            spans, stalls, windows = list(self._spans), list(self._stalls), list(self._windows)
        out_spans = []
        for entry in spans:
            name, phase, start, end, _, _, stream = entry
            out_spans.append({"name": name, "phase": phase, "host_start_ns": start, "host_end_ns": end,
                              "gpu_ms": self._settle(entry, 4), "stream": stream})
        by_kind = {kind: 0.0 for kind in STALL_KINDS}
        by_phase = {phase: 0.0 for phase in PHASES}
        details = []
        for entry in stalls:
            kind, phase, start, end, _, _, layer, source = entry
            measured = self._settle(entry, 4)
            idle = max(0.0, measured) if measured is not None else (end - start) / 1e6
            by_kind[kind] += idle
            by_phase[phase] += idle
            details.append({"kind": kind, "phase": phase, "source": source, "layer": layer, "gpu_ms": idle,
                            "host_start_ns": start, "host_end_ns": end})
        stall_ms = sum(by_kind.values())
        if self._window_cache is None or self._window_cache[0] != len(windows):
            self._window_cache = (len(windows), _union_ms(windows))
        window_ms = self._window_cache[1]
        transition = [item for item in out_spans if item["name"] == "prompt_transition"]
        return {
            "schema": SCHEMA, "clock": self.clock, "spans": out_spans,
            "exposed_stall": {"count": len(details), "gpu_ms": stall_ms, "by_kind": by_kind, "by_phase": by_phase},
            "hidden_rebuild_gpu_ms": max(0.0, window_ms - stall_ms) if windows else 0.0,
            "prompt_transition": {"kind": self._transition,
                                  "gpu_ms": sum(item["gpu_ms"] or 0.0 for item in transition),
                                  "host_ms": sum((item["host_end_ns"] - item["host_start_ns"]) / 1e6
                                                 for item in transition)},
            "tiers": self.tiers, "bytes": dict(self.bytes),
            # beyond the contract's keys: the evidence behind the sums
            "stalls": details, "restore_window_gpu_ms": window_ms if windows else None,
            "restore_windows": [{"source": source, "host_start_ns": start, "host_end_ns": end}
                                for source, start, end, _, _ in windows],
            "generation_started_ns": self.generation_started_ns, "generation_rule": self.generation_rule,
            "cuda_events": self.events}


class Segments:
    """Contiguous spans partitioning one interval (the synchronous restore): :meth:`switch` ends the current segment
    and starts the next at the same instant (one shared event), so the segments' sum is the whole interval. Active
    in the opening thread's context (:func:`active_segments`) until :meth:`close`."""

    def __init__(self, telemetry: WorkloadTelemetry, name: str):
        _check(name)
        self.telemetry = telemetry
        if telemetry.cuda and telemetry.compute_stream is None:
            telemetry.compute_stream = torch.cuda.current_stream(telemetry.device)
        self.name, self.closed = name, False
        self.start, self.event, self.phase = time.monotonic_ns(), telemetry._record(), telemetry.phase
        self._token = _ACTIVE.set(self)
        telemetry._open_segments.append(self)

    def _close_segment(self) -> tuple[int, Any]:
        tel = self.telemetry
        event = tel._record() if self.event is not None else None
        now = time.monotonic_ns()
        with tel._lock:
            tel._spans.append([self.name, self.phase, self.start, now, self.event, event,
                               "compute" if self.event is not None else "host"])
        return now, event

    def switch(self, name: str) -> None:
        _check(name)
        if self.closed or name == self.name:
            return
        self.start, self.event = self._close_segment()
        self.name, self.phase = name, self.telemetry.phase

    @property
    def current(self) -> str:
        return self.name

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._close_segment()
        with contextlib.suppress(ValueError):
            self.telemetry._open_segments.remove(self)
        try:
            _ACTIVE.reset(self._token)
        except ValueError:  # closed in another context (never on the request path): just clear it
            _ACTIVE.set(None)


def active_segments() -> Segments | None:
    """The partitioned restore window open in this thread's context, if any."""
    segments = _ACTIVE.get()
    return None if segments is None or segments.closed else segments


def active_telemetry() -> WorkloadTelemetry | None:
    segments = active_segments()
    return None if segments is None else segments.telemetry


@contextlib.contextmanager
def tier_segment(tier: str):
    """Inside the active restore window: attribute the enclosed work to ``restore_sync.<tier>``, then switch back."""
    segments = active_segments()
    if segments is None:
        yield
        return
    previous = segments.current
    segments.switch(f"restore_sync.{tier}")
    try:
        yield
    finally:
        segments.switch(previous)


def raw_tier_label(tiers: Any) -> str:
    """One label for a RAW install's ranges: ``gpu_raw`` when every range is a device copy, else ``host_raw`` (a mixed
    install is attributed to the copy-bound tier; per-tier counts and bytes stay in ``tiers``)."""
    names = {item[0] if isinstance(item, tuple) else item for item in tiers}
    return "gpu_raw" if names and names <= {"gpu_raw"} else "host_raw"


def workload_tier(label: str) -> str:
    """W1.1: a restore record's tier label in the telemetry vocabulary (:data:`RESTORE_TIERS`; ``store`` ->
    ``page_cache``)."""
    return TIER_ALIASES.get(label, label)


def offload_tier_label(labels: Any) -> str:
    """W1.1: one label for a raw_offload install's ranges (``gpu_raw`` / ``host_raw`` / ``nvme`` / ``store``): the
    slowest tier present, in the telemetry vocabulary (per-range tiers stay in ``tiers``)."""
    names = {workload_tier(str(label)) for label in labels}
    return next((tier for tier in _TIER_ORDER if tier in names), "host_raw")


@contextlib.contextmanager
def request_scope(telemetry: Any):
    """W1.1: the controller step of ``telemetry``'s request (``begin_request``): :func:`drain_span` records into it.
    None: nothing (telemetry off)."""
    if telemetry is None:
        yield
        return
    token = _REQUEST.set(telemetry)
    try:
        yield
    finally:
        _REQUEST.reset(token)


@contextlib.contextmanager
def drain_span():
    """W1.1: a controller drain (a fork's source drain, a rewind's switch drain) as a ``drain`` span of the request in
    :func:`request_scope`; a no-op elsewhere (controls, checkpoints, closes, telemetry off)."""
    telemetry = _REQUEST.get()
    if telemetry is None:
        yield
        return
    with telemetry.span("drain"):
        yield


@contextlib.contextmanager
def session_open_span():
    """W1.2: a closed session's reopen (manifest load, tree registry, prewarm) as a ``session_open`` span of the
    request in :func:`request_scope`; a no-op elsewhere (the ``open`` control, telemetry off)."""
    telemetry = _REQUEST.get()
    if telemetry is None:
        yield
        return
    with telemetry.span("session_open"):
        yield


def compressed_tier_label(labels: Any) -> str:
    """One label for a compressed call: ``compressed_prepared`` when every range's pack was prepared, else
    ``compressed_cold``."""
    names = set(labels)
    return "compressed_prepared" if names and names <= {"compressed_prepared"} else "compressed_cold"


def bytes_per_token(*, num_layers: int, num_heads: int, head_dim: int, capture_dtypes: Mapping[str, Any]) -> int:
    """Raw K + V bytes of one token of one stream, every layer (capture dtypes)."""
    size = sum(torch.empty((), dtype=capture_dtypes[target]).element_size() for target in ("K", "V"))
    return int(num_layers) * int(num_heads) * int(head_dim) * size


def restore_bytes(report: Mapping[str, Any], per_token: int, *, representation: str | None = None) -> dict[str, int]:
    """Raw-equivalent attended KV bytes of a restore (exact, from its tier records; every layer, both targets, every
    CFG stream; ``per_token``: one token of one stream, :func:`bytes_per_token`):

    * ``read``: every attended range restored from the archive (GPU raw, host raw, compressed; a RAW reference
      archive's store reads); a resident resume or a new session reads nothing;
    * ``decoded``: the compressed ranges (decoded and reconstructed into the caches);
    * ``h2d``: the ranges copied host to device (host raw; raw_offload's store reads; the RAW reference archive's
      staged rows).

    Compressed payload bytes are not counted here (``tiers`` names the compressed ranges)."""
    rate = int(per_token)
    out = {"read": 0, "decoded": 0, "h2d": 0}
    if report.get("mode") != "archive":
        return out
    lease = report.get("lease") or {}
    tiers = lease.get("tiers") if isinstance(lease, Mapping) else None
    if isinstance(tiers, Mapping) and tiers.get("per_chunk") is not None:
        for item in tiers["per_chunk"]:
            nbytes = int(item.get("tokens") or 0) * rate
            out["read"] += nbytes
            if str(item.get("tier", "")).startswith("compressed"):
                out["decoded"] += nbytes
            elif item.get("tier") in _H2D_TIERS:  # W1.1: raw_offload's store reads are host-to-device copies too
                out["h2d"] += nbytes
        return out
    plans = report.get("plans") or {}
    tokens = sum(int(span.get("length") or 0) for plan in plans.values() for span in plan.get("spans", ()))
    out["read"] = tokens * rate
    if representation == "raw":
        out["h2d"] = out["read"]
    return out


def _union_ms(windows: list) -> float:
    """Union of the source windows (ms): event offsets from the first window's start; host times off CUDA."""
    if not windows:
        return 0.0
    intervals = None
    if all(event0 is not None and event1 is not None for _, _, _, event0, event1 in windows):
        reference = windows[0][3]
        try:
            intervals = sorted((_offset(reference, event0), _offset(reference, event1))
                               for _, _, _, event0, event1 in windows)
        except RuntimeError:  # an event that never completed: the host clock instead
            intervals = None
    if intervals is None:
        intervals = sorted((start / 1e6, end / 1e6) for _, start, end, _, _ in windows)
    total, current = 0.0, None
    for start, end in intervals:
        if current is None or start > current[1]:
            if current is not None:
                total += current[1] - current[0]
            current = [start, end]
        else:
            current[1] = max(current[1], end)
    return total + (current[1] - current[0])


def _offset(reference: Any, event: Any) -> float:
    if event is reference:
        return 0.0
    event.synchronize()
    return float(reference.elapsed_time(event))


__all__ = ["PHASES", "PROMPT_TRANSITIONS", "RESTORE_TIERS", "SCHEMA", "SPAN_NAMES", "STALL_KINDS", "Segments",
           "WorkloadTelemetry", "active_segments", "active_telemetry", "bytes_per_token", "compressed_tier_label",
           "raw_tier_label", "restore_bytes", "tier_segment"]
