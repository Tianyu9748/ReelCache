"""Layer-overlapped restore: install a restored history's KV layer by layer while the model's forward reads it.

:func:`start_overlapped_restore` returns a :class:`LayerReadiness` at once. The caller has already allocated and bound
the destination (its native KV cache). Background threads, one per source, write it on private CUDA streams.

**Sources (O4).** A readiness has one or more producers ("sources"), each on its own thread and stream:
- ``"restore"``: the compressed restore (``restorer.restore``: decode the prepared packs, reconstruct, RoPE, store);
- ``"raw"``: a RAW install, layer by layer (``raw(layer)``): dense pre-RoPE K / V copied from the raw tier (GPU raw:
  device to device; host raw: host to device from the pinned buffers), the engine's RoPE, stored into the cache. The
  synchronous RAW path runs the same per-layer calls on the caller's stream
  (``reelcache.integrations.sglang.restore.DenseInstall``), so both write the same bits.

Layer ``L`` is readable once every source has issued its writes of that layer on its stream: each source records a
CUDA event there after the layer's writes, and the layer's host flag is set when the last source marked it. A mixed
history (some ranges raw, some compressed) therefore has ONE readiness per layer covering both sources. The consumer,
i.e. the attention of layer ``L``, calls ``readiness.wait(L, stream)`` before it reads the layer:
- a host wait until every source issued the layer;
- then a device-side wait of the consumer's stream on each source's event of the layer.

So the GPU never reads a layer before its writes complete, and the host is never blocked longer than until they were
issued.

**Order.** :class:`~reelcache.archive_restore.LineageRestorer` submits the decode jobs, then reconstructs layer by layer
in increasing order. When it binds the first destination of layer ``L``, every earlier layer has been issued on the
restore stream, so those layers are marked then. The LongCat overlapped restore uses the same pattern and measured
bitwise identical. Other restorers, e.g. ``SoftwareLineageRestorer``, which goes node by node, mark every layer when
the restore returns: correct, but with no overlap. The RAW install marks each layer once it issued it.

**Integrity and failures.**
- The compressed source marks its last layer only after the restore returned, i.e. after its integrity gate
  (asynchronous payload verification, D-E) passed. A forward that reads every layer therefore cannot complete on data
  that failed verification. A RAW layer is checked (rows, keys) before its writes are issued.
- ``require_prepared`` (the default) refuses a history whose packs are not prepared: overlapped restores decode
  prepared, already verified packs.
- A failure of any source is delivered, typed, to every later :meth:`LayerReadiness.wait` and to
  :meth:`LayerReadiness.join`, and stops the other sources at their next layer. A ``reelcache.errors`` exception keeps
  its class (``IntegrityError`` stays ``IntegrityError``, raised ``from`` the original); anything else becomes
  ``StateError``.
- No layer is released after a failure; layers released before it were completely written by every source.
- Commit or emit nothing derived from the restored KV before :meth:`LayerReadiness.join` returned. It returns once
  every source, including the compressed restore's gate, has finished and the RAW install's work has completed.

**Seam support (Step 11).**
- ``readiness.layer(i)`` is a per-layer view whose ``wait()`` / ``ready()`` the attention cache of layer ``i`` calls.
- ``after=`` chains restores in the order the forward consumes them (cond, then uncond): every source of a chained
  restore starts when every source of the previous one finished, and inherits its failure.
- ``wait_event=`` orders each source's first write after the caller's preparation on its own stream (e.g. the
  install zeroing the live slots, the RoPE tables).
- The threads run in the caller's ``torch.inference_mode`` / grad mode, captured at the call. SGLang's KV caches are
  inference tensors.
- :meth:`LayerReadiness.cancel` stops every source at its next layer. :meth:`LayerReadiness.quiesce` (cancel if
  needed, join, synchronize the streams) leaves no restore work in flight, so the caches can then be reset or freed.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any, Callable, Sequence

import torch

from .errors import ConfigurationError, ReelCacheError, StateError

#: The compressed restore's source name and the RAW install's.
RESTORE, RAW = "restore", "raw"


class LayerReadiness:
    """Per-layer readiness of a destination whose layers are still being restored (see the module docstring).

    Each producer ("source") calls :meth:`mark` after it has issued layer ``i``'s writes on its stream (a CUDA event
    is recorded there; on the CPU the writes are complete when ``mark`` is called), or :meth:`fail` once with the
    exception. Layer ``i`` is ready when every source marked it. The consumer calls :meth:`wait` before reading layer
    ``i``. ``waited_seconds[i]`` is the consumer's host wait for layer ``i``; ``marked_at[i]`` the
    ``time.perf_counter()`` at which the last source marked it, ``source_marked_at[source][i]`` each source's mark.
    """

    def __init__(self, layers: int, device: Any = None, *, timeout: float = 600.0,
                 sources: Sequence[str] = (RESTORE,)) -> None:
        if type(layers) is not int or layers < 1:
            raise ConfigurationError("layer readiness needs a positive layer count")
        if not timeout > 0:
            raise ConfigurationError("the layer wait timeout must be positive")
        sources = tuple(sources)
        if (not sources or len(set(sources)) != len(sources)
                or not all(isinstance(source, str) and source for source in sources)):
            raise ConfigurationError("readiness sources must be distinct non-empty names")
        self.layers = layers
        self.sources = sources
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.timeout = float(timeout)
        cuda = self.device.type == "cuda"
        self._flags = [threading.Event() for _ in range(layers)]
        self._marked = [False] * layers
        self._pending = [set(sources) for _ in range(layers)]
        self._events = [{source: torch.cuda.Event() if cuda else None for source in sources} for _ in range(layers)]
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._threads: dict[str, threading.Thread] = {}
        self._finished: dict[str, float] = {}
        self._cancelled = threading.Event()
        self._stop = threading.Event()  # cancel() or a source's failure: every source stops at its next layer
        self.stream = None  # the compressed restore's CUDA stream (else the first source's)
        self.streams: dict[str, Any] = {}
        self.created_at = time.perf_counter()
        self.called_at = self.created_at
        self.marked_at: list[float | None] = [None] * layers
        self.source_marked_at: dict[str, list[float | None]] = {source: [None] * layers for source in sources}
        self.waited_seconds = [0.0] * layers
        self.info: dict[str, Any] = {}
        # workload telemetry (W1a; None: no events): an event pair on the consuming stream around every wait
        self.stall_recorder: Any = None

    def _source(self, source: str | None) -> str:
        if source is None:
            if len(self.sources) != 1:
                raise ConfigurationError(f"name the source of a readiness with sources {self.sources}")
            return self.sources[0]
        if source not in self.sources:
            raise ConfigurationError(f"unknown readiness source {source!r} (sources {self.sources})")
        return source

    # -- producer ---------------------------------------------------------------------------------------------
    def mark(self, layer: int, stream: Any = None, source: str | None = None) -> None:
        """``source``'s writes of layer ``layer`` are issued (on ``stream``; None: the calling thread's current
        stream). The layer is released once every source marked it."""
        source = self._source(source)
        with self._lock:
            pending = self._pending[layer]
            if self._error is not None or source not in pending:
                return
            event = self._events[layer][source]
            if event is not None:
                event.record(stream)
            now = time.perf_counter()
            pending.discard(source)
            self.source_marked_at[source][layer] = now
            complete = not pending
            if complete:
                self._marked[layer] = True
                self.marked_at[layer] = now
        if complete:
            self._flags[layer].set()

    def fail(self, error: BaseException) -> None:
        """A source failed: every later wait (and :meth:`join`) raises ``error``, typed; the other sources stop."""
        with self._lock:
            if self._error is None:
                self._error = error
        self._stop.set()
        for flag in self._flags:
            flag.set()

    def attach(self, thread: threading.Thread, source: str | None = None) -> None:
        self._threads[self._source(source)] = thread

    def is_marked(self, layer: int, source: str | None = None) -> bool:
        """Whether layer ``layer`` is released (``source`` None) or ``source`` marked it."""
        if source is None:
            return self._marked[layer]
        return self._source(source) not in self._pending[layer]

    @property
    def stopping(self) -> bool:
        """Cancelled, or a source failed: producers stop at their next layer."""
        return self._stop.is_set()

    # -- consumer ---------------------------------------------------------------------------------------------
    def wait(self, layer: int, stream: Any = None) -> None:
        """Before reading layer ``layer``: host-wait until every source issued it, then order ``stream`` (None: the
        current stream) after each source's writes. Raises the restore's failure, typed, or ``StateError`` after
        ``timeout``. With a ``stall_recorder`` (workload telemetry) the wait is bracketed by two events on the
        consuming stream: their elapsed time is the GPU idle time the wait caused."""
        recorder = self.stall_recorder
        stall = None if recorder is None else recorder.stall_begin(stream, ready=self._flags[layer].is_set())
        started = time.perf_counter()
        if not self._flags[layer].wait(self.timeout):
            raise StateError(f"layer {layer} of the overlapped restore was not issued within {self.timeout} s")
        self.waited_seconds[layer] += time.perf_counter() - started
        if self._error is not None:
            raise self._failure() from self._error
        target = None
        for event in self._events[layer].values():
            if event is not None:
                if target is None:
                    target = stream if stream is not None else torch.cuda.current_stream(self.device)
                target.wait_event(event)
        if stall is not None:
            recorder.stall_end(stall, layer=layer)

    def ready(self, layer: int) -> bool:
        return self._flags[layer].is_set() and self._error is None

    def layer(self, layer: int) -> "LayerWait":
        """The per-layer view the attention cache of ``layer`` holds (``wait()`` / ``ready()``)."""
        if type(layer) is not int or not 0 <= layer < self.layers:
            raise ConfigurationError(f"layer must be an integer in [0, {self.layers})")
        return LayerWait(self, layer)

    @property
    def error(self) -> BaseException | None:
        return self._error

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def done(self) -> bool:
        """Every source's producer thread finished (successfully, failed or cancelled)."""
        threads = list(self._threads.values())
        return (len(threads) == len(self.sources) and all(not thread.is_alive() for thread in threads))

    def cancel(self) -> None:
        """Stop the restore at its next layer: it fails with ``StateError`` and no later layer is released."""
        self._cancelled.set()
        self._stop.set()

    def _join_threads(self, timeout: float | None, message: str) -> None:
        deadline = time.perf_counter() + (self.timeout if timeout is None else timeout)
        for thread in list(self._threads.values()):
            thread.join(max(0.0, deadline - time.perf_counter()))
            if thread.is_alive():
                raise StateError(message)

    def _await_producers(self) -> None:
        """Wait (no timeout) until every attached producer finished (``after=`` chaining)."""
        for thread in list(self._threads.values()):
            thread.join()

    def quiesce(self, timeout: float | None = None) -> bool:
        """Leave no restore work in flight, e.g. before a failed request resets or frees its caches.

        Cancels the restore unless it already finished, waits for every source's thread, then synchronizes their
        streams. Never raises the restore's own failure: the request handles its error. Returns True if the restore
        completed successfully, so every layer was released.
        """
        if not self.done:
            self.cancel()
        self._join_threads(timeout, "the overlapped restore did not quiesce")
        streams = [stream for stream in self.streams.values() if stream is not None]
        if self.stream is not None and not any(self.stream is stream for stream in streams):
            streams.append(self.stream)
        for stream in streams:
            stream.synchronize()
        return self._error is None

    def join(self, timeout: float | None = None) -> dict[str, Any]:
        """Wait for every source; re-raise a failure, typed; return the report (``info``)."""
        self._join_threads(timeout, "the overlapped restore did not finish")
        if self._error is not None:
            raise self._failure() from self._error
        return dict(self.info)

    def _failure(self) -> ReelCacheError:
        error = self._error
        message = f"overlapped restore failed: {type(error).__name__}: {error}"
        if isinstance(error, ReelCacheError):
            try:
                return type(error)(message)
            except Exception:  # noqa: BLE001 - a reelcache error class with another signature
                pass
        return StateError(message)

    def _finish(self, source: str) -> None:
        """``source``'s producer is done; the last one fills the combined timings of ``info``."""
        with self._lock:
            self._finished[source] = time.perf_counter()
            last = len(self._finished) == len(self.sources)
        if last and self._error is None:
            called = self.called_at
            marks = [t for t in self.marked_at if t is not None]
            self.info.update({
                "overlapped": True, "sources": list(self.sources),
                "first_layer_ready_seconds_from_call": min(marks) - called if marks else None,
                "last_layer_ready_seconds_from_call": max(marks) - called if marks else None,
                "restore_complete_seconds_from_call": max(self._finished.values()) - called})

    def timeline(self) -> dict[str, Any]:
        """Marks (seconds after creation) and the consumer's host waits per layer; per source with several."""
        record = {"marked_at": [None if t is None else t - self.created_at for t in self.marked_at],
                  "waited_seconds": list(self.waited_seconds)}
        if len(self.sources) > 1:
            record["sources"] = {source: [None if t is None else t - self.created_at for t in marks]
                                 for source, marks in self.source_marked_at.items()}
        return record


def _layer_of(entry: Any) -> int:
    key = entry[0] if isinstance(entry, tuple) else entry
    return int(key.layer)


def _release(item: Any) -> None:
    release = getattr(item, "release", None)
    if release is not None:
        release()


class LayerWait:
    """One layer of a :class:`LayerReadiness`: what the attention cache of that layer waits on before reading it."""

    __slots__ = ("readiness", "index")

    def __init__(self, readiness: LayerReadiness, index: int) -> None:
        self.readiness, self.index = readiness, index

    def wait(self, stream: Any = None) -> None:
        self.readiness.wait(self.index, stream)

    def ready(self) -> bool:
        return self.readiness.ready(self.index)


def start_overlapped_restore(restorer: Any, node_ref: str | None = None, destination: Callable | None = None,
                             rope: Callable | None = None, *, layers: int,
                             ranges: Sequence[tuple[int, int]] | None = None, keys: Sequence[Any] | None = None,
                             resident: Sequence[Any] = (), stream: Any = None, require_prepared: bool = True,
                             progressive: bool | None = None, after: LayerReadiness | None = None,
                             wait_event: Any = None, timeout: float = 600.0, raw: Callable[[int], Any] | None = None,
                             raw_stream: Any = None, raw_depth: int = 1, device: Any = None, telemetry: Any = None,
                             **options: Any) -> LayerReadiness:
    """Start restoring into ``destination`` layer by layer; returns its :class:`LayerReadiness` at once.

    Compressed source (``restorer`` not None): ``node_ref`` / ``destination`` / ``rope`` / ``ranges`` / ``keys`` /
    ``resident`` / ``options`` are the restorer's (``LineageRestorer.restore``; e.g. ``layer_ahead=d`` bounds the
    canvas working set to about ``d + 2`` layers). ``stream``: its CUDA stream (default: a new one).
    ``require_prepared``: refuse a history whose packs are not prepared (``LineageRestorer.prewarm`` / session open
    prepares them); raised here, before anything starts. ``progressive``: mark layers while the restorer issues them
    (default: for a ``LineageRestorer``, whose reconstruction is layer-ordered).

    RAW source (``raw`` not None, O4): ``raw(layer)`` issues layer ``layer``'s RAW writes on the current stream and
    returns its pending work (``release()`` waits for it and frees what it holds, e.g. staging; e.g.
    ``integrations.sglang.restore.DenseInstall.layer``, whose fast path holds nothing and never waits). It runs for
    layers ``0 .. layers - 1`` on ``raw_stream`` (default: a new one, so host-to-device copies overlap the compressed
    restore's work). ``raw_depth``: pending layers kept before the oldest is released (staging bound:
    ``raw_depth + 1`` layers). The thread ends after one wait for everything it issued.

    ``restorer=None`` starts a RAW install alone. ``layers``: the model's layer count; every layer is marked by the
    time every source returned. ``after``: start once that restore finished (cond before uncond), inheriting its
    failure or cancellation. ``wait_event``: a CUDA event every source's stream waits on before its first write (the
    caller's preparation of the destination). ``device``: default the restorer's, else ``raw.device``, else CPU.

    The threads run in the caller's ``torch.inference_mode`` and grad mode, captured at this call: thread-local modes
    are not inherited by new threads. So they write inference-tensor destinations (SGLang's KV caches) exactly as the
    synchronous restore does when called in the same mode. The restorer keeps its internal canvases and bases normal
    tensors whatever the mode.

    ``telemetry`` (workload telemetry, W1a; None: nothing recorded): the readiness's ``stall_recorder`` (an event pair
    on the consuming stream around every wait) and each source stream's active window (``window_begin`` /
    ``window_end``: its first work to its last layer).
    """
    if type(layers) is not int or layers < 1:
        raise ConfigurationError("layers must be a positive integer")
    if type(require_prepared) is not bool:
        raise ConfigurationError("require_prepared must be boolean")
    if type(raw_depth) is not int or raw_depth < 0:
        raise ConfigurationError("raw_depth must be a nonnegative integer")
    if after is not None and not isinstance(after, LayerReadiness):
        raise ConfigurationError("after must be the LayerReadiness of an earlier overlapped restore")
    if restorer is None and raw is None:
        raise ConfigurationError("an overlapped restore needs a restorer, a RAW install or both")
    if raw is not None and not callable(raw):
        raise ConfigurationError("raw must be callable (layer -> pending work)")
    if restorer is not None and (node_ref is None or destination is None):
        raise ConfigurationError("a compressed overlapped restore needs node_ref and destination")
    if device is None:
        device = getattr(restorer, "device", None) if restorer is not None else None
        if device is None:
            device = getattr(raw, "device", None) or getattr(getattr(raw, "__self__", None), "device", None) or "cpu"
    device = torch.device(device)
    if restorer is not None and require_prepared:
        prepared = getattr(restorer, "prepared", None)
        if prepared is None:
            raise StateError("an overlapped restore of prepared packs needs a restorer with a prepared-pack cache")
        skipped = {item for item in resident if isinstance(item, str)}
        history = restorer.history(node_ref, ranges=ranges)
        missing = [item["packet"] for item in history
                   if item["node_id"] not in skipped and not prepared.contains(item["packet"])]
        if missing:
            raise StateError(f"overlapped restore: {len(missing)} of {len(history)} history packs are not prepared "
                             "(prepare the session first, e.g. LineageRestorer.prewarm)")
    if restorer is not None and progressive is None:
        from .archive_restore import LineageRestorer

        progressive = isinstance(restorer, LineageRestorer)
    cuda = device.type == "cuda"
    if restorer is not None and stream is None and cuda:
        stream = torch.cuda.Stream(device=device)
    if raw is not None and raw_stream is None and cuda:
        raw_stream = torch.cuda.Stream(device=device)
    sources = ((RESTORE,) if restorer is not None else ()) + ((RAW,) if raw is not None else ())
    readiness = LayerReadiness(layers, device, timeout=timeout, sources=sources)
    readiness.stall_recorder = telemetry
    readiness.streams = {source: value for source, value in ((RESTORE, stream), (RAW, raw_stream))
                         if source in sources}
    readiness.stream = readiness.streams[sources[0]]
    inference, grad = torch.is_inference_mode_enabled(), torch.is_grad_enabled()  # entered again in the threads
    called = readiness.called_at = time.perf_counter()
    lock = threading.Lock()
    state = {"current": -1, "bound": 0}

    def begin(own: Any) -> None:
        """Every source: after the chained restore, unless stopped; its stream after the caller's preparation."""
        if after is not None:  # the restore the forward consumes first finishes first
            after._await_producers()
            if after.error is not None:  # its failure or cancellation, inherited
                raise after.error
        if readiness.stopping:
            raise StateError("the overlapped restore was cancelled")
        if cuda:
            torch.cuda.set_device(device)
        if wait_event is not None and own is not None:
            own.wait_event(wait_event)  # the destination is prepared before the first write

    @contextlib.contextmanager
    def modes(own: Any):
        # The caller's thread-local modes: SGLang serves under torch.inference_mode, and its KV caches (the
        # destination) are inference tensors that only inference mode may write in place.
        with (torch.cuda.stream(own) if own is not None else contextlib.nullcontext()), \
                torch.inference_mode(inference), torch.set_grad_enabled(grad):
            yield

    def bound(node_id, entry, tile):
        if readiness.stopping:
            raise StateError("the overlapped restore was cancelled")
        layer = _layer_of(entry)
        if not 0 <= layer < layers:
            raise ConfigurationError(f"restored layer {layer} lies outside the model's {layers} layers")
        with lock:
            state["bound"] += 1
            if progressive and layer != state["current"]:
                if layer < state["current"]:
                    raise StateError("the restorer issued layers out of order; an overlapped restore needs "
                                     "layer-ordered reconstruction")
                for done in range(layer):  # every earlier layer's reconstruction has been issued on ``stream``
                    if not readiness.is_marked(done, RESTORE):
                        readiness.mark(done, stream, RESTORE)
                state["current"] = layer
        return destination(node_id, entry, tile)

    def run_restore():
        began = time.perf_counter()
        window = None
        try:
            begin(stream)
            if telemetry is not None:
                window = telemetry.window_begin(RESTORE, stream)
            with modes(stream):
                result = restorer.restore(node_ref, bound, rope, ranges=ranges, keys=keys, resident=resident,
                                          stream=stream, **options)
            for handle in result.pop("handles", None) or []:
                _release(handle)
            for layer in range(layers):  # the remaining layers, after the restore and its integrity gate returned
                readiness.mark(layer, stream, RESTORE)
            if window is not None:
                telemetry.window_end(window)
            finished = time.perf_counter()
            marks = [t for t in readiness.source_marked_at[RESTORE] if t is not None]
            readiness.info.update({key: value for key, value in result.items() if key != "history"})
            readiness.info.update({
                "progressive": progressive, "bound_entries": state["bound"],
                "thread_start_delay_seconds": began - called,
                "restore_source": {"first_mark_seconds_from_call": min(marks) - called,
                                   "last_mark_seconds_from_call": max(marks) - called,
                                   "complete_seconds_from_call": finished - called}})
        except BaseException as exc:  # noqa: BLE001 - every waiter re-raises it, typed
            readiness.fail(exc)
        finally:
            readiness._finish(RESTORE)

    def run_raw():
        began = time.perf_counter()
        pending: list[Any] = []
        issued: list[float] = []
        released = 0.0
        window = None
        try:
            begin(raw_stream)
            if telemetry is not None:
                window = telemetry.window_begin(RAW, raw_stream)
            with modes(raw_stream):
                for layer in range(layers):
                    if readiness.stopping:
                        raise StateError("the overlapped restore was cancelled")
                    pending.append(raw(layer))
                    readiness.mark(layer, raw_stream, RAW)
                    issued.append(time.perf_counter())
                    while len(pending) > raw_depth:
                        clock = time.perf_counter()
                        _release(pending.pop(0))
                        released += time.perf_counter() - clock
                if window is not None:
                    telemetry.window_end(window)
                while pending:  # the install's work has completed when the thread ends (the lease may then go)
                    clock = time.perf_counter()
                    _release(pending.pop(0))
                    released += time.perf_counter() - clock
                if raw_stream is not None:  # layers that hold nothing to release (fast path): one wait, at the end
                    clock = time.perf_counter()
                    raw_stream.synchronize()
                    released += time.perf_counter() - clock
            finished = time.perf_counter()
            stats = getattr(getattr(raw, "__self__", None), "stats", None)
            readiness.info["raw_install"] = {
                "layers": layers, "depth": raw_depth, "thread_start_delay_seconds": began - called,
                "first_mark_seconds_from_call": issued[0] - called,
                "last_mark_seconds_from_call": issued[-1] - called, "complete_seconds_from_call": finished - called,
                "release_wait_seconds": released,
                **(stats() if callable(stats) else {})}
        except BaseException as exc:  # noqa: BLE001 - every waiter re-raises it, typed
            readiness.fail(exc)  # first: the waiters and the other sources stop now
            for item in pending:  # waits for the issued work, then frees its staging
                with contextlib.suppress(BaseException):
                    _release(item)
            if raw_stream is not None:  # nothing it issued still reads the sources when the thread ends
                with contextlib.suppress(BaseException):
                    raw_stream.synchronize()
        finally:
            readiness._finish(RAW)

    threads = []
    if restorer is not None:
        threads.append((RESTORE, threading.Thread(target=run_restore, name="reelcache-overlapped-restore",
                                                  daemon=True)))
    if raw is not None:
        threads.append((RAW, threading.Thread(target=run_raw, name="reelcache-overlapped-raw", daemon=True)))
    for source, thread in threads:
        readiness.attach(thread, source)
    for _, thread in threads:
        thread.start()
    return readiness


def start_overlapped_install(raw: Callable[[int], Any], *, layers: int, **kwargs: Any) -> LayerReadiness:
    """A RAW install alone, layer-overlapped (:func:`start_overlapped_restore` with ``restorer=None``)."""
    return start_overlapped_restore(None, raw=raw, layers=layers, **kwargs)


__all__ = ["RAW", "RESTORE", "LayerReadiness", "LayerWait", "start_overlapped_install", "start_overlapped_restore"]
