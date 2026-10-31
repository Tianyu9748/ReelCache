"""Persistent NVDEC decoder pool with asynchronous submission and bounded device canvases.

``DecoderPool`` runs ``workers`` threads on one GPU. Each worker owns an LRU of
zero-latency :class:`~reelcache.nvdec.CanvasSession` objects, one per canvas
geometry (an NVDEC session cannot switch geometry, and the SDK creates the
hardware decoder at its first picture, which costs tens of milliseconds).
:meth:`DecoderPool.prewarm` creates and warms the sessions of known geometries
on every worker ahead of time, so session creation stays off the restore
critical path; :meth:`DecoderPool.prewarm_shapes` does the same for canvas
shapes no pack exists for yet (synthesized pictures), e.g. every canonical
node canvas once per process. Jobs are routed to the worker with the least outstanding work:
queued bitstream bytes plus a session-creation charge for every queued job that
must first create its session (:func:`cold_session_bytes`: ``COLD_SESSION_BYTES``
per session, whatever the canvas -- measured, creating a session costs ~30 ms on
an RTX PRO 6000 and 41-52 ms on an A100 from 160x160 to 3040x1632 canvases, and
concurrent creations serialize: 8 at once take ~8x one), so the first jobs of a
geometry spread over the workers instead of queueing on the first warm one.

Sessions are created only synchronously: by :meth:`DecoderPool.prewarm` /
:meth:`DecoderPool.prewarm_shapes` (attach, session open, prepare, seal-time
preparation), and by a job the router sent to a worker without the session
(that job's own creation, charged as above). The pool never creates a session
in the background. Explicit warm-ups never delay a restore either, even when a
prewarm runs on another thread (e.g. seal-time preparation on an archival
worker) while a restore decodes: a worker starts a warm-up only when it has no
job queued, one warm-up runs at a time across the pool, the router never sends
a job to a worker while it creates a warm-up session (unless every worker is
-- a single-worker pool), and a warm-up whose session a job created meanwhile
is dropped. ``prewarm`` still returns once every requested session exists;
concurrent creations largely serialize in the driver (8 at once take ~8x
one), so one at a time costs an attach warm-up only a few percent (A100-40GB:
+0.3-0.6 s on 5.9-12.5 s). A restore overlapping a prewarm that creates
sessions decodes on one worker fewer meanwhile (A100-40GB: +26-61 ms, never a
wait on a creation). (CR-17-5: a session creation also waits for all work
already queued on the GPU -- ~0.35 s behind 0.3 s of queued kernels -- so any
creation a job queues behind can cost up to the GPU queue's depth: background
warm-ups -- queued behind the restore's jobs, then idle-priority, then one at a
time after an idle grace period -- each still delayed restores.) A geometry first met
without prewarm therefore stays on the workers its first jobs created it on
until their backlog exceeds one creation charge: prewarm what restores will
meet (the canonical canvases of ``reelcache.packed_decode`` at attach,
``LineageRestorer.prewarm`` / ``prewarm_lineages`` at session open). Whether
workers map to distinct physical NVDEC engines is not assumed: the pool only
measures (per-worker busy time and jobs, scaling with workers).

``submit`` never blocks on decoding and returns a :class:`DecodeTicket` whose
CUDA event marks the canvas pictures as written. ``CanvasArena`` bounds the
uint8 device canvases: a released canvas becomes reusable only after every
event passed at release (e.g. ``restore_kernels`` handle events) completed, so
a canvas is never recycled under a pending reconstruction. Exceeding the
budget waits for retirements and then fails explicitly with ``CapacityError``.

Shutdown: :meth:`DecoderPool.close` lets queued jobs finish, then every worker
destroys its NVDEC sessions on its own thread. A process that exits normally
with pools still open (e.g. an in-process tool that never closes its bridge)
closes them at interpreter exit (:func:`close_all_pools`, registered with
``atexit`` by the first pool), while the interpreter and the CUDA context are
intact: no session is ever left to libnvcuvid's own exit handler, which can
deadlock (RTX PRO 6000, driver 595.91.07: ``exit()`` -> libnvcuvid ->
``pthread_join`` of a libnvcuvid thread blocked in ``pthread_mutex_lock``).

Device-memory pressure: each pool registers
``decoder_pool:<gpu_id>`` -> :meth:`DecoderPool.trim_device_cache` with
``reelcache.device_pressure`` and unregisters on close. The trim frees the idle
canvases of every arena the pool decodes into, never waiting on GPU work.
Sessions are kept.

Callers may run inside ``torch.inference_mode`` (SGLang serves requests under
it). Arena and warm-up canvases are always normal tensors: they are reused
across calls in either mode and written by decoder threads, which do not
inherit the submitter's thread-local mode. A caller-provided inference-tensor
canvas is written by its worker inside inference mode.

Write ordering (:func:`check_ordering`): a decode is ordered after the work
already queued on the submitting thread's current stream (``after="current"``,
the default) or after explicit CUDA events. Unordered decodes (``after=None``)
are accepted only into a canvas currently leased from a ``CanvasArena``; any
other canvas (e.g. a fresh ``torch.empty`` on the compute stream, whose memory
the allocator may have just recycled from work still queued there) is rejected
with ``ConfigurationError`` instead of being decoded concurrently with that work.
"""

from __future__ import annotations

import atexit
from collections import OrderedDict, deque
from dataclasses import dataclass
import functools
import os
import threading
import time
import weakref

from .device_pressure import register_device_trim
from .errors import CapacityError, ConfigurationError, IntegrityError, StateError, UnsupportedError
from .nvdec import CanvasSession, _canvas_planes, check_canvas, check_units, load_api, normal_tensors, write_mode

# Routing charge for a worker without a warm session of the job's geometry, in
# bitstream-byte equivalents: one session creation (hardware decoder + first
# picture) is ~30-50 ms whatever the canvas size (Step 5c calibration, jobs
# 4403103/4413408), ~5-9 MB of decoding at the measured 124-173 MB/s per worker.
COLD_SESSION_BYTES = 8 * 1024**2
# Attribute naming the CanvasArena that currently leases a canvas (None once released).
_ARENA_LEASE = "_reelcache_canvas_arena"
_AFTER = "after must be 'current', None or a sequence of CUDA events"
#: Seconds :func:`close_all_pools` waits for each pool's workers at interpreter exit (decodes in flight end first).
EXIT_CLOSE_SECONDS = 30.0
# Open pools of this process (their worker threads keep them alive until closed) and whether close_all_pools is
# registered with atexit (by the first pool).
_OPEN_POOLS = weakref.WeakSet()
_OPEN_LOCK = threading.Lock()
_EXIT_HOOK = []


def _track(pool):
    with _OPEN_LOCK:
        _OPEN_POOLS.add(pool)
        if not _EXIT_HOOK:
            atexit.register(close_all_pools)
            _EXIT_HOOK.append(close_all_pools)


def open_pools():
    """The open :class:`DecoderPool` objects this process created (not closed, or a timed-out close)."""
    with _OPEN_LOCK:
        return [pool for pool in _OPEN_POOLS if pool._owner_pid == os.getpid()]


def close_all_pools(timeout=None):
    """Close every open :class:`DecoderPool` of this process; returns ``{"closed", "timed_out"}`` pool counts.

    Registered with :mod:`atexit` when the process starts its first pool. A process exiting normally with pools
    open thus destroys every NVDEC session on its owning worker thread while the interpreter and the CUDA context
    are intact, instead of leaving live decoders to libnvcuvid's native exit handler (which deadlocked: Step 11 job
    4419034, RTX PRO 6000). Exit hooks registered later, e.g. a bridge's own close, run first. Pools inherited
    across ``fork`` belong to the parent and are skipped. ``timeout`` (None: ``EXIT_CLOSE_SECONDS``) bounds the
    wait for each pool's workers: a decode that never ends is left behind (counted in ``timed_out``) rather than
    hanging the exit.
    """
    timeout = EXIT_CLOSE_SECONDS if timeout is None else timeout
    closed = timed_out = 0
    for pool in open_pools():
        try:
            ended = pool.close(timeout=timeout)
        except BaseException:  # noqa: BLE001 - interpreter exit: every other pool is still closed
            ended = False
        closed += bool(ended)
        timed_out += not ended
    return {"closed": closed, "timed_out": timed_out}


def check_ordering(canvas, after):
    """Validate the write ordering of a decode into ``canvas``; returns ``"current"``, a list of events or None.

    ``"current"`` orders the decode after every operation already queued on the
    submitting thread's current stream (safe for any canvas allocated on, or
    last used by, that stream). Explicit CUDA events are waited for exactly: the
    caller asserts they cover every pending user of the canvas memory. No
    ordering (``None`` or no events) is accepted only for a canvas currently
    leased from a :class:`CanvasArena` (allocated on its idle private stream or
    recycled after its consumers' events); for any other canvas it would let the
    decoder write memory that queued work may still use, so it is rejected.
    """
    if isinstance(after, str):
        if after != "current":
            raise ConfigurationError(_AFTER)
        return after
    if after is not None:
        try:
            after = [event for event in after if event is not None]
        except TypeError:
            raise ConfigurationError(_AFTER) from None
        if any(not hasattr(event, "query") for event in after):
            raise ConfigurationError(_AFTER)
        if after:
            return after
    if getattr(canvas, _ARENA_LEASE, None) is None:
        raise ConfigurationError(
            "an unordered decode (after=None or no events) needs a canvas leased from a CanvasArena; a canvas "
            "allocated elsewhere (e.g. torch.empty on the compute stream) may reuse memory that work still queued "
            "on another stream uses, so the decode would race with it: pass after='current' or explicit CUDA events")
    return None


def cold_session_bytes(width, height, pix_fmt):
    """Routing charge of creating a decoder session for a ``width x height`` canvas, in bitstream-byte equivalents.

    ``COLD_SESSION_BYTES`` for every canvas: session creation does not grow with the picture (A100: 41.6 ms +
    0.75 ms per million decoded samples, 160x160 to 3040x1632, uncontended; RTX PRO 6000: 30-32 ms per session for
    Wan and LongCat canonical canvases alike). What makes cold sessions expensive is concurrency -- creations
    serialize -- which is why the pool creates sessions only synchronously (prewarm, or a job's own creation).
    """
    _canvas_planes(pix_fmt)
    for value, name in ((width, "width"), (height, "height")):
        if type(value) is not int or value < 1:
            raise ConfigurationError(f"canvas {name} must be a positive integer")
    return COLD_SESSION_BYTES


@functools.lru_cache(maxsize=64)
def synthetic_unit(width, height, pix_fmt):
    """One self-contained all-intra access unit ``(bytes, offset, length, 1)`` of a blank ``width x height`` canvas.

    Encoded once per process with the version-2 software encoder contract (``_hevc.encode_pictures``), so the
    session it creates is the one real packs of that canvas decode on. Used to warm canvas shapes for which no
    pack exists yet (:meth:`DecoderPool.prewarm_shapes`).
    """
    import numpy as np

    from . import _hevc

    for value, name in ((width, "width"), (height, "height")):
        if type(value) is not int or not 16 <= value <= _hevc.MAX_DIMENSION:
            raise ConfigurationError(f"canvas {name} must be an integer in [16, {_hevc.MAX_DIMENSION}]")
    planes = _canvas_planes(pix_fmt)
    stream, _, _ = _hevc.encode_pictures(np.zeros((1, planes, height, width), dtype=np.uint8), pix_fmt)
    (unit,) = _hevc.all_intra_access_units(stream, 1)
    if not unit["self_contained"]:
        raise IntegrityError("the synthesized warm-up picture is not a self-contained IDR access unit")
    return stream, unit["offset"], unit["length"], 1


class DecodeTicket:
    """Future of one submitted job; the CUDA event exists once the job was issued."""

    def __init__(self, nbytes, pictures):
        self.nbytes, self.pictures = nbytes, pictures
        self._done = threading.Event()
        self._event = self._error = None
        self.slot = None
        self.cold = False
        self.submitted = time.perf_counter()
        self.started = self.finished = None

    def _finish(self, event=None, error=None):
        self._event, self._error = event, error
        self.finished = time.perf_counter()
        self._done.set()

    def done(self):
        return self._done.is_set()

    def event(self, timeout=None):
        """Host-wait until the job was issued and return its CUDA completion event."""
        if not self._done.wait(timeout):
            raise TimeoutError("NVDEC job was not issued within the timeout")
        if self._error is not None:
            raise self._error
        return self._event

    def wait(self, stream, timeout=None):
        """Make ``stream`` wait for the decoded pictures (host-waits only for issue)."""
        event = self.event(timeout)
        if event is not None:
            stream.wait_event(event)

    def synchronize(self, timeout=None):
        event = self.event(timeout)
        if event is not None:
            event.synchronize()


@dataclass
class _Job:
    units: list
    canvas: object
    first: int
    ticket: DecodeTicket
    ready: object
    geometry: tuple
    warm: bool = False
    charge: int = 0


class _Worker:
    def __init__(self, index):
        self.index = index
        self.queue = deque()
        self.warm_queue = deque()  # explicit warm-ups not started: run only while ``queue`` is empty
        self.ready = threading.Condition(threading.Lock())
        self.outstanding_bytes = 0  # queued and running work: bitstream bytes + session-creation charges
        self.geometries = set()  # warm geometries or ones a queued/running job creates (routing view; pool lock)
        self.warm_tickets = {}  # geometry -> ticket of the warm-up queued or running here (pool lock)
        self.warming = False  # a warm-up runs here: the router sends no job to this worker (pool lock)
        self.jobs = 0
        # Records only (pool lock): jobs queued and not started, warm-up jobs queued or running, and each started
        # job's queue wait (submit -> start) with the parts spent behind a session creation / a warm-up job here.
        self.queued_jobs = 0
        self.warm_ups_in_flight = 0
        self.jobs_started = 0
        self.queue_wait_seconds = 0.0
        self.queue_wait_max_seconds = 0.0
        self.creation_wait_seconds = 0.0
        self.warm_up_wait_seconds = 0.0
        # Worker thread only: recent (start, end) of session creations and of warm-up jobs on this worker.
        self.creation_spans = deque(maxlen=64)
        self.warm_up_spans = deque(maxlen=64)
        self.busy_seconds = 0.0
        self.sessions_created = 0
        self.cold_jobs = 0
        self.evictions = 0
        self.create_seconds = 0.0
        self.warm_seconds = 0.0
        self.setup = {"caps": 0.0, "stream": 0.0, "create_decoder": 0.0, "first_picture": 0.0}
        self.error = None
        self.thread = None


class DecoderPool:
    """``workers`` NVDEC threads on one GPU; ``submit`` never blocks on decoding.

    Only version-2 all-intra streams are decoded here; inter-coded and legacy
    packets keep using :class:`~reelcache.nvdec.NVDECDecoder`.
    ``max_sessions_per_worker`` bounds each worker's geometry LRU.
    Sessions are created only by :meth:`prewarm` / :meth:`prewarm_shapes` and
    by the jobs routed to a worker without one; nothing runs in the background
    (``warm_all_workers`` is accepted only as False: the background warm-ups it
    enabled were removed, CR-17-5). Warm-ups yield to jobs: idle priority per
    worker, one at a time across the pool, and no job is routed to a worker
    while it creates one (unless every worker does: a single-worker pool).
    """

    def __init__(self, *, gpu_id, gpu_uuid, workers, max_sessions_per_worker=16, api=None, start_timeout=120.0,
                 warm_all_workers=False):
        import torch

        if type(gpu_id) is not int or gpu_id < 0:
            raise ConfigurationError("NVDEC gpu_id must be a nonnegative visible-device index")
        if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-") or len(gpu_uuid) < 8:
            raise ConfigurationError("NVDEC requires an explicit GPU UUID")
        if type(workers) is not int or not 1 <= workers <= 64:
            raise ConfigurationError("a decoder pool needs 1..64 workers")
        if type(max_sessions_per_worker) is not int or max_sessions_per_worker < 1:
            raise ConfigurationError("max_sessions_per_worker must be a positive integer")
        if warm_all_workers is not False:
            raise ConfigurationError(
                "background decoder warm-ups were removed (CR-17-5: a session creation overlapping a restore delayed "
                "it); create sessions synchronously with DecoderPool.prewarm / prewarm_shapes before restoring")
        if gpu_id >= torch.cuda.device_count():
            raise UnsupportedError("NVDEC device is not visible")
        actual = str(torch.cuda.get_device_properties(gpu_id).uuid).removeprefix("GPU-")
        if actual != gpu_uuid.removeprefix("GPU-"):
            raise ConfigurationError("NVDEC visible device does not match the granted GPU UUID")
        self._api = load_api() if api is None else api
        self.gpu_id, self.gpu_uuid = gpu_id, gpu_uuid
        self.device = torch.device("cuda", gpu_id)
        self.max_sessions_per_worker = max_sessions_per_worker
        self.warm_all_workers = False
        self._alloc_stream = None  # idle private stream for warm-up canvases (created on first use)
        self._owner_pid = os.getpid()
        self._lock = threading.Lock()
        self._closed = False
        self._warming = 0  # warm-ups running across the pool (at most one; pool lock)
        self._stats = {"jobs": 0, "pictures": 0, "bitstream_bytes": 0, "failures": 0, "cold_jobs": 0,
                       "prewarm_seconds": 0.0, "warm_ups": 0, "warm_ups_moot": 0, "trims": 0, "trimmed_bytes": 0}
        self._workers = [_Worker(index) for index in range(workers)]
        self._arenas = weakref.WeakSet()  # canvas arenas this pool decoded into (device-memory pressure trims)
        for worker in self._workers:
            worker.thread = threading.Thread(target=self._run, args=(worker,), name=f"reelcache-nvdec-{worker.index}",
                                             daemon=True)
            worker.thread.start()
        _track(self)
        self._trim_handle = register_device_trim(f"decoder_pool:{gpu_id}", self.trim_device_cache, device=gpu_id)

    @property
    def workers(self):
        return len(self._workers)

    def _run(self, worker):
        import torch

        sessions = OrderedDict()
        try:
            torch.cuda.set_device(self.device)
        except BaseException as exc:  # noqa: BLE001 - delivered through every ticket
            worker.error = exc
        while True:
            with worker.ready:
                # Jobs first; a queued warm-up only while no job is queued here and no other warm-up runs.
                while True:
                    if worker.queue:
                        job = worker.queue.popleft()
                        break
                    if worker.warm_queue and self._start_warm_up(worker):
                        job = worker.warm_queue.popleft()
                        break
                    worker.ready.wait()
            if job is None:
                break
            job.ticket.started = time.perf_counter()
            if not job.warm:
                self._record_wait(worker, job.ticket)
            key = job.geometry
            try:
                if worker.error is not None:
                    raise worker.error
                session = sessions.get(key)
                if job.warm and session is not None:  # a job created this session meanwhile: nothing to warm
                    with self._lock:
                        self._stats["warm_ups_moot"] += 1
                    job.ticket._finish()
                    continue
                created = None
                if session is None:
                    started = created = time.perf_counter()
                    width, height, pix_fmt = key
                    session = CanvasSession(gpu_id=self.gpu_id, width=width, height=height, pix_fmt=pix_fmt,
                                            api=self._api)
                    if len(sessions) == self.max_sessions_per_worker:
                        _, evicted = sessions.popitem(last=False)
                        evicted.close()
                        with self._lock:
                            worker.evictions += 1
                            worker.geometries.discard(evicted.geometry)
                    sessions[key] = session
                    with self._lock:
                        worker.sessions_created += 1
                        worker.create_seconds += time.perf_counter() - started
                        if not job.warm:
                            worker.cold_jobs += 1
                            self._stats["cold_jobs"] += 1
                            job.ticket.cold = True
                else:
                    sessions.move_to_end(key)
                for event in job.ready:
                    session.stream.wait_event(event)
                fresh = session.stats["first_picture_seconds"] is None
                decoding = time.perf_counter()
                with write_mode(job.canvas):  # this thread never inherits the submitter's inference mode
                    event = session.decode(job.units, job.canvas, job.first)
                    job.canvas.record_stream(session.stream)
                if fresh:  # the creation ends with the first picture (the SDK creates the hardware decoder there)
                    worker.creation_spans.append((decoding if created is None else created,
                                                  decoding + session.stats["first_picture_seconds"]))
                    with self._lock:
                        for name, value in session.setup_seconds.items():
                            worker.setup[name] += value
                        worker.setup["first_picture"] += session.stats["first_picture_seconds"]
                job.ticket._finish(event=event)
            except BaseException as exc:  # noqa: BLE001 - delivered through the ticket
                failed = sessions.pop(key, None)
                if failed is not None:
                    try:
                        failed.close()
                    except BaseException:  # noqa: BLE001
                        pass
                with self._lock:
                    self._stats["failures"] += 1
                    worker.geometries.discard(key)
                job.ticket._finish(error=exc)
            finally:
                ended = time.perf_counter()
                busy = ended - job.ticket.started
                if job.warm:
                    worker.warm_up_spans.append((job.ticket.started, ended))
                with self._lock:
                    worker.outstanding_bytes -= job.charge
                    worker.busy_seconds += busy
                    if job.warm:
                        worker.warm_seconds += busy
                        worker.warm_ups_in_flight -= 1
                        worker.warm_tickets.pop(key, None)
                        worker.warming = False
                        self._warming -= 1
                if job.warm:  # the pool's warm-up slot is free: a worker holding queued warm-ups may start one
                    self._wake_warm_ups()
        with worker.ready:  # closed: warm-ups that never started are dropped
            dropped, worker.warm_queue = list(worker.warm_queue), deque()
        with self._lock:
            for job in dropped:
                worker.warm_ups_in_flight -= 1
                worker.warm_tickets.pop(job.geometry, None)
        for job in dropped:
            job.ticket._finish(error=StateError("the NVDEC pool closed before this warm-up ran"))
        for session in sessions.values():
            try:
                session.close()
            except BaseException:  # noqa: BLE001
                pass

    def _start_warm_up(self, worker):
        """Claim the pool's single warm-up slot for the warm-up at the head of ``worker.warm_queue`` (called by the
        worker holding ``worker.ready``). The warm-up is charged and its geometry becomes this worker's while it
        runs; the router sends no job to the worker until it ends."""
        with self._lock:
            if self._closed or self._warming or worker.queued_jobs:  # routed jobs (even mid-enqueue) go first
                return False
            job = worker.warm_queue[0]
            job.charge = job.ticket.nbytes + (0 if job.geometry in worker.geometries
                                              else cold_session_bytes(*job.geometry))
            worker.outstanding_bytes += job.charge
            worker.geometries.add(job.geometry)
            worker.warming = True
            self._warming += 1
            self._stats["warm_ups"] += 1
            return True

    def _wake_warm_ups(self):
        for other in self._workers:
            with other.ready:
                if other.warm_queue:
                    other.ready.notify()

    def _record_wait(self, worker, ticket):
        """Records only (worker thread, at a job's start): its queue wait and the parts of it spent while a session
        creation or a warm-up job ran on this worker (the worker's own recent spans; all ended by now)."""
        low, high = ticket.submitted, ticket.started

        def behind(spans):
            return sum(max(0.0, min(end, high) - max(start, low)) for start, end in spans)

        waited, creation, warm_up = high - low, behind(worker.creation_spans), behind(worker.warm_up_spans)
        with self._lock:
            worker.queued_jobs -= 1
            worker.jobs_started += 1
            worker.queue_wait_seconds += waited
            worker.queue_wait_max_seconds = max(worker.queue_wait_max_seconds, waited)
            worker.creation_wait_seconds += creation
            worker.warm_up_wait_seconds += warm_up

    def _choose(self, geometry):
        cold = cold_session_bytes(*geometry)

        def cost(worker):  # queued work, plus one session creation if this worker will not have the session
            return worker.outstanding_bytes + (0 if geometry in worker.geometries else cold), worker.index
        # A worker creating a warm-up session is blocked for as long as the creation waits (the GPU queue's depth).
        return min([worker for worker in self._workers if not worker.warming] or self._workers, key=cost)

    def _enqueue(self, worker, units, canvas, first, ticket, geometry, warm, after="current"):
        """Queue one job on ``worker`` (None: the router's choice, made under the same lock as the reservation, so a
        worker never starts a warm-up between the choice and the job's arrival), charged its bitstream bytes plus
        one session creation if the worker does not have (or queue) the geometry yet. A warm-up (``warm``) waits in
        the worker's warm-up queue instead, neither charged nor warm for routing until it starts
        (:meth:`_start_warm_up`)."""
        import torch

        if isinstance(after, str):
            if after != "current":
                raise ConfigurationError(_AFTER)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
            ready = [ready]
        elif after is None:
            ready = []
        else:
            ready = [event for event in after if event is not None]
            if any(not hasattr(event, "query") for event in ready):
                raise ConfigurationError(_AFTER)
        with self._lock:
            if self._closed:
                raise StateError("the NVDEC pool is closed")
            if worker is None:
                worker = self._choose(geometry)
            if warm:
                worker.warm_ups_in_flight += 1
                worker.warm_tickets[geometry] = ticket
                job = _Job(units, canvas, first, ticket, ready, geometry, True, 0)
            else:
                charge = ticket.nbytes + (0 if geometry in worker.geometries else cold_session_bytes(*geometry))
                worker.outstanding_bytes += charge
                worker.geometries.add(geometry)
                worker.jobs += 1
                worker.queued_jobs += 1
                self._stats["jobs"] += 1
                self._stats["pictures"] += ticket.pictures
                self._stats["bitstream_bytes"] += ticket.nbytes
                job = _Job(units, canvas, first, ticket, ready, geometry, False, charge)
            ticket.slot = worker.index
        with worker.ready:
            (worker.warm_queue if warm else worker.queue).append(job)
            worker.ready.notify()

    def _warm_canvas(self, planes, height, width):
        """A one-picture warm-up canvas without pending users (a normal tensor from an idle private stream)."""
        import torch

        with normal_tensors():
            if self.device.type != "cuda":
                return torch.empty((1, planes, height, width), dtype=torch.uint8, device=self.device)
            with self._lock:
                if self._alloc_stream is None:
                    self._alloc_stream = torch.cuda.Stream(device=self.device)
                stream = self._alloc_stream
            with torch.cuda.stream(stream):
                return torch.empty((1, planes, height, width), dtype=torch.uint8, device=self.device)

    def _warm_up(self, worker, geometry, unit):
        """Queue a one-picture decode of ``unit`` creating ``geometry``'s session on ``worker``; returns the ticket."""
        width, height, pix_fmt = geometry
        ticket = DecodeTicket(unit[2], 1)
        canvas = self._warm_canvas(_canvas_planes(pix_fmt), height, width)
        self._enqueue(worker, [unit], canvas, 0, ticket, geometry, True, None)
        return ticket

    def submit(self, units, canvas, first=0, *, pix_fmt="yuv444p", worker=None, after="current"):
        """Queue access units ``[(bytes, offset, length, 1)]`` for ``canvas[first:]``; returns a ticket.

        The canvas must stay alive and unread until the ticket's event; the pool
        records the canvas on the session stream for allocator safety. It may be
        a normal or an inference tensor (written in its own mode by the worker).
        ``after`` orders the canvas writes (:func:`check_ordering`): ``"current"``
        (default) waits for work already queued on the submitting thread's
        current stream (safe for any freshly allocated canvas, but it serializes
        decode behind that stream); explicit CUDA events are waited for exactly;
        ``None`` (no ordering) is accepted only for a canvas currently leased from
        a :class:`CanvasArena` and raises ``ConfigurationError`` for any other.
        """
        if os.getpid() != self._owner_pid:
            raise StateError("an NVDEC pool cannot cross a process boundary")
        units, pictures, nbytes = check_units(units)
        if pictures != len(units):
            raise ConfigurationError("NVDEC pool units hold exactly one access unit each")
        if type(first) is not int or first < 0:
            raise ConfigurationError("the first canvas picture must be a nonnegative integer")
        height, width = check_canvas(canvas, device=self.device, pix_fmt=pix_fmt, pictures=first + pictures)
        after = check_ordering(canvas, after)
        arena = getattr(canvas, _ARENA_LEASE, None)
        if arena is not None and arena not in self._arenas:  # its idle canvases join this pool's pressure trim
            with self._lock:
                self._arenas.add(arena)
        geometry = (width, height, pix_fmt)
        ticket = DecodeTicket(nbytes, pictures)
        if worker is None:
            target = None  # routed inside _enqueue
        elif type(worker) is int and 0 <= worker < len(self._workers):
            target = self._workers[worker]
        else:
            raise ConfigurationError("worker must index a pool worker")
        self._enqueue(target, units, canvas, first, ticket, geometry, False, after)
        return ticket

    def prewarm(self, samples, timeout=None):
        """Create and warm, on every worker, a session per ``(width, height, pix_fmt, unit)`` sample.

        ``unit`` is one access unit ``(bytes, offset, length, 1)`` of that
        geometry. Workers that already have (or queue a job of) a geometry are
        skipped; a warm-up another call queued is waited for, not repeated.
        Warm-ups yield to jobs (idle priority per worker, one at a time across
        the pool, no job routed to a worker while it warms), so a prewarm on
        another thread never delays a restore; returns once every requested
        session exists. Warming more geometries than
        ``max_sessions_per_worker`` evicts the least recently used sessions.
        Returns the wall seconds; failures raise.
        """
        started = time.perf_counter()
        tickets = []
        for width, height, pix_fmt, unit in samples:
            (unit,), _, _ = check_units([unit])
            geometry = (width, height, pix_fmt)
            for worker in self._workers:
                with self._lock:
                    if geometry in worker.geometries:
                        continue
                    pending = worker.warm_tickets.get(geometry)
                if pending is not None:
                    tickets.append(pending)
                    continue
                tickets.append(self._warm_up(worker, geometry, unit))
        for ticket in tickets:
            ticket.synchronize(timeout)
        seconds = time.perf_counter() - started
        with self._lock:
            self._stats["prewarm_seconds"] += seconds
        return seconds

    def prewarm_shapes(self, shapes, timeout=None):
        """Create a session for every ``(width, height, pix_fmt)`` canvas shape on every worker, in one call.

        For shapes no pack exists for yet (e.g. every canonical node canvas at process start, see
        ``reelcache.packed_decode.canonical_canvas_shapes``): each shape's access unit is synthesized once per
        process (:func:`synthetic_unit`). Shapes already warm or queued on every worker are skipped, so repeated
        calls cost only the check. More shapes than ``max_sessions_per_worker`` raise ``CapacityError`` (they
        would evict each other). Returns ``{"shapes", "warmed_shapes", "sessions_created", "encode_seconds",
        "session_seconds", "seconds"}``.
        """
        started = time.perf_counter()
        ordered = []
        for shape in shapes:
            if (not isinstance(shape, tuple) or len(shape) != 3 or type(shape[0]) is not int
                    or type(shape[1]) is not int or not isinstance(shape[2], str)):
                raise ConfigurationError("canvas shapes are (width, height, pix_fmt) tuples")
            if shape not in ordered:
                ordered.append(shape)
        if len(ordered) > self.max_sessions_per_worker:
            raise CapacityError(f"{len(ordered)} canvas shapes exceed max_sessions_per_worker="
                                f"{self.max_sessions_per_worker}; warm fewer shapes or allow more sessions")
        with self._lock:
            missing = [shape for shape in ordered if any(shape not in worker.geometries for worker in self._workers)]
            before = sum(worker.sessions_created for worker in self._workers)
        samples = [(*shape, synthetic_unit(*shape)) for shape in missing]
        encoded = time.perf_counter()
        session_seconds = self.prewarm(samples, timeout) if samples else 0.0
        with self._lock:
            created = sum(worker.sessions_created for worker in self._workers) - before
        return {"shapes": len(ordered), "warmed_shapes": len(missing), "sessions_created": created,
                "encode_seconds": encoded - started, "session_seconds": session_seconds,
                "seconds": time.perf_counter() - started}

    def stats(self):
        """Pool and per-worker statistics; the wait records are for per-restore deltas (records only).

        Gauges, read at a restore's submit: ``queued_jobs`` (jobs submitted, not started), ``warm_ups_in_flight``
        (``prewarm`` / ``prewarm_shapes`` warm-up jobs queued or running), ``warm_up_running`` (0 or 1: one runs
        at a time). Cumulative counters (take the difference across a restore): ``jobs`` (submitted),
        ``jobs_started``, ``queue_wait_seconds`` (sum over started jobs of submit -> start),
        ``creation_wait_seconds`` / ``warm_up_wait_seconds`` (the parts of those waits spent while a session
        creation -- a cold job's or a warm-up's -- / a warm-up job ran on the job's worker; the latter stays 0
        whenever the pool has more than one worker), ``sessions_created``, ``cold_jobs``, ``warm_ups`` (started)
        and ``warm_ups_moot`` (dropped: a job had created the session); ``queue_wait_max_seconds`` is the largest
        single wait so far. There are no background warm-ups (``background_warm_ups`` is ``"none"``,
        ``auto_warms`` always 0).
        """
        with self._lock:
            waits = {name: sum(getattr(worker, name) for worker in self._workers)
                     for name in ("queued_jobs", "warm_ups_in_flight", "jobs_started", "queue_wait_seconds",
                                  "creation_wait_seconds", "warm_up_wait_seconds", "sessions_created")}
            waits["queue_wait_max_seconds"] = max(worker.queue_wait_max_seconds for worker in self._workers)
            waits["warm_up_running"] = self._warming
            return {
                **self._stats, **waits, "gpu_id": self.gpu_id, "gpu_uuid": self.gpu_uuid, "closed": self._closed,
                "warm_all_workers": False, "cold_session_bytes": COLD_SESSION_BYTES,
                # No background session creation exists; "auto_warms" stays 0 for stats consumers of earlier pools.
                "background_warm_ups": "none", "auto_warms": 0,
                "workers": [{
                    "index": worker.index, "jobs": worker.jobs, "busy_seconds": worker.busy_seconds,
                    "queued": len(worker.queue), "outstanding_bytes": worker.outstanding_bytes,
                    "sessions_created": worker.sessions_created, "cold_jobs": worker.cold_jobs, "auto_warms": 0,
                    "evictions": worker.evictions, "create_seconds": worker.create_seconds,
                    "warm_seconds": worker.warm_seconds, "setup_seconds": dict(worker.setup),
                    "geometries": sorted(worker.geometries), "queued_jobs": worker.queued_jobs,
                    "warm_ups_in_flight": worker.warm_ups_in_flight, "jobs_started": worker.jobs_started,
                    "queue_wait_seconds": worker.queue_wait_seconds,
                    "queue_wait_max_seconds": worker.queue_wait_max_seconds,
                    "creation_wait_seconds": worker.creation_wait_seconds,
                    "warm_up_wait_seconds": worker.warm_up_wait_seconds, "warming": worker.warming,
                    "warm_queued": len(worker.warm_queue),
                } for worker in self._workers],
                "physical_engine_assignment": "not observable; measured by scaling only",
                "software_fallback": False,
            }

    def trim_device_cache(self, reason):
        """Device-memory pressure trim (``reelcache.device_pressure``, registered as ``decoder_pool:<gpu_id>``).

        Frees the idle decoded canvases of every :class:`CanvasArena` this pool
        decoded into (:meth:`CanvasArena.trim`) and returns the bytes freed.
        Canvases leased by a restore, or still read by its reconstructions,
        stay. Consumer events are only queried, so the trim never waits on GPU
        work. NVDEC sessions are kept: re-creating one would put a session
        creation on the next restore's critical path (CR-17-5). The registry
        runs ``torch.cuda.empty_cache()`` after the trims, which hands the
        freed canvases (cached on each arena's private stream) back to the
        device.
        """
        with self._lock:
            arenas = list(self._arenas)
        freed = sum(arena.trim(reason) for arena in arenas)
        with self._lock:
            self._stats["trims"] += 1
            self._stats["trimmed_bytes"] += freed
        return freed

    def close(self, timeout=None):
        """Stop the pool: queued jobs finish, then every worker destroys its NVDEC sessions on its own thread.

        Idempotent; also run at interpreter exit for pools still open (:func:`close_all_pools`). ``timeout`` bounds
        the wait for the workers (None: until they end). Returns True once every worker has ended, i.e. no session
        of this pool remains; a later call waits again for workers a timed-out call left running.
        """
        with self._lock:
            first, self._closed = not self._closed, True
        if first:
            handle = getattr(self, "_trim_handle", None)
            if handle is not None:
                handle.unregister()
            for worker in self._workers:
                with worker.ready:
                    worker.queue.append(None)
                    worker.ready.notify()
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        ended = True
        for worker in self._workers:
            thread = worker.thread
            if thread is None or thread is threading.current_thread():
                continue  # a worker closing its pool ends (closing its sessions) when it returns to its loop
            thread.join(None if deadline is None else max(0.0, deadline - time.monotonic()))
            ended = ended and not thread.is_alive()
        if ended:
            with _OPEN_LOCK:
                _OPEN_POOLS.discard(self)
        return ended

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class CanvasArena:
    """Bounded uint8 device canvases, recycled only after their consumer events complete.

    CUDA canvases are allocated on a private stream that runs no work, so a new
    canvas never aliases memory with pending users on another stream; decoders
    may therefore write arena canvases without waiting for the compute stream.
    An acquired canvas is leased until :meth:`release`; only leased canvases may
    be decoded into without ordering (``DecoderPool.submit(after=None)``), and a
    canvas is released exactly once. Canvases are normal tensors whatever the
    caller's ``torch.inference_mode``.

    Released canvases stay allocated, keyed by shape, for the next restore of
    the same canvases (no allocation on its path). A restore that fits the
    budget leases all of its canvases at once, so the arena keeps up to the
    largest such set (e.g. 1.28 GiB after a CausVid h=12 restore) until a new
    shape needs the room or :meth:`trim` frees them under device-memory
    pressure.
    """

    def __init__(self, device, budget_bytes, *, wait_seconds=30.0, poll_seconds=0.0002):
        import torch

        if type(budget_bytes) is not int or budget_bytes <= 0:
            raise ConfigurationError("canvas arena budget must be a positive integer")
        self.device = torch.device(device)
        self._stream = None
        if self.device.type == "cuda":
            self._stream = torch.cuda.Stream(device=self.device)
        self.budget_bytes, self.wait_seconds, self.poll_seconds = budget_bytes, wait_seconds, poll_seconds
        self._free = {}
        self._retiring = []
        self._lock = threading.Lock()
        self.used_bytes = 0
        self.stats = {"allocations": 0, "reuses": 0, "waits": 0, "evictions": 0, "peak_bytes": 0, "released": 0,
                      "trims": 0, "trimmed_canvases": 0, "trimmed_bytes": 0}
        self.last_trim = None

    @staticmethod
    def _size(shape):
        size = 1
        for value in shape:
            size *= value
        return size

    def _sweep(self):
        still = []
        for canvas, events in self._retiring:
            events = [event for event in events if not event.query()]
            if events:
                still.append((canvas, events))
            else:
                self._free.setdefault(tuple(canvas.shape), []).append(canvas)
        self._retiring = still

    def acquire(self, shape):
        import torch

        shape = tuple(int(value) for value in shape)
        if len(shape) != 4 or any(value < 1 for value in shape):
            raise ConfigurationError("canvas shape must be [pictures, planes, height, width]")
        size = self._size(shape)
        if size > self.budget_bytes:
            raise CapacityError("one canvas exceeds the whole canvas arena budget")
        deadline = time.perf_counter() + self.wait_seconds
        waited = False
        while True:
            with self._lock:
                self._sweep()
                free = self._free.get(shape)
                if free:
                    self.stats["reuses"] += 1
                    canvas = free.pop()
                    setattr(canvas, _ARENA_LEASE, self)
                    return canvas
                if self.used_bytes + size > self.budget_bytes:
                    for key in list(self._free):
                        while self._free[key] and self.used_bytes + size > self.budget_bytes:
                            evicted = self._free[key].pop()
                            self.used_bytes -= self._size(evicted.shape)
                            self.stats["evictions"] += 1
                        if not self._free[key]:
                            del self._free[key]
                if self.used_bytes + size <= self.budget_bytes:
                    self.used_bytes += size
                    self.stats["allocations"] += 1
                    self.stats["peak_bytes"] = max(self.stats["peak_bytes"], self.used_bytes)
                    with normal_tensors():
                        if self._stream is None:
                            canvas = torch.empty(shape, dtype=torch.uint8, device=self.device)
                        else:
                            with torch.cuda.stream(self._stream):
                                canvas = torch.empty(shape, dtype=torch.uint8, device=self.device)
                    setattr(canvas, _ARENA_LEASE, self)
                    return canvas
                pending = bool(self._retiring)
            if not pending or time.perf_counter() > deadline:
                raise CapacityError("canvas arena budget exhausted; release consumed canvases first")
            if not waited:
                self.stats["waits"] += 1
                waited = True
            time.sleep(self.poll_seconds)

    def release(self, canvas, after=None):
        """Return a leased canvas; it is reused only after ``after`` (a CUDA event or events) completed.

        Pass every event after which no reader or writer of the canvas remains
        (e.g. the decode ticket's event and each restore handle's event). A
        canvas that is not leased from this arena (released twice, or acquired
        elsewhere) raises ``StateError``: it would otherwise be handed out twice.
        """
        if after is None:
            events = []
        elif isinstance(after, (list, tuple)):
            events = [event for event in after if event is not None]
        else:
            events = [after]
        if any(not hasattr(event, "query") for event in events):
            raise ConfigurationError("canvas release events must be CUDA events")
        with self._lock:
            if getattr(canvas, _ARENA_LEASE, None) is not self:
                raise StateError("the canvas is not leased from this arena (released twice or acquired elsewhere)")
            setattr(canvas, _ARENA_LEASE, None)
            self._retiring.append((canvas, events))
            self.stats["released"] += 1

    def report(self):
        with self._lock:
            self._sweep()
            return {**self.stats, "budget_bytes": self.budget_bytes, "used_bytes": self.used_bytes,
                    "retiring": len(self._retiring), "free": sum(len(v) for v in self._free.values()),
                    "free_bytes": sum(self._size(canvas.shape) for canvases in self._free.values()
                                      for canvas in canvases),
                    "last_trim": None if self.last_trim is None else dict(self.last_trim)}

    def trim(self, reason=None):
        """Free every canvas no restore uses; returns the bytes freed (0 if none).

        Frees the free canvases: released, with every consumer event passed at
        release complete. Events are queried, never waited on. Leased canvases,
        and released ones whose decode or reconstruction may still be running,
        are not touched. A trim is therefore safe while restores run on other
        threads; their canvases join the free list later.

        The freed memory returns to PyTorch's caching allocator, cached on the
        arena's private stream. Only ``torch.cuda.empty_cache()``, which the
        device-memory pressure response runs once after every trim, gives it
        back to the device for other streams. The next restore re-allocates
        canvases of the trimmed shapes (counted in ``stats["allocations"]``).
        ``reason`` is recorded in ``report()["last_trim"]``. Every pool that
        decoded into the arena trims it under pressure
        (:meth:`DecoderPool.trim_device_cache`).
        """
        started = time.perf_counter()
        with self._lock:
            self._sweep()
            canvases = [canvas for free in self._free.values() for canvas in free]
            count, freed = len(canvases), sum(self._size(canvas.shape) for canvas in canvases)
            self._free = {}
            del canvases  # the arena's last references: their memory returns to the caching allocator
            self.used_bytes -= freed
            self.stats["trims"] += 1
            self.stats["trimmed_canvases"] += count
            self.stats["trimmed_bytes"] += freed
            self.last_trim = {"reason": None if reason is None else str(reason), "canvases": count, "bytes": freed,
                              "kept_bytes": self.used_bytes, "retiring": len(self._retiring),
                              "seconds": time.perf_counter() - started}
        return freed

