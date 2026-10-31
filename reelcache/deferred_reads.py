"""Lockstep per-matrix routines with batched host reads (no per-matrix host synchronization).

A *routine* is a generator that launches device work and yields a tuple of
tensors whose host values it needs next; it is resumed with a tuple of host
values (a Python ``bool``/``int``/``float`` for every 0-d tensor, a float64
NumPy array otherwise) and finally returns its result. :func:`run_lockstep`
advances many independent routines together, so a round of reads costs one
device-to-host transfer per device instead of one synchronization per matrix
and value. Under contention with a native forward on the same GPU every host
synchronization waits for the planning stream's queued work, so the number of
rounds, not the number of matrices, sets the planner's latency.

The routines below mirror :func:`reelcache.factorization.project_onto_basis`,
:func:`~reelcache.factorization.measure_source_energy`,
:func:`~reelcache.factorization.fast_condition_bound` and
:func:`reelcache.rank_method.span_energy` operation by operation: the same kernels
on the same inputs and the same host scalars, so values, exceptions and
decisions are identical; only *when* the host reads happen changes.
Divisions by host scalars stay divisions by Python floats (CUDA evaluates
them as a reciprocal multiply, unlike a division by a device tensor).
"""

from __future__ import annotations

import collections
import contextlib
import math
import threading
import time
import weakref
from collections.abc import Callable, Generator, Iterable, Sequence
from typing import Any

import numpy as np
import torch

from . import accel_b as _accel_b
from .errors import ConfigurationError, IntegrityError
from .exact_blas import mm
from .factorization import Projection, SourceEnergy, basis_condition_bound, validate_rank
from .graph_capture import current_graph_cache

#: Byte alignment of every segment of a batched upload (the CUDA caching allocator's block alignment),
#: so kernels that select vectorized paths by pointer alignment behave as on a fresh allocation.
UPLOAD_ALIGNMENT = 512

Routine = Generator[Sequence[torch.Tensor], tuple, Any]


class BatchedCall:
    """A lockstep request for device work batched with the round's other calls of the same ``key``.

    A routine may yield a ``BatchedCall`` instead of a tuple of tensors to read: :func:`run_lockstep` collects
    the round's calls, runs ``run(payloads)`` once per ``key`` (payloads in routine order; it returns one
    result per payload) and resumes each routine with its result. If ``run`` raises, the exception is thrown
    into every routine of that group (as if each had made the call inline).
    """

    __slots__ = ("key", "run", "payload")

    def __init__(self, key: Any, run, payload: Any):
        self.key, self.run, self.payload = key, run, payload


def _host_value(values: np.ndarray, tensor: torch.Tensor):
    if tensor.ndim == 0:
        value = values[0]
        if tensor.dtype == torch.bool:
            return bool(value)
        if tensor.dtype.is_floating_point:
            return float(value)
        return int(value)
    values = values.reshape(tuple(tensor.shape))
    return values.astype(bool) if tensor.dtype == torch.bool else values


def read_host(requests: Sequence[Sequence[torch.Tensor]]) -> list[tuple]:
    """Host values of every tensor of every request with one transfer per device.

    Values travel as float64, which represents every float16/bfloat16/float32
    value, boolean and small integer exactly.
    """
    output: list[list] = [[None] * len(request) for request in requests]
    by_device: dict[torch.device, list] = {}
    for row, request in enumerate(requests):
        for column, tensor in enumerate(request):
            if not isinstance(tensor, torch.Tensor):
                raise ConfigurationError("lockstep reads take tensors")
            by_device.setdefault(tensor.device, []).append((row, column, tensor))
    for items in by_device.values():
        parts = [tensor.detach().reshape(-1).to(torch.float64) for _, _, tensor in items]
        flat = torch.cat(parts) if len(parts) > 1 else parts[0]
        with blocked_wait("host_read"):  # the device synchronization allocates no device memory (dev22j8)
            flat = flat.cpu().numpy()
        offset = 0
        for (row, column, tensor), part in zip(items, parts):
            count = part.numel()
            output[row][column] = _host_value(flat[offset:offset + count], tensor)
            offset += count
    return [tuple(row) for row in output]


def _serve(requests: list) -> tuple[list, dict[int, BaseException]]:
    """Host values of the round's reads and results of its batched calls (positions as in ``requests``)."""
    results: list = [None] * len(requests)
    raised: dict[int, BaseException] = {}
    reads = [position for position, request in enumerate(requests) if not isinstance(request, BatchedCall)]
    if reads:
        for position, values in zip(reads, read_host([requests[position] for position in reads])):
            results[position] = values
    groups: dict[Any, list[int]] = {}
    for position, request in enumerate(requests):
        if isinstance(request, BatchedCall):
            groups.setdefault(request.key, []).append(position)
    for positions in groups.values():
        try:
            outputs = list(requests[positions[0]].run([requests[position].payload for position in positions]))
            if len(outputs) != len(positions):
                raise IntegrityError("a batched lockstep call returned the wrong number of results")
        except Exception as exc:  # noqa: BLE001 - thrown into each calling routine
            raised.update((position, exc) for position in positions)
            continue
        for position, output in zip(positions, outputs):
            results[position] = output
    return results, raised


_STEP_HOOK = threading.local()


@contextlib.contextmanager
def step_hook(hook: Callable[[], Any] | None, *, blocked: Callable[[str], Any] | None = None,
              phase: Callable[[str], Any] | None = None):
    """Call ``hook()`` in this thread before every routine step :func:`run_lockstep` runs (each resumption of a
    routine, i.e. between the Python work of consecutive steps) and at every :func:`yield_point` (thread-local; e.g.
    an archival worker yielding to restores in flight, SGLang Step 7). ``None`` removes the hook inside the block.

    dev22j8 (both optional, thread-local like the hook): ``blocked(label)`` returns the context manager entered around
    this thread's :func:`blocked_wait` sections (waits that allocate no device memory; the archive fitter counts a
    pause as landed meanwhile and holds it when the wait ends); ``phase(label)`` is told the step this thread enters
    (:func:`step_phase`; records only)."""
    previous = (getattr(_STEP_HOOK, "hook", None), getattr(_STEP_HOOK, "blocked", None),
                getattr(_STEP_HOOK, "phase", None))
    _STEP_HOOK.hook, _STEP_HOOK.blocked, _STEP_HOOK.phase = hook, blocked, phase
    try:
        yield
    finally:
        _STEP_HOOK.hook, _STEP_HOOK.blocked, _STEP_HOOK.phase = previous


def yield_point() -> None:
    """Run this thread's :func:`step_hook` (if any): a boundary where pausing is safe (no locks held)."""
    hook = getattr(_STEP_HOOK, "hook", None)
    if hook is not None:
        hook()


@contextlib.contextmanager
def blocked_wait(label: str):
    """A wait of this thread that allocates no device memory (dev22j8): a host read's device synchronization, a wait
    for a staging slot, an in-place encode, a dependency. Inside, the step hook's owner may count a pause as landed;
    the end of the wait is a step boundary (the owner's ``blocked`` context runs the hook there). A no-op without a
    ``blocked`` hook (see :func:`step_hook`)."""
    blocked = getattr(_STEP_HOOK, "blocked", None)
    if blocked is None:
        yield
        return
    with blocked(label):
        yield


def step_phase(label: str) -> None:
    """Name the step this thread enters for its step hook's owner (dev22j8; e.g. the archive fitter's phase at a pause
    acknowledgement timeout). Records only; a no-op without a ``phase`` hook."""
    phase = getattr(_STEP_HOOK, "phase", None)
    if phase is not None:
        phase(label)


def run_lockstep(routines: Iterable[Routine]) -> list:
    """Run routines in lockstep (one batched read per round); return their results in order.

    Exceptions keep the semantics of running the routines one after another:
    when routine ``i`` raises, routines after ``i`` are closed, earlier ones
    run to completion, and the exception of the lowest failing index is raised.
    Routines may also yield :class:`BatchedCall` requests (device work batched
    across the round's routines).
    """
    routines = list(routines)
    results: list = [None] * len(routines)
    errors: dict[int, BaseException] = {}
    sends: dict[int, Any] = {index: None for index in range(len(routines))}
    throws: dict[int, BaseException] = {}
    active = list(range(len(routines)))
    hook = getattr(_STEP_HOOK, "hook", None)
    try:
        while active:
            requests, waiting = [], []
            for index in active:
                if errors and index > min(errors):
                    routines[index].close()
                    continue
                if hook is not None:
                    hook()
                try:
                    if index in throws:
                        request = routines[index].throw(throws.pop(index))
                    else:
                        request = routines[index].send(sends[index])
                except StopIteration as stop:
                    results[index] = stop.value
                    continue
                except Exception as exc:  # noqa: BLE001 - re-raised below in sequential order
                    errors[index] = exc
                    continue
                requests.append(request if isinstance(request, BatchedCall) else tuple(request))
                waiting.append(index)
            if errors:
                first = min(errors)
                for index in [item for item in waiting if item > first]:
                    routines[index].close()
                kept = [(index, request) for index, request in zip(waiting, requests) if index < first]
                waiting = [index for index, _ in kept]
                requests = [request for _, request in kept]
            if not waiting:
                break
            served, raised = _serve(requests)
            sends = dict(zip(waiting, served))
            throws = {waiting[position]: exc for position, exc in raised.items()}
            active = waiting
    except BaseException:
        for routine in routines:
            routine.close()
        raise
    if errors:
        raise errors[min(errors)]
    return results


#: Finalization's projections and quantization keep the host temporaries of contiguous CPU tensors in
#: :class:`TileWorkspace` buffers; False restores the fresh-temporary expressions (the same values: a switch for tests
#: and A/B timing). Source fingerprints always use a workspace.
POOL_HOST_TEMPORARIES = True


class TileWorkspace:
    """Reusable host buffers for the large per-tile temporaries of host-side archival stages (source fingerprints in
    :mod:`reelcache.lineage_commit`, finalization's projections and quantization in :mod:`reelcache.node_packets`).

    A fresh temporary of tens or hundreds of MiB is a new anonymous mapping whose pages are faulted in (and zeroed) on
    first touch, so a stage that allocates one per tile runs at the speed of the host's page faults: a LongCat node's
    fingerprints and finalization faulted in ~600 GB, and ran 1.4-2.5x slower on a host whose memory was too
    fragmented for transparent huge pages (4.5x with THP off). Buffers taken from a workspace are faulted in once per
    workspace and reused; the arithmetic writing them is the same, so every result is bitwise unchanged.

    ``get(name, shape, dtype, device)`` returns a contiguous view of a buffer that grows to the largest request of its
    ``(name, dtype)``. Only host buffers are pooled: on CUDA it returns a fresh tensor (the caching allocator already
    reuses device blocks), so device memory behaviour is unchanged. Not thread-safe: one workspace per call.

    Lifetime rules (Step 7 review, freeze E):

    * a view is valid until the next ``get`` of the same name; never hold two live views of one name;
    * routines that share a workspace in :func:`run_lockstep` must not keep a view across a ``yield`` (the other
      routines reuse the buffers between their steps); a routine's own workspace is released before it yields, so
      N routines in lockstep never hold N sets of buffers;
    * memory is bounded by one buffer per name (its largest request) and freed with the workspace (``release`` or
      garbage collection). ``nbytes`` / ``peak_bytes`` count this workspace; :func:`workspace_stats` counts every live
      workspace of the process (recorded in the archive pipeline's stats).
    """

    def __init__(self):
        self._buffers: dict = {}
        self.nbytes = 0
        self.peak_bytes = 0
        self._held = [0]  # bytes this workspace holds, released from the process total when it is collected
        weakref.finalize(self, _workspace_released, self._held)

    def get(self, name: str, shape: Sequence[int], dtype: torch.dtype, device: Any) -> torch.Tensor:
        device = torch.device(device)
        shape = tuple(int(size) for size in shape)
        if device.type != "cpu":
            return torch.empty(shape, dtype=dtype, device=device)
        count = math.prod(shape)
        buffer = self._buffers.get((name, dtype))
        if buffer is None or buffer.numel() < count:
            if buffer is not None:  # release the smaller buffer before allocating the larger one
                self._account(-self._buffers.pop((name, dtype)).numel() * buffer.element_size())
            buffer = torch.empty(count, dtype=dtype)
            self._buffers[(name, dtype)] = buffer
            self._account(count * buffer.element_size())
        return buffer[:count].view(shape)

    def release(self, name: str | None = None) -> int:
        """Drop the buffers of ``name`` (every dtype), or all of them; returns the bytes released."""
        released = 0
        for key in [key for key in self._buffers if name is None or key[0] == name]:
            buffer = self._buffers.pop(key)
            released += buffer.numel() * buffer.element_size()
        self._account(-released)
        return released

    def _account(self, delta: int) -> None:
        self.nbytes += delta
        self.peak_bytes = max(self.peak_bytes, self.nbytes)
        self._held[0] += delta
        with _WORKSPACE_LOCK:
            _drain_released()
            _WORKSPACE_BYTES["live_bytes"] += delta
            _WORKSPACE_BYTES["peak_bytes"] = max(_WORKSPACE_BYTES["peak_bytes"], _WORKSPACE_BYTES["live_bytes"])
            if delta > 0:
                _WORKSPACE_BYTES["allocations"] += 1
                _WORKSPACE_BYTES["allocated_bytes"] += delta


_WORKSPACE_LOCK = threading.Lock()
_WORKSPACE_BYTES = {"live_bytes": 0, "peak_bytes": 0, "allocations": 0, "allocated_bytes": 0}
# Collected workspaces post their bytes here (deque appends are atomic): a finalizer that the garbage collector runs
# on a thread inside the lock never waits for it.
_RELEASED: collections.deque = collections.deque()


def _workspace_released(held: list) -> None:
    _RELEASED.append(held[0])
    held[0] = 0


def _drain_released() -> None:  # caller holds _WORKSPACE_LOCK
    while True:
        try:
            _WORKSPACE_BYTES["live_bytes"] -= _RELEASED.popleft()
        except IndexError:
            return


def workspace_stats(*, reset_peak: bool = False) -> dict[str, int]:
    """Host bytes of every live :class:`TileWorkspace` in this process: ``live_bytes`` now, ``peak_bytes`` since
    start (or the last reset), ``allocations`` / ``allocated_bytes`` (buffer growths, cumulative)."""
    with _WORKSPACE_LOCK:
        _drain_released()
        stats = dict(_WORKSPACE_BYTES)
        if reset_peak:
            _WORKSPACE_BYTES["peak_bytes"] = _WORKSPACE_BYTES["live_bytes"]
    return stats


def run_one(routine: Routine):
    """Run a single routine (one read per round)."""
    return run_lockstep([routine])[0]


def upload(tensors: Sequence[torch.Tensor], device: Any) -> list[torch.Tensor]:
    """``tensor.to(device)`` for many host tensors with one pinned host-to-device copy per dtype.

    Values are unchanged; every result is a contiguous view starting at a
    :data:`UPLOAD_ALIGNMENT`-byte boundary of one device buffer. Tensors
    already on ``device`` are returned as they are.
    """
    device = torch.device(device)
    result: list = [None] * len(tensors)
    pending: dict[torch.dtype, list[int]] = {}
    for index, tensor in enumerate(tensors):
        if tensor.device == device or device.type != "cuda" or tensor.device.type != "cpu":
            result[index] = tensor.to(device)
        else:
            pending.setdefault(tensor.dtype, []).append(index)
    for dtype, indices in pending.items():
        size = torch.empty((), dtype=dtype).element_size()
        step = max(1, UPLOAD_ALIGNMENT // size)
        offsets, total = [], 0
        for index in indices:
            offsets.append(total)
            total += -(-tensors[index].numel() // step) * step
        host = torch.empty(total, dtype=dtype, pin_memory=True)
        for index, offset in zip(indices, offsets):
            count = tensors[index].numel()
            host[offset:offset + count].copy_(tensors[index].reshape(-1))
        buffer = host.to(device, non_blocking=True)
        for index, offset in zip(indices, offsets):
            tensor = tensors[index]
            result[index] = buffer[offset:offset + tensor.numel()].view(tensor.shape)
    return result


def download(tensors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """``tensor.cpu().contiguous()`` for many tensors with one device-to-host transfer per device and dtype."""
    result: list = [None] * len(tensors)
    groups: dict[tuple, list[int]] = {}
    for index, tensor in enumerate(tensors):
        if tensor.device.type == "cpu":
            result[index] = tensor.contiguous()
        else:
            groups.setdefault((tensor.device, tensor.dtype), []).append(index)
    for indices in groups.values():
        parts = [tensors[index].detach().reshape(-1) for index in indices]
        flat = torch.cat(parts) if len(parts) > 1 else parts[0]
        with blocked_wait("host_read"):  # the device synchronization allocates no device memory (dev22j8)
            flat = flat.cpu()
        offset = 0
        for index, part in zip(indices, parts):
            count = part.numel()
            result[index] = flat[offset:offset + count].view(tensors[index].shape).clone()
            offset += count
    return result


# -- routines mirroring reelcache.factorization ------------------------------------------------
def _finite_rows(tensor: torch.Tensor) -> torch.Tensor:
    """:func:`finite_flag` of a host matrix in row chunks of ~4M elements (the same flag without a tensor-size
    boolean temporary)."""
    step = max(1, (1 << 22) // max(1, tensor.shape[-1]))
    return torch.stack([finite_flag(tensor[start:start + step]) for start in range(0, tensor.shape[0], step)]).all()


def finite_flag(tensor: torch.Tensor) -> torch.Tensor:
    """Device flag of ``isfinite(x).all() and isfinite(x.float()).all()`` (lossy-input admission).

    The FP32 conversion is exact for 16-bit and FP32 inputs (finiteness unchanged), so only wider inputs are
    converted (no full-size temporary for BF16/FP16/FP32 sources)."""
    if tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return torch.isfinite(tensor).all()
    return torch.isfinite(tensor).all() & torch.isfinite(tensor.float()).all()


def source_energy_routine(tensor: torch.Tensor, *, tile_rows: int = 1024):
    """:func:`reelcache.factorization.measure_source_energy` with two batched reads."""
    started = time.perf_counter()
    if (
        not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
        or tensor.device.type == "meta" or tensor.ndim != 2
        or min(tensor.shape) < 1 or not tensor.is_floating_point()
    ):
        raise ConfigurationError("source energy requires a nonempty floating matrix")
    if type(tile_rows) is not int or tile_rows < 1:
        raise ConfigurationError("tile_rows must be a positive integer")
    (scale,) = yield (torch.maximum(tensor.amax().abs(), tensor.amin().abs()),)
    if not math.isfinite(scale):
        raise ConfigurationError("source energy requires finite input")
    graphs = current_graph_cache(tensor) if scale else None
    if scale and _accel_b.fp32("energy", tensor) and _accel_b.fused_ok(tensor):  # accel B2 (opt-in): one fused FP32 pass
        values = yield _accel_b.energy_fp32(tensor, scale)
        flags = [True]
    elif graphs is not None:  # dev22l (B3, opt-in): the same kernels replayed from a CUDA graph
        (packed,) = graphs.run(("energy", tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tile_rows),
                               _energy_function(tile_rows), (tensor,), (1.0 / scale,))
        (values,) = yield (packed,)
        values = (float(values[0]), bool(values[1]))
        flags = [True]
    else:
        total = torch.zeros((), dtype=torch.float64, device=tensor.device)
        flags = []
        if scale:
            for start in range(0, tensor.shape[0], tile_rows):
                block = tensor[start:start + tile_rows].double() / scale
                flags.append(torch.isfinite(block).all())
                total += block.square().sum()
        request = (total,) if not flags else (total, torch.stack(flags).all())
        values = yield request
    if flags and not values[1]:
        raise ConfigurationError("source energy requires finite input")
    scaled = values[0]
    if not math.isfinite(scaled) or (scale != 0 and scaled <= 0):
        raise ConfigurationError("invalid original source energy")
    norm = scale * math.sqrt(scaled)
    energy = norm * norm
    if not math.isfinite(norm) or not math.isfinite(energy):
        raise ConfigurationError("original source energy exceeds FP64")
    return SourceEnergy(scale, scaled, norm, energy, scale == 0, time.perf_counter() - started)


PRODUCTS = ("fp32", "fp64")


def _check_products(products: str) -> None:
    if products not in PRODUCTS:
        raise ConfigurationError(f"measurement products must be one of {PRODUCTS}")


def condition_terms(basis: torch.Tensor, *, products: str = "fp32") -> tuple[torch.Tensor, torch.Tensor]:
    """Device terms ``(eta, norms)`` of :func:`reelcache.factorization.fast_condition_bound`.

    ``products="fp64"`` forms the Gram in FP64 (independent of the process-wide TF32 setting; the
    FP32 rounding allowance of :func:`finish_condition` then only makes the bound conservative).
    """
    _check_products(products)
    products = _accel_b.condition_products(products, basis)  # accel B2 (opt-in)
    rounded = basis.to(dtype=torch.float32 if products == "fp32" else torch.float64)
    gram = mm(rounded, rounded.T)
    norms = gram.diagonal().abs().max()
    gram.diagonal().sub_(1.0)
    return gram.abs().sum(1).max(), norms


def finish_condition(basis: torch.Tensor, eta: float, norms: float) -> float:
    """Host part of :func:`reelcache.factorization.fast_condition_bound` (exact FP64 bound when not near-orthonormal)."""
    rows, channels = basis.shape
    unit = 2.0 ** -24
    gamma = channels * unit / (1 - channels * unit)
    eta += rows * gamma * norms + rows * unit
    if not math.isfinite(eta) or eta >= 1 / 3:
        return basis_condition_bound(basis)
    return (1 + eta) / (1 - eta)


def condition_routine(basis: torch.Tensor, *, products: str = "fp32"):
    """:func:`reelcache.factorization.fast_condition_bound` with one batched read."""
    eta, norms = yield condition_terms(basis, products=products)
    return finish_condition(basis, eta, norms)


def projection_routine(tensor: torch.Tensor, basis: torch.Tensor, *, rank: int | None = None,
                       tile_rows: int = 1024, max_condition: float = 1e6, scale: float | None = None,
                       condition: Any = "fast", products: str = "fp32", workspace: TileWorkspace | None = None):
    """:func:`reelcache.factorization.project_onto_basis` with one batched read (two without ``scale``).

    ``scale`` is ``max |tensor|`` when the caller already read it (for example
    from :func:`source_energy_routine` of the same tensor). ``condition`` is
    ``"fast"`` (:func:`~reelcache.factorization.fast_condition_bound` of
    ``basis``, the planner's ``_condition_bound``), a precomputed float, or
    ``None`` (the exact FP64 bound of the projected view). ``products="fp64"``
    (the ``fp32_verified_fp64`` matmul policy) computes the coefficients
    ``P = X B^T`` tile by tile in FP64 (rounded once to the stored FP32) and
    the reconstruction ``P B`` of the stored factors in FP64, so the measured
    residual never depends on the process-wide TF32 setting; ``"fp32"`` is the
    reference arithmetic (FP32 products). ``workspace`` (:class:`TileWorkspace`) holds the host temporaries of a
    contiguous host ``tensor`` (the FP32 widening and the FP64 residual tiles) across tiles and calls; results are
    bitwise identical with or without it.
    """
    _check_products(products)
    started = time.perf_counter()
    if (
        not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
        or tensor.ndim != 2 or min(tensor.shape) < 1
        or not tensor.is_floating_point() or tensor.device.type == "meta"
    ):
        raise ConfigurationError("projection requires a nonempty finite FP32-compatible matrix")
    products = _accel_b.products(products, tensor)  # accel B2 (opt-in)
    # Host temporaries of a contiguous host tensor go to reusable buffers or row chunks (same values, same layouts);
    # other tensors keep the original expressions.
    pooled = POOL_HOST_TEMPORARIES and tensor.device.type == "cpu" and tensor.is_contiguous()
    fused = not pooled and ((products == "fp32" and _accel_b.fp32("residual", tensor) and _accel_b.fused_ok(tensor))
                            or _accel_b.recon_fp32(products, tensor))
    graphs = (None if pooled or fused or condition != "fast" or _accel_b.dot2_coefficients(products, tensor)
              else current_graph_cache(tensor))
    if graphs is not None and isinstance(basis, torch.Tensor) and basis.device == tensor.device:
        return (yield from _graphed_projection(graphs, tensor, basis, rank=rank, tile_rows=tile_rows,
                                               max_condition=max_condition, scale=scale, products=products,
                                               started=started))
    finite_tensor = _finite_rows(tensor) if pooled else finite_flag(tensor)
    if not isinstance(basis, torch.Tensor) or basis.layout != torch.strided:
        raise ConfigurationError("reuse basis must be a dense torch.Tensor")
    if basis.device.type == "meta" or basis.dtype != torch.float16:
        raise ConfigurationError("reuse basis must be the actual stored FP16 basis")
    if basis.ndim != 2 or basis.shape[1] != tensor.shape[1] or not 1 <= basis.shape[0] <= tensor.shape[1]:
        raise ConfigurationError("stored basis must have shape [canonical_rank, channels]")
    finite_basis = torch.isfinite(basis).all()
    stored = basis.detach()
    rank = stored.shape[0] if rank is None else rank
    validate_rank(rank, tuple(tensor.shape))
    if rank > stored.shape[0]:
        raise ConfigurationError("requested rank exceeds the immutable stored basis")
    if type(tile_rows) is not int or tile_rows < 1:
        raise ConfigurationError("tile_rows must be a positive integer")
    if (
        isinstance(max_condition, bool) or not isinstance(max_condition, (int, float))
        or not math.isfinite(max_condition) or max_condition < 1
    ):
        raise ConfigurationError("max_condition must be finite and at least one")
    terms = condition_terms(basis, products=products) if condition == "fast" else ()
    view = stored[:rank].to(device=tensor.device)
    b32 = view.float()
    coefficient_started = time.perf_counter()
    work = (TileWorkspace() if workspace is None else workspace) if pooled else None
    owned = pooled and workspace is None  # released before every yield (lockstep routines hold no buffers)
    if products == "fp32":
        if pooled and tensor.dtype != torch.float32:
            wide = work.get("projection_wide", tensor.shape, torch.float32, tensor.device)
            wide.copy_(tensor)
            coefficients = mm(wide, b32.T)
            del wide
            if owned:
                work.release("projection_wide")  # used once per call: no reuse, so no reason to keep it
        else:
            coefficients = mm(tensor.float(), b32.T)
    elif _accel_b.dot2_coefficients(products, tensor):  # accel B2 (opt-in products_df): compensated FP32 GEMM
        b64 = view.double()
        coefficients = _accel_b.dot2_gemm(tensor, view)
    else:
        b64 = view.double()
        coefficients = torch.empty((tensor.shape[0], rank), dtype=torch.float32, device=tensor.device)
        for start in range(0, tensor.shape[0], tile_rows):
            stop = min(start + tile_rows, tensor.shape[0])
            coefficients[start:stop] = tensor[start:stop].double() @ b64.T
    finite_coefficients = torch.isfinite(coefficients).all()
    coefficient_seconds = time.perf_counter() - coefficient_started
    if scale is None:
        (scale,) = yield (torch.maximum(tensor.amax().abs(), tensor.amin().abs()),)
    guard_started = time.perf_counter()
    source_energy = torch.zeros((), dtype=torch.float64, device=tensor.device)
    error_energy = torch.zeros_like(source_energy)
    if fused:  # accel B2 (opt-in): FP32 reconstructions and one fused residual kernel per tile
        source_energy, error_energy = _accel_b.residual_fp32(tensor, coefficients, b32, tile_rows, scale)
    for start in range(0, 0 if fused else tensor.shape[0], tile_rows):
        stop = min(start + tile_rows, tensor.shape[0])
        if pooled:
            shape = (stop - start, tensor.shape[1])
            if products == "fp32":  # exact_blas.mm of host tensors is torch.mm; out= runs the same GEMM
                reconstructed = work.get("projection_reconstructed", shape, torch.float32, tensor.device)
                torch.mm(coefficients[start:stop], b32, out=reconstructed)
            else:
                narrow = work.get("projection_coefficients64", (stop - start, rank), torch.float64, tensor.device)
                narrow.copy_(coefficients[start:stop])
                reconstructed = work.get("projection_reconstructed64", shape, torch.float64, tensor.device)
                torch.mm(narrow, b64, out=reconstructed)
                del narrow  # views keep their buffer alive: none may outlive the routine's release
            source = work.get("projection_source", shape, torch.float64, tensor.device)
            source.copy_(tensor[start:stop])
            error = work.get("projection_error", shape, torch.float64, tensor.device)
            torch.sub(source, reconstructed, out=error)  # the reconstruction widens exactly, as .to(float64) does
            if scale:
                source.div_(scale)
                error.div_(scale)
            squared = work.get("projection_squared", shape, torch.float64, tensor.device)
            source_energy += torch.square(source, out=squared).sum()
            error_energy += torch.square(error, out=squared).sum()
            del source, error, squared, reconstructed
            continue
        if products == "fp32":
            reconstructed = mm(coefficients[start:stop], b32)
        else:
            reconstructed = coefficients[start:stop].double() @ b64
        source = tensor[start:stop].to(dtype=torch.float64)
        error = source - reconstructed.to(dtype=torch.float64)
        if scale:
            source = source / scale
            error = error / scale
        source_energy += source.square().sum()
        error_energy += error.square().sum()
    if owned:
        work.release()
    values = yield (finite_tensor, finite_basis, finite_coefficients, source_energy, error_energy, *terms)
    tensor_ok, basis_ok, coefficients_ok, source_squared, error_squared = values[:5]
    if not tensor_ok:
        raise ConfigurationError("projection requires a nonempty finite FP32-compatible matrix")
    if not basis_ok:
        raise ConfigurationError("stored basis must be finite")
    condition_started = time.perf_counter()
    if condition == "fast":
        bound = finish_condition(basis, values[5], values[6])
    elif condition is None:
        bound = basis_condition_bound(view)
    else:
        bound = condition
    condition_seconds = time.perf_counter() - condition_started
    if not math.isfinite(bound) or bound > max_condition:
        raise ConfigurationError("stored basis is singular or ill-conditioned")
    if not coefficients_ok:
        raise ConfigurationError("projection overflowed FP32")
    if not math.isfinite(source_squared) or not math.isfinite(error_squared):
        raise ConfigurationError("nonfinite direct residual energy")
    zero = scale == 0.0
    residual = 0.0 if zero else math.sqrt(error_squared / source_squared)
    if not math.isfinite(residual):
        raise ConfigurationError("nonfinite relative projection residual")
    return Projection(
        coefficients=coefficients.contiguous(), relative_residual=residual,
        source_norm=scale * math.sqrt(source_squared), residual_norm=scale * math.sqrt(error_squared),
        zero_input=zero, condition_bound=bound, seconds=time.perf_counter() - started,
        coefficient_seconds=coefficient_seconds, guard_seconds=time.perf_counter() - guard_started,
        condition_seconds=condition_seconds,
    )


def stored_energy_routine(matrix: torch.Tensor, rows16: torch.Tensor, rank: int, tile_rows: int, *,
                          scale: float | None = None, products: str = "fp32"):
    """:func:`reelcache.rank_method.stored_energy` (direct residual of the stored FP16 rows)."""
    view = rows16[:rank]
    projection = yield from projection_routine(matrix, view, rank=rank, tile_rows=tile_rows, scale=scale,
                                               condition="fast", products=products)
    return 1.0 if projection.zero_input else 1.0 - projection.relative_residual ** 2


def span_energy_routine(coefficients: torch.Tensor, rows16: torch.Tensor, source_energy: float, *,
                        workspace: TileWorkspace | None = None):
    """:func:`reelcache.rank_method.span_energy` with one batched read. With ``workspace``, a contiguous host
    ``coefficients`` matrix's FP64 copy and squares use its buffers (the squares keep the solution's strides, so the
    sum runs in the same order): the same value."""
    if source_energy <= 0:
        return 1.0
    if _accel_b.fp32("span", coefficients) and rows16.dtype == torch.float16:  # accel B2 (opt-in)
        value = yield from _accel_b.span_energy_fp32(coefficients, rows16, source_energy)
        if value is not None:
            return value
    rows = rows16.double()
    factor, info = torch.linalg.cholesky_ex(rows @ rows.T)
    if (workspace is not None and POOL_HOST_TEMPORARIES and coefficients.device.type == "cpu"
            and coefficients.is_contiguous()):
        wide = workspace.get("span_coefficients", coefficients.shape, torch.float64, coefficients.device)
        wide.copy_(coefficients)
        solved = torch.linalg.solve_triangular(factor, wide.T, upper=False)
        del wide
        if solved.is_contiguous() or solved.mT.is_contiguous():  # dense: the strided view stays inside the buffer
            flat = workspace.get("span_squared", (solved.numel(),), torch.float64, coefficients.device)
            squared = flat.as_strided(solved.shape, solved.stride())
            total = torch.square(solved, out=squared).sum()
            del squared, flat
        else:
            total = solved.square().sum()
    else:
        solved = torch.linalg.solve_triangular(factor, coefficients.double().T, upper=False)
        total = solved.square().sum()
    info, total = yield (info, total)
    if info != 0:
        raise IntegrityError("stored basis rows are singular")
    return total / source_energy


# -- CUDA-graph forms (dev22l, ported to the accel B tree; reelcache.graph_capture; opt-in fitter_cuda_graphs) -----------
# Pure device functions with the eager routines' kernels in the same order. The one change: a division by the host
# scale becomes a multiply by its reciprocal held in a device scalar ``inv``. CUDA evaluates ``x / scale`` (a CPU
# scalar) as ``x * (1.0 / scale)`` with the reciprocal computed on the host in the op-math type, so the values are
# identical. Every function returns a tuple of tensors (a packed float64 vector of the values to read, plus the
# coefficients where callers keep them).
def _energy_function(tile_rows: int):
    def energy(tensor, inv):
        total = torch.zeros((), dtype=torch.float64, device=tensor.device)
        flags = []
        for start in range(0, tensor.shape[0], tile_rows):
            block = tensor[start:start + tile_rows].double() * inv
            flags.append(torch.isfinite(block).all())
            total += block.square().sum()
        return (torch.stack((total, torch.stack(flags).all().double())),)
    return energy


def _graph_front(tensor, basis, rank, tile_rows, products):
    """Flags, condition terms, the rank view's FP32/FP64 forms and the coefficients (``projection_routine``'s device
    work before the residual)."""
    finite_tensor = finite_flag(tensor)
    finite_basis = torch.isfinite(basis).all()
    eta, norms = condition_terms(basis, products=products)
    view = basis[:rank]
    b32 = view.float()
    b64 = None
    if products == "fp32":
        coefficients = mm(tensor.float(), b32.T)
    else:
        b64 = view.double()
        coefficients = torch.empty((tensor.shape[0], rank), dtype=torch.float32, device=tensor.device)
        for start in range(0, tensor.shape[0], tile_rows):
            stop = min(start + tile_rows, tensor.shape[0])
            coefficients[start:stop] = tensor[start:stop].double() @ b64.T
    finite_coefficients = torch.isfinite(coefficients).all()
    return finite_tensor, finite_basis, eta, norms, b32, b64, coefficients, finite_coefficients


def _graph_residual(tensor, coefficients, b32, b64, tile_rows, products, inv):
    """``projection_routine``'s residual tiles; ``inv`` None: no scaling (a zero scale)."""
    source_energy = torch.zeros((), dtype=torch.float64, device=tensor.device)
    error_energy = torch.zeros_like(source_energy)
    for start in range(0, tensor.shape[0], tile_rows):
        stop = min(start + tile_rows, tensor.shape[0])
        if products == "fp32":
            reconstructed = mm(coefficients[start:stop], b32)
        else:
            reconstructed = coefficients[start:stop].double() @ b64
        source = tensor[start:stop].to(dtype=torch.float64)
        error = source - reconstructed.to(dtype=torch.float64)
        if inv is not None:
            source = source * inv
            error = error * inv
        source_energy += source.square().sum()
        error_energy += error.square().sum()
    return source_energy, error_energy


def _projection_function(rank, tile_rows, products):
    """Known scale: the whole projection in one function (one graph)."""
    def projection(tensor, basis, inv):
        finite_tensor, finite_basis, eta, norms, b32, b64, coefficients, finite_coefficients = _graph_front(
            tensor, basis, rank, tile_rows, products)
        source_energy, error_energy = _graph_residual(tensor, coefficients, b32, b64, tile_rows, products, inv)
        packed = torch.stack((finite_tensor.double(), finite_basis.double(), finite_coefficients.double(),
                              source_energy, error_energy, eta.double(), norms.double()))
        return packed, coefficients
    return projection


def _projection_front_function(rank, tile_rows, products):
    """Unknown scale (finalization): the front and the scale ``max(|max x|, |min x|)``."""
    def front(tensor, basis):
        finite_tensor, finite_basis, eta, norms, _, _, coefficients, finite_coefficients = _graph_front(
            tensor, basis, rank, tile_rows, products)
        scale = torch.maximum(tensor.amax().abs(), tensor.amin().abs())
        packed = torch.stack((finite_tensor.double(), finite_basis.double(), finite_coefficients.double(),
                              eta.double(), norms.double(), scale.double()))
        return packed, coefficients
    return front


def _projection_residual_function(rank, tile_rows, products, scaled):
    def residual(tensor, basis, coefficients, inv):
        view = basis[:rank]
        b32 = view.float()
        b64 = None if products == "fp32" else view.double()
        source_energy, error_energy = _graph_residual(tensor, coefficients, b32, b64, tile_rows, products,
                                                      inv if scaled else None)
        return (torch.stack((source_energy, error_energy)),)
    return residual


def _graphed_projection(graphs, tensor, basis, *, rank, tile_rows, max_condition, scale, products, started):
    """:func:`projection_routine` (``condition="fast"``, device tensors) through ``graphs``: the same checks, reads
    (values packed per read) and results."""
    if basis.layout != torch.strided:
        raise ConfigurationError("reuse basis must be a dense torch.Tensor")
    if basis.device.type == "meta" or basis.dtype != torch.float16:
        raise ConfigurationError("reuse basis must be the actual stored FP16 basis")
    if basis.ndim != 2 or basis.shape[1] != tensor.shape[1] or not 1 <= basis.shape[0] <= tensor.shape[1]:
        raise ConfigurationError("stored basis must have shape [canonical_rank, channels]")
    stored = basis.detach()
    rank = stored.shape[0] if rank is None else rank
    validate_rank(rank, tuple(tensor.shape))
    if rank > stored.shape[0]:
        raise ConfigurationError("requested rank exceeds the immutable stored basis")
    if type(tile_rows) is not int or tile_rows < 1:
        raise ConfigurationError("tile_rows must be a positive integer")
    if (
        isinstance(max_condition, bool) or not isinstance(max_condition, (int, float))
        or not math.isfinite(max_condition) or max_condition < 1
    ):
        raise ConfigurationError("max_condition must be finite and at least one")
    layout = (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tuple(stored.shape), tuple(stored.stride()),
              rank, tile_rows, products)
    coefficient_started = time.perf_counter()
    if scale is not None and scale:
        packed, coefficients = graphs.run(("projection",) + layout, _projection_function(rank, tile_rows, products),
                                          (tensor, stored), (1.0 / scale,))
        coefficient_seconds = time.perf_counter() - coefficient_started
        guard_started = time.perf_counter()
        (values,) = yield (packed,)
        tensor_ok, basis_ok, coefficients_ok = bool(values[0]), bool(values[1]), bool(values[2])
        source_squared, error_squared, eta, norms = (float(values[3]), float(values[4]), float(values[5]),
                                                     float(values[6]))
    else:
        packed, coefficients = graphs.run(("projection_front",) + layout,
                                          _projection_front_function(rank, tile_rows, products), (tensor, stored))
        coefficient_seconds = time.perf_counter() - coefficient_started
        if scale is None:
            (front,) = yield (packed,)
            scale = float(front[5])
        else:  # a given zero scale: the front's flags and terms are read with the residual
            front = None
        guard_started = time.perf_counter()
        (residual,) = graphs.run(("projection_residual", bool(scale)) + layout,
                                 _projection_residual_function(rank, tile_rows, products, bool(scale)),
                                 (tensor, stored, coefficients), (1.0 / scale if scale else 1.0,))
        if front is None:
            (front, values) = yield (packed, residual)
        else:
            (values,) = yield (residual,)
        tensor_ok, basis_ok, coefficients_ok = bool(front[0]), bool(front[1]), bool(front[2])
        eta, norms = float(front[3]), float(front[4])
        source_squared, error_squared = float(values[0]), float(values[1])
    if not tensor_ok:
        raise ConfigurationError("projection requires a nonempty finite FP32-compatible matrix")
    if not basis_ok:
        raise ConfigurationError("stored basis must be finite")
    condition_started = time.perf_counter()
    bound = finish_condition(basis, eta, norms)
    condition_seconds = time.perf_counter() - condition_started
    if not math.isfinite(bound) or bound > max_condition:
        raise ConfigurationError("stored basis is singular or ill-conditioned")
    if not coefficients_ok:
        raise ConfigurationError("projection overflowed FP32")
    if not math.isfinite(source_squared) or not math.isfinite(error_squared):
        raise ConfigurationError("nonfinite direct residual energy")
    zero = scale == 0.0
    residual_norm = 0.0 if zero else math.sqrt(error_squared / source_squared)
    if not math.isfinite(residual_norm):
        raise ConfigurationError("nonfinite relative projection residual")
    return Projection(
        coefficients=coefficients.contiguous(), relative_residual=residual_norm,
        source_norm=scale * math.sqrt(source_squared), residual_norm=scale * math.sqrt(error_squared),
        zero_input=zero, condition_bound=bound, seconds=time.perf_counter() - started,
        coefficient_seconds=coefficient_seconds, guard_seconds=time.perf_counter() - guard_started,
        condition_seconds=condition_seconds,
    )
