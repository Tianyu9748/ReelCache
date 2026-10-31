"""CUDA-graph replay of the fitter's measurement kernel sequences (dev22l, proposed; default off).

The fitter launches ~43k small kernels per node and spends ~40 % of its time alone launching them from Python, which
also holds the GIL a host-bound generation thread needs (DESIGN_D_tiers §15.9). About 60 % of those launches are
the FP64 measurement passes of :mod:`reelcache.deferred_reads`: source energy, certification, verification and the
finalize measurement. They are fixed per-tile sequences between two lockstep reads. With ``fitter_cuda_graphs`` the
archive pipeline gives each of its GPU threads a :class:`GraphCache` (:func:`graph_scope`). Those routines then run
their device work through :meth:`GraphCache.run`, which replays a graph captured for a signature seen before (shapes,
strides, dtypes, rank, tile rows, products, variant) and otherwise runs the same function eagerly.

**Bitwise.** A replay runs the captured kernels with the captured launch configurations on copies of the inputs in
static buffers, aligned like fresh allocations. The one change from the eager routines: a division by a host scalar
(which CUDA evaluates as a multiply by the host-computed reciprocal, ``div_true_kernel_cuda``) becomes a multiply by
that same reciprocal held in a device scalar, i.e. the same arithmetic. Graphed, eagerly run and legacy calls give
identical values; the CUDA tests and the gpucheck identity dumps verify it.

**Memory.** Each (thread, stream) has one private memory pool shared by all its graphs; replays on one stream never
overlap. A graph copies its outputs into static output buffers before it ends, so the pool only holds intermediates
that the next capture reuses. The static input/output buffers grow to the largest signature. ``budget_bytes`` bounds
pool + buffers: beyond it no new graph is captured (eager, recorded). :meth:`GraphCache.trim` drops every graph, pool
and buffer; the pipeline registers it as a device trim, and graphs are recaptured lazily afterwards.
"""

from __future__ import annotations

import collections
import contextlib
import threading
import time
from typing import Any, Callable, Sequence

import torch

_SCOPE = threading.local()
#: Byte alignment of every static buffer segment (the caching allocator's block alignment, as for fresh tensors).
ALIGNMENT = 512
DEFAULT_BUDGET_BYTES = 1 << 30
DEFAULT_MAX_GRAPHS = 512


@contextlib.contextmanager
def graph_scope(cache: "GraphCache | None"):
    """Routines of :mod:`reelcache.deferred_reads` run on this thread inside the block use ``cache`` (None: eager)."""
    previous = getattr(_SCOPE, "cache", None)
    _SCOPE.cache = cache
    try:
        yield cache
    finally:
        _SCOPE.cache = previous


def current_graph_cache(tensor: torch.Tensor | None = None) -> "GraphCache | None":
    """This thread's cache when ``tensor`` (if given) is a CUDA tensor on the cache's device, else None."""
    cache = getattr(_SCOPE, "cache", None)
    if cache is None or not cache.enabled:
        return None
    if tensor is not None and (tensor.device.type != "cuda" or tensor.device != cache.device):
        return None
    return cache


class _Entry:
    __slots__ = ("graph", "inputs", "scalars", "outputs")

    def __init__(self, graph, inputs, scalars, outputs):
        self.graph, self.inputs, self.scalars, self.outputs = graph, inputs, scalars, outputs


class _StreamState:
    """The graphs, pool and static buffers of one stream (replays and copies on it are stream-ordered)."""

    def __init__(self):
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: collections.OrderedDict = collections.OrderedDict()
        self.specs: dict = {}
        self.seen: collections.Counter = collections.Counter()
        self.buffers: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self.pool_bytes = 0

    def buffer_bytes(self) -> int:
        return sum(buffer.numel() * buffer.element_size() for buffer in self.buffers.values())

    def drop_graphs(self) -> None:
        """Drop every graph. Their pool then has no user, and the caching allocator never lets a new capture share a
        pool without users, so later captures start a fresh pool (the old one is freed by the allocator)."""
        self.graphs.clear()
        self.pool = torch.cuda.graph_pool_handle()
        self.pool_bytes = 0


def _extent(size: Sequence[int], stride: Sequence[int]) -> int:
    if any(dimension == 0 for dimension in size):
        return 0
    return 1 + sum((dimension - 1) * step for dimension, step in zip(size, stride))


def _aligned(count: int, element: int) -> int:
    step = max(1, ALIGNMENT // element)
    return -(-count // step) * step


class GraphCache:
    """Per-thread CUDA-graph cache (see the module docstring).

    ``capture_after``: a signature is captured at this sighting; earlier sightings run eagerly through the same
    function, which also warms its kernels and library handles on the stream. ``max_graphs`` per stream (LRU).
    """

    def __init__(self, device: Any, *, budget_bytes: int | None = None, max_graphs: int = DEFAULT_MAX_GRAPHS,
                 capture_after: int = 2, name: str = "fitter"):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CUDA graphs need a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.budget_bytes = DEFAULT_BUDGET_BYTES if budget_bytes is None else int(budget_bytes)
        self.max_graphs, self.capture_after, self.name = int(max_graphs), int(capture_after), name
        self.enabled = True
        self._lock = threading.RLock()
        self._states: dict[int, _StreamState] = {}
        self.counters = {"calls": 0, "replays": 0, "eager_calls": 0, "captures": 0, "capture_failures": 0,
                         "capture_seconds": 0.0, "budget_refusals": 0, "evictions": 0, "invalidations": 0,
                         "trims": 0, "trimmed_bytes": 0, "last_capture_failure": None}

    # -- public ------------------------------------------------------------------------------------------------
    def run(self, key: Any, fn: Callable[..., tuple[torch.Tensor, ...]], tensors: Sequence[torch.Tensor],
            scalars: Sequence[float] = ()) -> tuple[torch.Tensor, ...]:
        """``fn(*tensors, *scalar_tensors)`` (a pure device function returning a tuple of tensors). Each scalar
        arrives as a 0-d float64 device tensor. A replay returns fresh copies of the graph's outputs, so callers
        may keep them across later calls."""
        with self._lock:
            self.counters["calls"] += 1
            stream = torch.cuda.current_stream(self.device)
            state = self._states.get(stream.cuda_stream)
            if state is None:
                state = self._states[stream.cuda_stream] = _StreamState()
            entry = state.graphs.get(key)
            if entry is not None:
                state.graphs.move_to_end(key)
                for view, tensor in zip(entry.inputs, tensors):
                    view.copy_(tensor)
                for view, value in zip(entry.scalars, scalars):
                    view.fill_(value)
                entry.graph.replay()
                self.counters["replays"] += 1
                return tuple(output.clone() for output in entry.outputs)
            state.seen[key] += 1
            spec = state.specs.get(key)
            if spec is not None and spec != "failed" and state.seen[key] >= self.capture_after:
                captured = self._capture(state, key, fn, tensors, scalars, spec)
                if captured is not None:
                    return captured
            outputs = self._eager(fn, tensors, scalars)
            if spec is None:
                state.specs[key] = tuple((tuple(output.shape), output.dtype) for output in outputs)
            return outputs

    def trim(self, reason: str = "trim") -> int:
        """Drop every graph, pool and static buffer (recaptured on demand); returns the device bytes released."""
        with self._lock:
            freed = sum(state.pool_bytes + state.buffer_bytes() for state in self._states.values())
            graphs = sum(len(state.graphs) for state in self._states.values())
            self._states.clear()
            self.counters["trims"] += 1
            self.counters["trimmed_bytes"] += freed
            self.counters["last_trim"] = {"reason": reason, "bytes": freed, "graphs": graphs}
            return freed

    def close(self) -> None:
        with self._lock:
            self.enabled = False
            self._states.clear()

    def memory_bytes(self) -> int:
        with self._lock:
            return sum(state.pool_bytes + state.buffer_bytes() for state in self._states.values())

    def stats(self) -> dict[str, Any]:
        with self._lock:
            states = list(self._states.values())
            return {**self.counters, "name": self.name, "device": str(self.device), "streams": len(states),
                    "graphs": sum(len(state.graphs) for state in states),
                    "signatures": sum(len(state.specs) for state in states),
                    "pool_bytes": sum(state.pool_bytes for state in states),
                    "buffer_bytes": sum(state.buffer_bytes() for state in states),
                    "budget_bytes": self.budget_bytes, "max_graphs": self.max_graphs,
                    "capture_after": self.capture_after}

    # -- internals ---------------------------------------------------------------------------------------------
    def _eager(self, fn, tensors, scalars):
        self.counters["eager_calls"] += 1
        values = [torch.full((), value, dtype=torch.float64, device=self.device) for value in scalars]
        return tuple(fn(*tensors, *values))

    def _views(self, state: _StreamState, role: str, layouts: Sequence[tuple]) -> list[torch.Tensor] | None:
        """Views of ``state``'s static ``role`` buffers for ``layouts`` [(size, stride, dtype)], grouped per dtype,
        each segment at an aligned offset. Growing a buffer invalidates the stream's graphs (their pointers)."""
        needs: dict[torch.dtype, int] = {}
        offsets = []
        for size, stride, dtype in layouts:
            element = torch.empty((), dtype=dtype).element_size()
            offset = needs.get(dtype, 0)
            offsets.append(offset)
            needs[dtype] = offset + _aligned(_extent(size, stride), element)
        grow = {dtype: count for dtype, count in needs.items()
                if (role, dtype) not in state.buffers or state.buffers[(role, dtype)].numel() < count}
        if grow:
            added = sum(count * torch.empty((), dtype=dtype).element_size() for dtype, count in grow.items())
            if self.memory_bytes() + added > self.budget_bytes:
                return None
            if state.graphs and any((role, dtype) in state.buffers for dtype in grow):
                state.drop_graphs()  # a replaced buffer moves: graphs captured on the old pointers go
                self.counters["invalidations"] += 1
            for dtype, count in grow.items():
                state.buffers[(role, dtype)] = torch.empty(count + count // 4, dtype=dtype, device=self.device)
        return [state.buffers[(role, dtype)][offset:offset + max(1, _extent(size, stride))]
                .as_strided(size, stride) for (size, stride, dtype), offset in zip(layouts, offsets)]

    def _capture(self, state, key, fn, tensors, scalars, spec):
        if self.memory_bytes() > self.budget_bytes:
            self.counters["budget_refusals"] += 1
            return None
        inputs = self._views(state, "in", [(tuple(t.shape), tuple(t.stride()), t.dtype) for t in tensors])
        scalar_views = self._views(state, "scalar", [((), (), torch.float64) for _ in scalars])
        outputs = self._views(state, "out", [(shape, _contiguous_stride(shape), dtype) for shape, dtype in spec])
        if inputs is None or scalar_views is None or outputs is None:
            self.counters["budget_refusals"] += 1
            return None
        for view, tensor in zip(inputs, tensors):
            view.copy_(tensor)
        for view, value in zip(scalar_views, scalars):
            view.fill_(value)
        graph = torch.cuda.CUDAGraph()
        clock = time.perf_counter()
        reserved = torch.cuda.memory_reserved(self.device)
        try:
            graph.capture_begin(pool=state.pool, capture_error_mode="thread_local")
            try:
                produced = fn(*inputs, *scalar_views)
                if len(produced) != len(outputs) or any(
                        tuple(item.shape) != tuple(view.shape) or item.dtype != view.dtype
                        for item, view in zip(produced, outputs)):
                    raise RuntimeError("graph function outputs changed between sightings")
                for view, item in zip(outputs, produced):
                    view.copy_(item)
                del produced
            finally:
                graph.capture_end()
        except Exception as exc:  # noqa: BLE001 - this signature runs eagerly from now on (recorded)
            del graph
            if not state.graphs:
                state.drop_graphs()
            state.specs[key] = "failed"
            self.counters["capture_failures"] += 1
            self.counters["last_capture_failure"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            return None
        state.pool_bytes += max(0, torch.cuda.memory_reserved(self.device) - reserved)
        self.counters["captures"] += 1
        self.counters["capture_seconds"] += time.perf_counter() - clock
        state.graphs[key] = _Entry(graph, inputs, scalar_views, outputs)
        if len(state.graphs) > self.max_graphs:
            state.graphs.popitem(last=False)  # the newest graph keeps the pool in use
            self.counters["evictions"] += 1
        graph.replay()
        self.counters["replays"] += 1
        return tuple(output.clone() for output in outputs)


def _contiguous_stride(shape: Sequence[int]) -> tuple[int, ...]:
    stride, step = [], 1
    for dimension in reversed(shape):
        stride.append(step)
        step *= max(1, dimension)
    return tuple(reversed(stride))


__all__ = ["ALIGNMENT", "DEFAULT_BUDGET_BYTES", "DEFAULT_MAX_GRAPHS", "GraphCache", "current_graph_cache",
           "graph_scope"]
