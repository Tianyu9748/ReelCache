"""Restore helpers: archived history -> engine RoPE -> native ``CausalSelfAttentionKVCache`` slots.

Only the attended context of the next chunk is restored (``VisibilityPlan``).
Every K/V write goes through :mod:`reelcache.restore_kernels`:

* RAW (:func:`restore_dense`): the exact captured pre-RoPE K / V
  (``DenseSource``); K is rotated by the engine's own ``rotate_key`` with the
  engine's own ``(cos, sin)`` at the recorded position, so the installed cache
  is bitwise the native cache.
* ``svd_hevc`` (:class:`LineageInstall`): the Step 6
  :class:`~reelcache.archive_restore.LineageRestorer` (packed NVDEC decode into
  device canvases, fused reconstruction ``P_hat Q`` with RoPE straight into the
  slots; ``RopeTable.from_cos_sin`` of the engine tables, or the engine's
  ``rotate_key`` with ``rope_mode="native"``), or the explicit
  ``SoftwareLineageRestorer``. No silent fallback.

:class:`AttentionKVLease` owns what a restore borrowed (store pins, staging,
reconstruction handles) until ``close``, which waits for every reconstruction
before releasing inputs.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import threading
import time
import traceback
from collections import OrderedDict
from typing import Any, Callable, Mapping, Sequence

import torch

from ... import restore_kernels as rk
from ...errors import ConfigurationError, IntegrityError, StateError, UnsupportedError
from ...types import MatrixKey
from .workload_telemetry import active_telemetry


class AttentionKVLease:
    """Resources of one restore; references never outlive :meth:`close`."""

    def __init__(self, *, store_lease: Any, plans: Mapping[str, Any], description: Mapping[str, Any]):
        self.plans = dict(plans)
        self.description = dict(description)
        self._store_lease = store_lease
        self._handles: list = []
        self._staging: list = []
        self._finalizers: list = []
        self.closed = False
        self.stats: dict[str, Any] = {"handles": 0, "staging_leases": 0, "jobs": 0, "seconds": {}}

    def track(self, *, handle=None, staging=None, finalizer: Callable[[], None] | None = None) -> None:
        if self.closed:
            raise StateError("attention KV lease is closed")
        if handle is not None:
            self._handles.append(handle)
            self.stats["handles"] += 1
        if staging is not None:
            self._staging.append(staging)
            self.stats["staging_leases"] += 1
        if finalizer is not None:
            self._finalizers.append(finalizer)

    @property
    def open_handles(self) -> int:
        return sum(not getattr(handle, "released", True) for handle in self._handles)

    def complete(self) -> AttentionKVLease:
        """The restore's records are complete (a plain restore's when it returns). An overlapped restore's archive
        sets its own ``complete`` (joins its calls, then fills the records) until :meth:`close`."""
        return self

    def wait(self, stream: Any = None) -> None:
        """Make ``stream`` (default: current) wait for every reconstruction (no host block)."""
        for handle in self._handles:
            waiter = getattr(handle, "wait", None)
            if callable(waiter):
                waiter(stream)

    def close(self) -> None:
        """Host-wait for every reconstruction, then release staging and pins (idempotent)."""
        if self.closed:
            return
        error = None
        # freeze F5: a layer-overlapped call still writing the caches finishes first (its failure is raised by
        # ``complete()``, not here). Its plain records stay in ``stats["overlap"]``; the readiness objects, the
        # archive's ``complete`` closure and the frames of a stored producer failure do not outlive the lease (a
        # stored exception's traceback frames can pin device memory: F2 census 4423272)
        calls = list(getattr(self, "readiness", None) or ())
        for _, readiness in calls:
            with contextlib.suppress(BaseException):
                readiness.join()
        if calls:
            records = [_readiness_record(readiness) for _, readiness in calls]
            overlap = self.stats.setdefault("overlap", {})
            for key in ("timelines", "producer", "errors"):
                overlap[key] = [record[key] for record in records]
            self.readiness = []
        self.__dict__.pop("complete", None)
        for handle in self._handles:
            try:
                release = getattr(handle, "release", None)
                if callable(release):
                    release()
                elif getattr(handle, "event", None) is not None:
                    handle.event.synchronize()
            except BaseException as exc:  # noqa: BLE001 - release everything, then report
                error = error or exc
        for staging in self._staging:
            try:
                staging.release()
            except BaseException as exc:  # noqa: BLE001
                error = error or exc
        for finalizer in self._finalizers:
            try:
                finalizer()
            except BaseException as exc:  # noqa: BLE001
                error = error or exc
        if self._store_lease is not None:
            self._store_lease.release()
        self.closed = True
        if error is not None:
            raise error

    def __enter__(self) -> AttentionKVLease:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _readiness_record(readiness: Any) -> dict[str, Any]:
    """Plain records of one finished overlapped call (Step 5c's ``LayerReadiness``); the locals of its stored
    failure's frames are cleared (the type, message and line numbers stay)."""
    record: dict[str, Any] = {"timelines": None, "producer": None, "errors": None}
    with contextlib.suppress(Exception):
        record["timelines"] = readiness.timeline()
    with contextlib.suppress(Exception):
        record["producer"] = dict(readiness.info or {})
    error = getattr(readiness, "error", None)
    if error is not None:
        record["errors"] = {"type": type(error).__name__, "message": str(error)[:2000]}
        seen: set[int] = set()
        while error is not None and id(error) not in seen:
            seen.add(id(error))
            traceback.clear_frames(error.__traceback__)
            error = error.__cause__ or error.__context__
    return record


def span_tables(plans: Mapping[str, Any], rope_cos_sin) -> dict[tuple[str, int], tuple[Any, Any]]:
    """Engine ``(cos, sin)`` of every attended span, keyed by ``(stream, global token start)``.

    Used as the engine returns them: tables computed under ``torch.inference_mode`` (SGLang) are inference
    tensors, which ``restore_kernels.RopeTable`` treats as immutable (its cached kernel copies are normal
    tensors)."""
    tables = {}
    for stream, plan in plans.items():
        for span in plan.spans:
            tables[(stream, span.token_start)] = rope_cos_sin(span.rope_start_frame, span.frames)
    return tables


#: Positions (latent frames) whose span tables the attach warm-up builds, in whole chunks of the profile's default
#: grid (CR-17-2): the visible windows and trained contexts of the bridge profiles (CausVid's 21 frames, LongLive 1.0's
#: 12-frame buffer, LongLive 2.0's 32-frame window) and their sessions of that length.
ROPE_WARM_FRAMES = 32
#: Device memory bound of the process-level span-table cache (least recently used entries are dropped beyond it).
ROPE_TABLE_CACHE_BYTES = 256 << 20


def _device_key(device: Any) -> str:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return str(device)


class RopeTableCache:
    """Process-level cache of the engine's ``(cos, sin)`` span tables (CR-17-2).

    Filled at attach for the profile's default grid (:func:`warm_rope_tables`) and by restores on a miss. Every entry
    is the engine's own output for the same ``(start_frame, frames, grid, device)`` (one engine per worker), so a
    cached table and a recomputed one are bitwise identical; the cache never changes how tables are computed. The
    tensors are never written (inference tensors when built under ``torch.inference_mode``, like SGLang's)."""

    def __init__(self, max_bytes: int = ROPE_TABLE_CACHE_BYTES):
        self.max_bytes = int(max_bytes)
        self._entries: OrderedDict[tuple, tuple[Any, Any]] = OrderedDict()
        self._sizes: dict[tuple, int] = {}
        self._lock = threading.Lock()
        self.bytes = 0
        self.stats_ = {"warm_builds": 0, "hits": 0, "builds": 0, "evictions": 0}

    @staticmethod
    def key(start_frame: Any, frames: Any, grid: Any, device: Any) -> tuple:
        return float(start_frame), int(frames), int(grid[0]), int(grid[1]), _device_key(device)

    def get(self, key: tuple) -> tuple[Any, Any] | None:
        with self._lock:
            tables = self._entries.get(key)
            if tables is not None:
                self._entries.move_to_end(key)
            return tables

    def put(self, key: tuple, tables: tuple[Any, Any]) -> None:
        size = sum(item.numel() * item.element_size() for item in tables if isinstance(item, torch.Tensor))
        with self._lock:
            if key in self._entries:
                return
            self._entries[key] = tables
            self._sizes[key] = size
            self.bytes += size
            while self.bytes > self.max_bytes and len(self._entries) > 1:
                oldest, _ = self._entries.popitem(last=False)
                self.bytes -= self._sizes.pop(oldest)
                self.stats_["evictions"] += 1

    def bind(self, compute: Callable[[Any, int], tuple[Any, Any]], *, grid: Any, device: Any) -> BoundRopeTables:
        """``rope_cos_sin`` of one restore through the cache (``compute`` = the request's engine tables)."""
        return BoundRopeTables(self, compute, tuple(grid), device)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {**self.stats_, "entries": len(self._entries), "bytes": self.bytes, "max_bytes": self.max_bytes}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class BoundRopeTables:
    """The request's ``rope_cos_sin(start_frame, frames)`` served from a :class:`RopeTableCache`; a miss calls the
    engine exactly as before and keeps its result. Counts this restore's hits and builds."""

    def __init__(self, cache: RopeTableCache, compute: Callable[[Any, int], tuple[Any, Any]], grid: tuple,
                 device: Any):
        self.cache, self.compute, self.grid, self.device = cache, compute, grid, device
        self.hits = self.builds = 0
        self.build_seconds = 0.0

    def __call__(self, start_frame: Any, frames: int) -> tuple[Any, Any]:
        key = self.cache.key(start_frame, frames, self.grid, self.device)
        tables = self.cache.get(key)
        if tables is not None:
            self.hits += 1
            with self.cache._lock:
                self.cache.stats_["hits"] += 1
            return tables
        started = time.perf_counter()
        tables = self.compute(start_frame, frames)
        self.build_seconds += time.perf_counter() - started
        self.builds += 1
        with self.cache._lock:
            self.cache.stats_["builds"] += 1
        self.cache.put(key, tables)
        return tables

    def report(self) -> dict[str, Any]:
        return {"hits": self.hits, "builds": self.builds, "build_seconds": self.build_seconds}


def warm_rope_tables(report: dict[str, Any], transformer: Any, profile: Any, device: Any, cache: RopeTableCache,
                     *, frames: int = ROPE_WARM_FRAMES) -> None:
    """Attach warm-up of the restore's RoPE span tables (CR-17-2): the engine's ``causal_rotary_cos_sin`` for every
    whole chunk starting below ``frames`` latent frames at the profile's default grid, called exactly as SGLang's
    causal stage calls it for a restore (``(chunk_frames, height, width, start_frame=float(start), device=...)``,
    under ``torch.inference_mode``), kept in ``cache``; the first restore of a session within that range builds no
    table. Torch's global thread settings are left alone."""
    grid = profile.default_grid()
    tables = getattr(transformer, "causal_rotary_cos_sin", None)
    if grid is None or tables is None:
        report["components"]["rope_tables"] = {"skipped": "no profile geometry or no engine RoPE seam"}
        return

    def warm():
        chunk = profile.chunk_frames
        built = 0
        with torch.inference_mode():
            for start in range(0, frames, chunk):
                key = cache.key(float(start), chunk, grid, device)
                if cache.get(key) is None:
                    cache.put(key, tables(chunk, grid[0], grid[1], start_frame=float(start), device=device))
                    built += 1
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        with cache._lock:
            cache.stats_["warm_builds"] += built
        return {"spans": built, "chunk_frames": chunk, "frames": frames, "grid": list(grid), "bytes": cache.bytes}

    warmup_component(report, "rope_tables", warm)


def _native_rope(rotate_key, cos_sin_of: Callable[[Any], tuple[Any, Any]]):
    def apply(pre_rope, job):
        cos, sin = cos_sin_of(job)
        return rotate_key(pre_rope.unsqueeze(0), cos, sin)[0]
    return apply


class DenseInstall:
    """The exact RAW install of dense pre-RoPE K / V (:func:`restore_dense`), one layer at a time.

    ``layer(L)`` issues layer ``L``'s writes on the current stream and returns its pending work (``release()``).
    :func:`restore_dense` runs the layers in order on the caller's stream; a layer-overlapped RAW install
    (``reelcache.overlapped_restore``, O4) runs the same calls on its own stream. Both write the same bits.

    * **Fast path** (``fast``, O4 cut 2): every attended span's rows on the restore device (GPU raw: read in place)
      or in host memory (host raw: copied host to device, ``non_blocking`` from the pinned buffers). No residency
      staging and no host synchronization: temporaries are freed in stream order, and the caller waits once, on
      :meth:`fence` (GPU-side events, or one host wait before the sources are released). V is copied into its slots.
      K is rotated by the engine's RoPE:
      - ``rope_per="layer"`` (default; the synchronous path): K of all spans gathered once (host rows copied straight
        into the gather; no gather for a single device span), ONE RoPE call per layer, scattered into the slots. Fewest
        calls: best where the calls are on the critical path (A100, CausVid h8: +24 ms vs resident, the per-span
        calls +35 ms).
      - ``rope_per="span"`` (the layer-overlapped install): one RoPE call per device span, straight from its rows (no
        gather): fewest bytes moved, which is what an install running beside the forward costs it (A100, CausVid h8
        GPU raw overlapped: +10 ms vs resident; with the gather +19 ms). Host spans are still gathered (one call): their
        host-to-device copies then run back to back (per-span calls between them left the copy engine idle: host raw
        h8 overlapped +159 ms instead of +51 ms).
      Copies from the host are issued first in each layer (K into the gather, V into its slots), then the RoPE calls,
      the K scatter and the device V copies.
      Taken when the rows are in the dtypes the staged path would write without conversion (K in ``key_dtype``, V in
      the cache's dtype).
    * **In place** (``in_place``, F6; default: when the engine's ``rotate_key`` takes ``out=``, e.g. SGLang's
      ``causal_wan_rotate_key`` with Step 5c's ``out`` patch): the engine rotates each span's K straight into its
      cache slot, one call per span in either ``rope_per`` (fewest bytes AND fewest calls: no gather, no rotation
      output, no scatter copy). GPU raw then allocates nothing (``scratch_bytes`` 0); host spans keep their K gather
      (the host-to-device landing buffer) and are rotated from it into their slots. Same kernel, so the same bits.
      Taken per span when the slot can receive the rotation as is: the slot's dtype is the rotation's, or narrower
      for float32 keys (RAVEN: the engine rounds once, as the cast on store does), and its last dimension has unit
      stride; else the output-tensor path.
    * **Staged path** (freeze F4's ``restore_dense``, ``fast=False`` or other dtypes): per span and target, the rows
      staged on the device through ``residency``, then one reconstruction of the layer with the engine's RoPE
      (``native``) for K; ``release()`` waits for it, then frees its staging (bounded staging: one layer at a time).

    The engine's RoPE is elementwise per token, so rotating the gathered K in one call writes the bits of one call per
    span. Keys are rotated in their capture dtype (``key_dtype``, default the cache dtype) and cast on store, as the
    engine rotates its attention-input keys and stores them in the cache dtype. The engine's span tables are computed
    here, in the caller's thread (``tables_seconds``)."""

    def __init__(self, *, plans: Mapping[str, Any], caches: Mapping[str, Any], rope_cos_sin, rotate_key,
                 device: torch.device, residency: Any, grid: tuple[int, int],
                 source: Callable[[Any, str, MatrixKey], torch.Tensor], key_dtype: torch.dtype | None = None,
                 non_blocking: bool = False, fast: bool = True, rope_per: str = "layer",
                 in_place: bool | None = None):
        if rope_per not in ("layer", "span"):
            raise ConfigurationError("rope_per must be 'layer' or 'span'")
        if in_place is not None and type(in_place) is not bool:
            raise ConfigurationError("in_place must be a bool or None (the engine's rotate_key takes out=)")
        self.in_place = bool(fast) and (takes_out(rotate_key) if in_place is None else in_place)
        started = time.perf_counter()
        self.tables = span_tables(plans, rope_cos_sin)
        self.spans = [(stream, span) for stream, plan in plans.items() for span in plan.spans]
        self.fast, self.rope_per = bool(fast), rope_per
        self.fast_tables, self._subset_tables = None, {}
        if self.fast and rope_per == "layer" and not self.in_place and self.spans:  # the layer's single call's tables
            tables = [self.tables[(stream, span.token_start)] for stream, span in self.spans]
            self.fast_tables = tables[0] if len(tables) == 1 else (torch.cat([cos for cos, _ in tables]),
                                                                   torch.cat([sin for _, sin in tables]))
        self.tables_seconds = time.perf_counter() - started
        self.caches, self.rotate_key, self.residency, self.source = caches, rotate_key, residency, source
        self.device, self.grid, self.key_dtype = torch.device(device), tuple(grid), key_dtype
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.non_blocking = bool(non_blocking)
        self.jobs = self.bytes = self.staged_rows = self.in_place_rows = self.host_rows = 0
        self.fast_layers = self.staged_layers = self.rope_calls = self.in_place_ropes = self.scratch_bytes = 0
        self.decode_seconds = self.launch_seconds = self.release_seconds = 0.0
        # workload telemetry of the restore window open on the constructing thread (None: off): per-layer ``decode``
        # (rows read, host-to-device copies issued) then ``install`` (RoPE into the slots, V copies) spans
        self.telemetry, self._step = active_telemetry(), None

    def _rows(self, span: Any, stream: str, key: MatrixKey) -> torch.Tensor:
        values = self.source(span, stream, key)
        rows = values[span.node_offset:span.node_offset + span.length]
        if rows.shape[0] != span.length:
            raise IntegrityError(f"archived range is shorter than the attended span {span.to_dict()}")
        return rows

    def layer(self, layer: int) -> DenseLayer:
        """Issue layer ``layer`` on the current stream; returns its pending work (see the class docstring)."""
        started = time.perf_counter()
        if self.telemetry is not None:
            self._step = self.telemetry.begin("decode")
        rows = [(stream, span, self._rows(span, stream, MatrixKey(layer, "K")),
                 self._rows(span, stream, MatrixKey(layer, "V"))) for stream, span in self.spans]
        if self.fast and rows and all(self._fast_rows(stream, layer, k, v) for stream, _, k, v in rows):
            pending = self._fast_layer(layer, rows, started)
        else:
            pending = self._staged_layer(layer, rows, started)
        if self.telemetry is not None:
            self.telemetry.end(self._step)
        return pending

    def _fast_rows(self, stream: str, layer: int, k: torch.Tensor, v: torch.Tensor) -> bool:
        cache = self.caches[stream][layer]
        return (k.dtype == (self.key_dtype or cache.k.dtype) and v.dtype == cache.v.dtype
                and k.is_contiguous() and v.is_contiguous()
                and all(rows.device == self.device or rows.device.type == "cpu" for rows in (k, v)))

    def _tables_for(self, indices: tuple[int, ...]) -> tuple[Any, Any]:
        """The engine tables of the spans ``indices`` (span order), concatenated for one RoPE call over them."""
        if len(indices) == len(self.spans) and self.fast_tables is not None:
            return self.fast_tables
        tables = self._subset_tables.get(indices)
        if tables is None:
            parts = [self.tables[(self.spans[i][0], self.spans[i][1].token_start)] for i in indices]
            tables = parts[0] if len(parts) == 1 else (torch.cat([cos for cos, _ in parts]),
                                                       torch.cat([sin for _, sin in parts]))
            self._subset_tables[indices] = tables
        return tables

    def _rotate(self, keys: torch.Tensor, tokens: int, heads: int, head_dim: int, tables: tuple[Any, Any]):
        result = self.rotate_key(keys.reshape(tokens, heads, head_dim).unsqueeze(0), tables[0], tables[1])[0]
        self.rope_calls += 1
        if not isinstance(result, torch.Tensor) or result.numel() != tokens * heads * head_dim:
            raise IntegrityError("native RoPE callable returned a tensor of the wrong size")
        self.scratch_bytes += result.numel() * result.element_size()
        return result.reshape(tokens, heads, head_dim)

    def _into(self, slot: torch.Tensor, keys: torch.Tensor) -> bool:
        """Whether the engine can rotate ``keys`` straight into ``slot``: in place on; the slot's dtype is the
        rotation's, or narrower for float32 keys (RAVEN: float32 keys, bfloat16 cache; the engine rounds once, as the
        cast on store would); a unit-stride last dimension."""
        return self.in_place and rotates_into(keys.dtype, slot.dtype) and slot.stride(-1) == 1

    def _rotate_into(self, keys: torch.Tensor, slot: torch.Tensor, heads: int, head_dim: int,
                     tables: tuple[Any, Any]) -> None:
        tokens = slot.shape[0]
        out = slot.unsqueeze(0)
        result = self.rotate_key(keys.reshape(tokens, heads, head_dim).unsqueeze(0), tables[0], tables[1], out=out)
        self.rope_calls += 1
        self.in_place_ropes += 1
        if result is not None and not (isinstance(result, torch.Tensor) and result.data_ptr() == out.data_ptr()
                                       and result.shape == out.shape):
            raise IntegrityError("the engine's RoPE did not write its out= tensor")

    def _fast_layer(self, layer: int, rows: list, started: float) -> DenseLayer:
        first = self.caches[rows[0][0]][layer].k
        heads, head_dim = first.shape[-2], first.shape[-1]
        if self.rope_per == "span" or self.in_place:  # host spans gathered; device spans one call each, from their rows
            gather = tuple(i for i, (_, _, k, _) in enumerate(rows) if k.device != self.device)
            single = [i for i, (_, _, k, _) in enumerate(rows) if k.device == self.device]
        else:
            gather, single = tuple(range(len(rows))), []
        gathered = None
        if gather:  # copies in first, back to back: the host K into the gather (device K: one cat, or in place)
            keys = [rows[i][2] for i in gather]
            tokens = sum(rows[i][1].length for i in gather)
            if len(keys) == 1 and keys[0].device == self.device:
                gathered = keys[0]
            elif all(k.device == self.device for k in keys):
                gathered = torch.cat(keys)
                self.scratch_bytes += gathered.numel() * gathered.element_size()
            else:
                gathered = torch.empty((tokens, keys[0].shape[1]), dtype=keys[0].dtype, device=self.device)
                self.scratch_bytes += gathered.numel() * gathered.element_size()
                offset = 0
                for k in keys:
                    gathered[offset:offset + k.shape[0]].copy_(k, non_blocking=True)
                    offset += k.shape[0]
            for k in keys:
                if k.device == self.device:
                    self.in_place_rows += 1
                else:
                    self.host_rows += 1
        destinations = [(rk.NativeView(self.caches[stream][layer].k, "BTHD", token_offset=span.slot_start).slot(
            span.length), rk.NativeView(self.caches[stream][layer].v, "BTHD", token_offset=span.slot_start).slot(
            span.length)) for stream, span, _, _ in rows]
        for (stream, span, _, v), (_, slot) in zip(rows, destinations):
            if v.device != self.device:  # the host V straight into its slots, behind the host K
                slot.copy_(v.view(span.length, heads, head_dim), non_blocking=True)
                self.host_rows += 1
        decoded = time.perf_counter()
        if self.telemetry is not None:
            self._step = self.telemetry.next(self._step, "install")
        if gather:
            if all(self._into(destinations[i][0], gathered) for i in gather):
                offset = 0  # each span rotated from the gather straight into its slot
                for i in gather:
                    length = rows[i][1].length
                    self._rotate_into(gathered[offset:offset + length], destinations[i][0], heads, head_dim,
                                      self._tables_for((i,)))
                    offset += length
            else:
                result = self._rotate(gathered, gathered.shape[0], heads, head_dim, self._tables_for(gather))
                offset = 0
                for i in gather:
                    length = rows[i][1].length
                    destinations[i][0].copy_(result[offset:offset + length])
                    offset += length
        for i in single:
            _, span, k, _ = rows[i]
            if self._into(destinations[i][0], k):
                self._rotate_into(k, destinations[i][0], heads, head_dim, self._tables_for((i,)))
            else:
                destinations[i][0].copy_(self._rotate(k, span.length, heads, head_dim, self._tables_for((i,))))
            self.in_place_rows += 1
        for (_, span, _, v), (_, slot) in zip(rows, destinations):
            if v.device == self.device:
                slot.copy_(v.view(span.length, heads, head_dim))
                self.in_place_rows += 1
            self.bytes += 2 * span.length * heads * head_dim * v.element_size()
        self.decode_seconds += decoded - started
        self.launch_seconds += time.perf_counter() - decoded
        self.fast_layers += 1
        self.jobs += 2 * len(rows)
        return DenseLayer(self, None, [], 2 * len(rows))

    def _staged_layer(self, layer: int, rows: list, started: float) -> DenseLayer:
        by_table: dict[int, tuple[Any, Any]] = {}
        native = _native_rope(self.rotate_key, lambda job: by_table[id(job.rope.table)])
        jobs, staged, dtype = [], [], None
        try:
            for stream, span, k, v in rows:
                for target, values in (("K", k), ("V", v)):
                    buffer = self.caches[stream][layer].k if target == "K" else self.caches[stream][layer].v
                    dtype = buffer.dtype
                    nbytes = values.numel() * values.element_size()
                    staging = self.residency.acquire(nbytes, lambda values=values: self._stage(values))
                    staged.append(staging)
                    rope = None
                    if target == "K":
                        cos, sin = self.tables[(stream, span.token_start)]
                        table = rk.RopeTable.from_cos_sin(cos, sin)
                        by_table[id(table)] = (cos, sin)
                        rope = rk.RopePositions(table, 0)
                        self.rope_calls += 1
                    jobs.append(rk.ReconstructionJob(
                        rk.DenseSource(staging.tensor), rk.NativeView(buffer, "BTHD", token_offset=span.slot_start),
                        (span.frames, self.grid[0], self.grid[1]),
                        rope=rope, label=f"{stream}:{span.token_start}:{target}{layer}"))
                    self.bytes += nbytes
            decoded = time.perf_counter()
            if self.telemetry is not None:
                self._step = self.telemetry.next(self._step, "install")
            handle = rk.reconstruct(jobs, native_rope=native, native_rope_dtype=self.key_dtype or dtype)
            self.launch_seconds += time.perf_counter() - decoded
        except BaseException:
            for staging in staged:
                staging.release()
            raise
        finally:
            by_table.clear()
        self.decode_seconds += decoded - started
        self.staged_layers += 1
        self.jobs += len(jobs)
        return DenseLayer(self, handle, staged, len(jobs))

    def _stage(self, rows: torch.Tensor) -> torch.Tensor:
        self.staged_rows += 1
        return rows.to(self.device, copy=True, non_blocking=self.non_blocking).contiguous()

    def fence(self) -> Any:
        """After the last layer: the event of the current stream the fast path's writes and reads complete at (a
        ``restore_kernels.RestoreHandle``: ``wait(stream)`` on the GPU, ``release()`` one host wait), or None when no
        fast layer was issued or on the CPU. Release it before the sources (raw leases) are released."""
        if not self.fast_layers or self.device.type != "cuda":
            return None
        stream = torch.cuda.current_stream(self.device)
        event = torch.cuda.Event()
        event.record(stream)
        return rk.RestoreHandle(event=event, stream=stream, keepalive=None, stats={"fence": True})

    def stats(self) -> dict[str, Any]:
        return {"spans": len(self.spans), "jobs": self.jobs, "bytes": self.bytes, "fast": self.fast,
                "rope_per": self.rope_per, "in_place": self.in_place, "in_place_ropes": self.in_place_ropes,
                "scratch_bytes": self.scratch_bytes,
                "fast_layers": self.fast_layers, "staged_layers": self.staged_layers, "rope_calls": self.rope_calls,
                "staged_rows": self.staged_rows, "in_place_rows": self.in_place_rows, "host_rows": self.host_rows,
                "non_blocking": self.non_blocking, "tables_seconds": self.tables_seconds,
                "decode_seconds": self.decode_seconds, "launch_seconds": self.launch_seconds,
                "release_seconds": self.release_seconds}


#: Slot dtypes a float32 key rotation may be written into directly (SGLang's ``out=`` contract, F6 proposal).
NARROWER_OUT_DTYPES = (torch.bfloat16, torch.float16)


def takes_out(rotate_key: Any) -> bool:
    """Whether the engine's key rotation takes ``out=`` (write the rotated key into a given tensor; F6)."""
    try:
        return "out" in inspect.signature(rotate_key).parameters
    except (TypeError, ValueError):
        return False


def rotates_into(key_dtype: torch.dtype, slot_dtype: torch.dtype) -> bool:
    """Whether the engine's ``out=`` may receive the rotation of ``key_dtype`` keys in a ``slot_dtype`` slot: the
    same dtype, or a narrower one for float32 keys (rounded once, as the cast on store does). One rule for the
    install (``DenseInstall._into``) and its warm-up (:func:`in_place_rope_dtype`)."""
    return slot_dtype == key_dtype or (key_dtype == torch.float32 and slot_dtype in NARROWER_OUT_DTYPES)


class DenseLayer:
    """One issued layer of a :class:`DenseInstall`. ``release()``: a staged layer waits for its writes, then frees its
    staging; a fast layer holds nothing (its temporaries were freed in stream order)."""

    __slots__ = ("handle", "install", "jobs", "released", "staged")

    def __init__(self, install: DenseInstall, handle: Any, staged: list, jobs: int):
        self.install, self.handle, self.staged, self.jobs = install, handle, staged, jobs
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        if self.handle is None:
            return
        started = time.perf_counter()
        try:
            self.handle.release()  # waits for the layer's reconstruction (its inputs are the staging)
        finally:
            for staging in self.staged:
                staging.release()
            self.install.release_seconds += time.perf_counter() - started


def restore_dense(lease: AttentionKVLease, *, plans: Mapping[str, Any], caches: Mapping[str, Any], rope_cos_sin,
                  rotate_key, num_layers: int, device: torch.device, residency: Any, grid: tuple[int, int],
                  source: Callable[[Any, str, MatrixKey], torch.Tensor], key_dtype: torch.dtype | None = None,
                  fast: bool = True, rotated: Callable[[str, Any], bool] | None = None) -> None:
    """Exact RAW restore: dense pre-RoPE K / V per span, engine RoPE (``native``), layer by layer on the caller's
    stream (:class:`DenseInstall`). Staged layers are waited for one at a time (bounded staging); fast layers are not
    waited for: the lease holds the install's :meth:`~DenseInstall.fence`, which ``lease.wait()`` / ``close()`` wait
    on before the sources are released.

    Keys are rotated in their capture dtype (``key_dtype``, default the cache dtype) and cast on store, as the
    engine rotates its attention-input keys and stores them in the cache dtype.

    ``rotated(stream, span)`` (wl-w1b7): spans whose rows are already installed form -- post-RoPE K and V in the
    cache dtype, as a replay installed them (``raw_tier_promote="host"``) -- are copied into their slots as they
    are (no RoPE: bitwise what was installed), before each layer's RoPE spans; the lease holds their fence too."""
    direct: list[tuple[str, Any]] = []
    if rotated is not None:
        direct = [(stream, span) for stream, plan in plans.items() for span in plan.spans if rotated(stream, span)]
        if direct:
            plans = {stream: dataclasses.replace(plan, spans=tuple(span for span in plan.spans
                                                                   if not rotated(stream, span)))
                     for stream, plan in plans.items()}
    install = DenseInstall(plans=plans, caches=caches, rope_cos_sin=rope_cos_sin, rotate_key=rotate_key,
                           device=device, residency=residency, grid=grid, source=source, key_dtype=key_dtype,
                           fast=fast)
    lease.stats["seconds"]["rope_tables"] = install.tables_seconds
    lease.stats["spans"] = len(install.spans) + len(direct)
    for layer in range(num_layers):
        if direct:
            _copy_installed(direct, layer, caches, source, device)
            if not install.spans:
                continue
        pending = install.layer(layer)
        if pending.handle is not None:
            lease.track(handle=pending.handle)
        lease.stats["jobs"] += pending.jobs
        pending.release()
    fence = install.fence()
    if fence is not None:
        lease.track(handle=fence)
    if direct and torch.device(device).type == "cuda":  # the installed-form copies complete before the sources go
        stream = torch.cuda.current_stream(torch.device(device))
        event = torch.cuda.Event()
        event.record(stream)
        lease.track(handle=rk.RestoreHandle(event=event, stream=stream, keepalive=None, stats={"fence": True}))
    lease.stats["raw_install"] = {key: value for key, value in install.stats().items()
                                  if key in ("fast_layers", "staged_layers", "rope_calls", "in_place_rows",
                                             "host_rows", "staged_rows", "in_place_ropes", "scratch_bytes")}
    if direct:
        lease.stats["raw_install"]["installed_form_spans"] = len(direct)
    lease.stats["seconds"].update(decode=install.decode_seconds,
                                  reconstruct=install.launch_seconds + install.release_seconds)


def _copy_installed(spans: Sequence[tuple[str, Any]], layer: int, caches: Mapping[str, Any],
                    source: Callable[[Any, str, MatrixKey], torch.Tensor], device: Any) -> None:
    """wl-w1b7: copy installed-form rows (post-RoPE K, V in the cache dtype) straight into their slots."""
    for stream, span in spans:
        cache = caches[stream][layer]
        for target, buffer in (("K", cache.k), ("V", cache.v)):
            rows = source(span, stream, MatrixKey(layer, target))[span.node_offset:span.node_offset + span.length]
            if rows.shape[0] != span.length:
                raise IntegrityError(f"archived range is shorter than the attended span {span.to_dict()}")
            slot = rk.NativeView(buffer, "BTHD", token_offset=span.slot_start).slot(span.length)
            if rows.dtype != slot.dtype:
                raise IntegrityError(f"installed-form rows of {span.node_ref} are {rows.dtype}, the cache {slot.dtype}")
            slot.copy_(rows.view(slot.shape), non_blocking=rows.device.type == "cpu" and rows.is_pinned())


class LineageInstall:
    """Destination/RoPE callbacks of a lineage restore into the request's native caches."""

    def __init__(self, *, plans: Mapping[str, Any], caches: Mapping[str, Any], rope_cos_sin, rotate_key,
                 rope_mode: str):
        self.plans, self.caches = plans, caches
        self.tables = span_tables(plans, rope_cos_sin)
        self.slots = {(stream, span.token_start): span for stream, plan in plans.items() for span in plan.spans}
        self._ropes: dict[tuple[str, int], Any] = {}
        self._by_table: dict[int, tuple[Any, Any]] = {}
        self.rope_mode = rope_mode
        self.rotate_key = rotate_key
        self.single_stream = None if len(plans) != 1 else next(iter(plans))

    def ranges(self, stream: str | None = None) -> list[tuple[int, int]]:
        streams = [stream] if stream is not None else list(self.plans)
        return sorted({(span.token_start, span.token_start + span.length)
                       for name in streams for span in self.plans[name].spans})

    def _entry(self, entry) -> tuple[MatrixKey, str]:
        if isinstance(entry, MatrixKey):
            if self.single_stream is None:
                raise IntegrityError("a multi-stream restore received an entry without its stream")
            return entry, self.single_stream
        key, stream = entry
        return key, stream

    def _span(self, stream: str, tile) -> Any:
        span = self.slots.get((stream, int(tile.token_start)))
        if span is None:
            raise IntegrityError(f"decoded range at token {tile.token_start} is not attended by stream {stream!r}")
        return span

    def destination(self, node_id, entry, tile):
        key, stream = self._entry(entry)
        span = self._span(stream, tile)
        cache = self.caches[stream][key.layer]
        return rk.NativeView(cache.k if key.target == "K" else cache.v, "BTHD", token_offset=span.slot_start)

    def rope(self, node_id, entry, tile):
        key, stream = self._entry(entry)
        if key.target != "K":
            return None
        span = self._span(stream, tile)
        cached = self._ropes.get((stream, span.token_start))
        if cached is None:
            cos, sin = self.tables[(stream, span.token_start)]
            table = rk.RopeTable.from_cos_sin(cos, sin)
            self._by_table[id(table)] = (cos, sin)
            cached = self._ropes[(stream, span.token_start)] = rk.RopePositions(table, 0)
        return cached

    def options(self, dtype) -> dict[str, Any]:
        """Restorer options; ``dtype`` is the keys' capture dtype (the engine rotates in it, then casts)."""
        if self.rope_mode != "native":
            return {}
        return {"native_rope": _native_rope(self.rotate_key, lambda job: self._by_table[id(job.rope.table)]),
                "native_rope_dtype": dtype}


def kernel_warmup_shapes(profile: Any, model: Any, *, path: str, rope_mode: str = "fused",
                         grid: tuple[int, int] | None = None) -> list[Any]:
    """``restore_kernels.WarmupShape`` of every launch group the bridge's restores issue (CR-12p2-11).

    ``path="pictures"``: lineage restores (packed NVDEC canvases; K with fused RoPE, or the engine's RoPE in
    ``rope_mode="native"`` from pre-RoPE K in its capture dtype, V in the same calls), ranks 1 .. min(channels,
    chunk tokens). ``path="dense"``: RAW restores (dense rows, engine RoPE); K and V get separate shapes when their
    capture dtypes differ (RAVEN: FP32 K, BF16 V). The cache dtype is the profile's ``native_dtype``."""
    cache = getattr(torch, profile.native_dtype)
    key, value = (getattr(torch, model.dtype_for(target)) for target in ("K", "V"))
    heads, dim = profile.num_heads, profile.head_dim
    if path == "pictures":
        top = model.channels_for("K")
        if grid is not None:
            top = min(top, profile.chunk_frames * grid[0] * grid[1])
        if rope_mode == "native":
            return [rk.WarmupShape(heads, dim, dtype=cache, source="pictures", rope="native", native_rope_dtype=key,
                                   ranks=(1, top))]
        return [rk.WarmupShape(heads, dim, dtype=cache, source="pictures", rope="fused", ranks=(1, top))]
    if path != "dense":
        raise ValueError(f"unknown restore path {path!r}")
    if key == value:
        return [rk.WarmupShape(heads, dim, dtype=cache, source="dense", rope="native", source_dtype=key,
                               native_rope_dtype=key)]
    return [rk.WarmupShape(heads, dim, dtype=cache, source="dense", rope="native", source_dtype=key,
                           native_rope_dtype=key),
            rk.WarmupShape(heads, dim, dtype=cache, source="dense", rope="none", source_dtype=value)]


def warmup_plain(value: Any) -> Any:
    """JSON-safe copy of a warm-up report (numbers, strings, nested mappings/sequences; others as repr)."""
    if isinstance(value, Mapping):
        return {str(key): warmup_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [warmup_plain(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def warmup_component(report: dict[str, Any], name: str, call: Callable[[], Any]) -> None:
    """Run one warm-up component and record its seconds and result; a failure is recorded, never raised."""
    started = time.perf_counter()
    try:
        detail = call()
        report["components"][name] = {"seconds": time.perf_counter() - started, "detail": warmup_plain(detail)}
    except Exception as exc:  # noqa: BLE001 - warm-up only moves cold-start cost; the bridge works without it
        report["components"][name] = {"seconds": time.perf_counter() - started, "error": repr(exc)}


def warm_restore_kernels(report: dict[str, Any], device: Any, shapes: Callable[[], list[Any]]) -> None:
    """``restore_kernels.warmup`` (restore_kernels d34662fc+) for the bridge's launch groups, when the core has it."""
    warmup = getattr(rk, "warmup", None)
    if warmup is None:
        report["components"]["restore_kernels"] = {"skipped": "this core's restore_kernels has no warmup()"}
        return
    warmup_component(report, "restore_kernels", lambda: warmup(device, shapes()))


def release_cached_blocks(report: dict[str, Any], device: Any) -> None:
    """Return the CUDA caching allocator's unused blocks after a warm-up (blocks cached on the warm-up's streams
    are unusable by other streams and, under memory pressure, slowed the first real restore's allocation)."""
    import gc

    if torch.device(device).type != "cuda":
        return

    def release():
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        return {"reserved_bytes": torch.cuda.memory_reserved(device)}

    warmup_component(report, "release", release)


def engine_rope_token_classes(chunk_tokens: int) -> tuple[int, int]:
    """Token counts covering both specializations of SGLang's Triton RoPE kernel (``_rotary_embedding_kernel``
    specializes its ``num_tokens`` argument on divisibility by 16): one chunk, and the nearest count of the other
    class above it (at most one chunk more, so two chunks' tables cover it)."""
    chunk = int(chunk_tokens)
    other = chunk + 16 - chunk % 16 if chunk % 16 else chunk + 8
    return chunk, other


def in_place_rope_dtype(rotate_key: Any, key_dtype: torch.dtype, profile: Any) -> torch.dtype | None:
    """The cache dtype a RAW install rotates ``key_dtype`` keys into with the engine's ``out=`` (``DenseInstall`` in
    place: the engine takes ``out=`` and the cache, the profile's ``native_dtype``, can receive the rotation as is,
    :func:`rotates_into`), else None (the output-tensor path)."""
    native = getattr(profile, "native_dtype", None)
    if not native or not takes_out(rotate_key):
        return None
    cache = getattr(torch, native)
    return cache if rotates_into(key_dtype, cache) else None


def warm_engine_rope(report: dict[str, Any], transformer: Any, profile: Any, model: Any, device: Any) -> None:
    """Process-level warm-up of the engine's own RoPE (``causal_rotary_cos_sin`` + ``causal_rotate_key``). RAW
    installs call it (``rope_mode="native"`` and every raw-tier range); ``restore_kernels.warmup`` does not. Pre-RoPE
    keys in their capture dtype (outputs discarded), at both token-count classes (:func:`engine_rope_token_classes`).

    Freeze F6, cut 9: a restore rotates one chunk's span or a gather of spans (host raw, the synchronous install), while
    the forward only ever rotates one chunk. In P5 4430781, the first RAVEN h8 host-raw restore (8 x 4680 = 37440
    tokens, FP32 keys) compiled the divisible variant inside the request: 380 ms, Triton cache entry WBRSIP2F.

    Freeze F6, with Step 5c's RoPE-into-slots: an engine rotation that takes ``out=`` makes every RAW install rotate
    each span's keys straight into its cache slot (``DenseInstall`` in place), a launch the output-tensor calls do not
    cover: SGLang's Triton kernel is compiled and autotuned per argument dtypes, and the slot is the cache's dtype
    (RAVEN: float32 keys into the bfloat16 cache). So that variant is warmed too, at both classes, into an output of
    the cache dtype (:func:`in_place_rope_dtype`; recorded as ``in_place_dtype``)."""
    grid = profile.default_grid()
    tables = getattr(transformer, "causal_rotary_cos_sin", None)
    rotate = getattr(transformer, "causal_rotate_key", None)
    if grid is None or tables is None or rotate is None:
        report["components"]["engine_rope"] = {"skipped": "no profile geometry or no engine RoPE seam"}
        return

    def warm():
        counts = engine_rope_token_classes(profile.chunk_frames * grid[0] * grid[1])
        cos, sin = tables(2 * profile.chunk_frames, grid[0], grid[1], start_frame=0, device=device)
        dtype = getattr(torch, model.dtype_for("K"))
        in_place = in_place_rope_dtype(rotate, dtype, profile)
        for tokens in counts:
            keys = torch.zeros(1, tokens, profile.num_heads, profile.head_dim, dtype=dtype, device=device)
            rotate(keys, cos[:tokens], sin[:tokens])
            if in_place is not None:
                rotate(keys, cos[:tokens], sin[:tokens], out=torch.empty(keys.shape, dtype=in_place, device=device))
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        detail = {"tokens": list(counts), "dtype": str(dtype)}
        if in_place is not None:
            detail["in_place_dtype"] = str(in_place)
        return detail

    warmup_component(report, "engine_rope", warm)


def check_restorable(plans: Mapping[str, Any]) -> None:
    """Lineage restores install whole committed ranges; a partial attended range is explicit."""
    for plan in plans.values():
        for span in plan.spans:
            if span.node_offset != 0 or span.length != span.range_tokens:
                raise UnsupportedError(
                    "the packed restore installs whole committed chunk ranges; a partial visibility span "
                    f"(offset {span.node_offset}, length {span.length} of {span.range_tokens}) is not supported")



# -- per-restore records (records only: nothing is waited for, changed or released) ---------------------------------
#: the decoder pool's stats (``DecoderPool.stats()``, 5c's pool A6r), pool level and per worker: gauges read when a
#: restore is submitted, cumulative counters (their change over the restore) and the running maximum of one job's
#: queue wait (submit -> start). The explicit warm-ups that yield to jobs (a concurrent seal-time prewarm overlapping
#: a restore) show as ``warm_up_running`` / ``warm_ups`` / ``warm_ups_moot`` (pool) and ``warming`` / ``warm_queued``
#: (worker)
POOL_GAUGES = ("queued_jobs", "warm_ups_in_flight", "warm_up_running")
POOL_COUNTERS = ("jobs", "jobs_started", "cold_jobs", "sessions_created", "queue_wait_seconds",
                 "creation_wait_seconds", "warm_up_wait_seconds", "warm_ups", "warm_ups_moot")
WORKER_GAUGES = ("queued_jobs", "warm_ups_in_flight", "warming", "warm_queued")
WORKER_COUNTERS = ("jobs", "jobs_started", "cold_jobs", "sessions_created", "queue_wait_seconds",
                   "creation_wait_seconds", "warm_up_wait_seconds")
POOL_MAX = "queue_wait_max_seconds"


def decoder_pool_snapshot(pool: Any) -> dict[str, Any] | None:
    """The archive's decoder pool stats now, pool level and per worker, or None without a pool (CPU, RAW archive) or
    readable stats. A record must never fail a restore, so any error gives None."""
    stats = getattr(pool, "stats", None)
    if not callable(stats):
        return None
    started = time.perf_counter()
    try:
        raw = stats()

        def pick(source: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
            return {key: source.get(key) for key in keys}

        snapshot: dict[str, Any] = {
            "pool": pick(raw, (*POOL_GAUGES, *POOL_COUNTERS, POOL_MAX)),
            "workers": [pick(worker, (*WORKER_GAUGES, *WORKER_COUNTERS, POOL_MAX))
                        for worker in raw.get("workers") or []]}
    except Exception:  # noqa: BLE001 - records only
        return None
    snapshot["seconds"] = time.perf_counter() - started
    return snapshot


def _pool_level(old: Mapping[str, Any], new: Mapping[str, Any], gauges: tuple[str, ...],
                counters: tuple[str, ...]) -> dict[str, Any]:
    def change(key: str) -> Any:
        if isinstance(old.get(key), (int, float)) and isinstance(new.get(key), (int, float)):
            return new[key] - old[key]
        return None

    before, after = old.get(POOL_MAX), new.get(POOL_MAX)
    during = {key: change(key) for key in counters}
    started_jobs, waited = during.get("jobs_started"), during.get("queue_wait_seconds")
    return {"at_submit": {key: old.get(key) for key in gauges},
            "during_restore": during,
            "mean_queue_wait_seconds": waited / started_jobs if started_jobs and waited is not None else None,
            # the running maximum rose: some job started in this window waited longer than any job before it
            POOL_MAX: {"before": before, "after": after,
                       "rose": (isinstance(before, (int, float)) and isinstance(after, (int, float))
                                and after > before)},
            "after": {key: new.get(key) for key in gauges}}


def decoder_pool_record(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    """One restore's decoder-pool record, pool level and per worker: the gauges when the restore was submitted (jobs
    queued, warm-ups in flight / running; per worker: warming, warm-ups queued), the counters' change over the restore
    (jobs submitted / started, cold jobs, sessions created, queue waits and the parts of them behind a session creation
    or a warm-up; pool: warm-ups started and dropped as moot; mean wait per started job) and the running maximum wait
    before / after. The pool is process-wide: the changes include other requests' decode work in the same window."""
    if before is None or after is None:
        return {"available": False}
    return {"available": True, **_pool_level(before["pool"], after["pool"], POOL_GAUGES, POOL_COUNTERS),
            "workers": [_pool_level(old, new, WORKER_GAUGES, WORKER_COUNTERS)
                        for old, new in zip(before["workers"], after["workers"])],
            "stats_seconds": before["seconds"] + after["seconds"]}


class GcPauses:
    """Records only: Python's garbage collections and the seconds spent in them, per generation, accumulated by one
    ``gc.callbacks`` hook per process (:func:`install_gc_pauses`); a restore records the change over its window. It
    never calls ``gc.freeze()``, ``gc.disable()`` or ``gc.collect()``."""

    def __init__(self) -> None:
        self.collections = [0, 0, 0]
        self.seconds = [0.0, 0.0, 0.0]
        self._started: float | None = None

    def __call__(self, phase: str, info: Mapping[str, Any]) -> None:
        # CPython runs one collection at a time (callbacks hold the GIL): one start time suffices
        if phase == "start":
            self._started = time.perf_counter()
        elif phase == "stop" and self._started is not None:
            generation = min(max(int(info.get("generation", 2)), 0), 2)
            self.collections[generation] += 1
            self.seconds[generation] += time.perf_counter() - self._started
            self._started = None

    def snapshot(self) -> tuple[tuple[int, ...], tuple[float, ...]]:
        return tuple(self.collections), tuple(self.seconds)


GC_PAUSES = GcPauses()
_GC_INSTALL_LOCK = threading.Lock()


def install_gc_pauses() -> GcPauses:
    """Install the process's GC-pause hook (idempotent: never appended twice, whatever the number of bridges)."""
    import gc

    with _GC_INSTALL_LOCK:
        if not any(callback is GC_PAUSES for callback in gc.callbacks):
            gc.callbacks.append(GC_PAUSES)
    return GC_PAUSES


def gc_pause_record(before: tuple | None, after: tuple | None) -> dict[str, Any]:
    """One restore's garbage collections: per generation, the collections and seconds in its window."""
    if before is None or after is None:
        return {"available": False}
    collections = [new - old for old, new in zip(before[0], after[0])]
    seconds = [new - old for old, new in zip(before[1], after[1])]
    return {"available": True, "collections": collections, "seconds": seconds, "total_seconds": sum(seconds)}


# -- per-restore tier records (freeze D; records only) ----------------------------------------------------------------
#: The tier of a resident resume: the bridge continues on the session's native KV still in the stage caches, decided
#: before any archive restore (``reelcache.raw_tier.TIER_NAMES``; an archive restore labels each attended range
#: ``gpu_raw`` / ``host_raw`` / ``compressed_prepared`` / ``compressed_cold`` in ``lease.stats["tiers"]``).
RESIDENT_TIER = "resident"


def configured_verification_mode(config: Any) -> str:
    """The configured payload verification of compressed reads, which every restore reports (coordinator 23:35):
    ``sync`` with ``verify_reads``, else ``off`` (5c's D-E record adds the asynchronous mode)."""
    return "sync" if getattr(config, "verify_reads", False) else "off"


def resident_lease_record(streams: Sequence[str], chunks: Mapping[int, Mapping[str, Any]], *,
                          policy: str | None = None, verification_mode: str | None = None) -> dict[str, Any]:
    """A resident resume's records in the archive restore's lease shape (``restore.lease.tiers`` and
    ``restore.lease.verification``): every generated chunk of every CFG stream is ``resident``. The stage caches hold
    the session's native KV, so nothing is read or moved (no bytes) and no payload is verified: the configured
    verification mode, ``verified_payloads`` 0 and no payload wait. ``node`` is None (the request path reads no store
    record for it)."""
    per_chunk = [{"stream": stream, "chunk": index, "node": None, "token_start": item.get("token_start"),
                  "tokens": item.get("num_tokens"), "tier": RESIDENT_TIER}
                 for stream in streams for index, item in sorted(chunks.items())]
    return {"tiers": {"per_chunk": per_chunk, "chunks": {RESIDENT_TIER: len(per_chunk)} if per_chunk else {},
                      "bytes": {}, "raw_seconds": 0.0, "policy": policy},
            "verification": {"mode": verification_mode, "payload_wait_seconds": 0.0, "verified_payloads": 0}}


__all__ = ["AttentionKVLease", "BoundRopeTables", "GC_PAUSES", "GcPauses", "LineageInstall", "RESIDENT_TIER",
           "RopeTableCache", "check_restorable", "configured_verification_mode", "decoder_pool_record",
           "decoder_pool_snapshot", "gc_pause_record", "install_gc_pauses", "resident_lease_record", "restore_dense",
           "span_tables", "warm_rope_tables"]
