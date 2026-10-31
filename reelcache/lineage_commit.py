"""Staged latest-method lineage commits shared by synchronous and pipelined archival.

A :class:`CommitJob` runs one node commit (chunk, KV recache or LongCat segment)
through five stages; :meth:`reelcache.lineage.LineageRuntimeMixin.commit_node`
runs them back to back on the caller's thread, and
:class:`reelcache.archive_pipeline.ArchivePipeline` runs them on its fitter
thread, CPU encode workers and ordered publisher. Both produce the same bytes.

1. ``plan`` (decision device): pinned-registry predictions plus the lineage's
   own unpublished observations, sketch/predict/fit/verify/grow/trim,
   certify/share/augment/refit per entry (``MatrixKey`` or ``(MatrixKey,
   stream)``); new basis row blocks and versions are stored (staged) at once so
   the next chunk can test them.
2. ``finalize`` (decision device): the exact stored representation of every
   entry is measured again, quantized and laid into its node's canonical
   canvas; one D2H copy per node. The device sources may be released after
   :attr:`CommitJob.sources_released_event`.
3. ``encode`` (CPU worker): libx265 node packs (thread count never changes
   bytes), optional validation decode and post-codec error.
4. ``store`` (CPU worker): packets without embedded bases (tiles reference
   their basis versions), matrix records, measured coefficient rates; every
   new object stays leased as staging until publication or failure.
5. ``publish`` (ordered): the node manifest (with its execution state) and a
   compare-and-swap of the lineage head; then the registry learns the commit's
   observations.

Byte rates for the refit estimate follow ``RankMethodConfig.rate_lag``: chunk
``n`` uses the rate measured ``rate_lag`` nodes above its parent (0 = the
parent itself, the synchronous reference), which lets a pipeline plan chunk
``n`` while chunk ``n - 1`` is still encoding without changing any decision.
The cumulative refit rule's per-entry ``refit_debt`` is a planning result,
not a measurement: it enters the active bindings when a range's versions are
stored, so later ranges, pipelined children, forks and reopened lineages all
continue from the debt their parent node records.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import threading
import time
import traceback
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import torch

from . import accel_b as _accel_b
from . import accel_c as _accel_c
from .basis_versions import rows_to_bytes
from .errors import CapacityError, ConfigurationError, IntegrityError, StateError, UnsupportedError
from .exact_blas import mm
from .deferred_reads import TileWorkspace, blocked_wait, step_phase, upload, yield_point
from .node_packets import (
    CANVAS_RANK_LADDER, FinalizeItem, NodeCanvas, assemble_canvas, detach_pack_bases, encode_node_pack,
    finalize_tiles, pack_codec,
)
from .predictor_registry import observation_origin
from .rank_method import (
    IDENTITY_BASIS_REASON, ORTHOGONALIZERS, LatestMethodPlanner, MatrixPlan, RankMethodConfig, chosen_estimate_bytes,
    identity_rows, rate_identity,
)
from .run_telemetry import split_run_telemetry
from .types import (
    ActiveBasisBinding, CacheEpoch, ChunkNode, EnergyReport, FitObservation, MatrixKey, MatrixRecord,
    TensorContext, TokenRangeRecords, canonical_digest, entry_order, entry_parts, json_copy,
)

PACKINGS = ("node", "per_matrix")


def frame_free(error: BaseException | None) -> BaseException | None:
    """Make an exception safe to keep (tickets, lineages, jobs, warm-ups); returns it.

    A stored traceback keeps every frame it crossed alive with its locals. F2 census 4423272: a failed commit's
    stored error pinned its staged chunk (60 x [4680, 1536] BF16) through ``_fit``'s frame.

    For the exception and everything it chains to (``__cause__``, ``__context__``, an ``ArchiveFailure``'s ``cause``):

    * the formatted traceback is kept as a note;
    * the traceback's finished frames are cleared (executing frames are skipped);
    * the traceback itself is dropped.
    """
    pending, seen = [error], set()
    while pending:
        item = pending.pop()
        if not isinstance(item, BaseException) or id(item) in seen:
            continue
        seen.add(id(item))
        tb = item.__traceback__
        if tb is not None:
            add_note = getattr(item, "add_note", None)
            if callable(add_note):
                try:
                    add_note("reelcache: traceback when stored (frames released):\n"
                             + "".join(traceback.format_tb(tb)).rstrip())
                except Exception:  # noqa: BLE001 - diagnostics only
                    pass
            traceback.clear_frames(tb)
            item.__traceback__ = None
        pending.extend((item.__cause__, item.__context__, getattr(item, "cause", None)))
    return error


def raisable(error: BaseException) -> BaseException:
    """A copy of a stored exception for one raise (same type, args, attributes, notes and cause).

    Raising the stored object itself would attach the caller's frames to it, and so to whatever keeps it (tickets,
    lineages). If the type cannot be copied without its ``__init__``, the stored object is returned.
    """
    try:
        clone = type(error).__new__(type(error), *error.args)
        clone.__dict__.update(error.__dict__)
        clone.__cause__, clone.__context__ = error.__cause__, error.__context__
        clone.__suppress_context__ = error.__suppress_context__
        return clone
    except Exception:  # noqa: BLE001 - an exotic exception type: raise the stored one
        return error


@dataclass(frozen=True)
class ArchiveOptions:
    """Archival execution options (not part of the method fingerprint; recorded per node).

    ``packing="node"`` stores one all-intra node pack per committed range with
    tiles referencing basis versions (required for svd_hevc pipelines);
    ``"per_matrix"`` keeps the Step 4 reference (one prepared-factor packet
    per matrix with a CPU re-measurement of the source). ``canvas_ladder``
    selects canonical canvases (None: each node's own size). ``encode_threads``
    is libx265's thread pool per node encode. ``validate_decode`` decodes each
    pack once to record decoded-coefficient digests; ``measure_post_codec``
    additionally keeps FP32 coefficients on the host to report the post-codec
    error without copying sources. ``plan_memory_bytes`` bounds the GPU working set of planning and finalization
    (matrices are planned, then finalized, in consecutive groups that fit it; plans and stored bytes are identical
    to unbounded archival whenever the eigensolves are batch-independent; run telemetry, not hashed).
    """

    packing: str = "node"
    canvas_ladder: tuple[int, ...] | None = CANVAS_RANK_LADDER
    encode_threads: int = 1
    validate_decode: bool = True
    measure_post_codec: bool = False
    orthogonalizer: str = "batched"
    plan_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.packing not in PACKINGS:
            raise ConfigurationError(f"packing must be one of {PACKINGS}")
        if self.canvas_ladder is not None:
            ladder = tuple(self.canvas_ladder)
            if not ladder or any(type(step) is not int or step < 1 for step in ladder) or list(ladder) != sorted(
                    set(ladder)):
                raise ConfigurationError("canvas ladder must be strictly increasing positive integers")
            object.__setattr__(self, "canvas_ladder", ladder)
        if type(self.encode_threads) is not int or not 1 <= self.encode_threads <= 64:
            raise ConfigurationError("encode_threads must be an integer in [1, 64]")
        for name in ("validate_decode", "measure_post_codec"):
            if type(getattr(self, name)) is not bool:
                raise ConfigurationError(f"{name} must be boolean")
        if self.measure_post_codec and not self.validate_decode:
            raise ConfigurationError("post-codec errors are measured on the validation decode")
        if self.orthogonalizer not in ORTHOGONALIZERS:
            raise ConfigurationError(f"orthogonalizer must be one of {ORTHOGONALIZERS}")
        if self.plan_memory_bytes is not None and (type(self.plan_memory_bytes) is not int
                                                   or self.plan_memory_bytes < 1):
            raise ConfigurationError("plan_memory_bytes must be None or a positive byte count")

    def to_dict(self) -> dict[str, Any]:
        data = {"packing": self.packing, "canvas_ladder": None if self.canvas_ladder is None else list(
            self.canvas_ladder), "encode_threads": self.encode_threads, "validate_decode": self.validate_decode,
            "measure_post_codec": self.measure_post_codec, "orthogonalizer": self.orthogonalizer}
        if self.plan_memory_bytes is not None:
            data["plan_memory_bytes"] = self.plan_memory_bytes  # run telemetry (reelcache.run_telemetry): unhashed
        return data

    def output_identity(self) -> dict[str, Any]:
        """The options that determine stored bytes (thread counts do not)."""
        data = self.to_dict()
        data.pop("encode_threads")
        data.pop("plan_memory_bytes", None)
        data.pop("orthogonalizer")
        return data


_FINGERPRINT_PRIME = 2147483629


#: Elements per fingerprint row tile: the int64 temporaries (bits, squares, weighted terms) cover one tile of at most
#: this many elements (128 MiB each), not the matrix; CausVid-size matrices stay one tile, LongCat's take four.
FINGERPRINT_TILE_ELEMENTS = 1 << 24
_FINGERPRINT_COMBINATIONS = ((37, 101, False), (613, 211, False), (1, 17, True))


@torch.no_grad()
def _fingerprint_terms(tensor: torch.Tensor, workspace: TileWorkspace | None = None) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.numel() == 0:
        raise ConfigurationError("source fingerprints take a nonempty matrix")
    size = tensor.element_size()
    if size not in (2, 4):
        raise ConfigurationError("source fingerprints support 16- and 32-bit element types")
    rows, columns = tensor.shape
    if rows > (1 << 22) or columns > (1 << 14):
        raise ConfigurationError("source fingerprint geometry exceeds its overflow-free range")
    if _accel_b.fused_fingerprints(tensor):  # accel B4 (opt-in): one fused exact integer kernel per tile, same values
        return _accel_b.fingerprint_terms_fused(tensor, _FINGERPRINT_PRIME, FINGERPRINT_TILE_ELEMENTS)
    source = tensor.detach()
    device = source.device
    # Each row tile's widened bits, squares and weighted terms go to reused buffers (TileWorkspace: host buffers are
    # faulted in once per call, not once per tile); the arithmetic and so every fingerprint are unchanged.
    work = TileWorkspace() if workspace is None else workspace
    column = torch.arange(columns, device=device, dtype=torch.int64)
    weights = [(column * a + 7) % 1021 + 1 for a, _, _ in _FINGERPRINT_COMBINATIONS]
    totals = torch.zeros((1 if size == 2 else 2) * len(_FINGERPRINT_COMBINATIONS), dtype=torch.int64, device=device)
    # Row tiles: every per-row checksum is the same, and the int64 sums over rows are exact (< 2^63 in the
    # overflow-free range), so the fingerprint does not depend on the tiling.
    tile_rows = max(1, FINGERPRINT_TILE_ELEMENTS // columns)
    for start in range(0, rows, tile_rows):
        stop = min(rows, start + tile_rows)
        shape = (stop - start, columns)
        chunk = source[start:stop]
        if not chunk.is_contiguous():
            chunk = chunk.contiguous()
        bits = work.get("fingerprint_bits", shape, torch.int64, device)
        bits.copy_(chunk.view(torch.int16 if size == 2 else torch.int32))  # sign-extending widening, as .to(int64)
        if size == 2:
            parts = [bits]
        else:
            low = work.get("fingerprint_low", shape, torch.int64, device)
            high = work.get("fingerprint_high", shape, torch.int64, device)
            torch.bitwise_and(bits, 0xFFFF, out=low)
            torch.bitwise_right_shift(bits, 16, out=high)
            parts = [low, high]
        product = work.get("fingerprint_product", shape, torch.int64, device)
        per_row = work.get("fingerprint_per_row", shape[:1], torch.int64, device)
        row = torch.arange(start, stop, device=device, dtype=torch.int64)
        index = 0
        for part in parts:
            squared = None
            for (_, b, square), weight in zip(_FINGERPRINT_COMBINATIONS, weights):
                row_weights = (row * b + 3) % 1019 + 1
                if square:
                    if squared is None:
                        squared = work.get("fingerprint_squared", shape, torch.int64, device)
                        torch.mul(part, part, out=squared)
                    term = squared
                else:
                    term = part
                torch.mul(term, weight, out=product)
                torch.sum(product, 1, out=per_row)
                per_row.remainder_(_FINGERPRINT_PRIME)
                totals[index] += (per_row * row_weights).sum()
                index += 1
    return totals.remainder(_FINGERPRINT_PRIME)


def _fingerprint_text(tensor: torch.Tensor, values) -> str:
    rows, columns = tensor.shape
    return f"fp1:{rows}x{columns}:{tensor.dtype}:" + "".join(f"{int(value):08x}" for value in values)


def source_fingerprint(tensor: torch.Tensor) -> str:
    """Exact, device-independent integer fingerprint of a matrix's bit patterns.

    Modular linear and quadratic checksums of 16-bit parts (no overflow up to
    4M rows x 16K columns); identical on CPU and CUDA. It binds observation
    identities to their content without a host copy; it is not an integrity
    hash.
    """
    return _fingerprint_text(tensor, _fingerprint_terms(tensor).tolist())


def source_fingerprints(tensors) -> list[str]:
    """:func:`source_fingerprint` of many matrices with one device-to-host transfer per device.

    Matrices may mix element types (e.g. RAVEN's FP32 pre-RoPE K with BF16 V):
    each fingerprint records its own dtype and covers its own bit width.
    """
    tensors = list(tensors)
    workspace = TileWorkspace()  # one set of tile buffers for every matrix of the call
    terms = []
    for tensor in tensors:
        yield_point()  # dev22j8: a step boundary per matrix (the tiles only queue kernels; their workspace is reused)
        # accel B5 (opt-in): host sources are fingerprinted on the fit device (the same integers)
        terms.append(_fingerprint_terms(_accel_b.fingerprint_source(tensor), workspace))
    by_device: dict = {}
    for index, value in enumerate(terms):
        by_device.setdefault(value.device, []).append(index)
    host = [None] * len(tensors)
    for indices in by_device.values():
        flat = torch.cat([terms[index] for index in indices])
        with blocked_wait("host_read"):  # dev22j8: the device synchronization allocates no device memory
            flat = flat.tolist()
        offset = 0
        for index in indices:
            count = int(terms[index].numel())
            host[index] = flat[offset:offset + count]
            offset += count
    return [_fingerprint_text(tensor, values) for tensor, values in zip(tensors, host)]


@dataclass
class ParentState:
    """Planning view of a lineage head: published (``head_ref``) or a pipelined, unpublished job."""

    branch_id: str
    streams: tuple[str, ...]
    entries: tuple
    head_ref: str | None
    head_kind: str
    frontier: int
    cache_epoch_id: str
    used_ids: frozenset
    chunk_geometry: Mapping[str, tuple]
    first_hw: tuple | None
    segment_lineage: bool
    bindings: Mapping[Any, ActiveBasisBinding]
    rows: Mapping[Any, torch.Tensor | None]
    rate_sources: tuple
    pending_observations: tuple
    job: Any = None
    depth: int = 0

    def planning_bindings(self, lag: int, timeout: float | None = None) -> dict[Any, ActiveBasisBinding]:
        """Version fields of the head plus the coefficient rates measured ``lag`` nodes above it."""
        if lag == 0 and self.job is None:
            return dict(self.bindings)
        rates = None
        if lag < len(self.rate_sources):
            source = self.rate_sources[lag]
            rates = source.measured_rates(timeout) if isinstance(source, CommitJob) else source
        result = {}
        for entry, binding in self.bindings.items():
            rate, identity = (None, None) if rates is None else rates.get(entry, (None, None))
            result[entry] = replace(binding, coefficient_rate=rate, rate_identity=identity)
        return result

    def published_rates(self, timeout: float | None = None) -> dict:
        """The head's own (published) coefficient rates."""
        if self.job is not None:
            return self.job.measured_rates(timeout)
        return _binding_rates(self.bindings)


#: W1.4.3 (E1): the store object (id, bytes) of the shared identity basis rows per channel count (HEVC-only arm)
_IDENTITY_OBJECTS: dict[int, tuple[str, int]] = {}


def content_object(data: bytes, kind: str) -> tuple[str, int]:
    """W1.4.3 (E1, review round 2): the content-addressed store object (id, bytes) ``ObjectStore.put(data, kind=kind)``
    writes -- ``"basis"`` for a row block (``BasisVersionStore``), ``"blob"`` for a packet payload."""
    return f"{kind}:{hashlib.sha256(data).hexdigest()}", len(data)


def identity_object(channels: int) -> tuple[str, int]:
    """W1.4.3 (E1): the store object id and size of :func:`~.rank_method.identity_rows` (``channels``) as
    ``BasisVersionStore.put_root`` stores it (content-addressed ``basis`` block), computed once per process."""
    found = _IDENTITY_OBJECTS.get(channels)
    if found is None:
        found = _IDENTITY_OBJECTS[channels] = content_object(rows_to_bytes(identity_rows(channels)), "basis")
    return found


def advance_shape(parent: ParentState, kind: str, node_id: str, ranges: Sequence[tuple[str, Mapping]],
                  epoch_id: str | None) -> ParentState:
    """Geometry/identity fields of ``parent`` after a commit of ``(chunk_id, contexts)`` ranges."""
    frontier, chunk_geometry, first_hw = parent.frontier, dict(parent.chunk_geometry), parent.first_hw
    for chunk_id, contexts in ranges:
        context = next(iter(contexts.values()))
        if first_hw is None:
            first_hw = (context.height, context.width)
        if kind == "chunk":
            chunk_geometry[chunk_id] = (context.token_start, context.frames, context.height, context.width)
            frontier = context.token_end
        elif kind == "segment":
            frontier = context.token_end
    return replace(parent, head_ref=None, head_kind=kind, frontier=frontier,
                   cache_epoch_id=epoch_id if kind in ("recache", "segment") else parent.cache_epoch_id,
                   used_ids=parent.used_ids | {node_id}, chunk_geometry=chunk_geometry, first_hw=first_hw,
                   segment_lineage=parent.segment_lineage or kind == "segment", depth=parent.depth + 1)


def _binding_rates(bindings: Mapping[Any, ActiveBasisBinding]) -> dict[Any, tuple]:
    return {entry: (binding.coefficient_rate, binding.rate_identity) for entry, binding in bindings.items()}


@dataclass
class _Range:
    chunk_id: str
    contexts: dict
    sources: dict
    source_ids: dict = field(default_factory=dict)
    plans: dict = field(default_factory=dict)
    plan_reports: list = field(default_factory=list)
    versions: dict = field(default_factory=dict)
    views: dict = field(default_factory=dict)
    canvas: NodeCanvas | None = None
    encoded: Any = None
    records: dict = field(default_factory=dict)
    record_reports: list = field(default_factory=list)
    packet_ref: str | None = None


class CommitJob:
    """One node commit's staged state; stage methods may run on different threads in order."""

    def __init__(self, runtime, kind: str, node_id: str, parent: ParentState,
                 groups: Sequence[tuple[str, Mapping, Mapping]], *, method: RankMethodConfig,
                 options: ArchiveOptions, epoch_id: str | None = None, details: Mapping[str, Any] | None = None,
                 execution_state: Mapping[str, torch.Tensor] | None = None):
        if not isinstance(method, RankMethodConfig) or not isinstance(options, ArchiveOptions):
            raise ConfigurationError("commit jobs need a RankMethodConfig and ArchiveOptions")
        if options.packing != "node":
            raise UnsupportedError("staged commits produce node packs; per-matrix packets use the Step 4 path")
        pack_codec(method.codec)
        self.runtime, self.kind, self.node_id, self.parent = runtime, kind, node_id, parent
        self.method, self.options = method, options
        self.epoch_id, self.details = epoch_id, dict(details or {})
        self.execution_state = execution_state
        self.ranges = [_Range(chunk_id, dict(contexts), dict(matrices)) for chunk_id, matrices, contexts in groups]
        self.started = time.perf_counter()
        self.stage_seconds = {"plan": 0.0, "versions": 0.0, "finalize": 0.0, "transfer_wait": 0.0, "encode": 0.0,
                              "store": 0.0, "publish": 0.0}
        self.bindings_out: dict[Any, ActiveBasisBinding] = {}
        self.rows_out: dict[Any, torch.Tensor | None] = {}
        self.observations: list[FitObservation] = []
        self.registry_view = None
        self.staged_pins: list = []
        self.staged_objects: list = []
        # W1.4.3 (E1): ``archive_write(objects)`` around each store transaction that writes payload objects -- the
        # basis versions' row blocks, a range's packet payload blobs -- with ``{object id: bytes}`` of their content
        # addresses (an archive budget admits the physically absent ones, each once; None: no budget), entered outside
        # store transactions
        self.archive_write: Callable[[Mapping[str, int]], Any] | None = None
        self.sources_released_event = None
        self.node_ref: str | None = None
        self.node: ChunkNode | None = None
        self.report: dict[str, Any] = {}
        self._rates: dict | None = None
        self._published_rates: dict | None = None
        self._rates_ready = threading.Event()
        self.active_planned: dict[Any, ActiveBasisBinding] = {}
        self.finalize_device: str | None = None
        self.error: BaseException | None = None
        self.observed = False

    # -- dependencies ------------------------------------------------------------
    def measured_rates(self, timeout: float | None = None) -> dict:
        """The coefficient rates this node publishes (blocks until its store stage finished).

        Entries this commit set active carry their own measured tile rates;
        others inherit the parent's published rates.
        """
        if not self._rates_ready.wait(timeout):
            raise TimeoutError("coefficient rates of an ancestor commit are not available yet")
        if self.error is not None or self._rates is None:
            raise StateError(f"ancestor commit {self.node_id} failed; its rates are unavailable") from self.error
        if self._published_rates is None:
            inherited = self.parent.published_rates(timeout)
            self._published_rates = {entry: self._rates.get(entry, inherited.get(entry, (None, None)))
                                     for entry in self.parent.entries}
        return self._published_rates

    def fail(self, error: BaseException) -> None:
        if self.error is None:
            self.error = frame_free(error)
        self._rates_ready.set()

    def drop_sources(self) -> None:
        """A failed or cancelled job no worker runs any more: drop its source tensors (a finished job dropped them at
        finalize). The job itself may live on, e.g. as its lineage's ``last_job``.

        dev22j8 (R-D2b, F5.1 census 4432533): also its ranges' canvases and encoded packs. A range finalized but not
        encoded keeps its device canvas (``NodeCanvas.device_pictures``, 116 MiB per CausVid chunk) until the canvas's
        D2H copy is waited for: wait for it (the copy reads the device pictures and writes the host staging slot),
        then drop the canvas and the encoded pack."""
        for item in self.ranges:
            item.sources = {}
            canvas = item.canvas
            if canvas is not None:
                try:
                    canvas.wait()  # host-waits for the D2H event, then releases the device pictures
                except Exception:  # noqa: BLE001 - a failed wait (a broken CUDA context) still drops the references
                    pass
            item.canvas = None
            item.encoded = None

    def child_state(self) -> ParentState:
        """The planning view a pipelined child commit uses before this job publishes."""
        parent = self.parent
        shape = advance_shape(parent, self.kind, self.node_id,
                              [(item.chunk_id, item.contexts) for item in self.ranges], self.epoch_id)
        rows = dict(parent.rows)
        rows.update(self.rows_out)
        bindings = dict(parent.bindings)
        bindings.update(self.active_planned)
        prior = parent.pending_observations if parent.job is not None and not parent.job.observed else ()
        return replace(shape, bindings=bindings, rows=rows,
                       rate_sources=(self,) + parent.rate_sources[:self.method.rate_lag],
                       pending_observations=prior + tuple(self.observations), job=self)

    # -- stage 1: plan + versions ----------------------------------------------
    @torch.no_grad()
    def plan(self, registry, *, device: Any = "cpu", eigh=None, eigh_label: str | None = None, fitter=None,
             rate_timeout: float | None = None) -> None:
        """Plan every range in order (on ``fitter``'s device and owner thread when given)."""
        started = time.perf_counter()
        self.registry_view = registry.pin_with(self.parent.pending_observations)
        # Commit context of this job's observations (run context, not content: records keep the plain observation).
        origin = observation_origin(self.parent.branch_id, self.kind, self.node_id, self.registry_view)
        with blocked_wait("rate_wait"):  # dev22j8: waits for an ancestor's measured rates (rate_lag); no allocation
            bindings = self.parent.planning_bindings(self.method.rate_lag, rate_timeout)
        rows = dict(self.parent.rows)
        self.stage_seconds["plan_dependency_wait"] = time.perf_counter() - started
        if fitter is None:
            device = torch.device(device)
            if device.type == "cuda" and eigh is None:
                from .gpu_fitting import select_eigh_solver
                eigh_label, eigh = select_eigh_solver("auto")
        streams = self.parent.streams or (None,)
        for item in self.ranges:
            step_phase("fingerprints")
            yield_point()  # dev22j8: a step boundary before the source fingerprints (device tile workspaces)...
            clock = time.perf_counter()
            entries = list(item.sources)
            fingerprints = source_fingerprints([item.sources[entry] for entry in entries])
            yield_point()  # ...and one after them
            for entry, fingerprint in zip(entries, fingerprints):
                key, stream = entry_parts(entry)
                context = item.contexts[entry]
                item.source_ids[entry] = canonical_digest({
                    "model": self.runtime.model.fingerprint, "commit": self.kind, "node_id": self.node_id,
                    "chunk_id": item.chunk_id, "token_start": context.token_start, "token_end": context.token_end,
                    "key": key.to_dict(), "stream": stream, "source": fingerprint,
                })
            self.stage_seconds["plan_fingerprints"] = self.stage_seconds.get("plan_fingerprints", 0.0) + (
                time.perf_counter() - clock)
            step_phase("planner")
            for stream in streams:
                members = [entry for entry in item.sources if entry_parts(entry)[1] == stream]
                keys = {entry_parts(entry)[0]: entry for entry in members}
                sources = {key: item.sources[entry] for key, entry in keys.items()}
                contexts = {key: item.contexts[entry] for key, entry in keys.items()}
                local_bindings = {key: bindings[entry] for key, entry in keys.items()}
                parent_rows = {key: rows.get(entry) for key, entry in keys.items()}
                source_ids = {key: item.source_ids[entry] for key, entry in keys.items()}
                label = f"{self.kind}:{self.node_id}:{item.chunk_id}" + ("" if stream is None else f":{stream}")
                if fitter is not None:
                    bound = {} if self.options.plan_memory_bytes is None else {
                        "max_working_bytes": self.options.plan_memory_bytes}
                    plan = fitter.plan_chunk(self.method, self.registry_view, sources, contexts, local_bindings,
                                             parent_rows, chunk_id=label, source_ids=source_ids,
                                             orthogonalizer=self.options.orthogonalizer, **bound)
                else:
                    plan = LatestMethodPlanner(self.method, self.registry_view, device=device, eigh=eigh,
                                               eigh_label=eigh_label,
                                               orthogonalizer=self.options.orthogonalizer,
                                               max_working_bytes=self.options.plan_memory_bytes).plan(
                        sources, contexts, local_bindings, parent_rows, chunk_id=label, source_ids=source_ids)
                item.plan_reports.append({
                    "chunk_id": item.chunk_id, "stream": stream, "device": plan.device,
                    "eigh_solver": plan.eigh_solver, "orthogonalizer": plan.orthogonalizer,
                    "pinned_sequence": plan.pinned_sequence, "speculative_observations": len(
                        self.registry_view.speculative), "seconds": plan.seconds, "stages": plan.stages,
                    "counters": plan.counters,
                    **({"sub_batches": list(plan.batches)} if plan.batches else {}),
                    **({"plan_memory": plan.memory} if plan.memory is not None else {}),
                })
                for key, matrix_plan in plan.plans.items():
                    item.plans[keys[key]] = matrix_plan
                    self.observations.extend(replace(observation, origin=origin)
                                             for observation in matrix_plan.observations)
            self.stage_seconds["plan"] += time.perf_counter() - started
            yield_point()
            step_phase("versions")
            started = time.perf_counter()
            self._store_versions(item, bindings, rows)
            self.stage_seconds["versions"] += time.perf_counter() - started
            yield_point()  # dev22j8: a step boundary after the version store (its transaction is closed)
            started = time.perf_counter()
        for entry in self.parent.entries:
            self.rows_out.setdefault(entry, rows.get(entry))

    def _store_versions(self, item: _Range, bindings: dict, rows: dict) -> None:
        store, versions = self.runtime.store, self.runtime.basis_versions
        guard = contextlib.nullcontext()
        if self.archive_write is not None:  # W1.4.3 (E1): every row block (FP16) this transaction stores, by its
            # content address -- the HEVC-only arm's shared identity basis (cached), and identical blocks (e.g. the
            # one-row bases of zero-input matrices) once (review round 2) -- admitted before the store sees them when
            # physically absent (never written, or evicted: a rewrite); no eviction runs until this transaction has
            # leased what it writes and finds
            objects = dict(identity_object(int(plan.new_rows.shape[1])) if plan.reason == IDENTITY_BASIS_REASON
                           else content_object(rows_to_bytes(plan.new_rows), "basis")
                           for plan in item.plans.values() if plan.new_rows is not None)
            guard = self.archive_write(objects)
        with guard, store.transaction():
            try:
                for entry in sorted(item.plans, key=entry_order):
                    plan = item.plans[entry]
                    key = entry_parts(entry)[0]
                    parent_rows = rows.get(entry)
                    new_block = 0
                    if plan.decision == "augment":
                        ref, version = versions.put_extension(plan.parent_version_ref, plan.new_rows)
                        full = torch.cat((parent_rows, plan.new_rows))
                        new_block = version.added_bytes
                    elif plan.new_rows is not None:
                        ref, version = versions.put_root(key, plan.new_rows)
                        full = plan.new_rows
                        # wl-base (W1c): the HEVC-only arm's identity basis is one shared content-addressed block per
                        # channel count (stored once per store), not a per-matrix cost
                        new_block = 0 if plan.reason == IDENTITY_BASIS_REASON else version.added_bytes
                    else:
                        ref = plan.parent_version_ref
                        version = versions.load(ref)
                        full = parent_rows
                    if full is None or int(full.shape[0]) != version.total_rank:
                        raise IntegrityError("in-memory active rows disagree with their basis version")
                    item.versions[entry] = (ref, version, new_block)
                    item.views[entry] = full[:plan.rank_view].contiguous()
                    binding = bindings[entry]
                    if plan.sets_active:
                        if plan.decision in ("fresh", "refit", "diagnostic_full_width"):
                            fresh_rank, fresh_chunk = plan.rank_view, item.chunk_id
                        else:
                            fresh_rank, fresh_chunk = binding.fresh_rank, binding.fresh_chunk_id
                        # The cumulative refit rule's debt after this range (0 after a fresh fit or refit; the paper
                        # rule tracks none): later ranges of this commit and pipelined children plan with it.
                        debt = plan.refit_debt or 0.0
                        # Later ranges of this commit keep the commit's planning rates (rate_lag rule).
                        bindings[entry] = ActiveBasisBinding(ref, version.total_rank, version.group_id, fresh_rank,
                                                             fresh_chunk, binding.coefficient_rate,
                                                             binding.rate_identity, debt)
                        self.active_planned[entry] = ActiveBasisBinding(ref, version.total_rank, version.group_id,
                                                                        fresh_rank, fresh_chunk, refit_debt=debt)
                        rows[entry] = full
                        self.rows_out[entry] = full
                # Pin every version this range references in the transaction that wrote or found it, including
                # versions another session wrote first (content addresses deduplicate identical content): an abort
                # of that session must not collect them while this commit is pending (CR-14-7).
                referenced = sorted({ref for ref, _, _ in item.versions.values()})
                if referenced:
                    self.staged_pins.append(store.lease(referenced, purpose="staging"))
            finally:
                # Collection candidates if this commit fails: everything it wrote, new or deduplicated; collection
                # removes only objects nothing published or pinned reaches.
                self.staged_objects.extend(store.touched_objects)

    # -- stage 2: finalize on the decision device ------------------------------
    @torch.no_grad()
    def finalize(self, *, device: Any = None, staging=None, stream=None) -> None:
        """Final measurement, quantization and canvas layout; starts one D2H copy per range.

        ``staging(nbytes)`` optionally returns a (pinned) host uint8 buffer for
        each range's canvas; the caller releases it after :meth:`encode`.
        Every range's rank-view rows cross to the device in one transfer and
        its tiles are finalized together (:func:`reelcache.node_packets.finalize_tiles`).
        """
        started = time.perf_counter()
        self.finalize_memory = []
        for index, item in enumerate(self.ranges):
            entries = sorted(item.plans, key=entry_order)
            targets = [item.sources[entry].device if device is None else torch.device(device) for entry in entries]
            groups = self._finalize_groups(item, entries)
            workspace = TileWorkspace()  # the groups' host temporaries reuse one set of buffers
            tiles = []
            for group in groups:
                views: list = [None] * len(group)
                for target_device in dict.fromkeys(targets[position] for position in group):
                    chosen = [slot for slot, position in enumerate(group) if targets[position] == target_device]
                    placed = upload([item.views[entries[group[slot]]] for slot in chosen], target_device)
                    for slot, view in zip(chosen, placed):
                        views[slot] = view
                batch = []
                for position, view in zip(group, views):
                    entry, target_device = entries[position], targets[position]
                    plan = item.plans[entry]
                    batch.append(FinalizeItem(item.sources[entry].to(target_device), view, item.contexts[entry],
                                              self.method.spec_for(plan.rank_view), entry, plan.target,
                                              plan.diagnostic, item.views[entry]))
                # Per-matrix arithmetic (the lockstep only batches host reads): grouping never changes a tile.
                yield_point()
                tiles.extend(finalize_tiles(batch, tile_rows=self.method.tile_rows,
                                            keep_coefficients=self.options.measure_post_codec,
                                            products=self.method.measurement_products, workspace=workspace))
                del batch, views
            del workspace
            channels = max(context.channels for context in item.contexts.values())
            yield_point()
            self.finalize_device = str(tiles[0][1].device)
            item.canvas = assemble_canvas(self.node_id if len(self.ranges) == 1 else f"{self.node_id}:{item.chunk_id}",
                                          tiles, codec=pack_codec(self.method.codec),
                                          ladder=self.options.canvas_ladder, channels=channels)
            del tiles
            buffer = None if staging is None else staging(item.canvas.nbytes)
            if _accel_c.device_input(item.canvas):  # accel C (nvenc_device_input): NVENC reads the device canvas
                item.canvas.keep_on_device(stream)
            else:
                item.canvas.to_host(buffer=buffer, stream=stream)
        if torch.cuda.is_available() and any(isinstance(m, torch.Tensor) and m.device.type == "cuda"
                                             for item in self.ranges for m in item.sources.values()):
            device = next(m.device for item in self.ranges for m in item.sources.values())
            event = torch.cuda.Event()
            event.record(stream if stream is not None else torch.cuda.current_stream(device))
            self.sources_released_event = event
        for item in self.ranges:
            item.sources = {}
        self.stage_seconds["finalize"] += time.perf_counter() - started

    def _finalize_groups(self, item: _Range, entries: list) -> list[list[int]]:
        """Consecutive entry positions finalized together: all at once, or groups whose finalization tensors fit half
        of ``options.plan_memory_bytes`` (rows, FP32/FP64 coefficients, FP64 quantization ratios, pixels and three
        FP64 measurement tiles per matrix; the pixels of earlier groups stay until the canvas is assembled)."""
        positions = list(range(len(entries)))
        budget = self.options.plan_memory_bytes
        if budget is None or not entries:
            return [positions]
        estimates = []
        for entry in entries:
            rows, channels = (int(size) for size in item.sources[entry].shape)
            rank = int(item.views[entry].shape[0])
            estimates.append(rows * rank * (4 + 8 + 8 + 1) + rank * channels * 2
                             + 3 * 8 * min(rows, self.method.tile_rows) * channels)
        size = max(1, min(len(entries), (budget // 2) // max(estimates)))
        groups = [positions[start:start + size] for start in range(0, len(positions), size)]
        self.finalize_memory.append({"chunk_id": item.chunk_id, "max_working_bytes": budget,
                                     "matrix_bytes_estimate": max(estimates), "group_size": size,
                                     "groups": len(groups)})
        return groups

    # -- stage 3: CPU encode ------------------------------------------------------
    def encode(self, *, encoder=None, encoder_lock=None) -> None:
        """Encode every range's canvas not encoded yet (libx265, or the explicit NVENC ``encoder`` for NVENC
        specifications)."""
        for index in range(len(self.ranges)):
            self.encode_range(index, encoder=encoder, encoder_lock=encoder_lock)

    def encode_range(self, index: int, *, encoder=None, encoder_lock=None) -> bool:
        """Encode one finalized range's canvas; False (nothing done) when it is already encoded.

        Ranges are encoded independently (the bytes never depend on which thread or order encodes them), so an
        archive pipeline can stream a node's ranges through fewer host staging slots than the node has ranges.
        """
        item = self.ranges[index]
        if item.encoded is not None:
            return False
        if item.canvas is None:
            raise StateError(f"range {item.chunk_id} of {self.node_id} is not finalized")
        started = time.perf_counter()
        item.canvas.wait()
        self.stage_seconds["transfer_wait"] += time.perf_counter() - started
        started = time.perf_counter()
        item.encoded = encode_node_pack(item.canvas, threads=self.options.encode_threads,
                                        validate_decode=self.options.validate_decode, encoder=encoder,
                                        encoder_lock=encoder_lock)
        self.stage_seconds["encode"] += time.perf_counter() - started
        return True

    # -- stage 4: store packets and records -----------------------------------------
    def store(self, adopt=None) -> None:
        """Store packets and records; ``adopt(packet_ref, encoded)`` may prepare the pack for restore."""
        started = time.perf_counter()
        rates = {}
        runtime, store = self.runtime, self.runtime.store
        for item in self.ranges:
            metadata, payloads, _ = detach_pack_bases(item.encoded)
            guard = contextlib.nullcontext()
            if self.archive_write is not None:  # W1.4.3 (E1): the packet's payload blobs, admitted by content address
                # before the store sees them (each content once, review round 2), evictions kept out until the
                # transaction below has leased the packet
                guard = self.archive_write(dict(content_object(value, "blob") for value in payloads.values()))
            metrics = json_copy(item.encoded.metrics)
            tiles = item.canvas.tiles
            tile_bases = []
            for tile in tiles:
                ref, version, _ = item.versions[tile.entry]
                tile_bases.append({"basis_version": ref, "rank_view": tile.rank})
            with guard, store.transaction():
                try:
                    clock = time.perf_counter()
                    with _accel_c.known_payload_digests(item.encoded):  # accel C (hash_once): encode-time digests
                        blobs = {name: store.put(value) for name, value in payloads.items()}
                    self.stage_seconds["store_blobs"] = self.stage_seconds.get("store_blobs", 0.0) + (
                        time.perf_counter() - clock)
                    clock = time.perf_counter()
                    content_metrics, metrics_telemetry = split_run_telemetry(metrics)
                    packet_ref = store.put_manifest({
                        "type": "packet", "version": 1, "model": runtime.model.to_dict(), "metadata": metadata,
                        "payloads": blobs, "metrics": content_metrics, "basis_storage": "external_version_prefix_tiles",
                        "tile_bases": tile_bases,
                    }, references=sorted(set(blobs.values()) | {item["basis_version"] for item in tile_bases}))
                    if metrics_telemetry is not None:
                        store.put_telemetry(packet_ref, {"metrics": metrics_telemetry})
                    self.staged_pins.append(store.lease(packet_ref, purpose="staging"))
                    packet_bytes = sum(len(value) for value in payloads.values())
                    from .codecs import _json_bytes
                    metadata_bytes = len(_json_bytes(metadata))
                    stream_bytes = len(payloads["stream.hevc"] if "stream.hevc" in payloads  # wl-base: factor packs
                                       else payloads["coefficients.f16"])
                    self.stage_seconds["store_packet"] = self.stage_seconds.get("store_packet", 0.0) + (
                        time.perf_counter() - clock)
                    clock = time.perf_counter()
                    created = []
                    for index, tile in enumerate(tiles):
                        record_ref, report, rate = self._record(item, index, tile, metadata, metrics, packet_ref,
                                                                stream_bytes, packet_bytes, metadata_bytes)
                        item.records[tile.entry] = record_ref
                        item.record_reports.append(report)
                        if rate is not None:
                            rates[tile.entry] = rate
                        created.append(record_ref)
                    self.staged_pins.append(store.lease(sorted(set(created)), purpose="staging"))
                    self.stage_seconds["store_records"] = self.stage_seconds.get("store_records", 0.0) + (
                        time.perf_counter() - clock)
                    clock = time.perf_counter()
                finally:
                    self.staged_objects.extend(store.touched_objects)
            self.stage_seconds["store_commit"] = self.stage_seconds.get("store_commit", 0.0) + (
                time.perf_counter() - clock)
            item.packet_ref = packet_ref
            if adopt is not None:
                clock = time.perf_counter()
                adopt(packet_ref, item.encoded)
                self.stage_seconds["seal_prepare"] = self.stage_seconds.get("seal_prepare", 0.0) + (
                    time.perf_counter() - clock)
            item.canvas = None
            item.encoded = None
        missing = set(self.active_planned) - set(rates)
        if missing:
            raise IntegrityError("an active basis entry has no measured coefficient rate (no tile byte range)")
        self._rates = rates
        self._rates_ready.set()
        self.stage_seconds["store"] += time.perf_counter() - started

    def _record(self, item: _Range, index: int, tile, metadata, metrics, packet_ref, stream_bytes, packet_bytes,
                metadata_bytes):
        plan: MatrixPlan = item.plans[tile.entry]
        context, rank = tile.context, tile.rank
        version_ref, version, new_block = item.versions[tile.entry]
        spec = self.method.spec_for(rank)
        tile_meta = metadata["tiles"][index]
        tile_bytes = tile_meta["byte_range"][1] if tile_meta["byte_range"] else None
        label = f"{tile.key.layer}:{tile.key.target}" + ("" if tile.stream is None else f":{tile.stream}")
        post_codec = metrics.get("post_codec_relative_error", {}).get(label)
        regime = "residual" if plan.decision == "augment" else "fresh"
        prediction = next((row for row in reversed(plan.predictions) if row["regime"] == regime), None)
        if prediction is not None and prediction.get("predictor") is None:
            prediction = None
        energy = EnergyReport(plan.target, tile.ideal_energy, tile.stored_energy, post_codec, plan.observed_rank,
                              plan.bucket_rank, rank, tile.zero_input, plan.diagnostic)
        added_rows = 0 if plan.new_rows is None else int(plan.new_rows.shape[0])
        affine_bytes = len(tile.affine)
        actual = {
            "tile_stream_bytes": tile_bytes, "affine_bytes": affine_bytes,
            "packet_payload_bytes": packet_bytes, "packet_metadata_bytes": metadata_bytes,
            "new_basis_block_bytes": new_block,
            "incremental_bytes": (tile_bytes or 0) + affine_bytes + new_block,
            "inherited_view_rows": rank - min(rank, added_rows),
            "inherited_view_bytes": 2 * context.channels * (rank - min(rank, added_rows)),
        }
        decision = {
            **plan.summary(),
            "finalization": {
                "device": self.finalize_device,
                "stored_factor_energy": tile.stored_energy, "ideal_subspace_energy": tile.ideal_energy,
                "relative_residual": tile.relative_residual, "condition_bound": tile.condition_bound,
                "decision_device_energy": plan.decision_energy, "post_codec_relative_error": post_codec,
                "measurement": "direct tiled residual of FP32 P = A Q^T with the stored FP16 rows (decision device)",
            },
            "packet": {"tile": index, "canvas": dict(metadata["canvas"]), "options": self.options.output_identity()},
            "actual_bytes": actual,
            "estimate_vs_actual": None if plan.estimate is None else {
                "estimated_chosen_bytes": chosen_estimate_bytes(plan.estimate),
                "actual_incremental_bytes": actual["incremental_bytes"],
            },
        }
        # F4: the predictor revision and the decision's run telemetry (predictions, planner work) go to the
        # record's unhashed sidecar (they depend on cross-session scheduling); load_record(ref, telemetry=True).
        decision, decision_telemetry = split_run_telemetry(json_copy(decision))
        record = MatrixRecord(context, spec, packet_ref, version_ref, rank, energy, None, decision,
                              packet_tile=index, stream=tile.stream)
        record_ref = self.runtime.store.put_manifest(
            {"type": "matrix_record", "version": 1, "model": self.runtime.model.to_dict(), "record": record.to_dict()},
            references=[packet_ref, version_ref],
        )
        sidecar = {} if decision_telemetry is None else {"decision": decision_telemetry}
        if prediction is not None:
            sidecar["predictor_revision"] = prediction["predictor"]["revision_ref"]
        if sidecar:
            self.runtime.store.put_telemetry(record_ref, sidecar)
        rate = None
        if tile_bytes is not None and plan.sets_active:
            rate = (tile_bytes / (context.token_count * rank), rate_identity(context, spec))
        report = {
            "key": tile.key.to_dict(), "stream": tile.stream, "decision": plan.decision, "reason": plan.reason,
            "record": record_ref, "basis_version": version_ref, "group_id": version.group_id, "rank_view": rank,
            "version_total_rank": version.total_rank, "energy": energy.to_dict(), "actual_bytes": actual,
            "estimate": plan.estimate, "counters": plan.work["counters"],
        }
        return record_ref, report, rate

    # -- stage 5: publish ------------------------------------------------------------
    def build_node(self, head_ref: str, head: ChunkNode) -> tuple[ChunkNode, list]:
        """The node to publish under the (now published) parent ``head_ref``."""
        runtime = self.runtime
        rates = self._rates or {}
        published = {}
        for entry in self.parent.entries:
            if entry in self.active_planned:
                rate, identity = rates[entry]
                published[entry] = replace(self.active_planned[entry], coefficient_rate=rate, rate_identity=identity)
            else:
                key, stream = entry_parts(entry)
                published[entry] = head.binding(key, stream)
        self.published_bindings = published
        active = tuple(sorted(published.items(), key=lambda item: entry_order(item[0])))
        state_ref = None
        if self.execution_state is not None:
            from .serialization import serialize_tensors
            state_ref = runtime.store.put(serialize_tensors(self.execution_state))
        streams = self.parent.streams
        epochs = []
        if self.kind == "chunk":
            item = self.ranges[0]
            context = next(iter(item.contexts.values()))
            node = ChunkNode("chunk", item.chunk_id, head_ref, head.cache_epoch_id, head.epoch_ref,
                             context.token_start, context.token_end, context.frames, context.height, context.width,
                             context.token_end, tuple(item.records.items()), active, state_ref, streams,
                             packet_ref=item.packet_ref)
        else:
            ranges = []
            for item in self.ranges:
                context = next(iter(item.contexts.values()))
                ranges.append(TokenRangeRecords(item.chunk_id, context.token_start, context.token_end, context.frames,
                                                context.height, context.width, tuple(item.records.items()), streams,
                                                packet_ref=item.packet_ref))
            if self.kind == "recache":
                epoch = CacheEpoch(self.epoch_id, head.epoch_ref, "kv_recache", "global", tuple(ranges), self.details)
            else:
                epoch = CacheEpoch(self.epoch_id, head.epoch_ref, "longcat_segment", "segment_local", tuple(ranges),
                                   self.details)
            epoch_ref = runtime._put_epoch(epoch)
            epochs.append(epoch_ref)
            if self.kind == "recache":
                node = ChunkNode("recache", self.node_id, head_ref, self.epoch_id, epoch_ref, head.frontier,
                                 head.frontier, 0, 0, 0, head.frontier, (), active, state_ref, streams)
            else:
                first = ranges[0]
                node = ChunkNode("segment", self.node_id, head_ref, self.epoch_id, epoch_ref, 0, first.token_end,
                                 first.frames, first.height, first.width, first.token_end, (), active, state_ref,
                                 streams)
        return node, epochs

    def commit_report(self) -> dict[str, Any]:
        records = [row for item in self.ranges for row in item.record_reports]
        return {
            "kind": self.kind, "method": self.method.to_dict(), "options": self.options.to_dict(),
            "registry": {"registry_id": self.registry_view.registry_id,
                         "pinned_sequence": self.registry_view.sequence,
                         "speculative_observations": len(self.registry_view.speculative)},
            "rate_lag": self.method.rate_lag,
            "plans": [row for item in self.ranges for row in item.plan_reports],
            "decisions": {decision: sum(row["decision"] == decision for row in records)
                          for decision in sorted({row["decision"] for row in records})},
            "incremental_bytes": sum(row["actual_bytes"]["incremental_bytes"] for row in records),
            "new_basis_block_bytes": sum(row["actual_bytes"]["new_basis_block_bytes"] for row in records),
            "stage_seconds": dict(self.stage_seconds),
            **({"finalize_memory": list(self.finalize_memory)} if getattr(self, "finalize_memory", None) else {}),
        }

    def record_reports(self) -> list:
        return [{"chunk_id": item.chunk_id, **row} for item in self.ranges for row in item.record_reports]

    def release_staging(self, *, collect: bool) -> None:
        store = self.runtime.store
        try:
            with store.transaction():
                for lease in self.staged_pins:
                    lease.release()
            self.staged_pins = []
            if collect and self.staged_objects:
                store.collect_garbage(candidates=self.staged_objects)
        finally:
            self.staged_objects = [] if collect else self.staged_objects


@torch.no_grad()
def warm_commit_path(method: RankMethodConfig, registry, contexts: Mapping[Any, TensorContext], *,
                     options: ArchiveOptions | None = None, device: Any = "cpu", fitter=None, stream=None,
                     max_frames: int | None = None, seed: int = 0) -> dict[str, Any]:
    """Plan and finalize two synthetic chunks of the given geometry on this thread; nothing is stored.

    The first real commit otherwise pays every first use: cuBLAS/cuSOLVER
    handles and workspaces (per thread and stream), lazily loaded kernel modules
    (read from the library files, slow on a cold network file system), the
    eigensolver binding and caching-allocator growth. Chunk 0 runs the fresh
    path, chunk 1 the active-basis paths (certify/share/augment/refit), each
    followed by finalization (measurement, quantization, canvas, D2H copy).
    Synthetic sources come from a private generator (global RNG untouched);
    observations are discarded. ``max_frames`` caps the frames per context
    (smaller transient memory; the libraries and kernels warmed are the same).
    Returns timings, decisions, the synthetic sources' bytes (``"source_bytes"``)
    and the warm canvas (``"canvas"``, host pictures) for an optional encoder
    warm-up.
    """
    from hashlib import sha256
    options = ArchiveOptions() if options is None else options
    if not isinstance(method, RankMethodConfig) or not isinstance(options, ArchiveOptions):
        raise ConfigurationError("warm-up needs a RankMethodConfig and ArchiveOptions")
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ConfigurationError("max_frames must be None or a positive integer")
    if not contexts:
        raise ConfigurationError("warm-up needs the commit's contexts")
    started = time.perf_counter()
    device = torch.device(device) if fitter is None else torch.device(fitter._device())
    stream_label = entry_parts(next(iter(contexts)))[1]
    keys = {}
    for entry, context in contexts.items():
        key, stream_name = entry_parts(entry)
        if stream_name == stream_label:
            if max_frames is not None and context.frames > max_frames:
                context = replace(context, frames=max_frames)
            keys[key] = context
    keys = dict(sorted(keys.items(), key=lambda item: entry_order(item[0])))
    eigh = eigh_label = None
    if fitter is None and device.type == "cuda":
        from .gpu_fitting import select_eigh_solver
        eigh_label, eigh = select_eigh_solver("auto")
    from .serialization import resolve_dtype
    view = registry.pin()
    bindings = {key: ActiveBasisBinding() for key in keys}
    rows: dict[MatrixKey, torch.Tensor | None] = {key: None for key in keys}
    report = {"device": str(device), "keys": len(keys), "plan_seconds": [], "finalize_seconds": [],
              "decisions": [], "eigh_solver": None, "canvas": None}
    context_manager = torch.cuda.stream(stream) if stream is not None else _NullContext()
    with context_manager:
        # Synthetic sources are written on the planning stream itself (no cross-stream read before they exist).
        generator = torch.Generator(device=device).manual_seed(seed)
        sources = {}
        for key, context in keys.items():
            tokens, channels = context.token_count, context.channels
            rank = min(64, channels, tokens)
            basis = torch.linalg.qr(torch.randn(channels, rank, generator=generator, device=device)).Q.T
            decay = torch.linspace(2.0, 0.02, rank, device=device)
            value = mm(torch.randn(tokens, rank, generator=generator, device=device) * decay, basis)
            value += 0.01 * torch.randn(tokens, channels, generator=generator, device=device)
            sources[key] = value.to(resolve_dtype(context.dtype))
            del basis, value
        report["source_bytes"] = sum(int(source.numel()) * source.element_size() for source in sources.values())
        for chunk in range(2):
            if chunk:
                drift = (0.05, 0.1, 0.15, 0.25)
                for position, (key, source) in enumerate(sources.items()):
                    tokens, channels = source.shape
                    update = mm(torch.randn(tokens, 6, generator=generator, device=device),
                                torch.randn(6, channels, generator=generator, device=device))
                    source += (drift[position % len(drift)] * update).to(source.dtype)
            source_ids = {key: canonical_digest({"warmup": seed, "chunk": chunk, "key": key.to_dict()})
                          for key in keys}
            clock = time.perf_counter()
            # dev22j8: planned and finalized in a node's groups (``options.plan_memory_bytes``), so the warm-up's
            # working set -- the decode guard's learned peak -- is a node's
            if fitter is not None:
                bound = {} if options.plan_memory_bytes is None else {"max_working_bytes": options.plan_memory_bytes}
                plan = fitter.plan_chunk(method, view, sources, keys, bindings, rows, chunk_id=f"warmup:{chunk}",
                                         source_ids=source_ids, orthogonalizer=options.orthogonalizer, **bound)
            else:
                plan = LatestMethodPlanner(method, view, device=device, eigh=eigh, eigh_label=eigh_label,
                                           orthogonalizer=options.orthogonalizer,
                                           max_working_bytes=options.plan_memory_bytes).plan(
                    sources, keys, bindings, rows, chunk_id=f"warmup:{chunk}", source_ids=source_ids)
            report["plan_seconds"].append(time.perf_counter() - clock)
            report["eigh_solver"] = plan.eigh_solver
            report["decisions"].append({decision: sum(item.decision == decision for item in plan.plans.values())
                                        for decision in sorted({item.decision for item in plan.plans.values()})})
            clock = time.perf_counter()
            views, batch = {}, []
            for key, item in plan.plans.items():
                parent = rows[key]
                if item.decision == "augment":
                    full = torch.cat((parent, item.new_rows))
                elif item.new_rows is not None:
                    full = item.new_rows
                else:
                    full = parent
                views[key] = full[:item.rank_view].contiguous()
                if item.sets_active:
                    digest = sha256(f"warmup:{seed}:{chunk}:{key.layer}:{key.target}".encode()).hexdigest()
                    fresh = item.decision in ("fresh", "refit", "diagnostic_full_width")
                    bindings[key] = ActiveBasisBinding(
                        "manifest:" + digest, int(full.shape[0]), digest,
                        item.rank_view if fresh else bindings[key].fresh_rank,
                        f"warmup:{chunk}" if fresh else bindings[key].fresh_chunk_id,
                        refit_debt=item.refit_debt or 0.0)
                    rows[key] = full
            ordered = list(keys)
            placed = upload([views[key] for key in ordered], device)
            for key, placed_view in zip(ordered, placed):
                item = plan.plans[key]
                batch.append(FinalizeItem(sources[key], placed_view, keys[key], method.spec_for(item.rank_view), key,
                                          item.target, item.diagnostic, views[key]))
            tiles = []
            for group in _finalize_batches(batch, options.plan_memory_bytes, method.tile_rows):
                tiles.extend(finalize_tiles(group, tile_rows=method.tile_rows,
                                            keep_coefficients=options.measure_post_codec,
                                            products=method.measurement_products))
            del batch, placed
            channels = max(context.channels for context in keys.values())
            canvas = assemble_canvas(f"warmup-{chunk}", tiles, codec=pack_codec(method.codec),
                                     ladder=options.canvas_ladder, channels=channels)
            del tiles
            canvas.to_host(stream=stream)
            canvas.wait()
            report["finalize_seconds"].append(time.perf_counter() - clock)
            report["canvas"] = canvas
    del sources
    report["seconds"] = time.perf_counter() - started
    return report


def _finalize_batches(batch: list, budget: int | None, tile_rows: int) -> list[list]:
    """The warm-up's finalization groups (dev22j8): consecutive items whose finalization tensors fit half of
    ``budget`` (``plan_memory_bytes``), by :meth:`CommitJob._finalize_groups`'s estimate; one group when unbounded."""
    if budget is None or not batch:
        return [batch]
    estimates = []
    for item in batch:
        rows, channels = (int(size) for size in item.matrix.shape)
        rank = int(item.rows16.shape[0])
        estimates.append(rows * rank * (4 + 8 + 8 + 1) + rank * channels * 2
                         + 3 * 8 * min(rows, tile_rows) * channels)
    size = max(1, min(len(batch), (budget // 2) // max(estimates)))
    return [batch[start:start + size] for start in range(0, len(batch), size)]


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
