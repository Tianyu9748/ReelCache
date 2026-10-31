"""FP16 factor node packs (``reelcache.factor_pack.v1``): the SVD-only baseline of workload v2 (wl-base, W1c).

A committed range's matrices are stored as their fixed-rank factors without HEVC (user decision 2026-10-19: "FP16
factors stored as-is"):

* the coefficients ``P = A Q^T`` (the planner's verified FP32 projection) rounded ONCE to FP16, row-major
  ``[N, rank]``, all tiles of the range in ONE payload ``coefficients.f16`` with per-tile byte ranges;
* the FP16 basis rows ``Q`` as the same immutable external basis versions as an ``svd_hevc`` node pack's tiles
  (``basis.NNN.f16`` payloads in the in-memory pack, detached by the store and re-attached, checked, on load).

The pack keeps the node-pack conventions (tile metadata, ``tile_bases`` bindings, ``byte_range`` per tile, a
``canvas`` record), so the staged commit stages (:mod:`reelcache.lineage_commit`), the store, leases, garbage collection
and :meth:`reelcache.lineage.LineageRuntimeMixin.node_pack` handle it like a ``reelcache.tensor_pack.v2`` pack; no HEVC path
accepts it (a different schema), so nothing decodes it by mistake.

Finalization (:func:`finalize_factor_tiles`, on the planning device) measures the stored representation exactly:
``stored_factor_energy`` and ``ideal_subspace_energy`` as for every tile, plus the post-codec error of the stored FP16
factors, ``||A - P16 Q16||_F / ||A||_F`` (direct tiled residual, FP64 sums), recorded as the tile's
``post_codec_relative_error`` (the relative Frobenius error against the RAW source). FP16 coefficients that overflow
raise :class:`IntegrityError` (never a silent clamp).

Restore (:class:`FactorLineageRestorer`): the :class:`~reelcache.archive_restore.LineageRestorer` machinery -- prepared
packs in a byte-bounded host cache (``compressed_prepared``), cold loads from the store (``compressed_cold``), device
basis versions in :class:`~reelcache.archive_restore.DeviceBasisCache`, layer-ordered reconstruction (so the
layer-overlapped restore applies) -- with :class:`FactorRestorer` in place of the NVDEC :class:`PackedRestorer`: per
layer of a node one host-to-device copy of its FP16 coefficients on a copy stream, then
:func:`reelcache.restore_kernels.reconstruct` on ``CoefficientSource`` (FP16 values) with the basis prefix: the P.Q GEMM,
the fused RoPE and the native install of the ReelCache restore. Prepared coefficients stay in pageable host memory (no
pinning on the archival threads; a zero-copy view of the payload bytes). A cold history is staged by the compressed
arms' :class:`~reelcache.restore_staging.StagingEngine` when the restorer has one (W1.2: parallel range reads, the
payload verified while it is read, device bases staged through the pinned ring), else read by the store path.
"""

from __future__ import annotations

import hashlib
import math
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from . import restore_kernels as rk
from .archive_restore import DeviceBasisCache, LineageRestorer, RestoreActivity, _PreparedCache
from .codecs import FACTOR_PACK_SCHEMA, _descriptor, _factorization_domain, _from_bytes, _json_bytes, _portable_spec
from .deferred_reads import projection_routine, run_lockstep, span_energy_routine
from .errors import ConfigurationError, IntegrityError, StateError
from .exact_blas import mm
from .node_packets import FinalTile, FinalizeItem, NodeCanvas, _entry_label
from .packed_decode import NodeRef, PreparedPacket
from .types import CodecSpec, EncodedTensor, MatrixKey, TensorContext, entry_order, entry_parts

FACTOR_ENCODER_SCHEMA = "reelcache.factor_pack_encoder.v1"
COEFFICIENT_PAYLOAD = "coefficients.f16"
COEFFICIENT_DTYPE = "float16"
_PACK_FIELDS = {"schema", "byte_order", "node", "codec", "packing", "canvas", "tiles", "payloads"}
_TILE_FIELDS = {"key", "stream", "context", "codec", "shape", "dtype", "factorization_domain", "coefficient_shape",
                "coefficient_dtype", "basis", "byte_range"}
_FP16_MAX = 65504.0


def factor_codec(spec: CodecSpec) -> dict[str, Any]:
    """The pack codec record of an SVD-only (``kind="svd"``) specification: FP16 coefficients, no HEVC."""
    if not isinstance(spec, CodecSpec) or spec.kind != "svd":
        raise ConfigurationError("factor packs carry svd (FP16 factor) matrices only")
    return {"kind": "svd", "coefficient_dtype": COEFFICIENT_DTYPE}


def is_factor_pack(value: Any) -> bool:
    """Whether ``value`` (an EncodedTensor or a pack metadata mapping) is a factor pack."""
    metadata = getattr(value, "metadata", value)
    return isinstance(metadata, Mapping) and metadata.get("schema") == FACTOR_PACK_SCHEMA


# -- finalization (planning device) -------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class FactorValues:
    """A finalized tile's stored FP16 coefficients ``[N, rank]`` (device) and their exact post-codec error."""

    values: torch.Tensor
    post_codec_error: float

    @property
    def device(self) -> torch.device:
        return self.values.device


def _stored_error(matrix: torch.Tensor, values16: torch.Tensor, rows16: torch.Tensor, tile_rows: int,
                  products: str) -> torch.Tensor:
    """``||A - P16 Q16||_F^2`` on the device: row tiles, explicit IEEE FP32 (or FP64) products, FP64 sums."""
    total = torch.zeros((), dtype=torch.float64, device=matrix.device)
    basis = rows16.double() if products == "fp64" else rows16.float()
    for start in range(0, matrix.shape[0], tile_rows):
        stop = min(start + tile_rows, matrix.shape[0])
        if products == "fp64":
            block = matrix[start:stop].double() - values16[start:stop].double() @ basis
        else:
            block = (matrix[start:stop].float() - mm(values16[start:stop].float(), basis)).double()
        total += block.square().sum()
    return total


def _factor_measure(item: FinalizeItem, tile_rows: int, products: str, workspace=None):
    """Routine: the stored measurement, ideal energy, FP16 rounding and exact post-codec error of one tile."""
    rows16, spec = item.rows16, item.spec
    rank = int(rows16.shape[0])
    if rank != spec.rank or rows16.dtype != torch.float16 or rows16.ndim != 2:
        raise ConfigurationError("finalization needs the exact FP16 rank-view rows of the resolved spec")
    projection = yield from projection_routine(item.matrix, rows16, rank=rank, tile_rows=tile_rows,
                                               condition="fast", products=products, workspace=workspace)
    stored = 1.0 if projection.zero_input else 1.0 - projection.relative_residual ** 2
    if not item.diagnostic and not projection.zero_input and stored < item.target:
        raise IntegrityError(
            f"{item.entry}: finalized stored representation measures {stored:.9f}, below {item.target}")
    source_energy = float(projection.source_norm) ** 2
    if projection.zero_input:
        ideal = 1.0
    else:
        ideal = min(1.0, (yield from span_energy_routine(projection.coefficients, rows16, source_energy,
                                                         workspace=workspace)))
    coefficients = projection.coefficients.float()
    largest = coefficients.abs().amax()
    values16 = coefficients.to(torch.float16)
    error = _stored_error(item.matrix, values16, rows16, tile_rows, products)
    finite, peak, error_energy = yield (torch.isfinite(values16).all(), largest, error)
    if not finite or not peak <= _FP16_MAX:
        raise IntegrityError(f"{item.entry}: coefficient magnitude {float(peak):.6g} overflows FP16 (the SVD-only "
                             "baseline stores FP16 factors as-is)")
    post = 0.0 if projection.zero_input or source_energy <= 0 else math.sqrt(max(float(error_energy), 0.0)
                                                                             / source_energy)
    return projection, stored, ideal, source_energy, values16, post


@torch.no_grad()
def finalize_factor_tiles(items: Sequence[FinalizeItem], *, tile_rows: int, keep_coefficients: bool = False,
                          products: str = "fp32", workspace=None) -> list[tuple[FinalTile, FactorValues]]:
    """:func:`reelcache.node_packets.finalize_tiles` for FP16 factor tiles (host reads batched in lockstep).

    Returns per item its :class:`~reelcache.node_packets.FinalTile` (no video, affine or pictures) and its stored FP16
    coefficients with their post-codec error. ``keep_coefficients`` is accepted for the interface; the post-codec error
    is always measured (exactly, on the device)."""
    from . import deferred_reads
    from .basis_versions import rows_to_bytes

    if not items:
        return []
    for item in items:
        if not isinstance(item, FinalizeItem) or item.spec.kind != "svd":
            raise ConfigurationError("factor finalization takes svd FinalizeItems")
    work = deferred_reads.TileWorkspace() if workspace is None else workspace
    measured = run_lockstep(_factor_measure(item, tile_rows, products, work) for item in items)
    results = []
    for item, (projection, stored, ideal, source_energy, values16, post) in zip(items, measured):
        tile = FinalTile(
            item.entry, item.context, item.spec, {}, {}, b"",
            rows_to_bytes(item.rows16 if item.host_rows is None else item.host_rows), int(item.rows16.shape[0]),
            (0, 0), stored, float(projection.relative_residual), float(projection.condition_bound), ideal,
            source_energy, bool(projection.zero_input), None, None,
        )
        results.append((tile, FactorValues(values16.contiguous(), post)))
    return results


@dataclass
class FactorCanvas(NodeCanvas):
    """A finalized factor node: its tiles' FP16 coefficients concatenated in tile order (``device_pictures`` is that
    buffer viewed as bytes, so the archive pipeline's staging and D2H copy apply unchanged)."""

    offsets: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def pictures(self) -> int:
        return 0

    @property
    def nbytes(self) -> int:
        return int(self.canvas["bytes"])


def assemble_factor_canvas(node_id: str, tiles: Sequence[tuple[FinalTile, FactorValues]], *,
                           codec: Mapping[str, Any], ladder: Any = None, channels: int | None = None) -> FactorCanvas:
    """Concatenate finalized factor tiles (layer, target, stream order) into one device FP16 coefficient buffer."""
    if not tiles:
        raise ConfigurationError("a node pack needs at least one tile")
    ordered = sorted(tiles, key=lambda item: entry_order(item[0].entry))
    first = ordered[0][0].context
    for tile, values in ordered:
        context = tile.context
        if (context.model != first.model or context.token_start != first.token_start
                or (context.frames, context.height, context.width) != (first.frames, first.height, first.width)):
            raise ConfigurationError("a node pack cannot span two nodes: matrices differ in token range or geometry")
        if factor_codec(tile.spec) != dict(codec):
            raise ConfigurationError("one factor pack has one codec")
        if (not isinstance(values, FactorValues) or values.values.dtype != torch.float16
                or tuple(values.values.shape) != (context.token_count, tile.rank)):
            raise IntegrityError("factor tile values differ from the tile's [tokens, rank] FP16 shape")
    flat = torch.cat([values.values.reshape(-1) for _, values in ordered])
    offsets, cursor = [], 0
    for tile, values in ordered:
        length = values.values.numel() * 2
        offsets.append((cursor, length))
        cursor += length
    record = {"policy": "factor_tiles", "pictures": 0, "bytes": cursor, "coefficient_dtype": COEFFICIENT_DTYPE,
              "tiles": len(ordered), "node_max_rank": max(tile.rank for tile, _ in ordered)}
    contexts = {tile.entry: tile.context for tile, _ in ordered}
    return FactorCanvas(node_id, contexts, dict(codec), record, [tile for tile, _ in ordered], flat.view(torch.uint8),
                        offsets=offsets, errors=[values.post_codec_error for _, values in ordered])


def encode_factor_pack(node: FactorCanvas, *, threads: int = 1, validate_decode: bool = True, packing: str = "node",
                       ffmpeg_path: str | None = None, encoder=None, encoder_lock=None) -> EncodedTensor:
    """A finalized factor node as a ``reelcache.factor_pack.v1`` packet (bases embedded; the store detaches them).

    No encoder runs (``encoder`` must be None); ``threads`` / ``validate_decode`` / ``ffmpeg_path`` are accepted for
    the node-pack interface and ignored (the post-codec error was measured exactly at finalization)."""
    started = time.perf_counter()
    if not isinstance(node, FactorCanvas):
        raise ConfigurationError("encode_factor_pack takes a FactorCanvas")
    if encoder is not None:
        raise ConfigurationError("FP16 factor packs are stored as-is: no picture encoder (no HEVC)")
    if sys.byteorder != "little":
        raise ConfigurationError("factor packs require a little-endian host")
    node.wait()
    host = node.host_pictures
    if host is None:
        raise ConfigurationError("factor canvas has not been transferred to the host")
    flat = np.asarray(host).reshape(-1)
    nbytes = node.nbytes
    if flat.dtype != np.uint8 or flat.size < nbytes:
        raise IntegrityError("host factor canvas differs from its record")
    payload = flat[:nbytes].tobytes()
    payloads = {COEFFICIENT_PAYLOAD: payload}
    descriptors = {COEFFICIENT_PAYLOAD: _descriptor(payload, "uint8", (len(payload),))}
    tiles_meta, errors, basis_bytes = [], {}, 0
    for index, (tile, (offset, length), error) in enumerate(zip(node.tiles, node.offsets, node.errors)):
        key, stream_label = entry_parts(tile.entry)
        if length != tile.context.token_count * tile.rank * 2:
            raise IntegrityError("factor tile byte range differs from its coefficient shape")
        basis_name = f"basis.{index:03d}.f16"
        tiles_meta.append({
            "key": key.to_dict(), "stream": stream_label, "context": tile.context.to_dict(),
            "codec": _portable_spec(tile.spec), "shape": list(tile.context.shape), "dtype": tile.context.dtype,
            "factorization_domain": _factorization_domain(tile.context),
            "coefficient_shape": [tile.context.token_count, tile.rank], "coefficient_dtype": COEFFICIENT_DTYPE,
            "basis": {"payload": basis_name, "view_rank": tile.rank}, "byte_range": [offset, length],
        })
        payloads[basis_name] = tile.basis
        descriptors[basis_name] = _descriptor(tile.basis, "float16", (tile.rank, tile.context.channels))
        basis_bytes += len(tile.basis)
        errors[_entry_label(tile.entry)] = float(error)
    first = node.tiles[0].context
    canvas = {name: node.canvas[name] for name in ("policy", "pictures", "bytes", "coefficient_dtype", "tiles")}
    metadata = {
        "schema": FACTOR_PACK_SCHEMA, "byte_order": "little",
        "node": {"model": first.model.to_dict(), "token_start": first.token_start, "frames": first.frames,
                 "height": first.height, "width": first.width, "node_id": node.node_id},
        "codec": dict(node.codec),
        "packing": {"mode": packing, "layout": "factor_tiles", "tile_order": "layer,target,stream"},
        "canvas": canvas, "tiles": tiles_meta, "payloads": descriptors,
    }
    metadata_bytes = len(_json_bytes(metadata))
    payload_bytes = sum(map(len, payloads.values()))
    metrics = {
        "schema": FACTOR_ENCODER_SCHEMA, "tile_count": len(tiles_meta), "coefficient_bytes": len(payload),
        "basis_bytes": basis_bytes, "payload_bytes": payload_bytes, "metadata_bytes": metadata_bytes,
        "representation_bytes": metadata_bytes + payload_bytes, "node_max_rank": node.canvas.get("node_max_rank"),
        "post_codec_relative_error": errors, "post_codec_measurement": "exact direct residual of the stored FP16 "
        "factors on the finalization device (FP64 sums)", "hevc": None,
        "tile_stream_bytes": [length for _, length in node.offsets],
        "encode_seconds": time.perf_counter() - started,
    }
    return EncodedTensor(metadata, payloads, None, metrics)


# -- validation and preparation -----------------------------------------------------------------------------------
def validate_factor_pack(encoded: EncodedTensor, *, verify: bool = False, external_bases: bool = False) -> list:
    """Check a factor pack's structure; returns per tile ``(index, tile, context, spec, rank, offset, length)``.

    ``verify`` re-hashes the payloads present (stores already verify on read); ``external_bases``: the basis
    payloads are absent (detached), only their descriptors are checked."""
    if not isinstance(encoded, EncodedTensor) or encoded.basis is not None:
        raise IntegrityError("a factor pack is an EncodedTensor without a whole-packet basis")
    metadata = encoded.metadata
    if not isinstance(metadata, Mapping) or set(metadata) != _PACK_FIELDS:
        raise IntegrityError("factor pack metadata has missing or extraneous fields")
    if metadata["schema"] != FACTOR_PACK_SCHEMA or metadata["byte_order"] != "little":
        raise IntegrityError("not a little-endian reelcache.factor_pack.v1 packet")
    if dict(metadata["codec"]) != {"kind": "svd", "coefficient_dtype": COEFFICIENT_DTYPE}:
        raise IntegrityError("factor pack codec must be FP16 svd factors")
    tiles, descriptors, payloads = metadata["tiles"], metadata["payloads"], encoded.payloads
    if not isinstance(tiles, (list, tuple)) or not tiles or not isinstance(descriptors, Mapping):
        raise IntegrityError("factor pack needs tiles and payload descriptors")
    coefficients = payloads.get(COEFFICIENT_PAYLOAD)
    descriptor = descriptors.get(COEFFICIENT_PAYLOAD)
    if not isinstance(coefficients, bytes) or not isinstance(descriptor, Mapping):
        raise IntegrityError("factor pack lacks its coefficient payload")
    if descriptor.get("bytes") != len(coefficients) or (
            verify and descriptor.get("sha256") != hashlib.sha256(coefficients).hexdigest()):
        raise IntegrityError("factor pack coefficients differ from their descriptor")
    node = metadata["node"]
    names, entries, parsed, cursor = {COEFFICIENT_PAYLOAD}, set(), [], 0
    for index, tile in enumerate(tiles):
        if not isinstance(tile, Mapping) or set(tile) != _TILE_FIELDS:
            raise IntegrityError("factor pack tile has missing or extraneous fields")
        try:
            context = TensorContext.from_dict(tile["context"])
            spec = CodecSpec.from_dict(tile["codec"])
            key = MatrixKey.from_dict(tile["key"])
        except ConfigurationError as exc:
            raise IntegrityError("factor pack tile context, codec or key is invalid") from exc
        stream = tile["stream"]
        entry = key if stream is None else (key, stream)
        if context.key != key or entry in entries or spec.kind != "svd":
            raise IntegrityError("factor pack tiles must be distinct svd matrices of their own key")
        entries.add(entry)
        if (context.token_start, context.frames, context.height, context.width) != (
                node.get("token_start"), node.get("frames"), node.get("height"), node.get("width")):
            raise IntegrityError("factor pack tiles must share the node's token range and geometry")
        rank = spec.rank
        if (type(rank) is not int or not 1 <= rank <= min(context.shape)
                or tile["coefficient_shape"] != [context.token_count, rank]
                or tile["coefficient_dtype"] != COEFFICIENT_DTYPE or tile["shape"] != list(context.shape)
                or tile["basis"].get("view_rank") != rank):
            raise IntegrityError("factor pack tile shapes are inconsistent")
        offset, length = tile["byte_range"]
        if offset != cursor or length != context.token_count * rank * 2:
            raise IntegrityError("factor pack tile byte ranges must tile the coefficient payload in order")
        cursor += length
        name = tile["basis"].get("payload")
        basis = descriptors.get(name)
        if (not isinstance(basis, Mapping) or basis.get("dtype") != "float16"
                or basis.get("shape") != [rank, context.channels] or basis.get("bytes") != rank * context.channels * 2):
            raise IntegrityError("factor pack basis descriptor is not the tile's FP16 rank-view prefix")
        raw = payloads.get(name)
        if external_bases:
            if raw is not None:
                raise IntegrityError("a detached factor pack embeds a basis")
        elif not isinstance(raw, bytes) or len(raw) != basis["bytes"] or (
                verify and hashlib.sha256(raw).hexdigest() != basis.get("sha256")):
            raise IntegrityError("factor pack basis differs from its descriptor")
        names.add(name)
        parsed.append((index, tile, context, spec, rank, offset, length))
    if cursor != len(coefficients):
        raise IntegrityError("factor pack coefficient payload has bytes no tile covers")
    if set(descriptors) != names or set(payloads) != (names if not external_bases else {COEFFICIENT_PAYLOAD}):
        raise IntegrityError("factor pack payloads and descriptors do not match")
    return parsed


@dataclass(frozen=True, eq=False)
class FactorTile:
    """One matrix of a prepared factor pack: host FP16 coefficients and its FP16 basis (host or device)."""

    key: object
    layer: int
    target: str
    index: int
    basis: torch.Tensor | None
    rank: int
    grid: tuple
    token_start: int
    coefficients: torch.Tensor
    nbytes: int
    ready: object = None


def prepare_factor_packet(encoded: EncodedTensor, *, verify: bool = False, pin_memory: bool = False,
                          bases: Sequence[tuple[torch.Tensor, Any]] | None = None, executor: Any = None,
                          validated: list | None = None) -> PreparedPacket:
    """Validate a factor pack (bases embedded or attached) for device restore: one host FP16 coefficient buffer
    (pinned with ``pin_memory``), per-tile views and host FP16 bases. A :class:`~reelcache.packed_decode.PreparedPacket`
    with :class:`FactorTile` tiles (no pictures, canvas or access units).

    The coefficients are a zero-copy view of the immutable payload bytes (W1.2; ``pin_memory`` copies them into
    pinned memory). Restore staging (:class:`reelcache.restore_staging.StagingEngine`, like
    ``packed_decode.prepare_packet``) passes the pack with detached bases plus ``bases`` -- per tile ``(device FP16
    [rank, C] view, readiness)`` -- and ``validated`` (this function's validation of the same metadata, done while
    the payload was read); ``executor`` is accepted for that interface (no access units to scan)."""
    external = bases is not None
    if validated is None:
        parsed = validate_factor_pack(encoded, verify=verify, external_bases=external)
    else:
        parsed = list(validated)
        descriptor = encoded.metadata["payloads"].get(COEFFICIENT_PAYLOAD)
        payload = encoded.payloads.get(COEFFICIENT_PAYLOAD)
        if not isinstance(payload, bytes) or not isinstance(descriptor, Mapping) or descriptor.get("bytes") != len(
                payload) or sum(item[6] for item in parsed) != len(payload):
            raise IntegrityError("factor pack coefficients differ from their validated descriptor")
    if external and len(bases) != len(parsed):
        raise IntegrityError("staged factor pack bases differ from its tiles")
    payload = encoded.payloads[COEFFICIENT_PAYLOAD]
    with torch.inference_mode(False):
        buffer = _coefficient_buffer(payload, pin_memory)
        tiles = []
        for position, (index, tile, context, spec, rank, offset, length) in enumerate(parsed):
            view = buffer[offset // 2:(offset + length) // 2].view(context.token_count, rank)
            ready = None
            if external:
                basis, ready = bases[position]
                if (not isinstance(basis, torch.Tensor) or basis.dtype != torch.float16
                        or tuple(basis.shape) != (rank, context.channels)):
                    raise IntegrityError("a staged factor tile basis is not its FP16 rank-view prefix")
            else:
                basis = _from_bytes(encoded.payloads[tile["basis"]["payload"]], "float16", (rank, context.channels))
            entry = context.key if tile["stream"] is None else (context.key, tile["stream"])
            tiles.append(FactorTile(entry, context.key.layer, context.key.target, index, basis, rank,
                                    (context.frames, context.height, context.width), context.token_start, view,
                                    length, ready))
    metadata = encoded.metadata
    return PreparedPacket(FACTOR_PACK_SCHEMA, dict(metadata["node"]), 0, 0, "factor_fp16", 0, tuple(tiles), (), True,
                          len(payload), None)


def _coefficient_buffer(payload: bytes, pin_memory: bool) -> torch.Tensor:
    """The FP16 host coefficients of a pack: a read-only zero-copy view of the immutable payload bytes (the tensor
    keeps them alive; nothing writes it), or a pinned copy with ``pin_memory``."""
    if sys.byteorder != "little":
        raise ConfigurationError("factor packs require a little-endian host")
    if pin_memory:
        return torch.from_numpy(np.frombuffer(payload, dtype="<f2").copy()).pin_memory()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given buffer is not writable")
        return torch.frombuffer(payload, dtype=torch.float16)


def decode_factor_pack(encoded: EncodedTensor, *, keys: Sequence[Any] | None = None, device: Any = "cpu",
                       dtype: Any = torch.float32) -> dict[Any, torch.Tensor]:
    """Dense ``P16 Q16`` per entry (pre-RoPE K / V, FP32 products; diagnostics and tests)."""
    prepared = prepare_factor_packet(encoded)
    wanted = None if keys is None else set(keys)
    result = {}
    for tile in prepared.tiles:
        if wanted is not None and tile.key not in wanted:
            continue
        values = tile.coefficients.to(device=device, dtype=torch.float32)
        result[tile.key] = mm(values, tile.basis.to(device=device, dtype=torch.float32)).to(dtype)
    return result


# -- device restore ---------------------------------------------------------------------------------------------
class _FactorJob:
    __slots__ = ("node_id", "node_order", "tiles", "first_layer", "nbytes", "buffer", "views", "event", "pending",
                 "submitted")

    def __init__(self, node_id, node_order, tiles):
        self.node_id, self.node_order, self.tiles = node_id, node_order, tiles
        self.first_layer = min(tile.layer for tile in tiles)
        self.nbytes = sum(tile.nbytes for tile in tiles)
        self.buffer = self.event = None
        self.views: dict[int, torch.Tensor] = {}
        self.pending = {(node_id, tile.key) for tile in tiles}
        self.submitted = False


class FactorRestorer:
    """Plans and runs the device side of factor-pack restores (the :class:`~reelcache.packed_decode.PackedRestorer`
    interface the :class:`~reelcache.archive_restore.LineageRestorer` drives): one job per layer of a node, whose FP16
    coefficients cross to the device in one copy-stream transfer; ``budget_bytes`` bounds the coefficients on the
    device (:meth:`FactorLineageRestorer._fitting_layer_ahead`)."""

    def __init__(self, device: Any, *, budget_bytes: int):
        if type(budget_bytes) is not int or budget_bytes <= 0:
            raise ConfigurationError("the factor restorer needs a positive device byte budget")
        self.device, self.budget_bytes = torch.device(device), budget_bytes
        self.copy_stream = None
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self.copy_stream = torch.cuda.Stream(device=self.device)
        self.stats_ = {"batches": 0, "jobs": 0, "entries": 0, "skipped_resident": 0, "nodes_loaded": 0,
                       "nodes_skipped": 0, "upload_bytes": 0, "upload_issue_seconds": 0.0}

    def plan(self, nodes, *, resident=(), keys=None) -> "FactorBatch":
        nodes = list(nodes)
        if not nodes or any(not isinstance(node, NodeRef) for node in nodes):
            raise ConfigurationError("nodes must be a nonempty sequence of NodeRef")
        if len({node.node_id for node in nodes}) != len(nodes):
            raise ConfigurationError("node ids must be distinct")
        resident_nodes, resident_entries = set(), set()
        for item in resident:
            if isinstance(item, str):
                resident_nodes.add(item)
            elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                resident_entries.add(item)
            else:
                raise ConfigurationError("resident items are node ids or (node_id, key) entries")
        wanted = None if keys is None else set(keys)
        jobs, entries, skipped = [], {}, 0
        for order, node in enumerate(nodes):
            if node.node_id in resident_nodes:
                self.stats_["nodes_skipped"] += 1
                continue
            self.stats_["nodes_loaded"] += 1
            for packet in node.load():
                if packet.schema != FACTOR_PACK_SCHEMA:
                    raise ConfigurationError("the factor restorer restores factor packs only")
                by_layer: dict[int, list] = {}
                for tile in packet.tiles:
                    entry = (node.node_id, tile.key)
                    if wanted is not None and tile.key not in wanted:
                        continue
                    if entry in resident_entries:
                        skipped += 1
                        continue
                    if entry in entries:
                        raise ConfigurationError(f"entry {entry!r} appears in two packets")
                    if tile.basis is None:
                        raise StateError("a factor tile reached the restore without its basis")
                    by_layer.setdefault(tile.layer, []).append(tile)
                for layer in sorted(by_layer):
                    job = _FactorJob(node.node_id, order, by_layer[layer])
                    jobs.append(job)
                    for tile in job.tiles:
                        entries[(node.node_id, tile.key)] = (job, tile)
        jobs.sort(key=lambda job: (job.first_layer, job.node_order))
        self.stats_["skipped_resident"] += skipped
        return FactorBatch(self, jobs, entries, skipped)

    def prewarm(self, packets, timeout=None) -> float:
        """No decoder sessions to create (interface of :meth:`PackedRestorer.prewarm`)."""
        return 0.0

    def stats(self) -> dict[str, Any]:
        return dict(self.stats_)

    def trim_device_cache(self, reason=None) -> int:
        return 0


class FactorBatch:
    """Device coefficient uploads of one restore and their reconstructions (the
    :class:`~reelcache.packed_decode.RestoreBatch` interface). Entries are ``(node_id, key)``; one consumer thread."""

    def __init__(self, restorer: FactorRestorer, jobs: list, entries: dict, skipped: int):
        self.restorer, self.jobs, self._entries, self.skipped = restorer, jobs, entries, skipped
        self._cursor = 0
        self._closed = self._counted = False

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self, layer: int | None = None) -> list:
        return [(job.node_id, tile.key) for job in self.jobs for tile in job.tiles
                if layer is None or tile.layer == layer]

    @property
    def layers(self) -> list[int]:
        return sorted({tile.layer for job in self.jobs for tile in job.tiles})

    def _upload(self, job: _FactorJob) -> None:
        restorer = self.restorer
        if restorer.device.type != "cuda":
            job.views = {tile.index: tile.coefficients for tile in job.tiles}
            job.submitted = True
            return
        started = time.perf_counter()
        with torch.inference_mode(False), torch.cuda.device(restorer.device), \
                torch.cuda.stream(restorer.copy_stream):
            total = sum(tile.coefficients.numel() for tile in job.tiles)
            buffer = torch.empty(total, dtype=torch.float16, device=restorer.device)
            offset = 0
            for tile in job.tiles:
                count = tile.coefficients.numel()
                target = buffer[offset:offset + count]
                target.copy_(tile.coefficients.reshape(-1), non_blocking=True)
                job.views[tile.index] = target.view(tile.coefficients.shape)
                offset += count
            event = torch.cuda.Event()
            event.record(restorer.copy_stream)
        job.buffer, job.event, job.submitted = buffer, event, True
        restorer.stats_["upload_bytes"] += 2 * total
        restorer.stats_["upload_issue_seconds"] += time.perf_counter() - started

    def submit_through(self, layer: int | None) -> None:
        """Upload every job whose first layer is <= ``layer`` (None: all)."""
        if self._closed:
            raise StateError("the restore batch is closed")
        restorer = self.restorer
        while self._cursor < len(self.jobs):
            job = self.jobs[self._cursor]
            if layer is not None and job.first_layer > layer:
                break
            self._upload(job)
            restorer.stats_["jobs"] += 1
            restorer.stats_["entries"] += len(job.tiles)
            self._cursor += 1
        if self._cursor == len(self.jobs) and self.jobs and not self._counted:
            restorer.stats_["batches"] += 1
            self._counted = True

    def _job(self, entry):
        try:
            job, tile = self._entries[entry]
        except KeyError:
            raise ConfigurationError(f"{entry!r} is not part of this restore batch") from None
        if not job.submitted:
            self.submit_through(job.first_layer)
        return job, tile

    def wait_events(self, entries, timeout=None) -> list:
        events, seen, ready = [], set(), set()
        for entry in entries:
            job, tile = self._job(entry)
            if tile.ready is not None and id(tile.ready) not in ready:
                ready.add(id(tile.ready))
                staged = tile.ready.event(timeout) if callable(getattr(tile.ready, "event", None)) else tile.ready
                if staged is not None:
                    events.append(staged)
            if id(job) not in seen:
                seen.add(id(job))
                if job.event is not None:
                    events.append(job.event)
        return events

    def reconstruction_jobs(self, entries, destination, rope=None, timeout=None) -> list:
        result = []
        for entry in entries:
            job, tile = self._job(entry)
            node_id, key = entry
            result.append(rk.ReconstructionJob(
                rk.CoefficientSource(job.views[tile.index]), destination(node_id, key, tile), tile.grid,
                basis=tile.basis, rank=tile.rank, rope=None if rope is None else rope(node_id, key, tile),
                label=f"{node_id}:{key}"))
        return result

    def reconstruct(self, entries, destination, rope=None, *, stream=None, timeout=None, **options):
        """Reconstruct ``entries`` (P.Q GEMM, RoPE, install); returns the ``RestoreHandle`` (it keeps the device
        coefficients alive until its work completes)."""
        if self._closed:
            raise StateError("the restore batch is closed")
        entries = list(entries)
        jobs = self.reconstruction_jobs(entries, destination, rope, timeout)
        events = self.wait_events(entries, timeout)  # host-waits staged bases (tickets) on every device
        if self.restorer.device.type == "cuda":
            handle = rk.reconstruct(jobs, stream=stream, wait_events=events, **options)
        else:
            handle = rk.reconstruct(jobs, **options)
        for entry in entries:
            job, _ = self._entries[entry]
            job.pending.discard(entry)
            if not job.pending:  # the handle holds what its kernels read
                job.buffer, job.views = None, {}
        return handle

    def close(self) -> None:
        """Drop every device buffer (uploads not consumed are waited for first); idempotent."""
        if self._closed:
            return
        for job in self.jobs:
            if job.submitted and job.event is not None and job.buffer is not None:
                job.event.synchronize()
            job.buffer, job.views = None, {}
        self._closed = True

    def stats(self) -> dict[str, Any]:
        submitted = [job for job in self.jobs if job.submitted]
        return {"jobs": len(self.jobs), "submitted": len(submitted), "entries": len(self._entries),
                "skipped_resident": self.skipped, "coefficient_bytes": sum(job.nbytes for job in self.jobs)}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class FactorLineageRestorer(LineageRestorer):
    """:class:`~reelcache.archive_restore.LineageRestorer` for factor packs (see the module docstring): the same history
    resolution, prepared-pack cache, device basis cache, verification records and layer-ordered reconstruction, with
    :class:`FactorRestorer` (H2D + GEMM) in place of the NVDEC packed restorer, on CUDA or on the CPU (reference
    reconstruction). ``arena_bytes`` bounds the device FP16 coefficients of one restore."""

    def __init__(self, runtime, device: Any, *, arena_bytes: int = 2 * 1024**3, basis_budget_bytes: int = 1024**3,
                 prepared_budget_bytes: int = 2 * 1024**3, adopt_bases: bool = True, staging_engine: Any = None):
        # LineageRestorer.__init__ builds an NVDEC PackedRestorer on a decoder pool; the same state without one.
        if type(adopt_bases) is not bool:
            raise ConfigurationError("adopt_bases must be boolean")
        if staging_engine is not None and not callable(getattr(staging_engine, "stage", None)):
            raise ConfigurationError("a staging engine provides stage(runtime, history)")
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        # W1.2: a cold history is staged by the compressed arms' engine (parallel reads, asynchronous payload
        # verification, staged device bases); without one, the store path (LineageRestorer._load)
        self.staging_engine = staging_engine
        self._staged: dict[str, Any] = {}
        self.last_staging = None
        self._local = threading.local()
        self.last_verification = None
        self.activity = RestoreActivity()
        self.runtime, self.pool = runtime, None
        self.adopt_bases = adopt_bases
        self.block_cache_bytes = 256 * 1024**2
        self.device = device
        self.restorer = FactorRestorer(device, budget_bytes=arena_bytes)
        self.bases = DeviceBasisCache(runtime, device, basis_budget_bytes)
        self.prepared = _PreparedCache(prepared_budget_bytes)
        self._warm_geometries: set = set()
        self.stats_ = {"restores": 0, "entries": 0, "nodes": 0, "seconds": 0.0, "plan_seconds": 0.0,
                       "device_trims": 0, "device_trimmed_bytes": 0}
        self._trim_handle = None
        if device.type == "cuda":
            from .device_pressure import register_device_trim
            self._trim_handle = register_device_trim(f"restorer:{device.index}", self.trim_device_cache,
                                                     device=device)

    def _prepare_packet(self, encoded: EncodedTensor):
        return prepare_factor_packet(encoded)

    def _fitting_layer_ahead(self, batch, layers) -> int | None:
        """None when every job's device coefficients fit ``arena_bytes`` at once, else the deepest depth whose windows
        of ``d + 2`` consecutive layers fit (at least 0), like the NVDEC restorer's canvas windows."""
        per_layer: dict[int, int] = {}
        for job in batch.jobs:
            per_layer[job.first_layer] = per_layer.get(job.first_layer, 0) + job.nbytes
        budget = self.restorer.budget_bytes
        if sum(per_layer.values()) <= budget:
            return None
        sizes = [per_layer.get(layer, 0) for layer in layers]
        depth = 0
        while depth + 1 < len(sizes):
            window = depth + 3
            if max(sum(sizes[start:start + window]) for start in range(max(1, len(sizes) - window + 1))) > budget:
                break
            depth += 1
        return depth


__all__ = ["COEFFICIENT_PAYLOAD", "FACTOR_ENCODER_SCHEMA", "FACTOR_PACK_SCHEMA", "FactorBatch", "FactorCanvas",
           "FactorLineageRestorer", "FactorRestorer", "FactorTile", "FactorValues", "assemble_factor_canvas",
           "decode_factor_pack", "encode_factor_pack", "factor_codec", "finalize_factor_tiles", "is_factor_pack",
           "prepare_factor_packet", "validate_factor_pack"]
