"""Packed node streams for latest-method lineages: canonical canvases, device finalization, CPU encode.

A committed clean node is archived as ONE all-intra ``reelcache.tensor_pack.v2``
temporal stream (Step 5a format, decodable by ``reelcache.packed_decode``) whose
tiles reference immutable basis versions instead of embedding their bases.

* **Canonical canvases.** NVDEC sessions cannot change picture size, and with
  online ranks a node's widest tile changes from chunk to chunk. A node's
  canvas is therefore taken from a fixed per-geometry ladder
  (:func:`canonical_canvas`): the canvas of the smallest ladder rank that
  covers the node's widest tile, where each ladder canvas covers every rank up
  to its step. Decoder pools can pre-create sessions for these few sizes.
  Padding samples are zero; their byte cost is measured, not assumed.
* **Device finalization** (:func:`finalize_tile`): on the planning device the
  final stored representation is measured once more with the exact arithmetic
  of the planner's verification (FP16 rows, FP32 ``P = A Q^T``, tiled direct
  residual), quantized exactly like ``_hevc.quantize_coefficients`` (the
  float64 affine arithmetic is IEEE-identical on CPU and CUDA) and laid into
  the node canvas. Only uint8 canvas pictures, affines and (optionally) FP32
  ``P`` cross PCIe; the source matrix is never copied to the host.
* **CPU encode** (:func:`encode_node_pack`): one libx265 process per node with
  the validated thread settings (``pools=T``, ``frame-threads=3`` at T >= 4;
  thread counts never change the stream bytes), tile byte ranges, optional
  validation decode (decoded-coefficient SHA256 and post-codec error) and the
  exact Step 5a metadata. Its output equals ``codecs.encode_packed`` with
  prepared factors on the same canvas byte for byte (tested).
* **External bases.** :func:`detach_pack_bases` removes the ``basis.NNN.f16``
  payloads before storage (their descriptors stay); :func:`attach_pack_bases`
  re-inserts the assembled version prefixes after checking each against its
  descriptor, so a stored pack is never decoded with a substituted basis.
"""

from __future__ import annotations

import hashlib
import math
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from . import _hevc
from . import accel_c as _accel_c
from .codecs import (
    FACTOR_PACK_SCHEMA, PACK_SCHEMA, _descriptor, _factorization_domain, _json_bytes, _pack_padding, _portable_spec,
    _unpack_tile,
)
from .deferred_reads import projection_routine, run_lockstep, span_energy_routine, upload
from .errors import ConfigurationError, IntegrityError, UnsupportedError
from .types import CodecSpec, EncodedTensor, MatrixKey, TensorContext, entry_order, entry_parts, json_copy

#: Rank steps of the canonical canvas ladder (per frames/height/width/pix_fmt geometry).
CANVAS_RANK_LADDER = (32, 64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048,
                      3072, 4096, 6144, 8192)
PACK_ENCODER_SCHEMA = "reelcache.node_pack_encoder.v1"
_LADDER_CACHE: dict[tuple, tuple] = {}


def _tile_dims(frames: int, height: int, width: int, rank: int, pix_fmt: str) -> tuple[int, int]:
    layout = _hevc.video_layout(frames, height, width, rank, pix_fmt)
    return _hevc.align(layout["image_width"]), _hevc.align(layout["image_height"])


def canvas_ladder(frames: int, height: int, width: int, pix_fmt: str, *, max_rank: int,
                  ladder: Sequence[int] = CANVAS_RANK_LADDER) -> tuple[tuple[int, int, int], ...]:
    """``(rank_step, width, height)`` canvases covering every rank up to each step (cached)."""
    for value in (frames, height, width, max_rank):
        if type(value) is not int or value < 1:
            raise ConfigurationError("canvas ladder geometry and rank must be positive integers")
    steps = tuple(ladder)
    if not steps or list(steps) != sorted(set(steps)) or any(type(step) is not int or step < 1 for step in steps):
        raise ConfigurationError("canvas rank ladder must be strictly increasing positive integers")
    key = (frames, height, width, pix_fmt, max_rank, steps)
    cached = _LADDER_CACHE.get(key)
    if cached is not None:
        return cached
    rows, widest, tallest, rank = [], _hevc.MIN_CANVAS, _hevc.MIN_CANVAS, 0
    targets = [step for step in steps if step < max_rank] + [max_rank]
    for step in targets:
        while rank < step:
            rank += 1
            tile_width, tile_height = _tile_dims(frames, height, width, rank, pix_fmt)
            widest, tallest = max(widest, tile_width), max(tallest, tile_height)
        rows.append((step, widest, tallest))
    result = tuple(rows)
    _LADDER_CACHE[key] = result
    return result


def canonical_canvas(frames: int, height: int, width: int, pix_fmt: str, node_max_rank: int, *,
                     channels: int, ladder: Sequence[int] | None = CANVAS_RANK_LADDER) -> dict[str, Any]:
    """The node canvas for a node whose widest tile has ``node_max_rank`` coefficient columns.

    ``ladder=None`` selects the node's own (smallest covering) canvas, the
    Step 5a default, for byte/time comparisons.
    """
    if type(node_max_rank) is not int or not 1 <= node_max_rank <= channels:
        raise ConfigurationError("node rank must be a positive integer within the channel count")
    if ladder is None:
        tile_width, tile_height = _tile_dims(frames, height, width, node_max_rank, pix_fmt)
        return {"width": max(tile_width, _hevc.MIN_CANVAS), "height": max(tile_height, _hevc.MIN_CANVAS),
                "rank_step": None, "policy": "own_size"}
    for step, canvas_width, canvas_height in canvas_ladder(frames, height, width, pix_fmt, max_rank=channels,
                                                           ladder=ladder):
        if step >= node_max_rank:
            if canvas_width > _hevc.MAX_DIMENSION or canvas_height > _hevc.MAX_DIMENSION:
                raise UnsupportedError("canonical canvas exceeds the HEVC provider's maximum picture dimension")
            return {"width": canvas_width, "height": canvas_height, "rank_step": step,
                    "policy": "canonical_rank_ladder"}
    raise ConfigurationError("canvas ladder does not cover the node rank")


def pack_codec(spec: CodecSpec) -> dict[str, Any]:
    """The pack codec record of an all-intra svd_hevc specification.

    libx265 specifications (QP8/ultrafast) and explicit NVENC specifications
    (``reelcache.nvenc.nvenc_spec``; encoded only by an explicit
    :class:`~reelcache.nvenc.NVENCEncoder`, see :func:`encode_node_pack`) are
    accepted; nothing substitutes one encoder for the other.
    """
    if isinstance(spec, CodecSpec) and spec.kind == "svd":  # wl-base (W1c): FP16 factor packs, no HEVC
        from .factor_packs import factor_codec
        return factor_codec(spec)
    if not isinstance(spec, CodecSpec) or spec.kind != "svd_hevc" or spec.inter_prediction is not False:
        raise ConfigurationError("node packs carry only all-intra svd_hevc coefficient matrices")
    from .nvenc import is_nvenc_preset
    if is_nvenc_preset(spec.preset):
        from .nvenc import validate_spec_parameters
        validate_spec_parameters(spec)
    else:
        _hevc.validate_parameters(spec.pix_fmt, spec.qp, spec.preset)
    return {"kind": "svd_hevc", "pix_fmt": spec.pix_fmt, "qp": spec.qp, "preset": spec.preset,
            "inter_prediction": False}


def affine_from_extrema(lo: np.ndarray, hi: np.ndarray, layout: Mapping[str, Any]) -> np.ndarray:
    """``_hevc.quantize_coefficients``' covering affine from per-column FP32 extrema."""
    lo = np.asarray(lo, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    count = layout["protected_count"]
    if layout["mode"] == "per_channel":
        return _hevc._covering_affine(lo, hi)
    affine = np.zeros(layout["affine_shape"], dtype="<f4")
    if count < lo.shape[0]:
        affine[:, 0] = _hevc._covering_affine(lo[count:].min(), hi[count:].max())
    if count:
        affine[:, 1:] = _hevc._covering_affine(lo[:count], hi[:count])
    return affine


@torch.no_grad()
def quantize_device(coefficients: torch.Tensor, spec: CodecSpec) -> tuple[torch.Tensor, np.ndarray, dict]:
    """Exactly ``_hevc.quantize_coefficients`` on the coefficients' device: (uint8 pixels, affine, layout)."""
    if (not isinstance(coefficients, torch.Tensor) or coefficients.dtype != torch.float32
            or coefficients.ndim != 2 or min(coefficients.shape) < 1):
        raise ConfigurationError("coefficients must be a nonempty FP32 matrix")
    if not bool(torch.isfinite(coefficients).all()):
        raise ConfigurationError("coefficients must be finite")
    rank = int(coefficients.shape[1])
    layout = _hevc.normalization_layout(rank, spec.normalization, spec.protected_fraction)
    extrema = torch.stack((coefficients.amin(0), coefficients.amax(0))).cpu().numpy()
    affine = affine_from_extrema(extrema[0], extrema[1], layout)
    scale, offset = _hevc._expanded_affine(affine, layout, rank)
    scale64 = torch.from_numpy(np.ascontiguousarray(scale, dtype=np.float64)).to(coefficients.device)
    offset64 = torch.from_numpy(np.ascontiguousarray(offset, dtype=np.float64)).to(coefficients.device)
    ratios = (coefficients.double() - offset64) / scale64
    ratios = torch.where(scale64 > 0, ratios, torch.zeros((), dtype=torch.float64, device=coefficients.device))
    pixels = torch.round(ratios).clamp_(0, 255).to(torch.uint8)
    return pixels, affine.astype("<f4", copy=False), layout


def mosaic_device(pixels: torch.Tensor, video: Mapping[str, Any]) -> torch.Tensor:
    """``[frames * segments, planes, rows * height, columns * width]`` channel-tile pictures of a tile."""
    frames, height, width, rank = video["frames"], video["height"], video["width"], video["rank"]
    columns, rows = video["grid"]
    planes, segments = video["planes"], video["segments"]
    slots = video["channels_per_segment"]
    values = pixels.reshape(frames, height, width, rank)
    padded = segments * slots
    if padded != rank:
        values = torch.nn.functional.pad(values, (0, padded - rank))
    values = values.reshape(frames, height, width, segments, rows, columns, planes)
    return values.permute(3, 0, 6, 4, 1, 5, 2).reshape(segments * frames, planes, rows * height, columns * width)


@dataclass(frozen=True)
class FinalTile:
    """One finalized tile: placement, affine, basis prefix bytes and measured energies."""

    entry: Any
    context: TensorContext
    spec: CodecSpec
    video: Mapping[str, Any]
    normalization: Mapping[str, Any]
    affine: bytes
    basis: bytes
    rank: int
    pictures: tuple[int, int]
    stored_energy: float
    relative_residual: float
    condition_bound: float
    ideal_energy: float
    source_energy: float
    zero_input: bool
    coefficients: np.ndarray | None = None
    gram: np.ndarray | None = None

    @property
    def key(self) -> MatrixKey:
        return entry_parts(self.entry)[0]

    @property
    def stream(self) -> str | None:
        return entry_parts(self.entry)[1]


@dataclass
class NodeCanvas:
    """A node's finalized device canvas, its tiles and (after :meth:`to_host`) host pictures."""

    node_id: str
    contexts: Mapping[Any, TensorContext]
    codec: Mapping[str, Any]
    canvas: Mapping[str, Any]
    tiles: list
    device_pictures: torch.Tensor | None
    host_pictures: np.ndarray | None = None
    event: Any = None
    finalize_seconds: float = 0.0
    keep_device: bool = False  # accel C (nvenc_device_input): NVENC reads device_pictures; no host copy

    @property
    def pictures(self) -> int:
        return int(self.canvas["pictures"])

    @property
    def nbytes(self) -> int:
        planes = 3 if self.codec["pix_fmt"] == "yuv444p" else 1
        return self.pictures * planes * self.canvas["width"] * self.canvas["height"]

    def to_host(self, *, buffer: torch.Tensor | None = None, stream: Any = None) -> None:
        """Start the D2H copy of the canvas into ``buffer`` (pinned uint8 staging) or a new host array.

        On CUDA an event is recorded on ``stream`` (default: current); call
        :meth:`wait` before reading the host pictures.
        """
        source = self.device_pictures
        if source is None:
            raise ConfigurationError("node canvas has no device pictures")
        if source.device.type != "cuda":
            self.host_pictures = source.numpy()
            self.device_pictures = None
            return
        count = source.numel()
        if buffer is None:
            host = torch.empty(source.shape, dtype=torch.uint8)
        else:
            if (not isinstance(buffer, torch.Tensor) or buffer.dtype != torch.uint8 or buffer.device.type != "cpu"
                    or buffer.numel() < count or not buffer.is_contiguous()):
                raise ConfigurationError("canvas staging must be a contiguous host uint8 buffer large enough")
            host = buffer.reshape(-1)[:count].view(source.shape)
        host.copy_(source, non_blocking=buffer is not None and buffer.is_pinned())
        event = torch.cuda.Event()
        event.record(stream if stream is not None else torch.cuda.current_stream(source.device))
        self.event = event
        self.host_pictures = host.numpy()
        self._host_tensor = host

    def keep_on_device(self, stream: Any = None) -> None:
        """accel C (``nvenc_device_input``): no D2H copy; :meth:`wait` host-waits for the canvas and keeps it."""
        source = self.device_pictures
        event = torch.cuda.Event()
        event.record(stream if stream is not None else torch.cuda.current_stream(source.device))
        self.event, self.keep_device = event, True

    def wait(self) -> None:
        """Host-wait for the D2H copy; then device pictures are released."""
        if self.event is not None:
            self.event.synchronize()
            self.event = None
        if not self.keep_device:
            self.device_pictures = None


@torch.no_grad()
def finalize_tile(matrix: torch.Tensor, rows16: torch.Tensor, context: TensorContext, spec: CodecSpec, *,
                  entry: Any, target: float, diagnostic: bool, tile_rows: int, keep_coefficients: bool = False,
                  decision_band: float = 0.0, host_rows: torch.Tensor | None = None,
                  products: str = "fp32") -> tuple[FinalTile, torch.Tensor]:
    """Final stored measurement and exact quantization of one matrix on its device.

    Returns the tile (picture range unset) and its device uint8 pixels
    ``[N, rank]``. Raises IntegrityError if the stored representation misses
    its target (a decision-device disagreement is never published).
    ``host_rows`` optionally supplies the same rows on the host for the
    stored basis bytes (no device-to-host copy). ``products`` is the
    measurement product precision (``RankMethodConfig.measurement_products``).
    """
    return finalize_tiles([FinalizeItem(matrix, rows16, context, spec, entry, target, diagnostic, host_rows)],
                          tile_rows=tile_rows, keep_coefficients=keep_coefficients, products=products)[0]


@dataclass(frozen=True)
class FinalizeItem:
    """One matrix to finalize: device source and FP16 rank-view rows (``host_rows``: the same rows on the host)."""

    matrix: torch.Tensor
    rows16: torch.Tensor
    context: TensorContext
    spec: CodecSpec
    entry: Any
    target: float
    diagnostic: bool
    host_rows: torch.Tensor | None = None


def _identity_basis(item: FinalizeItem) -> bool:
    """wl-base (W1c): whether ``item`` uses the shared full-width identity basis of the HEVC-only baseline
    (``rank_method.identity_rows``; recognized by its host rows, never by content)."""
    from .rank_method import is_identity_view
    return item.host_rows is not None and is_identity_view(item.host_rows)


def _identity_measure(item: FinalizeItem, tile_rows: int):
    """Routine: :func:`_finalize_measure` of the identity basis. The coefficients ``P = A I^T`` are the matrix itself
    (its exact FP32 widening; the generic IEEE FP32 GEMM with the identity gives the same values), so the stored and
    ideal energies are 1 and no ``N x C x C`` product or FP64 span solve runs. Quantization is the generic one."""
    rank = int(item.rows16.shape[0])
    values = item.matrix.float()
    total = torch.zeros((), dtype=torch.float64, device=values.device)
    for start in range(0, values.shape[0], tile_rows):
        total += values[start:start + tile_rows].double().square().sum()
    finite, extrema, source_energy = yield (torch.isfinite(values).all(),
                                            torch.stack((values.amin(0), values.amax(0))), total)
    if not finite:
        raise ConfigurationError("coefficients must be finite")
    source_energy = float(source_energy)
    projection = SimpleNamespace(relative_residual=0.0, condition_bound=1.0, zero_input=source_energy == 0.0)
    layout = _hevc.normalization_layout(rank, item.spec.normalization, item.spec.protected_fraction)
    extrema = extrema.astype(np.float32)
    affine = affine_from_extrema(extrema[0], extrema[1], layout)
    return projection, 1.0, 1.0, source_energy, values, affine, layout


def _finalize_measure(item: FinalizeItem, tile_rows: int, products: str = "fp32", workspace=None):
    """Routine: the stored measurement, ideal energy and quantization extrema of one tile (one or more reads)."""
    rows16, spec = item.rows16, item.spec
    rank = int(rows16.shape[0])
    if rank != spec.rank or rows16.dtype != torch.float16 or rows16.ndim != 2:
        raise ConfigurationError("finalization needs the exact FP16 rank-view rows of the resolved spec")
    if _identity_basis(item):  # wl-base (W1c): the HEVC-only baseline's full-width identity basis
        return (yield from _identity_measure(item, tile_rows))
    projection = yield from projection_routine(item.matrix, rows16, rank=rank, tile_rows=tile_rows,
                                               condition="fast", products=products, workspace=workspace)
    stored = 1.0 if projection.zero_input else 1.0 - projection.relative_residual ** 2
    if not item.diagnostic and not projection.zero_input and stored < item.target:
        raise IntegrityError(
            f"{item.entry}: finalized stored representation measures {stored:.9f}, below {item.target}")
    coefficients = projection.coefficients
    source_energy = float(projection.source_norm) ** 2
    if projection.zero_input:
        ideal = 1.0
    else:
        ideal = min(1.0, (yield from span_energy_routine(coefficients, rows16, source_energy, workspace=workspace)))
    values = coefficients.float()
    if values.dtype != torch.float32 or values.ndim != 2 or min(values.shape) < 1:
        raise ConfigurationError("coefficients must be a nonempty FP32 matrix")
    finite, extrema = yield (torch.isfinite(values).all(), torch.stack((values.amin(0), values.amax(0))))
    if not finite:
        raise ConfigurationError("coefficients must be finite")
    layout = _hevc.normalization_layout(rank, spec.normalization, spec.protected_fraction)
    extrema = extrema.astype(np.float32)
    affine = affine_from_extrema(extrema[0], extrema[1], layout)
    return projection, stored, ideal, source_energy, values, affine, layout


@torch.no_grad()
def finalize_tiles(items: Sequence[FinalizeItem], *, tile_rows: int, keep_coefficients: bool = False,
                   products: str = "fp32", workspace=None) -> list[tuple[FinalTile, torch.Tensor]]:
    """:func:`finalize_tile` for many matrices with a few batched host reads instead of ~15 per matrix.

    Identical arithmetic (:mod:`reelcache.deferred_reads`): the projections,
    ideal energies and per-column extrema of every tile are read in lockstep,
    the covering affines are computed on the host exactly like
    :func:`quantize_device`, their expanded FP64 scale/offset vectors cross to
    the device in one transfer, and quantization runs per tile on the device.
    Errors follow the sequential order of the items. ``workspace``
    (:class:`reelcache.deferred_reads.TileWorkspace`; one per call by default, a
    caller finalizing several groups may pass one) holds the host temporaries of
    the projections and the quantization; results are bitwise identical.
    """
    from . import deferred_reads
    if items and items[0].spec.kind == "svd":  # wl-base (W1c): FP16 factor tiles (reelcache.factor_packs)
        from .factor_packs import finalize_factor_tiles
        return finalize_factor_tiles(items, tile_rows=tile_rows, keep_coefficients=keep_coefficients,
                                     products=products, workspace=workspace)
    work = deferred_reads.TileWorkspace() if workspace is None else workspace
    pooled = deferred_reads.POOL_HOST_TEMPORARIES
    measured = run_lockstep(_finalize_measure(item, tile_rows, products, work) for item in items)
    vectors = []
    for item, (_, _, _, _, values, affine, layout) in zip(items, measured):
        scale, offset = _hevc._expanded_affine(affine, layout, int(values.shape[1]))
        vectors.append(torch.from_numpy(np.ascontiguousarray(scale, dtype=np.float64)))
        vectors.append(torch.from_numpy(np.ascontiguousarray(offset, dtype=np.float64)))
    devices = [values.device for *_, values, _, _ in measured]
    placed: list = [None] * len(vectors)
    for device in dict.fromkeys(devices):
        indices = [index for index in range(len(vectors)) if devices[index // 2] == device]
        for index, tensor in zip(indices, upload([vectors[index] for index in indices], device)):
            placed[index] = tensor
    from .basis_versions import rows_to_bytes
    results = []
    for position, (item, (projection, stored, ideal, source_energy, values, affine, layout)) in enumerate(
            zip(items, measured)):
        scale64, offset64 = placed[2 * position], placed[2 * position + 1]
        if pooled and values.device.type == "cpu" and values.is_contiguous():
            # The same element-wise arithmetic in one reused host buffer (no fresh FP64 temporaries).
            ratios = work.get("finalize_ratios", values.shape, torch.float64, values.device)
            ratios.copy_(values)
            ratios.sub_(offset64).div_(scale64)
            ratios.masked_fill_(~(scale64 > 0), 0.0)
            pixels = ratios.round_().clamp_(0, 255).to(torch.uint8)
        else:
            ratios = (values.double() - offset64) / scale64
            ratios = torch.where(scale64 > 0, ratios, torch.zeros((), dtype=torch.float64, device=values.device))
            pixels = torch.round(ratios).clamp_(0, 255).to(torch.uint8)
        rank = int(item.rows16.shape[0])
        context, spec = item.context, item.spec
        video = _hevc.video_layout(context.frames, context.height, context.width, rank, spec.pix_fmt)
        gram = None
        host_coefficients = None
        if keep_coefficients:
            rows = item.rows16.double()
            # wl-base (W1c): the identity basis keeps no Gram (G = I; _post_codec_error takes ||P - P'|| directly)
            gram = None if _identity_basis(item) else (rows @ rows.T).cpu().numpy()
            host_coefficients = values.cpu().numpy()
        basis = rows_to_bytes(item.rows16 if item.host_rows is None else item.host_rows)
        tile = FinalTile(
            item.entry, context, spec, video, layout, affine.astype("<f4", copy=False).tobytes(), basis, rank,
            (0, 0), stored, float(projection.relative_residual), float(projection.condition_bound), ideal,
            source_energy, bool(projection.zero_input), host_coefficients, gram,
        )
        results.append((tile, pixels))
    return results


def assemble_canvas(node_id: str, tiles: Sequence[tuple[FinalTile, torch.Tensor]], *, codec: Mapping[str, Any],
                    ladder: Sequence[int] | None, channels: int) -> NodeCanvas:
    """Lay finalized tiles (in layer, target, stream order) into one temporal canonical canvas."""
    if not tiles:
        raise ConfigurationError("a node pack needs at least one tile")
    if codec.get("kind") == "svd":  # wl-base (W1c): FP16 factor tiles, no pictures (reelcache.factor_packs)
        from .factor_packs import assemble_factor_canvas
        return assemble_factor_canvas(node_id, tiles, codec=codec, ladder=ladder, channels=channels)
    ordered = sorted(tiles, key=lambda item: entry_order(item[0].entry))
    first = ordered[0][0].context
    planes = 3 if codec["pix_fmt"] == "yuv444p" else 1
    for tile, _ in ordered:
        context = tile.context
        if (context.model != first.model or context.token_start != first.token_start
                or (context.frames, context.height, context.width) != (first.frames, first.height, first.width)):
            raise ConfigurationError("a node pack cannot span two nodes: matrices differ in token range or geometry")
        if pack_codec(tile.spec) != dict(codec):
            raise ConfigurationError("one node pack requires one pixel format, QP and preset")
    node_rank = max(tile.rank for tile, _ in ordered)
    choice = canonical_canvas(first.frames, first.height, first.width, codec["pix_fmt"], node_rank,
                              channels=channels, ladder=ladder)
    width, height = choice["width"], choice["height"]
    for tile, _ in ordered:
        if tile.video["image_width"] > width or tile.video["image_height"] > height:
            raise IntegrityError("a tile exceeds its canonical canvas")
    pictures = sum(tile.video["frames"] * tile.video["segments"] for tile, _ in ordered)
    device = ordered[0][1].device
    canvas = torch.zeros((pictures, planes, height, width), dtype=torch.uint8, device=device)
    placed, cursor = [], 0
    for tile, pixels in ordered:
        count = tile.video["frames"] * tile.video["segments"]
        images = mosaic_device(pixels, tile.video)
        canvas[cursor:cursor + count, :, :images.shape[2], :images.shape[3]] = images
        placed.append(_replace_pictures(tile, (cursor, count)))
        cursor += count
    record = {"width": width, "height": height, "alignment": _hevc.CANVAS_ALIGNMENT, "pictures": pictures,
              "planes": planes, "pix_fmt": codec["pix_fmt"], "fill": 0, "rank_step": choice["rank_step"],
              "policy": choice["policy"], "node_max_rank": node_rank}
    contexts = {tile.entry: tile.context for tile in placed}
    return NodeCanvas(node_id, contexts, dict(codec), record, placed, canvas)


def _replace_pictures(tile: FinalTile, pictures: tuple[int, int]) -> FinalTile:
    from dataclasses import replace
    return replace(tile, pictures=pictures)


def _post_codec_error(tile: FinalTile, decoded: np.ndarray) -> float:
    """Exact FP32-reconstruction error from FP32 P, decoded P' and G = Q Q^T (no source copy).

    ``||A - P' Q||^2 = ||A||^2 - 2 <P, P'> + <P' G, P'>`` with ``P = A Q^T``; the
    final cast to the native dtype is not included.
    """
    if tile.zero_input or tile.source_energy <= 0:
        return 0.0
    restored = decoded.astype(np.float64)
    if tile.gram is None:  # wl-base (W1c): the identity basis, G = I: ||A - P'||^2 with P = A
        difference = tile.coefficients.astype(np.float64) - restored
        return math.sqrt(max(float(np.einsum("ij,ij->", difference, difference)), 0.0) / tile.source_energy)
    error = (tile.source_energy - 2.0 * float(np.einsum("ij,ij->", tile.coefficients.astype(np.float64), restored))
             + float(np.einsum("ij,ij->", restored @ tile.gram, restored)))
    return math.sqrt(max(error, 0.0) / tile.source_energy)


def _validation_decode(stream: bytes, *, pictures: int, height: int, width: int, pix_fmt: str, threads: int,
                       ffmpeg_path: str | None = None) -> tuple[np.ndarray, float]:
    """FFmpeg software decode of a whole canvas stream with ``threads`` decoder threads.

    The same strict command and geometry checks as ``_hevc.decode_pictures``
    (error explosion, per-picture format/size from ``showinfo``); decoder
    threading never changes decoded pixels.
    """
    import re
    binary = _hevc.discover_ffmpeg(ffmpeg_path)
    started = time.perf_counter()
    command = [
        binary, "-hide_banner", "-loglevel", "info", "-nostdin", "-xerror", "-err_detect", "explode",
        "-threads", str(threads), "-f", "hevc", "-i", "pipe:0", "-map", "0:v:0", "-an", "-vf", "showinfo",
        "-vsync", "0", "-pix_fmt", f"+{pix_fmt}", "-threads", "1", "-frames:v", str(pictures + 1),
        "-f", "rawvideo", "pipe:1",
    ]
    samples = pictures * (3 if pix_fmt == "yuv444p" else 1) * height * width
    result = _hevc._run(command, data=stream, integrity=True, timeout=_hevc.size_scaled_timeout(
        samples, seconds_per_megasample=_hevc._DECODE_SECONDS_PER_MEGASAMPLE))
    per_frame = height * width * (3 if pix_fmt == "yuv444p" else 3 / 2)
    if len(result.stdout) != pictures * int(per_frame):
        raise IntegrityError("HEVC decoded byte count/frame count differs from the canvas")
    log = result.stderr.decode("utf-8", errors="replace")
    observed = re.findall(r"\bn:\s*\d+.*?\bfmt:(\S+).*?\bs:(\d+)x(\d+)", log)
    if len(observed) != pictures or any(fmt != pix_fmt or int(w) != width or int(h) != height
                                        for fmt, w, h in observed):
        raise IntegrityError("HEVC native pixel format or geometry differs from the canvas")
    pixels = np.frombuffer(result.stdout, dtype=np.uint8)
    if pix_fmt == "yuv444p":
        images = pixels.reshape(pictures, 3, height, width)
    else:
        images = pixels.reshape(pictures, int(per_frame))[:, :height * width].reshape(pictures, 1, height, width)
    return images, time.perf_counter() - started


def _validation_decode_dispatch(stream: bytes, *, pictures: int, height: int, width: int, pix_fmt: str, threads: int,
                                ffmpeg_path: str | None, units, random_access: bool, encoder=None):
    """accel C (``validation_decoder="nvdec"``): the validation decode on NVDEC for self-contained all-intra streams
    (normative decoding: the same pictures as FFmpeg's); otherwise :func:`_validation_decode` (FFmpeg)."""
    if random_access and _accel_c.current().validation_decoder == "nvdec":
        return _accel_c.nvdec_validation_decode(
            stream, units, pictures=pictures, height=height, width=width, pix_fmt=pix_fmt,
            pool=_accel_c.validation_pool(), gpu_id=getattr(encoder, "gpu_id", None))
    return _validation_decode(stream, pictures=pictures, height=height, width=width, pix_fmt=pix_fmt,
                              threads=threads, ffmpeg_path=ffmpeg_path)


def encode_node_pack(node: NodeCanvas, *, threads: int = 1, validate_decode: bool = True,
                     packing: str = "node", ffmpeg_path: str | None = None, encoder=None,
                     encoder_lock=None) -> EncodedTensor:
    """Encode a finalized node canvas into a ``reelcache.tensor_pack.v2`` packet (libx265 or explicit NVENC).

    Bases are embedded as ``basis.NNN.f16`` payloads (exact rank-view prefixes)
    so the in-memory packet is a complete Step 5a pack; the store detaches them.
    ``validate_decode`` decodes the stream once (FFmpeg) to record each tile's
    decoded-coefficient SHA256 and, for tiles finalized with
    ``keep_coefficients``, the post-codec error.

    ``encoder`` is an explicit :class:`reelcache.nvenc.NVENCEncoder` (RTX PRO 6000
    only; opt-in): the node's codec must be that encoder's NVENC specification,
    the stream comes from NVENC (serialized by ``encoder_lock`` when given) and
    the packet records the encoder identity in ``metadata["hevc"]``. libx265
    never encodes an NVENC specification and NVENC never a libx265 one.
    """
    if node.codec.get("kind") == "svd":  # wl-base (W1c): FP16 factor packs are stored as-is (no HEVC)
        from .factor_packs import encode_factor_pack
        return encode_factor_pack(node, threads=threads, validate_decode=validate_decode, packing=packing,
                                  ffmpeg_path=ffmpeg_path, encoder=encoder, encoder_lock=encoder_lock)
    started = time.perf_counter()
    if packing not in ("node", "layer_group", "explicit"):
        raise ConfigurationError("packed packing label must be node, layer_group or explicit")
    if type(validate_decode) is not bool:
        raise ConfigurationError("validate_decode must be boolean")
    from .nvenc import is_nvenc_preset
    if encoder is None and is_nvenc_preset(node.codec["preset"]):
        raise UnsupportedError("an NVENC node specification needs an explicit NVENCEncoder; libx265 never "
                               "substitutes for NVENC")
    if encoder is not None:
        if not is_nvenc_preset(node.codec["preset"]):
            raise ConfigurationError("an explicit NVENC encoder encodes only its NVENC specifications; libx265 "
                                     "specifications stay on libx265")
        for tile in node.tiles:
            encoder.validate_spec(tile.spec)
    node.wait()
    images = node.host_pictures
    # accel C (nvenc_device_input): NVENC encodes the device canvas the finalizer kept (no host copy)
    on_device = images is None and node.keep_device and node.device_pictures is not None and encoder is not None
    if images is None and not on_device:
        raise ConfigurationError("node canvas has not been transferred to the host")
    record = node.canvas
    codec = node.codec
    planes = 3 if codec["pix_fmt"] == "yuv444p" else 1
    if on_device:
        if (tuple(node.device_pictures.shape) != (record["pictures"], planes, record["height"], record["width"])
                or node.device_pictures.dtype != torch.uint8):
            raise IntegrityError("device canvas differs from its node canvas record")
    elif images.shape != (record["pictures"], planes, record["height"], record["width"]) or images.dtype != np.uint8:
        raise IntegrityError("host canvas differs from its node canvas record")
    with_span = time.perf_counter()
    if encoder is None:
        stream, encode_seconds, version = _hevc.encode_pictures(images, codec["pix_fmt"], ffmpeg_path=ffmpeg_path,
                                                                threads=threads)
    else:
        lock_started = time.perf_counter()
        with encoder_lock if encoder_lock is not None else _NullLock():
            locked = time.perf_counter()
            if on_device:
                stream, encode_seconds, version = encoder.encode_device_pictures(node.device_pictures,
                                                                                 codec["pix_fmt"])
            else:
                stream, encode_seconds, version = encoder.encode_pictures(images, codec["pix_fmt"])
        if on_device:
            node.device_pictures = None  # encoded: the device canvas goes
    # accel C (gil_fastpaths): the NVENC encoder's verification parse of this very stream, else the base parse
    units = (_accel_c.cached_units(stream, record["pictures"])
             or _hevc.all_intra_access_units(stream, record["pictures"]))
    random_access = all(unit["self_contained"] for unit in units)
    first_context = node.tiles[0].context
    tiles_meta, payloads, descriptors = [], {"stream.hevc": stream}, {}
    descriptors["stream.hevc"] = _descriptor(stream, "uint8", (len(stream),))
    decoded = None
    decode_seconds = 0.0
    if validate_decode:
        decoded, decode_seconds = _validation_decode_dispatch(
            stream, pictures=record["pictures"], height=record["height"], width=record["width"],
            pix_fmt=codec["pix_fmt"], threads=threads, ffmpeg_path=ffmpeg_path, units=units,
            random_access=random_access, encoder=encoder,
        )
    post_codec = {}
    basis_bytes = 0
    for index, tile in enumerate(node.tiles):
        key, stream_label = entry_parts(tile.entry)
        first, count = tile.pictures
        meta = {
            "key": key.to_dict(), "stream": stream_label, "context": tile.context.to_dict(),
            "codec": _portable_spec(tile.spec), "shape": list(tile.context.shape), "dtype": tile.context.dtype,
            "factorization_domain": _factorization_domain(tile.context),
            "coefficient_shape": [tile.context.token_count, tile.rank],
            "basis": {"payload": f"basis.{index:03d}.f16", "view_rank": tile.rank},
            "affine": f"affine.{index:03d}.f32", "normalization": dict(tile.normalization),
            "video": dict(tile.video), "pictures": [first, count],
            "rect": [0, 0, tile.video["image_width"], tile.video["image_height"]],
            "byte_range": None, "decoded_coefficients_sha256": None,
        }
        if random_access:
            start, last = units[first], units[first + count - 1]
            meta["byte_range"] = [start["offset"], last["offset"] + last["length"] - start["offset"]]
        if decoded is not None:
            restored = _unpack_tile(decoded, meta, first, tile.video)
            affine = np.frombuffer(tile.affine, dtype="<f4").reshape(tile.normalization["affine_shape"])
            coefficients = _hevc.dequantize_coefficients(restored, affine, tile.normalization)
            meta["decoded_coefficients_sha256"] = (  # accel C (gil_fastpaths): the same digest without the copy
                _accel_c.coefficient_digest(coefficients) if _accel_c.fast()
                else hashlib.sha256(coefficients.astype("<f4", copy=False).tobytes()).hexdigest())
            if tile.coefficients is not None:
                post_codec[tile.entry] = _post_codec_error(tile, coefficients)
        basis_name, affine_name = meta["basis"]["payload"], meta["affine"]
        payloads[basis_name] = tile.basis
        descriptors[basis_name] = _descriptor(tile.basis, "float16", (tile.rank, tile.context.channels))
        payloads[affine_name] = tile.affine
        descriptors[affine_name] = _descriptor(tile.affine, "float32", tile.normalization["affine_shape"])
        basis_bytes += len(tile.basis)
        tiles_meta.append(meta)
    tiers = sys.modules.get("reelcache.decoded_tier") if decoded is not None else None
    if tiers is not None:  # accel A1: open decoded tiers keep this normative decode (no tier: never imported)
        tiers.offer_decoded(descriptors["stream.hevc"]["sha256"], decoded, tiles_meta, model=first_context.model)
    del decoded
    canvas_record = {name: record[name] for name in ("width", "height", "alignment", "pictures", "planes",
                                                     "pix_fmt", "fill")}
    metadata = {
        "schema": PACK_SCHEMA, "byte_order": "little",
        "node": {"model": first_context.model.to_dict(), "token_start": first_context.token_start,
                 "frames": first_context.frames, "height": first_context.height, "width": first_context.width,
                 "node_id": node.node_id},
        "codec": dict(codec),
        "hevc": _hevc.encoder_record(False) if encoder is None else encoder.encoder_record(codec["pix_fmt"]),
        "packing": {"mode": packing, "layout": "temporal", "pixel_identical_to_same_canvas_matrix": True,
                    "tile_order": "layer,target,stream",
                    "random_access": "tile_byte_ranges" if random_access else "whole_stream"},
        "canvas": canvas_record, "tiles": tiles_meta, "payloads": descriptors,
        "padding": _pack_padding(tiles_meta, canvas_record),
    }
    metadata_bytes = len(_json_bytes(metadata))
    payload_bytes = sum(map(len, payloads.values()))
    metrics = {
        "schema": PACK_ENCODER_SCHEMA, "tile_count": len(tiles_meta), "pictures": record["pictures"],
        "canvas_policy": record.get("policy"), "canvas_rank_step": record.get("rank_step"),
        "node_max_rank": record.get("node_max_rank"),
        "hevc_encode_seconds": encode_seconds, "hevc_encode_threads": threads,
        "x265_thread_params": _hevc.x265_thread_params(threads, record["pictures"]),
        "hevc_validation_decode": validate_decode, "hevc_validation_decode_seconds": decode_seconds,
        "hevc_inter_prediction": False, "hevc_picture_structure": "all_intra_idr", "ffmpeg_version": version,
        "hevc_software_only": True, "stream_bytes": len(stream), "basis_bytes": basis_bytes,
        "payload_bytes": payload_bytes, "metadata_bytes": metadata_bytes,
        "representation_bytes": metadata_bytes + payload_bytes,
        "post_codec_relative_error": {_entry_label(entry): value for entry, value in post_codec.items()},
        "encode_seconds": time.perf_counter() - started, "pack_seconds": time.perf_counter() - with_span,
        "tile_stream_bytes": [meta["byte_range"][1] if meta["byte_range"] else None for meta in tiles_meta],
        # Each picture's access-unit length per tile, in picture order (random-access packs; SGLang Step 7 format
        # addition): decoders find a tile's access units without scanning its byte range
        # (:func:`recorded_access_units`); packs without it are scanned as before.
        "tile_access_units": ([[unit["length"] for unit in units[meta["pictures"][0]:sum(meta["pictures"])]]
                               for meta in tiles_meta] if random_access else None),
    }
    if encoder is not None:
        metrics.update(ffmpeg_version=None, hevc_software_only=False, hevc_encoder=encoder.name,
                       hevc_encoder_version=version, hevc_encode_threads=None, x265_thread_params=None,
                       hevc_encoder_lock_wait_seconds=locked - lock_started)
    encoded = EncodedTensor(metadata, payloads, None, metrics)
    _accel_c.record_digests(encoded, payloads, descriptors)  # accel C (hash_once): the descriptors' digests
    return encoded


def recorded_access_units(metadata: Mapping[str, Any], metrics: Mapping[str, Any] | None) -> list | None:
    """Per tile, each picture's access unit as ``(offset, length)`` in the pack stream, from the lengths recorded at
    encode time (``metrics["tile_access_units"]``), or None when the pack does not record them (older packs, whole-
    stream packs): the caller then scans the tile's byte range as before.

    Checked against the tile metadata: one length per tile picture, positive, summing to the tile's byte range (the
    pack's metrics are part of its content-addressed manifest).
    """
    recorded = None if metrics is None else metrics.get("tile_access_units")
    if recorded is None:
        return None
    tiles = metadata.get("tiles")
    if not isinstance(recorded, list) or not isinstance(tiles, list) or len(recorded) != len(tiles):
        raise IntegrityError("recorded access units do not cover the pack's tiles")
    result = []
    for tile, lengths in zip(tiles, recorded):
        byte_range, pictures = tile.get("byte_range"), tile.get("pictures")
        if (not isinstance(lengths, list) or byte_range is None or not isinstance(pictures, list)
                or len(lengths) != pictures[1] or any(type(length) is not int or length <= 0 for length in lengths)
                or sum(lengths) != byte_range[1]):
            raise IntegrityError("recorded access units disagree with their tile's pictures and byte range")
        offset, spans = byte_range[0], []
        for length in lengths:
            spans.append((offset, length))
            offset += length
        result.append(spans)
    return result


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _entry_label(entry: Any) -> str:
    key, stream = entry_parts(entry)
    return f"{key.layer}:{key.target}" + ("" if stream is None else f":{stream}")


def detach_pack_bases(encoded: EncodedTensor) -> tuple[dict[str, Any], dict[str, bytes], list[bytes]]:
    """Split a node pack into (metadata, non-basis payloads, per-tile basis bytes)."""
    metadata = json_copy(encoded.metadata)
    if metadata.get("schema") not in (PACK_SCHEMA, FACTOR_PACK_SCHEMA) or encoded.basis is not None:
        raise ConfigurationError("only tensor_pack.v2 node packs detach per-tile bases")
    payloads = dict(encoded.payloads)
    bases = []
    for tile in metadata["tiles"]:
        name = tile["basis"]["payload"]
        raw = payloads.pop(name, None)
        if raw is None:
            raise IntegrityError("node pack lacks a tile basis payload")
        descriptor = metadata["payloads"][name]
        if len(raw) != descriptor["bytes"] or (  # accel C (hash_once): the digest of this very object, if recorded
                _accel_c.known_digest(encoded, name, raw) or hashlib.sha256(raw).hexdigest()) != descriptor["sha256"]:
            raise IntegrityError("node pack tile basis differs from its descriptor")
        bases.append(raw)
    return metadata, payloads, bases


def attach_pack_bases(metadata: Mapping[str, Any], payloads: Mapping[str, bytes],
                      bases: Sequence[bytes], *, check_digests: bool = True) -> EncodedTensor:
    """Re-insert assembled basis prefixes after checking each against its tile descriptor.

    ``check_digests=False`` skips only the descriptor SHA-256 (geometry, dtype and length are still checked), for
    prefixes assembled from row blocks the store hash-checked in a store whose manifests -- these descriptors
    included -- are hash-checked on read as well (``verify_reads``): the digest would re-hash authenticated bytes.
    """
    tiles = metadata["tiles"]
    if len(bases) != len(tiles):
        raise IntegrityError("one assembled basis prefix is required per pack tile")
    combined = dict(payloads)
    for tile, raw in zip(tiles, bases):
        name = tile["basis"]["payload"]
        descriptor = metadata["payloads"].get(name)
        if name in combined or not isinstance(descriptor, Mapping):
            raise IntegrityError("stored node pack embeds a basis or lacks its descriptor")
        channels = TensorContext.from_dict(tile["context"]).channels
        if descriptor.get("shape") != [tile["basis"]["view_rank"], channels] or descriptor.get("dtype") != "float16":
            raise IntegrityError("node pack basis descriptor is not the tile's rank-view prefix")
        if len(raw) != descriptor.get("bytes") or (
                check_digests and hashlib.sha256(raw).hexdigest() != descriptor.get("sha256")):
            raise IntegrityError("assembled basis version prefix differs from the packed tile descriptor")
        combined[name] = raw
    return EncodedTensor(metadata, combined, None, {})


def pack_tile_index(metadata: Mapping[str, Any]) -> dict[Any, int]:
    """Tile index of every ``MatrixKey`` / ``(MatrixKey, stream)`` entry of a pack."""
    result = {}
    for index, tile in enumerate(metadata["tiles"]):
        key = MatrixKey.from_dict(tile["key"])
        entry = key if tile["stream"] is None else (key, tile["stream"])
        result[entry] = index
    return result
