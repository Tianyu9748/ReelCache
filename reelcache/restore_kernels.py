"""Batched, fused reconstruction of restored KV matrices into native caches.

This is the reconstruction half of the optimized restore path (integration
report, section 3.5, item 3). One call reconstructs any number of restored
matrices -- one layer's K or V for one chunk each -- directly into native cache
slots::

    pre-RoPE K or V  =  dequantize(unpack(pixels)) @ Q[:rank]      (per chunk)
    native slot      <- cast(RoPE(K at its original positions))    (K)
    native slot      <- cast(V)                                    (V)

Conventions (unchanged from the ReelCache packet contract):

* A canonical matrix is ``[N, C]`` with ``N = frames * height * width`` tokens
  in (frame, row, column) order and ``C = heads * head_dim`` channels.
* The basis is the immutable FP16 row matrix ``Q[R, C]``; a chunk uses the
  prefix rank view ``Q[:rank]``. Several chunks may share one basis tensor and
  use different rank views; chunks with different bases may share one launch.
* Coefficients are ``P[N, rank]`` (FP32 canonically). HEVC packets carry them
  as 8-bit pixels with an FP32 affine per rank column; the dequantization is
  ``float(pixel) * scale + offset`` with two roundings (no FMA), bitwise equal
  to ``codecs.dequantize_device_coefficients``.
* The pixel layout of one matrix is ``reelcache.channel_tiles.v1``
  (``_hevc.video_layout``): rank column ``k`` lives in segment
  ``k // channels_per_segment``; inside a segment, cell ``i`` uses plane
  ``i % planes`` and grid cell ``divmod(i // planes, grid_columns)``; every
  frame of the chunk is one picture per segment. A :class:`PictureSource`
  gives, for each (segment, frame), a uint8 view whose origin is the matrix's
  rectangle on a decoded picture, so packed (schema v2) canvases are read in
  place.
* Only normalized pre-RoPE K is factorized. RoPE is re-applied at the original
  positions from caller-supplied cos/sin tables generated with the model's own
  frequency function (:class:`RopeTable`). Interleaved pairs ``(2j, 2j+1)`` are
  rotated as complex numbers, as in Wan, SGLang causal Wan and LongCat.
* Native layouts: Wan/SGLang ``[1, T, H, D]`` (``"BTHD"``), RAVEN packed
  ``[T, H, D]`` (``"THD"``) and LongCat ``[1, H, T, D]`` (``"BHTD"``); any
  strides are accepted, including non-contiguous views.

Backends: ``auto`` (the default: on CUDA the measured policy -- ``fused`` on
Triton <= 3.4, picture sources on ``two_pass`` on Triton >= 3.5; ``reference``
on CPU), ``reference`` (plain torch, FP64 or FP32 math; CPU or GPU), ``fused``
(one Triton kernel per launch group: unpack + dequantize + GEMM + RoPE + cast +
strided store), ``two_pass`` (a batched Triton unpack of pictures into a uint8
``[N, rank]`` matrix, then the fused kernel on it), ``staged`` (Triton unpack
-> cuBLAS GEMM with FP32 output -> Triton RoPE/cast/store; for comparison and
stage timing; its fp32/tf32 GEMMs follow the process's FP32 matmul precision,
which this module never sets) and ``interpret`` (the fused kernels under the Triton CPU
interpreter, which must be enabled with ``TRITON_INTERPRET=1`` before Triton is
imported; tests only). The module imports without CUDA or Triton; then only the
reference backend is available. There is no silent fallback: on CUDA tensors
``auto`` raises :class:`UnsupportedError` (with Triton's import error,
:data:`TRITON_IMPORT_ERROR`) when the compiled kernels are unavailable --
Triton missing or failing to import, or ``TRITON_INTERPRET=1`` -- exactly like
``fused``; the reference runs only when requested (``backend="reference"``) or
for CPU tensors.

Precision modes (FP32 accumulation everywhere): ``fp32`` (SIMT FMA),
``tf32``, ``tf32x3`` (3xTF32 split), ``fp16`` (coefficients rounded to FP16;
the FP16 basis is exact), ``fp16x2`` (coefficients split into FP16 hi + lo,
two tensor-core products: near-FP32 accuracy) and ``bf16``. Coefficients are
prescaled by an exact power of two in the FP16 modes. :func:`error_bound`
documents the elementwise bound each mode satisfies.

The module never changes process-wide torch settings (TF32 / FP32 matmul
precision, reduced-precision reductions, cuDNN, determinism, default dtype or
threads): the model generating in the same process keeps its numerics. The
Triton backends choose their precision per kernel; the two paths that run
torch GEMMs follow the process's settings and refuse with
:class:`UnsupportedError` when those do not give the requested precision
(``staged`` fp32/tf32 need IEEE/TF32 FP32 matmuls, ``staged`` fp16 needs FP32
accumulation, the FP32 reference needs IEEE FP32; see
:func:`fp32_matmul_precision`). Its only process-wide patch is a Triton
interpreter compatibility shim, applied in ``interpret`` mode (tests) only.

GPU work runs on a caller-provided stream (default: the current stream) after
optional ``wait_events``; the returned :class:`RestoreHandle` records a
completion event and keeps every input, table and temporary alive until it is
released (or, if dropped unreleased, until the event completes), so inputs are
never freed before the event. The FP32 device copies of a :class:`RopeTable`
are shared by all streams and carry their own completion event, which every
consumer stream waits on (:meth:`RopeTable.kernel_tables`).

Triton compiles one kernel variant per combination of compile-time parameters,
some of which depend on the call (its token-tile count on Triton <= 3.4, the
largest rank of a launch group, the rows' alignment); a cold process compiles
or loads each on first use, inside its first restores. :func:`warmup` compiles
and loads, ahead of time, every variant the restore path launches for a set of
:class:`WarmupShape` launch groups (call it when a worker attaches).
"""

import math
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .errors import ConfigurationError, IntegrityError, StateError, UnsupportedError

try:  # Optional: the reference backend is plain torch.
    import triton
    import triton.language as tl
except Exception as _import_error:  # noqa: BLE001 - any import failure disables the Triton kernels
    triton = None
    tl = None
    # Reported by every CUDA call that needs the kernels (backend "auto" included): never a silent fallback.
    TRITON_IMPORT_ERROR = f"{type(_import_error).__name__}: {_import_error}"
else:
    TRITON_IMPORT_ERROR = None

TRITON_AVAILABLE = triton is not None


def _triton_version():
    if not TRITON_AVAILABLE:
        return (0, 0)
    try:
        major, minor = triton.__version__.split(".")[:2]
        return int(major), int(minor)
    except ValueError:
        return (99, 0)


# Default policy measured on RTX PRO 6000 and A100-80GB (README, "policy" runs):
# Triton <= 3.4 compiles the in-kernel picture gather well with tokens across
# threads (PIC_T), so one fused kernel with 64-token tiles and up to 4 channel
# tiles per program wins (per-layer calls); Triton >= 3.5 (3.7 measured) prefers
# the batched two-pass picture path with 32-token tiles and one channel tile per
# program.
_TRITON_VERSION = _triton_version()
_LEGACY_TRITON = _TRITON_VERSION < (3, 5)

LAYOUTS = ("BTHD", "THD", "BHTD")
PRECISIONS = ("fp32", "tf32", "tf32x3", "fp16", "fp16x2", "bf16")
BACKENDS = ("auto", "reference", "fused", "two_pass", "staged", "interpret")
OUTPUT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
COEFFICIENT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

# Selected from the RTX PRO 6000 / A100-80GB measurements documented in
# artifacts/sglang_integration/benchmarks/restore_kernels/README.md.
_RECOMMENDED_PRECISION = {(8, 0): "fp16x2", (9, 0): "fp16x2", (12, 0): "fp16x2"}
_DEFAULT_PRECISION = "fp16x2"
_DEFAULT_CONFIG = {"BLOCK_M": 64 if _LEGACY_TRITON else 32, "BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4,
                   "num_stages": 2, "N_GROUP": 4 if _LEGACY_TRITON else 1, "PIC_T": None}
_ARCH_CONFIGS = {
    # A100 with Triton <= 3.4: 16-wide K steps measured equal or faster for every model (LongCat -25%).
    (8, 0): {**_DEFAULT_CONFIG, "BLOCK_K": 16} if _LEGACY_TRITON else dict(_DEFAULT_CONFIG),
    (9, 0): dict(_DEFAULT_CONFIG),
    (12, 0): dict(_DEFAULT_CONFIG),
}
# FP16 operands are prescaled per matrix by an exact power of two so that the
# largest coefficient magnitude lies in (2**13, 2**14]: no overflow in the
# hi/lo split and the low part keeps full FP16 precision (see _prescale).
_PRECISION_CODES = {name: index for index, name in enumerate(PRECISIONS)}

# Per-job descriptor (int64). The fused, staged and RoPE kernels share it.
_FIELDS = (
    "tokens", "rank", "frames", "grid_h", "grid_w",
    "src_ptr", "src_row_stride", "src_col_stride",
    "scale_ptr", "offset_ptr", "bound_ptr",
    "basis_ptr", "basis_row_stride",
    "dst_ptr", "dst_token_stride", "dst_head_stride", "dst_dim_stride",
    "rope", "cos_ptr", "sin_ptr", "rope_row_stride",
    "row_t", "row_h", "row_w", "sec_t", "sec_h", "rope_dense",
    "pic_slots", "pic_planes", "pic_grid_cols", "pic_plane_stride", "pic_row_stride", "pic_table",
    "pic_base", "pic_frame_stride", "work_ptr", "work_row_stride", "needs_bound",
)
_F = {name: index for index, name in enumerate(_FIELDS)}
_DESC = 40
assert len(_FIELDS) <= _DESC


def _int(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def _require_tensor(value, name):
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise ConfigurationError(f"{name} must be a strided torch.Tensor")
    if value.device.type == "meta":
        raise ConfigurationError(f"{name} cannot be a meta tensor")
    return value


def _host_vector(value, length, name):
    """FP32 host copy of a per-rank-column affine vector (scale or offset)."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ConfigurationError(f"{name} must be a host array; pass the packet affine payload")
        value = value.detach().numpy()
    array = np.asarray(value)
    if array.dtype != np.float32 or array.shape != (length,):
        raise ConfigurationError(f"{name} must be a float32 vector with one entry per rank column")
    if not np.isfinite(array).all():
        raise ConfigurationError(f"{name} must be finite")
    return np.ascontiguousarray(array)


def expand_affine(affine, normalization, rank):
    """Per-column (scale, offset) from a packet affine payload.

    Mirrors ``_hevc._expanded_affine`` for the ``per_channel``, ``global`` and
    ``leading_columns`` normalization layouts.
    """
    affine = np.asarray(affine)
    if affine.dtype != np.dtype("<f4") or affine.ndim != 2 or affine.shape[0] != 2:
        raise ConfigurationError("affine must be a little-endian float32 [2, records] array")
    if not isinstance(normalization, Mapping) or "mode" not in normalization:
        raise ConfigurationError("normalization must be the packet normalization layout")
    _int(rank, "rank", minimum=1)
    if normalization["mode"] == "per_channel":
        if affine.shape[1] != rank:
            raise ConfigurationError("per-channel affine must have one record per rank column")
        return affine[0].copy(), affine[1].copy()
    scale = np.full(rank, affine[0, 0], dtype=np.float32)
    offset = np.full(rank, affine[1, 0], dtype=np.float32)
    count = normalization["protected_count"]
    if type(count) is not int or not 0 <= count <= rank or affine.shape[1] != count + 1:
        raise ConfigurationError("affine record count differs from the protected columns")
    if count:
        scale[:count], offset[:count] = affine[0, 1:], affine[1, 1:]
    return scale, offset


# ---------------------------------------------------------------------------
# Destinations, RoPE tables and sources
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class NativeView:
    """A native cache tensor, its layout and the first destination token.

    ``"BTHD"``: Wan/SGLang ``[B, T, H, D]``; ``"THD"``: RAVEN packed
    ``[T, H, D]``; ``"BHTD"``: LongCat ``[B, H, T, D]``. Any strides are
    accepted; the kernels write element by element through them.
    """

    tensor: torch.Tensor
    layout: str = "BTHD"
    token_offset: int = 0
    batch_index: int = 0

    def __post_init__(self):
        tensor = _require_tensor(self.tensor, "destination tensor")
        if self.layout not in LAYOUTS:
            raise ConfigurationError(f"destination layout must be one of {LAYOUTS}")
        if tensor.ndim != (3 if self.layout == "THD" else 4):
            raise ConfigurationError(f"{self.layout} destination has the wrong number of dimensions")
        if tensor.dtype not in OUTPUT_DTYPES:
            raise ConfigurationError("destination dtype must be bfloat16, float16 or float32")
        _int(self.token_offset, "token_offset")
        _int(self.batch_index, "batch_index")
        if self.layout == "THD" and self.batch_index:
            raise ConfigurationError("a packed THD destination has no batch dimension")
        if self.layout != "THD" and self.batch_index >= tensor.shape[0]:
            raise ConfigurationError("batch_index is outside the destination")
        if self.head_dim % 2:
            raise ConfigurationError("head_dim must be even")
        if any(size > 1 and stride == 0 for size, stride in zip(tensor.shape, tensor.stride())):
            raise ConfigurationError("destination views with broadcast (zero) strides alias")

    @property
    def heads(self):
        return self.tensor.shape[1] if self.layout == "BHTD" else self.tensor.shape[-2]

    @property
    def head_dim(self):
        return self.tensor.shape[-1]

    @property
    def channels(self):
        return self.heads * self.head_dim

    @property
    def capacity(self):
        return self.tensor.shape[2] if self.layout == "BHTD" else self.tensor.shape[-3]

    def strides(self):
        """Element strides of (token, head, dim)."""
        stride = self.tensor.stride()
        if self.layout == "BHTD":
            return stride[2], stride[1], stride[3]
        return stride[-3], stride[-2], stride[-1]

    def element_offset(self):
        """Element offset of (batch_index, token_offset, head 0, dim 0)."""
        base = self.tensor.storage_offset()
        if self.layout != "THD":
            base += self.batch_index * self.tensor.stride(0)
        return base + self.token_offset * self.strides()[0]

    def slot(self, tokens):
        """A ``[tokens, H, D]`` view of the destination slot."""
        stop = self.token_offset + tokens
        if self.layout == "THD":
            return self.tensor[self.token_offset:stop]
        if self.layout == "BTHD":
            return self.tensor[self.batch_index, self.token_offset:stop]
        return self.tensor[self.batch_index, :, self.token_offset:stop].transpose(0, 1)


def wan_sections(pairs):
    """Wan/LongCat 3D split of ``pairs = head_dim // 2`` into (t, h, w) pairs."""
    _int(pairs, "pairs", minimum=1)
    return (pairs - 2 * (pairs // 3), pairs // 3, pairs // 3)


_UPLOAD_STREAMS = {}


def _upload_stream(device):
    """Private per-device stream for RoPE table uploads from host memory (never waits on caller work).

    Taken from Torch's high-priority stream pool so that it does not alias
    the default-priority pool streams callers use (decode sessions etc.).
    """
    stream = _UPLOAD_STREAMS.get(device)
    if stream is None:
        stream = _UPLOAD_STREAMS.setdefault(device, torch.cuda.Stream(device, priority=-1))
    return stream


class _KernelTables:
    """FP32 ``cos``/``sin`` of one :class:`RopeTable` on one device and the event that completes them.

    ``stamp`` is the table state the copies were made from (see
    ``RopeTable._stamp``). ``event`` is None once the copies are known to be
    complete (or when the table's own tensors are used in place); ``staging``
    keeps the pinned host source of an upload alive until then.
    """

    __slots__ = ("cos", "event", "sin", "staging", "stamp")

    def __init__(self, cos, sin, stamp, event=None, staging=None):
        self.cos, self.sin, self.stamp, self.event, self.staging = cos, sin, stamp, event, staging

    @classmethod
    def make(cls, cos, sin, device, stream, stamp):
        if device.type == "cuda" and all(table.device == device and table.dtype == torch.float32
                                         and table.is_contiguous() for table in (cos, sin)):
            return cls(cos, sin, stamp)  # the caller's tensors, ordered like every other input
        # Copies are persistent internal buffers: normal tensors whatever the caller's mode (they may be filled
        # under torch.inference_mode and reused outside it, or the reverse); they never require grad.
        with torch.inference_mode(False), torch.no_grad():
            return cls._copy(cos, sin, device, stream, stamp)

    @classmethod
    def _copy(cls, cos, sin, device, stream, stamp):
        if device.type != "cuda":
            return cls(cos.to(device=device, dtype=torch.float32).contiguous(),
                       sin.to(device=device, dtype=torch.float32).contiguous(), stamp)
        if cos.device.type == "cpu":
            # Host tables are read now (as a synchronous copy would) into private pinned memory; the upload
            # depends on no caller work, so it runs on a private stream and never blocks the host.
            staging = tuple(torch.empty(table.shape, dtype=torch.float32, pin_memory=True).copy_(table)
                            for table in (cos, sin))
            upload = _upload_stream(device)
            with torch.cuda.stream(upload):
                copies = tuple(torch.empty(table.shape, dtype=torch.float32, device=device) for table in staging)
                for target, source in zip(copies, staging):
                    target.copy_(source, non_blocking=True)
                event = torch.cuda.Event()
                event.record(upload)
            return cls(*copies, stamp, event, staging)
        # A device table is a caller input: convert it on the calling stream, after that call's wait_events.
        stream = torch.cuda.current_stream(device) if stream is None else stream
        with torch.cuda.stream(stream):
            copies = tuple(table.to(device=device, dtype=torch.float32).contiguous() for table in (cos, sin))
            event = torch.cuda.Event()
            event.record(stream)
        return cls(*copies, stamp, event)

    def order_before(self, device, stream):
        """Make ``stream`` (default: current) wait until the copies are written; no-op once complete."""
        event = self.event
        if event is None:
            return
        if event.query():
            self.event = None
            self.staging = None
            return
        (torch.cuda.current_stream(device) if stream is None else stream).wait_event(event)


@dataclass(frozen=True, eq=False)
class RopeTable:
    """cos/sin rotation tables ``[R, head_dim // 2]`` for interleaved pairs.

    ``sections=(t, h, w)`` makes the table separable: pair ``j`` of a token at
    (frame f, row y, column x) uses row ``t_row + f`` when ``j < t``, row
    ``h_row + y`` when ``j < t + h`` and row ``w_row + x`` otherwise, so the
    native Wan ``freqs`` tensor is used as is. ``sections=None`` is a dense
    per-token table (row ``row + n`` for token ``n``), e.g. SGLang's
    ``freqs_cis``. Tables are float64 or float32; GPU kernels use float32
    copies (:meth:`kernel_tables`), the reference uses the table's own precision.

    The device copies are cached on the table. Tables of normal tensors are
    re-copied automatically after in-place updates (version counters). Tables
    built from inference tensors (``torch.inference_mode``, e.g. inside SGLang)
    have no version counter: treat them as immutable, or call
    :meth:`invalidate` after updating ``cos``/``sin`` in place. Tables, jobs and
    calls may mix inference and normal tensors and run inside or outside
    ``torch.inference_mode``.
    """

    cos: torch.Tensor
    sin: torch.Tensor
    sections: tuple | None = None
    section_rows: tuple | None = None
    _device_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _inference: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self):
        cos, sin = _require_tensor(self.cos, "RoPE cos"), _require_tensor(self.sin, "RoPE sin")
        if (
            cos.ndim != 2 or cos.shape != sin.shape or cos.dtype != sin.dtype
            or cos.dtype not in (torch.float32, torch.float64) or cos.device != sin.device
            or min(cos.shape) < 1
        ):
            raise ConfigurationError("RoPE cos/sin must be matching float32/float64 [rows, head_dim // 2] tensors")
        # Fixed at tensor creation; inference tensors track no version counter (see _stamp).
        object.__setattr__(self, "_inference", cos.is_inference() or sin.is_inference())
        rows, pairs = cos.shape
        if self.sections is not None:
            sections = tuple(self.sections)
            if len(sections) != 3 or any(type(v) is not int or v < 0 for v in sections) or sum(sections) != pairs:
                raise ConfigurationError("RoPE sections must be three nonnegative pair counts summing to head_dim // 2")
            object.__setattr__(self, "sections", sections)
            limits = (rows, rows, rows) if self.section_rows is None else tuple(self.section_rows)
            if len(limits) != 3 or any(type(v) is not int or not 0 <= v <= rows for v in limits):
                raise ConfigurationError("section_rows must be three row counts within the table")
            object.__setattr__(self, "section_rows", limits)
        elif self.section_rows is not None:
            raise ConfigurationError("section_rows requires separable sections")

    @property
    def rows(self):
        return self.cos.shape[0]

    @property
    def pairs(self):
        return self.cos.shape[1]

    @property
    def head_dim(self):
        return 2 * self.pairs

    @classmethod
    def from_complex(cls, freqs, sections="wan"):
        """From complex ``freqs[rows, head_dim // 2]`` (Wan ``rope_params`` concatenation)."""
        freqs = _require_tensor(freqs, "freqs")
        if not freqs.is_complex() or freqs.ndim != 2:
            raise ConfigurationError("freqs must be a complex [rows, head_dim // 2] tensor")
        real = torch.view_as_real(freqs.resolve_conj())
        dtype = torch.float64 if freqs.dtype == torch.complex128 else torch.float32
        if sections == "wan":
            sections = wan_sections(freqs.shape[1])
        return cls(real[..., 0].to(dtype).contiguous(), real[..., 1].to(dtype).contiguous(), sections)

    @classmethod
    def from_axes(cls, temporal, height, width):
        """Separable table from per-axis complex tensors ``[positions, pairs]``.

        Use this for non-integer temporal positions (LongLive 2.0 multi-shot
        offsets): pass the model's own temporal frequencies for the restored
        frames. Unused cells are NaN so an out-of-range row is detectable.
        """
        axes = [_require_tensor(value, "axis freqs") for value in (temporal, height, width)]
        if any(not value.is_complex() or value.ndim != 2 for value in axes):
            raise ConfigurationError("axis frequencies must be complex [positions, pairs] tensors")
        if len({value.device for value in axes}) != 1:
            raise ConfigurationError("axis frequencies must share a device")
        dtype = torch.float64 if all(value.dtype == torch.complex128 for value in axes) else torch.float32
        rows = max(value.shape[0] for value in axes)
        pairs = sum(value.shape[1] for value in axes)
        cos = torch.full((rows, pairs), float("nan"), dtype=dtype, device=axes[0].device)
        sin = torch.full_like(cos, float("nan"))
        column = 0
        for value in axes:
            real = torch.view_as_real(value.resolve_conj()).to(dtype)
            cos[:value.shape[0], column:column + value.shape[1]] = real[..., 0]
            sin[:value.shape[0], column:column + value.shape[1]] = real[..., 1]
            column += value.shape[1]
        return cls(cos, sin, tuple(value.shape[1] for value in axes), tuple(value.shape[0] for value in axes))

    @classmethod
    def from_cos_sin(cls, cos, sin, sections=None):
        """From real tables, e.g. SGLang's dense ``freqs_cis = (cos, sin)``."""
        return cls(cos, sin, sections)

    @classmethod
    def longcat(cls, head_dim, num_frames, height, width, *, base=10000.0, device="cpu"):
        """LongCat-Video ``rope_3d`` tables, computed in float32 like the native code.

        Native: per-axis inverse frequencies and grid positions in float32, angles
        ``position * frequency`` in float32 and ``cos``/``sin`` of those float32
        angles, applied as ``x * cos + rotate_half(x) * sin`` on interleaved pairs.
        Note that LongCat's native KV cache stores pre-RoPE K; RoPE is applied when
        attending, so restored LongCat caches are written without rotation.
        """
        _int(head_dim, "head_dim", minimum=8)
        if head_dim % 8:
            raise ConfigurationError("LongCat 3D RoPE requires head_dim divisible by 8")
        for value, name in ((num_frames, "num_frames"), (height, "height"), (width, "width")):
            _int(value, name, minimum=1)
        dims = (head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6))
        tables = []
        for dim, count in zip(dims, (num_frames, height, width)):
            inverse = 1.0 / (base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
            grid = torch.from_numpy(np.linspace(0, count, count, endpoint=False, dtype=np.float32)).float()
            tables.append(torch.einsum("..., f -> ... f", grid, inverse))
        rows = max(table.shape[0] for table in tables)
        pairs = sum(table.shape[1] for table in tables)
        cos = torch.full((rows, pairs), float("nan"), dtype=torch.float32)
        sin = torch.full_like(cos, float("nan"))
        column = 0
        for table in tables:
            cos[:table.shape[0], column:column + table.shape[1]] = table.cos()
            sin[:table.shape[0], column:column + table.shape[1]] = table.sin()
            column += table.shape[1]
        return cls(
            cos.to(device), sin.to(device), tuple(table.shape[1] for table in tables),
            tuple(table.shape[0] for table in tables),
        )

    def _stamp(self):
        """State the cached copies depend on: version counters of normal tensors (in-place updates,
        ``set_``, ``resize_``), or only the storages of inference tensors, which track no version."""
        if self._inference:
            return (self.cos.data_ptr(), self.sin.data_ptr())
        return (self.cos._version, self.sin._version)

    def invalidate(self):
        """Drop the cached device copies (call after updating inference-tensor tables in place).

        Copies used by reconstructions in flight stay alive with their handles.
        """
        self._device_cache.clear()

    def kernel_tables(self, device, stream=None):
        """Contiguous float32 ``(cos, sin)`` on ``device``, valid for work enqueued next on ``stream``.

        One copy per device and table state is cached on this table and
        shared by every stream. On CUDA a host table is copied now into pinned
        memory and uploaded on a private stream; a device table that needs a
        conversion (float64, non-contiguous) is converted on ``stream`` (default:
        current) after the work already enqueued there, i.e. after the calling
        reconstruct()'s ``wait_events``. The copy's completion event is stored
        with it, and every consumer stream -- including other streams reusing
        the cached copy -- is made to wait on it until it has completed, so no
        launch can read cos/sin before they are written. A device table that
        is already contiguous float32 is used in place (no event). Copies are
        normal tensors made outside inference mode, so a cache filled inside
        ``torch.inference_mode`` serves calls outside it and vice versa; an
        inference-tensor table is treated as immutable (see :meth:`invalidate`).
        Callers that launch their own kernels must keep the returned tensors alive.
        """
        stamp = self._stamp()
        entry = self._device_cache.get(device)  # keyed by normalized torch.device; reconstruct() passes one
        if entry is None or entry.stamp != stamp:
            device = torch.device(device)
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            entry = self._device_cache.get(device)
            if entry is None or entry.stamp != stamp:
                entry = _KernelTables.make(self.cos, self.sin, device, stream, stamp)
                self._device_cache[device] = entry  # published only after its event was recorded
        if entry.event is not None:
            entry.order_before(device, stream)
        return entry.cos, entry.sin


@dataclass(frozen=True, eq=False)
class RopePositions:
    """Original positions of one chunk in a :class:`RopeTable`.

    Separable tables take ``rows=(t_row, h_row, w_row)``: the chunk's first
    frame uses temporal row ``t_row``. Dense tables take ``rows=first_row``.
    Per-model offsets (CausVid/LongLive start frames, RAVEN frame shifts,
    LongLive 2.0 multi-shot offsets, segment-local LongCat positions) are the
    caller's mapping into these rows.
    """

    table: RopeTable
    rows: Any = 0

    def __post_init__(self):
        if not isinstance(self.table, RopeTable):
            raise ConfigurationError("RoPE positions require a RopeTable")
        if self.table.sections is None:
            _int(self.rows, "dense RoPE row")
        else:
            rows = self.rows
            if type(rows) is int:
                rows = (rows, 0, 0)
            rows = tuple(rows)
            if len(rows) != 3:
                raise ConfigurationError("separable RoPE positions are (t_row, h_row, w_row)")
            for value in rows:
                _int(value, "RoPE row")
            object.__setattr__(self, "rows", rows)


@dataclass(frozen=True, eq=False)
class CoefficientSource:
    """Coefficients ``[N, >= rank]`` already unpacked from their pictures.

    Float32/float16/bfloat16 values are used as is. ``uint8`` values are the
    quantized pixels (e.g. ``NVDECDecoder.decode_pixels`` device output) and
    require the per-column FP32 affine ``scale``/``offset`` as host arrays.
    ``bound`` optionally bounds ``|values|`` for the FP16 modes' exact
    power-of-two prescale; without it the bound is reduced on the device.
    """

    values: torch.Tensor
    scale: Any = None
    offset: Any = None
    bound: float | None = None

    def __post_init__(self):
        values = _require_tensor(self.values, "coefficients")
        if values.ndim != 2:
            raise ConfigurationError("coefficients must be a [tokens, rank] matrix")
        if self.bound is not None and (
            isinstance(self.bound, bool) or not isinstance(self.bound, (int, float))
            or not math.isfinite(self.bound) or self.bound < 0
        ):
            raise ConfigurationError("bound must be a finite nonnegative upper bound on |coefficients|")
        if values.dtype == torch.uint8:
            if self.scale is None or self.offset is None:
                raise ConfigurationError("quantized coefficients require their affine scale and offset")
            object.__setattr__(self, "scale", _host_vector(self.scale, values.shape[1], "scale"))
            object.__setattr__(self, "offset", _host_vector(self.offset, values.shape[1], "offset"))
            if np.any(self.scale < 0):
                raise ConfigurationError("affine scales must be nonnegative")
        elif values.dtype in COEFFICIENT_DTYPES:
            if self.scale is not None or self.offset is not None:
                raise ConfigurationError("dequantized coefficients take no affine")
        else:
            raise ConfigurationError("coefficients must be uint8 pixels or float32/float16/bfloat16 values")

    @property
    def quantized(self):
        return self.values.dtype == torch.uint8


@dataclass(frozen=True, eq=False)
class PictureSource:
    """Quantized coefficients read in place from decoded pictures.

    ``pictures[segment][frame]`` is a uint8 view ``[planes, rows, columns]``
    whose origin is this matrix's rectangle; rows/columns must cover the
    matrix's ``grid`` of ``height x width`` cells. All views of one matrix
    must share their plane and row strides and have unit column stride.
    ``layout`` is this matrix's ``reelcache.channel_tiles.v1`` layout and
    ``scale``/``offset`` its expanded FP32 affine (see :func:`expand_affine`).
    """

    pictures: Any
    layout: Mapping
    scale: Any
    offset: Any

    def __post_init__(self):
        layout = self.layout
        required = ("frames", "height", "width", "rank", "planes", "grid", "channels_per_segment", "segments")
        if not isinstance(layout, Mapping) or any(name not in layout for name in required):
            raise ConfigurationError("picture layout must be a reelcache.channel_tiles.v1 layout")
        frames, height, width, rank = (layout[name] for name in ("frames", "height", "width", "rank"))
        planes, slots, segments = layout["planes"], layout["channels_per_segment"], layout["segments"]
        for value, name in (
            (frames, "frames"), (height, "height"), (width, "width"), (rank, "rank"),
            (planes, "planes"), (slots, "channels_per_segment"), (segments, "segments"),
        ):
            _int(value, f"layout {name}", minimum=1)
        columns, rows = layout["grid"]
        _int(columns, "grid columns", minimum=1)
        _int(rows, "grid rows", minimum=1)
        if slots != columns * rows * planes or segments != math.ceil(rank / slots):
            raise ConfigurationError("picture layout slot/segment counts are inconsistent")
        pictures = tuple(tuple(segment) for segment in self.pictures)
        if len(pictures) != segments or any(len(segment) != frames for segment in pictures):
            raise ConfigurationError("pictures must be indexed [segment][frame] for every segment and frame")
        strides = None
        device = None
        for segment in pictures:
            for picture in segment:
                _require_tensor(picture, "picture")
                if picture.dtype != torch.uint8 or picture.ndim != 3:
                    raise ConfigurationError("pictures must be uint8 [planes, rows, columns] views")
                if (
                    picture.shape[0] < planes or picture.shape[1] < rows * height
                    or picture.shape[2] < columns * width
                ):
                    raise ConfigurationError("picture view does not cover the matrix rectangle")
                if picture.stride(2) != 1:
                    raise ConfigurationError("picture rows must be contiguous (unit column stride)")
                current = (picture.stride(0), picture.stride(1))
                if strides is None:
                    strides, device = current, picture.device
                elif current != strides or picture.device != device:
                    raise ConfigurationError("pictures of one matrix must share plane/row strides and device")
        object.__setattr__(self, "pictures", pictures)
        object.__setattr__(self, "scale", _host_vector(self.scale, rank, "scale"))
        object.__setattr__(self, "offset", _host_vector(self.offset, rank, "offset"))
        if np.any(self.scale < 0):
            raise ConfigurationError("affine scales must be nonnegative")

    @property
    def device(self):
        return self.pictures[0][0].device

    @property
    def rank(self):
        return self.layout["rank"]

    @property
    def plane_stride(self):
        return self.pictures[0][0].stride(0)

    @property
    def row_stride(self):
        return self.pictures[0][0].stride(1)

    @classmethod
    def from_canvases(cls, canvases, layout, picture_indices, origins, scale, offset):
        """Views on decoded canvases ``[pictures, planes, height, width]``.

        ``picture_indices[segment][frame]`` selects the canvas picture and
        ``origins`` is one ``(y, x)`` rectangle origin for the matrix or a
        ``[segment][frame]`` table of origins (schema v2 tile map).
        """
        canvases = _require_tensor(canvases, "canvases")
        if canvases.ndim != 4 or canvases.dtype != torch.uint8:
            raise ConfigurationError("canvases must be uint8 [pictures, planes, height, width]")
        single = len(origins) == 2 and all(type(value) is int for value in origins)
        pictures = []
        for segment_index, segment in enumerate(picture_indices):
            row = []
            for frame_index, picture in enumerate(segment):
                y, x = origins if single else origins[segment_index][frame_index]
                _int(picture, "picture index")
                _int(y, "rectangle y")
                _int(x, "rectangle x")
                if picture >= canvases.shape[0]:
                    raise ConfigurationError("picture index is outside the canvases")
                row.append(canvases[picture, :, y:, x:])
            pictures.append(tuple(row))
        return cls(tuple(pictures), layout, scale, offset)


@dataclass(frozen=True, eq=False)
class DenseSource:
    """An uncompressed canonical ``[N, C]`` matrix (RAW archive, exact path)."""

    values: torch.Tensor

    def __post_init__(self):
        values = _require_tensor(self.values, "dense values")
        if values.ndim != 2 or values.dtype not in COEFFICIENT_DTYPES:
            raise ConfigurationError("dense values must be a float32/float16/bfloat16 [tokens, channels] matrix")


@dataclass(frozen=True, eq=False)
class ReconstructionJob:
    """Reconstruct one restored matrix into one native destination slot.

    ``grid=(frames, height, width)`` is the chunk's token geometry
    (``N = frames * height * width``). Factorized sources need ``basis``
    (canonical FP16 rows ``[R, C]``) and use the prefix view ``basis[:rank]``
    (``rank`` defaults to the source's coefficient columns). ``rope=None``
    writes without rotation (V, or K of a native cache that stores pre-RoPE K
    such as LongCat's).
    """

    source: Any
    destination: NativeView
    grid: tuple
    basis: torch.Tensor | None = None
    rank: int | None = None
    rope: RopePositions | None = None
    label: str | None = None

    def __post_init__(self):
        if not isinstance(self.source, (CoefficientSource, PictureSource, DenseSource)):
            raise ConfigurationError("source must be a CoefficientSource, PictureSource or DenseSource")
        if not isinstance(self.destination, NativeView):
            raise ConfigurationError("destination must be a NativeView")
        grid = tuple(self.grid)
        if len(grid) != 3:
            raise ConfigurationError("grid must be (frames, height, width)")
        for value in grid:
            _int(value, "grid size", minimum=1)
        object.__setattr__(self, "grid", grid)
        tokens = self.tokens
        channels = self.destination.channels
        if self.destination.token_offset + tokens > self.destination.capacity:
            raise ConfigurationError("destination slot exceeds the native cache capacity")
        if isinstance(self.source, DenseSource):
            if self.basis is not None or self.rank is not None:
                raise ConfigurationError("dense sources take no basis or rank")
            if tuple(self.source.values.shape) != (tokens, channels):
                raise ConfigurationError("dense values must be [tokens, heads * head_dim]")
        else:
            basis = _require_tensor(self.basis, "basis")
            if basis.ndim != 2 or basis.shape[1] != channels:
                raise ConfigurationError("basis must be [rows, heads * head_dim] of the destination")
            if basis.dtype not in (torch.float16, torch.float32):
                raise ConfigurationError("basis must be the canonical float16 rows (float32 for reference use)")
            if isinstance(self.source, PictureSource):
                layout = self.source.layout
                if (layout["frames"], layout["height"], layout["width"]) != grid:
                    raise ConfigurationError("picture layout geometry differs from the job grid")
                columns = layout["rank"]
            else:
                if self.source.values.shape[0] != tokens:
                    raise ConfigurationError("coefficient rows differ from the job token count")
                columns = self.source.values.shape[1]
            rank = columns if self.rank is None else self.rank
            _int(rank, "rank", minimum=1)
            if rank > basis.shape[0]:
                raise ConfigurationError("rank view exceeds the basis rows")
            if rank > columns or (isinstance(self.source, PictureSource) and rank != columns):
                raise ConfigurationError("rank view exceeds the available coefficient columns")
            object.__setattr__(self, "rank", rank)
        if self.rope is not None:
            if not isinstance(self.rope, RopePositions):
                raise ConfigurationError("rope must be RopePositions")
            table = self.rope.table
            if table.head_dim != self.destination.head_dim:
                raise ConfigurationError("RoPE table pairs differ from head_dim // 2")
            if table.sections is None:
                if self.rope.rows + tokens > table.rows:
                    raise ConfigurationError("dense RoPE rows exceed the table")
            else:
                limits = table.section_rows
                needed = (self.rope.rows[0] + grid[0], self.rope.rows[1] + grid[1], self.rope.rows[2] + grid[2])
                if any(need > limit for need, limit, pairs in zip(needed, limits, table.sections) if pairs):
                    raise ConfigurationError("separable RoPE rows exceed the table")

    @property
    def tokens(self):
        frames, height, width = self.grid
        return frames * height * width

    @property
    def channels(self):
        return self.destination.channels

    def tensors(self):
        """Every tensor this job reads or writes (for lifetime/stream tracking)."""
        result = [self.destination.tensor]
        source = self.source
        if isinstance(source, PictureSource):
            result.extend(picture for segment in source.pictures for picture in segment)
        else:
            result.append(source.values)
        if self.basis is not None:
            result.append(self.basis)
        if self.rope is not None:
            result.extend((self.rope.table.cos, self.rope.table.sin))
        return result


# ---------------------------------------------------------------------------
# Completion handle
# ---------------------------------------------------------------------------


_DEFERRED = []
_DEFERRED_LOCK = threading.RLock()  # re-entrant: __del__ may run during a sweep


def _sweep_deferred():
    """Drop references of collected-but-unreleased handles whose GPU work has finished."""
    with _DEFERRED_LOCK:
        if _DEFERRED:
            _DEFERRED[:] = [(event, refs) for event, refs in _DEFERRED if not event.query()]


def pending_deferred():
    """Number of collected handles whose inputs are still held until their event completes."""
    with _DEFERRED_LOCK:
        return len(_DEFERRED)


class RestoreHandle:
    """Completion of one :func:`reconstruct` call.

    ``event`` is recorded on the execution stream after the last write (None
    for host execution, which completes synchronously). The handle keeps all
    inputs, tables and temporaries alive until :meth:`release`; consumers on
    another stream call :meth:`wait` before reading the destinations. A handle
    collected before release parks its references until its event completes,
    so inputs are never freed early.
    """

    def __init__(self, *, event, stream, keepalive, stats):
        self._event = event
        self._stream = stream
        self._keepalive = keepalive
        self._released = False
        self.stats = stats

    @property
    def event(self):
        return self._event

    @property
    def stream(self):
        return self._stream

    @property
    def released(self):
        return self._released

    def query(self):
        return self._event is None or self._event.query()

    def synchronize(self):
        if self._event is not None:
            self._event.synchronize()

    def wait(self, stream=None):
        """Make ``stream`` (default: current) wait for completion; no host block."""
        if self._event is None:
            return
        if stream is None:
            stream = torch.cuda.current_stream(self._stream.device)
        stream.wait_event(self._event)

    def release(self):
        """Wait for completion, then drop every retained input and temporary."""
        if self._released:
            return
        self.synchronize()
        self._keepalive = None
        self._released = True
        _sweep_deferred()

    def __del__(self):
        try:
            if not self._released and self._event is not None and self._keepalive is not None:
                if not self._event.query():
                    with _DEFERRED_LOCK:
                        _DEFERRED.append((self._event, self._keepalive))
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass


# ---------------------------------------------------------------------------
# Validation and the reference backend
# ---------------------------------------------------------------------------


def _storage_key(tensor):
    try:
        return tensor.untyped_storage().data_ptr(), tensor.device
    except (RuntimeError, NotImplementedError):
        return tensor.data_ptr(), tensor.device


def _check_disjoint(jobs):
    """Reject jobs whose slots in the same destination view overlap (writes would race).

    Views are compared by storage, offset, shape, strides, layout and batch;
    distinct views that alias the same memory are the caller's responsibility.
    """
    spans = {}
    for index, job in enumerate(jobs):
        view = job.destination
        tensor = view.tensor
        key = (*_storage_key(tensor), tensor.storage_offset(), tuple(tensor.shape), tensor.stride(),
               view.layout, view.batch_index)
        spans.setdefault(key, []).append((view.token_offset, view.token_offset + job.tokens, index))
    for intervals in spans.values():
        intervals.sort()
        for (_, stop, first), (start, _, second) in zip(intervals, intervals[1:]):
            if start < stop:
                raise ConfigurationError(f"jobs {first} and {second} write overlapping destination tokens")


def _validate(jobs):
    if isinstance(jobs, ReconstructionJob):
        jobs = (jobs,)
    if not isinstance(jobs, Sequence) or isinstance(jobs, (str, bytes)):
        raise ConfigurationError("jobs must be a sequence of ReconstructionJob")
    jobs = tuple(jobs)
    if any(not isinstance(job, ReconstructionJob) for job in jobs):
        raise ConfigurationError("jobs must be ReconstructionJob instances")
    devices = set()
    for job in jobs:
        devices.add(job.destination.tensor.device)
        source = job.source
        if isinstance(source, PictureSource):
            devices.add(source.device)
        else:
            devices.add(source.values.device)
        if job.basis is not None:
            devices.add(job.basis.device)
    if len(devices) > 1:
        raise ConfigurationError("all job tensors (except RoPE tables) must share one device")
    _check_disjoint(jobs)
    return jobs, (devices.pop() if devices else None)


def unpack_pictures(source):
    """Quantized coefficient pixels ``[N, rank]`` from a :class:`PictureSource`.

    Identical to the NVDEC device unpack (``nvdec.unpack_device_frame``) and
    the software cell mapping in ``_hevc``.
    """
    layout = source.layout
    frames, height, width, rank = layout["frames"], layout["height"], layout["width"], layout["rank"]
    columns, rows = layout["grid"]
    planes, slots = layout["planes"], layout["channels_per_segment"]
    values = torch.empty((frames, height, width, rank), dtype=torch.uint8, device=source.device)
    for segment, pictures in enumerate(source.pictures):
        first = segment * slots
        count = min(slots, rank - first)
        for frame, picture in enumerate(pictures):
            cropped = picture[:planes, :rows * height, :columns * width]
            tiled = cropped.reshape(planes, rows, height, columns, width)
            values[frame, :, :, first:first + count] = tiled.permute(2, 4, 1, 3, 0).reshape(height, width, -1)[:, :, :count]
    return values.reshape(-1, rank)


def dequantize(pixels, scale, offset):
    """``float(pixel) * scale + offset`` with the eager (mul, then add) rounding."""
    scale = torch.from_numpy(np.ascontiguousarray(scale)).to(pixels.device)
    offset = torch.from_numpy(np.ascontiguousarray(offset)).to(pixels.device)
    return pixels.float().mul_(scale).add_(offset)


def reference_coefficients(job):
    """The FP32 coefficient matrix ``[N, rank]`` a job reconstructs from."""
    source = job.source
    if isinstance(source, PictureSource):
        return dequantize(unpack_pictures(source), source.scale, source.offset)
    if isinstance(source, CoefficientSource):
        if source.quantized:
            return dequantize(source.values[:, :job.rank], source.scale[:job.rank], source.offset[:job.rank])
        return source.values[:, :job.rank].float()
    raise ConfigurationError("dense sources have no coefficients")


def rope_cos_sin(positions, grid, *, dtype=torch.float64, device=None):
    """Per-token ``(cos, sin)`` ``[N, head_dim // 2]`` for a job's positions."""
    table = positions.table
    device = table.cos.device if device is None else device
    cos = table.cos.to(device=device, dtype=dtype)
    sin = table.sin.to(device=device, dtype=dtype)
    frames, height, width = grid
    tokens = frames * height * width
    if table.sections is None:
        first = positions.rows
        return cos[first:first + tokens], sin[first:first + tokens]
    index = torch.arange(tokens, device=device)
    frame, rest = index // (height * width), index % (height * width)
    coordinates = (frame, rest // width, rest % width)
    out_cos = torch.empty((tokens, table.pairs), dtype=dtype, device=device)
    out_sin = torch.empty_like(out_cos)
    column = 0
    for axis, pairs in enumerate(table.sections):
        rows = positions.rows[axis] + coordinates[axis]
        out_cos[:, column:column + pairs] = cos[rows, column:column + pairs]
        out_sin[:, column:column + pairs] = sin[rows, column:column + pairs]
        column += pairs
    return out_cos, out_sin


def apply_rope_reference(values, positions, grid, *, dtype=torch.float64):
    """Rotate interleaved pairs of ``values[N, H, D]`` in ``dtype`` (Wan complex product)."""
    cos, sin = rope_cos_sin(positions, grid, dtype=dtype, device=values.device)
    x = values.to(dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[:, None, :], sin[:, None, :]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


def _pre_rope(job, compute):
    if isinstance(job.source, DenseSource):
        return job.source.values
    coefficients = reference_coefficients(job)
    basis = job.basis[:job.rank]
    return coefficients.to(compute) @ basis.to(device=coefficients.device, dtype=compute)


def fp32_matmul_precision(device_type):
    """Precision this process lets FP32 torch matmuls on ``"cuda"`` or ``"cpu"`` use (read only).

    ``"ieee"``, or a reduced mode: ``"tf32"``/``"bf16"`` (the ``fp32_precision``
    settings of Torch >= 2.9, which the legacy ``allow_tf32`` and
    ``set_float32_matmul_precision`` setters also update) or ``"high"``/
    ``"medium"`` (``torch.get_float32_matmul_precision()`` of older Torch).
    These settings are process-wide and also govern the model's own matmuls,
    so this module never changes them: the Triton backends choose their
    precision per kernel, and the two paths that run torch GEMMs (``staged``
    fp32/tf32 and the FP32 reference) check this value and raise
    :class:`UnsupportedError` when it differs from what they need.
    """
    if device_type == "cuda":
        holder = torch.backends.cuda.matmul
    else:
        holder = getattr(getattr(torch.backends, "mkldnn", None), "matmul", None)
    precision = getattr(holder, "fp32_precision", None)  # resolves the generic torch.backends.fp32_precision
    if isinstance(precision, str):
        return "ieee" if precision in ("ieee", "none") else precision
    legacy = torch.get_float32_matmul_precision()
    return "ieee" if legacy == "highest" else legacy


def _require_fp32_matmul(device_type, accepted, what):
    actual = fp32_matmul_precision(device_type)
    if actual not in accepted:
        raise UnsupportedError(
            f"{what} needs {device_type} FP32 matmuls at {' / '.join(accepted)} precision, but this process "
            f"set {actual!r}. That setting is process-wide (it also governs the model's matmuls), so "
            "restore_kernels never changes it: use a Triton backend (precision chosen per kernel) or change "
            "the setting in your own code."
        )


def _reference_job(job, compute, native_rope, native_rope_dtype):
    heads, head_dim = job.destination.heads, job.destination.head_dim
    tokens = job.tokens
    slot = job.destination.slot(tokens)
    values = _pre_rope(job, compute)
    if native_rope is not None and job.rope is not None:
        dtype = slot.dtype if native_rope_dtype is None else native_rope_dtype
        result = native_rope(values.to(dtype).reshape(tokens, heads, head_dim), job)
        if not isinstance(result, torch.Tensor) or result.numel() != tokens * heads * head_dim:
            raise IntegrityError("native RoPE callable returned a tensor of the wrong size")
        slot.copy_(result.reshape(tokens, heads, head_dim))
        return
    values = values.reshape(tokens, heads, head_dim)
    if job.rope is not None:
        values = apply_rope_reference(values, job.rope, job.grid, dtype=compute)
    slot.copy_(values)


def reconstruct_reference(jobs, *, compute_dtype=torch.float64, native_rope=None, native_rope_dtype=None):
    """Plain-torch reference: FP64 (default) or FP32 math, then one cast.

    Coefficients are dequantized exactly as ReelCache does, multiplied by the
    rank view in ``compute_dtype`` and, for jobs with RoPE, rotated in
    ``compute_dtype`` with the table's own values. Runs on the jobs' device.
    FP32 math needs the process's FP32 matmul precision to be IEEE on that
    device (:func:`fp32_matmul_precision`; never changed here), otherwise
    :class:`UnsupportedError`.
    """
    jobs, device = _validate(jobs)
    if compute_dtype not in (torch.float32, torch.float64):
        raise ConfigurationError("reference compute dtype must be float32 or float64")
    if compute_dtype == torch.float32 and device is not None:
        _require_fp32_matmul(device.type, ("ieee",), "the FP32 reference (compute_dtype=torch.float32)")
    with torch.no_grad():
        for job in jobs:
            _reference_job(job, compute_dtype, native_rope, native_rope_dtype)


_UNIT_ROUNDOFF = {torch.bfloat16: 2.0 ** -8, torch.float16: 2.0 ** -11, torch.float32: 2.0 ** -24}
# Smallest-magnitude rounding floor of each output format (half its subnormal spacing).
_OUTPUT_FLOOR = {torch.bfloat16: 2.0 ** -134, torch.float16: 2.0 ** -25, torch.float32: 2.0 ** -150}


def _gemm_gamma(precision, rank):
    """Relative GEMM error per unit of sum_k |a_k| |q_k| for each precision mode."""
    accumulate = (rank + 1) * 2.0 ** -24
    return {
        "fp32": accumulate,
        "tf32": 2.0 ** -10 + accumulate,
        "tf32x3": 2.0 ** -20 + 3 * accumulate,
        "fp16": 2.0 ** -11 + accumulate,
        "fp16x2": 2.0 ** -21 + 2 * accumulate,
        "bf16": 2.0 ** -7 + 2.0 ** -16 + accumulate,
        "reference32": accumulate,
    }[precision]


def reference_values(job):
    """FP64 values ``y*[N, C]`` a job writes before the final cast, and ``sum_k |p_k| |q_k|``."""
    tokens, heads, head_dim = job.tokens, job.destination.heads, job.destination.head_dim
    if isinstance(job.source, DenseSource):
        values = job.source.values.double()
        magnitude = torch.zeros_like(values)
    else:
        coefficients = reference_coefficients(job).double()
        basis = job.basis[:job.rank].to(device=coefficients.device, dtype=torch.float64)
        values = coefficients @ basis
        magnitude = coefficients.abs() @ basis.abs()
    if job.rope is not None:
        values = apply_rope_reference(values.reshape(tokens, heads, head_dim), job.rope, job.grid).reshape(tokens, -1)
        pairs = magnitude.reshape(tokens, -1, 2)
        magnitude = (pairs[..., 0] + pairs[..., 1]).repeat_interleave(2, dim=-1)
    return values, magnitude


def error_bound(job, precision, out_dtype=None):
    """Elementwise bound on |written - y*| for a precision mode (first-order analysis).

    ``|y_hat - y*| <= u_out |y*| + (1 + u_out) E + floor`` with
    ``E = gamma(precision, rank) * S + 2**-21 (|y_e| + |y_o|)`` where ``S`` is
    ``sum_k |p_k| |q_k|`` (summed over each rotated pair for K), ``gamma``
    covers operand rounding (TF32/FP16/BF16; the FP16 basis is exact in
    FP16/TF32) plus FP32 accumulation, and the last term covers FP32 RoPE
    arithmetic and FP32 tables. Returns ``(y*, bound)`` in FP64.
    """
    values, magnitude = reference_values(job)
    out_dtype = job.destination.tensor.dtype if out_dtype is None else out_dtype
    unit = _UNIT_ROUNDOFF[out_dtype]
    rank = job.rank or 0
    error = _gemm_gamma(precision, rank) * magnitude
    if job.rope is not None:
        tokens = job.tokens
        pairs = values.reshape(tokens, -1, 2).abs()
        error = error + 2.0 ** -21 * (pairs[..., 0] + pairs[..., 1]).repeat_interleave(2, dim=-1)
    return values, unit * values.abs() + (1 + unit) * error + _OUTPUT_FLOOR[out_dtype]


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:
    _K_DESC = tl.constexpr(_DESC)
    _K_TOKENS = tl.constexpr(_F["tokens"])
    _K_RANK = tl.constexpr(_F["rank"])
    _K_FRAMES = tl.constexpr(_F["frames"])
    _K_GRID_H = tl.constexpr(_F["grid_h"])
    _K_GRID_W = tl.constexpr(_F["grid_w"])
    _K_SRC_PTR = tl.constexpr(_F["src_ptr"])
    _K_SRC_RS = tl.constexpr(_F["src_row_stride"])
    _K_SRC_CS = tl.constexpr(_F["src_col_stride"])
    _K_SCALE_PTR = tl.constexpr(_F["scale_ptr"])
    _K_OFFSET_PTR = tl.constexpr(_F["offset_ptr"])
    _K_BOUND_PTR = tl.constexpr(_F["bound_ptr"])
    _K_BASIS_PTR = tl.constexpr(_F["basis_ptr"])
    _K_BASIS_RS = tl.constexpr(_F["basis_row_stride"])
    _K_DST_PTR = tl.constexpr(_F["dst_ptr"])
    _K_DST_TS = tl.constexpr(_F["dst_token_stride"])
    _K_DST_HS = tl.constexpr(_F["dst_head_stride"])
    _K_DST_DS = tl.constexpr(_F["dst_dim_stride"])
    _K_ROPE = tl.constexpr(_F["rope"])
    _K_COS_PTR = tl.constexpr(_F["cos_ptr"])
    _K_SIN_PTR = tl.constexpr(_F["sin_ptr"])
    _K_ROPE_RS = tl.constexpr(_F["rope_row_stride"])
    _K_ROW_T = tl.constexpr(_F["row_t"])
    _K_ROW_H = tl.constexpr(_F["row_h"])
    _K_ROW_W = tl.constexpr(_F["row_w"])
    _K_SEC_T = tl.constexpr(_F["sec_t"])
    _K_SEC_H = tl.constexpr(_F["sec_h"])
    _K_ROPE_DENSE = tl.constexpr(_F["rope_dense"])
    _K_PIC_SLOTS = tl.constexpr(_F["pic_slots"])
    _K_PIC_PLANES = tl.constexpr(_F["pic_planes"])
    _K_PIC_GCOLS = tl.constexpr(_F["pic_grid_cols"])
    _K_PIC_PS = tl.constexpr(_F["pic_plane_stride"])
    _K_PIC_RS = tl.constexpr(_F["pic_row_stride"])
    _K_PIC_TABLE = tl.constexpr(_F["pic_table"])
    _K_PIC_BASE = tl.constexpr(_F["pic_base"])
    _K_PIC_FS = tl.constexpr(_F["pic_frame_stride"])
    _K_WORK_PTR = tl.constexpr(_F["work_ptr"])
    _K_WORK_RS = tl.constexpr(_F["work_row_stride"])
    _K_NEEDS_BOUND = tl.constexpr(_F["needs_bound"])

    @triton.jit
    def _i32(desc, field):
        return tl.load(desc + field).to(tl.int32)

    @triton.jit
    def _token_coordinates(desc, offs_m):
        """(frame, row, column) of each token of the tile (int32)."""
        grid_h = _i32(desc, _K_GRID_H)
        grid_w = _i32(desc, _K_GRID_W)
        plane = grid_h * grid_w
        frame = offs_m // plane
        rest = offs_m - frame * plane
        row = rest // grid_w
        return frame, row, rest - row * grid_w

    @triton.jit
    def _source_setup(desc, offs_m, frame, trow, tcol, SOURCE: tl.constexpr, SRC_DTYPE: tl.constexpr,
                      PIC_AFFINE: tl.constexpr, SRC_ALIGN: tl.constexpr):
        """Loop-invariant addressing of a job's coefficient source (int32 in-job offsets)."""
        scale_ptr = tl.load(desc + _K_SCALE_PTR).to(tl.pointer_type(tl.float32))
        offset_ptr = tl.load(desc + _K_OFFSET_PTR).to(tl.pointer_type(tl.float32))
        table = tl.load(desc + _K_PIC_TABLE).to(tl.pointer_type(tl.int32))
        if SOURCE == 2:
            src = tl.load(desc + _K_PIC_BASE).to(tl.pointer_type(tl.uint8))
            row_stride = _i32(desc, _K_PIC_RS)
            rows = trow * row_stride + tcol
            if PIC_AFFINE:
                rows = rows + frame * _i32(desc, _K_PIC_FS)
            col_stride = _i32(desc, _K_GRID_W)
            frames = _i32(desc, _K_FRAMES)
            slots = _i32(desc, _K_PIC_SLOTS)
            planes = _i32(desc, _K_PIC_PLANES)
            grid_cols = _i32(desc, _K_PIC_GCOLS)
            plane_stride = _i32(desc, _K_PIC_PS)
            cell_stride = _i32(desc, _K_GRID_H) * row_stride
        else:
            address = tl.load(desc + _K_SRC_PTR)
            row_stride = tl.load(desc + _K_SRC_RS).to(tl.int32)
            if SOURCE == 0:
                src = address.to(tl.pointer_type(SRC_DTYPE))
            else:
                src = address.to(tl.pointer_type(tl.uint8))
            if SRC_ALIGN > 0:
                src = tl.multiple_of(src, 16)
                row_stride = tl.multiple_of(row_stride, SRC_ALIGN)
            rows = offs_m * row_stride
            col_stride = _i32(desc, _K_SRC_CS)
            frames = col_stride
            slots = col_stride
            planes = col_stride
            grid_cols = col_stride
            plane_stride = col_stride
            cell_stride = col_stride
        return src, rows, col_stride, table, frames, slots, planes, grid_cols, plane_stride, cell_stride, scale_ptr, offset_ptr

    @triton.jit
    def _load_coefficients(src, rows, col_stride, table, frames, slots, planes, grid_cols, plane_stride,
                           cell_stride, scale_ptr, offset_ptr, frame, kk, mask, k_mask,
                           SOURCE: tl.constexpr, PIC_AFFINE: tl.constexpr, UNIT_COLUMNS: tl.constexpr,
                           PIC_T: tl.constexpr = False, RAW: tl.constexpr = False):
        """Dequantized FP32 coefficient tile ``[BLOCK_M, BLOCK_K]`` (0 outside the rank).

        ``PIC_T`` gathers pictures as ``[BLOCK_K, BLOCK_M]`` so neighbouring
        threads read neighbouring tokens (bytes of one picture row).
        """
        if SOURCE == 2:
            segment = kk // slots
            cell = kk - segment * slots
            plane = cell % planes
            cell = cell // planes
            cell_row = cell // grid_cols
            cell_col = cell - cell_row * grid_cols
            k_offset = plane * plane_stride + cell_row * cell_stride + cell_col * col_stride
            if PIC_T:
                mask_t = tl.trans(mask)
                if PIC_AFFINE:
                    k_offset = k_offset + tl.load(table + segment * frames, mask=k_mask, other=0)
                    offset = k_offset[:, None] + rows[None, :]
                else:
                    picture = tl.load(table + segment[:, None] * frames + frame[None, :], mask=mask_t, other=0)
                    offset = picture + k_offset[:, None] + rows[None, :]
                pixels = tl.trans(tl.load(src + offset, mask=mask_t, other=0))
            else:
                if PIC_AFFINE:
                    k_offset = k_offset + tl.load(table + segment * frames, mask=k_mask, other=0)
                    offset = rows[:, None] + k_offset[None, :]
                else:
                    picture = tl.load(table + segment[None, :] * frames + frame[:, None], mask=mask, other=0)
                    offset = picture + rows[:, None] + k_offset[None, :]
                pixels = tl.load(src + offset, mask=mask, other=0)
        else:
            if UNIT_COLUMNS:
                offset = rows[:, None] + kk[None, :]
            else:
                offset = rows[:, None] + kk[None, :] * col_stride
            pixels = tl.load(src + offset, mask=mask, other=0)
        if RAW:
            a = pixels
        elif SOURCE == 0:
            a = pixels.to(tl.float32)
        else:
            scale = tl.load(scale_ptr + kk, mask=k_mask, other=0.0)
            shift = tl.load(offset_ptr + kk, mask=k_mask, other=0.0)
            a = pixels.to(tl.float32) * scale[None, :] + shift[None, :]
        return a

    @triton.jit
    def _pow2(exponent):
        """Exact float32 2**exponent for an integer-valued exponent in [-126, 127]."""
        return ((exponent.to(tl.int32) + 127) << 23).to(tl.float32, bitcast=True)

    @triton.jit
    def _prescale(desc, PRECISION: tl.constexpr):
        """Exact powers of two (scale, inverse) bringing max|coefficient| into [2**13, 2**14]."""
        if PRECISION == 3 or PRECISION == 4:
            bound = tl.load(tl.load(desc + _K_BOUND_PTR).to(tl.pointer_type(tl.float32)))
            exponent = tl.where(bound > 0, 14.0 - tl.math.ceil(tl.math.log2(tl.maximum(bound, 1e-30))), 0.0)
            exponent = tl.minimum(tl.maximum(exponent, -100.0), 100.0)
            scale = _pow2(exponent)
            inverse = _pow2(-exponent)
        else:
            scale = 1.0
            inverse = 1.0
        return scale, inverse

    @triton.jit
    def _epilogue(desc, acc, offs_m, m_mask, frame, trow, tcol, n0, channels,
                  HEAD_DIM: tl.constexpr, OUT_DTYPE: tl.constexpr, ANY_ROPE: tl.constexpr,
                  DST_ALIGNED: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        """RoPE on interleaved pairs (K), cast and store of one ``[BLOCK_M, BLOCK_N]`` tile."""
        HALF: tl.constexpr = BLOCK_N // 2
        if ANY_ROPE:
            if tl.load(desc + _K_ROPE) != 0:
                even, odd = tl.split(tl.reshape(acc, [BLOCK_M, HALF, 2]))
                pair = ((n0 + 2 * tl.arange(0, HALF)) % HEAD_DIM) // 2
                row_stride = _i32(desc, _K_ROPE_RS)
                if tl.load(desc + _K_ROPE_DENSE) != 0:
                    rows = (_i32(desc, _K_ROW_T) + offs_m)[:, None] + pair[None, :] * 0
                else:
                    sec_t = _i32(desc, _K_SEC_T)
                    sec_h = _i32(desc, _K_SEC_H)
                    rows_t = _i32(desc, _K_ROW_T) + frame
                    rows_h = _i32(desc, _K_ROW_H) + trow
                    rows_w = _i32(desc, _K_ROW_W) + tcol
                    rows = tl.where(
                        pair[None, :] < sec_t, rows_t[:, None],
                        tl.where(pair[None, :] < sec_t + sec_h, rows_h[:, None], rows_w[:, None]),
                    )
                index = rows * row_stride + pair[None, :]
                rmask = m_mask[:, None] & ((n0 + 2 * tl.arange(0, HALF)) < channels)[None, :]
                cos = tl.load(tl.load(desc + _K_COS_PTR).to(tl.pointer_type(tl.float32)) + index, mask=rmask, other=1.0)
                sin = tl.load(tl.load(desc + _K_SIN_PTR).to(tl.pointer_type(tl.float32)) + index, mask=rmask, other=0.0)
                out = tl.reshape(tl.join(even * cos - odd * sin, even * sin + odd * cos), [BLOCK_M, BLOCK_N])
            else:
                out = acc
        else:
            out = acc
        cols = n0 + tl.arange(0, BLOCK_N)
        dst = tl.load(desc + _K_DST_PTR).to(tl.pointer_type(OUT_DTYPE))
        token_stride = tl.load(desc + _K_DST_TS).to(tl.int32)
        head_stride = tl.load(desc + _K_DST_HS).to(tl.int32)
        if DST_ALIGNED:
            dst = tl.multiple_of(dst, 16)
            token_stride = tl.multiple_of(token_stride, 8)
            head_stride = tl.multiple_of(head_stride, 8)
            column = (cols // HEAD_DIM) * head_stride + cols % HEAD_DIM
        else:
            column = (cols // HEAD_DIM) * head_stride + (cols % HEAD_DIM) * _i32(desc, _K_DST_DS)
        tl.store(dst + (offs_m[:, None] * token_stride + column[None, :]), out.to(OUT_DTYPE),
                 mask=m_mask[:, None] & (cols < channels)[None, :])

    @triton.jit
    def _fused_restore_kernel(
        desc_ptr, tile_ptr, n_tiles, channels,
        HEAD_DIM: tl.constexpr, SOURCE: tl.constexpr, SRC_DTYPE: tl.constexpr, OUT_DTYPE: tl.constexpr,
        PRECISION: tl.constexpr, ANY_ROPE: tl.constexpr, PIC_AFFINE: tl.constexpr, DST_ALIGNED: tl.constexpr,
        BASIS_ALIGNED: tl.constexpr, SRC_ALIGN: tl.constexpr, UNIT_COLUMNS: tl.constexpr, N_GROUP: tl.constexpr,
        PIC_T: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Unpack + dequantize + rank-view GEMM + RoPE + cast + native store.

        Each program owns ``BLOCK_M`` tokens of one job and ``N_GROUP``
        consecutive ``BLOCK_N`` channel tiles (coefficient gathers of later
        tiles hit L1).
        """
        pid = tl.program_id(0)
        n_groups = tl.cdiv(n_tiles, N_GROUP)
        pid_g = pid % n_groups
        pid_t = pid // n_groups
        job = tl.load(tile_ptr + 2 * pid_t).to(tl.int64)
        m0 = tl.load(tile_ptr + 2 * pid_t + 1)
        desc = desc_ptr + job * _K_DESC
        tokens = _i32(desc, _K_TOKENS)
        rank = _i32(desc, _K_RANK)
        offs_m = m0 + tl.arange(0, BLOCK_M)
        m_mask = offs_m < tokens
        frame, trow, tcol = _token_coordinates(desc, offs_m)
        basis = tl.load(desc + _K_BASIS_PTR).to(tl.pointer_type(tl.float16))
        basis_rs = tl.load(desc + _K_BASIS_RS).to(tl.int32)
        if BASIS_ALIGNED:
            basis = tl.multiple_of(basis, 16)
            basis_rs = tl.multiple_of(basis_rs, 8)
        pre, inverse = _prescale(desc, PRECISION)
        (src, rows, col_stride, table, frames, slots, planes, grid_cols, plane_stride, cell_stride,
         scale_ptr, offset_ptr) = _source_setup(desc, offs_m, frame, trow, tcol, SOURCE, SRC_DTYPE, PIC_AFFINE,
                                                SRC_ALIGN)
        for group in range(0, N_GROUP):
            pid_n = pid_g * N_GROUP + group
            if pid_n < n_tiles:
                n0 = pid_n * BLOCK_N
                cols = n0 + tl.arange(0, BLOCK_N)
                n_mask = cols < channels
                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for k0 in range(0, rank, BLOCK_K):
                    kk = k0 + tl.arange(0, BLOCK_K)
                    k_mask = kk < rank
                    a = _load_coefficients(src, rows, col_stride, table, frames, slots, planes, grid_cols,
                                           plane_stride, cell_stride, scale_ptr, offset_ptr, frame, kk,
                                           m_mask[:, None] & k_mask[None, :], k_mask, SOURCE, PIC_AFFINE,
                                           UNIT_COLUMNS, PIC_T)
                    b = tl.load(basis + (kk[:, None] * basis_rs + cols[None, :]),
                                mask=k_mask[:, None] & n_mask[None, :], other=0.0)
                    if PRECISION == 0:
                        acc = tl.dot(a, b.to(tl.float32), acc, input_precision="ieee")
                    elif PRECISION == 1:
                        acc = tl.dot(a, b.to(tl.float32), acc, input_precision="tf32")
                    elif PRECISION == 2:
                        acc = tl.dot(a, b.to(tl.float32), acc, input_precision="tf32x3")
                    elif PRECISION == 3:
                        acc = tl.dot((a * pre).to(tl.float16), b, acc)
                    elif PRECISION == 4:
                        scaled = a * pre
                        hi = scaled.to(tl.float16)
                        acc = tl.dot(hi, b, acc)
                        acc = tl.dot((scaled - hi.to(tl.float32)).to(tl.float16), b, acc)
                    else:
                        acc = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), acc)
                if PRECISION == 3 or PRECISION == 4:
                    acc = acc * inverse
                _epilogue(desc, acc, offs_m, m_mask, frame, trow, tcol, n0, channels,
                          HEAD_DIM, OUT_DTYPE, ANY_ROPE, DST_ALIGNED, BLOCK_M, BLOCK_N)

    @triton.jit
    def _unpack_kernel(desc_ptr, tile_ptr, k_tiles,
                       SOURCE: tl.constexpr, SRC_DTYPE: tl.constexpr, WORK_DTYPE: tl.constexpr,
                       PRESCALE: tl.constexpr, PIC_AFFINE: tl.constexpr, SRC_ALIGN: tl.constexpr,
                       UNIT_COLUMNS: tl.constexpr, PIC_T: tl.constexpr, RAW: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
        """Step 1 of two: coefficients into a dense ``[N, rank]`` work matrix.

        ``RAW`` copies the quantized pixels (uint8 work, dequantized later in
        the fused kernel); otherwise the work holds dequantized coefficients.
        """
        pid = tl.program_id(0)
        pid_k = pid % k_tiles
        pid_t = pid // k_tiles
        job = tl.load(tile_ptr + 2 * pid_t).to(tl.int64)
        m0 = tl.load(tile_ptr + 2 * pid_t + 1)
        desc = desc_ptr + job * _K_DESC
        tokens = _i32(desc, _K_TOKENS)
        rank = _i32(desc, _K_RANK)
        if pid_k * BLOCK_K < rank:
            offs_m = m0 + tl.arange(0, BLOCK_M)
            m_mask = offs_m < tokens
            kk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = kk < rank
            frame, trow, tcol = _token_coordinates(desc, offs_m)
            (src, rows, col_stride, table, frames, slots, planes, grid_cols, plane_stride, cell_stride,
             scale_ptr, offset_ptr) = _source_setup(desc, offs_m, frame, trow, tcol, SOURCE, SRC_DTYPE, PIC_AFFINE,
                                                    SRC_ALIGN)
            mask = m_mask[:, None] & k_mask[None, :]
            a = _load_coefficients(src, rows, col_stride, table, frames, slots, planes, grid_cols, plane_stride,
                                   cell_stride, scale_ptr, offset_ptr, frame, kk, mask, k_mask, SOURCE, PIC_AFFINE,
                                   UNIT_COLUMNS, PIC_T, RAW)
            if PRESCALE:
                pre, inverse = _prescale(desc, 3)
                a = a * pre
            work = tl.load(desc + _K_WORK_PTR).to(tl.pointer_type(WORK_DTYPE))
            work_rs = _i32(desc, _K_WORK_RS)
            tl.store(work + (offs_m[:, None] * work_rs + kk[None, :]), a.to(WORK_DTYPE), mask=mask)

    @triton.jit
    def _rope_store_kernel(desc_ptr, tile_ptr, n_tiles, channels,
                           HEAD_DIM: tl.constexpr, WORK_DTYPE: tl.constexpr, OUT_DTYPE: tl.constexpr,
                           PRESCALE: tl.constexpr, ANY_ROPE: tl.constexpr, DST_ALIGNED: tl.constexpr,
                           WORK_ALIGNED: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        """Staged step 3 (and dense sources): RoPE + cast + native store of ``[N, C]`` rows."""
        pid = tl.program_id(0)
        pid_n = pid % n_tiles
        pid_t = pid // n_tiles
        job = tl.load(tile_ptr + 2 * pid_t).to(tl.int64)
        m0 = tl.load(tile_ptr + 2 * pid_t + 1)
        desc = desc_ptr + job * _K_DESC
        tokens = _i32(desc, _K_TOKENS)
        offs_m = m0 + tl.arange(0, BLOCK_M)
        m_mask = offs_m < tokens
        frame, trow, tcol = _token_coordinates(desc, offs_m)
        n0 = pid_n * BLOCK_N
        cols = n0 + tl.arange(0, BLOCK_N)
        work = tl.load(desc + _K_WORK_PTR).to(tl.pointer_type(WORK_DTYPE))
        work_rs = tl.load(desc + _K_WORK_RS).to(tl.int32)
        if WORK_ALIGNED:
            work = tl.multiple_of(work, 16)
            work_rs = tl.multiple_of(work_rs, 8)
        acc = tl.load(work + (offs_m[:, None] * work_rs + cols[None, :]),
                      mask=m_mask[:, None] & (cols < channels)[None, :], other=0.0).to(tl.float32)
        if PRESCALE:
            pre, inverse = _prescale(desc, 3)
            acc = acc * inverse
        _epilogue(desc, acc, offs_m, m_mask, frame, trow, tcol, n0, channels,
                  HEAD_DIM, OUT_DTYPE, ANY_ROPE, DST_ALIGNED, BLOCK_M, BLOCK_N)

    @triton.jit
    def _bound_kernel(desc_ptr, tile_ptr, k_tiles, bounds_ptr,
                      SRC_DTYPE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
        """max|coefficient| per job for float sources (FP16 prescale), via integer atomics."""
        pid = tl.program_id(0)
        pid_k = pid % k_tiles
        pid_t = pid // k_tiles
        job = tl.load(tile_ptr + 2 * pid_t).to(tl.int64)
        m0 = tl.load(tile_ptr + 2 * pid_t + 1)
        desc = desc_ptr + job * _K_DESC
        tokens = _i32(desc, _K_TOKENS)
        rank = _i32(desc, _K_RANK) * _i32(desc, _K_NEEDS_BOUND)
        offs_m = m0 + tl.arange(0, BLOCK_M)
        kk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = (offs_m < tokens)[:, None] & (kk < rank)[None, :]
        src = tl.load(desc + _K_SRC_PTR).to(tl.pointer_type(SRC_DTYPE))
        offset = offs_m[:, None] * _i32(desc, _K_SRC_RS) + kk[None, :] * _i32(desc, _K_SRC_CS)
        a = tl.abs(tl.load(src + offset, mask=mask, other=0.0).to(tl.float32))
        top = tl.max(tl.max(a, axis=1), axis=0)
        tl.atomic_max(bounds_ptr + job, top.to(tl.int32, bitcast=True))


_KERNEL_NAMES = ("_fused_restore_kernel", "_unpack_kernel", "_rope_store_kernel", "_bound_kernel")


def _interpreted():
    """True when Triton was imported with ``TRITON_INTERPRET=1`` (CPU interpreter)."""
    return TRITON_AVAILABLE and type(_fused_restore_kernel).__name__ == "InterpretedFunction"


def _compiled_unavailable():
    """Why the compiled (GPU) Triton kernels cannot run in this process, or None when they can."""
    if not TRITON_AVAILABLE:
        return f"Triton could not be imported ({TRITON_IMPORT_ERROR})"
    if _interpreted():
        return "Triton runs in interpreter mode (TRITON_INTERPRET=1 was set before it was imported)"
    return None


def _resolve_backend(backend, cuda):
    """``(backend, interpret)`` for jobs on CUDA (``cuda=True``) or host tensors; never a silent fallback.

    ``auto`` is the optimized Triton path on CUDA and the reference on host
    tensors. When the compiled kernels are unavailable, CUDA calls raise
    :class:`UnsupportedError` (``auto`` like ``fused``) instead of running the
    FP64 reference, which runs only for ``backend="reference"`` or host tensors.
    """
    if backend == "reference" or (backend == "auto" and not cuda):
        return "reference", False
    interpret = backend == "interpret" or (backend in ("fused", "two_pass") and not cuda and _interpreted())
    if interpret:
        if cuda:
            raise UnsupportedError("the Triton interpreter backend runs on CPU tensors only")
        if not TRITON_AVAILABLE:
            raise UnsupportedError(f"Triton could not be imported ({TRITON_IMPORT_ERROR}); "
                                   "the interpret backend is unavailable")
        if not _interpreted():
            raise UnsupportedError("the interpret backend needs TRITON_INTERPRET=1 set before Triton is imported")
        return ("fused" if backend == "interpret" else backend), True
    if not cuda:
        raise UnsupportedError(f"the {backend} backend needs CUDA tensors; use 'reference' or 'interpret'")
    reason = _compiled_unavailable()
    if reason is not None:
        if backend == "auto":
            raise UnsupportedError(
                f"backend='auto' cannot run the optimized CUDA reconstruction: {reason}. The FP64 reference "
                "is never substituted silently; pass backend='reference' to request it explicitly."
            )
        raise UnsupportedError(f"the {backend} backend is unavailable: {reason}")
    return ("fused" if backend == "auto" else backend), False


def _kernels(mode):
    """The Triton kernels for ``mode`` ("compiled" GPU kernels or the CPU "interpret"er).

    The Triton interpreter is process-wide: it must be selected with
    ``TRITON_INTERPRET=1`` before Triton is imported, and then replaces every
    compiled kernel. Tests run it in a separate process.
    """
    if not TRITON_AVAILABLE:
        raise UnsupportedError(f"Triton could not be imported ({TRITON_IMPORT_ERROR}); "
                               "only the reference backend is available")
    if mode == "interpret" and not _interpreted():
        raise UnsupportedError("the interpret backend needs TRITON_INTERPRET=1 set before Triton is imported")
    if mode == "compiled" and _interpreted():
        raise UnsupportedError("Triton runs in interpreter mode (TRITON_INTERPRET=1); compiled kernels are unavailable")
    if mode == "interpret":
        _interpreter_numpy_shim()
    return {name: globals()[name] for name in _KERNEL_NAMES}


def _interpreter_numpy_shim():
    """Let the Triton 3.4 interpreter index with loaded scalars under NumPy >= 2.4.

    The interpreter keeps scalar loads as 1-element arrays and converts them
    with ``int(array)``, which NumPy 2.4 rejects. Interpreter-only (tests).
    """
    import inspect

    from triton.runtime import interpreter
    original = interpreter._patch_lang_tensor
    if getattr(original, "_reelcache_shim", False):
        return
    try:
        if "squeeze" in inspect.getsource(original):  # fixed upstream (Triton >= 3.5)
            return
    except (OSError, TypeError):
        pass

    def patched(tensor, *args, **kwargs):
        result = original(tensor, *args, **kwargs)
        tensor.__index__ = lambda self: int(np.asarray(self.handle.data).reshape(-1)[0])
        return result

    patched._reelcache_shim = True
    interpreter._patch_lang_tensor = patched


# ---------------------------------------------------------------------------
# Launch planning
# ---------------------------------------------------------------------------


_CAPABILITIES = {}


def _capability(device):
    if device not in _CAPABILITIES:
        _CAPABILITIES[device] = torch.cuda.get_device_capability(device)
    return _CAPABILITIES[device]


def device_config(device):
    """Tile configuration for a CUDA device (by compute capability)."""
    device = torch.device(device)
    if device.type != "cuda":
        return dict(_DEFAULT_CONFIG)
    return dict(_ARCH_CONFIGS.get(_capability(device), _DEFAULT_CONFIG))


def recommended_precision(device):
    """Measured recommendation per GPU architecture (see the benchmark README)."""
    device = torch.device(device)
    if device.type != "cuda":
        return _DEFAULT_PRECISION
    return _RECOMMENDED_PRECISION.get(_capability(device), _DEFAULT_PRECISION)


def _triton_dtype(dtype):
    return {torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16,
            torch.uint8: tl.uint8}[dtype]




class _Arena:
    """Small per-launch tables packed into one pinned buffer and one H2D copy."""

    def __init__(self):
        self._parts = []
        self._size = 0

    def add(self, array):
        array = np.ascontiguousarray(array)
        offset = (self._size + 15) // 16 * 16
        self._parts.append((offset, array))
        self._size = offset + array.nbytes
        return offset

    def materialize(self, device, stream, finalize):
        """Allocate, let ``finalize(base_address)`` patch arrays, then copy once."""
        size = max(16, (self._size + 15) // 16 * 16)
        cuda = device.type == "cuda"
        host = torch.empty(size, dtype=torch.uint8, pin_memory=cuda)
        target = torch.empty(size, dtype=torch.uint8, device=device) if cuda else host
        finalize(target.data_ptr())
        buffer = host.numpy()
        for offset, array in self._parts:
            buffer[offset:offset + array.nbytes] = array.reshape(-1).view(np.uint8)
        if cuda:
            target.copy_(host, non_blocking=True)  # reconstruct() runs under the execution stream
        return host, target


def _tiles(jobs, block_m):
    """(job, first token) of every BLOCK_M token tile, as int32 [tiles, 2]."""
    counts = np.fromiter(((job.tokens + block_m - 1) // block_m for job in jobs), dtype=np.int64, count=len(jobs))
    total = int(counts.sum())
    first = np.repeat(np.cumsum(counts) - counts, counts)
    tiles = np.empty((total, 2), dtype=np.int32)
    tiles[:, 0] = np.repeat(np.arange(len(jobs), dtype=np.int32), counts)
    tiles[:, 1] = (np.arange(total, dtype=np.int64) - first) * block_m
    return tiles


def _coefficient_bound(source, rank):
    """Upper bound of |pixel * scale + offset| over 8-bit pixels (selects a power-of-two prescale)."""
    scale, offset = source.scale[:rank], source.offset[:rank]
    return float(max(np.abs(offset).max(), np.abs(np.float32(255) * scale + offset).max())) * 1.0001


def _source_kind(job):
    source = job.source
    if isinstance(source, PictureSource):
        return 2, torch.uint8
    if isinstance(source, CoefficientSource):
        return (1, torch.uint8) if source.quantized else (0, source.values.dtype)
    return 3, source.values.dtype


def _address(tensor):
    return tensor.data_ptr()


def _destination_address(view):
    tensor = view.tensor
    return tensor.data_ptr() + (view.element_offset() - tensor.storage_offset()) * tensor.element_size()


_INT32_MAX = 2 ** 31 - 1


def _span(sizes_strides):
    """Largest element offset reachable from a base through (size, stride) pairs."""
    return sum((size - 1) * abs(stride) for size, stride in sizes_strides if size > 0)


def _check_int32(value, what):
    if value > _INT32_MAX:
        raise UnsupportedError(f"{what} exceeds the kernels' int32 in-job offsets; split the job")


def _picture_addressing(source):
    """(base address, int32 [segment, frame] byte offsets, affine frame stride or None)."""
    pointers = [[picture.data_ptr() for picture in segment] for segment in source.pictures]
    base = min(min(row) for row in pointers)
    offsets = [[pointer - base for pointer in row] for row in pointers]
    layout = source.layout
    extent = ((layout["planes"] - 1) * source.plane_stride
              + (layout["grid"][1] * layout["height"] - 1) * source.row_stride
              + layout["grid"][0] * layout["width"])
    _check_int32(max(max(row) for row in offsets) + extent, "picture addressing span")
    frames = len(offsets[0])
    stride = 0 if frames == 1 else offsets[0][1] - offsets[0][0]
    if any(row[frame] - row[0] != frame * stride for row in offsets for frame in range(frames)):
        stride = None
    elif stride:
        _check_int32(abs(stride) * frames, "picture frame stride")
    return base, np.asarray(offsets, dtype=np.int32), stride


class _Plan:
    """Descriptors and side tables for one homogeneous launch group."""

    def __init__(self, jobs, device, stream, *, need_bounds=False, need_prescale=True, work=None, work_itemsize=4):
        self.jobs = jobs
        self.device = device
        self.keepalive = []
        last_table = cos = sin = None
        arena = _Arena()
        desc = np.zeros((len(jobs), _DESC), dtype=np.int64)
        bounds = np.zeros(len(jobs), dtype=np.float32)
        patches = []
        self.bound_jobs = []
        self.pic_affine = True
        self.dst_aligned = True
        self.basis_aligned = True
        self.src_aligned = True
        self.work_aligned = True
        self.unit_columns = True
        self.src_elements = 0
        for index, job in enumerate(jobs):
            row = desc[index]
            row[_F["tokens"]] = job.tokens
            row[_F["frames"]], row[_F["grid_h"]], row[_F["grid_w"]] = job.grid
            view = job.destination
            address = _destination_address(view)
            strides = view.strides()
            row[_F["dst_ptr"]] = address
            row[_F["dst_token_stride"]], row[_F["dst_head_stride"]], row[_F["dst_dim_stride"]] = strides
            _check_int32(_span(zip((job.tokens, view.heads, view.head_dim), strides)), "destination slot span")
            self.dst_aligned &= (address % 16 == 0 and strides[2] == 1 and strides[0] % 8 == 0 and strides[1] % 8 == 0)
            source = job.source
            if isinstance(source, DenseSource):
                values = source.values
                if values.stride(1) != 1:
                    raise ConfigurationError("dense GPU sources need unit channel stride")
                row[_F["work_ptr"]], row[_F["work_row_stride"]] = _address(values), values.stride(0)
                _check_int32(_span(zip(values.shape, values.stride())), "dense source span")
                self.work_aligned &= _address(values) % 16 == 0 and values.stride(0) * values.element_size() % 16 == 0
            else:
                rank = job.rank
                row[_F["rank"]] = rank
                basis = job.basis
                if basis.dtype != torch.float16 or basis.stride(1) != 1:
                    raise ConfigurationError("GPU kernels need the canonical float16 basis with unit column stride")
                row[_F["basis_ptr"]], row[_F["basis_row_stride"]] = _address(basis), basis.stride(0)
                _check_int32(_span(((rank, basis.stride(0)), (basis.shape[1], 1))), "basis span")
                self.basis_aligned &= _address(basis) % 16 == 0 and basis.stride(0) % 8 == 0
                if isinstance(source, PictureSource):
                    layout = source.layout
                    base, offsets, stride = _picture_addressing(source)
                    row[_F["pic_base"]] = base
                    patches.append((index, "pic_table", arena.add(offsets)))
                    row[_F["pic_slots"]] = layout["channels_per_segment"]
                    row[_F["pic_planes"]] = layout["planes"]
                    row[_F["pic_grid_cols"]] = layout["grid"][0]
                    row[_F["pic_plane_stride"]] = source.plane_stride
                    row[_F["pic_row_stride"]] = source.row_stride
                    if stride is None:
                        self.pic_affine = False
                    else:
                        row[_F["pic_frame_stride"]] = stride
                    quantized = True
                else:
                    values = source.values
                    row[_F["src_ptr"]] = _address(values)
                    row[_F["src_row_stride"]], row[_F["src_col_stride"]] = values.stride()
                    _check_int32(_span(((job.tokens, values.stride(0)), (rank, values.stride(1)))), "coefficient span")
                    self.unit_columns &= values.stride(1) == 1
                    self.src_aligned &= (_address(values) % 16 == 0 and values.stride(1) == 1
                                         and values.stride(0) * values.element_size() % 16 == 0)
                    self.src_elements = 16 // values.element_size()
                    quantized = source.quantized
                if quantized:
                    patches.append((index, "scale_ptr", arena.add(source.scale[:rank])))
                    patches.append((index, "offset_ptr", arena.add(source.offset[:rank])))
                    if need_prescale:
                        bounds[index] = _coefficient_bound(source, rank)
                elif source.bound is not None:
                    bounds[index] = source.bound
                elif need_bounds:
                    self.bound_jobs.append(index)
                    row[_F["needs_bound"]] = 1
                if work is not None:
                    row[_F["work_ptr"]], row[_F["work_row_stride"]] = work[index]
                    self.work_aligned &= work[index][0] % 16 == 0 and work[index][1] * work_itemsize % 16 == 0
            if job.rope is not None:
                table = job.rope.table
                if table is not last_table:
                    # FP32 copies ordered before this launch on ``stream`` (shared across streams via an event).
                    last_table = table
                    cos, sin = table.kernel_tables(device, stream)
                    self.keepalive.extend((cos, sin))
                row[_F["rope"]] = 1
                row[_F["cos_ptr"]], row[_F["sin_ptr"]] = _address(cos), _address(sin)
                row[_F["rope_row_stride"]] = cos.stride(0)
                _check_int32(cos.numel(), "RoPE table")
                if table.sections is None:
                    row[_F["rope_dense"]] = 1
                    row[_F["row_t"]] = job.rope.rows
                else:
                    row[_F["row_t"]], row[_F["row_h"]], row[_F["row_w"]] = job.rope.rows
                    row[_F["sec_t"]], row[_F["sec_h"]] = table.sections[0], table.sections[1]
        self.desc = desc
        self.bounds_offset = arena.add(bounds)
        self.desc_offset = arena.add(desc)
        self._patches = patches
        self._arena = arena

    def upload(self, tiles, stream):
        tile_offset = self._arena.add(tiles)

        def finalize(base):
            for index, name, offset in self._patches:
                self.desc[index, _F[name]] = base + offset
            self.desc[:, _F["bound_ptr"]] = base + self.bounds_offset + 4 * np.arange(len(self.jobs))

        host, buffer = self._arena.materialize(self.device, stream, finalize)
        self.keepalive.extend((host, buffer))
        desc = buffer[self.desc_offset:self.desc_offset + self.desc.nbytes].view(torch.int64)
        tile_view = buffer[tile_offset:tile_offset + tiles.nbytes].view(torch.int32)
        bounds = buffer[self.bounds_offset:self.bounds_offset + 4 * len(self.jobs)].view(torch.int32)
        return desc, tile_view, bounds


def _group(jobs, key):
    groups = {}
    for job in jobs:
        groups.setdefault(key(job), []).append(job)
    return list(groups.values())


_SM_COUNTS = {}


def _pic_t(config):
    """Gather orientation for pictures: explicit config, else the measured default for this Triton."""
    value = config.get("PIC_T")
    return _LEGACY_TRITON if value is None else bool(value)


def _channel_group(m_tiles, n_tiles, device, config):
    """Channel tiles per program: up to ``N_GROUP``, but keep >= 4 programs per SM.

    The private key ``config["_n_group"]`` (set by :func:`warmup` only) selects one value of that range.
    """
    limit = max(1, min(int(config.get("N_GROUP", 1)), n_tiles))
    if config.get("_n_group") is not None:
        return max(1, min(int(config["_n_group"]), limit))
    if device.type != "cuda":
        return limit
    if device not in _SM_COUNTS:
        _SM_COUNTS[device] = torch.cuda.get_device_properties(device).multi_processor_count
    target = 4 * _SM_COUNTS[device]
    return max(1, min(limit, (m_tiles * n_tiles) // target))


def _launch_options(kernels_mode, config):
    if kernels_mode == "interpret":
        return {}
    options = {"num_warps": config["num_warps"], "num_stages": config["num_stages"], "enable_fp_fusion": False}
    if config.get("maxnreg"):
        options["maxnreg"] = config["maxnreg"]
    return options


def _launch_bounds(kernels, plan, desc, tiles, num_tiles, bounds, config, mode):
    """Device max|coefficient| for float sources in FP16 modes (fills the bound slots).

    Launch groups are homogeneous in source kind, so every job of the plan is
    a float source here.
    """
    if not plan.bound_jobs:
        return 0
    kind, src_dtype = _source_kind(plan.jobs[0])
    if kind != 0:
        raise StateError("bound reduction requires a group of float coefficient sources")
    block_k = config["BLOCK_K"]
    k_tiles = max(math.ceil(job.rank / block_k) for job in plan.jobs)
    grid = (num_tiles * k_tiles,)
    kernels["_bound_kernel"][grid](
        desc, tiles, k_tiles, bounds, SRC_DTYPE=_triton_dtype(src_dtype),
        BLOCK_M=config["BLOCK_M"], BLOCK_K=block_k, **_launch_options(mode, {**config, "num_stages": 1}),
    )
    return 1


def _fused_group(kernels, jobs, device, precision, config, stream, mode, keepalive, stage_events=None):
    kind, src_dtype = _source_kind(jobs[0])
    code = _PRECISION_CODES[precision]
    need_bounds = code in (3, 4) and kind == 0
    plan = _Plan(jobs, device, stream, need_bounds=need_bounds, need_prescale=code in (3, 4))
    tiles_host = _tiles(jobs, config["BLOCK_M"])
    desc, tiles, bounds = plan.upload(tiles_host, stream)
    keepalive.extend(plan.keepalive)
    if stage_events is not None and mode != "interpret":
        stage_events.append(("start", _record(stream)))
    launches = _launch_bounds(kernels, plan, desc, tiles, tiles_host.shape[0], bounds, config, mode) if need_bounds else 0
    channels, head_dim = jobs[0].channels, jobs[0].destination.head_dim
    n_tiles = math.ceil(channels / config["BLOCK_N"])
    grid = (tiles_host.shape[0] * n_tiles,)
    n_group = _channel_group(tiles_host.shape[0], n_tiles, device, config)
    grid = (tiles_host.shape[0] * math.ceil(n_tiles / n_group),)
    interpret = mode == "interpret"
    options = _launch_options(mode, config)
    if kind == 0 and code in (0, 2, 3, 4) and "num_stages" in options:
        # Float tiles reformatted in registers before the dot (FP32 FMA, 3xTF32, FP16 prescale/split):
        # pipelining stages them through shared memory and ptxas spills (FP32 on Triton 3.4: ~11 KB
        # per thread on SM80 and SM120). One stage measured 1.0-3.5x faster on RTX PRO 6000 (FP32 on
        # Triton 3.4: ~79x); TF32 and BF16 keep the configured stages.
        options["num_stages"] = 1
    kernels["_fused_restore_kernel"][grid](
        desc, tiles, n_tiles, channels,
        HEAD_DIM=head_dim, SOURCE=kind, SRC_DTYPE=_triton_dtype(src_dtype),
        OUT_DTYPE=_triton_dtype(jobs[0].destination.tensor.dtype), PRECISION=code,
        ANY_ROPE=any(job.rope is not None for job in jobs), PIC_AFFINE=plan.pic_affine,
        DST_ALIGNED=plan.dst_aligned and not interpret, BASIS_ALIGNED=plan.basis_aligned and not interpret,
        SRC_ALIGN=plan.src_elements if plan.src_aligned and not interpret else 0, UNIT_COLUMNS=plan.unit_columns,
        N_GROUP=n_group, PIC_T=_pic_t(config),
        BLOCK_M=config["BLOCK_M"], BLOCK_N=config["BLOCK_N"], BLOCK_K=config["BLOCK_K"],
        **options,
    )
    if stage_events is not None and mode != "interpret":
        stage_events.append(("fused", _record(stream)))
    return launches + 1


def _two_pass_group(kernels, jobs, device, precision, config, stream, mode, keepalive, stage_events=None):
    """Batched unpack into a dense ``[N, rank]`` uint8 pixel matrix, then the fused kernel on it.

    The first pass replaces per-frame picture unpacking (one gather per pixel);
    the second is the fused dequantize + GEMM + RoPE + store on aligned rows.
    Both kernels share one descriptor upload: the unpack reads the picture
    fields and writes the work matrix, the fused pass reads it as its source.
    """
    kind, src_dtype = _source_kind(jobs[0])
    if kind != 2:
        return _fused_group(kernels, jobs, device, precision, config, stream, mode, keepalive, stage_events)
    code = _PRECISION_CODES[precision]
    strides = [(job.rank + 15) // 16 * 16 for job in jobs]
    total = sum(job.tokens * stride for job, stride in zip(jobs, strides))
    buffer = torch.empty(total, dtype=torch.uint8, device=device)
    keepalive.append(buffer)
    work, start, base = {}, 0, buffer.data_ptr()
    for index, (job, stride) in enumerate(zip(jobs, strides)):
        work[index] = (base + start, stride)
        start += job.tokens * stride
    plan = _Plan(jobs, device, stream, work=work, work_itemsize=1, need_prescale=code in (3, 4))
    for index, (address, stride) in work.items():
        row = plan.desc[index]
        row[_F["src_ptr"]], row[_F["src_row_stride"]], row[_F["src_col_stride"]] = address, stride, 1
    block_m = config["BLOCK_M"]
    tiles_host = _tiles(jobs, block_m)
    desc, tiles, _ = plan.upload(tiles_host, stream)
    keepalive.extend(plan.keepalive)
    interpret = mode == "interpret"
    block_k = config.get("UNPACK_BLOCK_K", 64)
    k_tiles = max(math.ceil(job.rank / block_k) for job in jobs)
    if stage_events is not None and not interpret:
        stage_events.append(("start", _record(stream)))
    kernels["_unpack_kernel"][(tiles_host.shape[0] * k_tiles,)](
        desc, tiles, k_tiles, SOURCE=2, SRC_DTYPE=tl.uint8, WORK_DTYPE=tl.uint8,
        PRESCALE=False, PIC_AFFINE=plan.pic_affine, SRC_ALIGN=0, UNIT_COLUMNS=True,
        PIC_T=_pic_t(config), RAW=True, BLOCK_M=block_m, BLOCK_K=block_k,
        **_launch_options(mode, {**config, "num_warps": 4, "num_stages": 1, "maxnreg": None}),
    )
    if stage_events is not None and not interpret:
        stage_events.append(("unpack", _record(stream)))
    channels, head_dim = jobs[0].channels, jobs[0].destination.head_dim
    n_tiles = math.ceil(channels / config["BLOCK_N"])
    n_group = _channel_group(tiles_host.shape[0], n_tiles, device, config)
    kernels["_fused_restore_kernel"][(tiles_host.shape[0] * math.ceil(n_tiles / n_group),)](
        desc, tiles, n_tiles, channels,
        HEAD_DIM=head_dim, SOURCE=1, SRC_DTYPE=tl.uint8,
        OUT_DTYPE=_triton_dtype(jobs[0].destination.tensor.dtype), PRECISION=code,
        ANY_ROPE=any(job.rope is not None for job in jobs), PIC_AFFINE=True,
        DST_ALIGNED=plan.dst_aligned and not interpret, BASIS_ALIGNED=plan.basis_aligned and not interpret,
        SRC_ALIGN=0 if interpret else 16, UNIT_COLUMNS=True, N_GROUP=n_group, PIC_T=False,
        BLOCK_M=block_m, BLOCK_N=config["BLOCK_N"], BLOCK_K=config["BLOCK_K"],
        **_launch_options(mode, config),
    )
    if stage_events is not None and not interpret:
        stage_events.append(("fused", _record(stream)))
    return 2


def _rope_store_group(kernels, jobs, device, config, stream, mode, keepalive, *,
                      work=None, work_dtype=None, prescale=False, plan_bounds=None):
    """RoPE + cast + store of dense rows: dense sources, or staged GEMM outputs ``work``."""
    if work is None:
        plan = _Plan(jobs, device, stream)
        work_dtype = jobs[0].source.values.dtype
    else:
        plan = _Plan(jobs, device, stream, work=work,
                     work_itemsize=torch.empty((), dtype=work_dtype).element_size())
    tiles_host = _tiles(jobs, config["BLOCK_M"])
    desc, tiles, bounds = plan.upload(tiles_host, stream)
    keepalive.extend(plan.keepalive)
    if plan_bounds is not None:
        with torch.cuda.stream(stream) if stream is not None else _null():
            bounds.copy_(plan_bounds)
    channels, head_dim = jobs[0].channels, jobs[0].destination.head_dim
    n_tiles = math.ceil(channels / config["BLOCK_N"])
    grid = (tiles_host.shape[0] * n_tiles,)
    kernels["_rope_store_kernel"][grid](
        desc, tiles, n_tiles, channels,
        HEAD_DIM=head_dim, WORK_DTYPE=_triton_dtype(work_dtype),
        OUT_DTYPE=_triton_dtype(jobs[0].destination.tensor.dtype), PRESCALE=prescale,
        ANY_ROPE=any(job.rope is not None for job in jobs), DST_ALIGNED=plan.dst_aligned and mode != "interpret",
        WORK_ALIGNED=plan.work_aligned and mode != "interpret",
        BLOCK_M=config["BLOCK_M"], BLOCK_N=config["BLOCK_N"], **_launch_options(mode, config),
    )
    return 1


def _check_staged_precision(precision):
    """The staged GEMM's precision follows process-wide torch settings: check them, never change them."""
    if precision == "fp32":
        _require_fp32_matmul("cuda", ("ieee",), "the staged fp32 GEMM (cuBLAS)")
    elif precision == "tf32":
        _require_fp32_matmul("cuda", ("tf32", "high"), "the staged tf32 GEMM (cuBLAS)")
    elif precision == "fp16" and getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False):
        raise UnsupportedError(
            "the staged fp16 GEMM needs FP32 accumulation, but this process allows FP16 accumulation in cuBLAS "
            "(torch.backends.cuda.matmul.allow_fp16_accumulation=True). That setting is process-wide, so "
            "restore_kernels never changes it: use a Triton backend or change the setting in your own code."
        )


def _staged_gemm(coefficients, basis, precision):
    """cuBLAS rank-view product with FP32 accumulation and FP32 output.

    Never changes process-wide torch settings. fp32 and tf32 use
    ``torch.mm`` on FP32 operands and need the process's FP32 matmul
    precision (:func:`fp32_matmul_precision`) to be IEEE or TF32
    respectively; otherwise :class:`UnsupportedError`. FP16/BF16 operands use
    ``torch.mm(..., out_dtype=torch.float32)``; a Torch without that overload
    raises :class:`UnsupportedError` -- there is no retry with an FP16/BF16
    output (which would round and could overflow the prescaled products) --
    and every other error (e.g. out of memory) propagates unchanged.
    """
    _check_staged_precision(precision)
    if precision in ("fp32", "tf32"):
        return torch.mm(coefficients, basis.float())
    operand = basis.to(torch.float16 if precision == "fp16" else torch.bfloat16)
    try:
        return torch.mm(coefficients, operand, out_dtype=torch.float32)
    except (TypeError, NotImplementedError) as error:
        raise UnsupportedError(
            f"the staged {precision} GEMM needs torch.mm(..., out_dtype=torch.float32) (FP32 output); "
            f"Torch {torch.__version__} rejected it: {type(error).__name__}: {error}"
        ) from error


def _staged_group(kernels, jobs, device, precision, config, stream, mode, keepalive, stage_events):
    """Triton unpack -> cuBLAS GEMM -> Triton RoPE/cast/store, with optional stage events."""
    if precision not in ("fp32", "tf32", "fp16", "bf16"):
        raise UnsupportedError("the staged cuBLAS backend supports fp32, tf32, fp16 and bf16")
    kind, src_dtype = _source_kind(jobs[0])
    work_dtype = {"fp32": torch.float32, "tf32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    prescale = precision == "fp16"
    # cuBLAS groups: jobs sharing a basis tensor and rank view stack into one GEMM.
    gemm_groups = {}
    for index, job in enumerate(jobs):
        key = (job.basis.data_ptr(), job.basis.stride(0), job.basis.shape[1], job.rank)
        gemm_groups.setdefault(key, []).append(index)
    coefficient_work = {}
    buffers = []
    with torch.cuda.stream(stream):
        for key, members in gemm_groups.items():
            rows = sum(jobs[index].tokens for index in members)
            buffer = torch.empty((rows, key[3]), dtype=work_dtype, device=device)
            buffers.append((key, members, buffer))
            start = 0
            for index in members:
                coefficient_work[index] = (buffer[start:].data_ptr(), buffer.stride(0))
                start += jobs[index].tokens
    keepalive.extend(buffer for _, _, buffer in buffers)
    plan = _Plan(jobs, device, stream, need_bounds=prescale and kind == 0, work=coefficient_work)
    tiles_host = _tiles(jobs, config["BLOCK_M"])
    desc, tiles, bounds = plan.upload(tiles_host, stream)
    keepalive.extend(plan.keepalive)
    launches = 0
    if stage_events is not None:
        stage_events.append(("start", _record(stream)))
    if plan.bound_jobs:
        launches += _launch_bounds(kernels, plan, desc, tiles, tiles_host.shape[0], bounds, config, mode)
    k_tiles = max(math.ceil(job.rank / config["BLOCK_K"]) for job in jobs)
    kernels["_unpack_kernel"][(tiles_host.shape[0] * k_tiles,)](
        desc, tiles, k_tiles, SOURCE=kind, SRC_DTYPE=_triton_dtype(src_dtype),
        WORK_DTYPE=_triton_dtype(work_dtype), PRESCALE=prescale, PIC_AFFINE=plan.pic_affine,
        SRC_ALIGN=plan.src_elements if plan.src_aligned and mode != "interpret" else 0,
        UNIT_COLUMNS=plan.unit_columns, PIC_T=_pic_t(config), RAW=False,
        BLOCK_M=config["BLOCK_M"], BLOCK_K=config["BLOCK_K"], **_launch_options(mode, config),
    )
    launches += 1
    if stage_events is not None:
        stage_events.append(("unpack", _record(stream)))
    outputs = {}
    with torch.cuda.stream(stream):
        for key, members, buffer in buffers:
            basis = jobs[members[0]].basis[:key[3]]
            product = _staged_gemm(buffer, basis, precision)
            keepalive.append(product)
            start = 0
            for index in members:
                outputs[index] = (product[start:].data_ptr(), product.stride(0))
                start += jobs[index].tokens
            launches += 1
            outputs_dtype = product.dtype
    if stage_events is not None:
        stage_events.append(("gemm", _record(stream)))
    launches += _rope_store_group(
        kernels, jobs, device, config, stream, mode, keepalive,
        work=outputs, work_dtype=outputs_dtype, prescale=prescale, plan_bounds=bounds,
    )
    if stage_events is not None:
        stage_events.append(("rope_store", _record(stream)))
    return launches


def _record(stream):
    event = torch.cuda.Event(enable_timing=True)
    event.record(stream)
    return event


def _group_key(job):
    kind, dtype = _source_kind(job)
    return (kind, dtype, job.destination.tensor.dtype, job.destination.head_dim, job.channels)


def _run_jobs(jobs, *, backend, precision, config, stream, device, mode, keepalive, stage_events,
              auto=False):
    kernels = _kernels(mode)
    launches = 0
    for group in _group(jobs, _group_key):
        kind = _source_kind(group[0])[0]
        if auto and kind == 2 and not _LEGACY_TRITON:
            launches += _two_pass_group(kernels, group, device, precision, config, stream, mode, keepalive,
                                        stage_events)
        elif kind == 3:
            launches += _rope_store_group(kernels, group, device, config, stream, mode, keepalive)
        elif backend == "staged":
            launches += _staged_group(kernels, group, device, precision, config, stream, mode,
                                      keepalive, stage_events)
        elif backend == "two_pass":
            launches += _two_pass_group(kernels, group, device, precision, config, stream, mode, keepalive,
                                        stage_events)
        else:
            launches += _fused_group(kernels, group, device, precision, config, stream, mode, keepalive,
                                     stage_events)
    return launches


def _native_rope_jobs(jobs, native_rope, native_rope_dtype, run, stream, keepalive):
    """Exact path: reconstruct pre-RoPE K, call the engine's own RoPE, copy into the slot."""
    staged = []
    prepared = []
    for job in jobs:
        heads, head_dim, tokens = job.destination.heads, job.destination.head_dim, job.tokens
        dtype = job.destination.tensor.dtype if native_rope_dtype is None else native_rope_dtype
        if isinstance(job.source, DenseSource) and job.source.values.dtype == dtype:
            prepared.append((job, job.source.values.reshape(tokens, heads, head_dim)))
            continue
        if dtype not in OUTPUT_DTYPES:
            raise ConfigurationError("native RoPE input dtype must be bfloat16, float16 or float32")
        with torch.cuda.stream(stream) if stream is not None else _null():
            temporary = torch.empty((tokens, heads, head_dim), dtype=dtype, device=job.destination.tensor.device)
        keepalive.append(temporary)
        staged.append(ReconstructionJob(
            job.source, NativeView(temporary, "THD"), job.grid, job.basis, job.rank, None, job.label,
        ))
        prepared.append((job, temporary))
    launches = run(staged) if staged else 0
    with torch.cuda.stream(stream) if stream is not None else _null():
        for job, values in prepared:
            result = native_rope(values, job)
            heads, head_dim, tokens = job.destination.heads, job.destination.head_dim, job.tokens
            if not isinstance(result, torch.Tensor) or result.numel() != tokens * heads * head_dim:
                raise IntegrityError("native RoPE callable returned a tensor of the wrong size")
            keepalive.append(result)
            job.destination.slot(tokens).copy_(result.reshape(tokens, heads, head_dim))
    return launches


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _matrix_flops(jobs):
    return sum(2 * job.tokens * job.rank * job.channels for job in jobs if job.rank)


def reconstruct(jobs, *, backend="auto", precision="auto", stream=None, wait_events=(), native_rope=None,
                native_rope_dtype=None, config=None, compute_dtype=torch.float64, stage_events=None):
    """Reconstruct restored matrices into their native cache slots.

    ``jobs``: :class:`ReconstructionJob` sequence; one call should cover all
    restored chunks of a layer (K and V) so the GPU does one batched launch per
    launch group (jobs sharing source kind, output dtype and head geometry).
    ``backend``: ``"auto"`` (CUDA: fused Triton, with picture sources on the
    batched two-pass path on Triton >= 3.5 as measured; CPU: reference; on
    CUDA it raises :class:`UnsupportedError` when the compiled Triton kernels
    are unavailable -- never a silent reference fallback),
    ``"fused"``, ``"two_pass"``, ``"staged"``, ``"reference"`` or
    ``"interpret"``. ``precision``: one of :data:`PRECISIONS` or ``"auto"``
    (measured per GPU, :func:`recommended_precision`). ``stream``: CUDA stream
    to run on (default: current); it first waits on ``wait_events`` (e.g.
    decoder completion). ``native_rope``: optional callable
    ``(pre_rope[N, H, D], job) -> post_rope`` used instead of the built-in
    rotation for jobs with RoPE (exact engine path; the input is
    ``native_rope_dtype``, default the destination dtype, and a dense source of
    that dtype is passed bitwise). ``config`` overrides tile parameters.
    ``stage_events``: optional list receiving ``(stage, event)`` CUDA timing
    events around the kernels ("start", "fused", "unpack", "gemm", "rope_store").

    Returns a :class:`RestoreHandle`; destinations are valid after its event.
    If launching fails part-way, the inputs of already enqueued work stay
    parked until that work completes, then the error is re-raised.
    """
    _sweep_deferred()
    jobs, device = _validate(jobs)
    if backend not in BACKENDS:
        raise ConfigurationError(f"backend must be one of {BACKENDS}")
    if precision != "auto" and precision not in PRECISIONS:
        raise ConfigurationError(f"precision must be 'auto' or one of {PRECISIONS}")
    if native_rope is not None and not callable(native_rope):
        raise ConfigurationError("native_rope must be callable")
    stats = {"jobs": len(jobs), "matrices": len(jobs), "flops": _matrix_flops(jobs), "launches": 0}
    if not jobs:
        stats.update(backend="none", precision=None)
        return RestoreHandle(event=None, stream=None, keepalive=None, stats=stats)
    cuda = device.type == "cuda"
    auto = backend == "auto"
    backend, interpret = _resolve_backend(backend, cuda)
    if precision == "auto":
        precision = recommended_precision(device)
    # Paths whose GEMM precision follows process-wide torch settings: refuse before any work is enqueued.
    if backend == "staged":
        if precision not in ("fp32", "tf32", "fp16", "bf16"):
            raise UnsupportedError("the staged cuBLAS backend supports fp32, tf32, fp16 and bf16")
        _check_staged_precision(precision)
    elif backend == "reference" and compute_dtype == torch.float32:
        _require_fp32_matmul(device.type, ("ieee",), "the FP32 reference (compute_dtype=torch.float32)")
    if cuda:
        stream = torch.cuda.current_stream(device) if stream is None else stream
        if not isinstance(stream, torch.cuda.Stream) or stream.device != device:
            raise ConfigurationError("stream must be a torch.cuda.Stream on the jobs' device")
    elif stream is not None or wait_events:
        raise ConfigurationError("host execution takes no CUDA stream or events")
    keepalive = [tensor for job in jobs for tensor in job.tensors()]
    stats.update(backend=backend, precision=precision if backend != "reference" else str(compute_dtype),
                 picture_path="two_pass" if auto and backend == "fused" and not _LEGACY_TRITON else backend)
    if cuda:
        for event in wait_events:
            stream.wait_event(event)
    if backend == "reference":
        with torch.cuda.stream(stream) if cuda else _null():
            reconstruct_reference(jobs, compute_dtype=compute_dtype, native_rope=native_rope,
                                  native_rope_dtype=native_rope_dtype)
    else:
        mode = "interpret" if interpret else "compiled"
        merged = {**(device_config(device) if cuda else dict(_DEFAULT_CONFIG)), **(config or {})}
        if merged["BLOCK_N"] % 2 or merged["BLOCK_N"] < 32:
            raise ConfigurationError("BLOCK_N must be even and >= 32")

        def run(selected):
            return _run_jobs(selected, backend=backend, precision=precision, config=merged, stream=stream,
                             device=device, mode=mode, keepalive=keepalive, stage_events=stage_events, auto=auto)

        exact = [job for job in jobs if native_rope is not None and job.rope is not None]
        ordinary = [job for job in jobs if not (native_rope is not None and job.rope is not None)]
        # Triton launches, allocations and copies all go to the execution stream.
        try:
            with torch.cuda.stream(stream) if cuda else _null():
                if ordinary:
                    stats["launches"] += run(ordinary)
                if exact:
                    stats["launches"] += _native_rope_jobs(exact, native_rope, native_rope_dtype, run,
                                                           stream if cuda else None, keepalive)
        except BaseException:
            if cuda:
                failed = torch.cuda.Event()
                failed.record(stream)
                with _DEFERRED_LOCK:
                    _DEFERRED.append((failed, keepalive))
            raise
    event = None
    if cuda:
        event = torch.cuda.Event()
        event.record(stream)
    return RestoreHandle(event=event, stream=stream, keepalive=keepalive, stats=stats)


# ---------------------------------------------------------------------------
# Warm-up: compile and load the kernels before the first restore
# ---------------------------------------------------------------------------

WARMUP_SOURCES = ("pictures", "uint8", "float", "dense")
WARMUP_ROPE_MODES = ("none", "fused", "native")
_WARMUP_GRID = (1, 4, 8)  # 32 tokens per synthetic job (one token tile)
_WARMUP_MAX_RANK = 65536
_WARMUP_LOCK = threading.Lock()
_WARMED = set()


@dataclass(frozen=True)
class WarmupShape:
    """One kind of launch group of the restore path (see :func:`warmup`).

    ``heads``, ``head_dim``, ``dtype`` and ``layout`` describe the native cache
    the jobs write: a view of a contiguous cache, as SGLang's KV caches are.
    ``source``: ``"pictures"`` (decoded canvases of the packed NVDEC restore),
    ``"uint8"`` or ``"float"`` coefficient matrices, or ``"dense"`` RAW rows.
    ``rope``: ``"fused"`` (the calls contain K jobs rotated by the kernel,
    usually with the V jobs of the same chunks), ``"native"`` (K rotated by the
    engine's ``native_rope`` callable from pre-RoPE K in ``native_rope_dtype``,
    default ``dtype``; V in the same calls) or ``"none"`` (no rotated job: V-only
    launch groups, LongCat's pre-RoPE cache). ``ranks=(min, max)``: rank range of
    the factorized matrices of one call. It selects the rank-dependent kernel
    specializations; the true range compiles no more variants than needed (the
    default covers every rank up to 1024; the K98/V95 profiles measured reach
    799). ``source_dtype``: dtype of float coefficients (default float32) or
    dense rows (default ``dtype``).
    """

    heads: int
    head_dim: int
    dtype: torch.dtype = torch.bfloat16
    layout: str = "BTHD"
    source: str = "pictures"
    rope: str = "fused"
    ranks: tuple = (1, 1024)
    source_dtype: torch.dtype | None = None
    native_rope_dtype: torch.dtype | None = None

    def __post_init__(self):
        _int(self.heads, "heads", minimum=1)
        _int(self.head_dim, "head_dim", minimum=1)
        if self.dtype not in OUTPUT_DTYPES:
            raise ConfigurationError(f"warm-up cache dtype must be one of {OUTPUT_DTYPES}")
        if self.layout not in LAYOUTS:
            raise ConfigurationError(f"warm-up layout must be one of {LAYOUTS}")
        if self.source not in WARMUP_SOURCES:
            raise ConfigurationError(f"warm-up source must be one of {WARMUP_SOURCES}")
        if self.rope not in WARMUP_ROPE_MODES:
            raise ConfigurationError(f"warm-up rope must be one of {WARMUP_ROPE_MODES}")
        if self.rope != "none" and self.head_dim % 2:
            raise ConfigurationError("RoPE needs an even head_dim")
        ranks = tuple(self.ranks) if isinstance(self.ranks, Sequence) and not isinstance(self.ranks, str) else ()
        if len(ranks) != 2:
            raise ConfigurationError("ranks must be (minimum, maximum)")
        low = _int(ranks[0], "minimum rank", minimum=1)
        high = _int(ranks[1], "maximum rank", minimum=low)
        if high > _WARMUP_MAX_RANK:
            raise ConfigurationError(f"warm-up ranks above {_WARMUP_MAX_RANK} are not supported")
        object.__setattr__(self, "ranks", (low, high))
        if self.source in ("float", "dense"):
            if self.source_dtype is not None and self.source_dtype not in COEFFICIENT_DTYPES:
                raise ConfigurationError(f"source_dtype must be one of {COEFFICIENT_DTYPES}")
        elif self.source_dtype is not None:
            raise ConfigurationError("source_dtype applies to float and dense sources only")
        if self.native_rope_dtype is not None:
            if self.rope != "native":
                raise ConfigurationError("native_rope_dtype applies to rope='native' only")
            if self.native_rope_dtype not in OUTPUT_DTYPES:
                raise ConfigurationError(f"native_rope_dtype must be one of {OUTPUT_DTYPES}")

    def describe(self):
        return {"heads": self.heads, "head_dim": self.head_dim, "dtype": str(self.dtype), "layout": self.layout,
                "source": self.source, "rope": self.rope, "ranks": list(self.ranks),
                "source_dtype": None if self.source_dtype is None else str(self.source_dtype),
                "native_rope_dtype": None if self.native_rope_dtype is None else str(self.native_rope_dtype)}


def _k_class(k_tiles):
    """Triton's specialization of an int argument: equal to 1 (a constexpr), divisible by 16, or neither."""
    return 1 if k_tiles == 1 else (16 if k_tiles % 16 == 0 else 0)


def _rank_recipes(ranks, block_ks, itemsize):
    """Rank tuples of synthetic launch groups covering every reachable rank-dependent specialization.

    A launch group's variant depends on ``ceil(max rank / BLOCK_K)`` (the
    ``k_tiles`` argument of the unpack and bound kernels; one class per entry of
    ``block_ks``) and, for coefficient sources of ``itemsize`` bytes, on whether
    every job's rows are 16-byte aligned (``rank * itemsize % 16 == 0``). One
    recipe per reachable combination: its largest rank sets the ``k_tiles``
    classes and a second, unaligned rank makes the group unaligned.
    """
    low, high = ranks
    recipes = {}
    unaligned = None
    for rank in range(low, high + 1):
        classes = tuple(_k_class(-(-rank // block)) for block in block_ks)
        aligned = itemsize is None or rank * itemsize % 16 == 0
        if not aligned and unaligned is None:
            unaligned = rank
        recipes.setdefault((classes, aligned), (rank,))
        if aligned and unaligned is not None:
            recipes.setdefault((classes, False), (rank, unaligned))
    return list(recipes.values())


def _warmup_path(shape, backend):
    """The group function :func:`reconstruct` uses for ``shape`` (mirrors ``_run_jobs``)."""
    if shape.source == "dense":
        return "rope_store"
    if backend == "staged":
        return "staged"
    if shape.source == "pictures" and (backend == "two_pass" or (backend == "auto" and not _LEGACY_TRITON)):
        return "two_pass"
    return "fused"


def _warmup_plan(shape, backend, precision, config, cuda):
    """(path, rank recipes, N_GROUP values) of one shape: every call-dependent specialization of its path."""
    path = _warmup_path(shape, backend)
    code = _PRECISION_CODES[precision]
    block_ks = []
    if path == "two_pass":
        block_ks.append(config.get("UNPACK_BLOCK_K", 64))
    if path == "staged":
        block_ks.append(config["BLOCK_K"])
    if shape.source == "float" and (precision == "fp16" if path == "staged" else code in (3, 4)):
        block_ks.append(config["BLOCK_K"])  # device bound reduction of float sources (FP16 prescale)
    block_ks = sorted(set(block_ks))
    itemsize = None
    if shape.source == "uint8":
        itemsize = 1
    elif shape.source == "float":
        itemsize = (shape.source_dtype or torch.float32).itemsize
    recipes = [()] if path == "rope_store" else _rank_recipes(shape.ranks, block_ks, itemsize)
    n_groups = [None]
    if cuda and path in ("fused", "two_pass"):
        n_tiles = math.ceil(shape.heads * shape.head_dim / config["BLOCK_N"])
        n_groups = list(range(1, max(1, min(int(config.get("N_GROUP", 1)), n_tiles)) + 1))
    return path, recipes, n_groups


def _warmup_destination(shape, tokens, device):
    heads, head_dim = shape.heads, shape.head_dim
    sizes = {"BTHD": (1, tokens, heads, head_dim), "THD": (tokens, heads, head_dim),
             "BHTD": (1, heads, tokens, head_dim)}[shape.layout]
    return torch.empty(sizes, dtype=shape.dtype, device=device)


def _warmup_source(shape, rank, tokens, device):
    """A synthetic source with production's data-dependent properties (contiguous rows, affine canvases)."""
    if shape.source == "dense":
        return DenseSource(torch.ones((tokens, shape.heads * shape.head_dim), dtype=shape.source_dtype or shape.dtype,
                                      device=device))
    if shape.source == "float":
        return CoefficientSource(torch.ones((tokens, rank), dtype=shape.source_dtype or torch.float32, device=device))
    scale, offset = np.full(rank, 1 / 255, dtype=np.float32), np.zeros(rank, dtype=np.float32)
    if shape.source == "uint8":
        return CoefficientSource(torch.zeros((tokens, rank), dtype=torch.uint8, device=device), scale, offset)
    from ._hevc import video_layout

    layout = video_layout(*_WARMUP_GRID, rank, "yuv444p")
    frames = layout["frames"]
    canvases = torch.zeros((layout["segments"] * frames, layout["planes"], layout["image_height"],
                            layout["image_width"]), dtype=torch.uint8, device=device)
    # Consecutive pictures per segment, one origin: the packed decoder's affine addressing.
    indices = [[segment * frames + frame for frame in range(frames)] for segment in range(layout["segments"])]
    return PictureSource.from_canvases(canvases, layout, indices, (0, 0), scale, offset)


def _warmup_jobs(shape, ranks, device):
    """K (rotated, unless ``rope="none"``) and V jobs per rank of one synthetic call (dense: no rank)."""
    ranks = ranks or (None,)
    tokens = math.prod(_WARMUP_GRID)
    targets = ("K", "V") if shape.rope != "none" else ("V",)
    destination = _warmup_destination(shape, tokens * len(targets) * len(ranks), device)
    basis = None
    if shape.source != "dense":
        basis = torch.zeros((max(ranks), shape.heads * shape.head_dim), dtype=torch.float16, device=device)
    table = None
    if shape.rope != "none":
        cos = torch.ones((tokens, shape.head_dim // 2), dtype=torch.float32, device=device)
        table = RopeTable.from_cos_sin(cos, torch.zeros_like(cos))
    jobs = []
    for rank in ranks:
        for target in targets:
            view = NativeView(destination, shape.layout, token_offset=len(jobs) * tokens)
            jobs.append(ReconstructionJob(
                _warmup_source(shape, rank, tokens, device), view, _WARMUP_GRID,
                basis=basis, rank=rank, rope=RopePositions(table, 0) if target == "K" else None,
                label=f"warmup:{target}:{rank}"))
    return jobs


def _identity_rope(values, job):
    """Stands in for the engine's ``native_rope`` callable during :func:`warmup`."""
    return values


def _warmup_precisions(precision, device):
    names = [precision] if isinstance(precision, str) else list(precision)
    if not names:
        raise ConfigurationError("warmup needs at least one precision")
    resolved = []
    for name in names:
        if name == "auto":
            name = recommended_precision(device)
        if name not in PRECISIONS:
            raise ConfigurationError(f"precision must be 'auto' or one of {PRECISIONS}")
        if name not in resolved:
            resolved.append(name)
    return resolved


def _variant_counts(device):
    """Compiled variants per kernel in Triton's in-memory cache of ``device``; None when not inspectable."""
    if device.type != "cuda" or not TRITON_AVAILABLE or _interpreted():
        return None
    counts = {}
    for name in _KERNEL_NAMES:
        caches = getattr(globals().get(name), "device_caches", None)
        if not isinstance(caches, dict):
            return None
        entry = caches.get(device.index)
        counts[name] = len(entry[0]) if entry else 0
    return counts


def warmup(device, shapes, *, backend="auto", precision="auto", config=None, stream=None):
    """Compile and load every kernel variant :func:`reconstruct` launches for ``shapes``, before the first restore.

    A cold process otherwise compiles (empty Triton cache: seconds) or loads
    (persistent cache) each kernel variant inside its first restores; call this
    once when a worker attaches. ``shapes``: :class:`WarmupShape` objects (or
    their keyword dicts) describing the launch groups of the restore path;
    ``backend``, ``precision`` (``"auto"``, one of :data:`PRECISIONS` or a
    sequence of them) and ``config`` as for :func:`reconstruct`.

    Each shape runs tiny synthetic jobs with production's data-dependent
    properties (contiguous caches, aligned canonical bases, the packed
    decoder's affine canvases) through the real launch path, once per
    call-dependent specialization: every channel-tile grouping (``N_GROUP``)
    the occupancy rule can choose, every rank class of the unpack/bound
    kernels' ``k_tiles`` argument within ``shape.ranks`` and, for coefficient
    sources, both row alignments. Outputs go to scratch buffers. With
    ``rope="native"`` an identity stands in for the engine's RoPE callable (warm
    the engine's own RoPE kernel separately).

    Never changes process-wide settings; waits only for its own launches (on
    ``stream``, default the current stream); serialized by a lock. Repeat calls
    for an already warmed (device, shape, backend, precision, config) do no
    work (``report["repeat"]``). Returns a report: wall ``seconds`` including
    GPU completion, ``calls``, ``launches``, per-shape details and, when
    Triton's in-memory cache can be inspected, compiled ``kernel_variants`` per
    kernel and ``new_variants``. On CPU tensors ``backend="auto"`` is the
    reference backend, which compiles nothing (``calls == 0``); like
    :func:`reconstruct`, CUDA without compiled Triton kernels raises
    :class:`UnsupportedError` (no silent fallback).
    """
    started = time.perf_counter()
    if isinstance(shapes, (WarmupShape, Mapping)):
        shapes = [shapes]
    shapes = [shape if isinstance(shape, WarmupShape) else WarmupShape(**shape) for shape in shapes]
    if not shapes:
        raise ConfigurationError("warmup needs at least one WarmupShape")
    if backend not in BACKENDS:
        raise ConfigurationError(f"backend must be one of {BACKENDS}")
    device = torch.device(device)
    cuda = device.type == "cuda"
    resolved, interpret = _resolve_backend(backend, cuda)  # raises exactly like reconstruct()
    if cuda and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    precisions = _warmup_precisions(precision, device)
    report = {"device": str(device), "backend": backend, "resolved_backend": resolved, "precisions": precisions,
              "triton": triton.__version__ if TRITON_AVAILABLE else None, "repeat": False, "warmed": 0,
              "skipped": 0, "calls": 0, "launches": 0, "kernel_variants": None, "new_variants": None, "shapes": []}
    if resolved == "reference":
        report.update(seconds=time.perf_counter() - started,
                      note="the reference backend (plain torch) compiles no kernels")
        return report
    merged = {**(device_config(device) if cuda else dict(_DEFAULT_CONFIG)), **(config or {})}
    config_key = tuple(sorted((name, repr(value)) for name, value in merged.items()))
    with (_WARMUP_LOCK, torch.cuda.device(device) if cuda else _null(),
          torch.cuda.stream(stream) if cuda and stream is not None else _null()):
        before = _variant_counts(device)
        for shape in shapes:
            for name in precisions:
                key = (str(device), backend, name, shape, config_key, interpret)
                path, recipes, n_groups = _warmup_plan(shape, backend, name, merged, cuda and not interpret)
                entry = {"shape": shape.describe(), "precision": name, "path": path,
                         "rank_recipes": [list(recipe) for recipe in recipes], "n_groups": n_groups,
                         "calls": 0, "launches": 0, "seconds": 0.0, "repeat": key in _WARMED}
                if entry["repeat"]:
                    report["skipped"] += 1
                    report["shapes"].append(entry)
                    continue
                shape_started = time.perf_counter()
                for recipe in recipes:
                    for n_group in n_groups:
                        call_config = dict(config or {})
                        if n_group is not None:
                            call_config["_n_group"] = n_group
                        jobs = _warmup_jobs(shape, recipe, device)
                        handle = reconstruct(jobs, backend=backend, precision=name, stream=stream,
                                             config=call_config,
                                             native_rope=_identity_rope if shape.rope == "native" else None,
                                             native_rope_dtype=shape.native_rope_dtype)
                        handle.release()
                        entry["calls"] += 1
                        entry["launches"] += handle.stats["launches"]
                entry["seconds"] = time.perf_counter() - shape_started
                _WARMED.add(key)
                report["warmed"] += 1
                report["calls"] += entry["calls"]
                report["launches"] += entry["launches"]
                report["shapes"].append(entry)
        after = _variant_counts(device)
    if after is not None:
        report["kernel_variants"] = after
        report["new_variants"] = sum(after.values()) - sum(before.values())
    report["repeat"] = report["warmed"] == 0
    report["seconds"] = time.perf_counter() - started
    return report


__all__ = [
    "BACKENDS", "LAYOUTS", "OUTPUT_DTYPES", "PRECISIONS", "TRITON_AVAILABLE", "TRITON_IMPORT_ERROR",
    "CoefficientSource", "DenseSource", "NativeView", "PictureSource", "ReconstructionJob",
    "RestoreHandle", "RopePositions", "RopeTable",
    "apply_rope_reference", "dequantize", "device_config", "error_bound", "expand_affine", "fp32_matmul_precision",
    "pending_deferred",
    "reconstruct",
    "reconstruct_reference", "recommended_precision", "reference_coefficients", "reference_values", "rope_cos_sin",
    "unpack_pictures", "wan_sections",
    "WARMUP_ROPE_MODES", "WARMUP_SOURCES", "WarmupShape", "warmup",
]
