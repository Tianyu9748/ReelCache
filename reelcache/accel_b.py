"""Accel study track B (2026-10-30): the archive fitter's GPU footprint. Every option is opt-in; with the defaults no
code path of the base tree changes (the hooks below are thread-local lookups that return the base path).

Options (``reelcache.integrations.sglang.config.ReelCacheRuntimeConfig`` fields, carried by :class:`FitterAccel` into
:class:`reelcache.archive_pipeline.ArchivePipeline`, which applies them on its fitter thread):

* **B1** ``fitter_green_context_sms=k``: the fitter's CUDA stream is the stream of a CUDA green context of ``k`` SMs
  (``torch.cuda.green_contexts``), so every kernel the fitter thread launches on it -- torch, cuBLAS through
  :mod:`reelcache.exact_blas` (torch's per-thread handle is bound to the current stream), cuSOLVER, Triton, CUDA-graph
  replays -- runs on at most ``k`` SMs. Green contexts live for the process (:func:`green_stream` never destroys one:
  the caching allocator may still hold blocks of its stream). ``fitter_green_workqueue`` (``"balanced"`` /
  ``"device_ctx"``) needs CUDA >= 13.1 (refused otherwise). Measured class (ii): cuSOLVER's QR (k >= 384 columns)
  and large batched eigensolves return different last bits inside a green context.
* **B3** ``fitter_cuda_graphs`` (+ ``fitter_graph_memory_bytes``): dev22l's CUDA graphs of the FP64 measurement passes
  (:mod:`reelcache.graph_capture`), bitwise.
* **B4** ``fused_fingerprints``: the source fingerprints' integer tile passes as one Triton kernel per matrix (exact
  int64 arithmetic: identical fingerprint strings); CPU tensors keep the base path.
* **B5** ``finalize_on_fit_device``: the pipeline finalizes on its CUDA fit device even when the sources are host
  tensors (LingBot's harness), and host sources are fingerprinted on that device (identical strings). Class (ii):
  the per-column extrema of the FP32 coefficients may differ in the last bits from the CPU's.
* **B2** ``fitter_fp64="off"`` (or ``"fp32:<pass>+<pass>"`` for single passes): FP32 replacements of the FP64 passes
  on CUDA tensors (class (ii), gated per pass on real nodes; :data:`FP32_PASSES`):

  - ``products``: the ``fp32_verified_fp64`` policy's FP64 measurement products (coefficients ``P = X B^T``, the
    reconstruction ``P B``, the condition Gram) become the IEEE-FP32 products of :mod:`reelcache.exact_blas` -- also
    independent of the global TF32 flags, the reason the policy existed (RAVEN). Narrower forms:
    ``products_verify`` (only the planner's verification projections; certification and finalization -- whose
    coefficients feed the augmentation residual and the stored pixels -- keep FP64) and ``products_recon`` (FP64
    coefficients everywhere; FP32 reconstruction ``P B`` in the fused residual kernel and an FP32 condition Gram);
    ``products_df``: the remaining FP64 coefficient GEMMs ``P = X B^T`` (``fp32_verified_fp64`` projections) as one
    Triton kernel of compensated FP32 dot products (Ogita-Rump-Oishi Dot2: error-free products by FMA and TwoSum,
    the result as if accumulated in twice the working precision, then rounded to FP32);
  - ``residual``: the direct-residual tiles (FP64 widening, scaling, squares and sums) become one fused Triton kernel
    per tile: FP32 differences and IEEE divisions by the scale, FP32 squares accumulated per element over the
    column blocks, then in FP64 (deterministic two-stage reduction);
  - ``energy``: the source-energy tiles likewise (one kernel per matrix);
  - ``span``: the ideal (span) energy ``<G^-1, M>`` (``G = Q Q^T`` of the FP16 rows, ``M = P^T P``) without the
    FP64 Gram, Cholesky and triangular solve: ``||P||^2`` (FP64-accumulated) plus the Neumann series
    ``sum_k (-1)^k tr(D^k M)`` of ``D = G - I`` with FP32 products, as many terms as the bound
    ``eta = max_i sum_j |D_ij|`` needs for a truncation below 1e-10 relative; ``eta > 0.05`` (or anything
    nonfinite) falls back to the FP64 routine for that matrix;
  - ``planner``: the planner's smaller FP64 element-wise steps -- scaled copies ``(A.double() / s).float()`` as IEEE
    FP32 divisions (bitwise the same values: double rounding of a quotient of 24-bit numbers is innocuous), and the
    share-prefix estimate, residual energy and sketch captured energies as FP32 squares accumulated in FP64.

  Quantization stays FP64 (stored pixels unchanged for unchanged coefficients).
* **B6** (tensor-core GEMMs for the IEEE-FP32 SIMT SGEMMs): not implemented -- at archival shapes plain TF32 is
  2.3-2.8x faster than the IEEE SIMT SGEMM but ~600x less accurate (relative error ~3e-4: beyond the planner's
  certified bounds and the 1e-6 gate), and an FP32-accurate split emulation (3xTF32) is 1.1-1.3x slower.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import threading
from typing import Any

import torch

from .errors import ConfigurationError

FP32_PASSES = ("products", "products_verify", "products_recon", "products_df", "residual", "energy", "span",
               "planner")
#: ``fitter_fp64="off"``: the passes that held the class-(ii) gates (decisions and ranks identical, |dE| <= 1e-6) on
#: real nodes of CausVid, RAVEN, LongLive 1.0 / 2.0 and LingBot (2026-10-30 replay gates; in RAVEN's real archive the
#: payload bytes stay identical, |dE| 2.2e-9). ``products`` is excluded: on RAVEN
#: its FP32 coefficients moved stored energies by up to 1.58e-6 (decisions still identical); ``products_verify`` +
#: ``products_recon`` keep the FP64-accurate coefficients and hold the gate (|dE| <= 1.9e-9, stored rows and pixels
#: identical). ``products_df`` (compensated FP32 coefficients) is excluded too: RAVEN's real archive moved stored
#: energies by 1.1e-6 with it (M1 job 4599445; 6.1e-7 in the replay gate); it stays selectable as ``fp32:...``.
OFF_PASSES = ("products_verify", "products_recon", "residual", "energy", "span", "planner")
GREEN_WORKQUEUES = ("balanced", "device_ctx")
#: Neumann span energy: the largest accepted ``eta`` and the truncation tolerance (relative to ``||P||^2``).
SPAN_ETA_MAX = 0.05
SPAN_TOLERANCE = 1e-10
SPAN_MAX_TERMS = 12


def parse_fitter_fp64(value: Any) -> frozenset:
    """``"on"`` (FP64, the base), ``"off"`` (:data:`OFF_PASSES` in FP32) or ``"fp32:<pass>[+<pass>...]"``."""
    if value == "on":
        return frozenset()
    if value == "off":
        return frozenset(OFF_PASSES)
    if isinstance(value, str) and value.startswith("fp32:"):
        names = [name for name in value[5:].split("+") if name]
        if names and all(name in FP32_PASSES for name in names):
            return frozenset(names)
    raise ConfigurationError(f"fitter_fp64 must be 'on', 'off' or 'fp32:<pass>[+<pass>...]' with passes from "
                             f"{FP32_PASSES}, got {value!r}")


@dataclasses.dataclass(frozen=True)
class FitterAccel:
    """Track B options of one archive pipeline (see the module docstring); all defaults = the base tree."""

    green_sms: int | None = None
    green_workqueue: str | None = None
    cuda_graphs: bool = False
    graph_memory_bytes: int | None = None
    fused_fingerprints: bool = False
    finalize_on_fit_device: bool = False
    fp32_passes: frozenset = frozenset()

    def __post_init__(self) -> None:
        if self.green_sms is not None and (type(self.green_sms) is not int or self.green_sms < 1):
            raise ConfigurationError("fitter_green_context_sms must be None or a positive SM count")
        if self.green_workqueue is not None and self.green_workqueue not in GREEN_WORKQUEUES:
            raise ConfigurationError(f"fitter_green_workqueue must be None or one of {GREEN_WORKQUEUES}")
        if self.green_workqueue is not None and self.green_sms is None:
            raise ConfigurationError("fitter_green_workqueue configures the fitter's green context: it needs "
                                     "fitter_green_context_sms")
        for name, field in (("cuda_graphs", "fitter_cuda_graphs"), ("fused_fingerprints", "fused_fingerprints"),
                            ("finalize_on_fit_device", "finalize_on_fit_device")):
            if type(getattr(self, name)) is not bool:
                raise ConfigurationError(f"{field} must be a bool")
        if self.graph_memory_bytes is not None and (type(self.graph_memory_bytes) is not int
                                                    or self.graph_memory_bytes < 1):
            raise ConfigurationError("fitter_graph_memory_bytes must be None or a positive byte count")
        passes = frozenset(self.fp32_passes)
        if not passes <= set(FP32_PASSES):
            raise ConfigurationError(f"unknown FP32 passes {sorted(passes - set(FP32_PASSES))}")
        object.__setattr__(self, "fp32_passes", passes)

    @property
    def default(self) -> bool:
        return self == FitterAccel()

    @property
    def needs_cuda(self) -> bool:
        return (self.green_sms is not None or self.cuda_graphs or self.finalize_on_fit_device
                or bool(self.fp32_passes))

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["fp32_passes"] = sorted(self.fp32_passes)
        return data

    @classmethod
    def from_config(cls, config: Any) -> FitterAccel | None:
        """The options of a runtime configuration (None at the defaults: the pipeline gets nothing new)."""
        accel = cls(green_sms=getattr(config, "fitter_green_context_sms", None),
                    green_workqueue=getattr(config, "fitter_green_workqueue", None),
                    cuda_graphs=getattr(config, "fitter_cuda_graphs", False),
                    graph_memory_bytes=getattr(config, "fitter_graph_memory_bytes", None),
                    fused_fingerprints=getattr(config, "fused_fingerprints", False),
                    finalize_on_fit_device=getattr(config, "finalize_on_fit_device", False),
                    fp32_passes=parse_fitter_fp64(getattr(config, "fitter_fp64", "on")))
        return None if accel.default else accel


# -- the fitter thread's scope ---------------------------------------------------------------------------------------
_LOCAL = threading.local()
_STATS_LOCK = threading.Lock()
STATS: dict[str, int] = {"energy_fused": 0, "residual_fused_tiles": 0, "span_neumann": 0, "span_fallbacks": 0,
                         "span_terms": 0, "products_fp32": 0, "planner_fp32": 0, "fingerprints_fused": 0,
                         "fingerprints_on_device": 0, "dot2_gemms": 0}


def _count(name: str, value: int = 1) -> None:
    with _STATS_LOCK:
        STATS[name] = STATS.get(name, 0) + value


def stats() -> dict[str, int]:
    with _STATS_LOCK:
        return dict(STATS)


@contextlib.contextmanager
def scope(accel: FitterAccel | None, *, fit_device: Any = None):
    """Apply ``accel`` to the archival routines this thread runs inside the block (None: the base paths)."""
    previous = (getattr(_LOCAL, "accel", None), getattr(_LOCAL, "fit_device", None))
    _LOCAL.accel = accel
    _LOCAL.fit_device = None if fit_device is None else torch.device(fit_device)
    try:
        yield
    finally:
        _LOCAL.accel, _LOCAL.fit_device = previous


def current() -> FitterAccel | None:
    return getattr(_LOCAL, "accel", None)


def fp32(name: str, tensor: torch.Tensor | None = None) -> bool:
    """Whether FP32 pass ``name`` applies to ``tensor`` (a CUDA tensor) on this thread."""
    accel = getattr(_LOCAL, "accel", None)
    if accel is None or name not in accel.fp32_passes:
        return False
    return tensor is None or (isinstance(tensor, torch.Tensor) and tensor.is_cuda)


def products(value: str, tensor: torch.Tensor) -> str:
    """The measurement products of a projection: ``"fp32"`` for ``"fp64"`` under the ``products`` pass (CUDA)."""
    if value == "fp64" and fp32("products", tensor):
        _count("products_fp32")
        return "fp32"
    return value


def verify_products(value: str, tensor: torch.Tensor) -> str:
    """The products of a planner verification: FP32 under ``products`` or ``products_verify`` (CUDA tensors)."""
    if value == "fp64" and (fp32("products", tensor) or fp32("products_verify", tensor)):
        _count("products_fp32")
        return "fp32"
    return value


def condition_products(value: str, tensor: torch.Tensor) -> str:
    """The condition Gram's products: :func:`products`, and FP32 under ``products_recon`` too (a verification under
    ``products_verify`` arrives with FP32 products already)."""
    if value == "fp64" and fp32("products_recon", tensor):
        return "fp32"
    return products(value, tensor)


def recon_fp32(value: str, tensor: torch.Tensor) -> bool:
    """``products_recon``: FP64 coefficients, FP32 reconstruction (the fused residual kernel) of a CUDA tensor."""
    return value == "fp64" and fp32("products_recon", tensor) and fused_ok(tensor)


def dot2_coefficients(value: str, tensor: torch.Tensor) -> bool:
    """``products_df``: an FP64-products projection's coefficients by the compensated Dot2 kernel (CUDA tensors)."""
    return value == "fp64" and fp32("products_df", tensor) and fused_ok(tensor)


DOT2_BM, DOT2_BN = 64, 64


def dot2_gemm(tensor: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``P = X R^T`` (FP32 ``[N, r]``) of a 2-D CUDA matrix ``X`` ``[N, C]`` and rows ``R`` ``[r, C]`` with compensated
    dot products (Dot2: as accurate as twice-FP32 accumulation, rounded once to FP32), deterministic."""
    kernels = _kernels()
    xt = tensor.T.to(torch.float32).contiguous()
    bt = rows.T.to(torch.float32).contiguous()
    n, r, c = int(tensor.shape[0]), int(rows.shape[0]), int(tensor.shape[1])
    out = torch.empty((n, r), dtype=torch.float32, device=tensor.device)
    with torch.cuda.device(tensor.device):
        kernels["dot2_gemm"][(-(-n // DOT2_BM), -(-r // DOT2_BN))](
            xt, bt, out, n, r, c, xt.stride(0), bt.stride(0), out.stride(0), BM=DOT2_BM, BN=DOT2_BN,
            num_warps=8, enable_fp_fusion=False)
    _count("dot2_gemms")
    return out


def fused_fingerprints(tensor: torch.Tensor) -> bool:
    accel = getattr(_LOCAL, "accel", None)
    return accel is not None and accel.fused_fingerprints and tensor.is_cuda and triton_available()


def fingerprint_source(tensor: torch.Tensor) -> torch.Tensor:
    """B5: a host source fingerprinted on this thread's fit device (an asynchronous copy on the current stream; the
    fingerprint is exact integer arithmetic, so the string is the host's)."""
    accel = getattr(_LOCAL, "accel", None)
    device = getattr(_LOCAL, "fit_device", None)
    if (accel is None or not accel.finalize_on_fit_device or device is None or device.type != "cuda"
            or tensor.device.type != "cpu"):
        return tensor
    _count("fingerprints_on_device")
    return tensor.to(device, non_blocking=tensor.is_pinned())


# -- B1: green contexts ----------------------------------------------------------------------------------------------
_GREEN: dict = {}
_GREEN_LOCK = threading.Lock()


def green_stream(device: Any, sms: int, workqueue: str | None = None):
    """The stream of this process's green context of ``sms`` SMs on ``device`` (created once, never destroyed)."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ConfigurationError("a green context needs a CUDA device")
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, int(sms), workqueue)
    with _GREEN_LOCK:
        entry = _GREEN.get(key)
        if entry is None:
            try:
                from torch.cuda import green_contexts
            except ImportError as exc:  # older torch
                raise ConfigurationError(f"torch {torch.__version__} has no torch.cuda.green_contexts") from exc
            if not getattr(green_contexts, "SUPPORTED", False):
                raise ConfigurationError(f"torch {torch.__version__} was built without CUDA green contexts")
            kwargs: dict[str, Any] = {"num_sms": int(sms), "device_id": index}
            if workqueue is not None:
                kwargs["workqueue_scope"] = workqueue
            try:
                with torch.cuda.device(index):
                    torch.cuda.current_stream(index).query()  # the primary context is current on this thread
                    context = green_contexts.GreenContext.create(**kwargs)
                    stream = context.Stream()
            except Exception as exc:  # noqa: BLE001 - surfaced as a configuration error
                raise ConfigurationError(f"a green context of {sms} SMs (workqueue {workqueue}) is unsupported "
                                         f"here: {type(exc).__name__}: {exc}") from exc
            entry = _GREEN[key] = (context, stream)
    return entry[1]


def green_contexts_alive() -> int:
    with _GREEN_LOCK:
        return len(_GREEN)


# -- Triton kernels (B2, B4) -------------------------------------------------------------------------------------------
_TRITON: dict = {}
_TRITON_LOCK = threading.Lock()
SQ_BLOCK_R, SQ_BLOCK_C = 8, 128
FP_BLOCK_R, FP_BLOCK_C = 8, 128


def triton_available() -> bool:
    return _kernels() is not None


def _kernels():
    if "kernels" in _TRITON:
        return _TRITON["kernels"]
    with _TRITON_LOCK:
        if "kernels" in _TRITON:
            return _TRITON["kernels"]
        try:
            import triton
            import triton.language as tl
        except Exception:  # noqa: BLE001 - no Triton: the base paths
            _TRITON["kernels"] = None
            return None

        @triton.jit
        def sq_partials(x_ptr, r_ptr, out_ptr, rows, cols, sx0, sx1, sr0, sr1, scale,
                        HAS_R: tl.constexpr, DIV: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
            # per program (BLOCK_R rows): sum (x/s)^2, sum ((x-r)/s)^2 and a nonfinite flag, in FP64
            pid = tl.program_id(0)
            offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
            rmask = offs_r < rows
            row_x = offs_r.to(tl.int64)[:, None] * sx0
            row_r = offs_r.to(tl.int64)[:, None] * sr0
            acc_x = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32)
            acc_d = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32)
            bad = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.int32)
            for c0 in range(0, cols, BLOCK_C):
                offs_c = c0 + tl.arange(0, BLOCK_C)
                mask = rmask[:, None] & (offs_c[None, :] < cols)
                x = tl.load(x_ptr + row_x + offs_c.to(tl.int64)[None, :] * sx1, mask=mask, other=0.0).to(tl.float32)
                bad = bad | tl.where(tl.abs(x) <= 3.4028234663852886e38, 0, 1)  # NaN and +-inf
                if DIV:
                    xs = tl.math.div_rn(x, scale)
                else:
                    xs = x
                acc_x += xs * xs
                if HAS_R:
                    r = tl.load(r_ptr + row_r + offs_c.to(tl.int64)[None, :] * sr1, mask=mask, other=0.0)
                    d = x - r.to(tl.float32)
                    if DIV:
                        d = tl.math.div_rn(d, scale)
                    acc_d += d * d
            sx = tl.sum(tl.sum(acc_x.to(tl.float64), axis=1), axis=0)
            sd = tl.sum(tl.sum(acc_d.to(tl.float64), axis=1), axis=0)
            nb = tl.max(tl.max(bad, axis=1), axis=0)
            tl.store(out_ptr + pid * 3, sx)
            tl.store(out_ptr + pid * 3 + 1, sd)
            tl.store(out_ptr + pid * 3 + 2, nb.to(tl.float64))

        @triton.jit
        def fingerprint_partials(x_ptr, out_ptr, rows, cols, row_start, s0, s1, prime,
                                 WIDE: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
            # lineage_commit._fingerprint_terms of BLOCK_R rows: per part (16-bit, or low/high of 32-bit) and
            # combination (a, b, square) in ((37, 101, 0), (613, 211, 0), (1, 17, 1)): sum over rows of
            # ((sum_c term * w_a(c)) mod p) * w_b(row); exact int64 (|row sums| < 2^54, partials < 2^63)
            pid = tl.program_id(0)
            offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
            rmask = offs_r < rows
            base = offs_r.to(tl.int64)[:, None] * s0
            a0 = tl.zeros([BLOCK_R], dtype=tl.int64)
            a1 = tl.zeros([BLOCK_R], dtype=tl.int64)
            a2 = tl.zeros([BLOCK_R], dtype=tl.int64)
            h0 = tl.zeros([BLOCK_R], dtype=tl.int64)
            h1 = tl.zeros([BLOCK_R], dtype=tl.int64)
            h2 = tl.zeros([BLOCK_R], dtype=tl.int64)
            for c0 in range(0, cols, BLOCK_C):
                offs_c = c0 + tl.arange(0, BLOCK_C)
                cmask = offs_c < cols
                mask = rmask[:, None] & cmask[None, :]
                bits = tl.load(x_ptr + base + offs_c.to(tl.int64)[None, :] * s1, mask=mask, other=0).to(tl.int64)
                column = offs_c.to(tl.int64)
                w0 = tl.where(cmask, (column * 37 + 7) % 1021 + 1, 0)
                w1 = tl.where(cmask, (column * 613 + 7) % 1021 + 1, 0)
                w2 = tl.where(cmask, (column * 1 + 7) % 1021 + 1, 0)
                if WIDE:
                    low = bits & 0xFFFF
                    high = bits >> 16
                    a0 += tl.sum(low * w0[None, :], axis=1)
                    a1 += tl.sum(low * w1[None, :], axis=1)
                    a2 += tl.sum(low * low * w2[None, :], axis=1)
                    h0 += tl.sum(high * w0[None, :], axis=1)
                    h1 += tl.sum(high * w1[None, :], axis=1)
                    h2 += tl.sum(high * high * w2[None, :], axis=1)
                else:
                    a0 += tl.sum(bits * w0[None, :], axis=1)
                    a1 += tl.sum(bits * w1[None, :], axis=1)
                    a2 += tl.sum(bits * bits * w2[None, :], axis=1)
            row = offs_r.to(tl.int64) + row_start
            r0 = tl.where(rmask, (row * 101 + 3) % 1019 + 1, 0)
            r1 = tl.where(rmask, (row * 211 + 3) % 1019 + 1, 0)
            r2 = tl.where(rmask, (row * 17 + 3) % 1019 + 1, 0)
            m0 = a0 % prime
            m0 = tl.where(m0 < 0, m0 + prime, m0)
            m1 = a1 % prime
            m1 = tl.where(m1 < 0, m1 + prime, m1)
            m2 = a2 % prime
            m2 = tl.where(m2 < 0, m2 + prime, m2)
            tl.store(out_ptr + pid * 6, tl.sum(m0 * r0, axis=0))
            tl.store(out_ptr + pid * 6 + 1, tl.sum(m1 * r1, axis=0))
            tl.store(out_ptr + pid * 6 + 2, tl.sum(m2 * r2, axis=0))
            if WIDE:
                n0 = h0 % prime
                n0 = tl.where(n0 < 0, n0 + prime, n0)
                n1 = h1 % prime
                n1 = tl.where(n1 < 0, n1 + prime, n1)
                n2 = h2 % prime
                n2 = tl.where(n2 < 0, n2 + prime, n2)
                tl.store(out_ptr + pid * 6 + 3, tl.sum(n0 * r0, axis=0))
                tl.store(out_ptr + pid * 6 + 4, tl.sum(n1 * r1, axis=0))
                tl.store(out_ptr + pid * 6 + 5, tl.sum(n2 * r2, axis=0))

        @triton.jit
        def dot2_gemm(xt_ptr, bt_ptr, out_ptr, n, r, c, sx, sb, so,
                      BM: tl.constexpr, BN: tl.constexpr):
            # out[i, j] = sum_k xt[k, i] * bt[k, j]: Dot2 (TwoProduct by FMA, TwoSum), FP32 result
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            offs_m = pid_m * BM + tl.arange(0, BM)
            offs_n = pid_n * BN + tl.arange(0, BN)
            mask_m = offs_m < n
            mask_n = offs_n < r
            hi = tl.zeros([BM, BN], dtype=tl.float32)
            lo = tl.zeros([BM, BN], dtype=tl.float32)
            for k in range(0, c):
                xv = tl.load(xt_ptr + k * sx + offs_m, mask=mask_m, other=0.0)
                bv = tl.load(bt_ptr + k * sb + offs_n, mask=mask_n, other=0.0)
                p = xv[:, None] * bv[None, :]
                e = tl.fma(xv[:, None], bv[None, :], -p)
                s = hi + p
                v = s - hi
                t = (hi - (s - v)) + (p - v)
                hi = s
                lo = lo + (t + e)
            out = out_ptr + offs_m.to(tl.int64)[:, None] * so + offs_n[None, :]
            tl.store(out, hi + lo, mask=mask_m[:, None] & mask_n[None, :])

        kernels = {"sq_partials": sq_partials, "fingerprint_partials": fingerprint_partials, "dot2_gemm": dot2_gemm}
        _TRITON["kernels"] = _self_test(kernels)
        return _TRITON["kernels"]


def _self_test(kernels):
    """Compile and check both kernels once on the current CUDA device (None: Triton unusable here, the base paths
    run; the error is kept in ``STATS["triton_error"]``)."""
    if not torch.cuda.is_available():
        return kernels
    try:
        x = torch.tensor([[3.0, -4.0], [0.5, 2.0]], device="cuda")
        out = torch.zeros((1, 3), dtype=torch.float64, device="cuda")
        kernels["sq_partials"][(1,)](x, x * 0, out, 2, 2, 2, 1, 2, 1, 4.0, HAS_R=True, DIV=True, BLOCK_R=SQ_BLOCK_R,
                                     BLOCK_C=SQ_BLOCK_C)
        bits = torch.tensor([[1, -2], [3, 4]], dtype=torch.int16, device="cuda")
        fp = torch.zeros((1, 6), dtype=torch.int64, device="cuda")
        kernels["fingerprint_partials"][(1,)](bits, fp, 2, 2, 0, 2, 1, 2147483629, WIDE=False, BLOCK_R=FP_BLOCK_R,
                                              BLOCK_C=FP_BLOCK_C)
        xt = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
        product = torch.zeros((2, 2), dtype=torch.float32, device="cuda")
        kernels["dot2_gemm"][(1, 1)](xt, xt, product, 2, 2, 2, 2, 2, 2, BM=DOT2_BM, BN=DOT2_BN, num_warps=8,
                                     enable_fp_fusion=False)
        torch.cuda.synchronize()
        if abs(float(out[0, 0]) - 1.828125) > 1e-12 or abs(float(out[0, 1]) - 1.828125) > 1e-12:
            raise RuntimeError(f"sq_partials self-test returned {out.tolist()}")
        if product.tolist() != [[10.0, 14.0], [14.0, 20.0]]:
            raise RuntimeError(f"dot2_gemm self-test returned {product.tolist()}")
        return kernels
    except Exception as exc:  # noqa: BLE001 - recorded; the base paths run
        with _STATS_LOCK:
            STATS["triton_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return None


_SQ_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def sq_partials(x: torch.Tensor, reconstructed: torch.Tensor | None, scale: float, out: torch.Tensor) -> None:
    """Write ``[programs, 3]`` FP64 partials (``sum (x/s)^2``, ``sum ((x - r)/s)^2``, nonfinite flag) of the 2-D CUDA
    tile ``x`` into ``out`` (``scale`` 0: no division). ``programs = ceil(rows / SQ_BLOCK_R)``."""
    kernels = _kernels()
    rows, cols = (int(size) for size in x.shape)
    has_r = reconstructed is not None
    r = reconstructed if has_r else x
    with torch.cuda.device(x.device):
        kernels["sq_partials"][(sq_programs(rows),)](
            x, r, out, rows, cols, x.stride(0), x.stride(1), r.stride(0), r.stride(1), float(scale) if scale else 1.0,
            HAS_R=has_r, DIV=bool(scale), BLOCK_R=SQ_BLOCK_R, BLOCK_C=SQ_BLOCK_C)


def sq_programs(rows: int) -> int:
    return -(-int(rows) // SQ_BLOCK_R)


def fused_ok(x: torch.Tensor) -> bool:
    return x.is_cuda and x.ndim == 2 and x.dtype in _SQ_DTYPES and _kernels() is not None


def sumsq64(x: torch.Tensor) -> torch.Tensor:
    """0-d FP64 ``sum(x * x)`` of a 2-D CUDA tensor: FP32 squares accumulated in FP64 (fused, deterministic)."""
    if not fused_ok(x):
        return torch.sum(x.float() * x.float(), dtype=torch.float64)
    out = torch.empty((sq_programs(x.shape[0]), 3), dtype=torch.float64, device=x.device)
    sq_partials(x, None, 0.0, out)
    return out[:, 0].sum()


def energy_fp32(tensor: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """B2 ``energy``: (0-d FP64 ``sum (x/scale)^2``, 0-d bool finite) of a CUDA matrix in one fused kernel."""
    _count("energy_fused")
    out = torch.empty((sq_programs(tensor.shape[0]), 3), dtype=torch.float64, device=tensor.device)
    sq_partials(tensor, None, scale, out)
    totals = out.sum(0)
    return totals[0], totals[2] == 0


def residual_fp32(tensor: torch.Tensor, coefficients: torch.Tensor, b32: torch.Tensor, tile_rows: int,
                  scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """B2 ``residual``: (source, error) 0-d FP64 sums of ``projection_routine``'s residual tiles with FP32
    reconstructions (:func:`reelcache.exact_blas.mm`) and one fused kernel per tile."""
    from .exact_blas import mm
    rows = int(tensor.shape[0])
    starts = list(range(0, rows, tile_rows))
    offsets, total = [], 0
    for start in starts:
        offsets.append(total)
        total += sq_programs(min(start + tile_rows, rows) - start)
    out = torch.empty((total, 3), dtype=torch.float64, device=tensor.device)
    for start, offset in zip(starts, offsets):
        stop = min(start + tile_rows, rows)
        reconstructed = mm(coefficients[start:stop], b32)
        sq_partials(tensor[start:stop], reconstructed, scale, out[offset:])
        del reconstructed
    _count("residual_fused_tiles", len(starts))
    totals = out.sum(0)
    return totals[0], totals[1]


def exact_div_fp32(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """``(tensor.double() / scale).float()`` as one IEEE FP32 division by a device scalar: the same values for 16- and
    32-bit inputs and an FP32-representable ``scale`` (the FP64 quotient lies within 2^-51 of the exact one, which is
    at least 2^-49 from any FP32 rounding midpoint, so both round alike)."""
    divisor = torch.full((), scale, dtype=torch.float32, device=tensor.device)
    return torch.div(tensor if tensor.dtype == torch.float32 else tensor.float(), divisor)


def planner_fp32(tensor: torch.Tensor, scale: float | None = None) -> bool:
    """Whether the ``planner`` pass applies to ``tensor`` (and ``scale`` is FP32-representable)."""
    if not fp32("planner", tensor) or tensor.dtype not in _SQ_DTYPES:
        return False
    if scale is not None and float(torch.tensor(scale, dtype=torch.float32)) != scale:
        return False
    _count("planner_fp32")
    return True


def span_energy_fp32(coefficients: torch.Tensor, rows16: torch.Tensor, source_energy: float):
    """B2 ``span``: routine of ``<G^-1, M> / source_energy`` (see the module docstring); returns None when the Neumann
    series cannot be certified (the caller then runs the FP64 routine)."""
    from .exact_blas import mm
    device = coefficients.device
    q = rows16.to(device=device, dtype=torch.float32)
    delta = mm(q, q.T)
    diagonal = torch.sum(q * q, dim=1, dtype=torch.float64)  # FP16 squares are exact in FP32: an exact diagonal
    delta.diagonal().copy_(diagonal - 1.0)
    (eta,) = yield (delta.abs().sum(1).max(),)
    if not math.isfinite(eta) or eta > SPAN_ETA_MAX:
        _count("span_fallbacks")
        return None
    terms = 0
    while terms < SPAN_MAX_TERMS and (eta ** (terms + 1)) / (1.0 - eta) > SPAN_TOLERANCE:
        terms += 1
    p = coefficients if coefficients.dtype == torch.float32 else coefficients.float()
    work = mm(p.T, p)
    traces = []
    for _ in range(terms):
        work = mm(delta, work)
        traces.append(torch.sum(work.diagonal(), dtype=torch.float64))
    leading = sumsq64(p)
    request = (leading, torch.stack(traces)) if traces else (leading,)
    values = yield request
    total = float(values[0])
    for index in range(terms):
        total += (-1.0) ** (index + 1) * float(values[1][index])
    if not math.isfinite(total):
        _count("span_fallbacks")
        return None
    _count("span_neumann")
    _count("span_terms", terms)
    return total / source_energy


def fingerprint_terms_fused(tensor: torch.Tensor, prime: int, tile_elements: int) -> torch.Tensor:
    """B4: ``lineage_commit._fingerprint_terms`` of a CUDA matrix (int64 ``[3]`` or ``[6]`` totals mod ``prime``) with
    one fused kernel per row tile of at most ``tile_elements`` elements; exact integer arithmetic, same values."""
    kernels = _kernels()
    source = tensor.detach()
    rows, columns = (int(size) for size in source.shape)
    wide = source.element_size() == 4
    bits = source.view(torch.int32 if wide else torch.int16)
    tile_rows = max(1, tile_elements // columns)
    starts = list(range(0, rows, tile_rows))
    programs = [-(-(min(start + tile_rows, rows) - start) // FP_BLOCK_R) for start in starts]
    out = torch.empty((sum(programs), 6), dtype=torch.int64, device=source.device)
    offset = 0
    with torch.cuda.device(source.device):
        for start, count in zip(starts, programs):
            chunk = bits[start:min(start + tile_rows, rows)]
            kernels["fingerprint_partials"][(count,)](
                chunk, out[offset:], int(chunk.shape[0]), columns, start, chunk.stride(0), chunk.stride(1),
                int(prime), WIDE=wide, BLOCK_R=FP_BLOCK_R, BLOCK_C=FP_BLOCK_C)
            offset += count
    _count("fingerprints_fused")
    totals = out.sum(0)[:6 if wide else 3]
    return totals.remainder(prime)


__all__ = ["FP32_PASSES", "FitterAccel", "OFF_PASSES", "condition_products", "current", "energy_fp32",
           "dot2_coefficients", "dot2_gemm", "exact_div_fp32", "recon_fp32", "verify_products",
           "fingerprint_source", "fingerprint_terms_fused", "fp32", "fused_fingerprints", "green_stream",
           "parse_fitter_fp64", "planner_fp32", "products", "residual_fp32", "scope", "span_energy_fp32", "stats",
           "sumsq64"]
