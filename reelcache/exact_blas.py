"""Explicit-precision BLAS for archival planning and finalization: IEEE FP32 products on CUDA whatever the flags.

Torch's FP32 matmul and triangular solve on CUDA read process-global flags at every launch
(``torch.backends.cuda.matmul.allow_tf32``, ``torch.set_float32_matmul_precision`` and, from Torch 2.9,
``torch.backends.cuda.matmul.fp32_precision``). A thread that toggles them -- e.g. a model region reproducing native
TF32 numerics (SGLang RAVEN's ``fp32_region``) -- would switch the archival fitter's products between TF32 and IEEE
FP32 while archive_native plans concurrently, making ranks, decisions and bytes timing-dependent. cuSOLVER routines
(Cholesky, QR, the batched eigensolvers) do not depend on these flags (measured).

This module calls cuBLAS directly (``ctypes``) on the calling thread's own Torch cuBLAS handle -- Torch keeps one per
thread and device, bound to the current stream and Torch's workspace -- after setting that handle's math mode to
``CUBLAS_DEFAULT_MATH`` (IEEE FP32), and :func:`mm` passes Torch's own arguments (``cublasCommonArgs``), so its result
equals Torch's ``a @ b`` with TF32 disabled, bit for bit, and never depends on the flags. Other dtypes and CPU tensors
use Torch (FP64 GEMMs never use TF32; CPU FP32 products do not read the CUDA flags).

:func:`trsm_right_upper` (``X R^-1``) is the flag-free replacement of ``torch.linalg.solve_triangular(R, X,
upper=True, left=False)`` for one matrix; batched algorithms call both per matrix, so a matrix's result never depends
on which other matrices share its batch.
"""

from __future__ import annotations

import ctypes
import os
import threading

import torch

from . import accel_c as _accel_c
from .errors import ConfigurationError, UnsupportedError

__all__ = ["mm", "trsm_right_upper", "library_path", "available", "IEEE_MATH_MODE"]

_OP_N, _OP_T = 0, 1
_SIDE_LEFT = 0
_FILL_LOWER = 0
_DIAG_NON_UNIT = 0
#: ``CUBLAS_DEFAULT_MATH``: IEEE FP32 for single-precision routines (no TF32 tensor-op math).
IEEE_MATH_MODE = 0

_lock = threading.Lock()
_lib = None
_lib_path: str | None = None


def _mapped_cublas() -> list[str]:
    """libcublas files (not libcublasLt) already mapped into this process, in mapping order."""
    try:
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return []
    found: list[str] = []
    for line in lines:
        parts = line.split(maxsplit=5)
        if len(parts) == 6:
            path = parts[5].strip()
            if path.startswith("/") and os.path.basename(path).startswith("libcublas.so") and path not in found:
                found.append(path)
    return found


def _library():
    global _lib, _lib_path
    if _lib is not None:
        return _lib
    with _lock:
        if _lib is None:
            if not torch.cuda.is_available():
                raise UnsupportedError("explicit-precision cuBLAS products need CUDA")
            torch.cuda.current_blas_handle()  # Torch maps its libcublas and creates this thread's handle
            candidates = _mapped_cublas()
            if not candidates:
                raise UnsupportedError("Torch's libcublas is not mapped into this process")
            lib = ctypes.CDLL(candidates[0])
            pointer, integer, scalar = ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)
            lib.cublasSetMathMode.argtypes = [pointer, integer]
            lib.cublasSetMathMode.restype = integer
            lib.cublasGetMathMode.argtypes = [pointer, ctypes.POINTER(integer)]
            lib.cublasGetMathMode.restype = integer
            lib.cublasSgemm_v2.argtypes = [pointer, integer, integer, integer, integer, integer, scalar, pointer,
                                           integer, pointer, integer, scalar, pointer, integer]
            lib.cublasSgemm_v2.restype = integer
            lib.cublasStrsm_v2.argtypes = [pointer, integer, integer, integer, integer, integer, integer, scalar,
                                           pointer, integer, pointer, integer]
            lib.cublasStrsm_v2.restype = integer
            _lib, _lib_path = lib, candidates[0]
    return _lib


def available() -> bool:
    """True when the flag-free CUDA path can be bound (CUDA present and Torch's libcublas mapped)."""
    try:
        _library()
    except UnsupportedError:
        return False
    return True


def library_path() -> str | None:
    """The libcublas file bound (None before the first CUDA product)."""
    return _lib_path


def _ieee_handle(device: torch.device) -> int:
    """This thread's Torch cuBLAS handle on ``device`` (current stream, Torch's workspace) in IEEE FP32 math mode."""
    if _accel_c.fast() and _lib is not None:  # accel C (gil_fastpaths): the same calls without the device context
        index = device.index
        if index is not None and torch._C._cuda_getDevice() == index:
            handle = torch._C._cuda_getCurrentBlasHandle()  # binds the current stream and Torch's workspace
            status = _set_math_mode(handle)
            if status != 0:
                raise UnsupportedError(f"cublasSetMathMode failed with status {status}")
            return handle
    lib = _library()
    with torch.cuda.device(device):
        handle = torch.cuda.current_blas_handle()
    status = lib.cublasSetMathMode(ctypes.c_void_p(handle), IEEE_MATH_MODE)
    if status != 0:
        raise UnsupportedError(f"cublasSetMathMode failed with status {status}")
    return handle


def _set_math_mode(handle: int) -> int:
    """``cublasSetMathMode(handle, IEEE_MATH_MODE)`` through a cached ctypes binding (accel C)."""
    function = _MATH_MODE[0] if _MATH_MODE else None
    if function is None:
        function = _library().cublasSetMathMode
        _MATH_MODE.append(function)
    return function(handle, IEEE_MATH_MODE)


_MATH_MODE: list = []


def _flag_free(*tensors: torch.Tensor) -> bool:
    """CUDA FP32 operands, the only case whose Torch product reads the global matmul flags."""
    return all(t.is_cuda and t.dtype == torch.float32 for t in tensors)


def _check_2d(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.layout != torch.strided:
        raise ConfigurationError(f"{name} must be a strided 2-D tensor")
    if tensor.is_conj() or tensor.is_neg():
        raise ConfigurationError(f"{name} must not carry a lazy conjugate or negative bit")


def _non_overlapping_and_dense(sizes, strides) -> bool:
    """``c10::TensorImpl::compute_non_overlapping_and_dense`` for two dimensions."""
    order = sorted((dim for dim in (0, 1) if sizes[dim] >= 2), key=lambda dim: strides[dim])
    required = 1
    for dim in order:
        if strides[dim] != required:
            return False
        required *= sizes[dim]
    return True


def _prepare(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Torch's ``prepare_matrix_for_cublas`` (``Blas.cpp``) for a real matrix: (operand, transpose flag)."""
    sizes, strides = tuple(tensor.shape), tuple(tensor.stride())
    if _non_overlapping_and_dense(sizes, strides):
        return tensor, tensor.is_contiguous()
    if strides[0] == 1 and strides[1] >= max(1, sizes[0]):
        return tensor, False
    if strides[1] == 1 and strides[0] >= max(1, sizes[1]):
        return tensor, True
    return tensor.clone(memory_format=torch.contiguous_format), True


def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` for 2-D tensors; on CUDA FP32 an IEEE FP32 cuBLAS GEMM independent of the global flags.

    The cuBLAS call is Torch's own for ``torch.mm`` into a fresh contiguous result (``addmm_out_cuda_impl``:
    ``cublasCommonArgs`` with a row-major result, ``_cublasAdjustLdLevel3``, ``cublasSgemm``), so the result equals
    Torch's with TF32 disabled.
    """
    _check_2d("a", a)
    _check_2d("b", b)
    if a.shape[1] != b.shape[0]:
        raise ConfigurationError(f"mm shapes {tuple(a.shape)} and {tuple(b.shape)} do not chain")
    if not _flag_free(a, b):
        return torch.mm(a, b)
    if a.device != b.device:
        raise ConfigurationError("mm operands must share a device")
    m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    if out.numel() == 0:
        return out
    if k == 0:
        return out.zero_()
    # A fresh contiguous result is transposed for cuBLAS (column-major): C^T = mat2^T mat1^T.
    mata, transpose_a = _prepare(b)
    matb, transpose_b = _prepare(a)
    transpose_a, transpose_b = not transpose_a, not transpose_b
    cm, ck, cn = int(mata.shape[1]), int(mata.shape[0]), int(matb.shape[0])
    lda = int(mata.stride(1 if transpose_a else 0))
    ldb = int(matb.stride(1 if transpose_b else 0))
    ldc = int(out.stride(0))
    if cn <= 1:
        ldc = max(cm, 1)
    if transpose_a:
        if cm <= 1:
            lda = max(ck, 1)
    elif ck <= 1:
        lda = max(cm, 1)
    if transpose_b:
        if ck <= 1:
            ldb = max(cn, 1)
    elif cn <= 1:
        ldb = max(ck, 1)
    one, zero = ctypes.c_float(1.0), ctypes.c_float(0.0)
    handle = _ieee_handle(a.device)
    status = _library().cublasSgemm_v2(
        ctypes.c_void_p(handle), _OP_T if transpose_a else _OP_N, _OP_T if transpose_b else _OP_N, cm, cn, ck,
        ctypes.byref(one), ctypes.c_void_p(mata.data_ptr()), lda, ctypes.c_void_p(matb.data_ptr()), ldb,
        ctypes.byref(zero), ctypes.c_void_p(out.data_ptr()), ldc,
    )
    if status != 0:
        raise UnsupportedError(f"cublasSgemm failed with status {status}")
    return out


def trsm_right_upper(factor: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
    """``block @ factor^-1`` for an upper-triangular ``[n, n]`` factor and a ``[rows, n]`` block (new tensor).

    On CUDA FP32 one cuBLAS ``strsm`` in IEEE FP32 (column-major: ``factor^T Y^T = block^T``, the lower triangle of
    the row-major factor's memory); otherwise ``torch.linalg.solve_triangular(factor, block, upper=True,
    left=False)``.
    """
    _check_2d("factor", factor)
    _check_2d("block", block)
    n = int(factor.shape[0])
    if factor.shape[1] != n or block.shape[1] != n:
        raise ConfigurationError("trsm_right_upper needs a square factor matching the block's columns")
    if not _flag_free(factor, block):
        return torch.linalg.solve_triangular(factor, block, upper=True, left=False)
    if factor.device != block.device:
        raise ConfigurationError("trsm operands must share a device")
    rows = int(block.shape[0])
    out = block.clone(memory_format=torch.contiguous_format)
    if out.numel() == 0:
        return out
    upper = factor.contiguous()
    one = ctypes.c_float(1.0)
    handle = _ieee_handle(block.device)
    status = _library().cublasStrsm_v2(
        ctypes.c_void_p(handle), _SIDE_LEFT, _FILL_LOWER, _OP_N, _DIAG_NON_UNIT, n, rows, ctypes.byref(one),
        ctypes.c_void_p(upper.data_ptr()), n, ctypes.c_void_p(out.data_ptr()), n,
    )
    if status != 0:
        raise UnsupportedError(f"cublasStrsm failed with status {status}")
    return out
