"""Deterministic factorization in the adapter's canonical channel domain.

K inputs are already normalized, *pre*-RoPE keys; this module must not apply
another normalization. V inputs are used directly. A stored basis is FP16 and
coefficients always follow ``P = X.float() @ B.float().T``. In particular, a
least-squares solve is not silently substituted after rounding the basis.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch

from .errors import CapacityError, ConfigurationError, UnsupportedError
from .types import TensorContext, canonical_dtype, json_copy


_DETECTOR_CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _detector_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(json_copy(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _detector_cpu_build_config() -> str:
    """Exclude only CUDAHooks' visibility-dependent, GPU-only diagnostics.

    The static Build settings line (including CUDA/CuDNN build versions), all
    CPU library/compiler details, and every unrecognized line remain bound.
    """
    cuda_runtime_prefixes = (
        "  - CUDA Runtime ",
        "  - Built with CUDA Runtime ",
        "  - NVCC architecture flags: ",
        "  - CuDNN ",
        "    - Built with CuDNN ",
        "  - Magma ",
    )
    return "".join(
        line for line in torch.__config__.show().splitlines(keepends=True)
        if not line.startswith(cuda_runtime_prefixes)
    )


def _detector_backend() -> str:
    return _detector_digest({
        "torch": str(torch.__version__), "build": _detector_cpu_build_config(),
        "architecture": platform.machine(), "cpu": torch.backends.cpu.get_cpu_capability(),
        "device": "cpu", "compute_dtype": "float32",
    })


def _detector_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _detector_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_detector_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class DetectorSpec:
    """The fixed acquisition recipe with an explicitly frozen CPU thread count."""

    solver: str = "qr_q2"
    device: str = "cpu"
    compute_dtype: str = "float32"
    norm_recipe: str = "scaled_original_fp64"
    power_iterations: int = 2
    initial_width: int = 128
    width_growth: int = 2
    width_cap: str = "min_rows_channels"
    energy_denominator: str = "full_original_frobenius_norm"
    stop_energy: float = 0.99
    feature_energies: tuple[float, ...] = (0.95, 0.99)
    base_seed: int = 0
    seed_recipe: str = "sha256_geometry_key_position_plus_attempt_v1"
    tile_rows: int = 1024
    cpu_threads: int = 1
    implementation_fingerprint: str = _DETECTOR_CODE_SHA256
    backend_fingerprint: str = field(default_factory=_detector_backend)
    schema_version: str = "reelcache.fixed_anchor_detector.v1"

    def __post_init__(self) -> None:
        fixed = {
            "solver": "qr_q2", "device": "cpu", "compute_dtype": "float32",
            "norm_recipe": "scaled_original_fp64", "initial_width": 128,
            "power_iterations": 2, "width_cap": "min_rows_channels",
            "energy_denominator": "full_original_frobenius_norm",
            "width_growth": 2, "stop_energy": 0.99, "base_seed": 0,
            "seed_recipe": "sha256_geometry_key_position_plus_attempt_v1",
            "tile_rows": 1024,
            "schema_version": "reelcache.fixed_anchor_detector.v1",
        }
        if any(type(getattr(self, key)) is not type(value) or getattr(self, key) != value
               for key, value in fixed.items()):
            raise ConfigurationError("DetectorSpec v1 is one fixed CPU acquisition recipe")
        if type(self.cpu_threads) is not int or self.cpu_threads < 1:
            raise ConfigurationError("detector CPU thread count must be a positive integer")
        if (
            not isinstance(self.feature_energies, (tuple, list))
            or any(type(value) is not float for value in self.feature_energies)
            or tuple(self.feature_energies) != (0.95, 0.99)
        ):
            raise ConfigurationError("the fixed detector measures both .95 and .99")
        object.__setattr__(self, "feature_energies", (0.95, 0.99))
        for value in (self.implementation_fingerprint, self.backend_fingerprint):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ConfigurationError("detector implementation/backend bindings require SHA256")

    @property
    def fingerprint(self) -> str:
        return _detector_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return json_copy({item.name: getattr(self, item.name) for item in fields(self)})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DetectorSpec:
        if not isinstance(value, Mapping) or set(value) != {item.name for item in fields(cls)}:
            raise ConfigurationError("detector specification must contain its complete versioned recipe")
        return cls(**dict(value))

    def validate_runtime(self) -> None:
        actual = {
            "implementation_fingerprint": _DETECTOR_CODE_SHA256,
            "backend_fingerprint": _detector_backend(),
            "cpu_threads": torch.get_num_threads(),
        }
        differences = [
            f"{name}: expected {getattr(self, name)!r}, got {value!r}"
            for name, value in actual.items() if getattr(self, name) != value
        ]
        if differences:
            raise ConfigurationError(
                "detector code, CPU backend, or thread contract differs from calibration: "
                + "; ".join(differences)
            )

    def seed_for(self, context: TensorContext) -> int:
        if not isinstance(context, TensorContext):
            raise ConfigurationError("detector seed requires the original TensorContext")
        digest = _detector_digest({
            "base_seed": self.base_seed, "layer": context.key.layer, "target": context.key.target,
            "token_start": context.token_start, "frames": context.frames,
            "height": context.height, "width": context.width,
            "layers": context.model.layers, "channels": context.channels,
        })
        return int(digest[:16], 16) % 2**63


@dataclass(frozen=True)
class Projection:
    coefficients: torch.Tensor
    relative_residual: float
    source_norm: float
    residual_norm: float
    zero_input: bool
    condition_bound: float
    seconds: float
    coefficient_seconds: float = 0.0
    guard_seconds: float = 0.0
    condition_seconds: float = 0.0


@dataclass(frozen=True)
class Factors:
    basis: torch.Tensor
    coefficients: torch.Tensor
    rank: int
    metrics: dict[str, Any] = field(default_factory=dict)


def tensor_sha256(tensor: torch.Tensor, *, tile_rows: int = 1024) -> str:
    """Hash the codec's little-endian row-major bytes without decomposing X."""
    if (
        not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
        or tensor.device.type == "meta" or tensor.ndim != 2
        or min(tensor.shape) < 1
    ):
        raise ConfigurationError("tensor digest requires a nonempty dense matrix")
    if type(tile_rows) is not int or tile_rows < 1:
        raise ConfigurationError("tile_rows must be positive")
    digest = hashlib.sha256()
    for start in range(0, tensor.shape[0], tile_rows):
        native = tensor[start:start + tile_rows].detach().contiguous().cpu()
        raw = native.view(torch.uint8).numpy().reshape(-1, native.element_size())
        if sys.byteorder != "little":
            raw = raw[:, ::-1]
        digest.update(raw.tobytes())
    return digest.hexdigest()


def validate_matrix(
    tensor: torch.Tensor, context: TensorContext, *, lossy: bool = True, check_values: bool = True,
) -> torch.Tensor:
    """Check geometry/identity without changing values or generation RNG.

    RAW permits all floating-point bit patterns, including NaN payloads.
    Lossy arithmetic requires finite values representable in FP32.
    ``check_values=False`` keeps every structural/key-space rule but defers the
    finite-value scan to a caller that repeats it on its device copy.
    """
    if not isinstance(context, TensorContext):
        raise ConfigurationError("context must be a TensorContext")
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise ConfigurationError("tensor must be a dense strided torch.Tensor")
    if tensor.device.type == "meta":
        raise ConfigurationError("meta tensors have no encodable values")
    if tensor.ndim != 2 or tuple(tensor.shape) != context.shape:
        raise ConfigurationError(f"tensor shape must equal context shape {context.shape}")
    if canonical_dtype(tensor.dtype) != context.dtype:
        raise ConfigurationError("tensor dtype differs from the declared matrix dtype")
    if context.key.target == "K":
        if context.key_space not in {"pre_rope", "post_rope"}:
            raise ConfigurationError("K requires an explicit pre_rope or post_rope key_space")
        if lossy and context.key_space != "pre_rope":
            raise ConfigurationError("SVD requires normalized pre-RoPE K; post-RoPE K is RAW-only")
    elif context.key_space != "none":
        raise ConfigurationError("V must be direct values with key_space='none'")
    result = tensor.detach()
    if lossy and check_values and (
        not bool(torch.isfinite(result).all())
        or not bool(torch.isfinite(result.float()).all())
    ):
        raise ConfigurationError("lossy input must be finite and representable in FP32")
    return result


def validate_rank(rank: int, shape: tuple[int, int]) -> int:
    if type(rank) is not int or not 1 <= rank <= min(shape):
        raise ConfigurationError(f"rank must be an integer in [1, {min(shape)}]")
    return rank


def validate_stored_basis(basis: torch.Tensor, channels: int) -> torch.Tensor:
    if not isinstance(basis, torch.Tensor) or basis.layout != torch.strided:
        raise ConfigurationError("reuse basis must be a dense torch.Tensor")
    if basis.device.type == "meta" or basis.dtype != torch.float16:
        raise ConfigurationError("reuse basis must be the actual stored FP16 basis")
    if (
        basis.ndim != 2 or basis.shape[1] != channels
        or not 1 <= basis.shape[0] <= channels
    ):
        raise ConfigurationError("stored basis must have shape [canonical_rank, channels]")
    if not bool(torch.isfinite(basis).all()):
        raise ConfigurationError("stored basis must be finite")
    return basis.detach()


def _projection_mark(work: dict[str, Any] | None, phase: str, *counters: str) -> None:
    if work is not None:
        work["failure_phase"] = phase
        for name in counters:
            work[name] += 1


@torch.no_grad()
def basis_condition_bound(
    basis: torch.Tensor, *, _work: dict[str, Any] | None = None,
) -> float:
    """Return an infinity-norm Gram condition bound without SVD/eigh.

    Cholesky is only a small stored-basis validity check, not a candidate
    factorization. A singular basis returns infinity and cannot be reused.
    """
    rounded = basis.to(dtype=torch.float64)
    _projection_mark(_work, "conditioning_gram", "matmul_count")
    gram = rounded @ rounded.T
    _projection_mark(_work, "conditioning_cholesky", "cholesky_count")
    chol, info = torch.linalg.cholesky_ex(gram, check_errors=False)
    if int(info.item()) != 0 or not bool(torch.isfinite(chol).all()):
        return math.inf
    _projection_mark(_work, "conditioning_inverse", "cholesky_inverse_count")
    inverse = torch.cholesky_inverse(chol)
    condition = float(
        (gram.abs().sum(1).max() * inverse.abs().sum(1).max()).item()
    )
    return condition if math.isfinite(condition) else math.inf


@torch.no_grad()
def fast_condition_bound(basis: torch.Tensor) -> float:
    """Cheap FP32 upper bound on :func:`basis_condition_bound` for near-orthonormal bases.

    With ``E = B Bᵀ - I`` and ``η = ‖E‖∞ < 1``, ``‖G‖∞ ≤ 1 + η`` and
    ``‖G⁻¹‖∞ ≤ 1 / (1 - η)``. FP16 products are exact in FP32; the FP32 sums
    get a rigorous ``γ_n`` slack. If the bound would exceed 2 (not near
    orthonormal), the exact FP64 Cholesky bound is returned instead.
    """
    rounded = basis.to(dtype=torch.float32)
    rows, channels = rounded.shape
    gram = rounded @ rounded.T
    norms = gram.diagonal().abs().max()
    gram.diagonal().sub_(1.0)
    eta = float(gram.abs().sum(1).max())
    unit = 2.0 ** -24
    gamma = channels * unit / (1 - channels * unit)
    eta += rows * gamma * float(norms) + rows * unit
    if not math.isfinite(eta) or eta >= 1 / 3:
        return basis_condition_bound(basis)
    return (1 + eta) / (1 - eta)


@torch.no_grad()
def project_onto_basis(
    tensor: torch.Tensor,
    basis: torch.Tensor,
    *,
    rank: int | None = None,
    tile_rows: int = 1024,
    max_condition: float = 1e6,
    _condition_bound: float | None = None,
    _work: dict[str, Any] | None = None,
) -> Projection:
    """Project once and measure a stable tiled *direct* residual.

    The matrix products use the actual stored FP16 basis promoted to FP32.
    Residual and source energies accumulate in FP64. This avoids cancellation
    in ``1 - ||P||² / ||X||²``, which is wrong for rounded, nonorthogonal bases.
    No random operations, SVD, or eigensolves are performed.
    """
    started = time.perf_counter()
    _projection_mark(_work, "projection_validation", "projection_count")
    if (
        not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
        or tensor.ndim != 2 or min(tensor.shape) < 1
        or not tensor.is_floating_point() or tensor.device.type == "meta"
        or not bool(torch.isfinite(tensor).all())
        or not bool(torch.isfinite(tensor.float()).all())
    ):
        raise ConfigurationError("projection requires a nonempty finite FP32-compatible matrix")
    stored = validate_stored_basis(basis, tensor.shape[1])
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
    copy_basis = stored.device != tensor.device
    copy_started = time.perf_counter()
    if copy_basis:
        _projection_mark(_work, "basis_device_copy", "basis_device_copy_count")
        if _work is not None:
            _work["basis_device_copy_bytes"] = None
    try:
        view = stored[:rank].to(device=tensor.device)
    finally:
        if _work is not None and copy_basis:
            _work["basis_device_copy_seconds"] += time.perf_counter() - copy_started
    if _work is not None and copy_basis:
        _work["basis_device_copy_bytes"] = view.numel() * view.element_size()
    condition_started = time.perf_counter()
    _projection_mark(_work, "conditioning")
    try:
        condition = (
            (basis_condition_bound(view) if _work is None else basis_condition_bound(view, _work=_work))
            if _condition_bound is None else _condition_bound
        )
    finally:
        condition_seconds = time.perf_counter() - condition_started
        if _work is not None:
            _work["condition_seconds"] += condition_seconds
    if not math.isfinite(condition) or condition > max_condition:
        raise ConfigurationError("stored basis is singular or ill-conditioned")
    b32 = view.float()
    coefficient_started = time.perf_counter()
    try:
        _projection_mark(_work, "coefficient_projection", "coefficient_projection_count", "matmul_count")
        coefficients = tensor.float() @ b32.T
        if not bool(torch.isfinite(coefficients).all()):
            raise ConfigurationError("projection overflowed FP32")
    finally:
        coefficient_seconds = time.perf_counter() - coefficient_started
        if _work is not None:
            _work["projection_coefficient_seconds"] += coefficient_seconds
    guard_started = time.perf_counter()
    try:
        _projection_mark(_work, "residual_guard", "source_norm_count", "residual_evaluation_count")
        scale = float(torch.maximum(tensor.amax().abs(), tensor.amin().abs()))
        source_energy = torch.zeros((), dtype=torch.float64, device=tensor.device)
        error_energy = torch.zeros_like(source_energy)
        for start in range(0, tensor.shape[0], tile_rows):
            stop = min(start + tile_rows, tensor.shape[0])
            source = tensor[start:stop].to(dtype=torch.float64)
            _projection_mark(_work, "residual_guard", "matmul_count", "residual_tile_count")
            reconstructed = coefficients[start:stop] @ b32
            error = source - reconstructed.to(dtype=torch.float64)
            if scale:
                source = source / scale
                error = error / scale
            source_energy += source.square().sum()
            error_energy += error.square().sum()
        source_squared, error_squared = float(source_energy), float(error_energy)
        if not math.isfinite(source_squared) or not math.isfinite(error_squared):
            raise ConfigurationError("nonfinite direct residual energy")
        zero = scale == 0.0
        residual = 0.0 if zero else math.sqrt(error_squared / source_squared)
        if not math.isfinite(residual):
            raise ConfigurationError("nonfinite relative projection residual")
    finally:
        guard_seconds = time.perf_counter() - guard_started
        if _work is not None:
            _work["projection_guard_seconds"] += guard_seconds
    _projection_mark(_work, "projection_finalization")
    return Projection(
        coefficients=coefficients.contiguous(),
        relative_residual=residual,
        source_norm=scale * math.sqrt(source_squared),
        residual_norm=scale * math.sqrt(error_squared),
        zero_input=zero,
        condition_bound=condition,
        seconds=time.perf_counter() - started,
        coefficient_seconds=coefficient_seconds,
        guard_seconds=guard_seconds,
        condition_seconds=condition_seconds,
    )


def _canonical_signs(basis: torch.Tensor) -> torch.Tensor:
    indices = basis.abs().argmax(dim=1, keepdim=True)
    pivots = basis.gather(1, indices)
    signs = torch.where(pivots < 0, -torch.ones_like(pivots), torch.ones_like(pivots))
    return basis * signs


@torch.no_grad()
def fit_basis(
    tensor: torch.Tensor,
    context: TensorContext,
    rank: int,
    *,
    method: str = "auto",
    seed: int = 0,
    power_iterations: int = 2,
) -> Factors:
    """Fit an ordered basis, round it once, then project using those bytes.

    Small/full-rank problems use exact SVD. Larger truncated problems use a
    seeded randomized range finder and a real reduced SVD, with a private
    device-local generator. There is no fork/reset of process-global RNG state.
    Determinism is for a fixed Torch/backend configuration, not a promise of
    identical LAPACK/cuSOLVER output across platforms or degenerate subspaces.
    """
    started = time.perf_counter()
    value = validate_matrix(tensor, context)
    validate_rank(rank, tuple(value.shape))
    gpu_methods = {"gesvd", "gesvdj", "gesvda", "torch_lowrank", "cupy", "gram"}
    if method not in {"auto", "exact", "randomized"} | gpu_methods:
        raise ConfigurationError("unsupported factorization method")
    if method in gpu_methods and value.device.type != "cuda":
        raise UnsupportedError("the requested factorization backend requires CUDA input")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ConfigurationError("factorization seed must be an integer in [0, 2**63)")
    if type(power_iterations) is not int or not 0 <= power_iterations <= 8:
        raise ConfigurationError("power_iterations must be an integer in [0, 8]")
    x = value.float()
    smallest = min(x.shape)
    sketch_rank = min(max(2 * rank, rank + 8), smallest)
    exact = method == "exact" or (
        method == "auto" and (smallest <= 512 or sketch_rank == smallest)
    )
    decomposition_started = time.perf_counter()
    qr_count = 0
    eigh_count = 0
    if method in {"gesvd", "gesvdj", "gesvda"}:
        _, _, vt = torch.linalg.svd(x, full_matrices=False, driver=method)
        fitted, algorithm = vt[:rank], f"cusolver_{method}"
    elif method == "torch_lowrank":
        # This API has no Generator argument; restore the selected RNG exactly.
        with torch.random.fork_rng(devices=[x.device.index]):
            torch.cuda.default_generators[x.device.index].manual_seed(seed)
            _, _, v = torch.svd_lowrank(x, q=sketch_rank, niter=power_iterations)
        fitted, algorithm = v[:, :rank].T, "torch_svd_lowrank"
        qr_count = 1 + 2 * power_iterations
    elif method == "cupy":
        try:
            import cupy
            import cupyx
        except ImportError as exc:
            raise UnsupportedError("CuPy SVD requires the optional cupy-cuda12x dependency") from exc
        with cupy.cuda.Device(x.device.index), cupy.cuda.ExternalStream(torch.cuda.current_stream(x.device).cuda_stream):
            with cupyx.errstate(linalg="raise"):
                _, _, vt = cupy.linalg.svd(cupy.from_dlpack(x), full_matrices=False)
            fitted = torch.from_dlpack(vt)[:rank]
        algorithm = "cupy_cusolver_svd"
    elif method == "gram":
        gram = x.T @ x
        _, vectors = torch.linalg.eigh((gram + gram.T) * 0.5)
        fitted, algorithm = vectors[:, -rank:].flip(1).T, "custom_fp32_gram_eigh"
        eigh_count = 1
    elif exact:
        _, _, vt = torch.linalg.svd(x, full_matrices=False)
        fitted = vt[:rank]
        algorithm = "exact_svd"
    else:
        if x.device.type not in {"cpu", "cuda"}:
            raise UnsupportedError("deterministic randomized SVD supports CPU and CUDA only")
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
        omega = torch.randn(
            x.shape[1], sketch_rank, dtype=torch.float32,
            device=x.device, generator=generator,
        )
        q, _ = torch.linalg.qr(x @ omega, mode="reduced")
        qr_count += 1
        for _ in range(power_iterations):
            z, _ = torch.linalg.qr(x.T @ q, mode="reduced")
            q, _ = torch.linalg.qr(x @ z, mode="reduced")
            qr_count += 2
        _, _, vt = torch.linalg.svd(q.T @ x, full_matrices=False)
        fitted = vt[:rank]
        algorithm = "randomized_svd"
    if value.device.type == "cuda":
        torch.cuda.current_stream(value.device).synchronize()
    decomposition_seconds = time.perf_counter() - decomposition_started
    basis = _canonical_signs(fitted).to(dtype=torch.float16).contiguous()
    if not bool(torch.isfinite(basis).all()):
        raise ConfigurationError("fitted basis is not finite in stored FP16")
    projection = project_onto_basis(value, basis, rank=rank)
    return Factors(
        basis=basis,
        coefficients=projection.coefficients,
        rank=rank,
        metrics={
            "factorization_method": algorithm,
            "factorization_seed": seed,
            "sketch_rank": sketch_rank if algorithm in ("randomized_svd", "torch_svd_lowrank") else None,
            "power_iterations": power_iterations if algorithm in ("randomized_svd", "torch_svd_lowrank") else 0,
            "decomposition_count": 1,
            "svd_count": 1 - eigh_count,
            "eigh_count": eigh_count,
            "qr_count": qr_count,
            "decomposition_seconds": decomposition_seconds,
            "projection_count": 1,
            "projection_seconds": projection.seconds,
            "projection_relative_residual": projection.relative_residual,
            "basis_condition_bound": projection.condition_bound,
            "zero_input": projection.zero_input,
            "fit_seconds": time.perf_counter() - started,
        },
    )


@dataclass(frozen=True)
class SourceEnergy:
    """An independently accumulated norm, including nonzero underflow cases."""

    scale: float
    scaled_energy: float
    source_norm: float
    source_energy: float
    zero_input: bool
    seconds: float


@dataclass(frozen=True)
class DirectionFit:
    """One ordered fit shared by the rank detector and the eventual encoder.

    Squared singular values are for ``X / energy.scale``. Their denominator is
    the independently measured, all-row ``energy.scaled_energy``, never the sum
    of this possibly truncated spectrum.
    """

    directions: torch.Tensor
    squared_singular_values: torch.Tensor
    energy: SourceEnergy
    shape: tuple[int, int]
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    @property
    def width(self) -> int:
        return int(self.directions.shape[0])

    def required_rank(self, energy_target: float) -> int | None:
        """Return a measured crossing, or None for an unresolved finite fit."""
        if (
            isinstance(energy_target, bool)
            or not isinstance(energy_target, (int, float))
            or not math.isfinite(energy_target)
            or not 0 < energy_target <= 1
        ):
            raise ConfigurationError("energy_target must be in (0, 1]")
        if self.energy.zero_input:
            return 0
        cumulative = self.squared_singular_values.double().cumsum(0)
        crossing = torch.nonzero(
            cumulative >= energy_target * self.energy.scaled_energy,
            as_tuple=False,
        )
        return None if crossing.numel() == 0 else int(crossing[0, 0]) + 1

    def energy_fraction(self, rank: int) -> float:
        if type(rank) is not int or not 0 <= rank <= self.width:
            raise ConfigurationError("energy rank is outside the fitted directions")
        if self.energy.zero_input:
            return 1.0
        return float(self.squared_singular_values[:rank].double().sum()) / self.energy.scaled_energy


class RankFitError(ConfigurationError):
    """A failed fit retains counts for operations that were actually invoked."""

    def __init__(self, message: str, metrics: Mapping[str, Any]):
        self.metrics = MappingProxyType(dict(metrics))
        super().__init__(message)


class RankProjectionError(ConfigurationError):
    """An unsuccessful stored projection retains its actual attempted work."""

    def __init__(self, message: str, metrics: Mapping[str, Any]):
        self.metrics = MappingProxyType(dict(metrics))
        super().__init__(message)

    def __reduce__(self):
        return type(self), (str(self), dict(self.metrics))


def rank_fit_workspace_bytes(
    shape: tuple[int, int],
    width: int,
    *,
    tile_rows: int = 1024,
    solver_workspace_bytes: int = 0,
) -> int:
    """Conservative live-tensor allowance plus a declared solver reserve.

    This is admission accounting, not a measured allocator peak. Backend
    allocator caches, the caller-owned source, and staged other layers are
    separate. Native runs must additionally report measured allocated/reserved
    peaks; this function does not infer library scratch from tensor sizes.
    """
    validate_rank(width, shape)
    for name, value in (("tile_rows", tile_rows), ("solver_workspace_bytes", solver_workspace_bytes)):
        if type(value) is not int or value < (1 if name == "tile_rows" else 0):
            raise ConfigurationError(f"{name} has an invalid value")
    rows, channels = shape
    tile = min(tile_rows, rows)
    live = 4 * (3 * rows * width + 4 * channels * width + 6 * width * width)
    return max(5 * rows * channels, live + 32 * tile * channels) + solver_workspace_bytes


def rank_projection_workspace_bytes(
    shape: tuple[int, int],
    rank: int,
    *,
    tile_rows: int = 1024,
    canonical_rank: int | None = None,
    solver_workspace_bytes: int = 0,
) -> int:
    """Admission allowance for actual FP32 P and the direct residual screen."""
    validate_rank(rank, shape)
    canonical_rank = rank if canonical_rank is None else canonical_rank
    if type(canonical_rank) is not int or not rank <= canonical_rank <= shape[1]:
        raise ConfigurationError("canonical basis rank is invalid")
    if type(tile_rows) is not int or tile_rows < 1:
        raise ConfigurationError("tile_rows must be positive")
    if type(solver_workspace_bytes) is not int or solver_workspace_bytes < 0:
        raise ConfigurationError("solver_workspace_bytes must be nonnegative")
    rows, channels = shape
    return (
        4 * rows * channels + 4 * rows * rank
        + 2 * canonical_rank * channels + 16 * rank * channels
        + 64 * rank * rank + 40 * min(tile_rows, rows) * channels
        + solver_workspace_bytes
    )


@torch.no_grad()
def measure_source_energy(
    tensor: torch.Tensor, *, tile_rows: int = 1024,
) -> SourceEnergy:
    """Measure the original values in scaled FP64 without an FP32 pre-round."""
    started = time.perf_counter()
    if (
        not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
        or tensor.device.type == "meta" or tensor.ndim != 2
        or min(tensor.shape) < 1 or not tensor.is_floating_point()
    ):
        raise ConfigurationError("source energy requires a nonempty floating matrix")
    if type(tile_rows) is not int or tile_rows < 1:
        raise ConfigurationError("tile_rows must be a positive integer")
    scale = float(torch.maximum(tensor.amax().abs(), tensor.amin().abs()))
    if not math.isfinite(scale):
        raise ConfigurationError("source energy requires finite input")
    total = torch.zeros((), dtype=torch.float64, device=tensor.device)
    if scale:
        for start in range(0, tensor.shape[0], tile_rows):
            block = tensor[start:start + tile_rows].double() / scale
            if not bool(torch.isfinite(block).all()):
                raise ConfigurationError("source energy requires finite input")
            total += block.square().sum()
    scaled = float(total)
    if not math.isfinite(scaled) or (scale != 0 and scaled <= 0):
        raise ConfigurationError("invalid original source energy")
    norm = scale * math.sqrt(scaled)
    energy = norm * norm
    if not math.isfinite(norm) or not math.isfinite(energy):
        raise ConfigurationError("original source energy exceeds FP64")
    return SourceEnergy(
        scale, scaled, norm, energy, scale == 0, time.perf_counter() - started,
    )


@torch.no_grad()
def fit_rank_directions(
    tensor: torch.Tensor,
    context: TensorContext,
    width: int,
    *,
    seed: int = 0,
    power_iterations: int = 2,
    tile_rows: int = 1024,
    max_workspace_bytes: int | None = None,
    solver_workspace_bytes: int = 0,
) -> DirectionFit:
    """Fit full-channel QR directions and their energy in one reduced SVD.

    Matrix products consume original rows in tiles. A private generator is the
    only RNG touched. Increasing the width is an explicitly counted refit;
    callers must not count a finite-width miss as an energy-qualified result.
    """
    started = time.perf_counter()
    if not isinstance(context, TensorContext):
        raise ConfigurationError("context must be a TensorContext")
    workspace = rank_fit_workspace_bytes(
        context.shape, width, tile_rows=tile_rows,
        solver_workspace_bytes=solver_workspace_bytes,
    )
    if max_workspace_bytes is not None:
        if type(max_workspace_bytes) is not int or max_workspace_bytes < 1:
            raise ConfigurationError("max_workspace_bytes must be positive")
        if workspace > max_workspace_bytes:
            raise CapacityError("rank fit exceeds its declared workspace allowance")
    value = validate_matrix(tensor, context)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ConfigurationError("factorization seed must be an integer in [0, 2**63)")
    if type(power_iterations) is not int or not 0 <= power_iterations <= 8:
        raise ConfigurationError("power_iterations must be an integer in [0, 8]")
    if value.device.type not in {"cpu", "cuda"}:
        raise UnsupportedError("deterministic rank fitting supports CPU and CUDA only")
    energy = measure_source_energy(value, tile_rows=tile_rows)
    metrics = {
        "factorization_method": "qr_reduced_svd",
        "factorization_seed": seed,
        "sketch_rank": width,
        "power_iterations": power_iterations,
        "decomposition_count": 0, "svd_count": 0, "eigh_count": 0,
        "qr_count": 0, "matmul_count": 0, "projection_count": 0,
        "source_norm_count": 1, "source_norm_seconds": energy.seconds,
        "range_seconds": 0.0, "qr_seconds": 0.0, "svd_seconds": 0.0,
        "decomposition_seconds": 0.0, "projection_seconds": 0.0,
        "workspace_admission_bytes": workspace,
        "solver_workspace_reserve_bytes": solver_workspace_bytes,
        "workspace_is_measured_peak": False,
        "timing_clock": "host_perf_counter",
        "subcomponent_timings_synchronized": value.device.type == "cpu",
        "zero_input": energy.zero_input,
    }
    if energy.zero_input:
        directions = torch.eye(
            width, value.shape[1], dtype=torch.float32, device=value.device,
        )
        spectrum = torch.zeros(width, dtype=torch.float64, device=value.device)
        metrics.update(factorization_method="zero_input_identity", fit_seconds=time.perf_counter() - started)
        return DirectionFit(directions, spectrum, energy, tuple(value.shape), metrics)

    decomposition_started = time.perf_counter()

    def fail(message, error=None):
        metrics["decomposition_seconds"] = time.perf_counter() - decomposition_started
        metrics["fit_seconds"] = time.perf_counter() - started
        raise RankFitError(message, metrics) from error

    def blocks():
        for start in range(0, value.shape[0], tile_rows):
            stop = min(start + tile_rows, value.shape[0])
            yield start, stop, (value[start:stop].double() / energy.scale).float()

    def multiply(operation):
        mark = time.perf_counter()
        metrics["matmul_count"] += 1
        try:
            result = operation()
        except RuntimeError as error:
            metrics["range_seconds"] += time.perf_counter() - mark
            fail("rank range multiplication failed", error)
        metrics["range_seconds"] += time.perf_counter() - mark
        return result

    def qr(matrix):
        mark = time.perf_counter()
        metrics["qr_count"] += 1
        try:
            q, _ = torch.linalg.qr(matrix, mode="reduced")
        except RuntimeError as error:
            metrics["qr_seconds"] += time.perf_counter() - mark
            fail("rank range QR failed", error)
        metrics["qr_seconds"] += time.perf_counter() - mark
        return q

    generator = torch.Generator(device=value.device)
    generator.manual_seed(seed)
    omega = torch.randn(
        value.shape[1], width, dtype=torch.float32,
        device=value.device, generator=generator,
    )
    y = torch.empty(value.shape[0], width, dtype=torch.float32, device=value.device)
    for start, stop, block in blocks():
        y[start:stop] = multiply(lambda: block @ omega)
    q = qr(y)
    del y, omega, block
    for _ in range(power_iterations):
        z = torch.zeros(value.shape[1], width, dtype=torch.float32, device=value.device)
        for start, stop, block in blocks():
            z += multiply(lambda: block.T @ q[start:stop])
        z = qr(z)
        del q
        y = torch.empty(value.shape[0], width, dtype=torch.float32, device=value.device)
        for start, stop, block in blocks():
            y[start:stop] = multiply(lambda: block @ z)
        q = qr(y)
        del y, z
    reduced = torch.zeros(width, value.shape[1], dtype=torch.float32, device=value.device)
    for start, stop, block in blocks():
        reduced += multiply(lambda: q[start:stop].T @ block)
    del q, block
    svd_started = time.perf_counter()
    metrics["svd_count"] = metrics["decomposition_count"] = 1
    try:
        _, singular, directions = torch.linalg.svd(reduced, full_matrices=False)
    except RuntimeError as error:
        metrics["svd_seconds"] = time.perf_counter() - svd_started
        fail("rank reduced SVD failed", error)
    metrics["svd_seconds"] = time.perf_counter() - svd_started
    if (
        not bool(torch.isfinite(singular).all())
        or not bool(torch.isfinite(directions).all())
    ):
        fail("rank direction fitting produced nonfinite values")
    directions = _canonical_signs(directions).contiguous()
    spectrum = singular.double().square()
    metrics["decomposition_seconds"] = time.perf_counter() - decomposition_started
    metrics["fit_seconds"] = time.perf_counter() - started
    return DirectionFit(directions, spectrum, energy, tuple(value.shape), metrics)


@dataclass(frozen=True)
class AnchorAcquisition:
    spec: DetectorSpec
    context: TensorContext
    source_sha256: str
    fit: DirectionFit
    r95: int | None
    r99: int | None
    attempts: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.spec, DetectorSpec) or not isinstance(self.fit, DirectionFit)
            or self.fit.shape != self.context.shape or self.fit.directions.device.type != "cpu"
            or self.fit.directions.dtype != torch.float32
        ):
            raise ConfigurationError("fixed acquisition must retain its own full-channel CPU fit")
        for rank in (self.r95, self.r99):
            if rank is not None and (type(rank) is not int or not 0 <= rank <= self.fit.width):
                raise ConfigurationError("invalid fixed detector crossing")
        object.__setattr__(self, "attempts", _detector_freeze(json_copy(self.attempts)))
        object.__setattr__(self, "metrics", _detector_freeze(json_copy(self.metrics)))

    @property
    def directions(self) -> torch.Tensor:
        return self.fit.directions

    def rank_at(self, tau: float) -> int | None:
        if tau not in (0.95, 0.99):
            raise ConfigurationError("fixed anchor features exist only for .95 and .99")
        return self.r95 if tau == 0.95 else self.r99

    def to_dict(self) -> dict[str, Any]:
        return json_copy({
            "schema_version": "reelcache.anchor_acquisition.v1",
            "detector_spec": self.spec.to_dict(), "detector_fingerprint": self.spec.fingerprint,
            "context": self.context.to_dict(), "source_sha256": self.source_sha256,
            "source_hash_format": "sha256-little-endian-row-major-bytes-no-header-v1",
            "final_width": self.fit.width, "features": {"0.95": self.r95, "0.99": self.r99},
            "status": "unresolved" if self.r99 is None else "zero" if self.fit.energy.zero_input else "resolved",
            "attempts": self.attempts, "metrics": self.metrics,
        })


class DetectorAcquisitionError(ConfigurationError):
    def __init__(self, message: str, metrics: Mapping[str, Any]):
        self.metrics = _detector_freeze(json_copy(metrics))
        super().__init__(message)

    def __reduce__(self):
        return type(self), (str(self), json_copy(self.metrics))


@torch.no_grad()
def acquire_anchor_directions(
    tensor: torch.Tensor, context: TensorContext, *,
    spec: DetectorSpec | None = None,
    max_workspace_bytes: int = 1 << 30,
    solver_workspace_bytes: int = 64 << 20,
) -> AnchorAcquisition:
    """Acquire fixed .95/.99 features and usable directions from one final fit.

    Every invocation owns one admitted original-dtype CPU copy. Recipe widths,
    stopping, seed, device and arithmetic never depend on learned predictors.
    A finite-width miss is never normalized into a successful crossing.
    """
    started = time.perf_counter()
    spec = DetectorSpec() if spec is None else spec
    if not isinstance(spec, DetectorSpec):
        raise ConfigurationError("fixed acquisition requires a DetectorSpec")
    spec.validate_runtime()
    if type(max_workspace_bytes) is not int or max_workspace_bytes < 1:
        raise ConfigurationError("detector workspace allowance must be positive")
    value = validate_matrix(tensor, context, lossy=False)
    if context.key_space != ("pre_rope" if context.key.target == "K" else "none"):
        raise ConfigurationError("fixed acquisition requires normalized pre-RoPE K or direct V")
    full_width = min(context.shape)
    width = min(spec.initial_width, full_width)
    copied_bytes = value.numel() * value.element_size()
    seed = spec.seed_for(context)
    counters = (
        "decomposition_count", "svd_count", "eigh_count", "eigvalsh_count", "qr_count",
        "projection_count", "matmul_count", "source_norm_count", "source_hash_count", "source_copy_count",
    )
    timings = (
        "fit_seconds", "decomposition_seconds", "source_norm_seconds", "qr_seconds",
        "svd_seconds", "range_seconds",
    )
    work = {name: 0 for name in counters}
    work.update({name: 0.0 for name in timings})
    work.update(
        operation_counts_complete=True, fit_attempts=0, finite_width_misses=0, source_copy_bytes=0,
        source_copy_seconds=0.0, source_hash_seconds=0.0,
        workspace_admission_bytes=0, workspace_limit_bytes=max_workspace_bytes,
        solver_workspace_reserve_bytes=solver_workspace_bytes, workspace_is_measured_peak=False,
        detector_fingerprint=spec.fingerprint, source_sha256=None,
        factorization_seed=seed, detector_device="cpu", source_device=str(value.device),
        compute_dtype="float32", source_dtype=context.dtype, cpu_threads=spec.cpu_threads,
        timing_clock="host_perf_counter", subcomponent_timings_synchronized=True,
        acquisition_scope="original_owned_cpu_copy_and_all_fixed_fits",
    )
    attempts = []

    def fail(reason: str, status: str, error: Exception | None = None) -> None:
        work.update(
            status=status, failure_reason=reason,
            failure_type=None if error is None else type(error).__name__,
            failure_detail=None if error is None else str(error),
            attempts=attempts, elapsed_seconds=time.perf_counter() - started,
        )
        raise DetectorAcquisitionError(reason, work) from error

    def admit(candidate_width: int) -> None:
        needed = copied_bytes + rank_fit_workspace_bytes(
            context.shape, candidate_width, tile_rows=spec.tile_rows,
            solver_workspace_bytes=solver_workspace_bytes,
        )
        if needed > max_workspace_bytes:
            work["requested_workspace_bytes"] = needed
            work["requested_fit_width"] = candidate_width
            fail("fixed_detector_workspace_limit", "OVER_BUDGET")
        work["workspace_admission_bytes"] = max(work["workspace_admission_bytes"], needed)

    def charge(metrics: Mapping[str, Any]) -> None:
        for name in counters:
            amount = metrics.get(name, 0)
            work[name] = None if work[name] is None or amount is None else work[name] + int(amount)
        for name in timings:
            amount = metrics.get(name, 0.0)
            work[name] = None if work[name] is None or amount is None else work[name] + float(amount)
        work["operation_counts_complete"] &= metrics.get("operation_counts_complete", True)

    admit(width)
    copy_started = time.perf_counter()
    work["source_copy_count"] = 1
    work["source_copy_bytes"] = None
    try:
        original = value.to(device="cpu", copy=True, memory_format=torch.contiguous_format)
        work["source_copy_bytes"] = copied_bytes
    except Exception as error:
        work["source_copy_seconds"] = time.perf_counter() - copy_started
        fail("fixed_detector_source_copy_failed", "UNRESOLVED", error)
    work["source_copy_seconds"] = time.perf_counter() - copy_started
    hash_started = time.perf_counter()
    work["source_hash_count"] += 1
    try:
        source_hash = tensor_sha256(original, tile_rows=spec.tile_rows)
        work["source_sha256"] = source_hash
    except Exception as error:
        work["source_hash_seconds"] = time.perf_counter() - hash_started
        fail("fixed_detector_source_hash_failed", "UNRESOLVED", error)
    work["source_hash_seconds"] = time.perf_counter() - hash_started
    fit = None
    while True:
        admit(width)
        attempt_seed = (seed + work["fit_attempts"]) % 2**63
        work["fit_attempts"] += 1
        mark = time.perf_counter()
        try:
            fit = fit_rank_directions(
                original, context, width, seed=attempt_seed, power_iterations=spec.power_iterations,
                tile_rows=spec.tile_rows, max_workspace_bytes=max_workspace_bytes - copied_bytes,
                solver_workspace_bytes=solver_workspace_bytes,
            )
        except RankFitError as error:
            charge(error.metrics)
            attempts.append({"width": width, "seed": attempt_seed, "status": "failed"})
            fail("fixed_detector_fit_failed", "UNRESOLVED", error)
        except Exception as error:
            work["known_work_before_failed_fit"] = {name: work[name] for name in (*counters, *timings)}
            work["operation_counts_complete"] = False
            for name in ("decomposition_count", "svd_count", "qr_count", "matmul_count", "source_norm_count"):
                work[name] = None
            work["failed_fit_seconds"] = time.perf_counter() - mark
            work["fit_seconds"] += work["failed_fit_seconds"]
            attempts.append({"width": width, "seed": attempt_seed, "status": "uninstrumented_failure"})
            fail("fixed_detector_fit_failed", "UNRESOLVED", error)
        charge(fit.metrics)
        try:
            crossing = fit.required_rank(spec.stop_energy)
        except Exception as error:
            attempts.append({"width": width, "seed": attempt_seed, "status": "crossing_failure"})
            fail("fixed_detector_crossing_failed", "UNRESOLVED", error)
        work["finite_width_misses"] += int(crossing is None)
        attempts.append({
            "width": width, "seed": attempt_seed,
            "status": "resolved" if crossing is not None else "unresolved",
            "stop_crossing": crossing,
        })
        if crossing is not None or width == full_width:
            break
        width = min(full_width, width * spec.width_growth)
        fit = None
    r95, r99 = fit.required_rank(0.95), fit.required_rank(0.99)
    work.update(
        final_width=width, finite_width_misses=sum(item["status"] == "unresolved" for item in attempts),
        status="unresolved" if r99 is None else "resolved",
        retained_direction_bytes=fit.directions.numel() * fit.directions.element_size(),
        retained_spectrum_bytes=fit.squared_singular_values.numel() * fit.squared_singular_values.element_size(),
        elapsed_seconds=time.perf_counter() - started,
    )
    return AnchorAcquisition(spec, context, source_hash, fit, r95, r99, tuple(attempts), work)


@torch.no_grad()
def project_rank_directions(
    tensor: torch.Tensor,
    directions: torch.Tensor,
    rank: int,
    *,
    tile_rows: int = 1024,
    max_workspace_bytes: int | None = None,
    solver_workspace_bytes: int = 0,
) -> Factors:
    """Round a selected prefix once and return the very P used by its guard.

    Unselected fitted directions are not serialized as gratuitous basis rows.
    No SVD, eigh, QR, or random draw is hidden in this projection step.
    Admitted failures raise RankProjectionError with attempted-work telemetry;
    unavailable counters are null rather than inferred to be zero.
    """
    if (
        not isinstance(tensor, torch.Tensor) or tensor.ndim != 2
        or tensor.layout != torch.strided or tensor.device.type == "meta"
        or min(tensor.shape) < 1 or not tensor.is_floating_point()
    ):
        raise ConfigurationError("rank projection requires a nonempty floating matrix")
    if (
        not isinstance(directions, torch.Tensor) or directions.ndim != 2
        or directions.layout != torch.strided or directions.device.type == "meta"
        or not directions.is_floating_point() or directions.shape[1] != tensor.shape[1]
        or not bool(torch.isfinite(directions).all())
    ):
        raise ConfigurationError("rank directions must be a finite full-channel matrix")
    validate_rank(rank, tuple(tensor.shape))
    if rank > directions.shape[0]:
        raise ConfigurationError("selected rank exceeds fitted directions")
    workspace = rank_projection_workspace_bytes(
        tuple(tensor.shape), rank, tile_rows=tile_rows,
        solver_workspace_bytes=solver_workspace_bytes,
    )
    if directions.device != tensor.device:
        workspace += 2 * rank * tensor.shape[1]
    if max_workspace_bytes is not None:
        if type(max_workspace_bytes) is not int or max_workspace_bytes < 1:
            raise ConfigurationError("max_workspace_bytes must be positive")
        if workspace > max_workspace_bytes:
            raise CapacityError("rank projection exceeds its declared workspace allowance")
    metrics = {
        "decomposition_count": 0, "svd_count": 0, "eigh_count": 0, "qr_count": 0,
        "projection_count": 0, "matmul_count": 0, "source_norm_count": 0,
        "cholesky_count": 0, "cholesky_inverse_count": 0,
        "basis_device_copy_count": 0, "basis_device_copy_bytes": 0,
        "basis_device_copy_seconds": 0.0,
        "coefficient_projection_count": 0, "residual_evaluation_count": 0, "residual_tile_count": 0,
        "projection_seconds": 0.0, "projection_coefficient_seconds": 0.0,
        "projection_guard_seconds": 0.0, "condition_seconds": 0.0, "basis_round_seconds": 0.0,
        "operation_counts_complete": True, "failure_phase": "basis_rounding",
        "tile_rows": tile_rows,
        "projection_convention": "P=X.float()@B.float().T",
        "guard_domain": "original_source_vs_fp32_P_at_stored_fp16_B",
        "workspace_admission_bytes": workspace,
        "solver_workspace_reserve_bytes": solver_workspace_bytes,
        "workspace_is_measured_peak": False, "timing_clock": "host_perf_counter",
        "subcomponent_timings_synchronized": tensor.device.type == "cpu",
    }
    projection_started = None
    try:
        rounding_started = time.perf_counter()
        try:
            basis = _canonical_signs(directions[:rank]).to(dtype=torch.float16).contiguous()
        finally:
            metrics["basis_round_seconds"] = time.perf_counter() - rounding_started
        projection_started = time.perf_counter()
        metrics["failure_phase"] = "projection_entry"
        projection = project_onto_basis(tensor, basis, rank=rank, tile_rows=tile_rows, _work=metrics)
        metrics.update(
            projection_seconds=projection.seconds,
            projection_relative_residual=projection.relative_residual,
            stored_svd_relative_residual=projection.relative_residual,
            post_hevc_relative_residual=None,
            source_norm=projection.source_norm, residual_norm=projection.residual_norm,
            basis_condition_bound=projection.condition_bound, zero_input=projection.zero_input,
            canonical_basis_rank=int(basis.shape[0]), view_rank=rank,
            basis_bytes=basis.numel() * basis.element_size(),
            coefficient_bytes=projection.coefficients.numel() * projection.coefficients.element_size(),
            coefficient_dtype="float32",
        )
        metrics.pop("failure_phase")
        return Factors(basis, projection.coefficients, rank, metrics)
    except Exception as error:
        if projection_started is not None:
            metrics["projection_seconds"] = time.perf_counter() - projection_started
            if metrics["projection_count"] == 0:
                metrics["operation_counts_complete"] = False
                for name in (
                    "projection_count", "matmul_count", "source_norm_count",
                    "cholesky_count", "cholesky_inverse_count", "coefficient_projection_count",
                    "basis_device_copy_count", "basis_device_copy_seconds", "basis_device_copy_bytes",
                    "residual_evaluation_count", "residual_tile_count",
                    "projection_coefficient_seconds", "projection_guard_seconds", "condition_seconds",
                ):
                    metrics[name] = None
        metrics.update(
            failure_type=type(error).__name__,
            failure_kind="configuration" if isinstance(error, ConfigurationError) else "backend",
        )
        raise RankProjectionError(str(error), metrics) from error
