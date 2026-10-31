"""Source-bound, uncentered, complete-channel FP64 energy spectra.

This is an offline oracle, not the cheap online rank detector. Feed original
normalized pre-RoPE K or direct V, never reconstructed factors, inverse-rotated
rounded keys, head-local channel shards, or a row sample. The caller attests the
source provenance; a digest detects changed bytes but cannot recover absent input.

The accumulator keeps one C-by-C Gram and bounded row tiles on CPU. Row shards
must cover the declared matrix in row order and retain *all* channels. Analysis
uses an adaptive power-of-two scale, without changing the source dtype or bytes:
``lambda_original = eigenvalues_scaled * 2**(2*scale_exponent)``. This notation
also represents energies outside the finite FP64 range without Infinity/NaN JSON.
There is no centering, extra K normalization, or leading-spectrum renormalization.

Numerical certification is conditional on ordinary IEEE FP64/BLAS/LAPACK
backward stability, not interval-arithmetic proof. The documented dimension-aware
error budget includes dot products, independent norm reduction, tile addition,
rescaling, and a conservative symmetric-eigensolver term. Only roundoff-sized
negative eigenvalues are clipped. Rank crossings inside this budget are returned
as intervals, including exact-looking threshold ties. The complete-rank endpoint
has energy fraction one by algebra, not by rescaling the computed spectrum.

No randomness or process-global thread/precision settings are changed. Fix the
backend build and thread settings for reproducibility. Canonical signs do not
make a repeated/near-degenerate eigenspace bit-identical across solvers. Memory
admission covers a conservative LAPACK/row-tile workspace estimate, not external
caller-owned input, process allocator caches, or unrelated resident matrices.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import torch

from .errors import CapacityError, ConfigurationError, IntegrityError, StateError


SCHEMA_VERSION = "reelcache.spectrum.v1"
HASH_FORMAT = "reelcache.original-row-major.v1"
DEFAULT_WORKSPACE_BYTES = 512 * 1024 * 1024
_EPS = sys.float_info.epsilon
_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}
_RETENTION = {
    "exact-input-retained", "verified-recapture-capable", "spectrum-only",
}


class SpectrumNumericalError(IntegrityError):
    """The complete spectrum cannot satisfy the declared FP64 checks.

    ``metrics`` is an optional read-only snapshot for an accumulator failure.
    Decomposition counts record invocations, including a solver that raised.
    ``total_seconds = exclusive_pipeline_seconds + eigensolve_seconds``:
    these cover admitted internal work through failure detection, excluding
    stream idle time, error publication and cleanup. Phase totals are nested
    diagnostics, not additional charges. The caller still measures whole-call
    elapsed time. Errors from standalone validation/SVD checks may have no
    metrics; absence means unavailable, never measured zero.
    """

    def __init__(self, *args: Any, metrics: Mapping[str, Any] | None = None) -> None:
        super().__init__(*args)
        if metrics is not None and not isinstance(metrics, Mapping):
            raise ConfigurationError("failure metrics must be a mapping")
        # Plain private state preserves normal Exception pickling for workers;
        # callers only receive a deeply read-only view.
        self._metrics = None if metrics is None else _plain_json(_freeze_json(metrics))

    @property
    def metrics(self) -> Mapping[str, Any] | None:
        return None if self._metrics is None else _freeze_json(self._metrics)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str, minimum: float | None = None) -> float:
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or (minimum is not None and value < minimum)
    ):
        raise ConfigurationError(f"{name} must be a finite number"
                                 + (f" >= {minimum}" if minimum is not None else ""))
    return float(value)


def _threshold(tau: Any) -> float:
    value = _finite(tau, "tau")
    if not 0.0 < value <= 1.0:
        raise ConfigurationError("tau must be in (0, 1]")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a nonempty string")
    return value


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ConfigurationError(f"{name} must be a lowercase SHA256 digest")
    return value


def _freeze_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        return _finite(value, "provenance number")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ConfigurationError("provenance object keys must be strings")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    raise ConfigurationError("provenance must contain only finite JSON values")


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _record(cls: type, payload: Any) -> Any:
    if not isinstance(payload, Mapping):
        raise ConfigurationError(f"{cls.__name__} must be an object")
    expected = {item.name for item in fields(cls)}
    if set(payload) != expected:
        raise ConfigurationError(f"{cls.__name__} fields differ from the schema")
    return cls(**dict(payload))


@dataclass(frozen=True)
class SpectrumSource:
    """An original-matrix declaration, not proof of native capture/recapture.

    ``tensor_sha256``, when supplied, is checked against the canonical hash below.
    The result always includes the computed hash. A source identity and explicit
    retention classification remain necessary even when that hash is available.
    Study/model/prompt/seed/boundary/history/position identities belong in
    ``provenance`` and must be validated by the native capture/study protocol.
    """

    identity: str
    shape: tuple[int, int]
    dtype: str
    target: str
    key_space: str
    retention: str = "spectrum-only"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    tensor_sha256: str | None = None
    representation: str = "original"

    def __post_init__(self) -> None:
        _text(self.identity, "source identity")
        if not isinstance(self.shape, (tuple, list)) or len(self.shape) != 2:
            raise ConfigurationError("source shape must be [rows, full_channels]")
        shape = tuple(_integer(n, "source dimension", 1) for n in self.shape)
        if max(shape) > 2**53:
            raise ConfigurationError("source dimensions must be exactly representable in FP64")
        object.__setattr__(self, "shape", shape)
        if not isinstance(self.dtype, str) or self.dtype not in _DTYPES:
            raise ConfigurationError("source dtype must be float16/bfloat16/float32/float64")
        if self.target not in ("K", "V"):
            raise ConfigurationError("source target must be K or V")
        if self.key_space != ("pre_rope" if self.target == "K" else "none"):
            raise ConfigurationError("K requires original pre_rope; V requires direct/none")
        if self.representation != "original":
            raise ConfigurationError("only original matrices are valid spectrum sources")
        if not isinstance(self.retention, str) or self.retention not in _RETENTION:
            raise ConfigurationError("source retention classification is required")
        if not isinstance(self.provenance, Mapping):
            raise ConfigurationError("source provenance must be a JSON object")
        object.__setattr__(self, "provenance", _freeze_json(self.provenance))
        if self.tensor_sha256 is not None:
            _digest(self.tensor_sha256, "source tensor_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {item.name: _plain_json(getattr(self, item.name)) for item in fields(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SpectrumSource:
        return _record(cls, payload)


@dataclass(frozen=True)
class SpectrumNumerics:
    """Multiplier for dimension-aware FP64 roundoff bounds (cannot be disabled).

    With u=FP64 epsilon and gamma(k)=k*u/(1-k*u), b=max tile rows,
    t=tile count, s=rescalings, and C=channels, the absolute error budgets are:

    eigenvalue: E * safety * gamma(b + 2*t + 6*s + 8*C + 4)
    input norm: E * safety * gamma(b*C + 2*t + 6*s + 4)
    symmetry:   2*E * safety * gamma(b + 2*t + 6*s + 4)
    channel DC: 4*E * safety * gamma(b + 2*t + 6*s + C + 4)

    E is the independently reduced scaled input energy. Prefix uncertainty uses
    r times the eigenvalue budget (including prefix-addition roundoff), and the
    input norm's own denominator interval. These deliberately conservative
    engineering tolerances require representative direct FP64 SVD qualification.
    """

    safety_factor: float = 8.0

    def __post_init__(self) -> None:
        value = _finite(self.safety_factor, "safety_factor", 1.0)
        if value > 1024:
            raise ConfigurationError("safety_factor must be <= 1024")
        object.__setattr__(self, "safety_factor", value)

    def to_dict(self) -> dict[str, Any]:
        return {"safety_factor": self.safety_factor}


def _gamma(count: int) -> float:
    value = count * _EPS
    if value >= 0.01:
        raise SpectrumNumericalError("dimensions exceed the FP64 error model's useful range")
    return value / (1.0 - value)


def _budgets(
    channels: int, tile_rows: int, tiles: int, rescalings: int,
    energy: float, numerics: SpectrumNumerics,
) -> dict[str, float]:
    base = tile_rows + 2 * tiles + 6 * rescalings + 4
    factor = numerics.safety_factor
    relative = {
        "eigenvalue_error_bound_scaled": factor * _gamma(base + 8 * channels),
        "norm_error_bound_scaled": factor * _gamma(
            tile_rows * channels + 2 * tiles + 6 * rescalings + 4
        ),
        "symmetry_tolerance_scaled": 2 * factor * _gamma(base),
        "mean_error_bound_scaled": 4 * factor * _gamma(base + channels),
    }
    if max(relative.values()) >= 0.01:
        raise SpectrumNumericalError("FP64 error budget is too large to certify this input")
    result = {name: energy * value for name, value in relative.items()}
    result["trace_tolerance_scaled"] = (
        result["norm_error_bound_scaled"]
        + (channels + 1) * result["eigenvalue_error_bound_scaled"]
    )
    result["spectrum_trace_tolerance_scaled"] = (
        (channels + 1) * result["eigenvalue_error_bound_scaled"]
    )
    return result


@dataclass(frozen=True)
class SpectrumDiagnostics:
    tile_rows: int
    tiles: int
    max_tile_rows: int
    rescalings: int
    gram_trace_scaled: float
    raw_eigenvalue_sum_scaled: float
    symmetry_error_scaled: float
    minimum_raw_eigenvalue_scaled: float
    negative_eigenvalue_mass_scaled: float
    structural_zero_tail_mass_scaled: float
    eigenvalue_error_bound_scaled: float
    norm_error_bound_scaled: float
    symmetry_tolerance_scaled: float
    mean_error_bound_scaled: float
    trace_tolerance_scaled: float
    spectrum_trace_tolerance_scaled: float
    direction_orthogonality_error: float | None
    direction_residual_scaled: float | None
    workspace_estimate_bytes: int
    workspace_limit_bytes: int

    def __post_init__(self) -> None:
        integer_fields = {
            "tile_rows", "tiles", "max_tile_rows", "rescalings",
            "workspace_estimate_bytes", "workspace_limit_bytes",
        }
        signed_fields = {"minimum_raw_eigenvalue_scaled", "raw_eigenvalue_sum_scaled"}
        optional_fields = {"direction_orthogonality_error", "direction_residual_scaled"}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in integer_fields:
                _integer(value, item.name, 0 if item.name == "rescalings" else 1)
            elif item.name in optional_fields and value is None:
                continue
            else:
                object.__setattr__(
                    self, item.name,
                    _finite(value, item.name, None if item.name in signed_fields else 0.0),
                )
        if self.max_tile_rows > self.tile_rows or self.rescalings >= self.tiles:
            raise ConfigurationError("inconsistent tile/rescaling diagnostics")
        if self.workspace_estimate_bytes > self.workspace_limit_bytes:
            raise CapacityError("spectrum workspace exceeds its admitted budget")

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(frozen=True)
class SpectrumSolver:
    algorithm: str
    torch_version: str
    backend_config_sha256: str
    num_threads: int
    num_interop_threads: int
    deterministic_algorithms: bool
    directions_requested: bool
    device: str = "cpu"
    compute_dtype: str = "float64"
    decomposition_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm, str) or self.algorithm not in {
            "torch.linalg.eigvalsh", "torch.linalg.eigh", "torch.linalg.svdvals",
        }:
            raise ConfigurationError("unrecognized spectrum solver")
        _text(self.torch_version, "torch_version")
        _digest(self.backend_config_sha256, "backend_config_sha256")
        _integer(self.num_threads, "num_threads", 1)
        _integer(self.num_interop_threads, "num_interop_threads", 1)
        if type(self.deterministic_algorithms) is not bool or type(self.directions_requested) is not bool:
            raise ConfigurationError("solver flags must be booleans")
        if self.device != "cpu" or self.compute_dtype != "float64":
            raise ConfigurationError("the oracle solver must use CPU FP64")
        if type(self.decomposition_count) is not int or self.decomposition_count != 1:
            raise ConfigurationError("one oracle result must use exactly one decomposition")
        if self.directions_requested != (self.algorithm == "torch.linalg.eigh"):
            raise ConfigurationError("directions require the same full eigh decomposition")

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @property
    def eigh_count(self) -> int:
        return int(self.algorithm == "torch.linalg.eigh")

    @property
    def eigvalsh_count(self) -> int:
        return int(self.algorithm == "torch.linalg.eigvalsh")

    @property
    def svd_count(self) -> int:
        return int(self.algorithm == "torch.linalg.svdvals")

    def work_counts(self) -> dict[str, int]:
        return {
            "decomposition_count": self.decomposition_count,
            "eigh_count": self.eigh_count,
            "eigvalsh_count": self.eigvalsh_count,
            "svd_count": self.svd_count,
        }


@dataclass(frozen=True)
class SpectrumTimings:
    """Successful-call telemetry, separate from reproducible numerical JSON.

    Hash-update and Gram-matmul/add times are subsets of accumulation.
    Eigensolve, direction canonicalization, and verification are subsets of
    finalization, not additional charges. Hash header/digest work remains in
    initialization/finalization; scaling, copies, norm/mean reductions and Gram
    rescaling remain in the unsplit part of accumulation.
    Initialization, update calls, and finalization exclude caller idle time
    between streamed shards. Callers must additionally measure end-to-end calls,
    source capture/staging, output persistence, cleanup, and failed attempts.
    """

    initialization_seconds: float
    accumulation_seconds: float
    eigensolve_seconds: float
    finalization_seconds: float
    hash_update_seconds: float = 0.0
    gram_matmul_add_seconds: float = 0.0
    direction_canonicalization_seconds: float = 0.0
    direction_verification_seconds: float = 0.0

    def __post_init__(self) -> None:
        for item in fields(self):
            object.__setattr__(
                self, item.name, _finite(getattr(self, item.name), item.name, 0.0),
            )
        if self.eigensolve_seconds > self.finalization_seconds:
            raise ConfigurationError("eigensolve time must be included in finalization time")
        if self.hash_update_seconds + self.gram_matmul_add_seconds > self.accumulation_seconds:
            raise ConfigurationError("hash and Gram timings must be included in accumulation")
        if (
            self.eigensolve_seconds + self.direction_canonicalization_seconds
            + self.direction_verification_seconds > self.finalization_seconds
        ):
            raise ConfigurationError("eigh and direction timings must be included in finalization")
        _finite(self.accounted_seconds, "accounted_seconds", 0.0)

    @property
    def accounted_seconds(self) -> float:
        return self.initialization_seconds + self.accumulation_seconds + self.finalization_seconds

    def to_dict(self) -> dict[str, float]:
        return {
            **{item.name: getattr(self, item.name) for item in fields(self)},
            "accounted_seconds": self.accounted_seconds,
        }


@lru_cache(maxsize=1)
def _backend_hash() -> str:
    return hashlib.sha256(torch.__config__.show().encode("utf-8")).hexdigest()


def _solver(algorithm: str, directions: bool = False) -> SpectrumSolver:
    return SpectrumSolver(
        algorithm=algorithm, torch_version=str(torch.__version__),
        backend_config_sha256=_backend_hash(), num_threads=torch.get_num_threads(),
        num_interop_threads=torch.get_num_interop_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        directions_requested=directions,
    )


@dataclass(frozen=True)
class RankBracket:
    tau: float
    lower: int
    upper: int
    rank: int | None
    status: str
    cdf_before_lower: float
    cdf_at_upper: float
    cdf_before_lower_bounds: tuple[float, float]
    cdf_at_upper_bounds: tuple[float, float]
    complete_endpoint_used: bool
    diagnostic: str

    def __post_init__(self) -> None:
        _threshold(self.tau)
        _integer(self.lower, "rank lower")
        _integer(self.upper, "rank upper")
        if self.lower > self.upper:
            raise ConfigurationError("rank interval is reversed")
        if not isinstance(self.status, str) or self.status not in {"exact", "interval", "zero"}:
            raise ConfigurationError("unknown rank status")
        expected = self.lower if self.lower == self.upper else None
        if self.rank != expected or (self.rank is not None and type(self.rank) is not int):
            raise ConfigurationError("only a singleton rank interval may be exact")
        if (
            (self.status == "zero") != (self.lower == self.upper == 0)
            or (self.status == "interval") != (self.lower < self.upper)
        ):
            raise ConfigurationError("rank status differs from its interval")
        for name in ("cdf_before_lower", "cdf_at_upper"):
            _finite(getattr(self, name), name, 0.0)
        for name in ("cdf_before_lower_bounds", "cdf_at_upper_bounds"):
            bounds = getattr(self, name)
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ConfigurationError("CDF bounds must have two endpoints")
            low, high = (_finite(value, name, 0.0) for value in bounds)
            if not low <= high <= 1.0:
                raise ConfigurationError("invalid CDF interval")
            object.__setattr__(self, name, (low, high))
        if type(self.complete_endpoint_used) is not bool:
            raise ConfigurationError("complete_endpoint_used must be boolean")
        _text(self.diagnostic, "rank diagnostic")
        if self.status != "zero" and not (
            self.cdf_before_lower_bounds[1] < self.tau
            <= self.cdf_at_upper_bounds[0]
        ):
            raise SpectrumNumericalError("rank interval does not certify its minimal bracket")

    def to_dict(self) -> dict[str, Any]:
        return {item.name: _plain_json(getattr(self, item.name)) for item in fields(self)}


def _prefixes(values: tuple[float, ...]) -> list[float]:
    total = correction = 0.0
    result = [0.0]
    for value in values:
        adjusted = value - correction
        updated = total + adjusted
        correction = (updated - total) - adjusted
        total = updated
        result.append(total)
    return result


def _rank_bracket(
    values: tuple[float, ...], energy: float, max_rank: int,
    eigen_error: float, norm_error: float, tau: float,
) -> RankBracket:
    tau = _threshold(tau)
    if energy == 0.0:
        return RankBracket(
            tau, 0, 0, 0, "zero", 0.0, 0.0, (0.0, 0.0), (0.0, 0.0),
            False, "zero source: mathematical rank 0, not a lossy CodecSpec rank",
        )
    sums = _prefixes(values)
    lower_cdf, upper_cdf = [0.0], [0.0]
    for rank in range(1, max_rank):
        low = max(0.0, sums[rank] - rank * eigen_error) / (energy + norm_error)
        high = (sums[rank] + rank * eigen_error) / (energy - norm_error)
        low = max(0.0, math.nextafter(low, -math.inf))
        high = math.nextafter(high, math.inf)
        if low > 1.0:
            raise SpectrumNumericalError("CDF lower bound exceeds the independent whole-input energy")
        lower_cdf.append(max(lower_cdf[-1], low))
        upper_cdf.append(min(1.0, max(upper_cdf[-1], high)))
    lower_cdf.append(1.0)
    upper_cdf.append(1.0)
    lower = next(rank for rank in range(1, max_rank + 1) if upper_cdf[rank] >= tau)
    upper = next(rank for rank in range(1, max_rank + 1) if lower_cdf[rank] >= tau)
    exact = lower == upper
    endpoint = upper == max_rank
    nominal_before, nominal_at = sums[lower - 1] / energy, sums[upper] / energy
    if exact and not endpoint and not nominal_before < tau <= nominal_at:
        raise SpectrumNumericalError("exact rank fails the unnormalized nominal CDF bracket")
    return RankBracket(
        tau=tau, lower=lower, upper=upper, rank=lower if exact else None,
        status="exact" if exact else "interval",
        cdf_before_lower=nominal_before, cdf_at_upper=nominal_at,
        cdf_before_lower_bounds=(lower_cdf[lower - 1], upper_cdf[lower - 1]),
        cdf_at_upper_bounds=(lower_cdf[upper], upper_cdf[upper]),
        complete_endpoint_used=endpoint,
        diagnostic=(
            "numerical uncertainty straddles threshold; use upper rank or verify original source"
            if not exact else (
                "certified minimal bracket; full-rank endpoint follows from complete channels"
                if endpoint else "certified minimal bracket under the declared FP64 error model"
            )
        ),
    )


def _workspace_bytes(shape: tuple[int, int], tile_rows: int, *, direct_svd: bool = False) -> int:
    rows, channels = shape
    tiled = 4 * min(rows, tile_rows) * channels
    # Includes Gram copies, vectors, symmetric scratch, LAPACK work, and tile copies.
    doubles = 12 * channels * channels + tiled + 64 * channels
    if direct_svd:
        doubles += 4 * rows * channels
    return 8 * doubles


def _admit(shape: tuple[int, int], tile_rows: int, limit: int, *, direct_svd: bool = False) -> int:
    _integer(tile_rows, "tile_rows", 1)
    _integer(limit, "max_workspace_bytes", 1)
    estimate = _workspace_bytes(shape, tile_rows, direct_svd=direct_svd)
    if estimate > limit:
        raise CapacityError(f"oracle workspace estimate {estimate} exceeds budget {limit} bytes")
    return estimate


@dataclass(frozen=True)
class SpectrumResult:
    source: SpectrumSource
    eigenvalues_scaled: tuple[float, ...]
    energy_scaled: float
    scale_exponent: int
    mean_energy_scaled: float
    diagnostics: SpectrumDiagnostics
    numerics: SpectrumNumerics
    solver: SpectrumSolver
    directions: torch.Tensor | None = field(default=None, repr=False, compare=False)
    timings: SpectrumTimings | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, SpectrumSource) or self.source.tensor_sha256 is None:
            raise ConfigurationError("result requires an original-source declaration and checked digest")
        if not isinstance(self.diagnostics, SpectrumDiagnostics):
            raise ConfigurationError("result diagnostics must be SpectrumDiagnostics")
        if not isinstance(self.numerics, SpectrumNumerics) or not isinstance(self.solver, SpectrumSolver):
            raise ConfigurationError("result requires validated numerical/solver identities")
        if self.timings is not None and not isinstance(self.timings, SpectrumTimings):
            raise ConfigurationError("result timings must be SpectrumTimings when present")
        if self.solver.algorithm not in {"torch.linalg.eigvalsh", "torch.linalg.eigh"}:
            raise ConfigurationError("primary results require the full channel Gram solver")
        if type(self.scale_exponent) is not int or not -1074 <= self.scale_exponent <= 1023:
            raise ConfigurationError("invalid source power-of-two scale exponent")
        energy = _finite(self.energy_scaled, "energy_scaled", 0.0)
        mean = _finite(self.mean_energy_scaled, "mean_energy_scaled", 0.0)
        object.__setattr__(self, "energy_scaled", energy)
        object.__setattr__(self, "mean_energy_scaled", mean)
        if not isinstance(self.eigenvalues_scaled, (tuple, list)):
            raise ConfigurationError("complete eigenvalues must be a sequence")
        values = tuple(_finite(value, "eigenvalue", 0.0) for value in self.eigenvalues_scaled)
        object.__setattr__(self, "eigenvalues_scaled", values)
        rows, channels = self.source.shape
        if len(values) != channels or any(a < b for a, b in zip(values, values[1:])):
            raise ConfigurationError("exactly C descending eigenvalues are required")
        if any(values[min(rows, channels):]):
            raise SpectrumNumericalError("N<C requires C-N explicit structural zero directions")
        diag = self.diagnostics
        if not (
            1 <= diag.max_tile_rows <= min(rows, diag.tile_rows)
            and math.ceil(rows / diag.max_tile_rows) <= diag.tiles <= rows
        ):
            raise ConfigurationError("tile counts cannot cover the declared source")
        estimate = _workspace_bytes(self.source.shape, diag.tile_rows)
        if diag.workspace_estimate_bytes != estimate:
            raise ConfigurationError("workspace diagnostics differ from source geometry")
        expected_budgets = _budgets(
            channels, diag.max_tile_rows, diag.tiles, diag.rescalings, energy, self.numerics,
        )
        if any(getattr(diag, name) != value for name, value in expected_budgets.items()):
            raise ConfigurationError("numerical bounds differ from the declared FP64 model")
        if energy == 0.0:
            if any(values) or mean or diag.gram_trace_scaled or self.scale_exponent != 0:
                raise SpectrumNumericalError("inconsistent zero-energy result")
        elif energy < 1.0:
            raise SpectrumNumericalError("nonzero scaled input must contain a value of magnitude >= 1")
        if mean > energy + diag.mean_error_bound_scaled + diag.norm_error_bound_scaled:
            raise SpectrumNumericalError("channel-mean/DC energy exceeds whole-input energy")
        if diag.symmetry_error_scaled > diag.symmetry_tolerance_scaled:
            raise SpectrumNumericalError("Gram has material asymmetry")
        if abs(diag.gram_trace_scaled - energy) > diag.trace_tolerance_scaled:
            raise SpectrumNumericalError("Gram trace differs from the independent full-input norm")
        if abs(diag.raw_eigenvalue_sum_scaled - diag.gram_trace_scaled) > diag.spectrum_trace_tolerance_scaled:
            raise SpectrumNumericalError("complete eigenvalue sum differs from Gram trace")
        if (
            diag.minimum_raw_eigenvalue_scaled < -diag.eigenvalue_error_bound_scaled
            or diag.negative_eigenvalue_mass_scaled > channels * diag.eigenvalue_error_bound_scaled
        ):
            raise SpectrumNumericalError("Gram has material negative eigenvalue mass")
        if diag.structural_zero_tail_mass_scaled > max(0, channels - rows) * diag.eigenvalue_error_bound_scaled:
            raise SpectrumNumericalError("material energy appeared in structural zero directions")
        direction_checks = (diag.direction_orthogonality_error, diag.direction_residual_scaled)
        if self.solver.directions_requested:
            if any(value is None for value in direction_checks):
                raise ConfigurationError("requested directions require eigensystem checks")
            ortho_tolerance = self.numerics.safety_factor * _gamma(8 * channels + 4) * math.sqrt(channels)
            if diag.direction_orthogonality_error > ortho_tolerance:
                raise SpectrumNumericalError("eigenvector directions are not orthonormal")
            if diag.direction_residual_scaled > 2 * math.sqrt(channels) * diag.eigenvalue_error_bound_scaled:
                raise SpectrumNumericalError("eigenvector residual exceeds the FP64 eigenvalue budget")
        elif any(value is not None for value in direction_checks):
            raise ConfigurationError("values-only solve cannot claim eigenvector verification")
        corrected_sum = (
            diag.raw_eigenvalue_sum_scaled + diag.negative_eigenvalue_mass_scaled
            - diag.structural_zero_tail_mass_scaled
        )
        if abs(math.fsum(values) - corrected_sum) > 8 * _EPS * channels * energy:
            raise SpectrumNumericalError("serialized spectrum differs from recorded clipping corrections")
        if self.directions is not None:
            directions = self.directions
            if (
                not self.solver.directions_requested or not isinstance(directions, torch.Tensor)
                or directions.device.type != "cpu" or directions.dtype != torch.float64
                or directions.layout != torch.strided or tuple(directions.shape) != (channels, channels)
                or not bool(torch.isfinite(directions).all())
            ):
                raise ConfigurationError("directions must be the full finite FP64 CPU row basis")
        self.rank_at(0.95)
        self.rank_at(0.99)

    @property
    def zero_energy(self) -> bool:
        return self.energy_scaled == 0.0

    @property
    def dc_energy_fraction(self) -> float:
        """Unclamped nominal DC share; uncertainty involves both mean and norm budgets."""
        return 0.0 if self.zero_energy else self.mean_energy_scaled / self.energy_scaled

    @property
    def r95(self) -> int | None:
        return self.rank_at(0.95).rank

    @property
    def r99(self) -> int | None:
        return self.rank_at(0.99).rank

    def work_counts(self) -> dict[str, int]:
        """Successful per-matrix logical operations, not tile/kernel counts."""
        return {
            **self.solver.work_counts(),
            "gram_build_count": 1,
            "source_hash_count": 1,
            "direction_verification_count": int(self.solver.directions_requested),
        }

    @property
    def energy(self) -> float | None:
        """Unscaled energy, or None if it overflows/underflows finite FP64."""
        try:
            value = math.ldexp(self.energy_scaled, 2 * self.scale_exponent)
        except OverflowError:
            return None
        return None if value == 0.0 and not self.zero_energy else value

    def rank_at(self, tau: float) -> RankBracket:
        return _rank_bracket(
            self.eigenvalues_scaled, self.energy_scaled, min(self.source.shape),
            self.diagnostics.eigenvalue_error_bound_scaled,
            self.diagnostics.norm_error_bound_scaled, tau,
        )

    def to_dict(self) -> dict[str, Any]:
        """Reproducible JSON; optional directions and separate timings are omitted."""
        return {
            "schema_version": SCHEMA_VERSION,
            "source": self.source.to_dict(),
            "hash_format": HASH_FORMAT,
            "hash_byte_order": sys.byteorder,
            "energy_definition": "uncentered_full_channel_squared_singular_values",
            "scale_exponent": self.scale_exponent,
            "unscale_energy_exponent": 2 * self.scale_exponent,
            "eigenvalues_scaled": list(self.eigenvalues_scaled),
            "energy_scaled": self.energy_scaled,
            "energy": self.energy,
            "mean_energy_scaled": self.mean_energy_scaled,
            "dc_energy_fraction": self.dc_energy_fraction,
            "zero_energy": self.zero_energy,
            "r95": self.r95,
            "r99": self.r99,
            "rank_brackets": {str(tau): self.rank_at(tau).to_dict() for tau in (0.95, 0.99)},
            "diagnostics": self.diagnostics.to_dict(),
            "numerics": self.numerics.to_dict(),
            "solver": self.solver.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SpectrumResult:
        """Revalidate all arrays, bounds, ranks and derived fields, not just syntax."""
        if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
            raise ConfigurationError("unsupported spectrum observation schema")
        try:
            result = cls(
                source=SpectrumSource.from_dict(payload["source"]),
                eigenvalues_scaled=payload["eigenvalues_scaled"],
                energy_scaled=payload["energy_scaled"],
                scale_exponent=payload["scale_exponent"],
                mean_energy_scaled=payload["mean_energy_scaled"],
                diagnostics=_record(SpectrumDiagnostics, payload["diagnostics"]),
                numerics=_record(SpectrumNumerics, payload["numerics"]),
                solver=_record(SpectrumSolver, payload["solver"]),
            )
        except KeyError as exc:
            raise ConfigurationError(f"missing spectrum field: {exc.args[0]}") from exc
        # Canonical JSON comparison rejects bool/int aliases, stale ranks, extra
        # fields and nonfinite numbers, while accepting the standard JSON types.
        try:
            supplied = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))
            expected = json.dumps(result.to_dict(), sort_keys=True, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("spectrum observation must contain finite JSON values") from exc
        if supplied != expected:
            raise ConfigurationError("spectrum observation has inconsistent or noncanonical derived fields")
        return result


def write_spectrum_json(result: SpectrumResult, path: str | os.PathLike[str]) -> str:
    """Durably publish one canonical full-spectrum JSON file without overwriting.

    The caller supplies an admitted, owned output path in an existing directory.
    A same-directory staging file is fsynced and hard-linked into place atomically;
    no partial JSON is published, and existing files/symlinks are never replaced.
    The returned SHA256 covers the exact canonical UTF-8 file bytes (no newline),
    not the original tensor or a model/context hash. Directions, timing telemetry,
    and original source tensors are not retained; callers persist separate work
    records and selected original verification captures as required.

    Failures propagate. A directory-fsync failure after publication can leave a
    complete file but never returns a success receipt; no existing output is
    removed to disguise a failure. This durability contract requires a filesystem
    supporting same-directory hard links and directory fsync.
    """
    if not isinstance(result, SpectrumResult):
        raise ConfigurationError("result must be a validated SpectrumResult")
    if not isinstance(path, (str, os.PathLike)) or (isinstance(path, str) and not path):
        raise ConfigurationError("an explicit spectrum output file path is required")
    target = Path(path)
    if target.name in {"", ".", ".."}:
        raise ConfigurationError("spectrum output must name a file")
    encoded = json.dumps(
        result.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    staged = target.with_name(
        f".{target.name}.part-{os.getpid()}-{time.monotonic_ns()}"
    )
    descriptor = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            if stream.write(encoded) != len(encoded):
                raise OSError("incomplete spectrum JSON write")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, target, follow_symlinks=False)
        directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return digest
    finally:
        staged.unlink(missing_ok=True)


@dataclass(frozen=True)
class SpectrumRowShard:
    """Optional explicit row offset; channels must never be partitioned."""

    row_start: int
    tensor: torch.Tensor

    def __post_init__(self) -> None:
        _integer(self.row_start, "row_start")


def _validate_tile(tensor: Any, source: SpectrumSource) -> None:
    if (
        not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
        or tensor.device.type == "meta" or tensor.ndim != 2
        or tensor.shape[0] < 1 or tensor.shape[1] != source.shape[1]
    ):
        raise ConfigurationError("row tiles must be nonempty dense [rows, full_channels] tensors")
    if tensor.dtype != _DTYPES[source.dtype]:
        raise ConfigurationError("row-tile dtype differs from the original declared source dtype")


def _new_hasher(source: SpectrumSource) -> Any:
    header = json.dumps(
        {"format": HASH_FORMAT, "shape": list(source.shape), "dtype": source.dtype,
         "byte_order": sys.byteorder}, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256((header + "\n").encode("utf-8"))


def _native_tile(tensor: torch.Tensor) -> torch.Tensor:
    native = tensor.detach().to(device="cpu").contiguous()
    if not bool(torch.isfinite(native).all()):
        raise ConfigurationError("original spectrum input must contain only finite values")
    return native


def _hash_tile(hasher: Any, native: torch.Tensor) -> None:
    hasher.update(memoryview(native.view(torch.uint8).numpy()))


def _canonical_directions(vectors: torch.Tensor) -> torch.Tensor:
    rows = vectors.flip(1).T.contiguous()
    pivots = rows.gather(1, rows.abs().argmax(dim=1, keepdim=True))
    rows.mul_(torch.where(pivots < 0, -1.0, 1.0))
    return rows


class StreamingGramAccumulator:
    """Single-use full-channel row stream; finalize releases its Gram workspace.

    ``update`` splits large input shards into ``tile_rows`` slices before any
    dtype/device copy. Device inputs are synchronously copied one tile at a time;
    the Gram/solver themselves are always CPU FP64. No source tiles are retained.
    Explicit ``row_start`` rejects gaps, duplicates and reordering. Without it,
    the caller attests that iteration order is original canonical row order.
    """

    def __init__(
        self, source: SpectrumSource, *, tile_rows: int = 1024,
        numerics: SpectrumNumerics | None = None,
        max_workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
    ) -> None:
        initialized_at = time.perf_counter()
        if not isinstance(source, SpectrumSource):
            raise ConfigurationError("source must be a SpectrumSource")
        self.source = source
        self.numerics = SpectrumNumerics() if numerics is None else numerics
        if not isinstance(self.numerics, SpectrumNumerics):
            raise ConfigurationError("numerics must be SpectrumNumerics")
        self._workspace = _admit(source.shape, tile_rows, max_workspace_bytes)
        self.tile_rows = tile_rows
        self.max_workspace_bytes = max_workspace_bytes
        self._gram: torch.Tensor | None = torch.zeros(
            (source.shape[1], source.shape[1]), dtype=torch.float64, device="cpu",
        )
        self._channel_sum: torch.Tensor | None = torch.zeros(source.shape[1], dtype=torch.float64)
        self._hash = _new_hasher(source)
        self._energy = self._energy_correction = 0.0
        self._scale_exponent: int | None = None
        self._rows = self._tiles = self._max_tile_rows = self._rescalings = 0
        self._closed = False
        self._accumulation_seconds = 0.0
        self._hash_update_seconds = self._gram_matmul_add_seconds = 0.0
        self._eigh_count = self._eigvalsh_count = 0
        self._eigensolve_seconds = 0.0
        self._initialization_seconds = time.perf_counter() - initialized_at

    def _annotate_failure(
        self, error: SpectrumNumericalError, phase: str, *,
        finalization_seconds: float = 0.0, pending_accumulation_seconds: float = 0.0,
    ) -> None:
        accumulation = self._accumulation_seconds + pending_accumulation_seconds
        exclusive = (
            self._initialization_seconds + accumulation
            + max(0.0, finalization_seconds - self._eigensolve_seconds)
        )
        error._metrics = {
            "operation_counts_complete": True,
            "decomposition_count": self._eigh_count + self._eigvalsh_count,
            "eigh_count": self._eigh_count,
            "eigvalsh_count": self._eigvalsh_count,
            "svd_count": 0,
            "failure_phase": phase,
            "initialization_seconds": self._initialization_seconds,
            "accumulation_seconds": accumulation,
            "finalization_seconds": finalization_seconds,
            "eigensolve_seconds": self._eigensolve_seconds,
            "eigh_seconds": self._eigensolve_seconds if self._eigh_count else 0.0,
            "eigvalsh_seconds": self._eigensolve_seconds if self._eigvalsh_count else 0.0,
            "exclusive_pipeline_seconds": exclusive,
            "total_seconds": exclusive + self._eigensolve_seconds,
            "workspace_admission_bytes": self._workspace,
            "workspace_limit_bytes": self.max_workspace_bytes,
        }

    def close(self) -> None:
        self._closed = True
        self._gram = self._channel_sum = None
        self._hash = None

    def __enter__(self) -> StreamingGramAccumulator:
        if self._closed:
            raise StateError("spectrum accumulator is closed")
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @torch.no_grad()
    def update(self, tensor: torch.Tensor, *, row_start: int | None = None) -> None:
        if self._closed:
            raise StateError("spectrum accumulator is closed")
        started_at = time.perf_counter()
        try:
            _validate_tile(tensor, self.source)
            if row_start is not None:
                _integer(row_start, "row_start")
                if row_start != self._rows:
                    raise ConfigurationError("row stream has a gap, duplicate, or out-of-order shard")
            if self._rows + tensor.shape[0] > self.source.shape[0]:
                raise ConfigurationError("row stream exceeds the declared complete source")
            for start in range(0, tensor.shape[0], self.tile_rows):
                native = _native_tile(tensor[start:start + self.tile_rows])
                hash_started_at = time.perf_counter()
                _hash_tile(self._hash, native)
                self._hash_update_seconds += time.perf_counter() - hash_started_at
                values = native.to(dtype=torch.float64, copy=True)
                maximum = max(abs(float(values.min())), abs(float(values.max())))
                if maximum:
                    exponent = math.frexp(maximum)[1] - 1
                    if self._scale_exponent is None:
                        self._scale_exponent = exponent
                    elif exponent > self._scale_exponent:
                        ratio = math.ldexp(1.0, self._scale_exponent - exponent)
                        self._gram.mul_(ratio).mul_(ratio)
                        self._channel_sum.mul_(ratio)
                        self._energy *= ratio
                        self._energy *= ratio
                        self._energy_correction *= ratio
                        self._energy_correction *= ratio
                        self._scale_exponent = exponent
                        self._rescalings += 1
                    values.div_(math.ldexp(1.0, self._scale_exponent))
                    gram_started_at = time.perf_counter()
                    self._gram.add_(values.T @ values)
                    self._gram_matmul_add_seconds += time.perf_counter() - gram_started_at
                    self._channel_sum.add_(values.sum(dim=0))
                    independent_energy = float(values.square().sum())
                    adjusted = independent_energy - self._energy_correction
                    updated = self._energy + adjusted
                    self._energy_correction = (updated - self._energy) - adjusted
                    self._energy = updated
                self._rows += native.shape[0]
                self._tiles += 1
                self._max_tile_rows = max(self._max_tile_rows, native.shape[0])
        except Exception as exc:
            if isinstance(exc, SpectrumNumericalError):
                self._annotate_failure(
                    exc, "accumulation",
                    pending_accumulation_seconds=time.perf_counter() - started_at,
                )
            self.close()
            raise
        finally:
            self._accumulation_seconds += time.perf_counter() - started_at

    @torch.no_grad()
    def finalize(self, *, return_directions: bool = False) -> SpectrumResult:
        if self._closed:
            raise StateError("spectrum accumulator is closed")
        started_at = time.perf_counter()
        phase = "source_validation"
        try:
            if type(return_directions) is not bool:
                raise ConfigurationError("return_directions must be a boolean")
            if self._rows != self.source.shape[0]:
                raise ConfigurationError("row stream is incomplete; no row sampling is permitted")
            digest = self._hash.hexdigest()
            if self.source.tensor_sha256 is not None and digest != self.source.tensor_sha256:
                raise IntegrityError("original-source tensor hash mismatch")
            source = replace(self.source, tensor_sha256=digest)
            gram = self._gram
            phase = "gram_validation"
            if not bool(torch.isfinite(gram).all()) or not math.isfinite(self._energy):
                raise SpectrumNumericalError("nonfinite accumulated Gram or independent norm")
            budgets = _budgets(
                source.shape[1], self._max_tile_rows, self._tiles,
                self._rescalings, self._energy, self.numerics,
            )
            trace = float(gram.diagonal().sum())
            symmetry = float(torch.linalg.vector_norm(gram - gram.T))
            if symmetry > budgets["symmetry_tolerance_scaled"]:
                raise SpectrumNumericalError("Gram has material asymmetry")
            if abs(trace - self._energy) > budgets["trace_tolerance_scaled"]:
                raise SpectrumNumericalError("Gram trace differs from the independent full-input norm")
            symmetric = (gram + gram.T) * 0.5
            try:
                phase = "eigensolve"
                eigensolve_started_at = time.perf_counter()
                try:
                    if return_directions:
                        self._eigh_count += 1
                        ascending, vectors = torch.linalg.eigh(symmetric)
                        algorithm = "torch.linalg.eigh"
                    else:
                        self._eigvalsh_count += 1
                        ascending = torch.linalg.eigvalsh(symmetric)
                        algorithm = "torch.linalg.eigvalsh"
                finally:
                    self._eigensolve_seconds += time.perf_counter() - eigensolve_started_at
                eigensolve_seconds = self._eigensolve_seconds
                phase = "direction_canonicalization"
                if return_directions:
                    canonicalization_started_at = time.perf_counter()
                    directions = _canonical_directions(vectors)
                    direction_canonicalization_seconds = time.perf_counter() - canonicalization_started_at
                else:
                    directions = None
                    direction_canonicalization_seconds = 0.0
            except RuntimeError as exc:
                raise SpectrumNumericalError("complete FP64 eigensolver failed") from exc
            phase = "spectrum_validation"
            if (
                tuple(ascending.shape) != (source.shape[1],)
                or ascending.dtype != torch.float64 or ascending.device.type != "cpu"
                or not bool(torch.isfinite(ascending).all())
                or bool((ascending[1:] < ascending[:-1]).any())
            ):
                raise SpectrumNumericalError("eigensolver returned an incomplete/nonfinite spectrum")
            raw = ascending.flip(0)
            eigen_error = budgets["eigenvalue_error_bound_scaled"]
            negative_mass = float((-raw.clamp_max(0)).sum())
            if (
                float(raw.min()) < -eigen_error
                or negative_mass > source.shape[1] * eigen_error
            ):
                raise SpectrumNumericalError("Gram has material negative eigenvalue mass")
            corrected = raw.clamp_min(0)
            structural_tail = corrected[min(source.shape):]
            structural_mass = float(structural_tail.sum())
            if structural_tail.numel() and float(structural_tail.max()) > eigen_error:
                raise SpectrumNumericalError("material energy in N<C structural null directions")
            structural_tail.zero_()
            if directions is None:
                orthogonality = direction_residual = None
                direction_verification_seconds = 0.0
            else:
                phase = "direction_verification"
                verification_started_at = time.perf_counter()
                identity = torch.eye(source.shape[1], dtype=torch.float64, device="cpu")
                orthogonality = float(torch.linalg.vector_norm(directions @ directions.T - identity))
                direction_residual = float(torch.linalg.vector_norm(
                    symmetric @ directions.T - directions.T * raw.unsqueeze(0),
                ))
                direction_verification_seconds = time.perf_counter() - verification_started_at
            phase = "result_validation"
            diag = SpectrumDiagnostics(
                tile_rows=self.tile_rows, tiles=self._tiles, max_tile_rows=self._max_tile_rows,
                rescalings=self._rescalings, gram_trace_scaled=trace,
                raw_eigenvalue_sum_scaled=math.fsum(raw.tolist()),
                symmetry_error_scaled=symmetry, minimum_raw_eigenvalue_scaled=float(raw.min()),
                negative_eigenvalue_mass_scaled=negative_mass,
                structural_zero_tail_mass_scaled=structural_mass,
                direction_orthogonality_error=orthogonality,
                direction_residual_scaled=direction_residual,
                workspace_estimate_bytes=self._workspace,
                workspace_limit_bytes=self.max_workspace_bytes, **budgets,
            )
            mean_energy = float(self._channel_sum.square().sum()) / source.shape[0]
            result = SpectrumResult(
                source=source, eigenvalues_scaled=tuple(corrected.tolist()),
                energy_scaled=self._energy,
                scale_exponent=0 if self._scale_exponent is None else self._scale_exponent,
                mean_energy_scaled=mean_energy, diagnostics=diag, numerics=self.numerics,
                solver=_solver(algorithm, return_directions), directions=directions,
            )
            object.__setattr__(result, "timings", SpectrumTimings(
                initialization_seconds=self._initialization_seconds,
                accumulation_seconds=self._accumulation_seconds,
                eigensolve_seconds=eigensolve_seconds,
                finalization_seconds=time.perf_counter() - started_at,
                hash_update_seconds=self._hash_update_seconds,
                gram_matmul_add_seconds=self._gram_matmul_add_seconds,
                direction_canonicalization_seconds=direction_canonicalization_seconds,
                direction_verification_seconds=direction_verification_seconds,
            ))
            return result
        except SpectrumNumericalError as exc:
            self._annotate_failure(
                exc, phase, finalization_seconds=time.perf_counter() - started_at,
            )
            raise
        finally:
            self.close()


def exact_spectrum(
    matrix: torch.Tensor, *, source: SpectrumSource, tile_rows: int = 1024,
    numerics: SpectrumNumerics | None = None, return_directions: bool = False,
    max_workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
) -> SpectrumResult:
    """Analyze one original matrix; optional directions share its one eigensolve."""
    if not isinstance(source, SpectrumSource):
        raise ConfigurationError("source must be a SpectrumSource")
    _validate_tile(matrix, source)
    if tuple(matrix.shape) != source.shape:
        raise ConfigurationError("matrix shape differs from the complete source shape")
    with StreamingGramAccumulator(
        source, tile_rows=tile_rows, numerics=numerics, max_workspace_bytes=max_workspace_bytes,
    ) as accumulator:
        accumulator.update(matrix, row_start=0)
        return accumulator.finalize(return_directions=return_directions)


def spectrum_from_row_shards(
    shards: Iterable[torch.Tensor | SpectrumRowShard], *, source: SpectrumSource,
    tile_rows: int = 1024, numerics: SpectrumNumerics | None = None,
    return_directions: bool = False, max_workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
) -> SpectrumResult:
    """Stream all full-channel MAGI-style row shards; never average shard ranks."""
    with StreamingGramAccumulator(
        source, tile_rows=tile_rows, numerics=numerics, max_workspace_bytes=max_workspace_bytes,
    ) as accumulator:
        for shard in shards:
            if isinstance(shard, SpectrumRowShard):
                accumulator.update(shard.tensor, row_start=shard.row_start)
            else:
                accumulator.update(shard)
        return accumulator.finalize(return_directions=return_directions)


@dataclass(frozen=True)
class SVDCrosscheckResult:
    source_sha256: str
    eigenvalues_scaled: tuple[float, ...]
    energy_scaled: float
    scale_exponent: int
    rank_brackets: tuple[RankBracket, RankBracket]
    maximum_energy_fraction_difference: float
    absolute_energy_tolerance: float
    relative_tolerance: float
    svd_eigenvalue_error_bound_scaled: float
    norm_error_bound_scaled: float
    workspace_estimate_bytes: int
    solver: SpectrumSolver

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "reelcache.spectrum-svd-crosscheck.v1",
            "passed": True, "source_sha256": self.source_sha256,
            "eigenvalues_scaled": list(self.eigenvalues_scaled),
            "energy_scaled": self.energy_scaled, "scale_exponent": self.scale_exponent,
            "rank_brackets": {str(item.tau): item.to_dict() for item in self.rank_brackets},
            "maximum_energy_fraction_difference": self.maximum_energy_fraction_difference,
            "absolute_energy_tolerance": self.absolute_energy_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "svd_eigenvalue_error_bound_scaled": self.svd_eigenvalue_error_bound_scaled,
            "norm_error_bound_scaled": self.norm_error_bound_scaled,
            "workspace_estimate_bytes": self.workspace_estimate_bytes,
            "solver": self.solver.to_dict(),
        }


@torch.no_grad()
def direct_svd_crosscheck(
    matrix: torch.Tensor, result: SpectrumResult, *,
    absolute_energy_tolerance: float = 1e-12, relative_tolerance: float = 1e-10,
    max_workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
) -> SVDCrosscheckResult:
    """Check identical original bytes against a *complete* direct CPU FP64 SVD.

    This deliberately allocates an admitted N-by-C FP64 copy and runs a separate
    decomposition. Use only on a bounded verification subset and ambiguous cases,
    not inside an online selector. Per-eigenvalue comparison allows
    ``atol*input_energy + rtol*svd_eigenvalue + Gram_bound + SVD_bound``.
    The SVD squared-value budget is ``E*(2*delta + delta**2)``, with
    ``delta=safety*gamma(8*max(N,C)+4)``. Overlapping rank intervals corroborate
    rather than magically resolve a threshold tie.
    """
    if not isinstance(result, SpectrumResult):
        raise ConfigurationError("crosscheck requires a validated SpectrumResult")
    atol = _finite(absolute_energy_tolerance, "absolute_energy_tolerance", 0.0)
    rtol = _finite(relative_tolerance, "relative_tolerance", 0.0)
    if max(atol, rtol) > 0.01:
        raise ConfigurationError("SVD comparison tolerances must be <= 0.01")
    source = result.source
    _validate_tile(matrix, source)
    if tuple(matrix.shape) != source.shape:
        raise ConfigurationError("SVD reference shape differs from the original source")
    tile_rows = result.diagnostics.tile_rows
    estimate = _admit(source.shape, tile_rows, max_workspace_bytes, direct_svd=True)
    copied = torch.empty(source.shape, dtype=torch.float64, device="cpu")
    hasher = _new_hasher(source)
    energies = []
    scale = math.ldexp(1.0, result.scale_exponent)
    for start in range(0, source.shape[0], tile_rows):
        native = _native_tile(matrix[start:start + tile_rows])
        _hash_tile(hasher, native)
        part = copied[start:start + native.shape[0]]
        part.copy_(native)
        part.div_(scale)
        energies.append(float(part.square().sum()))
    if hasher.hexdigest() != source.tensor_sha256:
        raise IntegrityError("direct SVD must use the identical original source bytes")
    energy = math.fsum(energies)
    norm_error = _budgets(
        source.shape[1], min(tile_rows, source.shape[0]), len(energies), 0,
        energy, result.numerics,
    )["norm_error_bound_scaled"]
    if (
        not math.isfinite(energy)
        or abs(energy - result.energy_scaled) > norm_error + result.diagnostics.norm_error_bound_scaled
    ):
        raise SpectrumNumericalError("direct-source norm differs from Gram oracle's independent norm")
    try:
        singular_values = torch.linalg.svdvals(copied)
    except RuntimeError as exc:
        raise SpectrumNumericalError("direct FP64 SVD failed") from exc
    if (
        tuple(singular_values.shape) != (min(source.shape),)
        or singular_values.dtype != torch.float64 or singular_values.device.type != "cpu"
        or not bool(torch.isfinite(singular_values).all())
        or bool((singular_values < 0).any())
        or bool((singular_values[1:] > singular_values[:-1]).any())
    ):
        raise SpectrumNumericalError("direct SVD returned incomplete/nonfinite singular values")
    values = tuple(singular_values.square().tolist()) + (0.0,) * max(0, source.shape[1] - source.shape[0])
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise SpectrumNumericalError("direct SVD squared spectrum is nonfinite")
    delta = result.numerics.safety_factor * _gamma(8 * max(source.shape) + 4)
    if delta >= 0.01:
        raise SpectrumNumericalError("direct SVD dimension-aware error budget is too large")
    svd_error = energy * (2 * delta + delta * delta)
    gram_error = result.diagnostics.eigenvalue_error_bound_scaled
    differences = [abs(left - right) for left, right in zip(values, result.eigenvalues_scaled)]
    if any(
        difference > atol * energy + rtol * reference + gram_error + svd_error
        for difference, reference in zip(differences, values)
    ):
        raise SpectrumNumericalError("direct FP64 SVD spectrum disagrees with channel Gram")
    if abs(math.fsum(values) - energy) > len(values) * svd_error + norm_error:
        raise SpectrumNumericalError("direct SVD complete spectrum disagrees with input norm")
    brackets = tuple(
        _rank_bracket(values, energy, min(source.shape), svd_error, norm_error, tau)
        for tau in (0.95, 0.99)
    )
    for bracket in brackets:
        gram_bracket = result.rank_at(bracket.tau)
        if max(bracket.lower, gram_bracket.lower) > min(bracket.upper, gram_bracket.upper):
            raise SpectrumNumericalError("direct SVD and Gram oracle rank intervals are disjoint")
    return SVDCrosscheckResult(
        source_sha256=source.tensor_sha256, eigenvalues_scaled=values,
        energy_scaled=energy, scale_exponent=result.scale_exponent,
        rank_brackets=brackets,
        maximum_energy_fraction_difference=0.0 if energy == 0 else max(differences) / energy,
        absolute_energy_tolerance=atol, relative_tolerance=rtol,
        svd_eigenvalue_error_bound_scaled=svd_error,
        norm_error_bound_scaled=norm_error,
        workspace_estimate_bytes=estimate, solver=_solver("torch.linalg.svdvals"),
    )
