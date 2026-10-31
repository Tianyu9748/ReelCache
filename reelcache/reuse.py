"""Explicit first-parent, projection-gated reuse of immutable FP16 bases.

The projection convention follows the historical temporal-reuse work, but the
gate measures the rounded basis's tiled direct residual rather than assuming
orthonormality. No prompt-only shortcut or newly fitted candidate is consulted.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from .errors import ConfigurationError
from .factorization import (
    Factors,
    Projection,
    basis_condition_bound,
    fit_basis,
    project_onto_basis,
    validate_matrix,
    validate_rank,
    validate_stored_basis,
)
from .types import TensorContext


@dataclass(frozen=True)
class ReuseDecision:
    accepted: bool
    reason: str
    projection: Projection | None
    seconds: float


def validate_threshold(threshold: float) -> float:
    if (
        isinstance(threshold, bool) or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold) or not 0 <= threshold <= 1
    ):
        raise ConfigurationError("reuse_threshold must be a finite relative residual in [0, 1]")
    return float(threshold)


@torch.no_grad()
def try_reuse_basis(
    tensor: torch.Tensor,
    context: TensorContext,
    basis: torch.Tensor,
    rank: int,
    threshold: float,
) -> ReuseDecision:
    """Return the *usable* projection on acceptance; never fit a candidate.

    A conservative FP32-sized boundary band rejects uncertain threshold cases.
    Exact zero-error and zero-input cases are explicit and accept even at a
    threshold of zero. Rank increases, nonfinite/singular bases and failed gates
    are observable rejects, not zero-padded success.
    """
    started = time.perf_counter()
    value = validate_matrix(tensor, context)
    validate_rank(rank, tuple(value.shape))
    limit = validate_threshold(threshold)

    def reject(reason: str, projection: Projection | None = None) -> ReuseDecision:
        return ReuseDecision(False, reason, projection, time.perf_counter() - started)

    if not isinstance(basis, torch.Tensor) or basis.dtype != torch.float16:
        raise ConfigurationError("reuse requires the stored FP16 basis, not a new FP32 candidate")
    if (
        basis.layout != torch.strided or basis.device.type == "meta"
        or basis.ndim != 2 or min(basis.shape) < 1
    ):
        raise ConfigurationError("reuse basis must be a nonempty dense matrix")
    if basis.shape[1] != value.shape[1] or basis.shape[0] > basis.shape[1]:
        return reject("incompatible_basis_geometry")
    if rank > basis.shape[0]:
        return reject("rank_increase_requires_refit")
    if not bool(torch.isfinite(basis).all()):
        return reject("nonfinite_basis")
    stored = validate_stored_basis(basis, value.shape[1])
    condition = basis_condition_bound(stored[:rank].to(device=value.device))
    if not math.isfinite(condition) or condition > 1e6:
        return reject("ill_conditioned_basis")
    try:
        projection = project_onto_basis(
            value, stored, rank=rank, _condition_bound=condition
        )
    except ConfigurationError:
        return reject("invalid_projection")
    residual = projection.relative_residual
    if projection.zero_input or residual == 0.0:
        reason = "zero_input" if projection.zero_input else "exact_projection"
        return ReuseDecision(True, reason, projection, time.perf_counter() - started)
    if residual > limit:
        return reject("residual_above_threshold", projection)
    boundary = 8 * torch.finfo(torch.float32).eps * max(1.0, limit)
    if limit - residual <= boundary:
        return reject("threshold_boundary", projection)
    return ReuseDecision(True, "residual_within_threshold", projection, time.perf_counter() - started)


@torch.no_grad()
def factorize_tensor(
    tensor: torch.Tensor,
    context: TensorContext,
    rank: int,
    *,
    reuse_basis: torch.Tensor | None = None,
    reuse_threshold: float | None = None,
) -> Factors:
    """Try one compatible parent, reuse its returned projection, or refit.

    ``decomposition_count`` counts SVD/eigh operations, not the small Cholesky
    validity screen. All rejected projection costs and rank-induced refits are
    included in the returned counters/timings.
    """
    started = time.perf_counter()
    value = validate_matrix(tensor, context)
    validate_rank(rank, tuple(value.shape))
    if reuse_threshold is not None:
        validate_threshold(reuse_threshold)
    if reuse_basis is not None and reuse_threshold is None:
        raise ConfigurationError("a reuse basis requires an explicit reuse_threshold")
    decision = None
    if reuse_basis is not None:
        decision = try_reuse_basis(value, context, reuse_basis, rank, reuse_threshold)
    if decision is not None and decision.accepted:
        projection = decision.projection
        assert projection is not None
        factors = Factors(
            basis=reuse_basis.detach(),
            coefficients=projection.coefficients,
            rank=rank,
            metrics={
                "factorization_method": "stored_basis_projection",
                "factorization_seed": None,
                "sketch_rank": None,
                "power_iterations": 0,
                "decomposition_count": 0,
                "svd_count": 0,
                "eigh_count": 0,
                "qr_count": 0,
                "decomposition_seconds": 0.0,
                "projection_count": 1,
                "projection_seconds": projection.seconds,
                "projection_relative_residual": projection.relative_residual,
                "basis_condition_bound": projection.condition_bound,
                "zero_input": projection.zero_input,
                "fit_seconds": 0.0,
            },
        )
    else:
        factors = fit_basis(value, context, rank)
    metrics = dict(factors.metrics)
    rejected_projection = (
        decision.projection if decision is not None and not decision.accepted else None
    )
    if rejected_projection is not None:
        metrics["projection_count"] += 1
        metrics["projection_seconds"] += rejected_projection.seconds
    metrics.update(
        reuse_attempts=int(decision is not None),
        reuse_accepted=bool(decision is not None and decision.accepted),
        reuse_rejections=int(decision is not None and not decision.accepted),
        reuse_reason=decision.reason if decision is not None else "no_candidate",
        reuse_seconds=decision.seconds if decision is not None else 0.0,
        attempted_relative_residual=(
            decision.projection.relative_residual
            if decision is not None and decision.projection is not None else None
        ),
        canonical_basis_rank=int(factors.basis.shape[0]),
        view_rank=rank,
        total_factorization_seconds=time.perf_counter() - started,
    )
    return Factors(factors.basis, factors.coefficients, rank, metrics)
