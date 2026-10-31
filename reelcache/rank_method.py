"""Paper-bound latest-method rank selection, verification and basis reuse.

This is the synchronous reference of the intra-chunk compressor and the
inherit/certify/augment/refit rule of ``tex/design.tex`` (Algorithm 1), on CPU
or CUDA. Conventions: canonical matrices ``A[N, C]``; stored row bases
``Q[r, C]`` in FP16; coefficients ``P = A.float() @ Q.float().T`` in FP32;
reconstruction ``P @ Q``. Only normalized pre-RoPE K and direct V are accepted.

Per matrix and energy target ``E_t``:

* **Anchor and prediction.** A column-orthonormal ``m``-direction sketch ``S``
  of ``X`` is built with QR-stabilized subspace iteration (``q`` power
  iterations, private generator). ``E_m = ||X S||^2 / ||X||^2``. If
  ``E_m >= E_t`` the sketch bucket ``m`` is the first rank bucket; otherwise
  ``x = ln((1 - E_m) / (1 - E_t))`` and ``r_hat = exp(a + b x)`` from the pinned
  learner of the regime (fresh or residual).
* **Fit, verify, grow, trim.** One FP32 Gram eigendecomposition returns all
  energy-ordered directions at once. The predicted bucket is verified with the
  actual stored representation (FP16 rows, FP32 coefficients, tiled direct
  residual); a short bucket grows by one bucket at a time and an over-wide one
  is trimmed to the smallest bucket whose ideal energy can reach the target
  (Eckart-Young bound) and whose stored representation verifies, without
  another fit. The label is the unrounded smallest sufficient rank within the
  fitted directions.
* **Inherit, certify, augment, refit.** With an active basis, the chunk is
  projected on the stored rows; ``e = 1 - ||R||^2 / ||A||^2``. If ``e >= E_t``
  the shortest bucket-aligned prefix that verifies is shared with no candidate
  fit. Otherwise the residual is fitted at ``(E_t - e) / (1 - e)``, its
  directions are re-orthogonalized against the inherited span (FP32 QR + two
  Gram-Schmidt passes; inherited bytes unchanged) and the combined stored rows are
  verified on ``A``. A refit starts a new basis group when the refit rule
  (``RankMethodConfig.refit_rule``) prefers a fresh fit of the most recent
  compatible fresh rank ``r_f`` to the augmentation. ``"paper"``: the
  incremental augmented chunk is estimated to store more bytes than the fresh
  fit, ``B_aug > B_fresh`` (``reelcache.refit_estimate.v1``; the default again since
  the user decision of 2026-10-17). ``"cumulative"`` (selectable, a ski-rental
  rule; the default of freezes D to F): an augmentation's coefficient excess over
  the fresh fit, ``excess`` (coefficient and scale bytes of the grown view
  minus the fresh fit's), recurs in every later chunk that keeps the grown
  basis, while the refit's extra basis bytes, ``cost``, are paid once. Each
  matrix's active binding carries ``refit_debt``, the excess accrued since its
  last fresh fit or refit; the chunk refits iff ``debt + excess >= h * cost``
  with the hysteresis ``h = refit_hysteresis`` (default 1, the plain
  ski-rental test; ``h > 1`` also charges the regrowth that follows a refit),
  or when ``B_aug > B_fresh`` (with no debt: exactly that paper test, bitwise,
  ties augment), otherwise it
  augments and accrues ``max(0, excess)``; a fresh fit or refit resets the
  debt (``reelcache.refit_estimate.v2`` records ``debt_before``, ``excess``,
  ``cost`` and ``debt_after``; every plan states its ``refit_debt``).
  ``refit_debt_on_share`` also charges shared chunks their excess over the
  fresh fit and refits a shared chunk whose debt reaches the cost.

Target 1.0 is a labeled full-width diagnostic that never consults or trains a
learner. Zero inputs are represented exactly with a one-row view.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch

from . import accel_b as _accel_b
from .errors import ConfigurationError, IntegrityError
from .deferred_reads import (
    BatchedCall, download, finite_flag, projection_routine, read_host, run_lockstep, run_one, source_energy_routine,
    step_phase, stored_energy_routine, upload, yield_point,
)
from .exact_blas import mm, trsm_right_upper
from .factorization import _canonical_signs, validate_matrix
from .predictor_registry import RegistryView
from .types import (
    RANK_BUCKET, ActiveBasisBinding, CodecSpec, EnergyTargets, FitObservation, MatrixKey,
    PredictorKey, TensorContext, canonical_digest,
)


METHOD_SCHEMA = "reelcache.rank_method.v1"
SOLVERS = ("gram_eigh", "warm_start", "topk_rr")
#: ``RankMethodConfig.refit_rule``: ``"paper"`` (the default again since the user decision of 2026-10-17) is
#: ``tex/design.tex``'s per-chunk test; ``"cumulative"`` (selectable; the default of freezes D to F, user decision
#: 2026-10-16) refits when the coefficient bytes an augmented basis keeps storing beyond a fresh fit, accrued since the
#: last fresh fit or refit, reach the refit's one-time basis cost.
REFIT_RULES = ("cumulative", "paper")
REFIT_ESTIMATE_V1 = "reelcache.refit_estimate.v1"
REFIT_ESTIMATE_V2 = "reelcache.refit_estimate.v2"
DECISIONS = ("fresh", "share", "augment", "refit", "zero_input", "diagnostic_full_width")
STAGES = ("transfer", "energy", "certify", "share_prefix", "residual", "sketch", "predict", "gram",
          "eigh", "verify", "augment_orthogonalize", "augment_verify", "estimate", "refit_sketch",
          "refit_gram", "refit_eigh", "refit_verify", "warm_start", "topk", "refit_topk")
#: wl-base (workload-v2 W1c): fixed-rank arms (``RankMethodConfig.fixed_rank_k`` / ``fixed_rank_v`` /
#: ``fixed_basis``). Their plans are ``fresh`` decisions with one of these reasons; the energy target is recorded but
#: not enforced (``diagnostic``), no learner is consulted or trained and no basis is shared.
FIXED_BASES = ("fitted", "identity")
FIXED_RANK_REASON = "fixed_rank_policy"
IDENTITY_BASIS_REASON = "identity_basis"
_IDENTITY_ROWS: dict[int, torch.Tensor] = {}


def identity_rows(channels: int) -> torch.Tensor:
    """The full-width FP16 identity basis ``[C, C]`` (host, shared and read-only by convention): the HEVC-only arm's
    coefficients ``P = A I^T`` are the matrix itself."""
    rows = _IDENTITY_ROWS.get(channels)
    if rows is None:
        rows = _IDENTITY_ROWS[channels] = torch.eye(channels, dtype=torch.float16)
    return rows


def is_identity_view(rows: torch.Tensor | None) -> bool:
    """Whether ``rows`` is (a view of) the shared identity basis of :func:`identity_rows` (no content comparison)."""
    if rows is None or rows.ndim != 2 or rows.shape[0] != rows.shape[1] or rows.device.type != "cpu":
        return False
    shared = _IDENTITY_ROWS.get(int(rows.shape[1]))
    return shared is not None and rows.data_ptr() == shared.data_ptr() and rows.stride() == shared.stride()


class RankTargetUnreachable(ConfigurationError):
    """The configured energy cannot be met even at full width; work is attached."""

    def __init__(self, message: str, work: Mapping[str, Any]):
        self.work = dict(work)
        super().__init__(message)


@dataclass(frozen=True)
class RankMethodConfig:
    """Versioned configuration of the paper method (defaults: paper m = 32, q = 2).

    ``solver`` selects how a fit's candidate directions are computed; the semantics (predicted bucket -> fit ->
    exact verification of the stored factors -> grow/trim; share/augment/refit) never change. Default since SGLang
    Step 7 Part C (user decision 2026-10-15): ``"topk_rr"`` (top-k Rayleigh-Ritz, ``topk_margin``/``topk_steps``);
    ``"gram_eigh"`` (the full Gram eigendecomposition, the earlier default) and ``"warm_start"`` remain explicit
    options. The solver is always serialized (``to_dict``), so stored methods keep their solver.

    ``refit_rule`` decides between a verified augmentation and a refit. ``"paper"`` (the default; user decision
    2026-10-17, reverting freeze D's default) refits iff this chunk's augmented bytes exceed a fresh fit's, ignores
    the hysteresis and reproduces archives without a refit rule bitwise. ``"cumulative"`` (selectable; user decision
    2026-10-16) carries a per-matrix ``refit_debt`` in the lineage's active binding and refits when the debt plus this
    chunk's recurring excess reaches ``refit_hysteresis`` times the refit's one-time basis cost (default 1, see
    :func:`refit_debt_test`; with no debt: exactly the paper's test). ``refit_debt_on_share`` (cumulative only, default off) also accrues the
    excess of shared chunks and refits a shared chunk whose debt reaches the cost. Both are serialized only away
    from the paper rule, so a method dict without ``refit_rule`` (every method before freeze D) is the paper rule.
    """

    targets: EnergyTargets
    codec: CodecSpec
    anchor_rank: int = RANK_BUCKET
    bucket: int = RANK_BUCKET
    power_iterations: int = 2
    tile_rows: int = 1024
    seed: int = 0
    reuse: bool = True
    refit: bool = True
    solver: str = "topk_rr"
    warm_steps: int = 1
    decision_band: float = 1e-6
    trim_slack: float = 1e-5
    coefficient_bytes_proxy: float | None = None
    metadata_bytes_estimate: int = 4096
    batch_size: int = 60
    rate_lag: int = 0
    # Matmul precision policy (see require_fp32_matmul): "fp32" (reference; CUDA planning refuses while the
    # process allows TF32) or "fp32_verified_fp64" (declared up front by profiles whose generation runs with
    # TF32: every acceptance measurement uses FP64 products, so energy targets hold whatever the global flags).
    matmul_precision: str = "fp32"
    # Top-k Rayleigh-Ritz fits (solver="topk_rr"): directions per fit = predicted bucket + topk_margin, refined by
    # topk_steps Gram products from a warm start before the Rayleigh-Ritz solve (see topk_directions). Defaults:
    # the setting closest to the exact solver's ranks/bytes on the real captures at the same plan time (Part C).
    topk_margin: int = 64
    topk_steps: int = 3
    refit_rule: str = "paper"
    refit_debt_on_share: bool = False
    # Cumulative rule: a debt refits once it reaches ``refit_hysteresis`` times the refit's basis cost (see
    # refit_debt_test). 1 (default) is the plain ski-rental test; the refinement study of 2026-10-17 measured 2 as
    # the setting that keeps fits and planner time within 1.25x of the paper rule.
    refit_hysteresis: float = 1.0
    # wl-base (W1c) fixed-rank arms, serialized only when set (earlier fingerprints unchanged): every K / V matrix is
    # fitted fresh at exactly ``fixed_rank_k`` / ``fixed_rank_v`` (clamped to its full width; the uniform-rank
    # ablation sets both to one rank, the SVD baseline C/6 and C/4), or ``fixed_basis="identity"`` takes the
    # full-width identity basis without a fit (the HEVC-only baseline). Both need ``reuse=False``.
    fixed_rank_k: int | None = None
    fixed_rank_v: int | None = None
    fixed_basis: str = "fitted"

    def __post_init__(self) -> None:
        if not isinstance(self.targets, EnergyTargets):
            raise ConfigurationError("rank method requires explicit EnergyTargets")
        if self.matmul_precision not in MATMUL_PRECISIONS:
            raise ConfigurationError(f"matmul_precision must be one of {MATMUL_PRECISIONS}")
        if not isinstance(self.codec, CodecSpec) or self.codec.kind not in ("svd", "svd_hevc"):
            raise ConfigurationError("the rank method encodes svd or svd_hevc coefficient packets")
        if self.codec.inter_prediction is None:
            raise ConfigurationError("legacy implicit-GOP svd_hevc specifications are decode-only")
        for name, low, high in (("anchor_rank", 1, 4096), ("bucket", 1, 4096), ("power_iterations", 0, 8),
                                ("tile_rows", 1, 1 << 20), ("warm_steps", 0, 4), ("batch_size", 1, 256),
                                ("metadata_bytes_estimate", 0, 1 << 30), ("rate_lag", 0, 8),
                                ("topk_margin", 0, 4096), ("topk_steps", 0, 8)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ConfigurationError(f"{name} must be an integer in [{low}, {high}]")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ConfigurationError("seed must be an integer in [0, 2**63)")
        for name in ("reuse", "refit", "refit_debt_on_share"):
            if type(getattr(self, name)) is not bool:
                raise ConfigurationError(f"{name} must be boolean")
        if self.solver not in SOLVERS:
            raise ConfigurationError(f"solver must be one of {SOLVERS}")
        if type(self.refit_rule) is not str or self.refit_rule not in REFIT_RULES:
            raise ConfigurationError(f"refit_rule must be one of {REFIT_RULES}")
        if self.refit_debt_on_share and self.refit_rule != "cumulative":
            raise ConfigurationError("refit_debt_on_share extends the cumulative refit rule only")
        hysteresis = self.refit_hysteresis
        if (isinstance(hysteresis, bool) or not isinstance(hysteresis, (int, float)) or not math.isfinite(hysteresis)
                or not 1 <= hysteresis <= 16):
            raise ConfigurationError("refit_hysteresis must be a number in [1, 16]")
        object.__setattr__(self, "refit_hysteresis", float(hysteresis))
        for name, high in (("decision_band", 1e-3), ("trim_slack", 1e-2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= high:
                raise ConfigurationError(f"{name} must be in [0, {high}]")
        proxy = self.coefficient_bytes_proxy
        if proxy is not None and (isinstance(proxy, bool) or not isinstance(proxy, (int, float))
                                  or not math.isfinite(proxy) or proxy <= 0):
            raise ConfigurationError("coefficient byte proxy must be finite and positive")
        if (self.fixed_rank_k is None) != (self.fixed_rank_v is None):
            raise ConfigurationError("fixed_rank_k and fixed_rank_v are given together")
        if self.fixed_rank_k is not None and any(type(value) is not int or value < 1
                                                 for value in (self.fixed_rank_k, self.fixed_rank_v)):
            raise ConfigurationError("fixed ranks must be positive integers")
        if self.fixed_basis not in FIXED_BASES:
            raise ConfigurationError(f"fixed_basis must be one of {FIXED_BASES}")
        if self.fixed_basis == "identity" and (self.fixed_rank_k is not None or self.codec.kind != "svd_hevc"):
            raise ConfigurationError("the identity basis is full-width and HEVC-coded (svd_hevc, no fixed ranks)")
        if self.fixed_policy and self.reuse:
            raise ConfigurationError("fixed-rank arms share no basis across chunks: reuse must be False")

    @property
    def fixed_policy(self) -> str | None:
        """``"identity"``, ``"fixed_rank"`` or None (the adaptive energy-target method)."""
        if self.fixed_basis == "identity":
            return "identity"
        return "fixed_rank" if self.fixed_rank_k is not None else None

    def fixed_rank_for(self, target: str) -> int | None:
        """The configured fixed rank of a K or V matrix (None without a fixed-rank policy)."""
        return self.fixed_rank_k if target == "K" else self.fixed_rank_v

    @property
    def detector(self) -> str:
        return f"sketch_qr_subspace.v1:m{self.anchor_rank}:q{self.power_iterations}"

    def solver_id(self, *, warm: bool = False) -> str:
        if self.solver == "topk_rr":
            return f"fp32_gram_topk_rr.v1:m{self.topk_margin}:s{self.topk_steps}"
        return f"fp32_gram_warm_rr.v1:s{self.warm_steps}" if warm else "fp32_gram_eigh.v1"

    def spec_for(self, rank: int) -> CodecSpec:
        return dataclasses.replace(self.codec, rank=rank)

    def predictor_key(self, context: TensorContext, regime: str, *, warm: bool = False) -> PredictorKey:
        return PredictorKey(context.model.fingerprint, context.key.layer, context.key.target,
                            context.token_count, context.channels, self.solver_id(warm=warm),
                            self.detector, regime)

    @property
    def coefficient_proxy(self) -> float:
        if self.coefficient_bytes_proxy is not None:
            return float(self.coefficient_bytes_proxy)
        return 4.0 if self.codec.kind == "svd" else 1.0

    def to_dict(self) -> dict[str, Any]:
        codec = self.codec.to_dict()
        codec.pop("ffmpeg_path", None)
        codec.pop("rank", None)
        data = {
            "schema": METHOD_SCHEMA, "targets": self.targets.to_dict(), "codec": codec,
            **{name: getattr(self, name) for name in (
                "anchor_rank", "bucket", "power_iterations", "tile_rows", "seed", "reuse", "refit",
                "solver", "warm_steps", "decision_band", "trim_slack", "coefficient_bytes_proxy",
                "metadata_bytes_estimate", "batch_size",
            )},
            "detector": self.detector,
        }
        # Omitted at their defaults so earlier method fingerprints are unchanged.
        if self.rate_lag:
            data["rate_lag"] = self.rate_lag
        if self.matmul_precision != "fp32":
            data["matmul_precision"] = self.matmul_precision
        if self.solver == "topk_rr":
            data["topk_margin"] = self.topk_margin
            data["topk_steps"] = self.topk_steps
        # Serialized away from the paper rule (the rule of every method before freeze D), whose fingerprints stay.
        if self.refit_rule != "paper":
            data["refit_rule"] = self.refit_rule
        if self.refit_debt_on_share:
            data["refit_debt_on_share"] = True
        # The hysteresis refines the cumulative rule only; at 1 the identity is the plain ski-rental rule's.
        if self.refit_rule == "cumulative" and self.refit_hysteresis != 1.0:
            data["refit_hysteresis"] = self.refit_hysteresis
        if self.fixed_rank_k is not None:  # wl-base (W1c) fixed-rank arms only
            data["fixed_ranks"] = {"K": self.fixed_rank_k, "V": self.fixed_rank_v}
        if self.fixed_basis != "fitted":
            data["fixed_basis"] = self.fixed_basis
        return data

    @property
    def measurement_products(self) -> str:
        """Product precision of acceptance measurements (``reelcache.deferred_reads`` ``products``)."""
        return "fp64" if self.matmul_precision == "fp32_verified_fp64" else "fp32"

    @property
    def fingerprint(self) -> str:
        return canonical_digest(self.to_dict())


def rate_identity(context: TensorContext, spec: CodecSpec) -> str:
    """Identity under which a measured coefficient byte rate may estimate another packet."""
    codec = spec.to_dict()
    for name in ("rank", "ffmpeg_path"):
        codec.pop(name, None)
    return canonical_digest({
        "codec": codec, "frames": context.frames, "height": context.height, "width": context.width,
        "channels": context.channels, "target": context.key.target, "dtype": context.dtype,
    })


def estimate_bytes(config: RankMethodConfig, context: TensorContext, coefficient_rank: int, new_rows: int,
                   binding: ActiveBasisBinding) -> dict[str, Any]:
    """The paper's per-chunk byte comparison terms; never a claimed encoded size."""
    spec = config.spec_for(max(1, coefficient_rank))
    if binding.coefficient_rate is not None and binding.rate_identity == rate_identity(context, spec):
        rate, kind = binding.coefficient_rate, "measured_coefficient_rate"
    else:
        rate, kind = config.coefficient_proxy, "factor_byte_proxy"
    scales = 0
    if config.codec.kind == "svd_hevc":
        from . import _hevc
        layout = _hevc.normalization_layout(max(1, coefficient_rank), config.codec.normalization,
                                            config.codec.protected_fraction)
        scales = 4 * layout["affine_shape"][0] * layout["affine_shape"][1]
    coefficients = rate * context.token_count * coefficient_rank
    basis = 2 * context.channels * new_rows
    total = coefficients + basis + scales + config.metadata_bytes_estimate
    return {"estimate_type": kind, "coefficient_rate": rate, "coefficient_rank": coefficient_rank,
            "new_basis_rows": new_rows, "coefficient_bytes": coefficients, "basis_bytes": basis,
            "scale_bytes": scales, "metadata_bytes": config.metadata_bytes_estimate, "total_bytes": total}


def refit_debt_test(debt: float, candidate: Mapping[str, Any], fresh: Mapping[str, Any], *,
                    hysteresis: float = 1.0) -> tuple[bool, float, float]:
    """``(refit, excess, cost)`` of the cumulative refit rule for one chunk's candidate vs a fresh fit.

    ``candidate`` and ``fresh`` are :func:`estimate_bytes` terms (the augmented or shared representation, and a
    fresh fit at the lineage's fresh rank). ``excess``: the candidate's per-chunk bytes beyond the fresh fit's that
    recur in every later chunk while its basis stays active (coefficient and normalization-scale bytes; the
    metadata estimate is the same on both sides); ``cost``: the fresh fit's basis bytes beyond the candidate's,
    paid once. With a debt the chunk refits iff ``debt + excess >= cost``. With no debt the test is the paper's
    ``B_candidate > B_fresh`` on the same totals (``B_candidate - B_fresh = excess - cost``), bitwise, so a tie
    keeps the candidate as the paper rule does.

    A hysteresis ``h > 1`` raises the debt threshold to ``h * cost``: a refit also pays for the rows the lineage
    grows again after it (a fresh basis fits one chunk; the next chunks augment it back) and the estimate's
    average coefficient rate overstates the bytes of the trailing ranks it saves. A raised threshold never drops a
    refit the chunk's own bytes call for (the paper test).
    """
    excess = (candidate["coefficient_bytes"] + candidate["scale_bytes"]) - (
        fresh["coefficient_bytes"] + fresh["scale_bytes"])
    cost = fresh["basis_bytes"] - candidate["basis_bytes"]
    paper = candidate["total_bytes"] > fresh["total_bytes"]
    if debt <= 0:
        return paper, excess, cost
    if hysteresis == 1.0:
        return debt + excess >= cost, excess, cost
    return paper or debt + excess >= hysteresis * cost, excess, cost


def chosen_estimate_bytes(estimate: Mapping[str, Any]) -> float | None:
    """Estimated bytes of the representation a refit estimate chose (a matrix record's ``estimate_vs_actual``).

    Reads ``reelcache.refit_estimate.v1`` and ``.v2`` documents (v2 share documents name their candidate ``shared``).
    """
    if estimate.get("decision") == "refit":
        return (estimate["fresh"] or {}).get("total_bytes")
    return (estimate["shared"] if "shared" in estimate else estimate["augmented"])["total_bytes"]


@dataclass
class DirectionSet:
    """Energy-ordered orthonormal FP32 rows of one fit and their ideal energies."""

    rows: torch.Tensor
    energies: torch.Tensor
    total: float
    complete: bool
    solver: str
    extender: Callable[[DirectionSet, int], DirectionSet] | None = None

    @property
    def width(self) -> int:
        return int(self.rows.shape[0])

    def ideal(self, rank: int) -> float:
        if self.total <= 0:
            return 1.0
        return float(self.energies[:rank].sum()) / self.total

    def crossing(self, fraction: float) -> int | None:
        """Smallest rank whose ideal energy reaches ``fraction`` within these directions."""
        if self.total <= 0:
            return 0
        cumulative = self.energies.cumsum(0)
        hits = torch.nonzero(cumulative >= fraction * self.total, as_tuple=False)
        return None if hits.numel() == 0 else int(hits[0, 0]) + 1


@dataclass
class MatrixPlan:
    """One matrix's verified decision on the decision device, before encoding."""

    key: MatrixKey
    decision: str
    reason: str
    target: float
    parent_version_ref: str | None
    new_rows: torch.Tensor | None
    rank_view: int
    bucket_rank: int
    observed_rank: int | None
    decision_energy: float
    observations: tuple[FitObservation, ...]
    predictions: tuple[dict[str, Any], ...]
    work: dict[str, Any]
    estimate: dict[str, Any] | None
    fresh_rank: int | None
    zero_input: bool = False
    diagnostic: bool = False
    sets_active: bool = True
    # Cumulative refit rule: the debt the matrix's next active binding carries (0 after a fresh fit or refit);
    # None under the paper rule, which tracks no debt (its bindings and records are unchanged).
    refit_debt: float | None = None

    def summary(self) -> dict[str, Any]:
        data = {
            "schema": "reelcache.rank_decision.latest.v1", "key": self.key.to_dict(),
            "decision": self.decision, "reason": self.reason, "target": self.target,
            "parent_version": self.parent_version_ref,
            "new_rows": 0 if self.new_rows is None else int(self.new_rows.shape[0]),
            "rank_view": self.rank_view, "bucket_rank": self.bucket_rank,
            "observed_rank": self.observed_rank, "decision_energy": self.decision_energy,
            "observations": [item.to_dict() for item in self.observations],
            "predictions": list(self.predictions), "work": self.work, "estimate": self.estimate,
            "zero_input": self.zero_input, "diagnostic_full_width": self.diagnostic,
            "sets_active": self.sets_active,
        }
        if self.refit_debt is not None:
            data["refit_debt"] = self.refit_debt
        return data


@dataclass
class ChunkPlan:
    plans: dict[MatrixKey, MatrixPlan]
    stages: dict[str, float]
    counters: dict[str, int]
    device: str
    eigh_solver: str | None
    pinned_sequence: int
    seconds: float = 0.0
    batches: list[dict[str, Any]] = field(default_factory=list)
    orthogonalizer: str = "householder"
    # Working-set bound of this plan (``LatestMethodPlanner(max_working_bytes=...)``): budget, groups, estimate.
    memory: dict[str, Any] | None = None


def _seed(config: RankMethodConfig, context: TensorContext, regime: str, chunk_id: str, purpose: str) -> int:
    digest = canonical_digest({
        "seed": config.seed, "model": context.model.fingerprint, "key": context.key.to_dict(),
        "token_start": context.token_start, "regime": regime, "chunk": chunk_id, "purpose": purpose,
    })
    return int(digest[:16], 16) % 2**63


def _bucket(rank: float, bucket: int, full: int) -> int:
    if not math.isfinite(rank):
        return full
    return int(min(full, max(bucket, math.ceil(max(rank, 1.0) / bucket) * bucket)))


def _legal_prefixes(total: int, bucket: int) -> list[int]:
    return sorted({*range(bucket, total, bucket), total})


#: Share of the device's total memory one stacked sketch batch may use (a device constant: deterministic splits).
SKETCH_MEMORY_FRACTION = 0.125
MATMUL_PRECISIONS = ("fp32", "fp32_verified_fp64")


def process_allows_tf32() -> bool:
    """Whether FP32 CUDA matmuls of this process may run as TF32 (process-global flags, read only)."""
    return bool(torch.backends.cuda.matmul.allow_tf32) or torch.get_float32_matmul_precision() != "highest"


def require_fp32_matmul(device: torch.device, policy: str = "fp32") -> None:
    """Validate the method's matmul precision policy; the process-global flags are never read or toggled.

    Every FP32 product of planning, verification and finalization is an explicit IEEE FP32 cuBLAS product on CUDA
    (:mod:`reelcache.exact_blas`), independent of ``allow_tf32`` / ``set_float32_matmul_precision`` -- which another
    thread (e.g. a model's TF32 region) may toggle at any time -- so no flag state is refused (SGLang Step 7 Part C;
    before, CUDA planning under ``"fp32"`` refused a process that allowed TF32). ``"fp32"``: IEEE FP32 acceptance
    products; ``"fp32_verified_fp64"``: every acceptance measurement in FP64 products (``reelcache.deferred_reads``).
    """
    if policy not in MATMUL_PRECISIONS:
        raise ConfigurationError(f"matmul precision policy must be one of {MATMUL_PRECISIONS}")


def default_eigh(grams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ascending eigenvalues and *row* eigenvectors of a symmetric batch (any device)."""
    values, columns = torch.linalg.eigh(grams)
    return values, columns.transpose(1, 2)


class _NullSpan:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Clock:
    """Stage timing on the planning stream.

    CUDA stages are timed with events recorded on the planning device's
    current stream and read once by :meth:`finish`: no host synchronization
    per stage (synchronizing the whole device around every stage also waited
    for a concurrent native forward's queued kernels). ``synchronize=True``
    restores device-synchronized wall-clock spans for profiling. CPU stages
    use wall-clock spans.
    """

    def __init__(self, device: torch.device, *, synchronize: bool = False):
        self.device = device
        self.synchronize = synchronize
        self.stages = {name: 0.0 for name in STAGES}
        self._events: list = []

    def sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def time(self, stage: str):
        clock = self
        events = self.device.type == "cuda" and not self.synchronize

        class _Span:
            def __enter__(self):
                step_phase("planner:" + stage)  # dev22j8: names the planner's stage (pause ack-timeout records)
                if events:
                    self.begin = torch.cuda.Event(enable_timing=True)
                    self.begin.record(torch.cuda.current_stream(clock.device))
                else:
                    clock.sync()
                    self.started = time.perf_counter()

            def __exit__(self, *exc):
                if events:
                    end = torch.cuda.Event(enable_timing=True)
                    end.record(torch.cuda.current_stream(clock.device))
                    clock._events.append((stage, self.begin, end))
                else:
                    clock.sync()
                    clock.stages[stage] += time.perf_counter() - self.started

        return _Span()

    def finish(self) -> dict[str, float]:
        """Stage seconds (waits once for the last recorded event)."""
        if self._events:
            self._events[-1][2].synchronize()
            for stage, begin, end in self._events:
                self.stages[stage] += begin.elapsed_time(end) / 1e3
            self._events.clear()
        return dict(self.stages)


@torch.no_grad()
def anchor_sketches(scaled: Sequence[torch.Tensor], totals: Sequence[float], seeds: Sequence[int], *,
                    anchor_rank: int, power_iterations: int, max_stack_bytes: int | None = None,
                    record: list | None = None) -> list[dict[str, Any]]:
    """Per-matrix QR subspace-iteration sketches ``S = orth((X^T X)^(q+1) Omega)``.

    Each item uses its own private generator and is computed on its own with explicit IEEE FP32 products
    (:mod:`reelcache.exact_blas`), so its sketch depends neither on the process-global matmul flags nor on the other
    items; no stacked batch is formed (peak memory: one item's products). Returns ``captured_energy`` and the
    sketch (column-orthonormal ``[C, m]``) per item, reading every captured energy in one host transfer.
    ``max_stack_bytes`` and ``record`` are accepted for compatibility: nothing is stacked, so nothing is split.
    """
    if not (len(scaled) == len(totals) == len(seeds)):
        raise ConfigurationError("sketch inputs, totals and seeds must align")
    sketches, captured, widths = [], [], []
    for index, matrix in enumerate(scaled):
        yield_point()  # dev22j8: a step boundary per matrix (the loop queues kernels; a pause lands between items)
        rows, channels = (int(size) for size in matrix.shape)
        width = min(anchor_rank, rows, channels)
        generator = torch.Generator(device=matrix.device)
        generator.manual_seed(seeds[index])
        sketch = torch.randn(channels, width, generator=generator, device=matrix.device, dtype=torch.float32)
        for _ in range(power_iterations + 1):
            sketch = torch.linalg.qr(mm(matrix.T, mm(matrix, sketch)), mode="reduced").Q
        if _accel_b.planner_fp32(matrix):  # accel B2 (opt-in): FP32 squares accumulated in FP64
            captured.append(_accel_b.sumsq64(mm(matrix, sketch)))
        else:
            captured.append(mm(matrix, sketch).double().square().sum())
        sketches.append(sketch)
        widths.append(width)
    captured_host = read_host([tuple(captured)])[0] if captured else ()
    results = []
    for index, value in enumerate(captured_host):
        total = totals[index]
        energy = 1.0 if total <= 0 else value / total
        if not math.isfinite(energy):
            raise ConfigurationError("anchor sketch produced a nonfinite captured energy")
        results.append({
            "captured_energy": min(1.0, max(0.0, energy)), "sketch": sketches[index], "width": widths[index],
            "matmuls": 2 * (power_iterations + 1) + 1, "qr": power_iterations + 1,
        })
    return results


def _symmetric_gram(matrix: torch.Tensor) -> torch.Tensor:
    """``(X^T X + (X^T X)^T) / 2`` with an explicit IEEE FP32 product."""
    yield_point()  # dev22j8: a step boundary per matrix's Gram product
    gram = mm(matrix.T, matrix)
    return (gram + gram.T) * 0.5


@torch.no_grad()
def gram_directions(scaled: Sequence[torch.Tensor], totals: Sequence[float], *,
                    eigh: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
                    batch_size: int = 60, solver: str = "fp32_gram_eigh.v1",
                    timer: Callable[[str], Any] | None = None) -> list[DirectionSet]:
    """All energy-ordered directions of each matrix from one batched FP32 Gram eigh.

    The Grams are explicit IEEE FP32 products (:mod:`reelcache.exact_blas`), the batched eigensolver is cuSOLVER
    (independent of the global matmul flags; a matrix's eigenpairs depend on its call's batch size class unless the
    eigensolver fixes the sub-batch, see :class:`reelcache.gpu_fitting.BoundedEigh`). ``timer(stage)`` optionally
    returns a context manager used to attribute the Gram products (``"gram"``) and the eigensolve (``"eigh"``).
    """
    eigh = default_eigh if eigh is None else eigh
    timer = (lambda _stage: _NullSpan()) if timer is None else timer
    results: list[DirectionSet | None] = [None] * len(scaled)
    groups: dict[tuple, list[int]] = {}
    for index, matrix in enumerate(scaled):
        groups.setdefault((int(matrix.shape[1]), matrix.device), []).append(index)
    for (_, _), indices in groups.items():
        for start in range(0, len(indices), batch_size):
            members = indices[start:start + batch_size]
            with timer("gram"):
                grams = torch.stack([_symmetric_gram(scaled[index]) for index in members])
            with timer("eigh"):
                values, vectors = eigh(grams)
            if not bool(torch.isfinite(values).all() & torch.isfinite(vectors).all()):
                raise ConfigurationError("batched eigensolver returned nonfinite values")
            for position, index in enumerate(members):
                rows = _canonical_signs(vectors[position].flip(0)).contiguous()
                energies = values[position].flip(0).double().clamp_min(0)
                results[index] = DirectionSet(rows, energies, totals[index], True, solver)
            del grams, values, vectors
    return results  # type: ignore[return-value]


def _orthonormalize_columns(block: torch.Tensor) -> torch.Tensor:
    """Scaled CholeskyQR2 (explicit IEEE FP32) with an explicit Householder QR repair when Cholesky breaks down."""
    work = block / block.norm(dim=0, keepdim=True).clamp_min(1e-30)
    for _ in range(2):
        factor, info = torch.linalg.cholesky_ex(mm(work.T, work), upper=True)
        if int(info) != 0:
            return torch.linalg.qr(block, mode="reduced").Q
        work = trsm_right_upper(factor, work)
    return work


@torch.no_grad()
def warm_start_directions(scaled: torch.Tensor, total: float, parent_rows: torch.Tensor, *, steps: int,
                          pad: int, seed: int) -> DirectionSet:
    """Explicit warm start from a parent basis: Gram product, CholeskyQR, small Rayleigh-Ritz eigh.

    The fitted subspace is ``[parent rows, pad random rows]`` refined by
    ``steps`` subspace iterations with ``G = X^T X``; growth extends it by a
    deflated random block and repeats Rayleigh-Ritz on the whole span, so
    earlier work is reused rather than refitted. Products are explicit IEEE FP32.
    """
    channels = scaled.shape[1]
    gram = _symmetric_gram(scaled)
    generator = torch.Generator(device=scaled.device)
    generator.manual_seed(seed)

    def ritz(columns: torch.Tensor, extra_steps: int) -> DirectionSet:
        basis = _orthonormalize_columns(columns)
        for _ in range(extra_steps):
            basis = _orthonormalize_columns(mm(gram, basis))
        projected = mm(mm(basis.T, gram), basis)
        values, vectors = torch.linalg.eigh((projected + projected.T) * 0.5)
        rows = _canonical_signs(mm(basis, vectors.flip(1)).T).contiguous()
        result = DirectionSet(rows, values.flip(0).double().clamp_min(0), total,
                              rows.shape[0] >= channels, "fp32_gram_warm_rr.v1")
        result.extender = extend
        return result

    def extend(current: DirectionSet, extra: int) -> DirectionSet:
        width = min(channels, current.width + extra)
        block = torch.randn(channels, width - current.width, generator=generator, device=scaled.device,
                            dtype=torch.float32)
        existing = current.rows.T
        for _ in range(2):
            block = block - mm(existing, mm(existing.T, block))
        return ritz(torch.cat((existing, block), dim=1), 1)

    width = min(channels, parent_rows.shape[0] + pad)
    start = parent_rows.float().T
    if width > start.shape[1]:
        block = torch.randn(channels, width - start.shape[1], generator=generator, device=scaled.device,
                            dtype=torch.float32)
        start = torch.cat((start, block), dim=1)
    return ritz(start, steps)


def _orthonormalize_each(blocks: Sequence[torch.Tensor], failed: Sequence[torch.Tensor] | None = None):
    """Column-orthonormal ``[C, k]`` per matrix by scaled CholeskyQR2 (explicit IEEE FP32), without a host read.

    Returns ``(bases, failed)``: ``failed`` holds per-matrix device flags accumulating Cholesky breakdowns; those
    bases are left non-orthonormal and the caller recomputes them with Householder QR.
    """
    bases, flags = [], []
    for position, block in enumerate(blocks):
        yield_point()  # dev22j8: a step boundary per matrix
        flag = torch.zeros((), dtype=torch.bool, device=block.device) if failed is None else failed[position]
        identity = torch.eye(block.shape[1], dtype=block.dtype, device=block.device)
        work = block / block.norm(dim=0, keepdim=True).clamp_min(1e-30)
        for _ in range(2):
            factor, info = torch.linalg.cholesky_ex(mm(work.T, work), upper=True)
            flag = flag | (info != 0)
            work = trsm_right_upper(torch.where(flag, identity, factor), work)
        bases.append(work)
        flags.append(flag)
    return bases, flags


def _ritz_each(grams: Sequence[torch.Tensor], bases: Sequence[torch.Tensor], eigh) -> tuple[list, list]:
    """Rayleigh-Ritz on each ``span(basis)``: descending Ritz values ``[k]`` and Ritz rows ``[k, C]`` per matrix.

    Projections and rotations are explicit IEEE FP32 products; the ``k x k`` eigenproblems are one batched call.
    """
    projected = []
    for gram, basis in zip(grams, bases):
        yield_point()  # dev22j8: a step boundary per matrix
        product = mm(basis.T, mm(gram, basis))
        projected.append((product + product.T) * 0.5)
    values, vectors = eigh(torch.stack(projected))
    rows = [mm(basis, vectors[position].T.clone()).T.flip(0) for position, basis in enumerate(bases)]
    return [values[position].flip(0) for position in range(len(bases))], rows


def _householder_iteration(gram: torch.Tensor, block: torch.Tensor, steps: int, eigh):
    """Reference subspace iteration of one ``[C, C]`` Gram (Householder QR at every step) and its Rayleigh-Ritz."""
    basis = torch.linalg.qr(block, mode="reduced").Q
    for _ in range(steps):
        basis = torch.linalg.qr(mm(gram, basis), mode="reduced").Q
    values, rows = _ritz_each([gram], [basis], eigh)
    return values[0], rows[0]


#: Largest ``max |V^T V - I|`` accepted from the batched CholeskyQR2 (above it the matrix is recomputed).
TOPK_ORTHOGONALITY_TOLERANCE = 1e-3


@torch.no_grad()
def topk_directions(scaled: Sequence[torch.Tensor], totals: Sequence[float], widths: Sequence[int],
                    starts: Sequence[torch.Tensor | None], seeds: Sequence[int], *,
                    eigh: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
                    margin: int = 64, steps: int = 3, align: int = 1, batch_size: int = 60,
                    solver: str = "fp32_gram_topk_rr.v1",
                    timer: Callable[[str], Any] | None = None) -> list[DirectionSet]:
    """Leading energy-ordered directions by blocked subspace iteration and Rayleigh-Ritz (no full eigensolve).

    Each matrix gets ``k = min(C, width + margin)`` directions (rounded up to a multiple of ``align``, the rank
    bucket, so that verified widths stay on the bucket grid): its start block (``starts``: warm-start columns
    ``[C, s]`` such as the anchor sketch subspace or a parent basis, then seeded Gaussian columns) is refined by
    ``steps`` products with ``G = X^T X`` (scaled CholeskyQR2), and the ``k x k`` projection of G is solved by the
    batched eigensolver; matrices whose k reaches C get the exact full eigendecomposition. Every product is an
    explicit IEEE FP32 product computed per matrix (:mod:`reelcache.exact_blas`): a matrix's products depend neither
    on the global matmul flags nor on the other matrices of its batch (its batched ``k x k`` eigensolve depends on
    the call's batch size class unless the eigensolver fixes the sub-batch). Matrices with equal ``(C, k)``
    share one batched eigensolve and one host read (matrices whose CholeskyQR2 broke down are recomputed with
    Householder QR). Every incomplete set carries a :class:`_TopkExtender`: growth past k doubles the subspace
    (the current Ritz vectors plus fresh seeded columns, ``steps`` Gram products, Rayleigh-Ritz; the exact full
    eigensolve once it reaches C) instead of refitting; the planner batches the extensions requested in one
    lockstep round. The directions only steer the search; every accepted rank is verified on the stored factors.
    """
    eigh = default_eigh if eigh is None else eigh
    timer = (lambda _stage: _NullSpan()) if timer is None else timer
    if not (len(scaled) == len(totals) == len(widths) == len(starts) == len(seeds)):
        raise ConfigurationError("top-k inputs, totals, widths, starts and seeds must align")
    results: list[DirectionSet | None] = [None] * len(scaled)
    groups: dict[tuple, list[int]] = {}
    for index, matrix in enumerate(scaled):
        channels = int(matrix.shape[1])
        k = int(min(channels, -(-(max(1, int(widths[index])) + margin) // align) * align))
        groups.setdefault((channels, k, matrix.device), []).append(index)

    for (channels, k, device), indices in groups.items():
        for start in range(0, len(indices), batch_size):
            members = indices[start:start + batch_size]
            with timer("gram"):
                grams = [_symmetric_gram(scaled[index]) for index in members]
            if k >= channels:
                with timer("eigh"):
                    values, vectors = eigh(torch.stack(grams))
                if not bool(torch.isfinite(values).all() & torch.isfinite(vectors).all()):
                    raise ConfigurationError("batched eigensolver returned nonfinite values")
                for position, index in enumerate(members):
                    results[index] = DirectionSet(_canonical_signs(vectors[position].flip(0)).contiguous(),
                                                  values[position].flip(0).double().clamp_min(0), totals[index],
                                                  True, solver)
                del grams, values, vectors
                continue
            with timer("topk"):
                blocks = []
                for index in members:
                    block = torch.empty((channels, k), dtype=torch.float32, device=device)
                    used = 0
                    begin = starts[index]
                    if begin is not None and begin.numel():
                        used = min(k, int(begin.shape[1]))
                        block[:, :used] = begin[:, :used].to(device=device, dtype=torch.float32)
                    if used < k:
                        generator = torch.Generator(device=device)
                        generator.manual_seed(int(seeds[index]))
                        block[:, used:] = torch.randn(channels, k - used, generator=generator, device=device,
                                                      dtype=torch.float32)
                    blocks.append(block)
                bases, failed = _orthonormalize_each(blocks)
                for _ in range(steps):
                    bases, failed = _orthonormalize_each([mm(gram, basis) for gram, basis in zip(grams, bases)],
                                                         failed)
                identity = torch.eye(k, dtype=torch.float32, device=device)
                fallback = torch.eye(channels, k, dtype=torch.float32, device=device)
                for position, basis in enumerate(bases):
                    drift = (mm(basis.T, basis) - identity).abs().amax()
                    failed[position] = failed[position] | ~(drift <= TOPK_ORTHOGONALITY_TOLERANCE)
                    # Rayleigh-Ritz stays well defined for broken-down matrices (recomputed below).
                    bases[position] = torch.where(failed[position], fallback, basis)
            with timer("eigh"):
                values, rows = _ritz_each(grams, bases, eigh)
            finite = [torch.isfinite(value).all() & torch.isfinite(row).all() for value, row in zip(values, rows)]
            flags = read_host([tuple(failed) + tuple(finite)])[0]  # one host read per batch
            failed_host, finite_host = list(flags[:len(members)]), list(flags[len(members):])
            for position, index in enumerate(members):
                if failed_host[position]:
                    value, row = _householder_iteration(grams[position], blocks[position], steps, eigh)
                    finite_host[position] = bool(torch.isfinite(value).all() & torch.isfinite(row).all())
                    values[position], rows[position] = value, row
                if not finite_host[position]:
                    raise ConfigurationError("top-k Rayleigh-Ritz returned nonfinite values")
                result = DirectionSet(_canonical_signs(rows[position]).contiguous(),
                                      values[position].double().clamp_min(0), totals[index], False, solver)
                result.extender = _TopkExtender(scaled[index], totals[index], int(seeds[index]), eigh, solver,
                                                steps, batch_size)
                results[index] = result
            del grams, values, rows, bases, blocks
    return results  # type: ignore[return-value]


class _TopkExtender:
    """Extends a top-k :class:`DirectionSet` (callable inline, or as a batched lockstep request)."""

    __slots__ = ("matrix", "total", "seed", "eigh", "solver", "steps", "batch_size")

    def __init__(self, matrix, total, seed, eigh, solver, steps, batch_size):
        self.matrix, self.total, self.seed, self.eigh = matrix, total, seed, eigh
        self.solver, self.steps, self.batch_size = solver, steps, batch_size

    def __call__(self, current: DirectionSet, extra: int) -> DirectionSet:
        return _extend_topk_batch([(self, current, extra)])[0]

    def request(self, current: DirectionSet, extra: int) -> BatchedCall:
        return BatchedCall(("topk_extend", id(self.eigh), self.solver, self.steps), _extend_topk_batch,
                           (self, current, extra))


def _extend_topk_batch(payloads) -> list[DirectionSet]:
    """Extend many top-k direction sets together: ``(extender, current, extra)`` -> wider set each.

    The new width doubles the current one (at least ``extra`` more, at most C); the start block is the current
    Ritz vectors plus columns from a width-dependent seed, refined like a first fit (no full refit unless the
    width reaches C, where the result is the exact full eigensolve).
    """
    results: list[DirectionSet | None] = [None] * len(payloads)
    groups: dict[tuple, list[int]] = {}
    for position, (extender, _current, _extra) in enumerate(payloads):
        groups.setdefault((id(extender.eigh), extender.solver, extender.steps, extender.batch_size),
                          []).append(position)
    for positions in groups.values():
        first = payloads[positions[0]][0]
        matrices, totals, widths, starts, seeds = [], [], [], [], []
        for position in positions:
            extender, current, extra = payloads[position]
            channels = int(extender.matrix.shape[1])
            width = min(channels, current.width + max(int(extra), current.width))
            matrices.append(extender.matrix)
            totals.append(extender.total)
            widths.append(width)
            starts.append(current.rows.T)
            seeds.append((extender.seed + width) % 2**63)
        extended = topk_directions(matrices, totals, widths, starts, seeds, eigh=first.eigh, margin=0,
                                   steps=first.steps, align=1, batch_size=first.batch_size, solver=first.solver)
        for position, result in zip(positions, extended):
            results[position] = result
    return results  # type: ignore[return-value]


def stored_energy(matrix: torch.Tensor, rows16: torch.Tensor, rank: int, tile_rows: int) -> float:
    """Direct-residual energy of the actual stored FP16 rows and FP32 coefficients (explicit IEEE FP32 products)."""
    return run_one(stored_energy_routine(matrix, rows16, rank, tile_rows))


@torch.no_grad()
def span_energy(coefficients: torch.Tensor, rows16: torch.Tensor, source_energy: float) -> float:
    """Ideal energy of the orthogonal projection onto span(rows) given ``P = A Q^T``."""
    if source_energy <= 0:
        return 1.0
    rows = rows16.double()
    factor, info = torch.linalg.cholesky_ex(rows @ rows.T)
    if int(info) != 0:
        raise IntegrityError("stored basis rows are singular")
    solved = torch.linalg.solve_triangular(factor, coefficients.double().T, upper=False)
    return float(solved.square().sum()) / source_energy


class _KeyWork:
    """Mutable per-matrix planning state (decision device)."""

    def __init__(self, key, context, matrix, target, binding, parent_rows16, base_source_id):
        self.key, self.context, self.matrix, self.target = key, context, matrix, target
        self.binding, self.parent_rows16, self.base_source_id = binding, parent_rows16, base_source_id
        self.energy = None
        self.scaled = None
        self.counters = {
            "sketch_count": 0, "sketch_matmuls": 0, "sketch_qr": 0, "prediction_count": 0,
            "candidate_fit_count": 0, "gram_count": 0, "eigh_count": 0, "verification_count": 0,
            "growth_steps": 0, "trim_verifications": 0, "trimmed_buckets": 0,
            "certify_projection_count": 0, "prefix_verification_count": 0,
            "residual_fit_count": 0, "orthogonalization_count": 0, "refit_count": 0,
            "warm_start_count": 0, "subspace_extensions": 0, "batched_orthogonalizations": 0,
            "orthogonalization_fallbacks": 0,
        }
        self.verifications: list[dict[str, Any]] = []
        self.predictions: list[dict[str, Any]] = []
        self.observations: list[FitObservation] = []
        self.plan: MatrixPlan | None = None
        self.certified: float | None = None
        self.certify_projection = None
        self.residual = None
        self.residual_total = None
        self.residual_target = None
        self.sketches: dict[str, dict[str, Any]] = {}
        self.directions: dict[str, DirectionSet] = {}
        self.prediction_state: dict[str, dict[str, Any]] = {}
        self.augment_result = None
        self.batched_orthogonal = None
        self.refit_reason = None
        self.estimate = None
        self.share_refit = False


class _BatchedExtension(Exception):
    """Growth needs more batched-orthogonalized directions than were computed."""


ORTHOGONALIZERS = ("householder", "batched")


class LatestMethodPlanner:
    """Plans one complete chunk (or recache range) on one device.

    ``orthogonalizer`` selects the numerical execution of the augmentation
    re-orthogonalization (not part of the method fingerprint, like the
    eigensolver): ``"householder"`` is the per-matrix reference (FP32
    Householder QR of the inherited span and of every candidate direction);
    ``"batched"`` computes the same leading QR factor of each matrix's first
    candidate directions with batched CholeskyQR2 in one launch group. Any
    matrix whose batched factor shows a dependent or near-dependent column, or
    whose growth needs more directions than were batched, is recomputed with
    the reference (counted as ``orthogonalization_fallbacks``).

    Host reads are batched across matrices (:mod:`reelcache.deferred_reads`):
    every per-matrix decision step (energy, certify, share prefix, residual,
    verify/grow/trim, labels) is a routine run in lockstep with the chunk's
    other matrices, so a chunk costs a few dozen device-to-host reads instead
    of thousands of per-matrix synchronizations. Kernels, host scalars and the
    order of decisions per matrix are those of the sequential reference, so
    decisions, ranks, energies and rows are identical. ``synchronize_stages``
    restores device-synchronized stage timing (profiling only).
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        """wl-base (W1c): a fixed-rank configuration (``RankMethodConfig.fixed_policy``) gets
        :class:`_FixedRankPlanner`; every other configuration this class, unchanged."""
        config = args[0] if args else kwargs.get("config")
        if cls is LatestMethodPlanner and isinstance(config, RankMethodConfig) and config.fixed_policy is not None:
            cls = _FixedRankPlanner
        return super().__new__(cls)

    def __init__(self, config: RankMethodConfig, view: RegistryView, *, device: Any = "cpu",
                 eigh: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
                 eigh_label: str | None = None, orthogonalizer: str = "householder",
                 synchronize_stages: bool = False, max_working_bytes: int | None = None):
        if max_working_bytes is not None and (type(max_working_bytes) is not int or max_working_bytes < 1):
            raise ConfigurationError("max_working_bytes must be None or a positive byte count")
        if not isinstance(config, RankMethodConfig) or not isinstance(view, RegistryView):
            raise ConfigurationError("planning requires a RankMethodConfig and a pinned RegistryView")
        if view.anchor_rank != config.anchor_rank:
            raise ConfigurationError("registry anchor rank differs from the method's anchor rank")
        if orthogonalizer not in ORTHOGONALIZERS:
            raise ConfigurationError(f"orthogonalizer must be one of {ORTHOGONALIZERS}")
        if type(synchronize_stages) is not bool:
            raise ConfigurationError("synchronize_stages must be boolean")
        self.config, self.view = config, view
        self.orthogonalizer = orthogonalizer
        self.synchronize_stages = synchronize_stages
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ConfigurationError("the latest-method reference runs on CPU or CUDA")
        self.eigh = default_eigh if eigh is None else eigh
        self.eigh_label = ("torch_linalg_eigh" if eigh is None else eigh_label) or "caller_supplied"
        self.max_working_bytes = max_working_bytes
        if max_working_bytes is not None:
            from .gpu_fitting import BoundedEigh
            if isinstance(self.eigh, BoundedEigh):
                # Half of the working set for the eigensolver's Gram copies and workspace (same solver and sub-batch
                # policy: matrices solved one per call -- the default from n = 4096, e.g. LongCat -- or in another
                # fixed sub-batch get bitwise the eigenpairs of the unbounded planner).
                self.eigh = self.eigh.derive(memory_fraction=1.0, total_memory=max(1, max_working_bytes // 2))

    # -- public -------------------------------------------------------------
    @torch.no_grad()
    def plan(self, sources: Mapping[MatrixKey, torch.Tensor], contexts: Mapping[MatrixKey, TensorContext],
             bindings: Mapping[MatrixKey, ActiveBasisBinding],
             parent_rows: Mapping[MatrixKey, torch.Tensor | None], *, chunk_id: str,
             source_ids: Mapping[MatrixKey, str]) -> ChunkPlan:
        started = time.perf_counter()
        require_fp32_matmul(self.device, self.config.matmul_precision)
        clock = _Clock(self.device, synchronize=self.synchronize_stages)
        eigh_calls_before = len(getattr(self.eigh, "calls", ()))
        self._sketch_batches = []
        keys = sorted(sources, key=lambda key: (key.layer, key.target))
        if set(keys) != set(contexts) or set(keys) != set(bindings) or set(keys) != set(parent_rows) \
                or set(keys) != set(source_ids):
            raise ConfigurationError("planning inputs must cover the same matrix keys")
        with clock.time("transfer"):
            works = self._admit(keys, sources, contexts, bindings, parent_rows, source_ids)
        groups, memory = self._working_groups(works)
        for group in groups:
            self._plan_group(clock, group, chunk_id)
            # New rows return to the host in one transfer per group; the group's working tensors are released
            # before the next group starts (bounded working set).
            pending = [work.plan for work in group
                       if work.plan.new_rows is not None and work.plan.new_rows.device.type != "cpu"]
            if pending:
                for plan, rows in zip(pending, download([plan.new_rows for plan in pending])):
                    plan.new_rows = rows
            for work in group:
                work.scaled = work.residual = work.certify_projection = None
                work.directions.clear()
        plans = {work.key: work.plan for work in works}
        counters: dict[str, int] = {}
        for work in works:
            for name, value in work.counters.items():
                counters[name] = counters.get(name, 0) + value
        for decision in DECISIONS:
            counters[f"decision_{decision}"] = sum(plan.decision == decision for plan in plans.values())
        stages = clock.finish()
        result = ChunkPlan(plans, stages, counters, str(self.device), self.eigh_label,
                           self.view.sequence, time.perf_counter() - started)
        result.orthogonalizer = self.orthogonalizer
        result.memory = memory
        # Eigensolver calls split into sub-batches (cuSOLVER size limit or memory bound; same solver).
        result.batches.extend({"stage": "eigh", **call} for call in getattr(self.eigh, "calls", ())[eigh_calls_before:]
                              if call["sub_batch"] < call["batch"])
        result.batches.extend(self._sketch_batches)
        return result

    def _plan_group(self, clock: _Clock, works: list[_KeyWork], chunk_id: str) -> None:
        """Every planning stage for one group of matrices (the whole chunk unless a working-set bound applies)."""
        with clock.time("energy"):
            energies = run_lockstep(source_energy_routine(work.matrix, tile_rows=self.config.tile_rows)
                                    for work in works)
            for work, energy in zip(works, energies):
                work.energy = energy
        regimes: dict[int, str] = {}
        candidates = []
        for index, work in enumerate(works):
            if work.energy.zero_input:
                self._plan_zero(work)
                regimes[index] = "zero"
            elif work.target == 1.0:
                regimes[index] = "fresh"
            elif self.config.reuse and work.parent_rows16 is not None:
                candidates.append(index)
            else:
                regimes[index] = "fresh"
        # Certify, then share a verified prefix or prepare the residual (one lockstep for every
        # inheriting matrix; their stage time is reported under "certify").
        with clock.time("certify"):
            outcomes = run_lockstep(self._certify_and_decide(works[index]) for index in candidates)
        regimes.update(zip(candidates, outcomes))
        fresh_jobs = [work for index, work in enumerate(works) if regimes[index] == "fresh"]
        residual_jobs = [work for index, work in enumerate(works) if regimes[index] == "residual"]
        self._fit_jobs(clock, fresh_jobs, residual_jobs, chunk_id, refit=False)
        with clock.time("verify"):
            run_lockstep(self._plan_fresh(
                work, decision="diagnostic_full_width" if work.target == 1.0 else "fresh",
                reason="target_is_full_width_diagnostic" if work.target == 1.0
                else "no_active_basis" if work.parent_rows16 is None else "reuse_disabled")
                for work in fresh_jobs)
        if self.orthogonalizer == "batched" and residual_jobs:
            self._orthogonalize_batch(clock, residual_jobs)
        with clock.time("augment_verify"):
            run_lockstep(self._plan_augment(work) for work in residual_jobs)
        refits = [work for work in residual_jobs if work.plan is None]
        if refits:
            self._fit_jobs(clock, refits, [], chunk_id, refit=True)
            for work in refits:
                work.counters["refit_count"] += 1
            with clock.time("refit_verify"):
                run_lockstep(self._plan_fresh(work, decision="refit", reason=work.refit_reason, refit=True)
                             for work in refits)

    def _matrix_working_bytes(self, work: _KeyWork) -> int:
        """Upper estimate of one matrix's planning tensors outside the eigensolver.

        FP32 scaled copy, FP32 residual, FP32 projection coefficients (rank <= C), three FP32 ``C x C`` blocks (Gram,
        its symmetrized copy, fitted directions) and three FP64 measurement tiles.
        """
        rows, channels = (int(size) for size in work.matrix.shape)
        return 3 * 4 * rows * channels + 3 * 4 * channels * channels + 3 * 8 * min(rows, self.config.tile_rows) * channels

    def _working_groups(self, works: list[_KeyWork]) -> tuple[list[list[_KeyWork]], dict[str, Any] | None]:
        """Consecutive matrix groups whose planning tensors fit ``max_working_bytes`` (one group when unbounded).

        A matrix's plan does not depend on its group whenever its eigensolves do not depend on the batch they are
        solved in (per-matrix products, the registry pinned for the whole chunk): on the CPU, and on CUDA when the
        eigensolver fixes the sub-batch (one matrix per call from n = 4096 by default, e.g. LongCat). Then bounded
        and unbounded plans are bitwise identical; otherwise (CUDA, n < 4096, whole-batch calls) a matrix's
        eigenpairs follow the size class of its group's call (last-bit differences).
        """
        if self.max_working_bytes is None or not works:
            return [works], None
        per_matrix = max(self._matrix_working_bytes(work) for work in works)
        size = max(1, min(len(works), (self.max_working_bytes // 2) // per_matrix))
        groups = [works[start:start + size] for start in range(0, len(works), size)]
        return groups, {"max_working_bytes": self.max_working_bytes, "eigh_bytes": self.max_working_bytes // 2,
                        "matrix_bytes_estimate": per_matrix, "group_size": size, "groups": len(groups),
                        "estimated_peak_bytes": size * per_matrix + self.max_working_bytes // 2}

    def _sketch_stack_bytes(self) -> int | None:
        if self.device.type != "cuda":
            return None
        return int(SKETCH_MEMORY_FRACTION * torch.cuda.get_device_properties(self.device).total_memory)

    def _admit(self, keys, sources, contexts, bindings, parent_rows, source_ids) -> list[_KeyWork]:
        """Validate and place the chunk's matrices and active rows (one finiteness read, one row upload).

        Errors follow the sequential reference: key by key, a key's
        nonfinite values before its row/binding mismatch.
        """
        works, failure, pending = [], None, None
        for key in keys:
            try:
                context = contexts[key]
                if not isinstance(context, TensorContext) or context.key != key:
                    raise ConfigurationError("planning context key mismatch")
                if context.key_space != ("pre_rope" if key.target == "K" else "none"):
                    raise ConfigurationError("only normalized pre-RoPE K and direct V are factorized")
                matrix = validate_matrix(sources[key], context, check_values=False).to(self.device)
            except ConfigurationError as exc:
                failure = exc
                break
            try:
                rows = parent_rows[key]
                binding = bindings[key]
                if (rows is None) != (binding.version_ref is None):
                    raise ConfigurationError("active basis rows and binding disagree")
                if rows is not None and (rows.dtype != torch.float16 or rows.ndim != 2
                                         or rows.shape[1] != context.channels
                                         or rows.shape[0] != binding.total_rank):
                    raise ConfigurationError("active basis rows do not match their binding")
            except ConfigurationError as exc:
                failure, pending = exc, matrix
                break
            works.append(_KeyWork(key, context, matrix, self.config.targets.for_target(key),
                                  binding, rows, source_ids[key]))
        checked = [work.matrix for work in works] + ([] if pending is None else [pending])
        if checked:
            (finite,) = read_host([(torch.stack([finite_flag(matrix) for matrix in checked]),)])[0]
            for ok in finite:
                if not ok:
                    raise ConfigurationError("lossy input must be finite and representable in FP32")
        if failure is not None:
            raise failure
        inheriting = [work for work in works if work.parent_rows16 is not None]
        for work, rows in zip(inheriting, upload([work.parent_rows16 for work in inheriting], self.device)):
            work.parent_rows16 = rows
        return works

    # -- decisions (routines: generators run in lockstep, see reelcache.deferred_reads) -------------
    def _verify(self, work: _KeyWork, rows16: torch.Tensor, rank: int, purpose: str):
        energy = yield from stored_energy_routine(work.matrix, rows16, rank, self.config.tile_rows,
                                                  scale=work.energy.scale,
                                                  products=_accel_b.verify_products(  # accel B2 (opt-in)
                                                      self.config.measurement_products, work.matrix))
        work.counters["verification_count"] += 1
        work.verifications.append({"purpose": purpose, "rank": rank, "stored_energy": energy})
        return energy

    def _passes(self, work: _KeyWork, energy: float, target: float | None = None) -> bool:
        return energy >= (work.target if target is None else target) + self.config.decision_band

    def _plan_zero(self, work: _KeyWork) -> None:
        rows = None
        if work.parent_rows16 is None:
            rows = torch.zeros(1, work.context.channels, dtype=torch.float16)
            rows[0, 0] = 1.0
        work.plan = MatrixPlan(
            work.key, "zero_input", "zero_matrix_represented_exactly", work.target,
            work.binding.version_ref if rows is None else None, rows, 1, 0, 0, 1.0, (), (),
            self._work(work), None, None, zero_input=True, sets_active=False,
            refit_debt=self._tracked_debt(work.binding.refit_debt),
        )

    def _tracked_debt(self, debt: float) -> float | None:
        """A plan's refit debt: tracked by the cumulative rule only (the paper rule's plans carry None)."""
        return None if self.config.refit_rule == "paper" else float(debt)

    def _cumulative_estimate(self, work: _KeyWork, candidate: dict[str, Any], *, kind: str,
                             fields: Mapping[str, Any]) -> dict[str, Any]:
        """``reelcache.refit_estimate.v2`` of the cumulative rule for an augmentation (``kind="augmented"``) or a
        shared chunk (``kind="shared"``, ``refit_debt_on_share``); ``debt_after`` is the debt the chunk leaves."""
        binding = work.binding
        fresh_rank, debt = binding.fresh_rank, binding.refit_debt
        fresh = None if fresh_rank is None else estimate_bytes(self.config, work.context, fresh_rank, fresh_rank,
                                                                binding)
        excess = cost = None
        refit = False
        refinement = {}
        if fresh is None:
            reason, after = "no_fresh_rank_estimate", debt
        else:
            hysteresis = self.config.refit_hysteresis
            test, excess, cost = refit_debt_test(debt, candidate, fresh, hysteresis=hysteresis)
            if hysteresis != 1.0:
                refinement = {"hysteresis": hysteresis, "threshold": hysteresis * cost}
            refit = self.config.refit and test
            if not self.config.refit:
                reason = "refit_disabled"
            elif debt > 0:
                reason = "refit_debt_reaches_cost" if refit else "refit_debt_below_cost"
            else:
                reason = f"{kind}_bytes_exceed_fresh" if refit else f"{kind}_bytes_within_fresh"
            after = 0.0 if refit else debt + max(0.0, excess)
        return {
            "schema": REFIT_ESTIMATE_V2, "rule": "cumulative", **fields, "r_fresh": fresh_rank,
            "fresh_rank_source": binding.fresh_chunk_id, kind: candidate, "fresh": fresh,
            "debt_before": debt, "excess": excess, "cost": cost, "debt_after": after, **refinement,
            "decision": "refit" if refit else ("augment" if kind == "augmented" else "share"), "reason": reason,
        }

    def _certify_and_decide(self, work: _KeyWork):
        """Certify the inherited basis; then share a verified prefix or prepare the residual fit.

        A shared chunk whose refit debt reached the cost (``refit_debt_on_share``) leaves without a plan and joins
        the residual jobs as a ``share_refit``: it has no residual to fit, so it skips the residual fit and the
        augmentation and is refitted with the chunk's other refits (one batched fit).
        """
        yield from self._certify(work)
        if work.certified >= work.target + self.config.decision_band:
            yield from self._plan_share(work)
            if work.share_refit:
                return "residual"
            return "share"
        yield from self._prepare_residual(work)
        return "residual"

    def _certify(self, work: _KeyWork):
        rows = work.parent_rows16
        projection = yield from projection_routine(work.matrix, rows, rank=rows.shape[0],
                                                   tile_rows=self.config.tile_rows, scale=work.energy.scale,
                                                   condition="fast", products=self.config.measurement_products)
        work.counters["certify_projection_count"] += 1
        work.certify_projection = projection
        work.certified = 1.0 if projection.zero_input else 1.0 - projection.relative_residual ** 2
        work.verifications.append({"purpose": "certify", "rank": int(rows.shape[0]),
                                   "stored_energy": work.certified})

    def _plan_share(self, work: _KeyWork):
        rows, total = work.parent_rows16, int(work.parent_rows16.shape[0])
        coefficients = work.certify_projection.coefficients
        if _accel_b.planner_fp32(coefficients):  # accel B2 (opt-in): FP32 squares accumulated in FP64
            estimate = (torch.sum(coefficients * coefficients, 0, dtype=torch.float64)
                        / work.energy.source_energy).cumsum(0)
        else:
            estimate = (coefficients.double().square().sum(0) / work.energy.source_energy).cumsum(0)
        prefixes = _legal_prefixes(total, self.config.bucket)
        (estimate,) = yield (estimate,)
        start = next((index for index, prefix in enumerate(prefixes)
                      if float(estimate[prefix - 1]) >= work.target + self.config.decision_band),
                     len(prefixes) - 1)
        chosen, energy = total, work.certified
        for prefix in prefixes[start:]:
            if prefix == total:
                break
            candidate = yield from self._verify(work, rows, prefix, "share_prefix")
            work.counters["prefix_verification_count"] += 1
            if self._passes(work, candidate):
                chosen, energy = prefix, candidate
                break
        debt = self._tracked_debt(work.binding.refit_debt)
        if self.config.refit_debt_on_share:
            shared = estimate_bytes(self.config, work.context, chosen, 0, work.binding)
            work.estimate = self._cumulative_estimate(work, shared, kind="shared",
                                                      fields={"r_view": chosen, "r_total": total})
            if work.estimate["decision"] == "refit":
                work.refit_reason = work.estimate["reason"]
                work.share_refit = True
                return
            debt = work.estimate["debt_after"]
        work.plan = MatrixPlan(
            work.key, "share", "inherited_basis_meets_target", work.target, work.binding.version_ref,
            None, chosen, 0, None, energy, (), (), self._work(work), work.estimate, None, sets_active=True,
            refit_debt=debt,
        )

    def _prepare_residual(self, work: _KeyWork):
        rows = work.parent_rows16.float()
        residual = work.matrix.float() - mm(work.certify_projection.coefficients, rows)
        fp32 = _accel_b.planner_fp32(residual)  # accel B2 (opt-in)
        energy, scale = yield (_accel_b.sumsq64(residual) if fp32 else residual.double().square().sum(),
                               residual.abs().max())
        if fp32 and scale:  # IEEE FP32 division: the same values as the FP64 quotient rounded to FP32
            work.residual = _accel_b.exact_div_fp32(residual, scale)
        else:
            work.residual = (residual.double() / scale).float() if scale else residual
        work.residual_total = 0.0 if not scale else energy / (scale * scale)
        remaining = max(1e-300, 1.0 - work.certified)
        work.residual_target = min(1.0, max(1e-12, (work.target - work.certified) / remaining))

    # -- fitting --------------------------------------------------------------
    @staticmethod
    def _scaled(work: _KeyWork) -> torch.Tensor:
        """FP32 ``A / max|A|`` for sketching and Gram products, created only for fitted matrices."""
        if work.scaled is None and _accel_b.planner_fp32(work.matrix, work.energy.scale):  # accel B2 (opt-in)
            work.scaled = _accel_b.exact_div_fp32(work.matrix, work.energy.scale)  # the same values (IEEE division)
        if work.scaled is None:
            work.scaled = (work.matrix.double() / work.energy.scale).float()
        return work.scaled

    def _fit_jobs(self, clock: _Clock, fresh: list[_KeyWork], residual: list[_KeyWork], chunk_id: str, *,
                  refit: bool) -> None:
        sketch_items = []
        for work in fresh:
            if work.target < 1.0:
                sketch_items.append((work, "fresh", self._scaled(work), work.energy.scaled_energy))
        for work in residual:
            if work.residual_total and work.residual_target < 1.0:
                sketch_items.append((work, "residual", work.residual, work.residual_total))
        with clock.time("refit_sketch" if refit else "sketch"):
            if sketch_items:
                results = anchor_sketches(
                    [item[2] for item in sketch_items], [item[3] for item in sketch_items],
                    [_seed(self.config, item[0].context, item[1], chunk_id, "sketch") for item in sketch_items],
                    anchor_rank=self.config.anchor_rank, power_iterations=self.config.power_iterations,
                    max_stack_bytes=self._sketch_stack_bytes(), record=self._sketch_batches,
                )
                for (work, regime, _, _), result in zip(sketch_items, results):
                    work.sketches[regime] = result
                    work.counters["sketch_count"] += 1
                    work.counters["sketch_matmuls"] += result["matmuls"]
                    work.counters["sketch_qr"] += result["qr"]
        with clock.time("predict"):
            for work, regime, _, _ in sketch_items:
                target = work.target if regime == "fresh" else work.residual_target
                warm = refit and regime == "fresh" and self.config.solver == "warm_start"
                self._predict(work, regime, target, warm)
        gram_items = []
        for work in fresh:
            if refit and self.config.solver == "warm_start" and work.parent_rows16 is not None:
                continue
            gram_items.append((work, "fresh", self._scaled(work), work.energy.scaled_energy))
        for work in residual:
            if not work.share_refit:
                gram_items.append((work, "residual", work.residual, work.residual_total))
        if gram_items:
            prefix = "refit_" if refit else ""
            if self.config.solver == "topk_rr":
                widths, starts, seeds = [], [], []
                for work, regime, _, _ in gram_items:
                    full = min(work.context.shape)
                    prediction = work.prediction_state.get(regime)
                    target = work.target if regime == "fresh" else work.residual_target
                    widths.append(full if target >= 1.0 else
                                  self.config.anchor_rank if prediction is None else prediction["initial_width"])
                    blocks = []
                    if regime == "fresh" and work.parent_rows16 is not None:
                        blocks.append(work.parent_rows16.float().T)
                    sketch = work.sketches.get(regime)
                    if sketch is not None:
                        blocks.append(sketch["sketch"])
                    starts.append(torch.cat(blocks, 1) if blocks else None)
                    seeds.append(_seed(self.config, work.context, regime, chunk_id, "topk"))
                directions = topk_directions(
                    [item[2] for item in gram_items], [item[3] for item in gram_items], widths, starts, seeds,
                    eigh=self.eigh, margin=self.config.topk_margin, steps=self.config.topk_steps,
                    align=self.config.bucket, batch_size=self.config.batch_size, solver=self.config.solver_id(),
                    timer=lambda stage: clock.time(prefix + stage),
                )
            else:
                directions = gram_directions(
                    [item[2] for item in gram_items], [item[3] for item in gram_items], eigh=self.eigh,
                    batch_size=self.config.batch_size, solver=self.config.solver_id(),
                    timer=lambda stage: clock.time(prefix + stage),
                )
            for (work, regime, _, _), result in zip(gram_items, directions):
                work.directions[regime] = result
                work.counters["gram_count"] += 1
                work.counters["eigh_count"] += 1
                work.counters["candidate_fit_count"] += 1
                if regime == "residual":
                    work.counters["residual_fit_count"] += 1
        if refit and self.config.solver == "warm_start":
            with clock.time("warm_start"):
                for work in fresh:
                    if work.parent_rows16 is None:
                        continue
                    prediction = work.prediction_state.get("fresh")
                    pad = self.config.bucket
                    if prediction is not None and prediction["initial_width"] > work.parent_rows16.shape[0]:
                        pad += prediction["initial_width"] - int(work.parent_rows16.shape[0])
                    work.directions["fresh"] = warm_start_directions(
                        self._scaled(work), work.energy.scaled_energy, work.parent_rows16,
                        steps=self.config.warm_steps,
                        pad=pad, seed=_seed(self.config, work.context, "fresh", chunk_id, "warm"),
                    )
                    work.counters["warm_start_count"] += 1
                    work.counters["candidate_fit_count"] += 1

    def _predict(self, work: _KeyWork, regime: str, target: float, warm: bool) -> None:
        sketch = work.sketches[regime]
        full = min(work.context.shape)
        anchor = sketch["captured_energy"]
        record: dict[str, Any] = {"regime": regime, "anchor_energy": anchor, "target": target,
                                  "sketch_width": sketch["width"]}
        if anchor >= target:
            record.update(kind="sketch_bucket", initial_width=min(full, self.config.anchor_rank),
                          predicted_rank=None, x=None)
        else:
            x = math.log((1.0 - anchor) / (1.0 - target))
            use = self.view.predict(self.config.predictor_key(work.context, regime, warm=warm), x)
            initial = _bucket(use.predicted_rank, self.config.bucket, full)
            clamped = not math.isfinite(use.predicted_rank) or use.predicted_rank > full
            record.update(kind="log_linear_clamped" if clamped else "log_linear", x=x,
                          predicted_rank=use.predicted_rank if math.isfinite(use.predicted_rank) else None,
                          initial_width=initial, predictor=use.to_dict())
        work.counters["prediction_count"] += 1
        work.prediction_state[regime] = record
        work.predictions.append(record)

    @staticmethod
    def _crossing(directions: DirectionSet, fraction: float):
        """:meth:`DirectionSet.crossing` as a routine (first hit by ``argmax`` instead of ``nonzero``)."""
        if directions.total <= 0:
            return 0
        hits = directions.energies.cumsum(0) >= fraction * directions.total
        found, first = yield (hits.any(), hits.to(torch.int32).argmax())
        return first + 1 if found else None

    def _observe(self, work: _KeyWork, regime: str, directions: DirectionSet, target: float, *,
                 warm: bool):
        label = yield from self._crossing(directions, target)
        prediction = work.prediction_state.get(regime)
        if (label is not None and prediction is not None and prediction["kind"] != "sketch_bucket"
                and label > self.config.anchor_rank and target < 1.0):
            use = prediction["predictor"]
            source_id = canonical_digest({"base": work.base_source_id, "regime": regime,
                                          "parent_version": work.binding.version_ref if regime == "residual" else None,
                                          "solver": self.config.solver_id(warm=warm)})
            work.observations.append(FitObservation(
                self.config.predictor_key(work.context, regime, warm=warm), source_id,
                prediction["anchor_energy"], target, prediction["x"], label,
                use["predicted_rank"], use["state"]["revision"],
            ))
        return label

    def _plan_fresh(self, work: _KeyWork, *, decision: str, reason: str, refit: bool = False):
        full = min(work.context.shape)
        directions = work.directions["fresh"]
        diagnostic = work.target == 1.0
        width = full if diagnostic else work.prediction_state["fresh"]["initial_width"]
        width, energy = yield from self._grow_trim(work, directions, width, full, work.target, None, diagnostic)
        warm = refit and self.config.solver == "warm_start" and work.parent_rows16 is not None
        directions = work.directions["fresh"]
        label = None
        if not diagnostic:
            label = yield from self._observe(work, "fresh", directions, work.target, warm=warm)
        # Device rows; plan() moves every new row block to the host in one transfer.
        rows = directions.rows[:width].to(torch.float16).contiguous()
        work.plan = MatrixPlan(
            work.key, decision, reason, work.target, None, rows, width, width, label, energy,
            tuple(work.observations), tuple(work.predictions), self._work(work), work.estimate, width,
            diagnostic=diagnostic, sets_active=True, refit_debt=self._tracked_debt(0.0),
        )

    def _grow_trim(self, work: _KeyWork, directions: DirectionSet, width: int, full: int, target: float,
                   prefix_rows: torch.Tensor | None, diagnostic: bool, regime: str = "fresh"):
        """Verify the predicted bucket, grow by one bucket while short, then trim."""
        bucket = self.config.bucket
        limit = full if prefix_rows is None else full - int(prefix_rows.shape[0])

        def rows_for(count: int):
            nonlocal directions
            while directions.width < count:
                extender = directions.extender
                if extender is None:
                    raise IntegrityError("fitted directions cannot be extended")
                if isinstance(extender, _TopkExtender):
                    directions = yield extender.request(directions, bucket)  # batched with the round
                else:
                    directions = extender(directions, bucket)
                work.directions[regime] = directions
                work.counters["subspace_extensions"] += 1
            added = directions.rows[:count].to(torch.float16)
            return added if prefix_rows is None else torch.cat((prefix_rows, added), 0)

        failed = 0

        def check(count: int, purpose: str):
            nonlocal failed
            rows = yield from rows_for(count)
            energy = yield from self._verify(work, rows, int(rows.shape[0]), purpose)
            if not self._passes(work, energy):
                failed = max(failed, count)
            return energy

        width = min(width, limit)
        energy = yield from check(width, "predicted_bucket")
        if diagnostic:
            return width, energy
        while not self._passes(work, energy) and width < limit:
            width = min(limit, width + bucket)
            work.counters["growth_steps"] += 1
            energy = yield from check(width, "growth")
        if not self._passes(work, energy):
            raise RankTargetUnreachable(
                f"{work.key}: target {work.target} is unreachable at {width} directions "
                f"(stored energy {energy:.9f})", self._work(work),
            )
        crossing = yield from self._crossing(directions, max(0.0, target - self.config.trim_slack))
        if crossing is not None:
            candidate = max(_bucket(crossing, bucket, limit), min(limit, failed + bucket) if failed else 0)
            while candidate < width:
                trial = yield from check(candidate, "trim")
                work.counters["trim_verifications"] += 1
                if self._passes(work, trial):
                    work.counters["trimmed_buckets"] += (width - candidate + bucket - 1) // bucket
                    width, energy = candidate, trial
                    break
                candidate = min(width, candidate + bucket)
        return width, energy

    def _plan_augment(self, work: _KeyWork):
        if work.share_refit:
            return  # a shared chunk refitted by the debt rule: no augmentation candidate
        parent = work.parent_rows16
        parent_rank, channels = int(parent.shape[0]), work.context.channels
        full = min(work.context.shape)
        room = full - parent_rank
        if room <= 0 or not work.residual_total:
            work.refit_reason = "inherited_basis_has_no_room" if room <= 0 else "residual_has_no_energy"
            return
        directions = work.directions["residual"]
        limit = min(room, directions.width)
        batched, work.batched_orthogonal = work.batched_orthogonal, None
        prediction = work.prediction_state.get("residual")
        start = self.config.anchor_rank if prediction is None else prediction["initial_width"]
        # Top-k residual directions (solver="topk_rr") cover the predicted bucket plus a margin; growth past them
        # extends the residual subspace and continues on the bucket grid instead of declaring the target unreachable.
        while True:
            if batched is not None:
                added, count = batched
                batched = None
                adjusted = DirectionSet(added, directions.energies[:count], directions.total, False,
                                        directions.solver)
                adjusted.extender = _raise_batched_extension
                independent, cap = count, limit
            else:
                adjusted, independent = self._orthogonalize_reference(work, parent, directions, limit)
                cap = independent
                if independent < 1:
                    work.refit_reason = "residual_directions_inside_inherited_span"
                    return
            try:
                try:
                    width, energy = yield from self._grow_trim(work, adjusted, min(start, independent),
                                                               parent_rank + cap, work.residual_target, parent,
                                                               False, regime="residual_adjusted")
                except RankTargetUnreachable:
                    if self._residual_extendable(directions, room):
                        directions, limit = yield from self._extend_residual(work, directions, room)
                        start = cap + self.config.bucket
                        continue
                    work.refit_reason = "augmentation_cannot_reach_target"
                    return
            except _BatchedExtension:
                # Growth went past the batched directions: redo this matrix with the reference.
                work.counters["orthogonalization_fallbacks"] += 1
                adjusted, independent = self._orthogonalize_reference(work, parent, directions, limit)
                if independent < 1:
                    work.refit_reason = "residual_directions_inside_inherited_span"
                    return
                try:
                    width, energy = yield from self._grow_trim(work, adjusted, min(start, independent),
                                                               parent_rank + independent, work.residual_target,
                                                               parent, False, regime="residual_adjusted")
                except RankTargetUnreachable:
                    if self._residual_extendable(directions, room):
                        directions, limit = yield from self._extend_residual(work, directions, room)
                        start = independent + self.config.bucket
                        continue
                    work.refit_reason = "augmentation_cannot_reach_target"
                    return
            break
        label = yield from self._observe(work, "residual", directions, work.residual_target, warm=False)
        augmented = estimate_bytes(self.config, work.context, parent_rank + width, width, work.binding)
        if self.config.refit_rule == "paper":
            fresh_rank = work.binding.fresh_rank
            fresh = None if fresh_rank is None else estimate_bytes(
                self.config, work.context, fresh_rank, fresh_rank, work.binding)
            refit = bool(self.config.refit and fresh is not None and augmented["total_bytes"] > fresh["total_bytes"])
            work.estimate = {
                "schema": REFIT_ESTIMATE_V1, "rule": "B_aug > B_fresh (current chunk only)",
                "r_parent": parent_rank, "r_added": width, "r_fresh": fresh_rank,
                "fresh_rank_source": work.binding.fresh_chunk_id, "augmented": augmented, "fresh": fresh,
                "decision": "refit" if refit else "augment",
                "reason": ("no_fresh_rank_estimate" if fresh is None
                           else "refit_disabled" if not self.config.refit
                           else "augmented_bytes_exceed_fresh" if refit else "augmented_bytes_within_fresh"),
            }
            debt = None
        else:
            work.estimate = self._cumulative_estimate(work, augmented, kind="augmented",
                                                      fields={"r_parent": parent_rank, "r_added": width})
            refit = work.estimate["decision"] == "refit"
            debt = work.estimate["debt_after"]
        if refit:
            # The verified residual fit still trains the residual learner; the chunk refits fresh.
            work.refit_reason = work.estimate["reason"]
            work.directions.pop("fresh", None)
            return
        # Device rows; plan() moves every new row block to the host in one transfer.
        rows = adjusted.rows[:width].to(torch.float16).contiguous()
        work.plan = MatrixPlan(
            work.key, "augment", "residual_fit_appended", work.target, work.binding.version_ref, rows,
            parent_rank + width, width, label, energy, tuple(work.observations), tuple(work.predictions),
            self._work(work), work.estimate, None, sets_active=True, refit_debt=debt,
        )

    @staticmethod
    def _residual_extendable(directions: DirectionSet, room: int) -> bool:
        """Top-k residual directions that may still be extended before an augmentation is declared unreachable."""
        return not directions.complete and directions.extender is not None and directions.width < room

    def _extend_residual(self, work: _KeyWork, directions: DirectionSet, room: int):
        extender = directions.extender
        if isinstance(extender, _TopkExtender):
            extended = yield extender.request(directions, self.config.bucket)  # batched with the round
        else:
            extended = extender(directions, self.config.bucket)
        work.directions["residual"] = extended
        work.counters["subspace_extensions"] += 1
        return extended, min(room, extended.width)

    def _orthogonalize_reference(self, work: _KeyWork, parent: torch.Tensor,
                                 directions: DirectionSet, limit: int) -> tuple[DirectionSet, int]:
        """Per-matrix reference: FP32 Householder QR of the inherited span, two Gram-Schmidt
        passes and a Householder QR of every candidate (error ~1e-6, far below FP16 rounding)."""
        inherited = torch.linalg.qr(parent.float().T, mode="reduced").Q
        candidates = directions.rows[:limit].float().T
        for _ in range(2):
            candidates = candidates - mm(inherited, mm(inherited.T, candidates))
        orthogonal, triangular = torch.linalg.qr(candidates, mode="reduced")
        diagonal = triangular.diagonal().abs()
        independent = int((diagonal > 1e-3).cumprod(0).sum())
        signs = torch.sign(triangular.diagonal()[:independent]).clamp_min(0) * 2 - 1
        added = (orthogonal[:, :independent] * signs).T.float()
        work.counters["orthogonalization_count"] += 1
        return DirectionSet(added, directions.energies[:independent], directions.total, False,
                            directions.solver), independent

    def _orthogonalize_batch(self, clock: _Clock, works: Sequence[_KeyWork]) -> None:
        """Batched CholeskyQR2 of the inherited spans and of each matrix's leading candidates.

        Mathematically the leading columns of the reference's positive-diagonal
        QR factor; matrices with a Cholesky breakdown or a candidate column at
        or below the reference's 1e-3 independence threshold fall back to the
        reference (their factor would differ only in near-dependent columns).
        """
        eligible = []
        for work in works:
            parent = work.parent_rows16
            parent_rank = int(parent.shape[0])
            room = min(work.context.shape) - parent_rank
            if room <= 0 or not work.residual_total:
                continue
            directions = work.directions["residual"]
            limit = min(room, directions.width)
            prediction = work.prediction_state.get("residual")
            initial = self.config.anchor_rank if prediction is None else prediction["initial_width"]
            count = min(limit, max(4 * self.config.bucket, initial + 2 * self.config.bucket))
            eligible.append((work, parent_rank, count))
        if not eligible:
            return
        with clock.time("augment_orthogonalize"):
            groups: dict[tuple, list] = {}
            for item in eligible:
                groups.setdefault((item[0].context.channels, item[0].parent_rows16.device), []).append(item)
            for (channels, device), members in groups.items():
                for start in range(0, len(members), self.config.batch_size):
                    self._orthogonalize_group(members[start:start + self.config.batch_size], channels, device)

    def _orthogonalize_group(self, members, channels: int, device) -> None:
        """CholeskyQR2 of each matrix's inherited span and leading candidates (explicit IEEE FP32, per matrix:
        a matrix's result never depends on the others), with one host read for the group's breakdown flags."""
        flags, outputs = [], []
        for work, parent_rank, count in members:
            inherited = work.parent_rows16.float().T.contiguous()
            candidates = work.directions["residual"].rows[:count].float().T.contiguous()
            failed = torch.zeros((), dtype=torch.bool, device=device)
            for _ in range(2):
                factor, info = torch.linalg.cholesky_ex(mm(inherited.T, inherited), upper=True)
                failed = failed | (info != 0)
                factor = torch.where(info != 0, torch.eye(parent_rank, device=device), factor)
                inherited = trsm_right_upper(factor, inherited)
            for _ in range(2):
                candidates = candidates - mm(inherited, mm(inherited.T, candidates))
            orthogonal, diagonal = candidates, None
            for step in range(2):
                factor, info = torch.linalg.cholesky_ex(mm(orthogonal.T, orthogonal), upper=True)
                failed = failed | (info != 0)
                if step == 0:
                    diagonal = factor.diagonal()
                factor = torch.where(info != 0, torch.eye(count, device=device), factor)
                orthogonal = trsm_right_upper(factor, orthogonal)
            # Independence uses the first-pass diagonal (the column norms after removing earlier columns).
            flags.append(failed | (diagonal <= 2e-3).any() | ~torch.isfinite(orthogonal).all())
            outputs.append(orthogonal)
        failed_host = read_host([tuple(flags)])[0]
        for (work, _, count), failed, orthogonal in zip(members, failed_host, outputs):
            if failed:
                work.counters["orthogonalization_fallbacks"] += 1
                continue
            work.counters["orthogonalization_count"] += 1
            work.counters["batched_orthogonalizations"] += 1
            work.batched_orthogonal = (orthogonal.T.contiguous(), count)

    def _work(self, work: _KeyWork) -> dict[str, Any]:
        return {
            "counters": dict(work.counters), "verifications": list(work.verifications),
            "certified_energy": work.certified, "residual_target": work.residual_target,
            "source_scale": None if work.energy is None else work.energy.scale,
        }


class _FixedRankPlanner(LatestMethodPlanner):
    """wl-base (W1c): :class:`LatestMethodPlanner` of the fixed-rank arms (``RankMethodConfig.fixed_policy``), which
    ``LatestMethodPlanner(...)`` constructs for such a configuration (see :meth:`LatestMethodPlanner.__new__`), so every
    construction site (commit jobs, the GPU fitter's ``plan_chunk``, the commit-path warm-up) plans them unchanged.
    The fixed arms never read an active basis (reuse is off): their matrices are admitted without parent rows."""

    def _admit(self, keys, sources, contexts, bindings, parent_rows, source_ids) -> list[_KeyWork]:
        empty = ActiveBasisBinding()
        return super()._admit(keys, sources, contexts, {key: empty for key in keys}, {key: None for key in keys},
                              source_ids)

    def _plan_group(self, clock: _Clock, works: list[_KeyWork], chunk_id: str) -> None:
        self._plan_fixed_group(clock, works, chunk_id)

    def _plan_fixed_group(self, clock: _Clock, works: list[_KeyWork], chunk_id: str) -> None:
        """Fixed-rank arms: every matrix gets a fresh basis of a configured width, with no energy target (the achieved
        energy is measured and recorded), no learner (nothing is predicted or observed) and no sharing.

        ``fixed_basis="identity"`` (the HEVC-only baseline): the shared full-width identity rows, no fit (its stored
        energy before the codec is exactly 1). Otherwise (the uniform-rank ablation, the SVD baseline): one fit per
        matrix with the configured solver (the paper method's, e.g. top-k Rayleigh-Ritz) at exactly
        ``min(fixed rank, N, C)`` directions, whose stored FP16 rows are measured like any plan's."""
        with clock.time("energy"):
            energies = run_lockstep(source_energy_routine(work.matrix, tile_rows=self.config.tile_rows)
                                    for work in works)
            for work, energy in zip(works, energies):
                work.energy = energy
        fitted = []
        for work in works:
            if work.energy.zero_input:
                self._plan_zero(work)
            elif self.config.fixed_policy == "identity":
                self._plan_identity(work)
            else:
                fitted.append(work)
        if not fitted:
            return
        widths = [min(self.config.fixed_rank_for(work.key.target), min(work.context.shape)) for work in fitted]
        scaled = [self._scaled(work) for work in fitted]
        totals = [work.energy.scaled_energy for work in fitted]
        if self.config.solver == "topk_rr":
            directions = topk_directions(
                scaled, totals, widths, [None] * len(fitted),
                [_seed(self.config, work.context, "fresh", chunk_id, "topk") for work in fitted], eigh=self.eigh,
                margin=self.config.topk_margin, steps=self.config.topk_steps, align=self.config.bucket,
                batch_size=self.config.batch_size, solver=self.config.solver_id(), timer=clock.time)
        else:
            directions = gram_directions(scaled, totals, eigh=self.eigh, batch_size=self.config.batch_size,
                                         solver=self.config.solver_id(), timer=clock.time)
        for work, result in zip(fitted, directions):
            work.directions["fresh"] = result
            work.counters["gram_count"] += 1
            work.counters["eigh_count"] += 1
            work.counters["candidate_fit_count"] += 1
        with clock.time("verify"):
            run_lockstep(self._plan_fixed(work, width) for work, width in zip(fitted, widths))

    def _plan_fixed(self, work: _KeyWork, width: int):
        directions = work.directions["fresh"]
        if directions.width < width:
            raise IntegrityError(f"{work.key}: the fit returned {directions.width} directions, fewer than the fixed "
                                 f"rank {width}")
        rows = directions.rows[:width].to(torch.float16).contiguous()
        energy = yield from self._verify(work, rows, width, "fixed_rank")
        work.plan = MatrixPlan(
            work.key, "fresh", FIXED_RANK_REASON, work.target, None, rows, width, width, None, energy, (), (),
            self._work(work), None, width, diagnostic=True, sets_active=True, refit_debt=self._tracked_debt(0.0),
        )

    def _plan_identity(self, work: _KeyWork) -> None:
        tokens, channels = work.context.shape
        if tokens < channels:
            raise ConfigurationError(f"{work.key}: the identity basis needs at least as many tokens as channels "
                                     f"({tokens} < {channels})")
        work.plan = MatrixPlan(
            work.key, "fresh", IDENTITY_BASIS_REASON, work.target, None, identity_rows(channels), channels, channels,
            None, 1.0, (), (), self._work(work), None, channels, diagnostic=True, sets_active=True,
            refit_debt=self._tracked_debt(0.0),
        )


def _raise_batched_extension(_current: DirectionSet, _extra: int) -> DirectionSet:
    raise _BatchedExtension()


def plan_chunk(config: RankMethodConfig, view: RegistryView, sources, contexts, bindings, parent_rows, *,
               chunk_id: str, source_ids, device: Any = "cpu", eigh=None, eigh_label=None,
               orthogonalizer: str = "householder") -> ChunkPlan:
    """Functional entry point for :class:`LatestMethodPlanner`."""
    return LatestMethodPlanner(config, view, device=device, eigh=eigh, eigh_label=eigh_label,
                               orthogonalizer=orthogonalizer).plan(
        sources, contexts, bindings, parent_rows, chunk_id=chunk_id, source_ids=source_ids,
    )
