"""Experimental, source-bound online ranks with atomic scalar-state candidates.

An accepted decision certifies only the declared pre-codec direct residual.
It is not an HEVC, perceptual-quality, or future-input guarantee. Preparation
never publishes state: the manager must atomically commit packets, decisions,
and ``next_state()`` under its existing head compare-and-swap.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import torch

from .errors import CapacityError, ConfigurationError, IntegrityError, StateError, UnsupportedError
from .factorization import (
    DetectorSpec,
    DetectorAcquisitionError,
    Factors,
    RankFitError,
    RankProjectionError,
    acquire_anchor_directions,
    fit_rank_directions,
    project_rank_directions,
    rank_projection_workspace_bytes,
    tensor_sha256,
    validate_matrix,
)
from .types import CodecSpec, MatrixKey, ModelIdentity, TensorContext, canonical_dtype, json_copy


POLICY_SCHEMA = "reelcache.rank_policy.v1"
STATE_SCHEMA = "reelcache.rank_state.v1"
DECISION_SCHEMA = "reelcache.rank_decision.v1"
PREPARED_SCHEMA = "reelcache.prepared_rank_matrix.v1"
PREDICTOR_SCHEMA = "reelcache.rank_predictor_profile.v1"
_UNSET = object()
_ITEM_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}
_COUNTERS = (
    "decomposition_count", "svd_count", "eigh_count", "eigvalsh_count", "qr_count",
    "projection_count", "matmul_count", "source_norm_count",
    "gram_build_count", "source_hash_count", "factor_hash_count", "direction_verification_count",
)
_PROJECTION_COUNTERS = (
    "cholesky_count", "cholesky_inverse_count", "coefficient_projection_count",
    "residual_evaluation_count", "residual_tile_count", "basis_device_copy_count",
)
_ALL_COUNTERS = (*_COUNTERS, *_PROJECTION_COUNTERS, "source_copy_count")
_TIMINGS = (
    "decomposition_seconds", "svd_seconds", "eigh_seconds", "qr_seconds",
    "range_seconds", "projection_seconds", "source_norm_seconds", "fit_seconds",
    "projection_coefficient_seconds", "projection_guard_seconds",
    "condition_seconds", "basis_round_seconds",
    "source_copy_seconds",
    "basis_device_copy_seconds",
)
_STUDY_SOURCE_FIELDS = (
    "case_id", "prompt_group", "split", "seed", "cell", "layer", "target",
    "boundary", "trajectory_id", "evidence_kind", "n_rows", "n_channels",
    "n_layers", "source_dtype", "source_domain",
)
_NATIVE_SOURCE_FIELDS = (
    "protocol_id", "launch_closure_id", "cell_id", "group_id", "fixture",
    "native_execution", "request_id", "arm_id", "branch_id", "parent_head",
)
_SOURCE_CHUNK_FIELDS = (
    "case_id", "prompt_group", "split", "seed", "cell", "boundary", "trajectory_id",
    "evidence_kind", *_NATIVE_SOURCE_FIELDS,
)
_FORBIDDEN_SOURCE_MEASUREMENTS = {
    "basis_sha256", "coefficients_sha256", "spectrum_sha256", "oracle_source_sha256",
    "oracle_observation", "eigenvalues_scaled", "energy_scaled", "scale_exponent",
    "rank", "estimated_rank", "exact_rank", "r95", "r99", "rank_brackets",
    "guard_relative_residual", "guard_passed",
}
_EXTRA_INTEGER_METRICS = {
    "fit_attempts", "guard_attempts", "finite_width_misses", "view_expansions",
    "refits", "reuse_attempts", "reuse_rejections", "failed_oracle_calls",
    "failed_projection_calls",
}
_NONNEGATIVE_METRICS = {
    "source_norm", "source_energy", "residual_norm", "projection_relative_residual",
    "stored_svd_relative_residual", "post_hevc_relative_residual", "basis_condition_bound",
    "fitted_energy_fraction", "estimated_cdf_before", "estimated_cdf_at",
}


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str, minimum: float = 0) -> float:
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value < minimum
    ):
        raise ConfigurationError(f"{name} must be finite and >= {minimum}")
    return float(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be nonempty text")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ConfigurationError(f"{name} must be a lowercase SHA256")
    return value


def _freeze(value: Any) -> Any:
    plain = json_copy(value)

    def freeze(item):
        if isinstance(item, dict):
            return MappingProxyType({key: freeze(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        return item

    return freeze(plain)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(json_copy(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _label(key: MatrixKey) -> str:
    return f"{key.layer}:{key.target}"


def _key(label: str) -> MatrixKey:
    if not isinstance(label, str) or re.fullmatch(r"(0|[1-9][0-9]*):[KV]", label) is None:
        raise ConfigurationError("matrix keys must be '<layer>:K' or '<layer>:V'")
    layer, target = label.split(":")
    return MatrixKey(int(layer), target)


def _matrix_map(values: Mapping, keys: tuple[MatrixKey, ...], name: str) -> dict:
    if not isinstance(values, Mapping) or set(values) != set(keys):
        raise ConfigurationError(f"{name} must cover every model K/V matrix")
    return {key: values[key] for key in keys}


def _schema_fields(cls, payload: Mapping, schema: str) -> dict:
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {item.name for item in dataclasses.fields(cls)}
        or payload.get("schema_version") != schema
    ):
        raise ConfigurationError(f"invalid {schema} fields")
    return dict(payload)


def _source_provenance(
    payload: Mapping[str, Any], context: TensorContext, *, chunk_id: str,
    parent_head: str | None, policy_hash: str, input_sha256: str | None = None,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ConfigurationError("source provenance must be a finite JSON mapping")
    provenance = json_copy(payload)
    if set(provenance) & _FORBIDDEN_SOURCE_MEASUREMENTS:
        raise ConfigurationError("source provenance cannot override measured spectrum, guard, or factor identities")
    study = any(name in provenance for name in (
        "case_id", "prompt_group", "trajectory_id", "evidence_kind", "protocol_id",
        "launch_closure_id", "cell_id", "group_id", "fixture", "native_execution",
    ))
    if study:
        if not set(_STUDY_SOURCE_FIELDS) <= set(provenance):
            raise ConfigurationError("study source provenance must bind complete case/geometry identities before fitting")
        for name in ("case_id", "prompt_group", "cell", "trajectory_id"):
            _text(provenance[name], f"source {name}")
        if provenance["split"] not in {"development", "calibration", "heldout"}:
            raise ConfigurationError("study source split is invalid")
        if provenance["evidence_kind"] not in {"native", "synthetic_fixture"}:
            raise ConfigurationError("study source evidence must distinguish native from synthetic")
        for name in ("seed", "boundary"):
            _integer(provenance[name], f"source {name}")
        if provenance["evidence_kind"] == "native":
            if not set(_NATIVE_SOURCE_FIELDS) <= set(provenance):
                raise ConfigurationError("native study provenance requires its protocol, launch closure and execution bindings")
            for name in ("protocol_id", "launch_closure_id"):
                _digest(provenance[name], f"source {name}")
            for name in ("request_id", "arm_id", "branch_id"):
                _text(provenance[name], f"source {name}")
            if provenance["fixture"] is not False or provenance["native_execution"] is not True:
                raise ConfigurationError("native study provenance cannot relabel fixture execution")
            if (provenance["boundary"] == 0) != (provenance["parent_head"] is None):
                raise ConfigurationError("native boundary and parent history bindings disagree")
        elif (
            ("fixture" in provenance and provenance["fixture"] is not True)
            or ("native_execution" in provenance and provenance["native_execution"] is not False)
        ):
            raise ConfigurationError("synthetic study provenance cannot claim native execution")
        for alias, canonical in (("cell_id", "cell"), ("group_id", "prompt_group")):
            if alias in provenance and provenance[alias] != provenance[canonical]:
                raise ConfigurationError("study census aliases disagree with the predeclared case identity")
    for name in ("original_bytes_sha256", "context_sha256"):
        if name in provenance:
            _digest(provenance[name], f"source {name}")
    expected = {
        "layer": context.key.layer, "target": context.key.target,
        "n_rows": context.token_count, "n_channels": context.channels,
        "n_layers": context.model.layers, "source_dtype": context.dtype,
        "source_domain": "normalized_pre_rope" if context.key.target == "K" else "direct",
        "policy_hash": policy_hash, "chunk_id": chunk_id, "parent_head": parent_head,
        "context_sha256": _fingerprint(context.to_dict()),
        "token_start": context.token_start, "token_end": context.token_end,
        "shape": list(context.shape), "dtype": context.dtype, "key_space": context.key_space,
    }
    if input_sha256 is not None:
        expected["original_bytes_sha256"] = input_sha256
    for name, value in expected.items():
        if name in provenance and (
            provenance[name] != value or (type(value) is int and type(provenance[name]) is not int)
        ):
            raise ConfigurationError(f"source provenance {name} conflicts with the authoritative tensor context")
    for name in ("context", "tensor_context"):
        if name in provenance and TensorContext.from_dict(provenance[name]) != context:
            raise ConfigurationError("source provenance tensor coordinates differ from the fitted original matrix")
    return _freeze(provenance)


def _validate_metric_scalars(
    metrics: Mapping[str, Any], *, allow_unknown_times: bool = False,
) -> None:
    """Validate telemetry fields, without treating arbitrary source labels as metrics."""
    for name, value in metrics.items():
        if name == "seconds" or name.endswith("_seconds"):
            if value is not None or not allow_unknown_times:
                _finite(value, f"metric {name}")
        elif name in _EXTRA_INTEGER_METRICS:
            _integer(value, f"metric {name}")
        elif name.endswith("_bytes"):
            if value is not None:
                _integer(value, f"metric {name}")
        elif name in _NONNEGATIVE_METRICS and value is not None:
            _finite(value, f"metric {name}")
    for name in ("oracle_timings", "oracle_failure_metrics", "oracle_solver", "projection_failure_metrics"):
        nested = metrics.get(name)
        if nested is None:
            continue
        if not isinstance(nested, Mapping):
            raise ConfigurationError(f"metric {name} must be an object or explicit null")
        _validate_metric_scalars(nested, allow_unknown_times=allow_unknown_times)
        for counter in _ALL_COUNTERS:
            if counter in nested and nested[counter] is not None:
                _integer(nested[counter], f"{name}.{counter}")


def _starting_rank(value: float, buckets: tuple[int, ...]) -> int:
    return next((rank for rank in buckets if rank >= math.ceil(value)), buckets[-1])


def _anchor_prediction(
    *, layer: int, template: tuple[float, ...], anchors: tuple[int, ...],
    current: Mapping[int, float], previous: Mapping[int, float],
    current_baselines: tuple[float | None, ...],
    previous_baselines: Mapping[int, float | None], mode: str,
    margin: float = 0.0, multiplier: float = 1.0,
    uniform_rank: int | None = None,
) -> tuple[float, tuple[dict[str, Any], ...]]:
    """The single ratio implementation used by calibration, fitting and replay."""
    if uniform_rank is not None:
        return float(uniform_rank), ()
    sources = []
    ratios = []
    if layer not in anchors and mode != "none":
        for anchor in anchors:
            current_allowed = mode == "staged" or (mode == "causal" and anchor < layer)
            if current_allowed and anchor in current:
                value, baseline, availability = current[anchor], current_baselines[anchor], "current"
            elif mode in {"causal", "lagged"} and anchor in previous:
                value, baseline, availability = previous[anchor], previous_baselines.get(anchor), "prior_committed"
            else:
                continue
            if baseline is None:
                continue
            rank = _finite(value, "anchor estimate")
            denominator = max(1.0, _finite(baseline, "detector anchor baseline"))
            ratio = rank / denominator
            ratios.append(ratio)
            sources.append({
                "layer": anchor, "estimated_rank": rank, "detector_baseline": denominator,
                "ratio": ratio, "availability": availability,
            })
    predicted = max(1.0, template[layer] * (max(ratios) if ratios else 1.0) * multiplier + margin)
    return predicted, tuple(sources)


@dataclass(frozen=True)
class RankPredictorProfile:
    """A calibration-only predictor configuration, not a qualified codec policy.

    Source/model/geometry and resource admission are supplied only when these
    profiles are bound into a RankPolicy. Training groups are provenance, never
    input features. Fixed-uniform rank already includes its calibration margin.
    """

    cell: str
    target: str
    tau: float
    history_regime: str
    execution_mode: str
    layer_template: tuple[float, ...]
    detector_baselines: Mapping[str, tuple[float | None, ...]]
    anchor_layers: tuple[int, ...]
    rank_margin: int
    rank_buckets: tuple[int, ...]
    policy_kind: str
    uniform_rank: int | None
    training_groups: tuple[str, ...]
    rank_statistic: str = "approximate_spectral_crossing"
    detector_fingerprint: str = field(default_factory=lambda: DetectorSpec().fingerprint)
    schema_version: str = PREDICTOR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PREDICTOR_SCHEMA:
            raise ConfigurationError("unsupported predictor profile schema")
        if self.rank_statistic != "approximate_spectral_crossing":
            raise ConfigurationError("v1 calibration anchors require raw approximate spectral crossings")
        _digest(self.detector_fingerprint, "fixed detector fingerprint")
        _text(self.cell, "predictor cell")
        if self.target not in {"K", "V"} or self.tau not in (0.95, 0.99):
            raise ConfigurationError("predictor profiles require K/V and tau 0.95/0.99")
        if self.history_regime not in {"startup", "later"}:
            raise ConfigurationError("predictor history must be startup or later")
        if self.execution_mode not in {"causal_lagged", "whole_clean_staged"}:
            raise ConfigurationError("predictor availability mode is invalid")
        if not isinstance(self.layer_template, (list, tuple)) or not self.layer_template:
            raise ConfigurationError("predictor requires a complete layer template")
        template = tuple(_finite(value, "layer template") for value in self.layer_template)
        object.__setattr__(self, "layer_template", template)
        if not isinstance(self.rank_buckets, (tuple, list)) or not self.rank_buckets:
            raise ConfigurationError("predictor requires rank buckets")
        buckets = tuple(_integer(rank, "rank bucket", 1) for rank in self.rank_buckets)
        if buckets != tuple(sorted(set(buckets))) or max(template) > buckets[-1]:
            raise ConfigurationError("rank buckets/template exceed the declared profile rank domain")
        object.__setattr__(self, "rank_buckets", buckets)
        if not isinstance(self.detector_baselines, Mapping) or set(self.detector_baselines) != {"startup", "later"}:
            raise ConfigurationError("detector baselines must include startup and later")
        baselines = {}
        for regime in ("startup", "later"):
            values = self.detector_baselines[regime]
            if not isinstance(values, (list, tuple)) or len(values) != len(template):
                raise ConfigurationError("detector baseline width differs from the layer template")
            baselines[regime] = tuple(
                None if value is None else _finite(value, "detector baseline", 1)
                for value in values
            )
            if any(value is not None and value > buckets[-1] for value in baselines[regime]):
                raise ConfigurationError("detector baseline exceeds profile rank dimensions")
        object.__setattr__(self, "detector_baselines", MappingProxyType(baselines))
        if not isinstance(self.anchor_layers, (tuple, list)):
            raise ConfigurationError("anchor layers must be a sequence")
        anchors = tuple(_integer(layer, "anchor layer") for layer in self.anchor_layers)
        if (
            len(anchors) not in {0, 1, 2, 4} or len(set(anchors)) != len(anchors)
            or any(layer >= len(template) for layer in anchors)
        ):
            raise ConfigurationError("a predictor must use 0/1/2/4 distinct in-range anchors")
        object.__setattr__(self, "anchor_layers", tuple(sorted(anchors)))
        _integer(self.rank_margin, "rank_margin")
        if self.policy_kind not in {"fixed_uniform", "fixed_layerwise", "anchors"}:
            raise ConfigurationError("unknown predictor policy_kind")
        if (self.policy_kind == "anchors") != bool(anchors):
            raise ConfigurationError("predictor kind and anchor budget disagree")
        if any(baselines[self.history_regime][layer] is None for layer in anchors):
            raise ConfigurationError("chosen current anchor lacks a calibrated detector baseline")
        if self.policy_kind == "fixed_uniform":
            if type(self.uniform_rank) is not int or self.uniform_rank not in buckets:
                raise ConfigurationError("uniform predictor needs an admissible frozen rank")
        elif self.uniform_rank is not None:
            raise ConfigurationError("non-uniform predictor cannot declare a uniform rank")
        if (
            not isinstance(self.training_groups, (tuple, list)) or not self.training_groups
            or any(not isinstance(group, str) or not group.strip() for group in self.training_groups)
            or len(set(self.training_groups)) != len(self.training_groups)
        ):
            raise ConfigurationError("predictor requires distinct training-group provenance")
        object.__setattr__(self, "training_groups", tuple(sorted(self.training_groups)))

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return json_copy({item.name: getattr(self, item.name) for item in dataclasses.fields(self)})

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RankPredictorProfile:
        if not isinstance(payload, Mapping) or "detector_fingerprint" not in payload:
            raise ConfigurationError("frozen predictor profile must bind its fixed detector recipe")
        return cls(**_schema_fields(cls, payload, PREDICTOR_SCHEMA))


def profile_builder(parameters: Mapping[str, Any]) -> RankPredictorProfile:
    """Callback for rank_analysis.calibrate_selector; no corpus/model access."""
    if not isinstance(parameters, Mapping):
        raise ConfigurationError("predictor parameters must be a mapping")
    values = dict(parameters)
    for alternate, canonical in (
        ("detector_anchor_baselines", "detector_baselines"),
        ("safety_margin", "rank_margin"),
    ):
        if alternate in values:
            if canonical in values:
                raise ConfigurationError("ambiguous duplicated predictor parameter")
            values[canonical] = values.pop(alternate)
    optional = {"rank_statistic", "detector_fingerprint"}
    expected = {item.name for item in dataclasses.fields(RankPredictorProfile)} - {"schema_version", *optional}
    if not expected <= set(values) or not set(values) <= expected | optional:
        raise ConfigurationError("calibration predictor parameters differ from the deployed schema")
    return RankPredictorProfile(**values)


def rank_predictor(
    profile: RankPredictorProfile, *, layer: int,
    current_anchors: Mapping[int, float], previous_anchors: Mapping[int, float],
    previous_regime: str | None, chunk_ordinal: int | None = None,
) -> int:
    """Calibration callback returning the same admissible initial rank as online."""
    if not isinstance(profile, RankPredictorProfile):
        raise ConfigurationError("rank_predictor requires a RankPredictorProfile")
    _integer(layer, "prediction layer")
    if layer >= len(profile.layer_template):
        raise ConfigurationError("prediction layer is outside the profile")
    if not isinstance(current_anchors, Mapping) or not isinstance(previous_anchors, Mapping):
        raise ConfigurationError("anchor features must be scalar mappings")
    if previous_regime not in {None, "startup", "later"} or (previous_anchors and previous_regime is None):
        raise ConfigurationError("previous-anchor features require their actual history regime")
    if chunk_ordinal is None:
        chunk_ordinal = 1 if profile.history_regime == "startup" else (
            3 if previous_regime == "later" else 2
        )
    _integer(chunk_ordinal, "chunk_ordinal", 1)
    if previous_regime == "later" and chunk_ordinal < 3:
        raise ConfigurationError("a later previous chunk requires a later current ordinal")
    stamp = 1 if previous_regime == "startup" else chunk_ordinal - 1
    current = {
        MatrixKey(anchor, profile.target): current_anchors[anchor]
        for anchor in profile.anchor_layers
        if anchor in current_anchors and layer not in profile.anchor_layers
        and (profile.execution_mode == "whole_clean_staged" or anchor < layer)
    }
    previous = {
        MatrixKey(anchor, profile.target): previous_anchors[anchor]
        for anchor in profile.anchor_layers if anchor in previous_anchors
    }
    return predict_rank(
        profile, MatrixKey(layer, profile.target), chunk_ordinal=chunk_ordinal,
        current_anchors=current, previous_anchors=previous,
        previous_anchor_chunks={key: stamp for key in previous},
    ).initial_rank


@dataclass(frozen=True)
class RankPrediction:
    predicted_rank: float
    initial_rank: int
    prediction_sources: tuple[Mapping[str, Any], ...]
    kind: str
    profile_fingerprint: str | None = None

    def __post_init__(self) -> None:
        _finite(self.predicted_rank, "predicted rank", 1)
        _integer(self.initial_rank, "initial rank", 1)
        object.__setattr__(self, "prediction_sources", _freeze(self.prediction_sources))

    @property
    def sources(self) -> tuple[Mapping[str, Any], ...]:
        return self.prediction_sources


@dataclass(frozen=True)
class RankPolicy:
    policy_id: str
    model: ModelIdentity
    geometry: tuple[int, int, int]
    source_dtypes: Mapping[MatrixKey, str]
    source_fingerprint: str
    geometry_fingerprint: str
    history_fingerprint: str
    calibration_fingerprint: str
    implementation_fingerprint: str
    energy_target: float
    template_ranks: Mapping[MatrixKey, float]
    allowed_ranks: Mapping[MatrixKey, tuple[int, ...]]
    codec: CodecSpec = field(default_factory=lambda: CodecSpec("svd", 1))
    anchors: tuple[MatrixKey, ...] = ()
    anchor_mode: str = "none"
    solver: str = "qr_q2"
    seed: int = 0
    prediction_margin: float = 1.0
    additive_margin: float = 0.0
    hysteresis_fraction: float = 0.0
    history_min: int = 1
    history_max: int = 3
    tile_rows: int = 1024
    oversample: int = 8
    fit_growth: float = 2.0
    max_fit_attempts: int = 4
    max_guard_attempts: int = 16
    max_fit_width: int | None = None
    source_limit_bytes: int = 1 << 30
    workspace_limit_bytes: int = 1 << 30
    solver_workspace_bytes: int = 64 << 20
    staging_limit_bytes: int = 0
    reuse_enabled: bool = False
    predictor_profiles: Mapping[str, RankPredictorProfile] = field(default_factory=dict)
    energy_targets: Mapping[str, float] | None = None
    anchor_statistic: str | None = None
    detector_spec: DetectorSpec = field(default_factory=DetectorSpec)
    schema_version: str = POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_SCHEMA:
            raise ConfigurationError("unsupported rank policy schema")
        _text(self.policy_id, "policy_id")
        if not isinstance(self.model, ModelIdentity):
            raise ConfigurationError("rank policy requires a ModelIdentity")
        if not isinstance(self.detector_spec, DetectorSpec):
            raise ConfigurationError("rank policy requires a typed fixed DetectorSpec")
        if not isinstance(self.geometry, (tuple, list)) or len(self.geometry) != 3:
            raise ConfigurationError("geometry must be frames, height, width")
        geometry = tuple(_integer(item, "geometry", 1) for item in self.geometry)
        object.__setattr__(self, "geometry", geometry)
        for name in (
            "source_fingerprint", "geometry_fingerprint", "history_fingerprint",
            "calibration_fingerprint", "implementation_fingerprint",
        ):
            _digest(getattr(self, name), name)
        energy = _finite(self.energy_target, "energy_target")
        if not 0 < energy <= 1:
            raise ConfigurationError("energy_target must be in (0, 1]")
        object.__setattr__(self, "energy_target", energy)
        if self.energy_targets is not None:
            if not isinstance(self.energy_targets, Mapping) or set(self.energy_targets) != {"K", "V"}:
                raise ConfigurationError("energy_targets must cover exactly K and V")
            targets = {key: _finite(value, "target energy") for key, value in self.energy_targets.items()}
            if any(not 0 < value <= 1 for value in targets.values()):
                raise ConfigurationError("target energies must be in (0, 1]")
            object.__setattr__(self, "energy_targets", MappingProxyType(targets))
        expected_statistic = "oracle_rank_upper" if self.solver == "exact_oracle" else "approximate_spectral_crossing"
        if self.anchor_statistic is None:
            object.__setattr__(self, "anchor_statistic", expected_statistic)
        elif self.anchor_statistic != expected_statistic:
            raise ConfigurationError("anchor statistic must match the declared detector, never substitute oracle labels")
        if not isinstance(self.codec, CodecSpec) or self.codec.kind == "raw":
            raise ConfigurationError("adaptive ranks require an explicitly lossy codec, never RAW fallback")
        dtypes = self.source_dtypes
        if isinstance(dtypes, Mapping) and set(dtypes) == {"K", "V"}:
            dtypes = {key: dtypes[key.target] for key in self.keys}
        dtypes = _matrix_map(dtypes, self.keys, "source_dtypes")
        object.__setattr__(self, "source_dtypes", MappingProxyType({
            key: canonical_dtype(value) for key, value in dtypes.items()
        }))
        templates = _matrix_map(self.template_ranks, self.keys, "template_ranks")
        object.__setattr__(self, "template_ranks", MappingProxyType({
            key: _finite(value, "template rank") for key, value in templates.items()
        }))
        ladders = _matrix_map(self.allowed_ranks, self.keys, "allowed_ranks")
        frozen_ladders = {}
        for key, values in ladders.items():
            if not isinstance(values, (tuple, list)) or not values:
                raise ConfigurationError("allowed ranks must be nonempty ordered sequences")
            ranks = tuple(_integer(rank, "allowed rank", 1) for rank in values)
            if tuple(sorted(set(ranks))) != ranks or ranks[-1] > min(self.tokens, self.model.channels_for(key)):
                raise ConfigurationError("allowed ranks must increase strictly within matrix dimensions")
            frozen_ladders[key] = ranks
        object.__setattr__(self, "allowed_ranks", MappingProxyType(frozen_ladders))
        if not isinstance(self.predictor_profiles, Mapping):
            raise ConfigurationError("predictor_profiles must be a mapping")
        profiles = dict(self.predictor_profiles)
        _integer(self.history_min, "history_min", 1)
        _integer(self.history_max, "history_max", 1)
        if profiles:
            regimes = ({"startup"} if self.history_min == 1 else set()) | (
                {"later"} if self.history_max >= 2 else set()
            )
            if set(profiles) != {f"{regime}:{target}" for regime in regimes for target in ("K", "V")}:
                raise ConfigurationError("predictor profiles must cover K/V in every supported history regime")
            for label, profile in profiles.items():
                if (
                    not isinstance(profile, RankPredictorProfile)
                    or label != f"{profile.history_regime}:{profile.target}"
                    or profile.tau != self.energy_for(profile.target)
                    or profile.rank_statistic != self.anchor_statistic
                    or profile.detector_fingerprint != self.detector_spec.fingerprint
                    or len(profile.layer_template) != self.model.layers
                    or any(
                        self.allowed_ranks[MatrixKey(layer, profile.target)] != profile.rank_buckets
                        for layer in range(self.model.layers)
                    )
                ):
                    raise ConfigurationError("predictor profile differs from policy target/energy/shape/rank domain")
                expected_mode = "staged" if profile.execution_mode == "whole_clean_staged" else "causal"
                if self.anchor_mode != expected_mode:
                    raise ConfigurationError("predictor profile availability differs from policy execution mode")
            if len({profile.cell for profile in profiles.values()}) != 1:
                raise ConfigurationError("one rank policy cannot combine different study cells")
            if self.prediction_margin != 1 or self.additive_margin != 0:
                raise ConfigurationError("profile margins cannot be silently stacked with policy margins")
        object.__setattr__(self, "predictor_profiles", MappingProxyType(profiles))
        if not isinstance(self.anchors, (tuple, list)):
            raise ConfigurationError("anchors must be a sequence of MatrixKey")
        anchors = tuple(self.anchors)
        if (
            any(not isinstance(key, MatrixKey) or key not in self.keys for key in anchors)
            or len(set(anchors)) != len(anchors)
        ):
            raise ConfigurationError("anchors must be unique policy keys")
        if profiles:
            state_keys = {
                MatrixKey(layer, profile.target)
                for profile in profiles.values() for layer in profile.anchor_layers
            }
            if set(anchors) != state_keys:
                raise ConfigurationError("policy anchors must be the frozen union of regime-specific anchor keys")
        elif any(sum(key.target == target for key in anchors) not in {0, 1, 2, 4} for target in ("K", "V")):
            raise ConfigurationError("anchor budgets must be 0, 1, 2, or 4 per target")
        object.__setattr__(self, "anchors", tuple(sorted(anchors)))
        if self.anchor_mode not in {"none", "staged", "causal", "lagged"}:
            raise ConfigurationError("unknown anchor availability mode")
        if self.anchor_mode == "none" and anchors:
            raise ConfigurationError("no-anchor mode cannot declare anchors")
        if self.solver not in {"qr_q2", "exact_oracle"}:
            raise ConfigurationError("solver must be qr_q2 or exact_oracle")
        if self.solver == "qr_q2" and any(self.energy_for(key) not in (0.95, 0.99) for key in anchors):
            raise ConfigurationError("fixed QR anchor features require .95 or .99 energy")
        _integer(self.seed, "seed")
        if self.seed >= 2**63:
            raise ConfigurationError("seed must be below 2**63")
        for name in (
            "history_min", "history_max", "tile_rows", "max_fit_attempts",
            "max_guard_attempts", "source_limit_bytes", "workspace_limit_bytes",
        ):
            _integer(getattr(self, name), name, 1)
        for name in ("oversample", "solver_workspace_bytes", "staging_limit_bytes"):
            _integer(getattr(self, name), name)
        if self.history_min > self.history_max:
            raise ConfigurationError("history support interval is reversed")
        if self.max_fit_width is not None:
            _integer(self.max_fit_width, "max_fit_width", 1)
            if any(self.max_fit_width < self.allowed_ranks[key][-1] for key in self.keys):
                raise ConfigurationError("max_fit_width cannot exclude declared allowed ranks")
        if type(self.reuse_enabled) is not bool:
            raise ConfigurationError("reuse_enabled must be boolean")
        if self.reuse_enabled:
            raise UnsupportedError("rank policy v1 binds approximate basis reuse OFF")
        for name, minimum in (
            ("prediction_margin", 1), ("additive_margin", 0),
            ("hysteresis_fraction", 0), ("fit_growth", 1),
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name, minimum))
        if self.fit_growth <= 1:
            raise ConfigurationError("fit_growth must exceed one")
        if any(self.source_bytes_for(key) > self.source_limit_bytes for key in self.keys):
            raise CapacityError("policy matrix exceeds its source allowance")
        if self.anchor_mode == "staged" and self.source_bytes_per_chunk > self.staging_limit_bytes:
            raise CapacityError("staged policy must admit the whole original clean chunk")

    @property
    def keys(self) -> tuple[MatrixKey, ...]:
        return tuple(MatrixKey(layer, target) for layer in range(self.model.layers) for target in ("K", "V"))

    @property
    def tokens(self) -> int:
        return math.prod(self.geometry)

    def source_bytes_for(self, key: MatrixKey) -> int:
        return self.tokens * self.model.channels_for(key) * _ITEM_BYTES[self.source_dtypes[key]]

    @property
    def source_bytes_per_chunk(self) -> int:
        return sum(self.source_bytes_for(key) for key in self.keys)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    @property
    def policy_hash(self) -> str:
        return self.fingerprint

    @property
    def identity(self) -> str:
        return self.fingerprint

    @property
    def reuse_threshold(self) -> None:
        return None

    def energy_for(self, key: MatrixKey | str) -> float:
        target = key.target if isinstance(key, MatrixKey) else key
        if target not in {"K", "V"}:
            raise ConfigurationError("energy target requires K or V")
        return self.energy_target if self.energy_targets is None else self.energy_targets[target]

    def active_anchors(self, history_chunks: int) -> tuple[MatrixKey, ...]:
        """At most 0/1/2/4 per target; ``anchors`` is the scalar-state union."""
        _integer(history_chunks, "history_chunks", 1)
        if not self.history_min <= history_chunks <= self.history_max:
            raise StateError("active anchors are outside policy history support")
        if not self.predictor_profiles:
            return self.anchors
        regime = "startup" if history_chunks == 1 else "later"
        return tuple(sorted(
            MatrixKey(layer, target)
            for target in ("K", "V")
            for layer in self.predictor_profiles[f"{regime}:{target}"].anchor_layers
        ))

    def factorization_seed_for(self, key: MatrixKey, token_start: int) -> int | None:
        if key not in self.keys:
            raise ConfigurationError("seed key is outside the policy")
        _integer(token_start, "token_start")
        if self.solver == "exact_oracle":
            return None
        if key in self.anchors:
            return self.detector_spec.seed_for(TensorContext(
                self.model, key, token_start, *self.geometry,
                "pre_rope" if key.target == "K" else "none", source_dtype=self.source_dtypes[key],
            ))
        return int(_fingerprint({
            "seed": self.seed, "layer": key.layer, "target": key.target, "token_start": token_start,
        })[:16], 16) % 2**63

    def predict_rank(
        self, key: MatrixKey, *, current_anchors: Mapping[MatrixKey, float],
        prior_state: RankState | None = None,
    ) -> RankPrediction:
        """Pure shared predictor for calibration-equivalent online fitting/replay."""
        return _policy_prediction(self, key, current_anchors=current_anchors, prior_state=prior_state)

    def validate_context(self, context: TensorContext) -> None:
        if (
            not isinstance(context, TensorContext) or context.model != self.model
            or context.key not in self.source_dtypes
            or (context.frames, context.height, context.width) != self.geometry
            or context.dtype != self.source_dtypes[context.key]
            or context.key_space != ("pre_rope" if context.key.target == "K" else "none")
        ):
            raise ConfigurationError("context differs from the bound rank-policy model/geometry/source domain")

    def validate_contexts(
        self, contexts: Mapping[MatrixKey, TensorContext], *,
        history_chunks: int | None = None,
    ) -> None:
        if not isinstance(contexts, Mapping) or set(contexts) != set(self.keys):
            raise ConfigurationError("rank policy contexts must cover every matrix")
        starts = set()
        for key in self.keys:
            context = contexts[key]
            self.validate_context(context)
            if context.key != key:
                raise ConfigurationError("context mapping key differs from its matrix")
            starts.add(context.token_start)
        if len(starts) != 1:
            raise ConfigurationError("rank policy contexts do not share one clean token range")
        if history_chunks is not None:
            _integer(history_chunks, "history_chunks", 1)
            if not self.history_min <= history_chunks <= self.history_max:
                raise StateError("clean chunk is outside the policy's bound history support")

    def spec_for(self, key: MatrixKey, rank: int) -> CodecSpec:
        if key not in self.allowed_ranks or type(rank) is not int or rank not in self.allowed_ranks[key]:
            raise ConfigurationError("resolved rank is not allowed by the policy")
        return dataclasses.replace(self.codec, rank=rank)

    def validate_decision(
        self, decision: RankDecision, context: TensorContext | None = None,
        spec: CodecSpec | None = None,
    ) -> None:
        if not isinstance(decision, RankDecision) or decision.policy_hash != self.fingerprint:
            raise IntegrityError("rank decision belongs to another policy")
        self.validate_context(decision.context)
        if context is not None and decision.context != context:
            raise IntegrityError("rank decision context mismatch")
        if decision.energy_target != self.energy_for(decision.key) or decision.solver != self.solver:
            raise IntegrityError("rank decision energy/solver mismatch")
        if decision.metrics.get("rank_statistic") != self.anchor_statistic:
            raise IntegrityError("rank decision anchor statistic differs from the bound detector")
        if (
            type(decision.metrics.get("tile_rows")) is not int
            or decision.metrics["tile_rows"] != self.tile_rows
            or decision.metrics.get("projection_convention") != "P=X.float()@B.float().T"
            or decision.metrics.get("guard_domain") != "original_source_vs_fp32_P_at_stored_fp16_B"
            or decision.metrics.get("guard_is_pre_codec") is not True
        ):
            raise IntegrityError("rank decision projection/guard recipe differs from the bound policy")
        if decision.status != "ACCEPTED" or not decision.guard_passed:
            raise StateError("an unresolved rank cannot be published")
        if self.solver == "qr_q2" and decision.key in self.anchors:
            acquisition = decision.metrics.get("anchor_acquisition")
            if (
                not isinstance(acquisition, Mapping)
                or acquisition.get("detector_fingerprint") != self.detector_spec.fingerprint
                or acquisition.get("source_sha256") != decision.input_sha256
                or acquisition.get("context") != _freeze(decision.context.to_dict())
                or acquisition.get("features", {}).get(str(decision.energy_target)) != decision.estimated_rank
                or decision.metrics.get("anchor_feature_rank") != decision.estimated_rank
                or decision.metrics.get("detector_fingerprint") != self.detector_spec.fingerprint
            ):
                raise IntegrityError("anchor decision differs from its fixed acquisition feature/source/recipe")
        expected = self.spec_for(decision.key, decision.rank)
        if decision.resolved_spec != expected or (spec is not None and spec != expected):
            raise IntegrityError("rank decision and resolved codec differ")

    def validate_history_decisions(
        self, decisions: Mapping[MatrixKey, RankDecision], prior_state: RankState,
    ) -> None:
        """Replay one complete decision set through the authoritative predictor."""
        if (
            not isinstance(decisions, Mapping) or set(decisions) != set(self.keys)
            or not isinstance(prior_state, RankState)
        ):
            raise IntegrityError("history validation requires every matrix decision and its prior scalar state")
        prior_state.validate(self)
        ordinal = prior_state.completed_chunks + 1
        if not self.history_min <= ordinal <= self.history_max:
            raise IntegrityError("decision history depth is outside the policy")
        contexts = {}
        identities = set()
        for key in self.keys:
            decision = decisions[key]
            self.validate_decision(decision)
            if decision.key != key or decision.estimated_rank is None:
                raise IntegrityError("history decision lacks its matrix's resolved detector statistic")
            if decision.metrics.get("chunk_ordinal") != ordinal:
                raise IntegrityError("decision chunk ordinal differs from the committed scalar history")
            if decision.metrics.get("prior_state_fingerprint") != prior_state.fingerprint:
                raise IntegrityError("decision did not use the declared prior scalar state")
            if (
                decision.metrics.get("factorization_seed")
                != self.factorization_seed_for(key, decision.context.token_start)
            ):
                raise IntegrityError("decision solver seed is not the policy's private deterministic seed")
            if prior_state.completed_chunks:
                if (
                    decision.context.token_start != prior_state.token_end
                    or decision.parent_head is None or decision.chunk_id == prior_state.last_chunk_id
                ):
                    raise IntegrityError("decision context is not the next committed clean boundary")
            elif decision.parent_head is not None:
                raise IntegrityError("first adaptive decision cannot have unbound prior factor history")
            identities.add((decision.chunk_id, decision.parent_head))
            contexts[key] = decision.context
        if len(identities) != 1:
            raise IntegrityError("matrix decisions do not belong to one atomic clean chunk")
        self.validate_contexts(contexts, history_chunks=ordinal)
        current = {key: float(decisions[key].estimated_rank) for key in self.anchors}
        for key in self.keys:
            decision = decisions[key]
            prediction = self.predict_rank(key, current_anchors=current, prior_state=prior_state)
            if (
                decision.predicted_rank != prediction.predicted_rank
                or decision.initial_rank != prediction.initial_rank
                or decision.prediction_sources != prediction.prediction_sources
                or decision.metrics.get("prediction_kind") != prediction.kind
                or decision.metrics.get("predictor_profile_fingerprint") != prediction.profile_fingerprint
            ):
                raise IntegrityError("decision prediction used incompatible history, baselines, or anchor availability")

    def to_dict(self) -> dict[str, Any]:
        result = {item.name: getattr(self, item.name) for item in dataclasses.fields(self)}
        result.update(
            model=self.model.to_dict(), geometry=list(self.geometry), codec=self.codec.to_dict(),
            anchors=[key.to_dict() for key in self.anchors],
            predictor_profiles={label: profile.to_dict() for label, profile in self.predictor_profiles.items()},
            detector_spec=self.detector_spec.to_dict(),
        )
        for name in ("source_dtypes", "template_ranks", "allowed_ranks"):
            result[name] = {_label(key): value for key, value in getattr(self, name).items()}
        return json_copy(result)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RankPolicy:
        if not isinstance(payload, Mapping) or "detector_spec" not in payload:
            raise ConfigurationError("rank policy must bind its fixed detector specification")
        values = _schema_fields(cls, payload, POLICY_SCHEMA)
        values["model"] = ModelIdentity.from_dict(values["model"])
        values["codec"] = CodecSpec.from_dict(values["codec"])
        values["detector_spec"] = DetectorSpec.from_dict(values["detector_spec"])
        values["anchors"] = tuple(MatrixKey.from_dict(key) for key in values["anchors"])
        values["predictor_profiles"] = {
            label: RankPredictorProfile.from_dict(profile)
            for label, profile in values["predictor_profiles"].items()
        }
        for name in ("source_dtypes", "template_ranks", "allowed_ranks"):
            values[name] = {_key(label): value for label, value in values[name].items()}
        return cls(**values)


@dataclass(frozen=True)
class RankState:
    policy_hash: str
    last_chunk_id: str | None = None
    parent_head: str | None = None
    token_end: int = 0
    completed_chunks: int = 0
    anchor_ranks: Mapping[MatrixKey, float] = field(default_factory=dict)
    anchor_chunks: Mapping[MatrixKey, int] = field(default_factory=dict)
    selected_ranks: Mapping[MatrixKey, int] = field(default_factory=dict)
    schema_version: str = STATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != STATE_SCHEMA:
            raise ConfigurationError("unsupported rank state schema")
        _digest(self.policy_hash, "state policy_hash")
        _integer(self.token_end, "token_end")
        _integer(self.completed_chunks, "completed_chunks")
        for name in ("last_chunk_id", "parent_head"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), name)
        for name in ("anchor_ranks", "anchor_chunks", "selected_ranks"):
            values = getattr(self, name)
            if not isinstance(values, Mapping) or any(not isinstance(key, MatrixKey) for key in values):
                raise ConfigurationError("rank state maps require MatrixKey keys")
            object.__setattr__(self, name, MappingProxyType(dict(values)))
        if set(self.anchor_ranks) != set(self.anchor_chunks):
            raise ConfigurationError("anchor values and history ages differ")
        for value in self.anchor_ranks.values():
            _finite(value, "saved anchor rank")
        for value in self.anchor_chunks.values():
            _integer(value, "anchor chunk", 1)
            if value > self.completed_chunks:
                raise ConfigurationError("saved anchor cannot come from the future")
        for value in self.selected_ranks.values():
            _integer(value, "saved selected rank", 1)
        if self.completed_chunks == 0:
            if self.last_chunk_id is not None or self.parent_head is not None or self.anchor_ranks or self.selected_ranks:
                raise ConfigurationError("empty rank state contains committed history")
        elif self.last_chunk_id is None or not self.selected_ranks or self.token_end < 1:
            raise ConfigurationError("committed rank state is incomplete")
        elif (self.completed_chunks == 1) != (self.parent_head is None):
            raise ConfigurationError("rank state parent identity differs from its history depth")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    @property
    def policy_identity(self) -> str:
        return self.policy_hash

    def validate(
        self, policy: RankPolicy, *, last_chunk_id: str | None = None,
        parent_head: str | None | object = _UNSET, token_end: int | None = None,
        completed_chunks: int | None = None,
    ) -> None:
        """Validate replayed scalar state against its policy and optional head facts."""
        if not isinstance(policy, RankPolicy) or self.policy_hash != policy.fingerprint:
            raise IntegrityError("rank state belongs to another policy")
        if self.completed_chunks:
            if set(self.selected_ranks) != set(policy.keys) or set(self.anchor_ranks) != set(policy.anchors):
                raise IntegrityError("rank state matrix/anchor coverage differs from policy")
            if not policy.history_min <= self.completed_chunks <= policy.history_max:
                raise IntegrityError("rank state is outside declared history support")
            for key in policy.keys:
                policy.spec_for(key, self.selected_ranks[key])
            if any(
                self.anchor_ranks[key] > min(policy.tokens, policy.model.channels_for(key))
                for key in policy.anchors
            ):
                raise IntegrityError("rank state anchor is outside source dimensions")
        if last_chunk_id is not None and self.last_chunk_id != last_chunk_id:
            raise IntegrityError("rank state last clean chunk mismatch")
        if parent_head is not _UNSET and self.parent_head != parent_head:
            raise IntegrityError("rank state parent head mismatch")
        if token_end is not None and self.token_end != _integer(token_end, "token_end"):
            raise IntegrityError("rank state token boundary mismatch")
        if completed_chunks is not None and self.completed_chunks != _integer(completed_chunks, "completed_chunks"):
            raise IntegrityError("rank state history depth mismatch")

    def to_dict(self) -> dict[str, Any]:
        result = {item.name: getattr(self, item.name) for item in dataclasses.fields(self)}
        for name in ("anchor_ranks", "anchor_chunks", "selected_ranks"):
            result[name] = {_label(key): value for key, value in sorted(getattr(self, name).items())}
        return json_copy(result)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RankState:
        values = _schema_fields(cls, payload, STATE_SCHEMA)
        for name in ("anchor_ranks", "anchor_chunks", "selected_ranks"):
            values[name] = {_key(label): value for label, value in values[name].items()}
        return cls(**values)


def bind_predictor_profiles(
    policy: RankPolicy, profiles: Mapping[str, RankPredictorProfile | Mapping[str, Any]],
) -> RankPolicy:
    """Bind frozen ``startup:K``/``later:V`` etc. profiles to a guarded policy.

    This binds *initial-rank prediction*, not a fixed-rank encoder or a
    quality-qualified production Profile. Static uniform/layerwise codec arms
    must still use their fixed CodecSpec maps instead of this adaptive guard.
    Each active regime has 0/1/2/4 anchors per target. Scalar state retains the
    frozen union across regimes. Union keys use fixed acquisition during their
    own encoding, and its directions are reused. The full union acquisition
    cost, not merely the current regime's active-anchor cost, must be charged.
    """
    if not isinstance(policy, RankPolicy) or not isinstance(profiles, Mapping) or not profiles:
        raise ConfigurationError("binding requires a RankPolicy and predictor-profile mapping")
    checked = {
        label: profile if isinstance(profile, RankPredictorProfile) else RankPredictorProfile.from_dict(profile)
        for label, profile in profiles.items()
    }
    modes = {profile.execution_mode for profile in checked.values()}
    if len(modes) != 1:
        raise ConfigurationError("one clean chunk cannot mix staged and causal predictor lifetimes")
    anchors = tuple(sorted({
        MatrixKey(layer, profile.target) for profile in checked.values() for layer in profile.anchor_layers
    }))
    templates, allowed, energies = {}, {}, {}
    for target in ("K", "V"):
        reference = checked.get(f"startup:{target}", checked.get(f"later:{target}"))
        if reference is None or len(reference.layer_template) != policy.model.layers:
            raise ConfigurationError("profile binding is missing target layers")
        energies[target] = reference.tau
        for layer in range(policy.model.layers):
            templates[MatrixKey(layer, target)] = reference.layer_template[layer]
            allowed[MatrixKey(layer, target)] = reference.rank_buckets
    return dataclasses.replace(
        policy, predictor_profiles=checked, anchors=anchors,
        anchor_mode="staged" if modes == {"whole_clean_staged"} else "causal",
        template_ranks=templates, allowed_ranks=allowed,
        prediction_margin=1.0, additive_margin=0.0,
        energy_targets=energies,
    )


def _policy_prediction(
    policy: RankPolicy, key: MatrixKey, *, current_anchors: Mapping[MatrixKey, float],
    prior_state: RankState | None,
) -> RankPrediction:
    prior = RankState(policy.fingerprint) if prior_state is None else prior_state
    if not isinstance(prior, RankState) or prior.policy_hash != policy.fingerprint:
        raise ConfigurationError("prediction state belongs to another policy")
    return predict_rank(
        policy, key, chunk_ordinal=prior.completed_chunks + 1,
        current_anchors=current_anchors, previous_anchors=prior.anchor_ranks,
        previous_anchor_chunks=prior.anchor_chunks,
    )


def predict_rank(
    policy: RankPolicy | RankPredictorProfile, key: MatrixKey, *,
    chunk_ordinal: int, current_anchors: Mapping[MatrixKey, float],
    previous_anchors: Mapping[MatrixKey, float],
    previous_anchor_chunks: Mapping[MatrixKey, int],
) -> RankPrediction:
    """Public pure prediction seam; no fabricated preparation or committed state.

    Both RankPreparation and the calibration callback use this function.
    Availability is checked before current future-anchor values are accessed.
    Prior scalar normalizers use each feature's recorded source regime.
    """
    ordinal = _integer(chunk_ordinal, "chunk_ordinal", 1)
    if (
        not isinstance(key, MatrixKey)
        or any(not isinstance(values, Mapping)
               for values in (current_anchors, previous_anchors, previous_anchor_chunks))
        or any(not isinstance(item, MatrixKey)
               for values in (current_anchors, previous_anchors, previous_anchor_chunks) for item in values)
        or set(previous_anchors) != set(previous_anchor_chunks)
    ):
        raise ConfigurationError("prediction requires MatrixKey-keyed scalar features and their source chunks")
    if any(
        _integer(chunk, "previous anchor chunk", 1) >= ordinal
        for chunk in previous_anchor_chunks.values()
    ):
        raise ConfigurationError("previous anchors cannot come from the current/future chunk")
    regime = "startup" if ordinal == 1 else "later"
    if isinstance(policy, RankPolicy):
        if key not in policy.keys:
            raise ConfigurationError("prediction key is outside the policy")
        if not policy.history_min <= ordinal <= policy.history_max:
            raise StateError("prediction is outside policy history support")
        active_keys = tuple(anchor for anchor in policy.active_anchors(ordinal) if anchor.target == key.target)
        profile = policy.predictor_profiles.get(f"{regime}:{key.target}")
        template = tuple(
            policy.template_ranks[MatrixKey(layer, key.target)] for layer in range(policy.model.layers)
        ) if profile is None else profile.layer_template
        mode = policy.anchor_mode
        margin = policy.additive_margin if profile is None else profile.rank_margin
        multiplier = policy.prediction_margin if profile is None else 1.0
        buckets = policy.allowed_ranks[key]
    elif isinstance(policy, RankPredictorProfile):
        profile = policy
        if key.target != profile.target or key.layer >= len(profile.layer_template) or profile.history_regime != regime:
            raise ConfigurationError("prediction key/current regime differs from the calibration profile")
        active_keys = tuple(MatrixKey(layer, key.target) for layer in profile.anchor_layers)
        template = profile.layer_template
        mode = "staged" if profile.execution_mode == "whole_clean_staged" else "causal"
        margin, multiplier, buckets = profile.rank_margin, 1.0, profile.rank_buckets
    else:
        raise ConfigurationError("predict_rank requires RankPolicy or RankPredictorProfile")
    anchor_layers = tuple(anchor.layer for anchor in active_keys)
    current = {}
    previous = {}
    previous_baselines = {}
    if key not in active_keys and (profile is None or profile.uniform_rank is None):
        for anchor in active_keys:
            current_allowed = mode == "staged" or (
                mode == "causal" and anchor.layer < key.layer
            )
            if current_allowed and anchor in current_anchors:
                current[anchor.layer] = current_anchors[anchor]
            elif anchor in previous_anchors:
                previous[anchor.layer] = previous_anchors[anchor]
                previous_regime = "startup" if previous_anchor_chunks[anchor] == 1 else "later"
                previous_baselines[anchor.layer] = (
                    template[anchor.layer] if profile is None else
                    profile.detector_baselines[previous_regime][anchor.layer]
                )
    predicted, raw_sources = _anchor_prediction(
        layer=key.layer, template=template, anchors=anchor_layers,
        current=current, previous=previous,
        current_baselines=template if profile is None else profile.detector_baselines[regime],
        previous_baselines=previous_baselines, mode=mode,
        margin=margin, multiplier=multiplier,
        uniform_rank=None if profile is None else profile.uniform_rank,
    )
    sources = []
    for feature in raw_sources:
        anchor = MatrixKey(feature["layer"], key.target)
        chunk = ordinal if feature["availability"] == "current" else previous_anchor_chunks[anchor]
        source = {
            "key": anchor.to_dict(), "estimated_rank": feature["estimated_rank"],
            "template_rank": template[anchor.layer], "ratio": feature["ratio"],
            "availability": feature["availability"], "history_age": ordinal - chunk,
            "chunk_ordinal": chunk,
        }
        if profile is not None:
            source.update(
                detector_baseline=feature["detector_baseline"],
                baseline_regime=regime if feature["availability"] == "current" else (
                    "startup" if chunk == 1 else "later"
                ),
            )
        sources.append(source)
    kind = (
        "fixed_uniform" if profile is not None and profile.uniform_rank is not None else
        "anchor_template" if key in active_keys else
        "max_available_anchor_ratio" if sources else "template"
    )
    return RankPrediction(
        predicted, _starting_rank(predicted, buckets), tuple(sources),
        kind, None if profile is None else profile.fingerprint,
    )


@dataclass(frozen=True)
class RankDecision:
    policy_hash: str
    chunk_id: str
    parent_head: str | None
    context: TensorContext
    input_sha256: str
    energy_target: float
    solver: str
    predicted_rank: float
    initial_rank: int
    status: str
    reason: str
    source_provenance: Mapping[str, Any] = field(default_factory=dict)
    rank: int | None = None
    resolved_spec: CodecSpec | None = None
    estimated_rank: int | None = None
    exact_rank: int | None = None
    rank_lower: int | None = None
    rank_upper: int | None = None
    estimate_status: str = "unresolved"
    fit_width: int = 0
    guard_threshold: float = 0.0
    guard_relative_residual: float | None = None
    guard_passed: bool = False
    zero_input: bool = False
    basis_sha256: str | None = None
    coefficients_sha256: str | None = None
    canonical_basis_rank: int = 0
    basis_bytes: int = 0
    coefficient_bytes: int = 0
    source_bytes: int = 0
    prediction_sources: tuple[Mapping[str, Any], ...] = ()
    attempts: tuple[Mapping[str, Any], ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = DECISION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != DECISION_SCHEMA:
            raise ConfigurationError("unsupported rank decision schema")
        _digest(self.policy_hash, "decision policy_hash")
        _digest(self.input_sha256, "input_sha256")
        _text(self.chunk_id, "chunk_id")
        _text(self.reason, "reason")
        if self.parent_head is not None:
            _text(self.parent_head, "parent_head")
        if not isinstance(self.context, TensorContext):
            raise ConfigurationError("rank decision requires TensorContext")
        if self.context.key_space != ("pre_rope" if self.key.target == "K" else "none"):
            raise ConfigurationError("rank decision source is not canonical pre-K/direct-V")
        object.__setattr__(self, "source_provenance", _source_provenance(
            self.source_provenance, self.context, chunk_id=self.chunk_id,
            parent_head=self.parent_head, policy_hash=self.policy_hash,
            input_sha256=self.input_sha256,
        ))
        energy = _finite(self.energy_target, "energy_target")
        if not 0 < energy <= 1 or self.guard_threshold != math.sqrt(1 - energy):
            raise ConfigurationError("guard threshold differs from the declared energy target")
        if self.solver not in {"qr_q2", "exact_oracle"}:
            raise ConfigurationError("rank decision has an unknown solver")
        _finite(self.predicted_rank, "predicted_rank")
        _integer(self.initial_rank, "initial_rank", 1)
        if self.status not in {"ACCEPTED", "UNRESOLVED", "OVER_BUDGET"}:
            raise ConfigurationError("invalid rank decision status")
        if self.estimate_status not in {"unresolved", "approximate", "exact", "interval", "zero"}:
            raise ConfigurationError("invalid rank estimate status")
        for name in ("estimated_rank", "exact_rank", "rank_lower", "rank_upper"):
            value = getattr(self, name)
            if value is not None:
                _integer(value, name)
                if value > min(self.context.shape):
                    raise ConfigurationError("rank estimate exceeds matrix dimensions")
        if (self.rank_lower is None) != (self.rank_upper is None):
            raise ConfigurationError("rank interval requires both endpoints")
        if self.rank_lower is not None and self.rank_lower > self.rank_upper:
            raise ConfigurationError("rank interval is reversed")
        if self.exact_rank is not None and (self.rank_lower != self.exact_rank or self.rank_upper != self.exact_rank):
            raise ConfigurationError("an exact rank requires a singleton bracket")
        if self.estimate_status == "approximate" and self.exact_rank is not None:
            raise ConfigurationError("approximate fit cannot claim an exact oracle rank")
        for name in ("fit_width", "canonical_basis_rank", "basis_bytes", "coefficient_bytes", "source_bytes"):
            _integer(getattr(self, name), name)
        if self.fit_width > min(self.context.shape):
            raise ConfigurationError("fit width exceeds matrix dimensions")
        if self.source_bytes != math.prod(self.context.shape) * _ITEM_BYTES[self.context.dtype]:
            raise ConfigurationError("source byte accounting differs from its original dtype")
        if type(self.guard_passed) is not bool or type(self.zero_input) is not bool:
            raise ConfigurationError("guard/zero flags must be boolean")
        if self.guard_relative_residual is not None:
            _finite(self.guard_relative_residual, "guard_relative_residual")
        if self.status == "ACCEPTED":
            _integer(self.rank, "resolved rank", 1)
            if (
                self.rank > min(self.context.shape) or self.rank > self.fit_width
                or not isinstance(self.resolved_spec, CodecSpec)
                or self.resolved_spec.kind == "raw" or self.resolved_spec.rank != self.rank
                or not self.guard_passed or self.guard_relative_residual is None
                or self.guard_relative_residual > self.guard_threshold
            ):
                raise ConfigurationError("accepted rank is not backed by a feasible guarded lossy representation")
            _digest(self.basis_sha256, "basis_sha256")
            _digest(self.coefficients_sha256, "coefficients_sha256")
            if (
                self.canonical_basis_rank < self.rank
                or self.canonical_basis_rank > self.context.channels
                or self.basis_bytes != 2 * self.canonical_basis_rank * self.context.channels
                or self.coefficient_bytes != 4 * self.context.token_count * self.rank
            ):
                raise ConfigurationError("accepted rank byte accounting is inconsistent")
        elif self.rank is not None or self.resolved_spec is not None or self.guard_passed:
            raise ConfigurationError("unresolved rank must not expose a publishable spec")
        object.__setattr__(self, "prediction_sources", _freeze(self.prediction_sources))
        object.__setattr__(self, "attempts", _freeze(self.attempts))
        object.__setattr__(self, "metrics", _freeze(self.metrics))
        _validate_metric_scalars(
            self.metrics,
            allow_unknown_times=self.status != "ACCEPTED" and self.metrics.get("operation_counts_complete") is False,
        )
        if self.status == "ACCEPTED" and (
            not set(_COUNTERS) <= set(self.metrics)
            or self.metrics.get("operation_counts_complete") is not True
        ):
            raise ConfigurationError("accepted decisions require complete explicit operation counters")
        if self.status == "ACCEPTED" and (
            self.metrics.get("zero_input") is not self.zero_input
            or self.metrics.get("projection_relative_residual") != self.guard_relative_residual
        ):
            raise ConfigurationError("accepted factor metrics disagree with the decision's actual guard")
        if self.metrics.get("source_provenance_sha256") != self.source_provenance_sha256:
            raise ConfigurationError("rank decision source provenance differs from its factor binding")
        observation = self.metrics.get("oracle_observation")
        if observation is not None:
            try:
                oracle_provenance = observation["source"]["provenance"]
                if (
                    any(oracle_provenance.get(name) != value for name, value in self.source_provenance.items())
                    or oracle_provenance.get("original_bytes_sha256") != self.input_sha256
                    or oracle_provenance.get("context") != _freeze(self.context.to_dict())
                ):
                    raise ConfigurationError("oracle observation was not bound to this decision's original source")
            except (KeyError, TypeError) as error:
                raise ConfigurationError("oracle source provenance is incomplete") from error
        for name in _ALL_COUNTERS:
            if name in self.metrics:
                if self.metrics[name] is None:
                    if self.status == "ACCEPTED" or self.metrics.get("operation_counts_complete") is not False:
                        raise ConfigurationError("unknown operation counts require an explicit failed/incomplete decision")
                else:
                    _integer(self.metrics[name], name)
        decomposition_fields = ("decomposition_count", "svd_count", "eigh_count")
        if (
            all(name in self.metrics for name in decomposition_fields)
            and all(self.metrics.get(name, 0) is not None for name in (*decomposition_fields, "eigvalsh_count"))
        ):
            if self.metrics["decomposition_count"] != (
                self.metrics["svd_count"] + self.metrics["eigh_count"] + self.metrics.get("eigvalsh_count", 0)
            ):
                raise ConfigurationError("rank decision decomposition counters disagree")

    @property
    def key(self) -> MatrixKey:
        return self.context.key

    @property
    def policy_identity(self) -> str:
        return self.policy_hash

    @property
    def selected_rank(self) -> int | None:
        return self.rank

    @property
    def source_sha256(self) -> str:
        return self.input_sha256

    @property
    def spec(self) -> CodecSpec | None:
        return self.resolved_spec

    @property
    def oracle_observation(self) -> Mapping[str, Any] | None:
        return self.metrics.get("oracle_observation")

    @property
    def resolved_rank(self) -> int | None:
        return self.rank

    @property
    def context_sha256(self) -> str:
        return _fingerprint(self.context.to_dict())

    @property
    def source_provenance_sha256(self) -> str:
        return _fingerprint(self.source_provenance)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        result = {item.name: getattr(self, item.name) for item in dataclasses.fields(self)}
        result["context"] = self.context.to_dict()
        result["resolved_spec"] = None if self.resolved_spec is None else self.resolved_spec.to_dict()
        return json_copy(result)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RankDecision:
        values = _schema_fields(cls, payload, DECISION_SCHEMA)
        values["context"] = TensorContext.from_dict(values["context"])
        if values["resolved_spec"] is not None:
            values["resolved_spec"] = CodecSpec.from_dict(values["resolved_spec"])
        return cls(**values)


@dataclass(frozen=True)
class PreparedRankMatrix:
    spec: CodecSpec
    factors: Factors
    decision: RankDecision
    schema_version: str = PREPARED_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PREPARED_SCHEMA:
            raise ConfigurationError("unsupported prepared rank schema")
        if (
            not isinstance(self.factors, Factors) or not isinstance(self.decision, RankDecision)
            or self.decision.status != "ACCEPTED" or self.spec != self.decision.resolved_spec
            or self.factors.rank != self.decision.rank
        ):
            raise ConfigurationError("prepared factors/spec/decision disagree")
        object.__setattr__(self, "factors", Factors(
            self.factors.basis, self.factors.coefficients, self.factors.rank,
            _freeze(self.factors.metrics),
        ))

    def validate(
        self, tensor: torch.Tensor, context: TensorContext,
        spec: CodecSpec | None = None, policy: RankPolicy | None = None,
    ) -> None:
        """Seal check before packing; hashing performs no candidate decomposition."""
        value = validate_matrix(tensor, context)
        decision, factors = self.decision, self.factors
        tile_rows = (
            policy.tile_rows if policy is not None else
            _integer(decision.metrics.get("tile_rows", 1024), "prepared hash tile_rows", 1)
        )
        if (
            decision.context != context or self.spec != decision.resolved_spec
            or (spec is not None and self.spec != spec) or factors.rank != decision.rank
        ):
            raise IntegrityError("prepared source context or resolved spec mismatch")
        if (
            not isinstance(factors.basis, torch.Tensor)
            or factors.basis.dtype != torch.float16
            or tuple(factors.basis.shape) != (decision.canonical_basis_rank, context.channels)
            or not isinstance(factors.coefficients, torch.Tensor)
            or factors.coefficients.dtype != torch.float32
            or tuple(factors.coefficients.shape) != (context.token_count, decision.rank)
        ):
            raise IntegrityError("prepared factors have the wrong stored basis/P domain")
        if (
            tensor_sha256(value, tile_rows=tile_rows) != decision.input_sha256
            or tensor_sha256(factors.basis, tile_rows=tile_rows) != decision.basis_sha256
            or tensor_sha256(factors.coefficients, tile_rows=tile_rows) != decision.coefficients_sha256
        ):
            raise IntegrityError("prepared source or factors changed after the residual guard")
        if policy is not None:
            policy.validate_decision(decision, context, self.spec)


class RankAbstention(StateError):
    """A failed attempt with attributable work, never an implicit RAW fallback."""

    def __init__(self, decision: RankDecision):
        self.decision = decision
        super().__init__(f"{decision.status}: {decision.reason}")


class RankPreparation:
    """One bounded clean-chunk attempt; no mutation of the prior scalar state.

    Staged preparation consumes only one source mapping value at a time. The
    caller nevertheless logically retains the *whole* clean source set until
    publication; disk-backed mappings do not make that storage disappear.
    Non-staged modes make no claim about the caller's physical source lifetime.
    Failed projections retain completed fits and attempted guard work in an
    abstention decision. Uninstrumented work is explicitly incomplete/null.
    """

    def __init__(
        self, policy: RankPolicy, prior_state: RankState | None = None, *,
        chunk_id: str, parent_head: str | None, token_start: int,
        source_provenance: Mapping[MatrixKey, Mapping[str, Any]] | None = None,
    ):
        if not isinstance(policy, RankPolicy):
            raise ConfigurationError("preparation requires a RankPolicy")
        _text(chunk_id, "chunk_id")
        _integer(token_start, "token_start")
        if parent_head is not None:
            _text(parent_head, "parent_head")
        prior = RankState(policy.fingerprint) if prior_state is None else prior_state
        if not isinstance(prior, RankState) or prior.policy_hash != policy.fingerprint:
            raise ConfigurationError("prior scalar state belongs to another policy")
        if prior.completed_chunks:
            if (
                parent_head is None or token_start != prior.token_end
                or chunk_id == prior.last_chunk_id
                or set(prior.selected_ranks) != set(policy.keys)
                or set(prior.anchor_ranks) != set(policy.anchors)
            ):
                raise ConfigurationError("prior rank state does not match the requested history boundary")
            for key in policy.keys:
                policy.spec_for(key, prior.selected_ranks[key])
            if any(
                prior.anchor_ranks[key] > min(policy.tokens, policy.model.channels_for(key))
                for key in policy.anchors
            ):
                raise ConfigurationError("saved anchor estimate exceeds its bound source dimensions")
        elif parent_head is not None:
            raise ConfigurationError("adaptive history cannot start on an unbound existing head")
        ordinal = prior.completed_chunks + 1
        if not policy.history_min <= ordinal <= policy.history_max:
            raise StateError("clean chunk is outside the policy's bound history support")
        self.policy = policy
        self.prior_state = prior
        self.chunk_id = chunk_id
        self.parent_head = parent_head
        self.token_start = token_start
        if source_provenance is not None and (
            not isinstance(source_provenance, Mapping) or set(source_provenance) != set(policy.keys)
        ):
            raise ConfigurationError("source provenance must cover the complete clean chunk")
        self._source_provenance = MappingProxyType({
            key: _freeze({} if source_provenance is None else source_provenance[key])
            for key in policy.keys
        })
        if any(not isinstance(value, Mapping) for value in self._source_provenance.values()):
            raise ConfigurationError("every source provenance entry must be a finite JSON object")
        for key, provenance in self._source_provenance.items():
            if (
                "source_fingerprint" in provenance
                and provenance["source_fingerprint"] != policy.source_fingerprint
            ):
                raise ConfigurationError("source provenance fingerprint differs from the policy source closure")
            context = TensorContext(
                policy.model, key, token_start, *policy.geometry,
                "pre_rope" if key.target == "K" else "none",
                source_dtype=policy.source_dtypes[key],
            )
            semantic_provenance = dict(provenance)
            if "context_sha256" in semantic_provenance:
                _digest(semantic_provenance.pop("context_sha256"), "source context_sha256")
            for name in ("context", "tensor_context"):
                if name in semantic_provenance:
                    declared = TensorContext.from_dict(semantic_provenance[name])
                    policy.validate_context(declared)
                    if declared.key != key or declared.token_start != token_start:
                        raise ConfigurationError("declared source provenance belongs to another clean matrix")
                    semantic_provenance.pop(name)
            _source_provenance(
                semantic_provenance, context, chunk_id=chunk_id,
                parent_head=parent_head, policy_hash=policy.fingerprint,
            )
        bound_study = [value for value in self._source_provenance.values() if "case_id" in value]
        if bound_study:
            if len(bound_study) != len(policy.keys):
                raise ConfigurationError("study provenance must bind every matrix in the atomic clean chunk")
            identities = {
                _fingerprint({name: value.get(name) for name in _SOURCE_CHUNK_FIELDS})
                for value in bound_study
            }
            if len(identities) != 1:
                raise ConfigurationError("matrix source provenance mixes unrelated clean-chunk identities")
        self._ordinal = ordinal
        self._accepted: dict[MatrixKey, RankDecision] = {}
        self._work_decisions: dict[MatrixKey, RankDecision] = {}
        self._cache: dict[MatrixKey, PreparedRankMatrix] = {}
        self._current_anchors: dict[MatrixKey, float] = {}
        self._cached_bytes = 0
        self._cache_peak = 0
        self._staged = False
        self._failed = False
        self._closed = False
        self._candidate_state: RankState | None = None
        self.failure_decision: RankDecision | None = None
        self._started = time.perf_counter()

    def __enter__(self) -> RankPreparation:
        self._ensure_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed or self._failed:
            raise StateError("rank preparation is closed or failed")

    def _check_context(self, context: TensorContext) -> None:
        self.policy.validate_context(context)
        if context.token_start != self.token_start:
            raise ConfigurationError("matrix token range differs from the prepared clean chunk")

    @property
    def decisions(self) -> Mapping[MatrixKey, RankDecision]:
        return MappingProxyType(dict(self._accepted))

    @property
    def metrics(self) -> Mapping[str, Any]:
        result = {name: 0 for name in _ALL_COUNTERS}
        for decision in self._work_decisions.values():
            for name in _ALL_COUNTERS:
                value = decision.metrics.get(name, 0)
                result[name] = None if result[name] is None or value is None else result[name] + int(value)
        result.update(
            policy_hash=self.policy.fingerprint,
            logical_source_bytes_per_chunk=self.policy.source_bytes_per_chunk,
            logical_staged_source_bytes=self.policy.source_bytes_per_chunk if self._staged else None,
            staging_charge_scope="once_per_clean_chunk_not_per_matrix",
            source_lifetime_claim="whole_clean_staged" if self._staged else "caller_lifetime_not_inferred",
            streaming_memory_claim=False,
            cached_factor_bytes=self._cached_bytes,
            peak_cached_factor_bytes=self._cache_peak,
            active_anchor_keys=[key.to_dict() for key in self.policy.active_anchors(self._ordinal)],
            scalar_state_anchor_keys=[key.to_dict() for key in self.policy.anchors],
            workspace_admission_peak_bytes=max(
                (int(item.metrics.get("workspace_admission_peak_bytes", 0))
                 for item in self._work_decisions.values()), default=0,
            ),
            workspace_is_measured_peak=False,
            prepared_matrix_count=len(self._accepted),
            operation_counts_complete=all(
                item.metrics.get("operation_counts_complete", True)
                for item in self._work_decisions.values()
            ),
            failed=self._failed,
            preparation_elapsed_seconds=time.perf_counter() - self._started,
        )
        return _freeze(result)

    def _fail(self, error: Exception) -> None:
        self._failed = True
        if isinstance(error, RankAbstention):
            self.failure_decision = error.decision
            self._work_decisions[error.decision.key] = error.decision
        self._cache.clear()
        self._cached_bytes = 0

    def prepare_staged(
        self, sources: Mapping[MatrixKey, torch.Tensor],
        contexts: Mapping[MatrixKey, TensorContext],
        reuse_bases: Mapping[MatrixKey, torch.Tensor] | None = None,
    ) -> None:
        """Declare whole-clean retention and fit only chosen staged anchors.

        Calling this for a causal/lagged control charges the shared staging but
        deliberately reads/fits no future source. Thus it cannot manufacture a
        streaming-memory comparison from an identical whole-source harness.
        """
        self._ensure_open()
        try:
            if self._accepted or self._staged:
                raise StateError("staging must precede all matrix preparation and occur once")
            if reuse_bases is not None and len(reuse_bases):
                raise UnsupportedError("rank policy v1 does not accept reused bases")
            if (
                not isinstance(sources, Mapping) or not isinstance(contexts, Mapping)
                or set(sources) != set(self.policy.keys) or set(contexts) != set(self.policy.keys)
            ):
                raise ConfigurationError("staging must describe every original clean K/V matrix")
            for key in self.policy.keys:
                context = contexts[key]
                self._check_context(context)
                if context.key != key:
                    raise ConfigurationError("staged mapping key and context key differ")
            if self.policy.source_bytes_per_chunk > self.policy.staging_limit_bytes:
                raise CapacityError("whole-clean source retention exceeds its declared staging allowance")
            self._staged = True
            if self.policy.anchor_mode != "staged":
                return
            for key in self.policy.active_anchors(self._ordinal):
                value = sources[key]
                prepared = self._fit_one(value, contexts[key])
                del value
                self._cache[key] = prepared
                self._work_decisions[key] = prepared.decision
                self._current_anchors[key] = float(prepared.decision.estimated_rank)
                self._cached_bytes += prepared.decision.basis_bytes + prepared.decision.coefficient_bytes
                self._cache_peak = max(self._cache_peak, self._cached_bytes)
        except Exception as error:
            self._fail(error)
            raise

    def prepare_matrix(
        self, tensor: torch.Tensor, context: TensorContext,
        reuse_basis: torch.Tensor | None = None,
    ) -> PreparedRankMatrix:
        self._ensure_open()
        try:
            self._check_context(context)
            if reuse_basis is not None:
                raise UnsupportedError("rank policy v1 binds approximate basis reuse OFF")
            key = context.key
            if key in self._accepted:
                raise StateError("matrix was already accepted in this clean chunk")
            if self.policy.anchor_mode == "staged" and not self._staged:
                raise StateError("staged policy requires its whole-clean capture barrier")
            if self.policy.anchor_mode == "causal" and key != self.policy.keys[len(self._accepted)]:
                raise StateError("causal preparation must follow ascending layer then K/V availability")
            prepared = self._cache.pop(key, None)
            if prepared is None:
                prepared = self._fit_one(tensor, context)
            else:
                mark = time.perf_counter()
                prepared.validate(tensor, context, policy=self.policy)
                metrics = dict(prepared.decision.metrics)
                metrics["cached_anchor_consumed"] = True
                metrics["source_hash_count"] += 1
                metrics["factor_hash_count"] += 2
                metrics["cache_seal_validation_seconds"] = time.perf_counter() - mark
                decision = dataclasses.replace(prepared.decision, metrics=metrics)
                prepared = PreparedRankMatrix(
                    prepared.spec,
                    Factors(prepared.factors.basis, prepared.factors.coefficients,
                            prepared.factors.rank, metrics),
                    decision,
                )
                self._cached_bytes -= decision.basis_bytes + decision.coefficient_bytes
            self.policy.validate_decision(prepared.decision, context, prepared.spec)
            self._accepted[key] = prepared.decision
            self._work_decisions[key] = prepared.decision
            if key in self.policy.anchors:
                self._current_anchors[key] = float(prepared.decision.estimated_rank)
            return prepared
        except Exception as error:
            self._fail(error)
            raise

    def next_state(self) -> RankState:
        self._ensure_open()
        if set(self._accepted) != set(self.policy.keys):
            raise StateError("next rank state requires every matrix's accepted guard")
        if self._candidate_state is None:
            self._candidate_state = RankState(
                policy_hash=self.policy.fingerprint,
                last_chunk_id=self.chunk_id, parent_head=self.parent_head,
                token_end=self.token_start + self.policy.tokens,
                completed_chunks=self._ordinal,
                anchor_ranks=self._current_anchors,
                anchor_chunks={key: self._ordinal for key in self._current_anchors},
                selected_ranks={key: self._accepted[key].rank for key in self.policy.keys},
            )
        return self._candidate_state

    def close(self) -> None:
        self._cache.clear()
        self._cached_bytes = 0
        self._closed = True

    def _prediction(self, key: MatrixKey) -> RankPrediction:
        return self.policy.predict_rank(
            key, current_anchors=self._current_anchors, prior_state=self.prior_state,
        )

    def _fit_one(self, tensor: torch.Tensor, context: TensorContext) -> PreparedRankMatrix:
        policy = self.policy
        started = time.perf_counter()
        self._check_context(context)
        validation_workspace = max(
            5 * math.prod(context.shape),
            4 * min(policy.tile_rows, context.token_count) * context.channels * _ITEM_BYTES[context.dtype],
        )
        if validation_workspace + self._cached_bytes > policy.workspace_limit_bytes:
            raise CapacityError("source validation/hashing exceeds the declared workspace allowance")
        validation_started = time.perf_counter()
        value = validate_matrix(tensor, context)
        validation_seconds = time.perf_counter() - validation_started
        source_hash_started = time.perf_counter()
        source_hash = tensor_sha256(value, tile_rows=policy.tile_rows)
        source_hash_seconds = time.perf_counter() - source_hash_started
        provenance = _source_provenance(
            self._source_provenance[context.key], context, chunk_id=self.chunk_id,
            parent_head=self.parent_head, policy_hash=policy.fingerprint, input_sha256=source_hash,
        )
        prediction = self._prediction(context.key)
        predicted, prediction_sources = prediction.predicted_rank, prediction.prediction_sources
        ladder = policy.allowed_ranks[context.key]
        initial = prediction.initial_rank
        full_width = min(context.shape)
        max_width = full_width if policy.max_fit_width is None else min(full_width, policy.max_fit_width)
        width = min(max_width, max(initial + policy.oversample, math.ceil(initial * policy.fit_growth)))
        fixed_anchor = policy.solver == "qr_q2" and context.key in policy.anchors
        if fixed_anchor:
            width = min(full_width, policy.detector_spec.initial_width)
        energy_target = policy.energy_for(context.key)
        threshold = math.sqrt(1 - energy_target)
        guard_band = 8 * torch.finfo(torch.float32).eps
        attempts: list[dict[str, Any]] = []
        work: dict[str, Any] = {name: 0 for name in _ALL_COUNTERS}
        work.update({name: 0.0 for name in _TIMINGS})
        work["source_hash_count"] = 1
        work.update(
            source_hash_seconds=source_hash_seconds,
            source_provenance_sha256=_fingerprint(provenance),
            study_source_provenance_bound=bool(provenance and "case_id" in provenance),
            source_validation_seconds=validation_seconds,
            factor_hash_seconds=0.0, fit_attempts=0, guard_attempts=0,
            operation_counts_complete=True, failed_oracle_calls=0,
            failed_projection_calls=0, failed_projection_seconds=0.0,
            finite_width_misses=0, view_expansions=0, refits=0,
            reuse_attempts=0, reuse_accepted=False, reuse_source_basis_id=None,
            reuse_policy_enabled=False, canonical_basis_rank=0, view_rank=None,
            source_dtype=context.dtype, coefficient_dtype="float32", basis_dtype="float16",
            rank_statistic=policy.anchor_statistic,
            tile_rows=policy.tile_rows,
            factor_bytes_scope="materialized_FP16_B_and_FP32_P_before_codec",
            serialized_payload_bytes=None, serialized_metadata_bytes=None,
            projection_convention="P=X.float()@B.float().T",
            guard_domain="original_source_vs_fp32_P_at_stored_fp16_B",
            guard_is_pre_codec=True, guard_boundary_band=guard_band,
            logical_source_bytes_per_chunk=policy.source_bytes_per_chunk,
            logical_staged_source_bytes=policy.source_bytes_per_chunk if self._staged else None,
            staging_charge_scope="once_per_clean_chunk_not_per_matrix",
            streaming_memory_claim=False,
            timing_clock="host_perf_counter",
            subcomponent_timings_synchronized=value.device.type == "cpu",
            workspace_admission_peak_bytes=validation_workspace + self._cached_bytes,
            workspace_is_measured_peak=False,
            solver_workspace_reserve_bytes=policy.solver_workspace_bytes,
            cached_factor_bytes_before=self._cached_bytes,
            cached_anchor_consumed=False, chunk_ordinal=self._ordinal,
            prior_state_fingerprint=self.prior_state.fingerprint,
            prediction_kind=prediction.kind,
            predictor_profile_fingerprint=prediction.profile_fingerprint,
            prediction_clipped_to_ladder=predicted > ladder[-1],
            fixed_anchor_acquisition=fixed_anchor,
            detector_fingerprint=policy.detector_spec.fingerprint if fixed_anchor else None,
            detector_seconds=0.0,
            basis_device_copy_bytes=0,
        )
        if fixed_anchor:
            work["anchor_feature_rank"] = None
        estimated = exact = lower = upper = None
        estimate_status = "unresolved"
        zero = False
        last_residual = None
        available_workspace = policy.workspace_limit_bytes - self._cached_bytes

        def record_work(metrics: Mapping[str, Any], extra_live: int = 0) -> None:
            for name in _ALL_COUNTERS:
                amount = metrics.get(name, 0)
                work[name] = None if work[name] is None or amount is None else work[name] + int(amount)
            work["operation_counts_complete"] &= metrics.get("operation_counts_complete", True)
            if "basis_device_copy_bytes" in metrics:
                amount = metrics["basis_device_copy_bytes"]
                work["basis_device_copy_bytes"] = (
                    None if amount is None or work["basis_device_copy_bytes"] is None
                    else work["basis_device_copy_bytes"] + amount
                )
            for name in _TIMINGS:
                amount = metrics.get(name)
                if name in metrics and (work[name] is None or amount is None):
                    work[name] = None
                elif amount is not None:
                    work[name] += float(amount)
            work["workspace_admission_peak_bytes"] = max(
                work["workspace_admission_peak_bytes"],
                int(metrics.get("workspace_admission_bytes", 0)) + self._cached_bytes + extra_live,
            )

        def decision(status: str, reason: str, factors: Factors | None = None) -> RankDecision:
            metrics = dict(work)
            metrics["decision_seconds"] = time.perf_counter() - started
            if factors is not None:
                metrics.update(
                    canonical_basis_rank=int(factors.basis.shape[0]), view_rank=factors.rank,
                    basis_bytes=factors.basis.numel() * factors.basis.element_size(),
                    coefficient_bytes=factors.coefficients.numel() * factors.coefficients.element_size(),
                )
                mark = time.perf_counter()
                basis_hash = tensor_sha256(factors.basis, tile_rows=policy.tile_rows)
                coefficients_hash = tensor_sha256(factors.coefficients, tile_rows=policy.tile_rows)
                metrics["factor_hash_seconds"] += time.perf_counter() - mark
                metrics["factor_hash_count"] += 2
                metrics["decision_seconds"] = time.perf_counter() - started
            else:
                basis_hash = coefficients_hash = None
            return RankDecision(
                policy_hash=policy.fingerprint, chunk_id=self.chunk_id, parent_head=self.parent_head,
                context=context, input_sha256=source_hash, energy_target=energy_target,
                solver=policy.solver, predicted_rank=predicted, initial_rank=initial,
                status=status, reason=reason,
                source_provenance=provenance,
                rank=None if factors is None else factors.rank,
                resolved_spec=None if factors is None else policy.spec_for(context.key, factors.rank),
                estimated_rank=work.get("anchor_feature_rank", estimated),
                exact_rank=exact, rank_lower=lower, rank_upper=upper,
                estimate_status=estimate_status, fit_width=width,
                guard_threshold=threshold, guard_relative_residual=last_residual,
                guard_passed=factors is not None, zero_input=zero,
                basis_sha256=basis_hash, coefficients_sha256=coefficients_hash,
                canonical_basis_rank=0 if factors is None else int(factors.basis.shape[0]),
                basis_bytes=0 if factors is None else factors.basis.numel() * factors.basis.element_size(),
                coefficient_bytes=0 if factors is None else factors.coefficients.numel() * factors.coefficients.element_size(),
                source_bytes=value.numel() * value.element_size(),
                prediction_sources=prediction_sources, attempts=tuple(attempts), metrics=metrics,
            )

        def abstain(reason: str, *, capacity: bool = False) -> None:
            raise RankAbstention(decision("OVER_BUDGET" if capacity else "UNRESOLVED", reason))

        local_seed = policy.factorization_seed_for(context.key, context.token_start)
        work["factorization_seed"] = local_seed
        directions = None
        reason = "finite_width_energy_unresolved"
        fit_limit = 1 if policy.solver == "exact_oracle" else policy.max_fit_attempts
        for fit_index in range(fit_limit):
            work["fit_attempts"] += 1
            if fit_index:
                work["refits"] += 1
            if available_workspace < 1:
                abstain("cached_anchor_factors_exhaust_workspace", capacity=True)
            try:
                if policy.solver == "exact_oracle":
                    from .spectra import SpectrumNumericalError, SpectrumSource, exact_spectrum

                    width = full_width
                    if policy.max_fit_width is not None and width > policy.max_fit_width:
                        abstain("exact_oracle_requires_full_spectrum_width")
                    mark = time.perf_counter()
                    try:
                        result = exact_spectrum(
                            value,
                            source=SpectrumSource(
                                identity=provenance.get("source_identity", (
                                    f"{policy.fingerprint}:{self.chunk_id}:{_label(context.key)}:"
                                    f"{work['source_provenance_sha256']}"
                                )),
                                shape=context.shape, dtype=context.dtype,
                                target=context.key.target, key_space=context.key_space,
                                retention="spectrum-only",
                                provenance={
                                    **json_copy(provenance),
                                    "policy_hash": policy.fingerprint, "context": context.to_dict(),
                                    "original_bytes_sha256": source_hash, "chunk_id": self.chunk_id,
                                    "parent_head": self.parent_head,
                                    "source_fingerprint": policy.source_fingerprint,
                                },
                            ),
                            tile_rows=policy.tile_rows, return_directions=True,
                            max_workspace_bytes=available_workspace,
                        )
                    except SpectrumNumericalError as error:
                        work["oracle_seconds"] = time.perf_counter() - mark
                        work["failed_oracle_calls"] += 1
                        failed_metrics = getattr(error, "metrics", None)
                        if isinstance(failed_metrics, Mapping):
                            work["oracle_failure_metrics"] = json_copy(failed_metrics)
                            work["operation_counts_complete"] = bool(
                                failed_metrics.get("operation_counts_complete", True)
                            )
                            record_work({
                                **failed_metrics,
                                "decomposition_seconds": failed_metrics.get("total_seconds", 0.0),
                                "fit_seconds": failed_metrics.get("total_seconds", 0.0),
                            })
                        else:
                            work["operation_counts_complete"] = False
                            work["decomposition_count"] = None
                            work["eigh_count"] = None
                            work["oracle_failure_metrics"] = None
                        attempts.append({
                            "kind": "exact_oracle", "status": "numerical_failure",
                            "reason": str(error),
                            "operation_counts_complete": work["operation_counts_complete"],
                        })
                        abstain("exact_oracle_numerical_failure")
                    oracle_seconds = time.perf_counter() - mark
                    bracket = result.rank_at(energy_target)
                    estimated, exact = bracket.upper, bracket.rank
                    lower, upper = bracket.lower, bracket.upper
                    estimate_status = bracket.status
                    zero = result.zero_energy
                    directions = result.directions
                    if directions is None:
                        raise IntegrityError("exact oracle did not return its own directions")
                    timing = result.timings
                    observation = result.to_dict()
                    work.update(
                        oracle_seconds=oracle_seconds, oracle_rank_bracket=bracket.to_dict(),
                        oracle_source_sha256=result.source.tensor_sha256,
                        oracle_solver=result.solver.to_dict(),
                        oracle_energy_scaled=result.energy_scaled,
                        oracle_energy_scale_exponent=result.scale_exponent,
                        exact_integer_rank=exact, oracle_device="cpu",
                        oracle_timing_subcomponents_available=timing is not None,
                        oracle_timings=None if timing is None else timing.to_dict(),
                        oracle_observation=observation,
                        oracle_observation_json_bytes=len(json.dumps(
                            observation, sort_keys=True, separators=(",", ":"), allow_nan=False,
                        ).encode()),
                        oracle_observation_scope="full_spectrum_no_direction_tensors",
                        factorization_method="exact_full_channel_gram_eigh",
                        factorization_seed=None, sketch_rank=None, power_iterations=0,
                    )
                    record_work({
                        **result.work_counts(),
                        "source_norm_count": 1,
                        "decomposition_seconds": oracle_seconds,
                        "fit_seconds": oracle_seconds,
                        "eigh_seconds": 0.0 if timing is None else timing.eigensolve_seconds,
                        "workspace_admission_bytes": result.diagnostics.workspace_estimate_bytes,
                    })
                    del result
                else:
                    if fixed_anchor and fit_index == 0:
                        acquisition = acquire_anchor_directions(
                            value, context, spec=policy.detector_spec,
                            max_workspace_bytes=available_workspace,
                            solver_workspace_bytes=policy.solver_workspace_bytes,
                        )
                        record_work(acquisition.metrics)
                        work["fit_attempts"] += acquisition.metrics["fit_attempts"] - 1
                        work["refits"] += max(0, acquisition.metrics["fit_attempts"] - 1)
                        work["finite_width_misses"] += acquisition.metrics["finite_width_misses"]
                        work["source_hash_seconds"] += acquisition.metrics["source_hash_seconds"]
                        work["detector_seconds"] = acquisition.metrics["elapsed_seconds"]
                        work["source_copy_bytes"] = acquisition.metrics["source_copy_bytes"]
                        work["anchor_acquisition"] = acquisition.to_dict()
                        work["anchor_feature_rank"] = acquisition.rank_at(energy_target)
                        work["anchor_feature_fit_width"] = acquisition.fit.width
                        attempts.extend({"kind": "anchor_acquisition", **item} for item in acquisition.attempts)
                        fit = acquisition.fit
                        width = fit.width
                        if acquisition.source_sha256 != source_hash:
                            abstain("fixed_anchor_source_changed")
                        if work["anchor_feature_rank"] is None:
                            abstain("fixed_anchor_feature_unresolved")
                        del acquisition
                    else:
                        fit = fit_rank_directions(
                            value, context, width,
                            seed=(local_seed + work["fit_attempts"] - 1) % 2**63, power_iterations=2,
                            tile_rows=policy.tile_rows, max_workspace_bytes=available_workspace,
                            solver_workspace_bytes=policy.solver_workspace_bytes,
                        )
                        record_work(fit.metrics)
                    estimated = fit.required_rank(energy_target)
                    zero = fit.energy.zero_input
                    estimate_status = "zero" if zero else "approximate" if estimated is not None else "unresolved"
                    if zero:
                        exact = lower = upper = 0
                    work.update(
                        factorization_method=fit.metrics["factorization_method"],
                        sketch_rank=width, power_iterations=0 if zero else 2,
                        source_norm=fit.energy.source_norm, source_energy=fit.energy.source_energy,
                        fitted_energy_fraction=fit.energy_fraction(width),
                        estimated_cdf_before=None if estimated is None or estimated == 0 else fit.energy_fraction(estimated - 1),
                        estimated_cdf_at=None if estimated is None else fit.energy_fraction(estimated),
                        encoding_estimated_rank=estimated,
                        encoding_estimate_status=estimate_status,
                    )
                    directions = fit.directions
                    del fit
            except DetectorAcquisitionError as error:
                record_work(error.metrics)
                work["fit_attempts"] += error.metrics.get("fit_attempts", 0) - 1
                work["refits"] += max(0, error.metrics.get("fit_attempts", 0) - 1)
                work["finite_width_misses"] += error.metrics.get("finite_width_misses", 0)
                work["detector_seconds"] = error.metrics["elapsed_seconds"]
                work["source_hash_seconds"] += error.metrics.get("source_hash_seconds", 0.0)
                work["source_copy_bytes"] = error.metrics.get("source_copy_bytes")
                work["anchor_acquisition_failure"] = json_copy(error.metrics)
                attempts.extend({"kind": "anchor_acquisition", **item}
                                for item in error.metrics.get("attempts", ()))
                abstain(str(error), capacity=error.metrics.get("status") == "OVER_BUDGET")
            except CapacityError as error:
                attempts.append({"kind": "admission", "fit_width": width, "reason": str(error)})
                abstain("fit_workspace_limit", capacity=True)
            except RankFitError as error:
                record_work(error.metrics)
                attempts.append({
                    "kind": "fit", "fit_index": fit_index + 1, "fit_width": width,
                    "seed": None if local_seed is None else (local_seed + work["fit_attempts"] - 1) % 2**63,
                    "status": "numerical_failure", "reason": str(error),
                })
                abstain("rank_decomposition_failed")
            attempts.append({
                "kind": "anchor_fit_reuse" if fixed_anchor and fit_index == 0 else "fit",
                "fit_index": fit_index + 1, "fit_width": width,
                "estimated_rank": estimated, "estimate_status": estimate_status,
                "exact_rank": exact,
                "status": "unresolved" if estimated is None else "crossing",
            })
            work["zero_input"] = zero
            work["source_bytes"] = value.numel() * value.element_size()
            if estimated is None:
                work["finite_width_misses"] += 1
                reason = "finite_width_energy_unresolved"
            else:
                candidates = [rank for rank in ladder if max(1, estimated) <= rank <= width]
                if zero:
                    candidates = [ladder[0]] if ladder[0] <= width else []
                previous = self.prior_state.selected_ranks.get(context.key)
                if (
                    candidates and policy.hysteresis_fraction > 0 and previous in candidates
                    and previous <= candidates[0] * (1 + policy.hysteresis_fraction)
                ):
                    candidates = [rank for rank in candidates if rank >= previous]
                    work["hysteresis_attempted_hold_rank"] = previous
                reason = "allowed_ladder_cannot_reach_estimated_energy"
                for candidate_index, rank in enumerate(candidates):
                    if work["guard_attempts"] >= policy.max_guard_attempts:
                        abstain("guard_attempt_limit")
                    live_directions = directions.numel() * directions.element_size()
                    required = rank_projection_workspace_bytes(
                        context.shape, rank, tile_rows=policy.tile_rows,
                        solver_workspace_bytes=policy.solver_workspace_bytes,
                    )
                    if directions.device != value.device:
                        required += 2 * rank * context.channels
                    if required + live_directions > available_workspace:
                        attempts.append({
                            "kind": "admission", "rank": rank,
                            "workspace_bytes": required + live_directions + self._cached_bytes,
                        })
                        abstain("projection_workspace_limit", capacity=True)
                    work["guard_attempts"] += 1
                    work["view_expansions"] += int(candidate_index > 0)
                    mark = time.perf_counter()
                    try:
                        factors = project_rank_directions(
                            value, directions, rank, tile_rows=policy.tile_rows,
                            max_workspace_bytes=available_workspace - live_directions,
                            solver_workspace_bytes=policy.solver_workspace_bytes,
                        )
                    except Exception as error:
                        elapsed = time.perf_counter() - mark
                        work["failed_projection_calls"] += 1
                        work["failed_projection_seconds"] += elapsed
                        work["failed_projection_time_scope"] = "outer_attempt_including_rounding_nested_in_decision"
                        work["known_work_before_failed_projection"] = {
                            name: work[name] for name in (*_ALL_COUNTERS, *_TIMINGS)
                        }
                        if isinstance(error, RankProjectionError):
                            record_work(error.metrics, live_directions)
                            work["projection_failure_metrics"] = json_copy(error.metrics)
                            retryable = (
                                error.metrics.get("failure_kind") == "configuration"
                                and error.metrics.get("operation_counts_complete") is True
                            )
                        else:
                            work["operation_counts_complete"] = False
                            for name in (
                                "projection_count", "matmul_count", "source_norm_count",
                                *_PROJECTION_COUNTERS,
                            ):
                                work[name] = None
                            work["projection_seconds"] += elapsed
                            for name in (
                                "projection_coefficient_seconds", "projection_guard_seconds",
                                "condition_seconds", "basis_round_seconds",
                                "basis_device_copy_seconds",
                            ):
                                work[name] = None
                            work["projection_failure_metrics"] = None
                            work["workspace_admission_peak_bytes"] = max(
                                work["workspace_admission_peak_bytes"],
                                required + live_directions + self._cached_bytes,
                            )
                            retryable = False
                        attempts.append({
                            "kind": "guard", "fit_index": fit_index + 1,
                            "rank": rank, "accepted": False, "reason": str(error),
                            "status": "projection_failure",
                            "failure_type": type(error.__cause__ or error).__name__,
                            "operation_counts_complete": work["operation_counts_complete"],
                            "elapsed_seconds": elapsed,
                            "work": work["projection_failure_metrics"],
                        })
                        if not retryable:
                            abstain("projection_backend_failure")
                        reason = "invalid_stored_projection"
                        continue
                    record_work(factors.metrics, live_directions)
                    last_residual = float(factors.metrics["projection_relative_residual"])
                    accepted = zero or last_residual == 0 or last_residual < threshold - guard_band
                    attempts.append({
                        "kind": "guard", "fit_index": fit_index + 1, "rank": rank,
                        "relative_residual": last_residual, "accepted": accepted,
                        "basis_bytes": factors.metrics["basis_bytes"],
                        "coefficient_bytes": factors.metrics["coefficient_bytes"],
                    })
                    if accepted:
                        if bool(factors.metrics["zero_input"]) != zero:
                            raise IntegrityError("fitted norm and original-source projection disagree on zero input")
                        work["zero_input"] = bool(factors.metrics["zero_input"])
                        if rank == work.get("hysteresis_attempted_hold_rank"):
                            work["hysteresis_held_rank"] = rank
                        work["basis_condition_bound"] = factors.metrics["basis_condition_bound"]
                        work["residual_norm"] = factors.metrics["residual_norm"]
                        work["projection_relative_residual"] = last_residual
                        work["stored_svd_relative_residual"] = last_residual
                        work["post_hevc_relative_residual"] = None
                        accepted_decision = decision(
                            "ACCEPTED", "zero_input_operational_rank" if zero else "direct_residual_guard",
                            factors,
                        )
                        return PreparedRankMatrix(
                            accepted_decision.resolved_spec,
                            Factors(factors.basis, factors.coefficients, rank, accepted_decision.metrics),
                            accepted_decision,
                        )
                    reason = (
                        "stored_precision_guard_failed_at_full_rank"
                        if rank == full_width else "direct_residual_guard_failed"
                    )
                    del factors
            del directions
            directions = None
            if width >= max_width or policy.solver == "exact_oracle" or fit_index + 1 >= fit_limit:
                break
            width = min(
                max_width,
                ladder[0] if zero and width < ladder[0]
                else max(width + 1, math.ceil(width * policy.fit_growth)),
            )
        abstain(reason)
