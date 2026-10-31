"""Deterministic selection of measured whole-model representation profiles.

Selection never fits a basis or evaluates future video quality. Evidence fields
are operator-supplied measurements, not a statistical or per-request guarantee.
Unqualified lossy configurations require an explicitly named experimental profile.
Dynamic ``online_rank.RankPolicy`` objects are a separate versioned experiment
contract; allowed rank sets are not quality-qualified static action vectors and
are never passed through this production profile selector.
"""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from .errors import CapacityError, ConfigurationError, StateError
from .types import CodecSpec, MatrixKey, ModelIdentity

FEATURE_NAMES = frozenset({
    "K_residual_max", "V_residual_max",
    "K_range_imbalance", "V_range_imbalance",
    "K_outlier_fraction", "V_outlier_fraction",
    "K_spatial_variation", "V_spatial_variation",
    "basis_age",
})


def _finite(value, name: str, *, minimum=None, maximum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ConfigurationError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{name} is below its valid range")
    if maximum is not None and value > maximum:
        raise ConfigurationError(f"{name} is above its valid range")
    return value


def _positive_int(value, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")


def _strings(values, name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ConfigurationError(f"{name} must be a sequence, not a string")
    values = tuple(values)
    if any(not isinstance(value, str) or not value for value in values):
        raise ConfigurationError(f"{name} must contain nonempty strings")
    if len(values) != len(set(values)):
        raise ConfigurationError(f"{name} must not contain duplicates")
    return values


def _geometry(value) -> tuple[int, int, int]:
    value = tuple(value)
    if len(value) != 3:
        raise ConfigurationError("geometry must contain frames, height and width")
    for dimension in value:
        _positive_int(dimension, "geometry dimension")
    return value


def _specs(specs: Mapping[MatrixKey, CodecSpec], model) -> Mapping:
    expected = {
        MatrixKey(layer, target)
        for layer in range(model.layers) for target in ("K", "V")
    }
    if set(specs) != expected or any(
        not isinstance(spec, CodecSpec) for spec in specs.values()
    ):
        raise ConfigurationError("a profile must specify every model layer's K and V")
    if any(spec.rank is not None and spec.rank > model.channels_for(key)
           for key, spec in specs.items()):
        raise ConfigurationError("profile rank exceeds canonical KV channels")
    return MappingProxyType(dict(specs))


def _spec_dict(specs) -> dict:
    return {
        f"{key.layer}:{key.target}": value.to_dict()
        for key, value in sorted(specs.items(), key=lambda item: (item[0].layer, item[0].target))
    }


@dataclass(frozen=True)
class QualityEvidence:
    reference_id: str
    mean_ssim: float
    minimum_frame_ssim: float
    mean_ssim_drop: float
    calibration_prompts: tuple[str, ...]
    validation_prompts: tuple[str, ...]
    validation_cases: tuple[str, ...]
    source_sha256: tuple[str, ...]
    held_out: bool = False
    native_validated: bool = False

    def __post_init__(self):
        if not isinstance(self.reference_id, str) or not self.reference_id:
            raise ConfigurationError("quality evidence requires a reference identity")
        _finite(self.mean_ssim, "mean_ssim", minimum=-1, maximum=1)
        _finite(self.minimum_frame_ssim, "minimum_frame_ssim", minimum=-1, maximum=1)
        _finite(self.mean_ssim_drop, "mean_ssim_drop", minimum=-2, maximum=2)
        if self.minimum_frame_ssim > self.mean_ssim:
            raise ConfigurationError("minimum frame SSIM exceeds mean SSIM")
        if type(self.held_out) is not bool or type(self.native_validated) is not bool:
            raise ConfigurationError("evidence flags must be boolean")
        for name in (
            "calibration_prompts", "validation_prompts",
            "validation_cases", "source_sha256",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        if any(re.fullmatch(r"[0-9a-f]{64}", value) is None
               for value in self.source_sha256):
            raise ConfigurationError("evidence source identities must be SHA256 hex")
        if len(self.validation_cases) < len(self.validation_prompts):
            raise ConfigurationError("validation cases cannot cover fewer than listed prompts")
        if self.held_out and set(self.calibration_prompts) & set(self.validation_prompts):
            raise ConfigurationError("held-out prompts overlap calibration prompts")


@dataclass(frozen=True)
class QualityBudget:
    reference_id: str = "per_channel"
    min_mean_ssim: float | None = None
    min_frame_ssim: float | None = None
    max_mean_ssim_drop: float | None = None
    min_calibration_prompts: int = 6
    min_validation_prompts: int = 4
    min_validation_cases: int = 8
    max_predicted_bytes: int | None = None
    max_predicted_distortion: float | None = None
    max_encode_seconds: float | None = None

    def __post_init__(self):
        if not isinstance(self.reference_id, str) or not self.reference_id:
            raise ConfigurationError("quality budget requires a reference identity")
        thresholds = (self.min_mean_ssim, self.min_frame_ssim, self.max_mean_ssim_drop)
        if any(value is not None for value in thresholds):
            if any(value is None for value in thresholds):
                raise ConfigurationError("all three SSIM limits must be specified together")
            _finite(self.min_mean_ssim, "min_mean_ssim", minimum=-1, maximum=1)
            _finite(self.min_frame_ssim, "min_frame_ssim", minimum=-1, maximum=1)
            _finite(self.max_mean_ssim_drop, "max_mean_ssim_drop", minimum=0, maximum=2)
        for name in (
            "min_calibration_prompts", "min_validation_prompts", "min_validation_cases",
        ):
            _positive_int(getattr(self, name), name)
        if self.max_predicted_bytes is not None:
            _positive_int(self.max_predicted_bytes, "max_predicted_bytes")
        for name in ("max_predicted_distortion", "max_encode_seconds"):
            if getattr(self, name) is not None:
                _finite(getattr(self, name), name, minimum=0)


@dataclass(frozen=True)
class Profile:
    profile_id: str
    model: ModelIdentity
    specs: Mapping[MatrixKey, CodecSpec]
    geometry: tuple[int, int, int]
    measured_stored_bytes: int
    predicted_distortion: float
    encode_seconds: float
    history_min: int = 1
    history_max: int = 1
    feature_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    evidence: QualityEvidence | None = None
    implementation_fingerprint: str | None = None

    def __post_init__(self):
        if not isinstance(self.profile_id, str) or not self.profile_id or self.profile_id == "RAW":
            raise ConfigurationError("profile_id must be nonempty and not reserved RAW")
        if not isinstance(self.model, ModelIdentity):
            raise ConfigurationError("profile requires a ModelIdentity")
        object.__setattr__(self, "specs", _specs(self.specs, self.model))
        object.__setattr__(self, "geometry", _geometry(self.geometry))
        _positive_int(self.measured_stored_bytes, "measured_stored_bytes")
        _finite(self.predicted_distortion, "predicted_distortion", minimum=0)
        _finite(self.encode_seconds, "encode_seconds", minimum=0)
        _positive_int(self.history_min, "history_min")
        _positive_int(self.history_max, "history_max")
        if self.history_max < self.history_min:
            raise ConfigurationError("history support interval is reversed")
        if not set(self.feature_bounds) <= FEATURE_NAMES:
            raise ConfigurationError("unsupported online feature in profile")
        bounds = {}
        for name, interval in self.feature_bounds.items():
            if len(interval) != 2:
                raise ConfigurationError("feature support must contain two endpoints")
            low, high = (_finite(value, name, minimum=0) for value in interval)
            if low > high:
                raise ConfigurationError("feature support interval is reversed")
            bounds[name] = (low, high)
        object.__setattr__(self, "feature_bounds", MappingProxyType(bounds))
        if self.evidence is not None and not isinstance(self.evidence, QualityEvidence):
            raise ConfigurationError("evidence must be QualityEvidence or None")
        if self.implementation_fingerprint is not None and (
            not isinstance(self.implementation_fingerprint, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.implementation_fingerprint) is None
        ):
            raise ConfigurationError("implementation fingerprint must be SHA256 hex")

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "model": self.model.to_dict(),
            "specs": _spec_dict(self.specs),
            "geometry": list(self.geometry),
            "measured_stored_bytes": self.measured_stored_bytes,
            "predicted_distortion": self.predicted_distortion,
            "encode_seconds": self.encode_seconds,
            "history_min": self.history_min,
            "history_max": self.history_max,
            "feature_bounds": {key: list(value) for key, value in self.feature_bounds.items()},
            "evidence": None if self.evidence is None else dataclasses.asdict(self.evidence),
            "implementation_fingerprint": self.implementation_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> Profile:
        values = dict(data)
        values["model"] = ModelIdentity.from_dict(values["model"])
        specs = {}
        for label, spec in values["specs"].items():
            if not isinstance(label, str) or re.fullmatch(r"(0|[1-9][0-9]*):[KV]", label) is None:
                raise ConfigurationError("matrix profile key must be '<layer>:K' or '<layer>:V'")
            layer, target = label.split(":")
            specs[MatrixKey(int(layer), target)] = CodecSpec.from_dict(spec)
        values["specs"] = specs
        if values.get("evidence") is not None:
            values["evidence"] = QualityEvidence(**values["evidence"])
        return cls(**values)


@dataclass(frozen=True)
class Selection:
    profile_id: str
    specs: Mapping[MatrixKey, CodecSpec]
    predicted_bytes: int
    predicted_distortion: float
    encode_seconds: float | None
    evidence_qualified: bool
    experimental: bool
    reason: str
    rejected: Mapping[str, str]

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id, "specs": _spec_dict(self.specs),
            "predicted_bytes": self.predicted_bytes,
            "predicted_distortion": self.predicted_distortion,
            "encode_seconds": self.encode_seconds,
            "evidence_qualified": self.evidence_qualified,
            "experimental": self.experimental,
            "reason": self.reason, "rejected": dict(self.rejected),
            "per_request_SSIM_guarantee": False,
            "costs_are_predictions_not_capacity_reservations": True,
        }


def qualification_failure(profile: Profile, budget: QualityBudget) -> str | None:
    if (
        budget.min_mean_ssim is None or budget.min_frame_ssim is None
        or budget.max_mean_ssim_drop is None
    ):
        return "automatic lossy selection requires an explicit absolute/relative SSIM budget"
    evidence = profile.evidence
    if evidence is None:
        return "no native quality evidence"
    if not evidence.native_validated or not evidence.held_out:
        return "native held-out validation is missing"
    if not evidence.source_sha256:
        return "quality evidence has no source identities"
    if evidence.reference_id != budget.reference_id:
        return "quality reference does not match the budget"
    if (
        len(evidence.calibration_prompts) < budget.min_calibration_prompts
        or len(evidence.validation_prompts) < budget.min_validation_prompts
        or len(evidence.validation_cases) < budget.min_validation_cases
    ):
        return "insufficient independent calibration/validation support"
    if (
        evidence.mean_ssim < budget.min_mean_ssim
        or evidence.minimum_frame_ssim < budget.min_frame_ssim
        or evidence.mean_ssim_drop > budget.max_mean_ssim_drop
    ):
        return "absolute or relative SSIM quality budget failed"
    return None


def select_profile(
    profiles: Sequence[Profile],
    *,
    model: ModelIdentity,
    geometry: tuple[int, int, int],
    history_chunks: int,
    features: Mapping[str, float],
    raw_bytes: int,
    budget: QualityBudget | None = None,
    explicit_profile: str | None = None,
    allow_experimental: bool = False,
    implementation_fingerprint: str | None = None,
) -> Selection:
    """Select the smallest supported qualified vector, or explicitly report RAW."""
    budget = QualityBudget() if budget is None else budget
    geometry = _geometry(geometry)
    _positive_int(history_chunks, "history_chunks")
    _positive_int(raw_bytes, "raw_bytes")
    if not isinstance(model, ModelIdentity) or not isinstance(budget, QualityBudget):
        raise ConfigurationError("model and budget must use their declared types")
    if not set(features) <= FEATURE_NAMES:
        raise ConfigurationError("features must be declared current-tensor statistics")
    for name, value in features.items():
        _finite(value, name, minimum=0)
    if type(allow_experimental) is not bool:
        raise ConfigurationError("allow_experimental must be boolean")
    if implementation_fingerprint is not None and (
        not isinstance(implementation_fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", implementation_fingerprint) is None
    ):
        raise ConfigurationError("implementation fingerprint must be SHA256 hex")
    if allow_experimental and explicit_profile is None:
        raise ConfigurationError("experimental selection requires an explicit profile")
    if any(not isinstance(profile, Profile) for profile in profiles):
        raise ConfigurationError("static selection accepts Profile vectors, not dynamic RankPolicy objects")
    identifiers = [profile.profile_id for profile in profiles]
    if len(identifiers) != len(set(identifiers)):
        raise ConfigurationError("duplicate profile identifiers")
    if explicit_profile is not None and explicit_profile not in identifiers:
        raise ConfigurationError("explicit profile is not registered")
    candidates = []
    rejected = {}
    for profile in profiles:
        if explicit_profile is not None and profile.profile_id != explicit_profile:
            continue
        reason = None
        if profile.model != model or profile.geometry != geometry:
            reason = "model identity or geometry is outside profile support"
        elif not profile.history_min <= history_chunks <= profile.history_max:
            reason = "history length is outside profile support"
        elif any(
            name not in features or not low <= features[name] <= high
            for name, (low, high) in profile.feature_bounds.items()
        ):
            reason = "current tensor features are outside profile support"
        elif (
            budget.max_predicted_bytes is not None
            and profile.measured_stored_bytes > budget.max_predicted_bytes
        ):
            reason = "predicted byte budget failed"
        elif (
            budget.max_predicted_distortion is not None
            and profile.predicted_distortion > budget.max_predicted_distortion
        ):
            reason = "decoded-distortion proxy budget failed"
        elif (
            budget.max_encode_seconds is not None
            and profile.encode_seconds > budget.max_encode_seconds
        ):
            reason = "predicted encode deadline failed"
        elif explicit_profile is None and profile.measured_stored_bytes >= raw_bytes:
            reason = "profile does not reduce predicted stored bytes relative to RAW"
        quality_failure = qualification_failure(profile, budget)
        if (
            implementation_fingerprint is None
            or profile.implementation_fingerprint != implementation_fingerprint
        ):
            quality_failure = "quality evidence is not bound to the current implementation"
        if reason is None and quality_failure is not None and not allow_experimental:
            reason = quality_failure
        if reason is not None:
            rejected[profile.profile_id] = reason
        else:
            candidates.append((profile, quality_failure))
    if candidates:
        profile, quality_failure = min(
            candidates,
            key=lambda item: (
                item[0].measured_stored_bytes,
                item[0].predicted_distortion,
                item[0].profile_id,
            ),
        )
        return Selection(
            profile.profile_id, profile.specs, profile.measured_stored_bytes,
            profile.predicted_distortion, profile.encode_seconds,
            quality_failure is None, quality_failure is not None,
            "supported measured profile" if quality_failure is None
            else f"explicit unqualified experiment: {quality_failure}",
            MappingProxyType(rejected),
        )
    if explicit_profile is not None:
        raise StateError(
            f"Explicit profile {explicit_profile!r} rejected: {rejected[explicit_profile]}"
        )
    if budget.max_predicted_bytes is not None and raw_bytes > budget.max_predicted_bytes:
        raise CapacityError("No supported compressed profile; RAW exceeds the byte budget")
    if budget.max_encode_seconds is not None:
        raise CapacityError("No supported compressed profile; RAW encoding cost is unqualified")
    specs = {
        MatrixKey(layer, target): CodecSpec()
        for layer in range(model.layers) for target in ("K", "V")
    }
    return Selection(
        "RAW", MappingProxyType(specs), raw_bytes, 0.0, None, True, False,
        "explicit RAW abstention: no supported quality-qualified lossy profile",
        MappingProxyType(rejected),
    )
