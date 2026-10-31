"""Build auditable profile registries from externally measured native outcomes."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from .errors import ConfigurationError, IntegrityError, ReelCacheError, StateError
from .policy import (
    FEATURE_NAMES, Profile, QualityEvidence, _finite, _positive_int,
)


def _canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def package_fingerprint(
    *, package_root: Path | None = None, extra_identity: Mapping | None = None
) -> str:
    """Bind package source; callers also supply codec/runtime dependency identities."""
    root = Path(__file__).parent if package_root is None else Path(package_root)
    sources = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }
    if not sources:
        raise ConfigurationError("implementation fingerprint requires package sources")
    return hashlib.sha256(_canonical({
        "sources": sources, "extra_identity": dict(extra_identity or {}),
    })).hexdigest()


def configuration_fingerprint(profile: Profile) -> str:
    data = profile.to_dict()
    fields = (
        "profile_id", "model", "specs", "geometry",
        "history_min", "history_max", "implementation_fingerprint",
    )
    return hashlib.sha256(_canonical({key: data[key] for key in fields})).hexdigest()


@dataclass(frozen=True)
class Observation:
    profile_id: str
    case_id: str
    prompt_id: str
    split: str
    configuration_sha256: str
    source_sha256: str
    reference_id: str
    ssim: tuple[float, ...]
    reference_ssim: tuple[float, ...]
    raw_bytes: int
    stored_bytes: int
    kv_relative_error: float
    encode_seconds: float
    history_chunks: int = 1
    features: Mapping[str, float] = field(default_factory=dict)
    native_validated: bool = False
    held_out_verified: bool = False

    def __post_init__(self):
        for name in ("profile_id", "case_id", "prompt_id", "reference_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ConfigurationError(f"{name} must be a nonempty string")
        if self.split not in {"calibration", "validation", "development"}:
            raise ConfigurationError("unknown observation split")
        for name in ("configuration_sha256", "source_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ConfigurationError(f"{name} must be SHA256 hex")
        for name in ("ssim", "reference_ssim"):
            values = tuple(getattr(self, name))
            if not values:
                raise ConfigurationError("observations require frame-level SSIM")
            for value in values:
                _finite(value, name, minimum=-1, maximum=1)
            object.__setattr__(self, name, values)
        if len(self.ssim) != len(self.reference_ssim):
            raise ConfigurationError("SSIM reference must cover the same frames")
        for name in ("raw_bytes", "stored_bytes", "history_chunks"):
            _positive_int(getattr(self, name), name)
        _finite(self.kv_relative_error, "kv_relative_error", minimum=0)
        _finite(self.encode_seconds, "encode_seconds", minimum=0)
        if not set(self.features) <= FEATURE_NAMES:
            raise ConfigurationError("unsupported observation features")
        for name, value in self.features.items():
            _finite(value, name, minimum=0)
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))
        for name in ("native_validated", "held_out_verified"):
            if type(getattr(self, name)) is not bool:
                raise ConfigurationError(f"{name} must be boolean")


def fit_profile(template: Profile, observations: Sequence[Observation]) -> Profile:
    """Fit costs/support on calibration only, then attach separate validation.

    Records must already refer to the exact frozen action vector and implementation.
    This function cannot establish that a caller's split declarations are truthful.
    """
    if not observations:
        raise ConfigurationError("profile fitting requires measured observations")
    config_hash = configuration_fingerprint(template)
    if len({row.case_id for row in observations}) != len(observations):
        raise ConfigurationError("duplicate cases must not inflate profile support")
    if any(
        row.profile_id != template.profile_id
        or row.configuration_sha256 != config_hash
        for row in observations
    ):
        raise ConfigurationError("observations do not match the frozen profile configuration")
    if any(not template.history_min <= row.history_chunks <= template.history_max
           for row in observations):
        raise ConfigurationError("observation lies outside frozen history support")
    fitting = [row for row in observations if row.split != "validation"]
    validation = [row for row in observations if row.split == "validation"]
    if not fitting:
        raise ConfigurationError("validation outcomes must not fit profile costs or support")
    names = set(fitting[0].features)
    if any(set(row.features) != names for row in observations):
        raise ConfigurationError("calibration and validation feature definitions differ")
    fitted_bounds = {
        name: (
            min(row.features[name] for row in fitting),
            max(row.features[name] for row in fitting),
        )
        for name in sorted(names)
    }
    bounds = dict(template.feature_bounds) or fitted_bounds
    if set(bounds) != names:
        raise ConfigurationError("frozen profile support does not match observation features")
    if any(
        not low <= row.features[name] <= high
        for row in observations for name, (low, high) in bounds.items()
    ):
        raise ConfigurationError(
            "out-of-support observations cannot be silently included as accepted validation"
        )
    evidence = None
    if validation:
        cal_prompts = tuple(sorted({row.prompt_id for row in fitting}))
        val_prompts = tuple(sorted({row.prompt_id for row in validation}))
        if set(cal_prompts) & set(val_prompts):
            raise ConfigurationError("validation prompts overlap calibration prompts")
        references = {row.reference_id for row in observations}
        if len(references) != 1:
            raise ConfigurationError("quality references differ between observations")
        frames = [score for row in validation for score in row.ssim]
        drops = [
            reference - score for row in validation
            for score, reference in zip(row.ssim, row.reference_ssim)
        ]
        evidence = QualityEvidence(
            reference_id=next(iter(references)),
            mean_ssim=sum(frames) / len(frames),
            minimum_frame_ssim=min(frames),
            mean_ssim_drop=sum(drops) / len(drops),
            calibration_prompts=cal_prompts,
            validation_prompts=val_prompts,
            validation_cases=tuple(row.case_id for row in validation),
            source_sha256=tuple(sorted({row.source_sha256 for row in observations})),
            native_validated=all(row.native_validated for row in observations),
            held_out=all(row.held_out_verified for row in validation),
        )
    return dataclasses.replace(
        template,
        measured_stored_bytes=max(row.stored_bytes for row in fitting),
        predicted_distortion=max(row.kv_relative_error for row in fitting),
        encode_seconds=max(row.encode_seconds for row in fitting),
        feature_bounds=bounds,
        evidence=evidence,
    )


def save_registry(path: Path, profiles: Sequence[Profile]) -> str:
    """Atomically publish an immutable, checksum-bound profile registry."""
    if any(not isinstance(profile, Profile) for profile in profiles):
        raise ConfigurationError("static registries accept Profile vectors; use save_rank_policy for dynamic policies")
    identifiers = [profile.profile_id for profile in profiles]
    if len(set(identifiers)) != len(identifiers):
        raise ConfigurationError("registry contains duplicate profiles")
    body = {
        "schema": "reelcache_profile_registry_v1",
        "profiles": [profile.to_dict() for profile in sorted(profiles, key=lambda p: p.profile_id)],
    }
    digest = hashlib.sha256(_canonical(body)).hexdigest()
    payload = _canonical({**body, "sha256": digest}) + b"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise IntegrityError("profile registry must not be a symlink")
    if path.exists():
        if path.read_bytes() != payload:
            raise StateError("refusing to overwrite a frozen profile registry")
        return digest
    descriptor, temporary = tempfile.mkstemp(prefix=".reelcache-profile-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)
    return digest


def load_registry(path: Path, *, expected_sha256: str | None = None) -> list[Profile]:
    path = Path(path)
    if path.is_symlink():
        raise IntegrityError("profile registry must not be a symlink")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise IntegrityError("profile registry is not valid JSON") from error
    if not isinstance(data, dict) or set(data) != {"schema", "profiles", "sha256"}:
        raise IntegrityError("invalid profile registry fields")
    if data["schema"] != "reelcache_profile_registry_v1" or not isinstance(data["profiles"], list):
        raise IntegrityError("unsupported profile registry schema")
    body = {"schema": data["schema"], "profiles": data["profiles"]}
    try:
        digest = hashlib.sha256(_canonical(body)).hexdigest()
    except (TypeError, ValueError) as error:
        raise IntegrityError("profile registry contains invalid values") from error
    if digest != data["sha256"] or (
        expected_sha256 is not None and digest != expected_sha256
    ):
        raise IntegrityError("profile registry checksum mismatch")
    try:
        profiles = [Profile.from_dict(value) for value in data["profiles"]]
    except (KeyError, TypeError, ValueError) as error:
        raise IntegrityError("profile registry contains malformed profiles") from error
    if len({profile.profile_id for profile in profiles}) != len(profiles):
        raise IntegrityError("profile registry contains duplicate identifiers")
    return profiles


def save_rank_policy(path: Path, policy) -> str:
    """Freeze one experimental policy, never a quality-qualified Profile.

    Allowed per-matrix ranks describe admissible executions, not a calibrated
    Cartesian product. The checksum covers the entire versioned policy.
    """
    from .online_rank import RankPolicy

    if not isinstance(policy, RankPolicy):
        raise ConfigurationError("rank policy registry requires an experimental RankPolicy")
    policy_data = policy.to_dict()
    if RankPolicy.from_dict(policy_data).to_dict() != policy_data:
        raise ConfigurationError("rank policy is not a canonical portable policy")
    body = {"schema": "reelcache_rank_policy_registry_v1", "policy": policy_data}
    digest = hashlib.sha256(_canonical(body)).hexdigest()
    payload = _canonical({**body, "sha256": digest}) + b"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise IntegrityError("rank policy registry must not be a symlink")
    if path.exists():
        if path.read_bytes() != payload:
            raise StateError("refusing to overwrite a frozen rank policy")
        return digest
    pending = path.with_name(f".{path.name}.pending-{uuid.uuid4().hex}")
    descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(pending, path)
        except FileExistsError as exc:
            raise StateError("rank policy registry was published concurrently") from exc
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        pending.unlink()
    return digest


def load_rank_policy(path: Path, *, expected_sha256: str | None = None):
    """Read a strictly versioned experimental policy without qualifying it."""
    from .online_rank import RankPolicy
    from .serialization import parse_json

    path = Path(path)
    if path.is_symlink():
        raise IntegrityError("rank policy registry must not be a symlink")
    data = parse_json(path.read_bytes())
    if (
        not isinstance(data, dict) or set(data) != {"schema", "policy", "sha256"}
        or data["schema"] != "reelcache_rank_policy_registry_v1"
    ):
        raise IntegrityError("invalid experimental rank policy registry")
    digest = hashlib.sha256(_canonical({
        "schema": data["schema"], "policy": data["policy"],
    })).hexdigest()
    if digest != data["sha256"] or (
        expected_sha256 is not None and expected_sha256 != digest
    ):
        raise IntegrityError("rank policy registry checksum mismatch")
    try:
        policy = RankPolicy.from_dict(data["policy"])
        if policy.to_dict() != data["policy"]:
            raise IntegrityError("rank policy registry has partial or noncanonical fields")
    except (AttributeError, ReelCacheError, TypeError, ValueError, KeyError) as exc:
        raise IntegrityError("rank policy registry contains a malformed policy") from exc
    return policy
