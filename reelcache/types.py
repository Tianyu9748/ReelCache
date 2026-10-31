"""Validated component contracts, independent of native model imports."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from . import accel_c as _accel_c
from .errors import ConfigurationError


_FLOAT_DTYPES = {
    "float16": "float16", "half": "float16",
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "float32": "float32", "float": "float32",
    "float64": "float64", "double": "float64",
}


def _positive_int(value: Any, name: str, *, zero: bool = False) -> None:
    if type(value) is not int or value < (0 if zero else 1):
        raise ConfigurationError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a nonempty string")


def canonical_dtype(value: Any) -> str:
    name = str(value).removeprefix("torch.")
    if name not in _FLOAT_DTYPES:
        raise ConfigurationError(f"unsupported KV dtype: {value!r}")
    return _FLOAT_DTYPES[name]


def json_copy(value: Any, *, name: str = "metadata") -> Any:
    """Validate JSON without accepting NaNs, objects, or non-string keys."""
    if _accel_c.fast():  # accel C (gil_fastpaths): one-pass copy of plain JSON; anything else takes the base path
        copied = _accel_c.json_copy(value)
        if copied is not None:
            return copied[0]
    def check(item: Any) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise ConfigurationError(f"{name} keys must be strings")
            for child in item.values():
                check(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                check(child)
            return
        raise ConfigurationError(f"{name} contains a non-JSON value: {type(item).__name__}")

    def plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {key: plain(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [plain(child) for child in item]
        return item

    try:
        check(value)
        return json.loads(json.dumps(plain(value), allow_nan=False))
    except (RecursionError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} is not a finite acyclic JSON value") from exc


@dataclass(frozen=True)
class ModelIdentity:
    name: str
    fingerprint: str
    layers: int
    kv_channels: int | Mapping[str, int]
    layout: str
    dtype: str | Mapping[str, str]

    def __post_init__(self) -> None:
        _text(self.name, "model name")
        _text(self.fingerprint, "model fingerprint")
        _positive_int(self.layers, "layers")
        _text(self.layout, "layout")
        if isinstance(self.dtype, Mapping):
            per_matrix = {f"{layer}:{target}" for layer in range(self.layers) for target in ("K", "V")}
            if set(self.dtype) not in ({"K", "V"}, per_matrix):
                raise ConfigurationError("dtype mapping must cover K/V or every layer:target matrix")
            object.__setattr__(self, "dtype", MappingProxyType(
                {key: canonical_dtype(value) for key, value in self.dtype.items()}
            ))
        else:
            object.__setattr__(self, "dtype", canonical_dtype(self.dtype))
        if isinstance(self.kv_channels, Mapping):
            if set(self.kv_channels) != {"K", "V"}:
                raise ConfigurationError("kv_channels mapping must contain exactly K and V")
            for target, channels in self.kv_channels.items():
                _positive_int(channels, f"{target} channels")
            object.__setattr__(self, "kv_channels", MappingProxyType(dict(self.kv_channels)))
        else:
            _positive_int(self.kv_channels, "kv_channels")

    def channels_for(self, target: str | MatrixKey) -> int:
        if isinstance(target, MatrixKey):
            target = target.target
        if not isinstance(target, str) or target not in {"K", "V"}:
            raise ConfigurationError("target must be K or V")
        if isinstance(self.kv_channels, Mapping):
            return self.kv_channels[target]
        return self.kv_channels

    def dtype_for(self, target: str | MatrixKey) -> str:
        key = target if isinstance(target, MatrixKey) else None
        target = key.target if key is not None else target
        if not isinstance(target, str) or target not in {"K", "V"}:
            raise ConfigurationError("target must be K or V")
        if key is not None and key.layer >= self.layers:
            raise ConfigurationError("matrix layer is outside the model")
        if not isinstance(self.dtype, Mapping):
            return self.dtype
        if target in self.dtype:
            return self.dtype[target]
        if key is None:
            raise ConfigurationError("layer-specific dtype requires a MatrixKey")
        return self.dtype[f"{key.layer}:{key.target}"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "fingerprint": self.fingerprint, "layers": self.layers,
            "kv_channels": dict(self.kv_channels) if isinstance(self.kv_channels, Mapping) else self.kv_channels,
            "layout": self.layout, "dtype": dict(self.dtype) if isinstance(self.dtype, Mapping) else self.dtype,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelIdentity:
        try:
            return cls(**dict(data))
        except (TypeError, KeyError) as exc:
            raise ConfigurationError("invalid model identity fields") from exc

    def __hash__(self) -> int:
        dtypes = tuple(self.dtype_for(MatrixKey(layer, target))
                       for layer in range(self.layers) for target in ("K", "V"))
        return hash((self.name, self.fingerprint, self.layers, self.channels_for("K"),
                     self.channels_for("V"), self.layout, dtypes))


@dataclass(frozen=True, order=True)
class MatrixKey:
    layer: int
    target: str

    def __post_init__(self) -> None:
        _positive_int(self.layer, "layer", zero=True)
        if not isinstance(self.target, str) or self.target not in {"K", "V"}:
            raise ConfigurationError("matrix target must be K or V")

    def to_dict(self) -> dict[str, Any]:
        return {"layer": self.layer, "target": self.target}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MatrixKey:
        try:
            return cls(**dict(data))
        except (TypeError, KeyError) as exc:
            raise ConfigurationError("invalid matrix key fields") from exc


@dataclass(frozen=True)
class TensorContext:
    """Canonical matrix coordinates with an optional explicit source precision.

    A normalized FP32 pre-RoPE K can retain a BF16 native model identity without
    being pre-rounded. Omitting source_dtype preserves the model's declared
    matrix dtype and the original seven-field serialized context.
    """

    model: ModelIdentity
    key: MatrixKey
    token_start: int
    frames: int
    height: int
    width: int
    key_space: str
    source_dtype: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelIdentity) or not isinstance(self.key, MatrixKey):
            raise ConfigurationError("context requires ModelIdentity and MatrixKey")
        if self.key.layer >= self.model.layers:
            raise ConfigurationError("matrix layer is outside the model")
        _positive_int(self.token_start, "token_start", zero=True)
        for name in ("frames", "height", "width"):
            _positive_int(getattr(self, name), name)
        if not isinstance(self.key_space, str) or self.key_space not in {"pre_rope", "post_rope", "none"}:
            raise ConfigurationError("key_space must be pre_rope, post_rope, or none")
        if self.key.target == "V" and self.key_space != "none":
            raise ConfigurationError("V matrices must use key_space='none'")
        if self.source_dtype is not None:
            object.__setattr__(self, "source_dtype", canonical_dtype(self.source_dtype))

    @property
    def channels(self) -> int:
        return self.model.channels_for(self.key)

    @property
    def dtype(self) -> str:
        return self.model.dtype_for(self.key) if self.source_dtype is None else self.source_dtype

    @property
    def token_count(self) -> int:
        return self.frames * self.height * self.width

    @property
    def token_end(self) -> int:
        return self.token_start + self.token_count

    @property
    def shape(self) -> tuple[int, int]:
        return self.token_count, self.channels

    def to_dict(self) -> dict[str, Any]:
        result = {
            "model": self.model.to_dict(), "key": self.key.to_dict(),
            "token_start": self.token_start, "frames": self.frames,
            "height": self.height, "width": self.width, "key_space": self.key_space,
        }
        if self.source_dtype is not None:
            result["source_dtype"] = self.source_dtype
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TensorContext:
        try:
            fields = dict(data)
            fields["model"] = ModelIdentity.from_dict(fields["model"])
            fields["key"] = MatrixKey.from_dict(fields["key"])
            return cls(**fields)
        except (TypeError, KeyError) as exc:
            raise ConfigurationError("invalid tensor context fields") from exc


@dataclass(frozen=True)
class CodecSpec:
    """Representation parameters; fraction controls only leading-column mode.

    Per-channel mode canonicalizes the fraction to one; global mode to zero.
    Leading-column mode retains the supplied fraction. Providers also record
    the actual mode-derived protected count in their normalization descriptor.

    ``inter_prediction`` selects the GOP of ``svd_hevc`` coefficient streams.
    ``False`` (the default) emits genuine all-intra streams; ``True`` emits the
    documented inter GOP (see ``reelcache._hevc.encoder_record``). Version-2
    svd_hevc specifications always serialize the flag. ``None`` is reserved for
    version-1 svd_hevc records that predate it: they keep their historical
    x265 preset-default GOP identity, still decode, and cannot be used for new
    encodes. RAW and SVD-only representations carry no HEVC stream, so their
    flag is canonically ``False`` and is not serialized (identity unchanged).
    """

    kind: str = "raw"
    rank: int | None = None
    normalization: str = "per_channel"
    protected_fraction: float = 1.0
    pix_fmt: str = "yuv444p"
    qp: int = 8
    preset: str = "ultrafast"
    ffmpeg_path: str | None = None
    inter_prediction: bool | None = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in {"raw", "svd", "svd_hevc"}:
            raise ConfigurationError("codec kind must be raw, svd, or svd_hevc")
        if self.kind == "raw":
            if self.rank is not None:
                raise ConfigurationError("raw encoding does not take a rank")
        else:
            _positive_int(self.rank, "rank")
        if (not isinstance(self.normalization, str)
                or self.normalization not in {"per_channel", "global", "leading_columns"}):
            raise ConfigurationError("unsupported normalization")
        if (isinstance(self.protected_fraction, bool)
                or not isinstance(self.protected_fraction, (int, float))
                or not math.isfinite(self.protected_fraction)
                or not 0 <= self.protected_fraction <= 1):
            raise ConfigurationError("protected_fraction must be in [0, 1]")
        if self.normalization == "global":
            object.__setattr__(self, "protected_fraction", 0.0)
        elif self.normalization == "per_channel":
            object.__setattr__(self, "protected_fraction", 1.0)
        _text(self.pix_fmt, "pix_fmt")
        _text(self.preset, "preset")
        if self.ffmpeg_path is not None:
            _text(self.ffmpeg_path, "ffmpeg_path")
            if "\x00" in self.ffmpeg_path:
                raise ConfigurationError("ffmpeg_path must not contain NUL")
        if type(self.qp) is not int or not 0 <= self.qp <= 51:
            raise ConfigurationError("qp must be an integer in [0, 51]")
        if self.inter_prediction is not None and type(self.inter_prediction) is not bool:
            raise ConfigurationError("inter_prediction must be a boolean")
        if self.kind != "svd_hevc":
            if self.inter_prediction:
                raise ConfigurationError("inter prediction applies only to svd_hevc coefficient streams")
            object.__setattr__(self, "inter_prediction", False)

    @property
    def legacy_gop(self) -> bool:
        """True only for a version-1 svd_hevc record without an explicit GOP choice."""
        return self.inter_prediction is None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "kind": self.kind, "rank": self.rank, "normalization": self.normalization,
            "protected_fraction": self.protected_fraction, "pix_fmt": self.pix_fmt,
            "qp": self.qp, "preset": self.preset, "ffmpeg_path": self.ffmpeg_path,
        }
        if self.kind == "svd_hevc" and self.inter_prediction is not None:
            result["inter_prediction"] = self.inter_prediction
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CodecSpec:
        """Parse a serialized specification without reinterpreting its version.

        A record without ``inter_prediction`` is version 1: svd_hevc records
        keep the legacy (``None``) GOP identity, never the new all-intra default.
        Configuration layers that intend the current default must say so.
        """
        try:
            fields = dict(data)
            if "inter_prediction" not in fields:
                fields["inter_prediction"] = None
            elif type(fields["inter_prediction"]) is not bool:
                raise ConfigurationError("serialized inter_prediction must be a boolean when present")
            return cls(**fields)
        except (TypeError, KeyError) as exc:
            raise ConfigurationError("invalid codec spec fields") from exc


@dataclass(frozen=True)
class EncodedTensor:
    metadata: Mapping[str, Any]
    payloads: Mapping[str, bytes]
    basis: bytes | None = None
    metrics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, Mapping) or not isinstance(self.payloads, Mapping):
            raise ConfigurationError("encoded metadata and payloads must be mappings")
        if not self.payloads:
            raise ConfigurationError("encoded tensors require at least one payload")
        for name, payload in self.payloads.items():
            _text(name, "payload name")
            if not isinstance(payload, bytes):
                raise ConfigurationError("encoded payloads must be immutable bytes")
        if self.basis is not None and not isinstance(self.basis, bytes):
            raise ConfigurationError("encoded basis must be immutable bytes")
        if self.metrics is not None and not isinstance(self.metrics, Mapping):
            raise ConfigurationError("metrics must be a mapping")
        object.__setattr__(self, "metadata", json_copy(self.metadata))
        object.__setattr__(self, "payloads", MappingProxyType(dict(self.payloads)))
        if self.metrics is not None:
            object.__setattr__(self, "metrics", json_copy(self.metrics, name="metrics"))


# ---------------------------------------------------------------------------
# Latest paper method (SGLang Step 4): energy targets, immutable basis
# versions, matrix records, lineage nodes, cache epochs and the paper's online
# ridge predictor state. These are durable, validated, immutable records; the
# mutable resource owners live in predictor_registry/rank_method/manager.

RANK_BUCKET = 32
_OBJECT_REF = re.compile(r"^(blob|basis|manifest):[0-9a-f]{64}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ENERGY_SLACK = 1e-6
BASIS_DOMAINS = MappingProxyType({"K": "normalized_pre_rope_k", "V": "direct_v"})
NODE_KINDS = ("root", "chunk", "recache", "segment")
EPOCH_CAUSES = ("initial", "kv_recache", "longcat_segment")
PREDICTOR_REGIMES = ("fresh", "residual")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def thaw_json(value: Any) -> Any:
    """Plain JSON copy of a frozen mapping/tuple structure."""
    if _accel_c.fast():  # accel C (gil_fastpaths): exact-type checks before the ABC checks, identical copy
        return _accel_c.thaw_json(value)
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(item) for item in value]
    return value


def frozen_json(value: Any, *, name: str = "metadata") -> Any:
    """Validated JSON frozen into read-only mappings and tuples (deep, not shallow)."""
    return _deep_freeze(json_copy(thaw_json(value), name=name))


def canonical_digest(value: Any) -> str:
    """SHA256 of canonical JSON (sorted keys, no whitespace, no NaN)."""
    return hashlib.sha256(json.dumps(
        json_copy(thaw_json(value)), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def object_ref(value: Any, name: str, *, kinds: tuple[str, ...] = ("manifest",)) -> str:
    if (not isinstance(value, str) or _OBJECT_REF.fullmatch(value) is None
            or value.split(":", 1)[0] not in kinds):
        raise ConfigurationError(f"{name} must be an immutable {'/'.join(kinds)} object reference")
    return value


def _hex64(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ConfigurationError(f"{name} must be a lowercase SHA256 hex digest")
    return value


def _fraction(value: Any, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 < value <= 1):
        raise ConfigurationError(f"{name} must be a finite fraction in (0, 1]")
    return float(value)


def _energy(value: Any, name: str) -> float:
    """A measured energy fraction: finite and not above one beyond rounding slack."""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value > 1 + _ENERGY_SLACK):
        raise ConfigurationError(f"{name} must be a finite measured energy fraction <= 1")
    return float(value)


def _optional_text(value: Any, name: str) -> None:
    if value is not None:
        _text(value, name)


def entry_key(value: Any, name: str = "matrix entry") -> Any:
    """A node entry key: ``MatrixKey`` or ``(MatrixKey, stream label)`` for CFG streams."""
    if isinstance(value, MatrixKey):
        return value
    if (isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], MatrixKey)
            and isinstance(value[1], str) and value[1] and value[1] == value[1].strip()):
        return value
    raise ConfigurationError(f"{name} keys must be MatrixKey or (MatrixKey, stream label)")


def entry_parts(value: Any) -> tuple[MatrixKey, str | None]:
    """``(MatrixKey, stream or None)`` of a node entry key."""
    value = entry_key(value)
    return (value, None) if isinstance(value, MatrixKey) else value


def entry_order(value: Any) -> tuple:
    key, stream = entry_parts(value)
    return key.layer, key.target, stream is not None, stream or ""


def _entry_to_dict(value: Any) -> dict[str, Any]:
    key, stream = entry_parts(value)
    return {"key": key.to_dict()} if stream is None else {"key": key.to_dict(), "stream": stream}


def _entry_from_dict(item: Mapping[str, Any]) -> Any:
    key = MatrixKey.from_dict(item["key"])
    stream = item.get("stream")
    return key if stream is None else entry_key((key, stream))


def _key_pairs(values: Any, name: str, convert) -> tuple:
    """Normalize ``{entry: value}`` or ``((entry, value), ...)`` to sorted unique pairs.

    Entries are ``MatrixKey`` or ``(MatrixKey, stream label)`` (CFG streams).
    """
    items = values.items() if isinstance(values, Mapping) else values
    if not isinstance(items, (tuple, list)) and not isinstance(values, Mapping):
        raise ConfigurationError(f"{name} must be a MatrixKey mapping or pair sequence")
    pairs = []
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ConfigurationError(f"{name} entries must be (MatrixKey, value) pairs")
        pairs.append((entry_key(item[0], name), convert(item[1])))
    pairs.sort(key=lambda pair: entry_order(pair[0]))
    if len({key for key, _ in pairs}) != len(pairs):
        raise ConfigurationError(f"{name} contains duplicate matrix keys")
    return tuple(pairs)


def _check_streams(streams: Any, entries: Sequence[Any], name: str) -> tuple[str, ...]:
    """Validate a node's CFG stream labels against its entry keys (complete per stream)."""
    streams = tuple(streams)
    if any(not isinstance(item, str) or not item or item != item.strip() for item in streams):
        raise ConfigurationError("stream labels must be nonempty strings without surrounding whitespace")
    if len(set(streams)) != len(streams) or list(streams) != sorted(streams):
        raise ConfigurationError("stream labels must be unique and sorted")
    labels = {entry_parts(item)[1] for item in entries}
    if not streams:
        if labels - {None}:
            raise ConfigurationError(f"{name} carry stream labels but the node declares no streams")
        return streams
    if None in labels or not labels <= set(streams):
        raise ConfigurationError(f"{name} of a multi-stream node must be labeled with declared streams")
    if entries:
        per_stream: dict[str, set] = {stream: set() for stream in streams}
        for item in entries:
            key, stream = entry_parts(item)
            per_stream[stream].add(key)
        reference = per_stream[streams[0]]
        if any(keys != reference for keys in per_stream.values()):
            raise ConfigurationError(f"{name} must cover the same matrices in every stream")
    return streams


@dataclass(frozen=True)
class EnergyTargets:
    """Explicit pre-codec energy fractions: normalized pre-RoPE K and direct V.

    ``provisional`` marks targets that only qualify mechanics, integration and
    performance until the energy sweeps select a model-specific profile.
    A fraction of exactly 1.0 is a labeled full-width numerical diagnostic,
    never a learned-tail request, and is not lossless RAW.
    """

    profile: str
    key: float
    value: float
    provisional: bool
    note: str | None = None

    def __post_init__(self) -> None:
        _text(self.profile, "energy profile")
        object.__setattr__(self, "key", _fraction(self.key, "K energy target"))
        object.__setattr__(self, "value", _fraction(self.value, "V energy target"))
        if type(self.provisional) is not bool:
            raise ConfigurationError("provisional must be boolean")
        _optional_text(self.note, "energy profile note")

    def for_target(self, target: str | MatrixKey) -> float:
        target = target.target if isinstance(target, MatrixKey) else target
        if target == "K":
            return self.key
        if target == "V":
            return self.value
        raise ConfigurationError("energy target selector must be K or V")

    def is_diagnostic(self, target: str | MatrixKey) -> bool:
        return self.for_target(target) == 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "reelcache.energy_targets.v1", "profile": self.profile, "key": self.key,
                "value": self.value, "provisional": self.provisional, "note": self.note}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EnergyTargets:
        fields = dict(data)
        if fields.pop("schema", None) != "reelcache.energy_targets.v1":
            raise ConfigurationError("unsupported energy target schema")
        try:
            return cls(**fields)
        except TypeError as exc:
            raise ConfigurationError("invalid energy target fields") from exc

    @property
    def fingerprint(self) -> str:
        return canonical_digest(self.to_dict())


_PROVISIONAL = "provisional K98/V95 until the energy sweeps select a model-specific profile"
ENERGY_TARGET_PROFILES = MappingProxyType({
    "causvid": EnergyTargets("causvid", 0.98, 0.95, False, "final: pre-RoPE K 98%, V 95%"),
    "raven": EnergyTargets("raven", 0.98, 0.95, True, _PROVISIONAL),
    "longlive1": EnergyTargets("longlive1", 0.98, 0.95, True, _PROVISIONAL),
    "longlive2": EnergyTargets("longlive2", 0.98, 0.95, True, _PROVISIONAL),
    "longcat_720p": EnergyTargets("longcat_720p", 0.98, 0.95, True, _PROVISIONAL),
})


def energy_targets(profile: str) -> EnergyTargets:
    """Built-in versioned profile; unknown models are unconfigured, never CausVid's."""
    try:
        return ENERGY_TARGET_PROFILES[profile]
    except (KeyError, TypeError) as exc:
        raise ConfigurationError(f"no energy target profile is configured for {profile!r}") from exc


@dataclass(frozen=True)
class EnergyReport:
    """Separate ideal-subspace, stored-factor and post-codec measurements.

    ``ideal_subspace_energy`` is the orthogonal projection onto the span of the
    stored rows of the view; ``stored_factor_energy`` is the actual direct
    residual of FP32 ``P = A Q^T`` with the FP16 rows ``Q``; the post-codec
    error is ``None`` until measured, never zero by default.
    """

    target: float
    ideal_subspace_energy: float
    stored_factor_energy: float
    post_codec_relative_error: float | None = None
    observed_rank: int | None = None
    bucket_rank: int = 0
    rank_view: int = 1
    zero_input: bool = False
    diagnostic_full_width: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _fraction(self.target, "energy target"))
        object.__setattr__(self, "ideal_subspace_energy",
                           _energy(self.ideal_subspace_energy, "ideal subspace energy"))
        object.__setattr__(self, "stored_factor_energy",
                           _energy(self.stored_factor_energy, "stored-factor energy"))
        if self.post_codec_relative_error is not None:
            value = self.post_codec_relative_error
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ConfigurationError("post-codec relative error must be finite and nonnegative or None")
            object.__setattr__(self, "post_codec_relative_error", float(value))
        if self.observed_rank is not None:
            _positive_int(self.observed_rank, "observed rank", zero=True)
        _positive_int(self.bucket_rank, "bucket rank", zero=True)
        _positive_int(self.rank_view, "rank view")
        for name in ("zero_input", "diagnostic_full_width"):
            if type(getattr(self, name)) is not bool:
                raise ConfigurationError(f"{name} must be boolean")
        if self.zero_input and (self.ideal_subspace_energy != 1.0 or self.stored_factor_energy != 1.0):
            raise ConfigurationError("a zero input is represented exactly")

    @property
    def meets_target(self) -> bool:
        return self.zero_input or self.stored_factor_energy >= self.target

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in (
            "target", "ideal_subspace_energy", "stored_factor_energy", "post_codec_relative_error",
            "observed_rank", "bucket_rank", "rank_view", "zero_input", "diagnostic_full_width",
        )}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EnergyReport:
        try:
            return cls(**dict(data))
        except TypeError as exc:
            raise ConfigurationError("invalid energy report fields") from exc


@dataclass(frozen=True)
class BasisVersion:
    """One immutable node of a per-layer/target basis DAG.

    A version stores only its newly added FP16 row block (``added_rows_ref``)
    and references its parent version; the canonical contiguous view is the
    parent's rows followed by the added rows, in that order, never reordered.
    ``view_sha256`` binds the assembled ``[total_rank, channels]`` bytes.
    """

    parent_ref: str | None
    added_rows_ref: str
    added_rank: int
    total_rank: int
    channels: int
    group_id: str
    key: MatrixKey
    model_fingerprint: str
    depth: int
    view_sha256: str
    dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.parent_ref is not None:
            object_ref(self.parent_ref, "basis parent version")
        object_ref(self.added_rows_ref, "added basis row block", kinds=("basis",))
        _positive_int(self.added_rank, "added rank")
        _positive_int(self.total_rank, "total rank")
        _positive_int(self.channels, "basis channels")
        _positive_int(self.depth, "basis version depth", zero=True)
        if not isinstance(self.key, MatrixKey):
            raise ConfigurationError("basis version key must be a MatrixKey")
        _text(self.model_fingerprint, "basis model fingerprint")
        _hex64(self.group_id, "basis group id")
        _hex64(self.view_sha256, "basis view digest")
        if self.dtype != "float16":
            raise ConfigurationError("stored basis rows are FP16")
        if self.total_rank > self.channels:
            raise ConfigurationError("basis rank cannot exceed its channel count")
        if (self.parent_ref is None) != (self.depth == 0):
            raise ConfigurationError("only a depth-zero root version has no parent")
        if self.parent_ref is None and self.total_rank != self.added_rank:
            raise ConfigurationError("a root version's total rank equals its block rank")
        if self.parent_ref is not None and self.total_rank <= self.added_rank:
            raise ConfigurationError("an extension version must add rows to a nonempty parent")

    @property
    def domain(self) -> str:
        return BASIS_DOMAINS[self.key.target]

    @property
    def parent_rank(self) -> int:
        return self.total_rank - self.added_rank

    @property
    def added_bytes(self) -> int:
        return self.added_rank * self.channels * 2

    @staticmethod
    def root_group_id(model_fingerprint: str, key: MatrixKey, rows_ref: str) -> str:
        return canonical_digest({"model": model_fingerprint, "key": key.to_dict(), "root_rows": rows_ref})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "reelcache.basis_version.v1", "parent": self.parent_ref,
            "added_rows": self.added_rows_ref, "added_rank": self.added_rank,
            "total_rank": self.total_rank, "channels": self.channels, "dtype": self.dtype,
            "group_id": self.group_id, "key": self.key.to_dict(), "domain": self.domain,
            "model_fingerprint": self.model_fingerprint, "depth": self.depth,
            "view_sha256": self.view_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BasisVersion:
        try:
            fields = dict(data)
            if fields.pop("schema") != "reelcache.basis_version.v1":
                raise ConfigurationError("unsupported basis version schema")
            domain = fields.pop("domain")
            version = cls(
                parent_ref=fields.pop("parent"), added_rows_ref=fields.pop("added_rows"),
                key=MatrixKey.from_dict(fields.pop("key")), **fields,
            )
        except (KeyError, TypeError) as exc:
            raise ConfigurationError("invalid basis version fields") from exc
        if domain != version.domain:
            raise ConfigurationError("basis version domain disagrees with its K/V key")
        return version


@dataclass(frozen=True)
class ActiveBasisBinding:
    """A lineage's active basis for one layer/target plus its refit memory.

    ``fresh_rank`` is the rank of the layer/target's most recent fresh fit on
    this lineage (the refit estimate). ``coefficient_rate`` is the measured
    packet bytes per coefficient value of the last encoded packet, valid only
    for the codec/geometry identity ``rate_identity``. ``refit_debt`` is the
    cumulative refit rule's state (``RankMethodConfig.refit_rule``): estimated
    coefficient bytes the active basis has stored beyond a fresh fit since the
    last fresh fit or refit. It is serialized only when nonzero, so bindings
    without it (every archive before the rule, the paper rule) read as debt 0.
    """

    version_ref: str | None = None
    total_rank: int = 0
    group_id: str | None = None
    fresh_rank: int | None = None
    fresh_chunk_id: str | None = None
    coefficient_rate: float | None = None
    rate_identity: str | None = None
    refit_debt: float = 0.0

    def __post_init__(self) -> None:
        if self.version_ref is None:
            if self.total_rank != 0 or self.group_id is not None:
                raise ConfigurationError("an empty binding has no rank or group")
        else:
            object_ref(self.version_ref, "active basis version")
            _positive_int(self.total_rank, "active basis rank")
            _hex64(self.group_id, "active basis group")
        if (self.fresh_rank is None) != (self.fresh_chunk_id is None):
            raise ConfigurationError("fresh rank and its chunk are recorded together")
        if self.fresh_rank is not None:
            _positive_int(self.fresh_rank, "fresh rank")
            _text(self.fresh_chunk_id, "fresh chunk id")
        if (self.coefficient_rate is None) != (self.rate_identity is None):
            raise ConfigurationError("a coefficient rate requires its codec/geometry identity")
        if self.coefficient_rate is not None:
            rate = self.coefficient_rate
            if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
                raise ConfigurationError("coefficient rate must be finite and positive")
            object.__setattr__(self, "coefficient_rate", float(rate))
            _hex64(self.rate_identity, "coefficient rate identity")
        debt = self.refit_debt
        if isinstance(debt, bool) or not isinstance(debt, (int, float)) or not math.isfinite(debt) or debt < 0:
            raise ConfigurationError("refit debt must be finite and nonnegative")
        object.__setattr__(self, "refit_debt", float(debt) if debt else 0.0)
        if self.refit_debt and (self.version_ref is None or self.fresh_rank is None):
            raise ConfigurationError("a refit debt needs an active basis and its fresh rank")

    def to_dict(self) -> dict[str, Any]:
        data = {name: getattr(self, name) for name in (
            "version_ref", "total_rank", "group_id", "fresh_rank", "fresh_chunk_id",
            "coefficient_rate", "rate_identity",
        )}
        # Omitted at zero: node manifests without a debt (the paper rule, earlier archives) are unchanged.
        if self.refit_debt:
            data["refit_debt"] = self.refit_debt
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ActiveBasisBinding:
        try:
            fields = dict(data)
            if "refit_debt" in fields and fields["refit_debt"] == 0:
                raise ConfigurationError("a zero refit debt is omitted")
            return cls(**fields)
        except TypeError as exc:
            raise ConfigurationError("invalid active basis binding fields") from exc


def _factor_pack_tile(codec: CodecSpec, decision: Any) -> bool:
    """wl-base (W1c): whether a record is a tile of a ``reelcache.factor_pack.v1`` node pack (FP16 svd factors, the
    SVD-only baseline): an svd codec whose decision names the pack's factor-tile canvas."""
    packet = decision.get("packet") if isinstance(decision, Mapping) else None
    canvas = packet.get("canvas") if isinstance(packet, Mapping) else None
    return codec.kind == "svd" and isinstance(canvas, Mapping) and canvas.get("policy") == "factor_tiles"


@dataclass(frozen=True)
class MatrixRecord:
    """One committed matrix: its packet (or tile of a node pack) plus its explicit basis view.

    Lossy records reference an immutable :class:`BasisVersion` and a legal
    prefix ``rank_view`` of its canonical rows; the packet's basis descriptor
    binds exactly those prefix bytes. RAW records carry no basis or rank.
    ``packet_tile`` is the tile index inside a ``reelcache.tensor_pack.v2`` node
    pack (None: an independent single-matrix packet); ``stream`` is the CFG
    stream label of a multi-stream node (None: single stream).
    """

    context: TensorContext
    codec: CodecSpec
    packet_ref: str
    basis_version_ref: str | None
    rank_view: int | None
    energy: EnergyReport | None
    predictor_revision_ref: str | None
    decision: Mapping[str, Any]
    packet_tile: int | None = None
    stream: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.context, TensorContext) or not isinstance(self.codec, CodecSpec):
            raise ConfigurationError("matrix record requires a TensorContext and CodecSpec")
        object_ref(self.packet_ref, "matrix packet")
        if self.codec.kind == "raw":
            if self.basis_version_ref is not None or self.rank_view is not None or self.energy is not None:
                raise ConfigurationError("RAW records carry no basis, rank view or factor energy")
        else:
            object_ref(self.basis_version_ref, "matrix basis version")
            _positive_int(self.rank_view, "rank view")
            if self.rank_view != self.codec.rank:
                raise ConfigurationError("rank view must equal the resolved codec rank")
            if not isinstance(self.energy, EnergyReport) or self.energy.rank_view != self.rank_view:
                raise ConfigurationError("lossy records require an energy report for their rank view")
            if self.context.key.target == "K" and self.context.key_space != "pre_rope":
                raise ConfigurationError("lossy K records must be normalized pre-RoPE K")
        if self.packet_tile is not None:
            _positive_int(self.packet_tile, "packet tile", zero=True)
            if not _factor_pack_tile(self.codec, self.decision) and (
                    self.codec.kind != "svd_hevc" or self.codec.inter_prediction is not False):
                raise ConfigurationError("only all-intra svd_hevc matrices are tiles of a node pack")
        if self.stream is not None:
            entry_key((self.context.key, self.stream), "matrix record stream")
        if self.predictor_revision_ref is not None:
            _text(self.predictor_revision_ref, "predictor revision reference")
        object.__setattr__(self, "decision", frozen_json(self.decision, name="matrix decision"))

    @property
    def entry(self) -> Any:
        return self.context.key if self.stream is None else (self.context.key, self.stream)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "schema": "reelcache.matrix_record.v1", "context": self.context.to_dict(),
            "codec": self.codec.to_dict(), "packet": self.packet_ref,
            "basis_version": self.basis_version_ref, "rank_view": self.rank_view,
            "energy": None if self.energy is None else self.energy.to_dict(),
            "predictor_revision": self.predictor_revision_ref,
            "decision": thaw_json(self.decision),
        }
        # Optional fields are omitted when unset so earlier records keep their exact bytes.
        if self.packet_tile is not None:
            data["packet_tile"] = self.packet_tile
        if self.stream is not None:
            data["stream"] = self.stream
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MatrixRecord:
        try:
            fields = dict(data)
            if fields.pop("schema") != "reelcache.matrix_record.v1":
                raise ConfigurationError("unsupported matrix record schema")
            energy = fields.pop("energy")
            if "packet_tile" in fields and fields["packet_tile"] is None:
                raise ConfigurationError("an unset packet tile is omitted, not null")
            if "stream" in fields and fields["stream"] is None:
                raise ConfigurationError("an unset stream label is omitted, not null")
            return cls(
                context=TensorContext.from_dict(fields.pop("context")),
                codec=CodecSpec.from_dict(fields.pop("codec")),
                packet_ref=fields.pop("packet"), basis_version_ref=fields.pop("basis_version"),
                rank_view=fields.pop("rank_view"),
                energy=None if energy is None else EnergyReport.from_dict(energy),
                predictor_revision_ref=fields.pop("predictor_revision"),
                decision=fields.pop("decision"), **fields,
            )
        except (KeyError, TypeError) as exc:
            raise ConfigurationError("invalid matrix record fields") from exc


@dataclass(frozen=True)
class TokenRangeRecords:
    """Matrix records of one re-materialized (or segment-local) token range."""

    chunk_id: str
    token_start: int
    token_end: int
    frames: int
    height: int
    width: int
    records: tuple[tuple[Any, str], ...]
    streams: tuple[str, ...] = ()
    # The range's node pack (packed commits since SGLang Step 7; None for per-matrix/RAW and older ranges): restore
    # finds the pack without loading every matrix record; audits still validate the records against it.
    packet_ref: str | None = None

    def __post_init__(self) -> None:
        _text(self.chunk_id, "range chunk id")
        if self.packet_ref is not None:
            object_ref(self.packet_ref, "range node pack")
        _positive_int(self.token_start, "range token start", zero=True)
        for name in ("frames", "height", "width"):
            _positive_int(getattr(self, name), f"range {name}")
        if self.token_end != self.token_start + self.frames * self.height * self.width:
            raise ConfigurationError("range token span must equal its geometry")
        object.__setattr__(self, "records", _key_pairs(
            self.records, "range records", lambda ref: object_ref(ref, "range matrix record"),
        ))
        if not self.records:
            raise ConfigurationError("a re-materialized range carries matrix records")
        object.__setattr__(self, "streams", _check_streams(
            self.streams, [key for key, _ in self.records], "range records"))

    def to_dict(self) -> dict[str, Any]:
        data = {
            "chunk_id": self.chunk_id, "token_start": self.token_start, "token_end": self.token_end,
            "frames": self.frames, "height": self.height, "width": self.width,
            "records": [{**_entry_to_dict(key), "record": ref} for key, ref in self.records],
        }
        if self.streams:
            data["streams"] = list(self.streams)
        if self.packet_ref is not None:
            data["packet"] = self.packet_ref
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TokenRangeRecords:
        try:
            fields = dict(data)
            records = tuple((_entry_from_dict(item), item["record"]) for item in fields.pop("records"))
            streams = tuple(fields.pop("streams", ()))
            if "streams" in data and not streams:
                raise ConfigurationError("an empty stream list is omitted")
            packet = fields.pop("packet", None)
            if "packet" in data and packet is None:
                raise ConfigurationError("an absent node pack is omitted")
            return cls(records=records, streams=streams, packet_ref=packet, **fields)
        except (KeyError, TypeError) as exc:
            raise ConfigurationError("invalid token range fields") from exc


@dataclass(frozen=True)
class CacheEpoch:
    """A cache version: initial, a KV recache (LongLive 1.0), or a LongCat segment.

    Restore resolves each attended token range to the newest version on the
    node's ancestor path; nodes before this epoch keep their earlier versions.
    """

    epoch_id: str
    parent_epoch_ref: str | None
    cause: str
    position_space: str
    ranges: tuple[TokenRangeRecords, ...]
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        _text(self.epoch_id, "epoch id")
        if self.parent_epoch_ref is not None:
            object_ref(self.parent_epoch_ref, "parent epoch")
        if self.cause not in EPOCH_CAUSES:
            raise ConfigurationError(f"epoch cause must be one of {EPOCH_CAUSES}")
        if self.position_space not in ("global", "segment_local"):
            raise ConfigurationError("epoch position space must be global or segment_local")
        ranges = tuple(self.ranges)
        if any(not isinstance(item, TokenRangeRecords) for item in ranges):
            raise ConfigurationError("epoch ranges must be TokenRangeRecords")
        ordered = sorted(ranges, key=lambda item: item.token_start)
        if ordered != list(ranges) or any(
            left.token_end > right.token_start for left, right in zip(ordered, ordered[1:])
        ):
            raise ConfigurationError("epoch ranges must be ordered and non-overlapping")
        if self.cause == "initial":
            if ranges or self.parent_epoch_ref is not None or self.position_space != "global":
                raise ConfigurationError("an initial epoch has no parent and no re-materialized ranges")
        elif self.cause == "kv_recache":
            if not ranges or self.parent_epoch_ref is None or self.position_space != "global":
                raise ConfigurationError("a KV recache epoch re-materializes global ranges of a parent epoch")
        elif (self.position_space != "segment_local" or len(ranges) != 1
              or ranges[0].token_start != 0 or self.parent_epoch_ref is None):
            raise ConfigurationError("a LongCat segment epoch holds one segment-local range starting at zero")
        object.__setattr__(self, "ranges", ranges)
        object.__setattr__(self, "details", frozen_json(self.details, name="epoch details"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "reelcache.cache_epoch.v1", "epoch_id": self.epoch_id,
            "parent_epoch": self.parent_epoch_ref, "cause": self.cause,
            "position_space": self.position_space,
            "ranges": [item.to_dict() for item in self.ranges], "details": thaw_json(self.details),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CacheEpoch:
        try:
            fields = dict(data)
            if fields.pop("schema") != "reelcache.cache_epoch.v1":
                raise ConfigurationError("unsupported cache epoch schema")
            return cls(
                epoch_id=fields.pop("epoch_id"), parent_epoch_ref=fields.pop("parent_epoch"),
                ranges=tuple(TokenRangeRecords.from_dict(item) for item in fields.pop("ranges")),
                **fields,
            )
        except (KeyError, TypeError) as exc:
            raise ConfigurationError("invalid cache epoch fields") from exc


@dataclass(frozen=True)
class ChunkNode:
    """One immutable state-tree node.

    ``root`` starts a lineage (initial epoch, no KV); ``chunk`` appends frames
    and carries their matrix records; ``recache`` adds no frames and publishes a
    new KV-recache epoch; ``segment`` publishes a LongCat segment-local epoch.
    ``active_basis`` is the complete per-key binding the next commit tests.
    """

    node_kind: str
    chunk_id: str
    parent_ref: str | None
    cache_epoch_id: str
    epoch_ref: str
    token_start: int
    token_end: int
    frames: int
    height: int
    width: int
    frontier: int
    matrix_records: tuple[tuple[Any, str], ...]
    active_basis: tuple[tuple[Any, ActiveBasisBinding], ...]
    execution_state_ref: str | None = None
    streams: tuple[str, ...] = ()
    # A chunk node's node pack (packed commits since SGLang Step 7; see TokenRangeRecords.packet_ref).
    packet_ref: str | None = None

    def __post_init__(self) -> None:
        if self.node_kind not in NODE_KINDS:
            raise ConfigurationError(f"node kind must be one of {NODE_KINDS}")
        _text(self.chunk_id, "node chunk id")
        if self.parent_ref is not None:
            object_ref(self.parent_ref, "parent node")
        if (self.node_kind == "root") != (self.parent_ref is None):
            raise ConfigurationError("exactly the root node has no parent")
        _text(self.cache_epoch_id, "cache epoch id")
        object_ref(self.epoch_ref, "cache epoch")
        for name in ("token_start", "token_end", "frames", "height", "width", "frontier"):
            _positive_int(getattr(self, name), f"node {name}", zero=True)
        span = self.frames * self.height * self.width
        if self.token_end != self.token_start + span:
            raise ConfigurationError("node token span must equal its geometry")
        object.__setattr__(self, "matrix_records", _key_pairs(
            self.matrix_records, "node matrix records", lambda ref: object_ref(ref, "node matrix record"),
        ))
        object.__setattr__(self, "active_basis", _key_pairs(
            self.active_basis, "active basis bindings", self._binding,
        ))
        streams = _check_streams(self.streams, [key for key, _ in self.matrix_records], "node matrix records")
        _check_streams(streams, [key for key, _ in self.active_basis], "active basis bindings")
        object.__setattr__(self, "streams", streams)
        if self.node_kind == "chunk":
            if not span or not self.matrix_records or self.frontier != self.token_end:
                raise ConfigurationError("a chunk node appends nonempty frames with matrix records")
        elif self.matrix_records:
            raise ConfigurationError("only chunk nodes carry their own matrix records")
        if self.node_kind in ("root", "recache") and span:
            raise ConfigurationError("root and recache nodes add no frames")
        if self.node_kind == "segment" and (self.token_start != 0 or not span or self.frontier != self.token_end):
            raise ConfigurationError("a segment node owns one segment-local range starting at zero")
        if self.execution_state_ref is not None:
            object_ref(self.execution_state_ref, "execution state", kinds=("blob",))
        if self.packet_ref is not None:
            object_ref(self.packet_ref, "node pack")
            if self.node_kind != "chunk":
                raise ConfigurationError("only chunk nodes carry their own node pack (epoch ranges carry theirs)")

    @staticmethod
    def _binding(value: Any) -> ActiveBasisBinding:
        if not isinstance(value, ActiveBasisBinding):
            raise ConfigurationError("active basis entries must be ActiveBasisBinding values")
        return value

    def binding(self, key: MatrixKey, stream: str | None = None) -> ActiveBasisBinding:
        entry = key if stream is None else (key, stream)
        for candidate, value in self.active_basis:
            if candidate == entry:
                return value
        return ActiveBasisBinding()

    def record_ref(self, key: MatrixKey, stream: str | None = None) -> str | None:
        entry = key if stream is None else (key, stream)
        for candidate, value in self.matrix_records:
            if candidate == entry:
                return value
        return None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "schema": "reelcache.chunk_node.v2" if self.streams else "reelcache.chunk_node.v1",
            "node_kind": self.node_kind, "chunk_id": self.chunk_id,
            "parent": self.parent_ref, "cache_epoch_id": self.cache_epoch_id, "epoch": self.epoch_ref,
            "token_start": self.token_start, "token_end": self.token_end, "frames": self.frames,
            "height": self.height, "width": self.width, "frontier": self.frontier,
            "matrices": [{**_entry_to_dict(key), "record": ref} for key, ref in self.matrix_records],
            "active_basis": [{**_entry_to_dict(key), "binding": value.to_dict()}
                             for key, value in self.active_basis],
            "execution_state": self.execution_state_ref,
        }
        if self.streams:
            data["streams"] = list(self.streams)
        if self.packet_ref is not None:
            data["packet"] = self.packet_ref
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChunkNode:
        try:
            fields = dict(data)
            schema = fields.pop("schema")
            if schema not in ("reelcache.chunk_node.v1", "reelcache.chunk_node.v2"):
                raise ConfigurationError("unsupported chunk node schema")
            streams = tuple(fields.pop("streams", ()))
            if (schema == "reelcache.chunk_node.v2") != bool(streams) or ("streams" in data and not streams):
                raise ConfigurationError("chunk node v2 (and only v2) declares its CFG streams")
            packet = fields.pop("packet", None)
            if "packet" in data and packet is None:
                raise ConfigurationError("an absent node pack is omitted")
            return cls(packet_ref=packet, 
                parent_ref=fields.pop("parent"), epoch_ref=fields.pop("epoch"),
                matrix_records=tuple((_entry_from_dict(item), item["record"])
                                     for item in fields.pop("matrices")),
                active_basis=tuple((_entry_from_dict(item), ActiveBasisBinding.from_dict(item["binding"]))
                                   for item in fields.pop("active_basis")),
                execution_state_ref=fields.pop("execution_state"), streams=streams, **fields,
            )
        except (KeyError, TypeError) as exc:
            raise ConfigurationError("invalid chunk node fields") from exc


@dataclass(frozen=True, order=True)
class PredictorKey:
    """Learner scope: model/source, layer, K|V, geometry, solver/detector, regime.

    Fresh fits and augmentation-residual fits never share one learner.
    """

    model_fingerprint: str
    layer: int
    target: str
    rows: int
    channels: int
    solver: str
    detector: str
    regime: str

    def __post_init__(self) -> None:
        _text(self.model_fingerprint, "predictor model fingerprint")
        MatrixKey(self.layer, self.target)
        _positive_int(self.rows, "predictor rows")
        _positive_int(self.channels, "predictor channels")
        _text(self.solver, "predictor solver")
        _text(self.detector, "predictor detector")
        if self.regime not in PREDICTOR_REGIMES:
            raise ConfigurationError(f"predictor regime must be one of {PREDICTOR_REGIMES}")

    @property
    def matrix_key(self) -> MatrixKey:
        return MatrixKey(self.layer, self.target)

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in (
            "model_fingerprint", "layer", "target", "rows", "channels", "solver", "detector", "regime",
        )}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PredictorKey:
        try:
            return cls(**dict(data))
        except TypeError as exc:
            raise ConfigurationError("invalid predictor key fields") from exc


@dataclass(frozen=True)
class OnlineRidgeState:
    """The paper's 2x2 online ridge state for ``ln r = a + b x``.

    ``M0 = s I`` and ``v0 = M0 (ln m, 1)`` (paper: s = 2, m = 32); each verified
    observation with ``z = (1, x)`` adds ``z z^T`` to M and ``z ln r`` to v.
    ``M`` is stored as its three distinct entries ``(m00, m01, m11)``.
    """

    m: tuple[float, float, float]
    v: tuple[float, float]
    observations: int = 0
    revision: int = 0
    anchor_rank: int = RANK_BUCKET
    prior_strength: float = 2.0

    def __post_init__(self) -> None:
        values = tuple(self.m), tuple(self.v)
        if len(values[0]) != 3 or len(values[1]) != 2 or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
            for item in (*values[0], *values[1])
        ):
            raise ConfigurationError("ridge state requires finite M (3 entries) and v (2 entries)")
        object.__setattr__(self, "m", tuple(float(item) for item in values[0]))
        object.__setattr__(self, "v", tuple(float(item) for item in values[1]))
        _positive_int(self.observations, "ridge observations", zero=True)
        _positive_int(self.revision, "ridge revision", zero=True)
        _positive_int(self.anchor_rank, "anchor rank")
        if (isinstance(self.prior_strength, bool) or not isinstance(self.prior_strength, (int, float))
                or not math.isfinite(self.prior_strength) or self.prior_strength <= 0):
            raise ConfigurationError("ridge prior strength must be finite and positive")
        object.__setattr__(self, "prior_strength", float(self.prior_strength))
        m00, m01, m11 = self.m
        if m00 <= 0 or m00 * m11 - m01 * m01 <= 0:
            raise ConfigurationError("ridge M must be symmetric positive definite")

    @classmethod
    def initial(cls, anchor_rank: int = RANK_BUCKET, prior_strength: float = 2.0) -> OnlineRidgeState:
        _positive_int(anchor_rank, "anchor rank")
        prior = float(prior_strength)
        return cls((prior, 0.0, prior), (prior * math.log(anchor_rank), prior), 0, 0, anchor_rank, prior)

    def coefficients(self) -> tuple[float, float]:
        m00, m01, m11 = self.m
        v0, v1 = self.v
        det = m00 * m11 - m01 * m01
        return (m11 * v0 - m01 * v1) / det, (m00 * v1 - m01 * v0) / det

    def log_rank(self, x: float) -> float:
        a, b = self.coefficients()
        return a + b * float(x)

    def predict(self, x: float) -> float:
        """Unrounded ``exp(a + b x)``; ``inf`` when it exceeds float range."""
        value = self.log_rank(x)
        return math.inf if value > 700 else math.exp(value)

    def updated(self, x: float, observed_rank: int | float) -> OnlineRidgeState:
        if (isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                or isinstance(observed_rank, bool) or not isinstance(observed_rank, (int, float))
                or not math.isfinite(observed_rank) or observed_rank <= 0):
            raise ConfigurationError("ridge updates need a finite x and positive observed rank")
        x, label = float(x), math.log(float(observed_rank))
        m00, m01, m11 = self.m
        v0, v1 = self.v
        return OnlineRidgeState(
            (m00 + 1.0, m01 + x, m11 + x * x), (v0 + label, v1 + x * label),
            self.observations + 1, self.revision + 1, self.anchor_rank, self.prior_strength,
        )

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "reelcache.online_ridge_state.v1", "m": list(self.m), "v": list(self.v),
                "observations": self.observations, "revision": self.revision,
                "anchor_rank": self.anchor_rank, "prior_strength": self.prior_strength}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OnlineRidgeState:
        try:
            fields = dict(data)
            if fields.pop("schema") != "reelcache.online_ridge_state.v1":
                raise ConfigurationError("unsupported ridge state schema")
            return cls(m=tuple(fields.pop("m")), v=tuple(fields.pop("v")), **fields)
        except (KeyError, TypeError) as exc:
            raise ConfigurationError("invalid ridge state fields") from exc


@dataclass(frozen=True)
class FitObservation:
    """A verified training observation; never a retained KV matrix.

    ``observation_id`` binds only the learner scope and the source identity, so
    a retried or replayed commit of the same source is idempotent. ``origin``
    (optional, run context, never serialized: not content) names the commit
    context that produced the payload (:func:`reelcache.predictor_registry.observation_origin`):
    a repeated identity from the same origin -- or without origins -- must carry
    the same payload (else an integrity error); from another origin (another
    session or lineage committing identical content, or a re-commit after the
    learners changed) the first applied payload wins and the repeat is skipped.
    """

    key: PredictorKey
    source_id: str
    anchor_energy: float
    target_energy: float
    x: float
    observed_rank: int
    predicted_rank: float | None = None
    prediction_revision: int | None = None
    origin: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, PredictorKey):
            raise ConfigurationError("observation key must be a PredictorKey")
        if self.origin is not None:
            _hex64(self.origin, "observation origin")
        _hex64(self.source_id, "observation source id")
        energy = self.anchor_energy
        if isinstance(energy, bool) or not isinstance(energy, (int, float)) or not 0 <= energy < 1:
            raise ConfigurationError("anchor energy must be in [0, 1)")
        target = _fraction(self.target_energy, "observation target")
        if target >= 1 or energy >= target:
            raise ConfigurationError("observations come from below-target anchors and sub-unit targets")
        x = self.x
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
            raise ConfigurationError("observation x must be finite and positive")
        _positive_int(self.observed_rank, "observed rank")
        if self.predicted_rank is not None and (
            isinstance(self.predicted_rank, bool) or not isinstance(self.predicted_rank, (int, float))
            or math.isnan(self.predicted_rank) or self.predicted_rank <= 0
        ):
            raise ConfigurationError("predicted rank must be positive or None")
        if self.prediction_revision is not None:
            _positive_int(self.prediction_revision, "prediction revision", zero=True)
        object.__setattr__(self, "anchor_energy", float(energy))
        object.__setattr__(self, "target_energy", target)
        object.__setattr__(self, "x", float(x))

    @property
    def observation_id(self) -> str:
        return canonical_digest({"key": self.key.to_dict(), "source_id": self.source_id})

    @property
    def payload_digest(self) -> str:
        return canonical_digest({"anchor_energy": self.anchor_energy, "target_energy": self.target_energy,
                                 "x": self.x, "observed_rank": self.observed_rank})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "reelcache.fit_observation.v1", "key": self.key.to_dict(), "source_id": self.source_id,
            "anchor_energy": self.anchor_energy, "target_energy": self.target_energy, "x": self.x,
            "observed_rank": self.observed_rank,
            "predicted_rank": None if self.predicted_rank is None or not math.isfinite(self.predicted_rank)
            else self.predicted_rank,
            "prediction_revision": self.prediction_revision,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FitObservation:
        try:
            fields = dict(data)
            if fields.pop("schema") != "reelcache.fit_observation.v1":
                raise ConfigurationError("unsupported fit observation schema")
            return cls(key=PredictorKey.from_dict(fields.pop("key")), **fields)
        except (KeyError, TypeError) as exc:
            raise ConfigurationError("invalid fit observation fields") from exc
