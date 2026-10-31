"""Model profiles of the shared causal-Wan bridge: data, not code paths (report section 3.6).

CausVid, LongLive 1.0, RAVEN and LongLive 2.0 run on SGLang's
``CausalWanSelfAttention``; their differences (checkpoint binding, chunking,
visibility, RoPE convention, lifecycle) are recorded here. The bridge ships
the LongLive 2.0 profile; out-of-tree profiles (the Step 12 profile package's
``bridge_profile_fields``; core never references that package) are
registered with :func:`register_profile` or passed inline as
``ReelCacheRuntimeConfig.profile_fields``.

Matching is explicit. A profile matches an engine model only by **exact model
path** (a registered path, the same path normalized, or a Hugging Face cache
snapshot ``.../models--org--repo/snapshots/<rev>`` of a registered repo id at
the bound revision) and, when ``transformers`` is not empty, the transformer
class. Several profiles may share a transformer class (the three 1.3B
profiles share ``ReelCacheCausalWanTransformer3DModel``), so a class alone never
selects one.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from ...errors import ConfigurationError, UnsupportedError

VISIBILITY_POLICIES = ("full_history", "sink_window", "global_sink_pinned_window")
# Spellings used by profile packages for the same native semantics.
VISIBILITY_ALIASES = {"sink_rolling_window": "sink_window"}
ROPE_CONVENTIONS = ("sglang_causal_wan",)
RECACHE_POLICIES = ("none", "kv_recache_on_prompt_switch", "optional_scene_cut_clean_recache")
DTYPES = ("bfloat16", "float16", "float32")
_HF_SNAPSHOT = re.compile(r"(?:^|/)models--([^/]+?)--([^/]+)/snapshots/([0-9a-f]{40})/?$")


def normalize_model_path(path: Any) -> dict[str, Any]:
    """Identity of a model path: the normalized path and, for an HF cache snapshot, its repo id and revision."""
    text = str(path or "").strip()
    normalized = os.path.normpath(text) if text else ""
    match = _HF_SNAPSHOT.search(normalized)
    if match:
        return {"path": normalized, "repo": f"{match.group(1)}/{match.group(2)}", "revision": match.group(3)}
    return {"path": normalized, "repo": None, "revision": None}


@dataclass(frozen=True)
class ModelProfile:
    """Per-model configuration of the shared causal-Wan bridge.

    ``visibility`` names the native cache policy; the bridge never re-derives
    slot placement from it (it replays SGLang's own cache logic) but checks the
    engine's cache geometry against ``sink_frames``/``window_frames``
    (``window_frames`` is the KV buffer length in latent frames, sink
    included; ``None`` for a request-sized ``full_history`` buffer).
    ``rope`` is the positional convention used to re-rotate restored pre-RoPE
    keys with the engine's own tables at their recorded positions.
    ``capture_dtypes`` (``{"K": ..., "V": ...}``) names the dtypes of the
    captured pre-RoPE K and V when they differ from the cache's
    ``native_dtype`` (RAVEN: FP32 keys at the attention input, BF16 values and
    cache); restore rotates keys in their capture dtype and casts on store, as
    the engine does. ``matmul_precision`` declares the rank method's matmul
    policy the model's generation needs (``fp32_verified_fp64`` when it runs
    TF32 matmuls; see ``ReelCacheRuntimeConfig.matmul_precision``).
    """

    name: str
    backbone: str
    transformers: tuple[str, ...]
    model_paths: tuple[str, ...]
    checkpoint: Mapping[str, Any]
    num_layers: int
    num_heads: int
    head_dim: int
    chunk_frames: int
    visibility: str
    sink_frames: int
    window_frames: int | None
    energy_profile: str
    native_dtype: str = "bfloat16"
    rope: str = "sglang_causal_wan"
    cfg_streams: tuple[str, ...] = ("cond",)
    recache: str = "none"
    lifecycle: tuple[str, ...] = ()
    defaults: Mapping[str, Any] = field(default_factory=dict)
    capture_dtypes: Mapping[str, str] | None = None
    matmul_precision: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "visibility", VISIBILITY_ALIASES.get(self.visibility, self.visibility))
        for name in ("transformers", "model_paths", "cfg_streams", "lifecycle"):
            value = getattr(self, name)
            if isinstance(value, str) or not isinstance(value, (tuple, list)):
                raise ConfigurationError(f"{name} must be a sequence of strings")
            object.__setattr__(self, name, tuple(value))
            if any(not isinstance(item, str) or not item for item in getattr(self, name)):
                raise ConfigurationError(f"{name} must hold nonempty strings")
        if not isinstance(self.name, str) or not self.name or "/" in self.name:
            raise ConfigurationError("profile name must be a nonempty string without '/'")
        if self.visibility not in VISIBILITY_POLICIES:
            raise ConfigurationError(f"visibility must be one of {VISIBILITY_POLICIES} (or an alias "
                                     f"{sorted(VISIBILITY_ALIASES)})")
        if self.rope not in ROPE_CONVENTIONS:
            raise ConfigurationError(f"rope must be one of {ROPE_CONVENTIONS}")
        if self.recache not in RECACHE_POLICIES:
            raise ConfigurationError(f"recache must be one of {RECACHE_POLICIES}")
        if self.native_dtype not in DTYPES:
            raise ConfigurationError(f"native_dtype must be one of {DTYPES}")
        if self.capture_dtypes is not None:
            if (not isinstance(self.capture_dtypes, Mapping) or set(self.capture_dtypes) != {"K", "V"}
                    or any(value not in DTYPES for value in self.capture_dtypes.values())):
                raise ConfigurationError(f"capture_dtypes maps exactly K and V to one of {DTYPES}")
            object.__setattr__(self, "capture_dtypes", MappingProxyType(dict(self.capture_dtypes)))
        if self.matmul_precision not in (None, "fp32", "fp32_verified_fp64"):
            raise ConfigurationError("matmul_precision must be None, 'fp32' or 'fp32_verified_fp64'")
        if not self.model_paths:
            raise ConfigurationError("a profile binds at least one exact model path")
        for name in ("num_layers", "num_heads", "head_dim", "chunk_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ConfigurationError(f"{name} must be a positive integer")
        if type(self.sink_frames) is not int or self.sink_frames < 0:
            raise ConfigurationError("sink_frames must be a nonnegative integer")
        if self.window_frames is None:
            if self.visibility != "full_history" or self.sink_frames:
                raise ConfigurationError("window_frames=None (a request-sized buffer) is the full_history policy "
                                         "without a sink")
        elif type(self.window_frames) is not int or self.window_frames < 1:
            raise ConfigurationError("window_frames must be a positive integer or None")
        elif self.window_frames <= self.sink_frames:
            raise ConfigurationError("the KV buffer must be longer than the sink")
        if not self.cfg_streams or set(self.cfg_streams) - {"cond", "uncond"}:
            raise ConfigurationError("cfg_streams are 'cond' and optionally 'uncond'")
        object.__setattr__(self, "checkpoint", MappingProxyType(dict(self.checkpoint)))
        object.__setattr__(self, "defaults", MappingProxyType(dict(self.defaults)))

    @property
    def channels(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def max_recache_ranges(self) -> int:
        """Upper bound on the chunk ranges one recache node archives: a prompt-switch recache rewrites at most the
        KV buffer minus the protected sink, in whole chunks (0 without prompt-switch recache or for a
        request-sized buffer, where the bound depends on the request)."""
        if self.recache != "kv_recache_on_prompt_switch" or self.window_frames is None:
            return 0
        return -(-(self.window_frames - self.sink_frames) // self.chunk_frames)

    def default_grid(self) -> tuple[int, int] | None:
        """Post-patch ``(height, width)`` token grid of the profile's default resolution, when its defaults
        name ``height``, ``width`` and ``tokens_per_frame`` with one integer square stride; else None."""
        height, width, tokens = (self.defaults.get(key) for key in ("height", "width", "tokens_per_frame"))
        if not all(type(value) is int and value > 0 for value in (height, width, tokens)):
            return None
        area = height * width
        if area % tokens:
            return None
        stride = int(round((area // tokens) ** 0.5))
        if stride < 1 or stride * stride * tokens != area or height % stride or width % stride:
            return None
        return height // stride, width // stride

    def capture_dtype(self, target: str) -> str:
        """Dtype of the captured ``K`` or ``V`` (the cache's ``native_dtype`` unless ``capture_dtypes`` says)."""
        return self.native_dtype if self.capture_dtypes is None else self.capture_dtypes[target]

    # -- matching ---------------------------------------------------------------------------
    def matches_path(self, model_path: Any) -> bool:
        """Exact path identity (see the module docstring); never a suffix or substring test."""
        target = normalize_model_path(model_path)
        if not target["path"]:
            return False
        for item in self.model_paths:
            if target["path"] == normalize_model_path(item)["path"]:
                return True
            if target["repo"] is not None and target["repo"] == item:
                revision = self.checkpoint.get("revision")
                return revision is None or revision == target["revision"]
        return False

    def matches_architecture(self, model: Mapping[str, Any]) -> bool:
        return not self.transformers or model.get("transformer") in self.transformers

    def matches(self, model: Mapping[str, Any]) -> bool:
        return self.matches_architecture(model) and self.matches_path(model.get("model_path"))

    def check_request(self, request: Any) -> None:
        """Reject engine geometry that differs from the profile (explicit, never adapted)."""
        problems = []
        if request.num_layers != self.num_layers:
            problems.append(f"layers {request.num_layers} != {self.num_layers}")
        if request.num_heads != self.num_heads or request.head_dim != self.head_dim:
            problems.append(f"heads {request.num_heads}x{request.head_dim} != {self.num_heads}x{self.head_dim}")
        if request.dtype != self.native_dtype:
            problems.append(f"cache dtype {request.dtype} != {self.native_dtype}")
        if any(size != self.chunk_frames for size in request.block_sizes):
            problems.append(f"block sizes {request.block_sizes} differ from {self.chunk_frames}-frame chunks")
        model = request.model
        frame = request.tokens_per_frame
        cache_tokens = model.get("cache_tokens")
        if self.window_frames is None:
            needed = (request.pos_start_base + sum(request.block_sizes)) * frame
            if not isinstance(cache_tokens, int) or cache_tokens < needed:
                problems.append(f"full-history cache tokens {cache_tokens} < the request's {needed}")
            if model.get("attention_window_tokens") not in (None, cache_tokens):
                problems.append("a full-history cache attends to its whole buffer")
        elif cache_tokens != self.window_frames * frame:
            problems.append(f"cache tokens {cache_tokens} != {self.window_frames} frames")
        if model.get("sink_tokens") != self.sink_frames * frame:
            problems.append(f"sink tokens {model.get('sink_tokens')} != {self.sink_frames} frames")
        if model.get("global_sink_tokens") not in (0, self.sink_frames * frame):
            problems.append(f"global sink tokens {model.get('global_sink_tokens')} unexpected")
        if not request.streams or set(request.streams) - set(self.cfg_streams):
            problems.append(f"CFG streams {request.streams} are not the profile's {self.cfg_streams}")
        if problems:
            raise UnsupportedError(f"request does not match the {self.name} profile: " + "; ".join(problems))

    # -- plain data -----------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "reelcache.sglang.model_profile.v2", "name": self.name, "backbone": self.backbone,
            "transformers": list(self.transformers), "model_paths": list(self.model_paths),
            "checkpoint": dict(self.checkpoint), "num_layers": self.num_layers, "num_heads": self.num_heads,
            "head_dim": self.head_dim, "chunk_frames": self.chunk_frames, "visibility": self.visibility,
            "sink_frames": self.sink_frames, "window_frames": self.window_frames,
            "energy_profile": self.energy_profile, "native_dtype": self.native_dtype, "rope": self.rope,
            "cfg_streams": list(self.cfg_streams), "recache": self.recache, "lifecycle": list(self.lifecycle),
            "defaults": dict(self.defaults),
            # Optional fields appear only when set: existing profiles keep their fingerprints (archive roots).
            **({} if self.capture_dtypes is None else {"capture_dtypes": dict(self.capture_dtypes)}),
            **({} if self.matmul_precision is None else {"matmul_precision": self.matmul_precision}),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelProfile:
        """Build from plain fields (``to_dict`` output or Step 12's ``bridge_profile_fields``)."""
        if not isinstance(data, Mapping):
            raise ConfigurationError("profile fields must be a mapping")
        values = dict(data)
        schema = values.pop("schema", None)
        if schema not in (None, "reelcache.sglang.model_profile.v1", "reelcache.sglang.model_profile.v2"):
            raise ConfigurationError(f"unsupported profile schema {schema!r}")
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - known)
        if unknown:
            raise ConfigurationError(f"unknown profile fields: {unknown}")
        return cls(**values)


LONGLIVE2 = ModelProfile(
    name="longlive2",
    backbone="wan2.2-ti2v-5b-causal",
    transformers=("LongLive2Transformer3DModel",),
    model_paths=("Rabinovich/LongLive-2.0-5B-Diffusers",),
    checkpoint={
        "repo": "Rabinovich/LongLive-2.0-5B-Diffusers",
        "revision": "37664a64cfc38cdc5fabe88576961ef10646e4f1",
        "transformer_sha256": "17c1c2fc0cc8333c94c128f9ad868a71d89815707a2db5e35ce490d977a6a293",
        "source": "NVlabs/LongLive LongLive 2.0 (Wan2.2-TI2V-5B causal) in diffusers layout",
    },
    num_layers=30,
    num_heads=24,
    head_dim=128,
    chunk_frames=8,
    visibility="global_sink_pinned_window",
    sink_frames=8,
    window_frames=32,
    energy_profile="longlive2",
    cfg_streams=("cond", "uncond"),
    recache="optional_scene_cut_clean_recache",
    lifecycle=(
        "global sink = first chunk; multi_shot_sink pins each scene-cut chunk (visibility only)",
        "multi-shot RoPE offsets: rope_start_frame = start_frame + shot_index * multi_shot_rope_offset",
        "one KV version per frame: no recache epochs (shot_clean_recache off in SGLang)",
        "flow-matching DMD sampler; per-step noise from the request generator (restored with each checkpoint)",
    ),
    defaults={"height": 704, "width": 1280, "tokens_per_frame": 880, "num_inference_steps": 4},
)

_LOCK = threading.Lock()
_PROFILES: dict[str, ModelProfile] = {LONGLIVE2.name: LONGLIVE2}
PROFILES: Mapping[str, ModelProfile] = MappingProxyType(_PROFILES)


def register_profile(profile: ModelProfile | Mapping[str, Any], *, replace: bool = False) -> ModelProfile:
    """Register an out-of-tree profile (process-wide; call before the worker builds the bridge)."""
    if not isinstance(profile, ModelProfile):
        profile = ModelProfile.from_dict(profile)
    with _LOCK:
        existing = _PROFILES.get(profile.name)
        if existing is not None and existing != profile and not replace:
            raise ConfigurationError(f"a different profile named {profile.name!r} is already registered")
        clashes = [other.name for other in _PROFILES.values() if other.name != profile.name and any(
            other.matches_path(path) for path in profile.model_paths)]
        if clashes:
            raise ConfigurationError(f"profile {profile.name!r} binds a model path of {clashes}")
        _PROFILES[profile.name] = profile
    return profile


def profile_for(model: Mapping[str, Any], name: str | None = None, *,
                fields: Mapping[str, Any] | None = None) -> ModelProfile:
    """The configured profile (inline ``fields``, or a registered ``name``), else the one matching the model.

    An explicit profile must match the engine architecture and, for path-bound
    profiles without a transformer class, the exact model path; automatic
    selection always needs the exact model path.
    """
    if fields is not None:
        profile = ModelProfile.from_dict(fields)
    elif name is not None:
        try:
            profile = PROFILES[name]
        except KeyError:
            raise ConfigurationError(f"no shared causal-Wan profile named {name!r}; known: {sorted(PROFILES)}") from None
    else:
        matches = [profile for profile in PROFILES.values() if profile.matches(model)]
        if len(matches) != 1:
            raise UnsupportedError(
                f"no unique shared causal-Wan profile for {dict(model)!r} by exact model path "
                f"(registered: {sorted(PROFILES)}; name one with ReelCacheRuntimeConfig.profile)"
            )
        return matches[0]
    if not profile.matches_architecture(model) or (not profile.transformers and not profile.matches_path(
            model.get("model_path"))):
        raise UnsupportedError(f"profile {profile.name!r} does not match the engine model {dict(model)!r}")
    return profile


__all__ = ["LONGLIVE2", "ModelProfile", "PROFILES", "VISIBILITY_ALIASES", "normalize_model_path", "profile_for",
           "register_profile"]
