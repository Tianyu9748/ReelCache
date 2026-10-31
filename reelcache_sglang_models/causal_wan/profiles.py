"""Causal-Wan backbone profiles (CausVid, LongLive 1.0, RAVEN) as data on SGLang's shared causal-Wan path.

Report section 3.6: one shared causal-Wan integration; per-model differences are ``ModelProfile`` data
(checkpoint/conversion binding, chunk size, sampler and timesteps, visibility policy, RoPE offsets,
recache policy, CFG and native dtypes). This module is pure Python: it imports neither torch nor SGLang,
so profiles can be inspected, fingerprinted and validated anywhere. Step 11's typed ``ModelProfile``
(reelcache.integrations.sglang) is expected to consume ``ModelProfile.to_dict()`` (see the Step 12 README).

File bindings carry the SHA256 recorded by feasibility item 9 (job 24334381) for the native checkpoints
and by the Step 12 binding receipt (``step12_profiles/conversion/diffusers_binding.json``) for the
Wan2.1-T2V-1.3B-Diffusers components SGLang loads (text encoder, tokenizer, VAE). RAVEN binds
``raven_model.pt``; the CM-GRPO variants in the same folder are deliberately not bound.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Mapping

PROFILE_SCHEMA = "reelcache.causal_wan.model_profile.v1"
PROFILE_REVISION = "s12v1"

WAN21_T2V_1_3B = MappingProxyType({
    "family": "Wan2.1-T2V-1.3B causal",
    "num_layers": 30, "num_attention_heads": 12, "attention_head_dim": 128, "hidden_size": 1536,
    "ffn_dim": 8960, "text_dim": 4096, "freq_dim": 256, "in_channels": 16, "out_channels": 16,
    "patch_size": (1, 2, 2), "qk_norm": "rms_norm_across_heads", "cross_attn_norm": True, "eps": 1e-6,
    "text_len": 512, "rope_theta": 10000.0, "rope_max_frames": 1024,
})
GEOMETRY_480P = MappingProxyType({
    "height": 480, "width": 832, "vae_stride": (4, 8, 8), "latent_channels": 16, "latent_height": 60,
    "latent_width": 104, "token_height": 30, "token_width": 52, "tokens_per_latent_frame": 1560, "fps": 16,
})

#: Where the checkpoints live. ``MODEL_ROOT`` is a placeholder: point it at your model directory (the layout below and
#: in README section 4.5). The bindings record path, size and SHA-256, so profile identities follow these paths.
MODEL_ROOT = "/models"
HF_HUB_CACHE = f"{MODEL_ROOT}/huggingface/hub"
WAN_BASE_DIR = f"{MODEL_ROOT}/Wan2.1-T2V-1.3B"
RAVEN_WEIGHTS_DIR = f"{MODEL_ROOT}/raven_weights"
DIFFUSERS_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DIFFUSERS_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
DIFFUSERS_SNAPSHOT = f"{HF_HUB_CACHE}/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/{DIFFUSERS_REVISION}"
PROFILE_MODEL_ROOT = f"{MODEL_ROOT}/reelcache_sglang_profiles"

VISIBILITY_POLICIES = ("full_history", "window_21_frames", "sink_rolling_window")
SAMPLER_KINDS = ("dmd_flow_x0", "raven_consistency_v_lerp")
RECACHE_TRIGGERS = ("none", "prompt_switch")
STEP_DRAWS = ("between_steps", "every_step")
STEP_GENERATORS = ("request", "raven_fork")
DIT_COMPUTE_POLICIES = ("bf16", "self_forcing_bf16", "wan_fp32_regions")


def _plain(value):
    """dataclass / Mapping / tuple tree -> JSON-ready dicts and lists (MappingProxyType-safe asdict)."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


@dataclass(frozen=True)
class FileBinding:
    """One exact artifact: absolute path, SHA256 and size in bytes."""
    role: str
    path: str
    sha256: str
    bytes: int
    state_dict_key: str | None = None
    key_prefix: str | None = None
    note: str = ""

    def __post_init__(self):
        if not self.path.startswith("/"):
            raise ValueError(f"binding path must be absolute: {self.path}")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError(f"bad sha256 for {self.role}: {self.sha256}")
        if self.bytes <= 0:
            raise ValueError(f"bad size for {self.role}")


@dataclass(frozen=True)
class LoRASpec:
    rank: int
    alpha: float
    state_dict_key: str
    key_prefix: str
    target: str
    native_application: str
    sglang_application: str

    @property
    def scale(self) -> float:
        return self.alpha / self.rank


@dataclass(frozen=True)
class ConversionSpec:
    kind: str
    source_roles: tuple[str, ...]
    output_dtype: str
    transformer_class: str
    expected_tensors: int = 825
    lora: LoRASpec | None = None


@dataclass(frozen=True)
class NoiseSpec:
    initial_layout: str
    initial_dtype: str
    step_layout: str
    step_dtype: str
    step_generator: str
    step_draws: str
    native_source: str

    def __post_init__(self):
        if self.step_generator not in STEP_GENERATORS or self.step_draws not in STEP_DRAWS:
            raise ValueError("bad noise recipe")
        for layout in (self.initial_layout, self.step_layout):
            if sorted(layout.split(",")) != sorted("B,C,T,H,W".split(",")):
                raise ValueError(f"bad layout {layout}")


@dataclass(frozen=True)
class SamplerSpec:
    kind: str
    denoising_steps: tuple[int, ...]
    warp_denoising_step: bool
    timestep_shift: float
    num_train_timesteps: int
    resolved_timesteps: tuple[float, ...]
    scheduler: str
    context_timestep: int
    latent_state_dtype: str
    noise: NoiseSpec

    def __post_init__(self):
        if self.kind not in SAMPLER_KINDS:
            raise ValueError(self.kind)
        if len(self.resolved_timesteps) != len(self.denoising_steps):
            raise ValueError("one resolved timestep per denoising step")
        if list(self.resolved_timesteps) != sorted(self.resolved_timesteps, reverse=True):
            raise ValueError("timesteps must decrease")


@dataclass(frozen=True)
class VisibilityPolicy:
    name: str
    window_latent_frames: int | None
    sink_latent_frames: int
    pinned_sink: bool = False
    extrapolated_after_latent_frames: int | None = None
    description: str = ""

    def __post_init__(self):
        if self.name not in VISIBILITY_POLICIES:
            raise ValueError(f"unknown visibility policy {self.name}")
        if self.sink_latent_frames < 0:
            raise ValueError("negative sink")
        if self.window_latent_frames is not None and self.window_latent_frames <= self.sink_latent_frames:
            raise ValueError("window must exceed the sink")

    def cache_latent_frames(self, total_latent_frames: int) -> int:
        """Native KV buffer length (latent frames) for a request of ``total_latent_frames``."""
        if total_latent_frames <= 0:
            raise ValueError("total_latent_frames must be positive")
        if self.window_latent_frames is None:
            return total_latent_frames
        return self.window_latent_frames

    def attended_latent_frames(self, chunk_index: int, chunk_frames: int) -> tuple[tuple[int, int], ...]:
        """Global latent-frame ranges [start, end) chunk ``chunk_index`` attends to, its own frames included."""
        end = (chunk_index + 1) * chunk_frames
        if self.window_latent_frames is None or end <= self.window_latent_frames:
            return ((0, end),)
        start = end - (self.window_latent_frames - self.sink_latent_frames)
        if self.sink_latent_frames == 0:
            return ((start, end),)
        return ((0, self.sink_latent_frames), (start, end))

    def is_extrapolated(self, chunk_index: int, chunk_frames: int) -> bool:
        if self.extrapolated_after_latent_frames is None:
            return False
        return (chunk_index + 1) * chunk_frames > self.extrapolated_after_latent_frames


@dataclass(frozen=True)
class RecachePolicy:
    trigger: str
    recache_latent_frames: int = 0
    global_sink: bool = False
    reset_cross_attention: bool = False
    context_timestep: int = 0
    unmasked: bool = False
    upstream: str = ""
    description: str = ""

    def __post_init__(self):
        if self.trigger not in RECACHE_TRIGGERS:
            raise ValueError(self.trigger)
        if self.trigger == "prompt_switch" and self.recache_latent_frames <= 0:
            raise ValueError("prompt-switch recache needs a positive frame count")


@dataclass(frozen=True)
class RopeSpec:
    temporal_positions: str
    first_chunk_latent_frames: int
    theta: float
    native_cache_key_domain: str
    capture_key_domain: str = "pre_rope_normalized_k_at_attention_input"
    note: str = ""


@dataclass(frozen=True)
class DTypeSpec:
    native_dit_weights: str
    native_dit_compute: str
    native_text_encoder: str
    native_vae: str
    native_kv_cache_k: str
    native_kv_cache_v: str
    sglang_dit_precision: str
    sglang_text_encoder_precision: str
    sglang_vae_precision: str
    sglang_vae_decode_precision: str
    sglang_kv_cache: str
    # DiT precision policy applied by the pipeline (sglang_dit.DIT_PRECISION_POLICIES): "bf16" = SGLang's causal-Wan
    # block math unchanged (fp32 modulation/norm kernels, bf16 residual stream); "self_forcing_bf16" = the native
    # Self-Forcing lineage math (CausVid, LongLive 1.0: pipe.to(bf16), upstream fp32 autocast regions commented out,
    # every modulation/norm/residual op in bf16); "wan_fp32_regions" = original Wan2.1 mixed precision (fp32 weights
    # and residual stream, fp32 time embedding, modulation and head, bf16-autocast matmuls), which RAVEN runs.
    sglang_dit_compute: str = "bf16"
    # TF32 for the fp32 matmuls of the fp32 regions (mirrors the native harness' allow_tf32 setting). Set process-wide
    # around those GEMMs only and restored (sglang_dit.fp32_region: no flag-free equivalent, thread-safety assumption);
    # the archive then plans under matmul_precision="fp32_verified_fp64" (archive_matmul_precision).
    sglang_fp32_regions_tf32: bool = False

    def __post_init__(self):
        if self.sglang_dit_compute not in DIT_COMPUTE_POLICIES:
            raise ValueError(f"unknown DiT compute policy {self.sglang_dit_compute!r}")
        need = {"wan_fp32_regions": "fp32", "self_forcing_bf16": "bf16"}.get(self.sglang_dit_compute)
        if need is not None and self.sglang_dit_precision != need:
            raise ValueError(f"{self.sglang_dit_compute} needs {need} DiT weights (sglang_dit_precision={need!r})")
        if self.sglang_fp32_regions_tf32 and self.sglang_dit_compute != "wan_fp32_regions":
            raise ValueError("sglang_fp32_regions_tf32 applies to the wan_fp32_regions policy only")


@dataclass(frozen=True)
class CFGSpec:
    guidance_scale: float
    negative_branch: bool
    note: str = ""


@dataclass(frozen=True)
class NativeReference:
    repository: str
    revision: str
    entry_points: tuple[str, ...]
    harness: str
    attention: str


@dataclass(frozen=True)
class SGLangBinding:
    pipeline_name: str
    model_dir_name: str
    # SGLang's CausalWanTransformer3DModel + a cross-attention cache fix (sglang_dit.py), own architecture name.
    dit_class: str = "ReelCacheCausalWanTransformer3DModel"
    components_from: str = f"{DIFFUSERS_REPO}@{DIFFUSERS_REVISION}"

    @property
    def model_path(self) -> str:
        return f"{PROFILE_MODEL_ROOT}/{self.model_dir_name}"


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    display_name: str
    backbone: Mapping
    geometry: Mapping
    chunk_latent_frames: int
    files: tuple[FileBinding, ...]
    components: tuple[FileBinding, ...]
    conversion: ConversionSpec
    sampler: SamplerSpec
    default_visibility: str
    visibility_policies: tuple[VisibilityPolicy, ...]
    recache: RecachePolicy
    rope: RopeSpec
    cfg: CFGSpec
    dtypes: DTypeSpec
    native_reference: NativeReference
    sglang: SGLangBinding
    notes: tuple[str, ...] = ()
    revision: str = PROFILE_REVISION
    schema: str = PROFILE_SCHEMA

    def __post_init__(self):
        if self.chunk_latent_frames <= 0:
            raise ValueError("chunk_latent_frames must be positive")
        names = [p.name for p in self.visibility_policies]
        if len(set(names)) != len(names) or self.default_visibility not in names:
            raise ValueError(f"{self.profile_id}: default visibility {self.default_visibility} not in {names}")
        roles = [f.role for f in self.files]
        missing = [r for r in self.conversion.source_roles if r not in roles]
        if missing:
            raise ValueError(f"{self.profile_id}: conversion needs unbound roles {missing}")
        if len(set(roles)) != len(roles):
            raise ValueError(f"{self.profile_id}: duplicate file roles")
        if self.recache.trigger == "prompt_switch":
            vis = self.visibility(self.default_visibility)
            if vis.window_latent_frames != self.recache.recache_latent_frames:
                raise ValueError("the prompt-switch recache spans exactly the KV buffer (local_attn_size)")
            if self.recache.recache_latent_frames % self.chunk_latent_frames:
                raise ValueError("recache span must be whole chunks")

    def visibility(self, name: str | None = None) -> VisibilityPolicy:
        name = name or self.default_visibility
        for p in self.visibility_policies:
            if p.name == name:
                return p
        raise KeyError(f"{self.profile_id} has no visibility policy {name!r}; allowed: "
                       f"{[p.name for p in self.visibility_policies]}")

    def file(self, role: str) -> FileBinding:
        hits = [f for f in (*self.files, *self.components) if f.role == role]
        if len(hits) != 1:
            raise KeyError(f"{self.profile_id}: role {role!r} bound {len(hits)} times")
        return hits[0]

    def to_dict(self) -> dict:
        out = _plain(self)
        out["sglang"]["model_path"] = self.sglang.model_path
        if self.conversion.lora is not None:
            out["conversion"]["lora"]["scale"] = self.conversion.lora.scale
        return out

    def fingerprint(self) -> str:
        """Identity of the whole profile (sampler, visibility, recache, dtypes, bindings)."""
        return _digest(self.to_dict())

    def binding_fingerprint(self) -> str:
        """Identity of what a profile model directory contains: sources, conversion, components, backbone, path."""
        d = self.to_dict()
        keep = ("role", "path", "sha256", "bytes", "state_dict_key", "key_prefix")
        conv = {k: v for k, v in d["conversion"].items() if k != "lora"}
        if d["conversion"]["lora"] is not None:
            conv["lora"] = {k: d["conversion"]["lora"][k] for k in ("rank", "alpha", "state_dict_key", "key_prefix")}
        return _digest({"profile_id": d["profile_id"], "backbone": d["backbone"], "conversion": conv,
                        "files": [{k: f[k] for k in keep} for f in d["files"]],
                        "components": [{k: f[k] for k in keep} for f in d["components"]],
                        "sglang": d["sglang"]})


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ bindings
_T5_SHA256 = "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d"
_VAE_SHA256 = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
_TOKENIZER_NATIVE = (
    ("google/umt5-xxl/special_tokens_map.json", "7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d", 6623),
    ("google/umt5-xxl/spiece.model", "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458", 4548313),
    ("google/umt5-xxl/tokenizer_config.json", "ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05", 61728),
    ("google/umt5-xxl/tokenizer.json", "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b", 16837417),
)


def _native_text_vae(root: str) -> tuple[FileBinding, ...]:
    """T5/VAE/tokenizer the original repositories load (identical files under both roots, item 9)."""
    out = [FileBinding("native_text_encoder_t5", f"{root}/models_t5_umt5-xxl-enc-bf16.pth", _T5_SHA256, 11361920418),
           FileBinding("native_vae", f"{root}/Wan2.1_VAE.pth", _VAE_SHA256, 507609880)]
    for i, (rel, sha, size) in enumerate(_TOKENIZER_NATIVE):
        out.append(FileBinding(f"native_tokenizer_{i}", f"{root}/{rel}", sha, size))
    return tuple(out)


_WAN_BASE_CONFIG = FileBinding("base_config", f"{WAN_BASE_DIR}/config.json",
                               "ab37994c43740513f94b3ba6233a784035a67b43c8cde83c8f31aa90468c67ce", 249)
_WAN_BASE_DIT = FileBinding("base_dit_weights", f"{WAN_BASE_DIR}/diffusion_pytorch_model.safetensors",
                            "96b6b242ca1c2f24e9d02cd6596066fab6d310e2d7538f33ae267cb18d957e8f", 5676070424,
                            note="loaded by the native from_pretrained, then fully overwritten by the generator state")

# SGLang-side components (Wan2.1-T2V-1.3B-Diffusers @ 0fad780a, step12_profiles/conversion/diffusers_binding.json).
_D = DIFFUSERS_SNAPSHOT
DIFFUSERS_COMPONENTS: tuple[FileBinding, ...] = (
    FileBinding("diffusers_model_index", f"{_D}/model_index.json",
                "b8b28e022329acf975e027579a1d2dc3de44dba28f4c909b98a880fbf5b0de7d", 400),
    FileBinding("diffusers_scheduler_config", f"{_D}/scheduler/scheduler_config.json",
                "3fed2abbd9bbc301a74db01947198057ec5049808910dccab320925bf27bea6e", 751,
                note="loaded, then replaced by the profile's native scheduler"),
    FileBinding("diffusers_text_encoder_config", f"{_D}/text_encoder/config.json",
                "4087b6192155a4643f6d29fd326c4610103130674011ef4d0a53f8bce5de967d", 854),
    FileBinding("diffusers_text_encoder_0", f"{_D}/text_encoder/model-00001-of-00005.safetensors",
                "c0ef3a140898e228a3520c9adec60743d2e8e5b3d229651bb37f1a3921919f99", 4972389712),
    FileBinding("diffusers_text_encoder_1", f"{_D}/text_encoder/model-00002-of-00005.safetensors",
                "481c7b2b39771c44df6dd8d13ee12ed072d731b4a650bd092885d4d52db229ad", 4899225672),
    FileBinding("diffusers_text_encoder_2", f"{_D}/text_encoder/model-00003-of-00005.safetensors",
                "f93148bcc04052a169e1e49bfcf6125df6cf9bf243cb9c627da75266cf8e35c3", 4966309504),
    FileBinding("diffusers_text_encoder_3", f"{_D}/text_encoder/model-00004-of-00005.safetensors",
                "a451792c739c05bca4606190cc2dd16731411bac03b4cf6aacc5767321f857c9", 4999880704),
    FileBinding("diffusers_text_encoder_4", f"{_D}/text_encoder/model-00005-of-00005.safetensors",
                "7e76e18d224531b8197a46231cb53daf7f2f6ca707130252becf933026ac4eea", 2885866152),
    FileBinding("diffusers_text_encoder_index", f"{_D}/text_encoder/model.safetensors.index.json",
                "8af791f24a6447aa30c95786f40adf23d3c9df97470d2f36159c8f1113f120b4", 22476),
    FileBinding("diffusers_tokenizer_special_tokens", f"{_D}/tokenizer/special_tokens_map.json",
                "456b58fd240a06c743a7c2cf8008bec501240d68ebd1fc4018ea569505fea270", 7079),
    FileBinding("diffusers_tokenizer_spiece", f"{_D}/tokenizer/spiece.model",
                "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458", 4548313),
    FileBinding("diffusers_tokenizer_json", f"{_D}/tokenizer/tokenizer.json",
                "20a46ac256746594ed7e1e3ef733b83fbc5a6f0922aa7480eda961743de080ef", 16837459),
    FileBinding("diffusers_tokenizer_config", f"{_D}/tokenizer/tokenizer_config.json",
                "1d8d2a216bf8e70ac15b7ddcea566c4dd0433c024b39a58ca5e4c66bd78defbd", 61758),
    FileBinding("diffusers_vae_config", f"{_D}/vae/config.json",
                "f0c1cc1d7decb5badc384f54691746a27a9aeff49f7ebca974e583389342d527", 724),
    FileBinding("diffusers_vae", f"{_D}/vae/diffusion_pytorch_model.safetensors",
                "d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793", 507591892),
)
COMPONENT_DIRS = ("text_encoder", "tokenizer", "vae", "scheduler")

_BF16_DIT_DTYPES = dict(native_dit_weights="bfloat16", native_dit_compute="bfloat16 (no autocast)",
                        native_text_encoder="bfloat16", native_vae="bfloat16", native_kv_cache_k="bfloat16",
                        native_kv_cache_v="bfloat16", sglang_dit_precision="bf16",
                        sglang_text_encoder_precision="bf16", sglang_vae_precision="bf16",
                        sglang_vae_decode_precision="bf16", sglang_kv_cache="bfloat16",
                        sglang_dit_compute="self_forcing_bf16")

_DMD_NOISE = NoiseSpec(initial_layout="B,T,C,H,W", initial_dtype="bfloat16", step_layout="B,T,C,H,W",
                       step_dtype="bfloat16", step_generator="request", step_draws="between_steps",
                       native_source="")


# ------------------------------------------------------------------------------------------------ CausVid
CAUSVID = ModelProfile(
    profile_id="causvid",
    display_name="CausVid (Wan2.1-T2V-1.3B + CausVid autoregressive DMD generator)",
    backbone=WAN21_T2V_1_3B,
    geometry=GEOMETRY_480P,
    chunk_latent_frames=3,
    files=(
        FileBinding("generator_checkpoint", f"{MODEL_ROOT}/causvid/autoregressive_checkpoint/model.pt",
                    "8b75a848a525a6ab0b8cefa7f2a3997aefb581eb5824ad60bafb5337e81e521c", 11352649716,
                    state_dict_key="generator", key_prefix="model.",
                    note="fp32 'generator' (825 tensors); the 'critic' entry is not used"),
        _WAN_BASE_CONFIG, _WAN_BASE_DIT, *_native_text_vae(WAN_BASE_DIR),
    ),
    components=DIFFUSERS_COMPONENTS,
    conversion=ConversionSpec(kind="native_wan_to_diffusers", source_roles=("generator_checkpoint",),
                              output_dtype="float32", transformer_class="CausalWanTransformer3DModel"),
    sampler=SamplerSpec(
        kind="dmd_flow_x0", denoising_steps=(1000, 757, 522), warp_denoising_step=False, timestep_shift=8.0,
        num_train_timesteps=1000, resolved_timesteps=(1000.0, 757.0, 522.0),
        scheduler="DiffSynth FlowMatchScheduler(shift=8.0, sigma_min=0.0, extra_one_step=True), set_timesteps(1000, training=True)",
        context_timestep=0, latent_state_dtype="bfloat16",
        noise=NoiseSpec(initial_layout="B,T,C,H,W", initial_dtype="bfloat16", step_layout="B,T,C,H,W",
                        step_dtype="bfloat16", step_generator="request", step_draws="between_steps",
                        native_source="CausVid causal_inference.InferencePipeline.inference (harness copy passes an "
                                      "explicit generator; upstream uses the global CUDA RNG, same stream)")),
    default_visibility="full_history",
    visibility_policies=(
        VisibilityPolicy("full_history", None, 0, extrapolated_after_latent_frames=21,
                         description="every generated frame stays visible (M14, the native adapter); chunks beyond "
                                     "the 21 trained latent frames are labeled extrapolated"),
        VisibilityPolicy("window_21_frames", 21, 0,
                         description="control: sliding window equal to the 21-frame training context (current "
                                     "chunk + 6 previous chunks); bounded memory; its own RAW references"),
    ),
    recache=RecachePolicy(trigger="none"),
    rope=RopeSpec(temporal_positions="absolute_latent_frame (chunk i starts at frame 3*i)", first_chunk_latent_frames=3,
                  theta=10000.0, native_cache_key_domain="post_rope_bfloat16 [1,T,12,128]"),
    cfg=CFGSpec(guidance_scale=1.0, negative_branch=False,
                note="DMD-distilled: no classifier-free guidance at inference (real_guidance_scale is training-only)"),
    dtypes=DTypeSpec(**_BF16_DIT_DTYPES),
    native_reference=NativeReference(
        repository="tianweiy/CausVid (harness copy of causvid/models/wan/causal_inference.py and the Wan model code)",
        revision="ReelCache-Energy36 causvid_energy_vbench_grid_20260924/native (M06 grid harness; source_manifest.json)",
        entry_points=("third_party.causvid.modeling.causal_inference.InferencePipeline",
                      "native_causvid.build_pipeline", "native_causvid.instance_cross_attention_sdpa"),
        harness="step12_profiles/scripts/native_ref.py CausVidNative",
        attention="flash_attn 2 self-attention; torch SDPA cross-attention (M06 binding); TF32 off"),
    sglang=SGLangBinding(pipeline_name="ReelCacheCausVidProfilePipeline", model_dir_name="reelcache-causvid-480p-s12v1"),
    notes=("The native InferencePipeline allocates a 32760-token (21-frame) cache; full_history sizes the cache to the "
           "request instead, as the ReelCache native adapter and M14 do.",),
)

# ------------------------------------------------------------------------------------------------ LongLive 1.0
LONGLIVE1 = ModelProfile(
    profile_id="longlive1",
    display_name="LongLive 1.0 (Wan2.1-T2V-1.3B + LongLive base + LoRA rank 256)",
    backbone=WAN21_T2V_1_3B,
    geometry=GEOMETRY_480P,
    chunk_latent_frames=3,
    files=(
        FileBinding("generator_base_checkpoint",
                    f"{MODEL_ROOT}/LongLive-1.3B/models/longlive_base.pt",
                    "10a2aa8fcf89c77d9033f4c117405412a690e289625766619d293f0c5a208ee7", 5676334208,
                    state_dict_key="generator", key_prefix="model.", note="fp32 'generator' (825 tensors)"),
        FileBinding("lora_rank256", f"{MODEL_ROOT}/LongLive-1.3B/models/lora.pt",
                    "c4e43b87d62d4b0614b496773639f1ab170a7ee486dc23407901e9d3a5ebc07a", 2800056690,
                    state_dict_key="generator_lora", key_prefix="base_model.model.",
                    note="PEFT state dict, 600 fp32 tensors (30 blocks x 10 Linear x A/B); 'critic_lora' unused"),
        _WAN_BASE_CONFIG, _WAN_BASE_DIT, *_native_text_vae(WAN_BASE_DIR),
    ),
    components=DIFFUSERS_COMPONENTS,
    conversion=ConversionSpec(
        kind="native_wan_to_diffusers", source_roles=("generator_base_checkpoint", "lora_rank256"),
        output_dtype="float32", transformer_class="CausalWanTransformer3DModel",
        lora=LoRASpec(rank=256, alpha=256.0, state_dict_key="generator_lora", key_prefix="base_model.model.",
                      target="every nn.Linear of CausalWanAttentionBlock: self_attn.{q,k,v,o}, cross_attn.{q,k,v,o}, ffn.0, ffn.2",
                      native_application="PEFT LoRA, unmerged, bf16 (peft.get_peft_model + set_peft_model_state_dict; "
                                         "pipeline.to(bfloat16))",
                      sglang_application="merged at conversion in float64: W + (alpha/rank) * B @ A, stored fp32, "
                                         "cast to bf16 by SGLang at load")),
    sampler=SamplerSpec(
        kind="dmd_flow_x0", denoising_steps=(1000, 750, 500, 250), warp_denoising_step=True, timestep_shift=5.0,
        num_train_timesteps=1000, resolved_timesteps=(1000.0, 937.5, 833.3333, 625.0),
        scheduler="DiffSynth FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True), set_timesteps(1000, training=True)",
        context_timestep=0, latent_state_dtype="bfloat16",
        noise=NoiseSpec(initial_layout="B,T,C,H,W", initial_dtype="bfloat16", step_layout="B,T,C,H,W",
                        step_dtype="bfloat16", step_generator="request", step_draws="between_steps",
                        native_source="LongLive1.0 interactive_inference.py / inference.py: set_seed then torch.randn "
                                      "and torch.randn_like on the global CUDA RNG (same stream as a request generator)")),
    default_visibility="sink_rolling_window",
    visibility_policies=(
        VisibilityPolicy("sink_rolling_window", 12, 3,
                         description="12-frame KV buffer (local_attn_size) = 3-frame global sink + 9-frame rolling "
                                     "window that includes the current chunk"),
    ),
    recache=RecachePolicy(
        trigger="prompt_switch", recache_latent_frames=12, global_sink=True, reset_cross_attention=True,
        context_timestep=0, unmasked=True,
        upstream=("NVlabs/LongLive@c6a27e771e803b1c12ffa28b82ba69b42472020e LongLive1.0/pipeline/"
                  "interactive_causal_inference.py:34-96; KV write/attention LongLive1.0/wan/modules/causal_model.py:206-347"),
        description=("before the first chunk under the new prompt: reset cross-attention caches; one clean (timestep "
                     "context_noise=0), unmasked forward over the last min(12, generated) frames under the new prompt; "
                     "is_recompute protects the sink (global_sink) so only the frames beside it are written back; cache "
                     "indices are not moved; cross-attention caches reset again")),
    rope=RopeSpec(temporal_positions="absolute_latent_frame (chunk i starts at frame 3*i; the recache keeps the "
                                     "original positions of the recomputed frames)",
                  first_chunk_latent_frames=3, theta=10000.0,
                  native_cache_key_domain="post_rope_bfloat16 [1,12*1560,12,128] ring buffer (sink + rolling window)"),
    cfg=CFGSpec(guidance_scale=1.0, negative_branch=False, note="DMD-distilled; no classifier-free guidance"),
    dtypes=DTypeSpec(**_BF16_DIT_DTYPES),
    native_reference=NativeReference(
        repository="NVlabs/LongLive (LongLive1.0)", revision="c6a27e771e803b1c12ffa28b82ba69b42472020e",
        entry_points=("pipeline.causal_inference.CausalInferencePipeline.inference",
                      "pipeline.interactive_causal_inference.InteractiveCausalInferencePipeline.inference"),
        harness="step12_profiles/scripts/native_ref.py LongLive1Native",
        attention="flash_attn 2 (self and cross)"),
    sglang=SGLangBinding(pipeline_name="ReelCacheLongLive1ProfilePipeline", model_dir_name="reelcache-longlive1-480p-s12v1"),
    notes=("The vendored RAVEN/third_party/longlive copy lacks the interactive (prompt-switch) path; the parity "
           "reference is the upstream commit above.",),
)

# ------------------------------------------------------------------------------------------------ RAVEN
RAVEN = ModelProfile(
    profile_id="raven",
    display_name="RAVEN (Wan2.1-T2V-1.3B causal + raven_model.pt)",
    backbone=WAN21_T2V_1_3B,
    geometry=GEOMETRY_480P,
    chunk_latent_frames=3,
    files=(
        FileBinding("generator_checkpoint", f"{MODEL_ROOT}/RAVEN-model-weights/raven_model.pt",
                    "73e41928df3c7a90bca2b3299e1970ce0fdf60c4d1635e214c1b7af5a982e986", 5676256254,
                    state_dict_key=None, key_prefix="",
                    note="flat fp32 state dict (825 tensors); cmgrpo_raven_{full,merge}.pt / cmgrpo_raven_lora are NOT bound"),
        *_native_text_vae(RAVEN_WEIGHTS_DIR),
    ),
    components=DIFFUSERS_COMPONENTS,
    conversion=ConversionSpec(kind="native_wan_to_diffusers", source_roles=("generator_checkpoint",),
                              output_dtype="float32", transformer_class="CausalWanTransformer3DModel"),
    sampler=SamplerSpec(
        kind="raven_consistency_v_lerp", denoising_steps=(1000, 938, 833, 625), warp_denoising_step=False,
        timestep_shift=5.0, num_train_timesteps=1000, resolved_timesteps=(1000.0, 938.0, 833.0, 625.0),
        scheduler=("RAVEN LinearInterpolationSchedule(T=1000, v_lerp) + TrailingSamplingTimesteps(T=1000 int, 4 steps, "
                   "shift 5) -> integer timesteps round(1000*5t/(1+4t)); ConsistencySampler"),
        context_timestep=0, latent_state_dtype="float32",
        noise=NoiseSpec(initial_layout="B,C,T,H,W", initial_dtype="float32", step_layout="B,C,T,H,W",
                        step_dtype="float32", step_generator="raven_fork", step_draws="every_step",
                        native_source=("RAVEN prepare_inference_inputs: torch.empty(1,16,T,60,104,fp32).normal_(RandomState(seed)); "
                                       "ConsistencySampler.step_to: torch.randn([16,3,60,104]) from RandomState(seed).fork(0) "
                                       "= seed yield_seed(seed), drawn at every step including the last"))),
    default_visibility="full_history",
    visibility_policies=(
        VisibilityPolicy("full_history", None, 0, description="sink 0, window None as shipped (planb_raven_local.jsonc)"),
    ),
    recache=RecachePolicy(trigger="none"),
    rope=RopeSpec(temporal_positions="absolute_latent_frame: frame_shifts = independent_first_chunk + (i-1)*chunk = 3*i "
                                     "(continuation offset 3; the M06 harness used 0, which is invalid)",
                  first_chunk_latent_frames=3, theta=10000.0,
                  native_cache_key_domain="post_rope_float32 packed [T*1560,12,128] (NaiveCache); V bfloat16",
                  note="The native cache keeps fp32 post-RoPE K; attention casts it to bf16. SGLang caches bf16 "
                       "post-RoPE K; ReelCache captures pre-RoPE K at the attention input."),
    cfg=CFGSpec(guidance_scale=1.0, negative_branch=False,
                note="planb_raven_local.jsonc guidance_scale 1.0: the negative caches exist but are never used"),
    dtypes=DTypeSpec(native_dit_weights="float32", native_dit_compute="bfloat16 autocast (time embedding, modulation and "
                                                                     "head in float32)",
                     native_text_encoder="bfloat16", native_vae="float32 weights, bfloat16 autocast",
                     native_kv_cache_k="float32 (post-RoPE)", native_kv_cache_v="bfloat16",
                     sglang_dit_precision="fp32", sglang_text_encoder_precision="bf16", sglang_vae_precision="fp32",
                     sglang_vae_decode_precision="bf16", sglang_kv_cache="bfloat16",
                     sglang_dit_compute="wan_fp32_regions", sglang_fp32_regions_tf32=True),
    native_reference=NativeReference(
        repository="RAVEN (local clone)", revision="3bacebd2a33114986fd5c6e3de1e1053401935ca",
        entry_points=("project.meta_models.causal_wan2_1_t2v.CausalWan2_1_T2V (infer/cache/pred)",
                      "raven_runtime.run_native_rollout (prepost_rope_key_20260928 harness)"),
        harness="step12_profiles/scripts/native_ref.py RavenNative",
        attention="flash_attn 2 forced (no catch-all fallback); TF32 on as in the harness"),
    sglang=SGLangBinding(pipeline_name="ReelCacheRavenProfilePipeline", model_dir_name="reelcache-raven-480p-s12v1"),
    notes=("RAVEN's sampler differs from the DMD stage (integer timesteps, v_lerp x0 in fp32, a separate forked step-noise "
           "generator drawn at every step, fp32 latent state): the profile uses a thin RAVEN stage for it.",
           "RAVEN keeps original Wan2.1 mixed precision (fp32 weights and residual stream; time embedding, modulation and "
           "head in fp32; matmuls under bf16 autocast) while SGLang's causal-Wan block keeps a bf16 residual stream: the "
           "profile loads fp32 weights and applies the wan_fp32_regions DiT policy (sglang_dit.py)."),
)

PROFILES: Mapping[str, ModelProfile] = MappingProxyType({p.profile_id: p for p in (CAUSVID, LONGLIVE1, RAVEN)})

# ------------------------------------------------------------------------------------------------ prompt switches
#: How a profile applies ``prompt_switches`` (workload v2, freeze W1a). Deliberately NOT a ``ModelProfile`` field:
#: ``to_dict()`` / ``fingerprint()`` (and so the bridge profile, its archive root and every node id) stay F6's.
#: ``recache``: LongLive 1.0's native KV recache epoch at the switch. ``cross_attention_only`` (CausVid, RAVEN; user
#: decision 2026-10-19): blocks from the switch chunk on cross-attend to the new prompt; the cross-attention caches are
#: reset at the switch; archived and self-attention KV are never recomputed (no recache). Switch chunks only need to
#: be > 0 (LongLive 1.0's "no switch before chunk 5" is its sink rule, not imposed here).
PROMPT_SWITCH_RECACHE = "recache"
PROMPT_SWITCH_CROSS_ATTENTION = "cross_attention_only"
CROSS_ATTENTION_SWITCH_PROFILES = frozenset({"causvid", "raven"})


def prompt_switch_mode(profile: ModelProfile) -> str | None:
    """``recache`` (LongLive 1.0), ``cross_attention_only`` (CausVid, RAVEN) or None (no prompt switch)."""
    if profile.recache.trigger == "prompt_switch":
        return PROMPT_SWITCH_RECACHE
    if profile.profile_id in CROSS_ATTENTION_SWITCH_PROFILES:
        return PROMPT_SWITCH_CROSS_ATTENTION
    return None


def switch_block_conditioning(profile: ModelProfile, prompt: str, switches, num_blocks: int) -> tuple[str, ...] | None:
    """Per-block conditioning ids of a request (SGLang ``CausalKVRequest.block_conditioning``); ``switches``: the
    request's normalized ``prompt_switches``, or None when the request has no such field.

    LongLive 1.0: always ``lifecycle.block_conditioning_ids`` (``s<segment>:<prompt digest>`` per block). CausVid /
    RAVEN: the same ids exactly when the request carries ``prompt_switches`` (a list, possibly empty: the switch path,
    so a session's restored blocks are validated per block); without the field none, as F6 (the request conditioning
    covers every block; node records and ids unchanged). A session keeps one form: restored blocks generated under the
    other one are refused by the bridge (``same_blocks``). Other profiles: None."""
    mode = prompt_switch_mode(profile)
    if mode is None or (mode == PROMPT_SWITCH_CROSS_ATTENTION and switches is None):
        return None
    from .lifecycle import block_conditioning_ids  # lifecycle imports this module

    return block_conditioning_ids(prompt, list(switches or ()), num_blocks)


def get_profile(profile_id: str) -> ModelProfile:
    try:
        return PROFILES[profile_id]
    except KeyError:
        raise KeyError(f"unknown causal-Wan profile {profile_id!r}; known: {sorted(PROFILES)}") from None


def profile_for_model_path(model_path: str) -> ModelProfile:
    """Exact registered model directory -> profile (no family-name guessing)."""
    hits = [p for p in PROFILES.values() if p.sglang.model_path == model_path.rstrip("/")]
    if len(hits) != 1:
        raise KeyError(f"{model_path!r} is not an exact causal-Wan profile path")
    return hits[0]


def sglang_capture_dtypes(profile: ModelProfile) -> dict | None:
    """Dtypes of the normalized pre-RoPE K and the V at SGLang's attention input, when they differ from the cache.

    ``wan_fp32_regions`` (RAVEN) keeps Wan2.1's fp32 weights: the q/k RMSNorm multiplies the bf16 projection by its
    fp32 weight, so the pre-RoPE key is float32 while V and the post-RoPE cache are bfloat16 (the engine rotates the
    fp32 key, then casts with ``.type_as(v)``). The other policies capture both in the cache dtype (None)."""
    if profile.dtypes.sglang_dit_compute == "wan_fp32_regions":
        return {"K": "float32", "V": profile.dtypes.sglang_kv_cache}
    return None


def archive_matmul_precision(profile: ModelProfile) -> str | None:
    """Rank-method matmul policy the archive needs: generation that enables TF32 (RAVEN's fp32 regions, toggled by
    the profile only inside those regions) flips the process-global flags while the fitter plans on another thread,
    so planning must not depend on them (Step 7 ``fp32_verified_fp64``: every acceptance measured in FP64).

    This is RAVEN's TF32 policy together with :func:`sglang_dit.fp32_region`: the flag is open only around three
    GEMM sites per forward, on the forward thread; the bridge restores on that thread between forwards, and
    restore_kernels' production paths and FP64 reference ignore the flag. Other code that needs IEEE FP32 cuBLAS
    products on another thread of the worker (restore_kernels' staged fp32 backend or FP32 reference, planning with
    ``matmul_precision="fp32"``) must not run while a RAVEN forward is in flight. With TF32 directions the archived
    bases of a RAVEN session may differ between runs (acceptance does not: FP64)."""
    return "fp32_verified_fp64" if profile.dtypes.sglang_fp32_regions_tf32 else None


def bridge_profile_fields(profile: ModelProfile, visibility: str | None = None, *, extended: bool = False) -> dict:
    """Plain-data view for Step 11's ``reelcache.integrations.sglang.profiles.ModelProfile`` (phase 2 input).

    The three 1.3B profiles share ``CausalWanTransformer3DModel``, so they must be matched by exact model path
    (``transformers`` is empty). ``full_history`` caches are sized by the request, so ``window_frames`` is None
    there. ``extended`` adds the fields of bridges that accept them (Step 11 CR-12p2-2/3: ``capture_dtypes``,
    ``matmul_precision``) when the profile needs them (RAVEN); older bridges reject unknown fields.
    """
    vis = profile.visibility(visibility)
    bridge_visibility = "full_history" if vis.window_latent_frames is None else "sink_window"
    extra = {}
    if extended:
        capture = sglang_capture_dtypes(profile)
        if capture is not None:
            extra["capture_dtypes"] = capture
        precision = archive_matmul_precision(profile)
        if precision is not None:
            extra["matmul_precision"] = precision
    return extra | {
        "name": profile.profile_id if visibility in (None, profile.default_visibility) else f"{profile.profile_id}:{vis.name}",
        "backbone": "wan2.1-t2v-1.3b-causal", "transformers": (), "model_paths": (profile.sglang.model_path,),
        "checkpoint": {"binding_fingerprint": profile.binding_fingerprint(), "profile_fingerprint": profile.fingerprint(),
                       "sources": {f.role: {"path": f.path, "sha256": f.sha256} for f in profile.files
                                   if f.role in profile.conversion.source_roles},
                       "components": f"{DIFFUSERS_REPO}@{DIFFUSERS_REVISION}"},
        "num_layers": int(profile.backbone["num_layers"]), "num_heads": int(profile.backbone["num_attention_heads"]),
        "head_dim": int(profile.backbone["attention_head_dim"]), "chunk_frames": profile.chunk_latent_frames,
        "visibility": bridge_visibility, "sink_frames": vis.sink_latent_frames, "window_frames": vis.window_latent_frames,
        "energy_profile": profile.profile_id, "native_dtype": profile.dtypes.sglang_kv_cache, "rope": "sglang_causal_wan",
        "cfg_streams": ("cond",),
        "recache": "kv_recache_on_prompt_switch" if profile.recache.trigger == "prompt_switch" else "none",
        "lifecycle": tuple(x for x in (vis.description, profile.recache.description) if x),
        "defaults": {"height": profile.geometry["height"], "width": profile.geometry["width"],
                     "tokens_per_frame": profile.geometry["tokens_per_latent_frame"],
                     "num_inference_steps": len(profile.sampler.denoising_steps)},
    }
