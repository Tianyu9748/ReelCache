"""SGLang configuration classes for the LongCat-Video continuation port (imports SGLang)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig
from sglang.multimodal_gen.configs.models.encoders import T5Config
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType, PipelineConfig
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams

from .context_parallel import optimal_split
from .spec import (
    DIRECT_720P,
    MAX_SEQUENCE_LENGTH,
    NATIVE_NEGATIVE_PROMPT,
    LongCatVideoArch,
)


@dataclass
class LongCatVideoArchConfig(DiTArchConfig):
    """Fields of ``weights/LongCat-Video/dit/config.json`` (kernel/parallel flags are recorded only)."""

    in_channels: int = 16
    out_channels: int = 16
    hidden_size: int = 4096
    depth: int = 48
    num_heads: int = 32
    caption_channels: int = 4096
    mlp_ratio: int = 4
    adaln_tembed_dim: int = 512
    frequency_embedding_size: int = 256
    patch_size: tuple = (1, 2, 2)
    text_tokens_zero_pad: bool = True
    enable_bsa: bool = False
    cp_split_hw: list | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.num_attention_heads = self.num_heads
        self.num_channels_latents = self.in_channels
        # Module names equal the native checkpoint keys: identity mapping.
        self.param_names_mapping = {}
        self.reverse_param_names_mapping = {}
        self.lora_param_names_mapping = {}

    def to_longcat_arch(self) -> LongCatVideoArch:
        return LongCatVideoArch.from_config_dict({
            "in_channels": self.in_channels, "out_channels": self.out_channels, "hidden_size": self.hidden_size,
            "depth": self.depth, "num_heads": self.num_heads, "caption_channels": self.caption_channels,
            "mlp_ratio": self.mlp_ratio, "adaln_tembed_dim": self.adaln_tembed_dim,
            "frequency_embedding_size": self.frequency_embedding_size, "patch_size": list(self.patch_size),
            "text_tokens_zero_pad": self.text_tokens_zero_pad, "enable_bsa": self.enable_bsa,
            "cp_split_hw": self.cp_split_hw})


@dataclass
class LongCatVideoDiTConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=LongCatVideoArchConfig)
    prefix: str = "longcat_video"


def _longcat_vae_config() -> WanVAEConfig:
    cfg = WanVAEConfig()
    cfg.load_encoder = True  # continuation re-encodes the last 13 RGB frames every segment
    cfg.load_decoder = True
    cfg.use_feature_cache = True  # native causal chunked encode/decode (1 + 4k frames, frame-by-frame decode)
    cfg.use_tiling = False
    cfg.use_temporal_tiling = False
    cfg.use_parallel_tiling = False
    cfg.use_parallel_encode = False
    cfg.use_parallel_decode = False
    return cfg


@dataclass
class LongCatVideoContinuationConfig(PipelineConfig):
    """Direct 720p video continuation (native distilled VC path): one GPU, or native LongCat context parallelism
    (``sp_degree = ulysses_degree = N``; 2-D spatial split, Ulysses all-to-all, replicated weights)."""

    task_type: ModelTaskType = ModelTaskType.F2V
    native_only_components: tuple[str, ...] = ("transformer", "vae", "text_encoder")
    should_use_guidance: bool = False
    enable_autocast: bool = False
    disable_autocast: bool = True
    flow_shift: float | None = None  # keep scheduler_config.json shift (12.0); checked at stage time

    dit_config: DiTConfig = field(default_factory=LongCatVideoDiTConfig)
    dit_precision: str = "bf16"

    vae_config: VAEConfig = field(default_factory=_longcat_vae_config)
    vae_precision: str = "bf16"  # native LongCat runs the VAE in BF16 (encode and decode)
    vae_decode_precision: str | None = "bf16"
    vae_tiling: bool = False
    vae_sp: bool = False

    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (T5Config(),))
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    text_encoder_extra_args: list[dict] = field(default_factory=lambda: [{"max_length": MAX_SEQUENCE_LENGTH}])

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True

    def validate_server_args(self, server_args) -> None:
        super().validate_server_args(server_args)
        problems = []
        for name in ("tp_size", "ring_degree", "kv_gather_degree", "dp_size"):
            value = getattr(server_args, name, None)
            if value is not None and value > 1:
                problems.append(f"{name}={value}")
        if getattr(server_args, "enable_cfg_parallel", False):
            problems.append("enable_cfg_parallel")
        sp = getattr(server_args, "sp_degree", None) or 1
        ulysses = getattr(server_args, "ulysses_degree", None) or 1
        if sp > 1 or ulysses > 1:
            # native LongCat CP = Ulysses over the whole group (no K/V-gather, ring or TP composition)
            if ulysses != sp:
                problems.append(f"sp_degree={sp} with ulysses_degree={ulysses} (context parallelism needs "
                                "--ulysses-degree equal to --sp-degree)")
            if getattr(server_args, "sp_split_auto", False):
                problems.append("auto-assigned SP split (set --ulysses-degree explicitly)")
            heads = getattr(self.dit_config.arch_config, "num_heads", 32)
            if heads % max(sp, 1):
                problems.append(f"{heads} heads not divisible by sp_degree={sp}")
            # native generate_vc rounds buckets to 16 * max(split): the ported 627/960 tables are those for 16/32
            if max(optimal_split(sp)) > 2:
                problems.append(f"sp_degree={sp} (native split {optimal_split(sp)} selects other bucket tables)")
        if problems:
            raise ValueError("LongCat-Video continuation port runs on one GPU or with native context parallelism "
                             "(sp_degree = ulysses_degree); unsupported: " + ", ".join(problems))

    def adjust_num_frames(self, num_frames, *, log_adjustment: bool = True):
        return num_frames


@dataclass
class LongCatVideoSamplingParams(SamplingParams):
    height: int = DIRECT_720P.height
    width: int = DIRECT_720P.width
    num_frames: int = DIRECT_720P.num_frames
    fps: int = 15
    num_inference_steps: int = 16
    guidance_scale: float = 1.0
    negative_prompt: str = NATIVE_NEGATIVE_PROMPT
    resolution: str = field(default="720p", metadata={"batch_sig_exclude": True})
    num_cond_frames: int = field(default=DIRECT_720P.num_cond_frames, metadata={"batch_sig_exclude": True})
    num_segments: int = field(default=1, metadata={"batch_sig_exclude": True})
    conditioning_cache_backend: str | None = field(default=None, metadata={"batch_sig_exclude": True})
    session_id: str | None = field(default=None, metadata={"batch_sig_exclude": True})
    # Resume/rewind at a segment boundary: the request's conditioning frames are that boundary's frames, segment
    # indices start at ``start_segment_index`` and ``generator_state`` (the hex state recorded in the segment record)
    # reproduces the original noise draw. Defaults keep the plain request unchanged.
    start_segment_index: int = field(default=0, metadata={"batch_sig_exclude": True})
    generator_state: str | None = field(default=None, metadata={"batch_sig_exclude": True})
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [(DIRECT_720P.width, DIRECT_720P.height), (832, 480)])

    @classmethod
    def video_request_extra_fields(cls) -> frozenset[str]:
        # /v1/videos forwards only declared model fields: the number of segments of a (session) request. Resume
        # state (start segment, generator state, conditioning frames) never comes from HTTP clients: a session's
        # boundary comes from its archive (session_provider).
        return super().video_request_extra_fields() | frozenset({"num_segments"})

    def _validate(self):
        super()._validate()
        if self.num_segments < 1:
            raise ValueError("num_segments must be >= 1")
        if self.start_segment_index is None or int(self.start_segment_index) < 0:
            raise ValueError("start_segment_index must be >= 0")
        if self.resolution not in ("480p", "720p"):
            raise ValueError("resolution must be '480p' or '720p'")
        if self.guidance_scale != 1.0:
            raise ValueError("only the distilled sampler (guidance_scale 1.0) is ported")

    def _validate_with_pipeline_config(self, pipeline_config):
        # A ReelCache session resume/fork/rewind (``causal_kv_session`` with an action other than ``generate``) takes
        # its boundary conditioning (13 frames + generator state) from the session archive and must not carry a
        # video (the session provider refuses one): the archive serves F2V's video input. The rest of the F2V input
        # contract is unchanged; every other request is validated by SGLang as before.
        handle = getattr(self, "causal_kv_session", None)
        if isinstance(handle, Mapping) and handle.get("action", "generate") != "generate" and not self.video_path:
            if self.image_path is not None:
                raise ValueError("input_reference is not supported for F2V models.")
            return
        super()._validate_with_pipeline_config(pipeline_config)
