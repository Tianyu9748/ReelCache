"""LongCat-Video direct-720p continuation: architecture, geometry and sampler specification.

Pure Python (no torch, no SGLang). Every number here is taken from the native LongCat-Video
source at commit ``6b3f4b8`` and its HF checkpoint revision ``03b5552``:

* ``weights/LongCat-Video/dit/config.json`` (architecture),
* ``longcat_video/utils/bukcet_config.py`` (aspect-ratio buckets),
* ``longcat_video/pipeline_longcat_video.py`` (continuation geometry, distilled sigmas),
* ``weights/LongCat-Video/scheduler/scheduler_config.json`` (flow shift 12),
* ``weights/LongCat-Video/vae/config.json`` (Wan 2.1 VAE, 4x temporal / 8x spatial).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

LONGCAT_VIDEO_COMMIT = "6b3f4b8582a8bc3f20f795735f5383716c4ba794"
HF_REPO_ID = "meituan-longcat/LongCat-Video"
HF_REVISION = "03b55529b1d1d4045f5fbe14d65c8c6e8116b278"
DIT_CLASS_NAME = "LongCatVideoTransformer3DModel"
PIPELINE_NAME = "LongCatVideoContinuationPipeline"

# Native text conditioning: UMT5-XXL, padded/truncated to 512 tokens (pipeline_longcat_video.py).
MAX_SEQUENCE_LENGTH = 512
NATIVE_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly "
    "drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy "
    "background, three legs, many people in the background, walking backwards"
)

# Wan 2.1 VAE (weights/LongCat-Video/vae/config.json).
VAE_SCALE_FACTOR_TEMPORAL = 4
VAE_SCALE_FACTOR_SPATIAL = 8
VAE_Z_DIM = 16
VAE_LATENTS_MEAN = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)
VAE_LATENTS_STD = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916,
)


@dataclass(frozen=True)
class LongCatVideoArch:
    """DiT architecture of ``weights/LongCat-Video/dit/config.json``."""

    in_channels: int = 16
    out_channels: int = 16
    hidden_size: int = 4096
    depth: int = 48
    num_heads: int = 32
    caption_channels: int = 4096
    mlp_ratio: int = 4
    adaln_tembed_dim: int = 512
    frequency_embedding_size: int = 256
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_tokens_zero_pad: bool = True
    norm_eps: float = 1e-6
    rope_base: float = 10000.0

    # Native config keys that select kernels or parallelism, not math. They are recorded and
    # validated (block-sparse attention and context parallelism are not part of this port).
    IGNORED_KEYS = ("enable_flashattn3", "enable_flashattn2", "enable_xformers", "bsa_params", "_class_name",
                    "_diffusers_version")

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.head_dim % 8:
            raise ValueError("LongCat 3D RoPE needs head_dim divisible by 8")
        if tuple(self.patch_size)[0] != 1:
            raise ValueError("LongCat-Video patch embedding must not compress time (patch_size[0] == 1)")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def ffn_hidden_dim(self) -> int:
        """``FeedForwardSwiGLU(dim, int(hidden * mlp_ratio))`` sizing (multiple_of=256)."""
        hidden = int(2 * int(self.hidden_size * self.mlp_ratio) / 3)
        return 256 * ((hidden + 256 - 1) // 256)

    @property
    def rope_axis_dims(self) -> tuple[int, int, int]:
        d = self.head_dim
        return d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)

    @property
    def patch_numel(self) -> int:
        t, h, w = self.patch_size
        return t * h * w

    @classmethod
    def from_config_dict(cls, config: Mapping[str, Any]) -> "LongCatVideoArch":
        """Build from the native ``dit/config.json``; reject settings this port does not implement."""
        cfg = dict(config)
        if cfg.get("enable_bsa"):
            raise ValueError("block-sparse attention (enable_bsa) is the refinement path; not part of this port")
        cp = cfg.get("cp_split_hw")
        if cp not in (None, [1, 1], (1, 1)):
            raise ValueError(f"context-parallel cp_split_hw={cp!r} is not supported by this port")
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        kwargs = {}
        for key, value in cfg.items():
            if key in cls.IGNORED_KEYS or key in ("enable_bsa", "cp_split_hw"):
                continue
            if key not in known:
                raise ValueError(f"unknown LongCat-Video DiT config key {key!r}")
            kwargs[key] = tuple(value) if key == "patch_size" else value
        return cls(**kwargs)


LONGCAT_VIDEO_ARCH = LongCatVideoArch()

# ---------------------------------------------------------------------------------------------------
# Aspect-ratio buckets (longcat_video/utils/bukcet_config.py). Single GPU: cp_split_hw = [1, 1], so the
# pipeline's scale_factor_spatial = vae_scale_factor_spatial * 2 = 16 selects ASPECT_RATIO_627/960.
# Dict order matters: the native lookup sorts keys by distance with a stable sort.
# ---------------------------------------------------------------------------------------------------
ASPECT_RATIO_627 = {
    "0.26": ([320, 1216], 1), "0.31": ([352, 1120], 1), "0.38": ([384, 1024], 1), "0.43": ([416, 960], 1),
    "0.52": ([448, 864], 1), "0.58": ([480, 832], 1), "0.67": ([512, 768], 1), "0.74": ([544, 736], 1),
    "0.86": ([576, 672], 1), "0.95": ([608, 640], 1), "1.05": ([640, 608], 1), "1.17": ([672, 576], 1),
    "1.29": ([704, 544], 1), "1.35": ([736, 544], 1), "1.50": ([768, 512], 1), "1.67": ([800, 480], 1),
    "1.73": ([832, 480], 1), "2.00": ([896, 448], 1), "2.31": ([960, 416], 1), "2.58": ([992, 384], 1),
    "2.75": ([1056, 384], 1), "3.09": ([1088, 352], 1), "3.70": ([1184, 320], 1), "3.80": ([1216, 320], 1),
    "3.90": ([1248, 320], 1), "4.00": ([1280, 320], 1),
}
ASPECT_RATIO_960 = {
    "0.25": ([480, 1920], 1), "0.29": ([512, 1792], 1), "0.32": ([544, 1696], 1), "0.36": ([576, 1600], 1),
    "0.40": ([608, 1504], 1), "0.49": ([672, 1376], 1), "0.54": ([704, 1312], 1), "0.59": ([736, 1248], 1),
    "0.69": ([800, 1152], 1), "0.74": ([832, 1120], 1), "0.82": ([864, 1056], 1), "0.88": ([896, 1024], 1),
    "0.94": ([928, 992], 1), "1.00": ([960, 960], 1), "1.07": ([992, 928], 1), "1.14": ([1024, 896], 1),
    "1.22": ([1056, 864], 1), "1.31": ([1088, 832], 1), "1.35": ([1120, 832], 1), "1.44": ([1152, 800], 1),
    "1.70": ([1248, 736], 1), "2.00": ([1344, 672], 1), "2.05": ([1376, 672], 1), "2.47": ([1504, 608], 1),
    "2.53": ([1536, 608], 1), "2.83": ([1632, 576], 1), "3.06": ([1664, 544], 1), "3.12": ([1696, 544], 1),
    "3.62": ([1856, 512], 1), "3.93": ([1888, 480], 1), "4.00": ([1920, 480], 1),
}
BUCKETS = {"480p": ASPECT_RATIO_627, "720p": ASPECT_RATIO_960}
SINGLE_GPU_BUCKET_SCALE = VAE_SCALE_FACTOR_SPATIAL * 2


def bucket_for_condition(height: int, width: int, resolution: str = "720p") -> tuple[str, int, int]:
    """Native ``get_condition_shape``: closest bucket to the first condition frame's height/width."""
    if resolution not in BUCKETS:
        raise ValueError(f"unsupported resolution {resolution!r}; expected one of {sorted(BUCKETS)}")
    if height <= 0 or width <= 0:
        raise ValueError("condition frame size must be positive")
    table = BUCKETS[resolution]
    ratio = height / width
    key = sorted(table.keys(), key=lambda x: abs(float(x) - ratio))[0]
    target_h, target_w = table[key][0]
    if target_h % SINGLE_GPU_BUCKET_SCALE or target_w % SINGLE_GPU_BUCKET_SCALE:
        raise ValueError(f"bucket {key} {target_h}x{target_w} not divisible by {SINGLE_GPU_BUCKET_SCALE}")
    return key, target_h, target_w


def latent_frames(num_frames: int) -> int:
    return 1 + (num_frames - 1) // VAE_SCALE_FACTOR_TEMPORAL


@dataclass(frozen=True)
class ContinuationGeometry:
    """One video-continuation segment (``generate_vc``) at a bucket resolution."""

    height: int = 704
    width: int = 1312
    num_frames: int = 93
    num_cond_frames: int = 13
    arch: LongCatVideoArch = field(default=LONGCAT_VIDEO_ARCH)

    def __post_init__(self) -> None:
        if self.num_frames % VAE_SCALE_FACTOR_TEMPORAL != 1:
            raise ValueError("num_frames - 1 must be divisible by the VAE temporal factor (4)")
        if self.num_cond_frames % VAE_SCALE_FACTOR_TEMPORAL != 1 or not 0 < self.num_cond_frames < self.num_frames:
            raise ValueError("num_cond_frames must be 4k+1 and smaller than num_frames")
        scale = VAE_SCALE_FACTOR_SPATIAL * self.arch.patch_size[1]
        if self.height % scale or self.width % scale:
            raise ValueError(f"height/width must be divisible by {scale}")

    @property
    def latent_frames(self) -> int:
        return latent_frames(self.num_frames)

    @property
    def cond_latent_frames(self) -> int:
        return latent_frames(self.num_cond_frames)

    @property
    def new_latent_frames(self) -> int:
        return self.latent_frames - self.cond_latent_frames

    @property
    def latent_hw(self) -> tuple[int, int]:
        return self.height // VAE_SCALE_FACTOR_SPATIAL, self.width // VAE_SCALE_FACTOR_SPATIAL

    @property
    def token_hw(self) -> tuple[int, int]:
        lh, lw = self.latent_hw
        return lh // self.arch.patch_size[1], lw // self.arch.patch_size[2]

    @property
    def tokens_per_latent_frame(self) -> int:
        th, tw = self.token_hw
        return th * tw

    @property
    def cond_tokens(self) -> int:
        return self.cond_latent_frames * self.tokens_per_latent_frame

    @property
    def new_tokens(self) -> int:
        return self.new_latent_frames * self.tokens_per_latent_frame

    @property
    def new_rgb_frames(self) -> int:
        return self.num_frames - self.num_cond_frames

    @property
    def latent_shape(self) -> tuple[int, int, int, int, int]:
        lh, lw = self.latent_hw
        return 1, self.arch.in_channels, self.latent_frames, lh, lw

    @property
    def cond_token_grid(self) -> tuple[int, int, int]:
        return (self.cond_latent_frames, *self.token_hw)

    @property
    def full_token_grid(self) -> tuple[int, int, int]:
        return (self.latent_frames, *self.token_hw)

    def cache_bytes(self, bytes_per_element: int = 2) -> int:
        a = self.arch
        return a.depth * 2 * self.cond_tokens * a.hidden_size * bytes_per_element

    def as_dict(self) -> dict:
        return {"height": self.height, "width": self.width, "num_frames": self.num_frames,
                "num_cond_frames": self.num_cond_frames, "latent_frames": self.latent_frames,
                "cond_latent_frames": self.cond_latent_frames, "new_latent_frames": self.new_latent_frames,
                "latent_hw": list(self.latent_hw), "token_hw": list(self.token_hw),
                "tokens_per_latent_frame": self.tokens_per_latent_frame, "cond_tokens": self.cond_tokens,
                "new_tokens": self.new_tokens, "new_rgb_frames": self.new_rgb_frames,
                "cache_bytes_bf16": self.cache_bytes()}


DIRECT_720P = ContinuationGeometry()


@dataclass(frozen=True)
class DistilledSampler:
    """Official distilled continuation sampler (``generate_vc(use_distill=True)``, cfg_step_lora)."""

    num_inference_steps: int = 16
    guidance_scale: float = 1.0
    use_distill: bool = True
    enhance_hf: bool = False
    num_train_timesteps: int = 1000
    num_distill_sample_steps: int = 50
    flow_shift: float = 12.0
    lora_name: str = "cfg_step_lora"
    lora_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not self.use_distill or self.enhance_hf:
            raise ValueError("only the distilled sampler (use_distill=True, enhance_hf=False) is ported")
        if self.guidance_scale != 1.0:
            raise ValueError("the distilled sampler runs without CFG (guidance_scale 1.0)")
        if not 0 < self.num_inference_steps <= self.num_distill_sample_steps:
            raise ValueError("num_inference_steps must be in [1, num_distill_sample_steps]")


DISTILLED_SAMPLER = DistilledSampler()
