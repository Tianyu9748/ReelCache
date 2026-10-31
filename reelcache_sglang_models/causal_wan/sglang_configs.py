"""SGLang pipeline/sampling configs of the causal-Wan profiles (imports SGLang).

Each profile gets its own ``PipelineConfig`` and ``SamplingParams`` subclass so the registry maps its exact
model directory to exactly one config pair. Values come from ``profiles.ModelProfile``; nothing is guessed
from family names.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import ClassVar

from sglang.multimodal_gen.configs.pipeline_configs.wan import WanT2V480PConfig
from sglang.multimodal_gen.configs.sample.wan import WanT2V_1_3B_SamplingParams
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from .profiles import ModelProfile, get_profile, prompt_switch_mode

logger = init_logger(__name__)


def wan_native_prompt_clean(text: str) -> str:
    """The original Wan tokenizer's ``clean='whitespace'``: ftfy + double html.unescape + strip, collapse spaces."""
    import ftfy  # required: the native tokenizer applies it (no silent skip)

    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class ReelCacheCausalWanProfileConfig(WanT2V480PConfig):
    """Shared causal-Wan profile config; subclasses set ``profile_id``."""

    profile_id: ClassVar[str] = ""
    is_causal: bool = True
    flow_shift: float | None = None
    dmd_denoising_steps: list[int] | None = field(default=None)
    warp_denoising_step: bool = False
    context_noise: int = 0
    vae_tiling: bool = False
    vae_sp: bool = False
    # Server-level override of the profile's default visibility policy (None: profile default).
    context_policy: str | None = None

    @classmethod
    def profile(cls) -> ModelProfile:
        return get_profile(cls.profile_id)

    def __post_init__(self):
        super().__post_init__()
        p = self.profile()
        s = p.sampler
        self.dmd_denoising_steps = list(s.denoising_steps)
        self.warp_denoising_step = s.warp_denoising_step
        self.flow_shift = s.timestep_shift
        self.context_noise = s.context_timestep
        d = p.dtypes
        self.dit_precision = d.sglang_dit_precision
        self.precision = d.sglang_dit_precision
        self.text_encoder_precisions = (d.sglang_text_encoder_precision,)
        self.vae_precision = d.sglang_vae_precision
        self.vae_decode_precision = d.sglang_vae_decode_precision
        self.preprocess_text_funcs = (wan_native_prompt_clean,)
        vis = p.visibility()
        arch = self.dit_config.arch_config
        arch.num_frames_per_block = p.chunk_latent_frames
        arch.local_attn_size = -1 if vis.window_latent_frames is None or p.recache.trigger == "none" \
            else vis.window_latent_frames
        arch.sink_size = vis.sink_latent_frames
        if self.context_policy is not None:
            _check_policy(p, self.context_policy)

    def adjust_num_frames(self, num_frames: int, *, log_adjustment: bool = True) -> int:
        stride = self.vae_config.arch_config.scale_factor_temporal
        chunk = self.profile().chunk_latent_frames
        latent = (num_frames - 1) // stride + 1
        if (num_frames - 1) % stride == 0 and latent % chunk == 0:
            return num_frames
        adjusted_latent = max(chunk, latent // chunk * chunk)
        adjusted = (adjusted_latent - 1) * stride + 1
        if log_adjustment:
            logger.warning("%s: num_frames %s does not map to whole %s-frame chunks; using %s",
                           self.profile_id, num_frames, chunk, adjusted)
        return adjusted


@dataclass
class ReelCacheCausalWanSamplingParams(WanT2V_1_3B_SamplingParams):
    """Sampling params of a causal-Wan profile; extra fields are request-scoped and excluded from batching."""

    profile_id: ClassVar[str] = ""
    guidance_scale: float = 1.0
    negative_prompt: str = ""
    num_frames: int = 81
    num_inference_steps: int | None = None
    seed: int = 0
    # Visibility policy for this request (None: server/profile default), e.g. CausVid "window_21_frames".
    context_policy: str | None = field(default=None, metadata={"batch_sig_exclude": True})
    # Prompt switches: [{"chunk": n, "prompt": "..."}]: chunk n (0-based) and later use the new prompt (LongLive 1.0:
    # KV recache at the switch; CausVid / RAVEN: cross-attention only, profiles.prompt_switch_mode).
    prompt_switches: list[dict] | None = field(default=None, metadata={"batch_sig_exclude": True})

    def __post_init__(self):
        if self.num_inference_steps is None:
            self.num_inference_steps = len(get_profile(self.profile_id).sampler.denoising_steps)
        super().__post_init__()

    @classmethod
    def video_request_extra_fields(cls) -> frozenset[str]:
        """Model-owned fields ``/v1/videos`` forwards to these params (SGLang forwards declared extras only):
        ``prompt_switches`` (LongLive 1.0 prompt switches) and ``context_policy`` (e.g. CausVid
        ``window_21_frames``); without this declaration the HTTP server silently drops both (Step 14 CR-14-1)."""
        return super().video_request_extra_fields() | frozenset({"prompt_switches", "context_policy"})

    def _validate(self):
        super()._validate()
        p = get_profile(self.profile_id)
        if self.num_inference_steps != len(p.sampler.denoising_steps):
            raise ValueError(f"{p.profile_id} uses its fixed {len(p.sampler.denoising_steps)}-step schedule "
                             f"{list(p.sampler.denoising_steps)}; got num_inference_steps={self.num_inference_steps}")
        if self.guidance_scale != p.cfg.guidance_scale:
            raise ValueError(f"{p.profile_id} runs without classifier-free guidance (guidance_scale "
                             f"{p.cfg.guidance_scale}); got {self.guidance_scale}")
        if self.context_policy is not None:
            _check_policy(p, self.context_policy)
        if self.prompt_switches:
            if prompt_switch_mode(p) is None:
                raise ValueError(f"{p.profile_id} has no prompt-switch semantics")
            normalize_prompt_switches(self.prompt_switches)


def _check_policy(profile: ModelProfile, name: str) -> None:
    try:
        profile.visibility(name)
    except KeyError as exc:
        raise ValueError(str(exc)) from None


def normalize_prompt_switches(switches) -> list[dict]:
    if not switches:
        return []
    out, last = [], 0
    for item in switches:
        if not isinstance(item, dict) or set(item) != {"chunk", "prompt"}:
            raise ValueError("prompt_switches entries must be {'chunk': int, 'prompt': str}")
        chunk, prompt = item["chunk"], item["prompt"]
        if not isinstance(chunk, int) or chunk <= last or not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt_switches need strictly increasing chunk indices > 0 and non-empty prompts")
        out.append({"chunk": chunk, "prompt": prompt})
        last = chunk
    return out


@dataclass
class CausVidProfileConfig(ReelCacheCausalWanProfileConfig):
    profile_id: ClassVar[str] = "causvid"


@dataclass
class CausVidSamplingParams(ReelCacheCausalWanSamplingParams):
    profile_id: ClassVar[str] = "causvid"


@dataclass
class LongLive1ProfileConfig(ReelCacheCausalWanProfileConfig):
    profile_id: ClassVar[str] = "longlive1"


@dataclass
class LongLive1SamplingParams(ReelCacheCausalWanSamplingParams):
    profile_id: ClassVar[str] = "longlive1"


@dataclass
class RavenProfileConfig(ReelCacheCausalWanProfileConfig):
    profile_id: ClassVar[str] = "raven"


@dataclass
class RavenSamplingParams(ReelCacheCausalWanSamplingParams):
    profile_id: ClassVar[str] = "raven"


CONFIGS = {
    "causvid": (CausVidProfileConfig, CausVidSamplingParams),
    "longlive1": (LongLive1ProfileConfig, LongLive1SamplingParams),
    "raven": (RavenProfileConfig, RavenSamplingParams),
}
