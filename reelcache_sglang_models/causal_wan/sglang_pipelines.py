"""SGLang composed pipelines of the causal-Wan profiles (imports SGLang).

All three are ``WanCausalDMDPipeline`` subclasses (the shared causal-Wan path, ``CausalWanTransformer3DModel``).
Each is bound to exactly one profile model directory: ``_load_config`` verifies the directory's binding
(converted transformer SHA256, Diffusers component links/blobs, binding fingerprint) and fails loudly on any
mismatch; ``initialize_pipeline`` installs the profile's native scheduler, checks the loaded DiT and applies the
profile's DiT precision policy (``sglang_dit.ReelCacheCausalWanTransformer3DModel.apply_precision_policy``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines.wan_causal_dmd_pipeline import WanCausalDMDPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs

from .convert import model_index_for, verify_profile_dir
from .profiles import ModelProfile, get_profile
from .sampling import scheduler_for
from .sglang_configs import CONFIGS
from .sglang_stages import (
    ProfileCausalDenoisingStage,
    ProfileLatentPreparationStage,
    ProfileTextEncodingStage,
    RavenProfileDenoisingStage,
)

VERIFY_ENV = "REELCACHE_PROFILE_VERIFY"  # "full" (default): re-hash the converted transformer; "size": size only


class ReelCacheCausalWanProfilePipeline(WanCausalDMDPipeline):
    """Shared base; not registered itself."""

    pipeline_name = "ReelCacheCausalWanProfilePipeline"
    profile_id: ClassVar[str] = ""
    is_video_pipeline = True
    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]

    @classmethod
    def profile(cls) -> ModelProfile:
        return get_profile(cls.profile_id)

    def _load_config(self) -> dict[str, Any]:
        profile = self.profile()
        root = Path(self.model_path)
        if not root.is_dir():
            raise ValueError(f"{self.pipeline_name} needs its bound profile directory {profile.sglang.model_path}; "
                             f"got {self.model_path!r}")
        mode = os.environ.get(VERIFY_ENV, "full")
        if mode not in ("full", "size"):
            raise ValueError(f"{VERIFY_ENV} must be 'full' or 'size'")
        self.profile_binding_report = verify_profile_dir(profile, root, full_hash=mode == "full")
        self.model_path = str(root)
        return model_index_for(profile)

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError(f"{self.pipeline_name} supports monolithic deployment only, got {role.value!r}")

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        profile = self.profile()
        expected_cfg, _ = CONFIGS[profile.profile_id]
        if type(server_args.pipeline_config) is not expected_cfg:
            raise TypeError(f"{self.pipeline_name} needs {expected_cfg.__name__}, got "
                            f"{type(server_args.pipeline_config).__name__}")
        transformer = self.get_module("transformer")
        if type(transformer).__name__ != profile.sglang.dit_class:
            raise TypeError(f"expected {profile.sglang.dit_class}, loaded {type(transformer).__name__}")
        cfg, bb = transformer.config, profile.backbone
        got = {"num_layers": cfg.num_layers, "num_attention_heads": cfg.num_attention_heads,
               "attention_head_dim": cfg.attention_head_dim, "ffn_dim": cfg.ffn_dim,
               "num_frames_per_block": cfg.num_frames_per_block}
        want = {"num_layers": bb["num_layers"], "num_attention_heads": bb["num_attention_heads"],
                "attention_head_dim": bb["attention_head_dim"], "ffn_dim": bb["ffn_dim"],
                "num_frames_per_block": profile.chunk_latent_frames}
        if got != want:
            raise ValueError(f"{self.pipeline_name}: loaded DiT config {got} != profile {want}")
        d = profile.dtypes
        self.dit_precision_report = transformer.apply_precision_policy(d.sglang_dit_compute,
                                                                       tf32=d.sglang_fp32_regions_tf32)
        self.modules["scheduler"] = scheduler_for(profile)

    def denoising_stage_cls(self):
        return RavenProfileDenoisingStage if self.profile().sampler.kind == "raven_consistency_v_lerp" \
            else ProfileCausalDenoisingStage

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        profile = self.profile()
        self.add_stage(InputValidationStage())
        self.add_stage(ProfileTextEncodingStage(text_encoders=[self.get_module("text_encoder")],
                                                tokenizers=[self.get_module("tokenizer")]))
        self.add_stage(ProfileLatentPreparationStage(scheduler=self.get_module("scheduler"),
                                                     transformer=self.get_module("transformer"), profile=profile))
        self.add_stage(self.denoising_stage_cls()(transformer=self.get_module("transformer"),
                                                  scheduler=self.get_module("scheduler"), profile=profile))
        self.add_standard_decoding_stage()


class ReelCacheCausVidProfilePipeline(ReelCacheCausalWanProfilePipeline):
    pipeline_name = "ReelCacheCausVidProfilePipeline"
    profile_id: ClassVar[str] = "causvid"
    pipeline_config_cls, sampling_params_cls = CONFIGS["causvid"]


class ReelCacheLongLive1ProfilePipeline(ReelCacheCausalWanProfilePipeline):
    pipeline_name = "ReelCacheLongLive1ProfilePipeline"
    profile_id: ClassVar[str] = "longlive1"
    pipeline_config_cls, sampling_params_cls = CONFIGS["longlive1"]


class ReelCacheRavenProfilePipeline(ReelCacheCausalWanProfilePipeline):
    pipeline_name = "ReelCacheRavenProfilePipeline"
    profile_id: ClassVar[str] = "raven"
    pipeline_config_cls, sampling_params_cls = CONFIGS["raven"]


PIPELINES = {"causvid": ReelCacheCausVidProfilePipeline, "longlive1": ReelCacheLongLive1ProfilePipeline,
             "raven": ReelCacheRavenProfilePipeline}
