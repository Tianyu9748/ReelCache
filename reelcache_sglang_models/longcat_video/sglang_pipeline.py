"""SGLang composed pipeline for LongCat-Video direct-720p continuation (imports SGLang)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import ComposedPipelineBase
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision_types import PRECISION_TO_TYPE

from .sglang_configs import LongCatVideoContinuationConfig, LongCatVideoSamplingParams
from .session_provider import PROVIDER_SPEC
from .sglang_loader import LongCatVideoTransformerLoader, record_verified_checkpoint
from .sglang_stages import (
    LongCatVideoContinuationStage,
    LongCatVideoInputStage,
    LongCatVideoTextEncodingStage,
)
from .spec import DIT_CLASS_NAME, PIPELINE_NAME
from .weights import verify_checkpoint

# The native checkpoint has no diffusers model_index.json (its model_index.json is {"model_name": ...}).
# This index maps SGLang component slots to the native subfolders; "transformer" loads from dit/.
MODEL_INDEX: dict[str, Any] = {
    "_class_name": PIPELINE_NAME,
    "_diffusers_version": "0.32.0",
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "AutoTokenizer"],
    "vae": ["diffusers", "AutoencoderKLWan"],
    "transformer": ["diffusers", DIT_CLASS_NAME],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
}
TRANSFORMER_SUBFOLDER = "dit"


class LongCatVideoContinuationPipeline(ComposedPipelineBase):
    pipeline_name = PIPELINE_NAME
    # --reelcache-enable: SGLang's backend factory loads this provider (Step 14 session controls, segment boundaries)
    reelcache_session_provider = PROVIDER_SPEC
    is_video_pipeline = True
    pipeline_config_cls = LongCatVideoContinuationConfig
    sampling_params_cls = LongCatVideoSamplingParams
    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]
    component_loaders = {"transformer": LongCatVideoTransformerLoader}

    def _load_config(self) -> dict[str, Any]:
        root = Path(self.model_path)
        if not root.is_dir():
            raise ValueError(f"LongCat-Video port needs the hash-bound local checkpoint directory; got {self.model_path!r}")
        report = verify_checkpoint(root)
        record_verified_checkpoint(root, report)
        self.weight_binding_report = report
        self.model_path = str(root)
        return dict(MODEL_INDEX)

    def _resolve_component_path(self, server_args: ServerArgs, module_name: str, load_module_name: str) -> str:
        if module_name == "transformer" and server_args.component_paths.get(module_name) is None:
            return os.path.join(self.model_path, TRANSFORMER_SUBFOLDER)
        return super()._resolve_component_path(server_args, module_name, load_module_name)

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError(f"{PIPELINE_NAME} supports monolithic deployment only, got {role.value!r}")

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        server_args.pipeline_config.validate_server_args(server_args)
        transformer = self.get_module("transformer")
        if type(transformer).__name__ != DIT_CLASS_NAME or not hasattr(transformer, "longcat_load_report"):
            raise RuntimeError("LongCat-Video transformer was not loaded through LongCatVideoTransformerLoader")
        vae = self.get_module("vae")
        if getattr(vae, "encoder", None) is None:
            raise RuntimeError("LongCat-Video continuation needs the VAE encoder (load_encoder=True)")

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(LongCatVideoInputStage(), "longcat_video_input_stage")
        self.add_stage(LongCatVideoTextEncodingStage(text_encoder=self.get_module("text_encoder"),
                                                     tokenizer=self.get_module("tokenizer"),
                                                     prompt_dtype=PRECISION_TO_TYPE[
                                                         server_args.pipeline_config.dit_precision]),
                       "longcat_video_text_encoding_stage")
        self.add_stage(LongCatVideoContinuationStage(transformer=self.get_module("transformer"),
                                                     vae=self.get_module("vae"),
                                                     scheduler=self.get_module("scheduler")),
                       "longcat_video_continuation_stage")


EntryClass = LongCatVideoContinuationPipeline
