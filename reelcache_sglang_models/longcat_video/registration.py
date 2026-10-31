"""Register the LongCat-Video continuation port with SGLang Diffusion (imports SGLang)."""

from __future__ import annotations

import threading

from .spec import DIT_CLASS_NAME, HF_REPO_ID, PIPELINE_NAME

DIT_MODULE_PATH = "reelcache_sglang_models.longcat_video.sglang_dit:LongCatVideoTransformer3DModel"
_LOCK = threading.Lock()
_REGISTERED: dict | None = None


def register_longcat_video() -> dict:
    """Idempotently register the DiT architecture and the composed pipeline in this process."""
    global _REGISTERED
    with _LOCK:
        if _REGISTERED is not None:
            return dict(_REGISTERED)
        import sglang.multimodal_gen.registry as pipeline_registry
        from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

        if not hasattr(pipeline_registry, "register_pipeline"):
            raise ImportError("sglang.multimodal_gen.registry is not initialized; cannot register LongCat-Video")
        from .sglang_configs import LongCatVideoContinuationConfig, LongCatVideoSamplingParams
        from .sglang_pipeline import LongCatVideoContinuationPipeline

        ModelRegistry.register_model(DIT_CLASS_NAME, DIT_MODULE_PATH)
        pipeline_registry.register_pipeline(
            LongCatVideoContinuationPipeline,
            sampling_param_cls=LongCatVideoSamplingParams,
            pipeline_config_cls=LongCatVideoContinuationConfig,
            hf_model_paths=[HF_REPO_ID],
        )
        _REGISTERED = {"pipeline": PIPELINE_NAME, "dit_architecture": DIT_CLASS_NAME, "dit_module": DIT_MODULE_PATH,
                       "hf_model_paths": [HF_REPO_ID]}
        return dict(_REGISTERED)


def registered() -> dict | None:
    return None if _REGISTERED is None else dict(_REGISTERED)
