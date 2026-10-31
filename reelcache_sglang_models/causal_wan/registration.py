"""Register the causal-Wan profile pipelines with SGLang Diffusion (imports SGLang inside the function only).

Each pipeline is registered to exactly ONE model path: its profile directory under
``profiles.PROFILE_MODEL_ROOT`` (no detectors, no family-name patterns). Before registering, the function
checks that none of those paths collides with an existing registry entry in either direction (exact path,
SGLang's basename/substring partial matching, non-diffusers patterns), so existing models - in particular
LongLive 2.0 and the Wan2.1 Diffusers repos - keep resolving to their own pipelines. Failures raise.
"""

from __future__ import annotations

import os
import threading

from .profiles import PROFILES

_LOCK = threading.Lock()
_REGISTERED: dict | None = None


def _short(path: str) -> str:
    return path.rstrip("/").split("/")[-1].lower()


def collision_report(pipeline_registry, model_paths: list[str]) -> list[str]:
    """Problems that would make an exact profile path resolve ambiguously (empty list = collision-free)."""
    problems = []
    known = dict(pipeline_registry._MODEL_HF_PATH_TO_NAME)
    patterns = dict(pipeline_registry.KNOWN_NON_DIFFUSERS_DIFFUSION_MODEL_PATTERNS)
    for path in model_paths:
        norm = os.path.normpath(path).lower()
        mine = _short(path)
        if path in known or path.lower() in patterns:
            problems.append(f"{path}: already registered")
        for other in known:
            theirs = _short(other)
            if theirs in mine:
                problems.append(f"{path}: existing entry {other!r} ({theirs}) is a substring of the profile basename")
            if mine in theirs:
                problems.append(f"{path}: profile basename is a substring of existing entry {other!r}")
            if f"models--{other.lower().replace('/', '--')}" in norm:
                problems.append(f"{path}: contains the HF cache fragment of {other!r}")
        for pattern in patterns:
            if "/" not in pattern and pattern in norm:
                problems.append(f"{path}: contains the non-diffusers pattern {pattern!r}")
            if "/" in pattern and _short(pattern) == mine:
                problems.append(f"{path}: same basename as non-diffusers pattern {pattern!r}")
        detected = pipeline_registry.get_non_diffusers_pipeline_name(path)
        if detected is not None:
            problems.append(f"{path}: already resolves to pipeline {detected!r}")
    return problems


def register_causal_wan_profiles() -> dict:
    """Idempotently register the CausVid, LongLive 1.0 and RAVEN profile pipelines in this process."""
    global _REGISTERED
    with _LOCK:
        if _REGISTERED is not None:
            return dict(_REGISTERED)
        import sglang.multimodal_gen.registry as pipeline_registry
        from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

        if not hasattr(pipeline_registry, "register_pipeline"):
            raise ImportError("sglang.multimodal_gen.registry is not initialized; cannot register causal-Wan profiles")
        from .sglang_configs import CONFIGS
        from .sglang_dit import DIT_ARCH_NAME, DIT_MODULE_PATH
        from .sglang_pipelines import PIPELINES

        pipeline_registry._discover_and_register_pipelines()
        paths = [PROFILES[pid].sglang.model_path for pid in PIPELINES]
        problems = collision_report(pipeline_registry, paths)
        if DIT_ARCH_NAME in ModelRegistry.get_supported_archs():
            problems.append(f"DiT architecture {DIT_ARCH_NAME} already registered")
        if any(PROFILES[pid].sglang.dit_class != DIT_ARCH_NAME for pid in PIPELINES):
            problems.append("profile DiT class differs from the registered architecture")
        if problems:
            raise RuntimeError("causal-Wan profile registration would collide: " + "; ".join(problems))
        ModelRegistry.register_model(DIT_ARCH_NAME, DIT_MODULE_PATH)
        out = {"dit_architecture": {"name": DIT_ARCH_NAME, "module": DIT_MODULE_PATH}}
        for pid, pipe_cls in PIPELINES.items():
            cfg_cls, sp_cls = CONFIGS[pid]
            path = PROFILES[pid].sglang.model_path
            pipeline_registry.register_pipeline(pipe_cls, sampling_param_cls=sp_cls, pipeline_config_cls=cfg_cls,
                                                hf_model_paths=[path])
            out[pid] = {"pipeline": pipe_cls.pipeline_name, "model_path": path,
                        "pipeline_config": cfg_cls.__name__, "sampling_params": sp_cls.__name__}
        _REGISTERED = out
        return dict(out)


def registered() -> dict | None:
    return None if _REGISTERED is None else dict(_REGISTERED)
