"""ReelCache bridge for SGLang Diffusion: shared causal-Wan KV archive (``--reelcache-enable``).

SGLang loads this package only when ReelCache is explicitly enabled and calls
:func:`create_sglang_backend` once per GPU worker; ReelCache core never imports
SGLang, and this package imports SGLang only through the objects SGLang passes
in. Public pieces:

* :class:`ReelCacheRuntimeConfig` (validated plain configuration),
* :class:`ModelProfile` / :data:`LONGLIVE2` / :func:`register_profile`
  (shared-backbone profile data, exact model-path matching),
* :class:`CausalWanKVBridge` (capture, commit, visibility, restore/install,
  recache epochs, resident native KV across requests) on the Step 6 runtime
  (:class:`LineageSessionArchive`: lineages, ordered asynchronous archival
  across request boundaries, packed device restore) or the exact RAW reference
  (:class:`RawSessionArchive`),
* :class:`CleanChunkBuilder`, :class:`VisibilityPlan` / :class:`TokenSpan`,
  :class:`AttentionKVLease`, :class:`SGLangSessionState`,
  :class:`SGLangSessionController` (open, pause, checkpoint, resume, rewind,
  fork, close),
* :class:`SessionControls` / :class:`SessionControlError`: the worker's typed
  session control surface (``bridge.control(op, args)``, Step 14).
"""

from __future__ import annotations

from typing import Any, Mapping

from ...errors import ConfigurationError
from .archive import RawSessionArchive
from .bridge import CausalWanKVBridge
from .capture import CleanChunkBuilder
from .config import ReelCacheRuntimeConfig
from .controls import SessionControlError, SessionControls, prompt_change_capability
from .lineage_archive import LineageSessionArchive
from .nodes import BridgeNode, CoverageRange
from .profiles import LONGLIVE2, PROFILES, ModelProfile, profile_for, register_profile
from .restore import AttentionKVLease
from .session import SGLangSessionController, SGLangSessionState
from .visibility import TokenSpan, VisibilityPlan


def _causal_stage_model(pipeline: Any, server_args: Any) -> dict[str, Any]:
    stages = [stage for stage in getattr(pipeline, "stages", ()) if callable(getattr(stage, "set_causal_kv_backend", None))]
    transformers = {type(stage.transformer).__name__ for stage in stages if getattr(stage, "transformer", None) is not None}
    if len(transformers) != 1:
        raise ConfigurationError(f"expected one causal transformer class in the pipeline, found {sorted(transformers)}")
    return {"transformer": transformers.pop(), "model_path": getattr(server_args, "model_path", None),
            "pipeline": type(pipeline).__name__}


def _session_provider(pipeline: Any) -> Any:
    """The pipeline-declared session provider factory (``reelcache_session_provider``: ``"module:function"`` or a
    callable), for pipelines whose sessions live behind their own seam (no causal denoising stage)."""
    import importlib

    provider = getattr(pipeline, "reelcache_session_provider", None)
    if provider is None or callable(provider):
        return provider
    module, _, name = str(provider).partition(":")
    if not module or not name:
        raise ConfigurationError(f"reelcache_session_provider must be 'module:function', got {provider!r}")
    return getattr(importlib.import_module(module), name)


def create_sglang_backend(*, config: Mapping[str, Any], pipeline: Any, server_args: Any) -> Any:
    """SGLang's backend factory (``--reelcache-enable``): validate configuration, pick the profile, build the bridge.

    A pipeline without causal denoising stages that declares ``reelcache_session_provider`` gets that provider's
    backend (same worker interface: ``control``, ``inflight_control``, ``set_boundary_hook``, ``close``)."""
    import torch

    has_causal_stage = any(callable(getattr(stage, "set_causal_kv_backend", None))
                           for stage in getattr(pipeline, "stages", ()))
    provider = None if has_causal_stage else _session_provider(pipeline)
    if provider is not None:
        # The provider receives the --reelcache-config dict itself. SGLang's backend_config() also carries its
        # --reelcache-hevc-inter-prediction flag (a bridge codec option): passed on only when set, so a provider that
        # does not know it refuses it explicitly instead of every provider having to accept it.
        provider_config = dict(config)
        if provider_config.get("hevc_inter_prediction") is False:
            del provider_config["hevc_inter_prediction"]
        backend = provider(config=provider_config, pipeline=pipeline, server_args=server_args)
        startup = getattr(backend, "startup", None)
        if isinstance(startup, dict):
            startup.setdefault("gc_freeze", freeze_long_lived_objects())
        else:
            freeze_long_lived_objects()
        return backend
    runtime_config = ReelCacheRuntimeConfig.from_dict(config)
    model = _causal_stage_model(pipeline, server_args)
    profile = profile_for(model, runtime_config.profile, fields=runtime_config.profile_fields)
    if not torch.cuda.is_available():
        raise ConfigurationError("the ReelCache SGLang bridge runs inside a CUDA GPU worker")
    device = torch.device("cuda", torch.cuda.current_device())
    bridge = CausalWanKVBridge(runtime_config, profile, device=device, model_path=model["model_path"])
    # Process-level warm-up of the engine's RoPE for native-RoPE restores (RAW, rope_mode="native").
    stages = [stage for stage in getattr(pipeline, "stages", ()) if callable(getattr(stage, "set_causal_kv_backend", None))]
    transformer = getattr(stages[0], "transformer", None) if stages else None
    if transformer is not None:
        bridge.warm_engine_rope(transformer)
    bridge.startup["gc_freeze"] = freeze_long_lived_objects()
    return bridge


_FROZEN = []


def freeze_long_lived_objects() -> dict:
    """Once per worker process, after model load and the attach warm-up: collect, then ``gc.freeze()`` the heap.

    Later full collections then skip the long-lived objects (model, SGLang, indices) instead of scanning them:
    in these workers a forced full collection took ~0.5 s (LongLive 547 ms, RAVEN 469 ms) and 0.0 ms after the
    freeze; such pauses are the likely cause of the 0.77 s restore stall (Step 5b restore_stall diagnosis).
    No numerical effect; objects created later (sessions, archive records) are still collected."""
    import gc
    import time

    if _FROZEN:
        return {**_FROZEN[0], "repeat": True}
    started = time.perf_counter()
    collected = gc.collect()
    collected_seconds = time.perf_counter() - started
    gc.freeze()
    report = {"collected": collected, "collect_seconds": collected_seconds, "frozen_objects": gc.get_freeze_count(),
              "seconds": time.perf_counter() - started}
    _FROZEN.append(report)
    return report


__all__ = [
    "AttentionKVLease", "BridgeNode", "CausalWanKVBridge", "CleanChunkBuilder", "CoverageRange", "LONGLIVE2",
    "LineageSessionArchive", "ModelProfile", "ReelCacheRuntimeConfig", "PROFILES", "RawSessionArchive",
    "SGLangSessionController", "SGLangSessionState", "SessionControlError", "SessionControls", "TokenSpan",
    "VisibilityPlan", "create_sglang_backend", "freeze_long_lived_objects", "profile_for",
    "prompt_change_capability", "register_profile",
]
