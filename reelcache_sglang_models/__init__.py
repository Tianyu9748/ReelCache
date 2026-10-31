"""Out-of-tree SGLang Diffusion model package of the ReelCache integration.

SGLang imports this package when ``SGLANG_EXTERNAL_MODEL_PACKAGE=reelcache_sglang_models`` is set
(``runtime/utils/external_model_package.py``); the import then registers the LongCat-Video
continuation pipeline and its DiT, and the causal-Wan profile pipelines (CausVid, LongLive 1.0, RAVEN;
``causal_wan.registration``) with SGLang's registries. Imported anywhere else, the package
registers nothing and does not import SGLang (core ``reelcache`` never imports it). A process that did
not set the variable can call :func:`register` explicitly. Registration failures are raised, never
swallowed.
"""

from __future__ import annotations

import os

EXTERNAL_PACKAGE_ENV = "SGLANG_EXTERNAL_MODEL_PACKAGE"

__all__ = ["EXTERNAL_PACKAGE_ENV", "loaded_by_sglang", "register", "registered"]


def loaded_by_sglang() -> bool:
    """True when SGLang's external-package hook names this package."""
    return os.environ.get(EXTERNAL_PACKAGE_ENV, "").strip() == __name__


def register() -> dict:
    """Register every model of this package with SGLang (imports SGLang; idempotent)."""
    from .causal_wan.registration import register_causal_wan_profiles
    from .longcat_video.registration import register_longcat_video

    return {"longcat_video": register_longcat_video(), "causal_wan": register_causal_wan_profiles()}


def registered() -> dict:
    from .longcat_video import registration

    return {"longcat_video": registration.registered()}


if loaded_by_sglang():
    register()
