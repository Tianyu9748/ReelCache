"""LongCat-Video 13.6B direct-720p continuation port for SGLang Diffusion.

SGLang-free modules (importable without SGLang): :mod:`.spec`, :mod:`.weights`, :mod:`.rope`,
:mod:`.schedule`, :mod:`.conditioning_cache`, :mod:`.lora`, :mod:`.dit_core`, :mod:`.video_io`,
:mod:`.text`. Modules prefixed ``sglang_`` and :mod:`.registration` import SGLang and are only
imported by :func:`reelcache_sglang_models.register`. See ``DESIGN.md``.
"""

from .spec import (  # noqa: F401
    DIRECT_720P,
    DISTILLED_SAMPLER,
    DIT_CLASS_NAME,
    HF_REPO_ID,
    HF_REVISION,
    LONGCAT_VIDEO_ARCH,
    LONGCAT_VIDEO_COMMIT,
    PIPELINE_NAME,
    ContinuationGeometry,
    DistilledSampler,
    LongCatVideoArch,
)
