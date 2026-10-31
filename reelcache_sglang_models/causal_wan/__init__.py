"""CausVid, LongLive 1.0 and RAVEN as profiles on SGLang's shared causal-Wan path (Step 12).

Pure modules (no SGLang import; importable in ReelCache's CPU suite):

* ``profiles``  - ``ModelProfile`` data: bindings, sampler, visibility, RoPE, recache, CFG and dtypes.
* ``sampling``  - native scheduler/sampler semantics (DiffSynth flow-match scheduler, RAVEN v-lerp step).
* ``recache``   - LongLive 1.0 prompt-switch KV-recache index math and the CacheEpoch-ready record.
* ``convert``   - native -> diffusers conversion (+ LoRA merge) and profile model directories.
* ``lifecycle`` - ReelCache bridge session decisions (resume after restored chunks / prompt switches, recache
                  cache-epoch identity, ``--reelcache-config`` with the bridge profile inline, request cache geometry).

SGLang-side modules (``sglang_*``, ``registration``) import SGLang and are only reached through
``registration.register_causal_wan_profiles()`` (called by ``reelcache_sglang_models.register()`` when SGLang's
``SGLANG_EXTERNAL_MODEL_PACKAGE`` hook imports the package). With the SGLang branch ``reelcache-integration`` and
``--reelcache-enable`` the profile denoising stages attach the ReelCache bridge (``reelcache.integrations.sglang``): pass
``lifecycle.bridge_runtime_config(profile, store_root=...)`` as ``--reelcache-config``. The stages keep their KV caches
across requests, so a session's ``resume`` continues on its resident native KV when no other request wrote it.
"""

from .profiles import CAUSVID, LONGLIVE1, PROFILES, RAVEN, ModelProfile, get_profile, profile_for_model_path

__all__ = ["CAUSVID", "LONGLIVE1", "PROFILES", "RAVEN", "ModelProfile", "get_profile", "profile_for_model_path"]
