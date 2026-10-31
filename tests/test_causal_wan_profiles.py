"""Step 12: causal-Wan backbone profiles (CausVid, LongLive 1.0, RAVEN) - pure modules, no SGLang.

Covers ``reelcache_sglang_models.causal_wan`` profiles/sampling/recache/convert: profile data and validation,
exact bindings against feasibility item 9, visibility math, the native schedulers (bitwise against the
original repositories' scheduler code), RAVEN's sampler pieces, the LongLive 1.0 recache plan, the
native->diffusers key mapping, LoRA merge, profile-directory build/verify (tamper detection) and import
isolation. SGLang-side behavior (registration, configs, KV-cache recompute rule) is checked by
``artifacts/sglang_integration/sglang/step12_profiles/sglang_cpu/step12_sglang_cpu_checks.py`` in the CUDA 13.2 env.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reelcache_sglang_models.causal_wan import CAUSVID, LONGLIVE1, PROFILES, RAVEN, get_profile, profile_for_model_path  # noqa: E402
from reelcache_sglang_models.causal_wan import convert as cv  # noqa: E402
from reelcache_sglang_models.causal_wan import lifecycle as lc  # noqa: E402
from reelcache_sglang_models.causal_wan import profiles as pf  # noqa: E402
from reelcache_sglang_models.causal_wan import sampling as sm  # noqa: E402
from reelcache_sglang_models.causal_wan.recache import RecacheRecord, plan_recache  # noqa: E402
from tests.assets import asset  # noqa: E402

#: Test assets: the feasibility item-9 weight-hash receipt (paths under profiles.MODEL_ROOT), CausVid's native
#: modeling/flow_match.py and LongLive 1.0's utils/scheduler.py (upstream commit c6a27e7).
ITEM09 = ("feasibility", "item09.json")
CAUSVID_FLOW_MATCH = ("upstream", "causvid", "modeling", "flow_match.py")
LONGLIVE_SCHEDULER = ("upstream", "LongLive1.0", "utils", "scheduler.py")
PURE = ["profiles", "sampling", "recache", "convert", "lifecycle"]
SGLANG_SIDE = ["sglang_configs", "sglang_cache", "sglang_dit", "sglang_stages", "sglang_pipelines"]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ProfileDataTests(unittest.TestCase):
    def test_three_profiles_on_the_shared_backbone(self):
        self.assertEqual(sorted(PROFILES), ["causvid", "longlive1", "raven"])
        for p in PROFILES.values():
            self.assertEqual(p.chunk_latent_frames, 3)
            self.assertEqual(p.backbone["num_layers"], 30)
            self.assertEqual(p.backbone["num_attention_heads"] * p.backbone["attention_head_dim"], 1536)
            self.assertEqual(p.geometry["tokens_per_latent_frame"], 1560)
            self.assertEqual((p.geometry["height"], p.geometry["width"]), (480, 832))
            self.assertEqual(p.sglang.dit_class, "ReelCacheCausalWanTransformer3DModel")
            self.assertEqual(cv.transformer_config_for(p)["_class_name"], "ReelCacheCausalWanTransformer3DModel")
            self.assertEqual(p.cfg.guidance_scale, 1.0)
            self.assertFalse(p.cfg.negative_branch)
            self.assertEqual(p.rope.capture_key_domain, "pre_rope_normalized_k_at_attention_input")
            self.assertIs(get_profile(p.profile_id), p)
            self.assertIs(profile_for_model_path(p.sglang.model_path), p)
            self.assertIs(profile_for_model_path(p.sglang.model_path + "/"), p)
        with self.assertRaises(KeyError):
            get_profile("longlive2")
        with self.assertRaises(KeyError):
            profile_for_model_path("/data/CausVid")

    def test_sampler_values(self):
        self.assertEqual(CAUSVID.sampler.denoising_steps, (1000, 757, 522))
        self.assertFalse(CAUSVID.sampler.warp_denoising_step)
        self.assertEqual(CAUSVID.sampler.timestep_shift, 8.0)
        self.assertEqual(LONGLIVE1.sampler.denoising_steps, (1000, 750, 500, 250))
        self.assertTrue(LONGLIVE1.sampler.warp_denoising_step)
        self.assertEqual(LONGLIVE1.sampler.timestep_shift, 5.0)
        self.assertEqual(RAVEN.sampler.kind, "raven_consistency_v_lerp")
        self.assertEqual(RAVEN.sampler.noise.step_generator, "raven_fork")
        self.assertEqual(RAVEN.sampler.noise.step_draws, "every_step")
        self.assertEqual(RAVEN.sampler.latent_state_dtype, "float32")
        for p in (CAUSVID, LONGLIVE1):
            self.assertEqual(p.sampler.noise.initial_layout, "B,T,C,H,W")
            self.assertEqual(p.sampler.noise.step_draws, "between_steps")

    def test_dit_precision_policies(self):
        # Self-Forcing lineage runs bf16 end to end natively; RAVEN keeps Wan2.1's fp32 regions (fp32 weights).
        for p in (CAUSVID, LONGLIVE1):
            self.assertEqual((p.dtypes.sglang_dit_precision, p.dtypes.sglang_dit_compute), ("bf16", "self_forcing_bf16"))
            self.assertFalse(p.dtypes.sglang_fp32_regions_tf32)
            self.assertEqual(p.dtypes.native_dit_weights, "bfloat16")
        d = RAVEN.dtypes
        self.assertEqual((d.sglang_dit_precision, d.sglang_dit_compute), ("fp32", "wan_fp32_regions"))
        self.assertTrue(d.sglang_fp32_regions_tf32)  # the native harness sets allow_tf32
        self.assertEqual(d.native_dit_weights, "float32")
        self.assertEqual(pf.DIT_COMPUTE_POLICIES, ("bf16", "self_forcing_bf16", "wan_fp32_regions"))
        with self.assertRaises(ValueError):
            dataclasses.replace(d, sglang_dit_compute="fp16_everything")
        with self.assertRaises(ValueError):  # the fp32 regions need fp32 master weights
            dataclasses.replace(d, sglang_dit_precision="bf16")
        with self.assertRaises(ValueError):  # the native bf16 math needs bf16 weights
            dataclasses.replace(CAUSVID.dtypes, sglang_dit_precision="fp32")
        with self.assertRaises(ValueError):  # TF32 only concerns the fp32 regions
            dataclasses.replace(CAUSVID.dtypes, sglang_fp32_regions_tf32=True)
        self.assertEqual(dataclasses.replace(CAUSVID.dtypes, sglang_dit_compute="bf16").sglang_dit_compute, "bf16")
        self.assertEqual(dataclasses.replace(RAVEN, dtypes=dataclasses.replace(d, sglang_fp32_regions_tf32=False))
                         .binding_fingerprint(), RAVEN.binding_fingerprint())

    def test_visibility_and_recache_policies(self):
        self.assertEqual(CAUSVID.default_visibility, "full_history")
        self.assertEqual({v.name for v in CAUSVID.visibility_policies}, {"full_history", "window_21_frames"})
        self.assertEqual(LONGLIVE1.visibility().window_latent_frames, 12)
        self.assertEqual(LONGLIVE1.visibility().sink_latent_frames, 3)
        self.assertEqual(LONGLIVE1.recache.trigger, "prompt_switch")
        self.assertTrue(LONGLIVE1.recache.global_sink)
        self.assertEqual(LONGLIVE1.recache.recache_latent_frames, 12)
        self.assertIn("c6a27e771e803b1c12ffa28b82ba69b42472020e", LONGLIVE1.recache.upstream)
        self.assertEqual(CAUSVID.recache.trigger, "none")
        self.assertEqual(RAVEN.recache.trigger, "none")
        self.assertEqual(RAVEN.visibility().window_latent_frames, None)
        with self.assertRaises(KeyError):
            RAVEN.visibility("window_21_frames")

    def test_visibility_math(self):
        full, win = CAUSVID.visibility("full_history"), CAUSVID.visibility("window_21_frames")
        self.assertEqual(full.cache_latent_frames(36), 36)
        self.assertEqual(win.cache_latent_frames(36), 21)
        self.assertEqual(full.attended_latent_frames(11, 3), ((0, 36),))
        self.assertEqual(win.attended_latent_frames(6, 3), ((0, 21),))
        self.assertEqual(win.attended_latent_frames(7, 3), ((3, 24),))
        self.assertEqual(win.attended_latent_frames(11, 3), ((15, 36),))
        self.assertEqual([i for i in range(12) if full.is_extrapolated(i, 3)], list(range(7, 12)))
        self.assertFalse(any(win.is_extrapolated(i, 3) for i in range(12)))
        ll = LONGLIVE1.visibility()
        self.assertEqual(ll.cache_latent_frames(39), 12)
        self.assertEqual(ll.attended_latent_frames(2, 3), ((0, 9),))
        self.assertEqual(ll.attended_latent_frames(3, 3), ((0, 12),))
        # chunk c11 (index 10): sink c1 + c9, c10 + itself = 12 frames
        self.assertEqual(ll.attended_latent_frames(10, 3), ((0, 3), (24, 33)))
        with self.assertRaises(ValueError):
            full.cache_latent_frames(0)

    def test_validation_rejects_bad_profiles(self):
        with self.assertRaises(ValueError):
            pf.VisibilityPolicy("full_history", 3, 3)
        with self.assertRaises(ValueError):
            pf.VisibilityPolicy("bogus", None, 0)
        with self.assertRaises(ValueError):
            pf.FileBinding("x", "relative/path", "0" * 64, 1)
        with self.assertRaises(ValueError):
            pf.FileBinding("x", "/abs", "XYZ", 1)
        with self.assertRaises(ValueError):
            dataclasses.replace(CAUSVID, default_visibility="sink_rolling_window")
        with self.assertRaises(ValueError):
            dataclasses.replace(LONGLIVE1, recache=dataclasses.replace(LONGLIVE1.recache, recache_latent_frames=9))
        with self.assertRaises(ValueError):
            dataclasses.replace(CAUSVID, files=CAUSVID.files[1:])  # generator binding removed
        with self.assertRaises(ValueError):
            pf.SamplerSpec(**{**{f.name: getattr(CAUSVID.sampler, f.name) for f in dataclasses.fields(pf.SamplerSpec)},
                              "resolved_timesteps": (522.0, 757.0, 1000.0)})
        with self.assertRaises(ValueError):
            pf.NoiseSpec("B,T,C,H", "bfloat16", "B,T,C,H,W", "bfloat16", "request", "between_steps", "")

    def test_fingerprints(self):
        for p in PROFILES.values():
            d = p.to_dict()
            json.dumps(d)
            self.assertEqual(p.fingerprint(), p.fingerprint())
            self.assertEqual(len(p.fingerprint()), 64)
            self.assertEqual(d["sglang"]["model_path"], p.sglang.model_path)
        self.assertEqual(len({p.fingerprint() for p in PROFILES.values()}), 3)
        self.assertEqual(LONGLIVE1.to_dict()["conversion"]["lora"]["scale"], 1.0)
        noted = dataclasses.replace(CAUSVID, notes=("changed",))
        self.assertNotEqual(noted.fingerprint(), CAUSVID.fingerprint())
        self.assertEqual(noted.binding_fingerprint(), CAUSVID.binding_fingerprint())
        moved = dataclasses.replace(CAUSVID, files=(dataclasses.replace(CAUSVID.files[0], sha256="1" * 64),
                                                    *CAUSVID.files[1:]))
        self.assertNotEqual(moved.binding_fingerprint(), CAUSVID.binding_fingerprint())

    def test_bindings_equal_feasibility_item9(self):
        item = json.loads(asset(*ITEM09).read_text())
        bound = {}
        for model in ("causvid", "longlive1", "raven"):
            for row in item["files_by_model"][model]:
                bound[row["path"]] = (row["sha256"], row["bytes"])
        checked = 0
        for p in PROFILES.values():
            for f in p.files:
                self.assertIn(f.path, bound, f"{p.profile_id}:{f.role}")
                self.assertEqual((f.sha256, f.bytes), bound[f.path], f"{p.profile_id}:{f.role}")
                checked += 1
        self.assertEqual(checked, 26)  # CausVid 9, LongLive 1.0 10, RAVEN 7 (all item-9 required files)
        self.assertTrue(RAVEN.file("generator_checkpoint").path.endswith("/raven_model.pt"))
        self.assertFalse(any("cmgrpo" in f.path for f in RAVEN.files))
        self.assertEqual(LONGLIVE1.file("lora_rank256").state_dict_key, "generator_lora")
        self.assertEqual(LONGLIVE1.conversion.lora.rank, 256)
        self.assertEqual(LONGLIVE1.conversion.lora.alpha, 256.0)
        for p in PROFILES.values():
            self.assertEqual(p.components, pf.DIFFUSERS_COMPONENTS)
            for c in p.components:
                self.assertTrue(c.path.startswith(pf.DIFFUSERS_SNAPSHOT))

    def test_registered_paths_are_exact_and_distinct(self):
        paths = [p.sglang.model_path for p in PROFILES.values()]
        names = [p.sglang.pipeline_name for p in PROFILES.values()]
        self.assertEqual(len(set(paths)), 3)
        self.assertEqual(len(set(names)), 3)
        for p in PROFILES.values():
            base = p.sglang.model_dir_name.lower()
            self.assertTrue(p.sglang.model_path.startswith(pf.PROFILE_MODEL_ROOT + "/"))
            self.assertNotIn("wan2.1-t2v-1.3b", base)  # no SGLang Wan repo short name inside the basename
            self.assertNotIn("longlive-2.0", base)
            self.assertNotIn("wanpipeline", p.sglang.pipeline_name.lower())  # SGLang's Wan detector
            self.assertTrue(p.sglang.pipeline_name.startswith("ReelCache"))

    def test_bridge_profile_fields(self):
        ll = pf.bridge_profile_fields(LONGLIVE1)
        self.assertEqual((ll["visibility"], ll["sink_frames"], ll["window_frames"], ll["recache"]),
                         ("sink_window", 3, 12, "kv_recache_on_prompt_switch"))
        self.assertEqual(ll["transformers"], ())  # shared DiT class: match by exact path only
        self.assertEqual(ll["model_paths"], (LONGLIVE1.sglang.model_path,))
        cv_full = pf.bridge_profile_fields(CAUSVID)
        self.assertEqual((cv_full["visibility"], cv_full["window_frames"], cv_full["name"]), ("full_history", None, "causvid"))
        cv_win = pf.bridge_profile_fields(CAUSVID, "window_21_frames")
        self.assertEqual((cv_win["visibility"], cv_win["window_frames"], cv_win["name"]),
                         ("sink_window", 21, "causvid:window_21_frames"))
        rv = pf.bridge_profile_fields(RAVEN)
        self.assertEqual((rv["num_layers"], rv["num_heads"], rv["head_dim"], rv["chunk_frames"], rv["native_dtype"]),
                         (30, 12, 128, 3, "bfloat16"))
        json.dumps(rv)


class SamplingTests(unittest.TestCase):
    def test_flow_match_scheduler_equals_native_causvid(self):
        native = _load_module(asset(*CAUSVID_FLOW_MATCH), "_s12_cv_flow_match").FlowMatchScheduler(
            shift=8.0, sigma_min=0.0, extra_one_step=True)
        native.set_timesteps(1000, training=True)
        ours = sm.scheduler_for(CAUSVID)
        self.assertTrue(torch.equal(native.sigmas, ours.sigmas))
        self.assertTrue(torch.equal(native.timesteps, ours.timesteps))
        g = torch.Generator().manual_seed(0)
        x = torch.randn(3, 16, 6, 8, generator=g).to(torch.bfloat16)
        n = torch.randn(3, 16, 6, 8, generator=g).to(torch.bfloat16)
        for t in (757, 522):
            tt = t * torch.ones(3, dtype=torch.long)
            self.assertTrue(torch.equal(native.add_noise(x, n, tt), ours.add_noise(x, n, tt)))
        self.assertEqual(sm.resolve_dmd_timesteps(CAUSVID.sampler, ours).tolist(), [1000, 757, 522])

    def test_flow_match_scheduler_equals_upstream_longlive(self):
        mod = _load_module(asset(*LONGLIVE_SCHEDULER), "_s12_ll_scheduler")
        native = mod.FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        native.set_timesteps(1000, training=True)
        ours = sm.scheduler_for(LONGLIVE1)
        self.assertTrue(torch.equal(native.sigmas, ours.sigmas))
        self.assertTrue(torch.equal(native.timesteps, ours.timesteps))
        warped = sm.resolve_dmd_timesteps(LONGLIVE1.sampler, ours)
        # upstream CausalInferencePipeline.__init__ warp
        steps = torch.tensor([1000, 750, 500, 250], dtype=torch.long)
        table = torch.cat((native.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
        self.assertTrue(torch.equal(warped, table[1000 - steps]))
        for got, want in zip(warped.tolist(), LONGLIVE1.sampler.resolved_timesteps):
            self.assertAlmostEqual(got, want, places=3)

    def test_raven_timesteps_rng_and_step(self):
        self.assertEqual(sm.raven_trailing_timesteps(4, 5.0), list(RAVEN.sampler.denoising_steps))
        self.assertEqual(sm.raven_trailing_timesteps(4, 5.0), [1000, 938, 833, 625])
        self.assertEqual(sm.yield_seed(0), 12345)
        self.assertEqual(sm.yield_seed(20001019), (1103515245 * 20001019 + 12345) % 2 ** 31)
        g = torch.Generator().manual_seed(1)
        x0 = torch.randn(16, 3, 4, 6, generator=g)
        eps = torch.randn(16, 3, 4, 6, generator=g)
        xt = sm.raven_forward(x0, eps, 833)
        v = eps - x0  # v_lerp target
        self.assertTrue(torch.allclose(sm.raven_x0(v, xt, 833), x0, atol=1e-5))
        self.assertTrue(torch.equal(sm.raven_forward(x0, eps, 0), x0))  # last step: s = 0 returns x0 exactly
        self.assertEqual(sm.raven_x0(v, xt, 833).dtype, torch.float32)


class RecachePlanTests(unittest.TestCase):
    FS = 1560

    def plan(self, frames_done, global_sink=True, local_end_frames=None):
        local_end = (local_end_frames if local_end_frames is not None else min(12, frames_done)) * self.FS
        return plan_recache(current_start_frame=frames_done, local_attn_frames=12, sink_frames=3, frame_seq=self.FS,
                            global_end_tokens=frames_done * self.FS, local_end_tokens=local_end,
                            global_sink=global_sink)

    def test_report_example_switch_before_c11(self):
        p = self.plan(30)
        self.assertEqual((p.num_recache_frames, p.recache_start_frame), (12, 18))
        self.assertTrue(p.is_recompute)
        self.assertEqual(p.written_global_frames, (21, 30))  # c8-c10
        self.assertEqual(p.discarded_global_frames, (18, 21))  # c7 recomputed, not written
        self.assertEqual(p.kept_sink_slots, (0, 3 * self.FS))
        self.assertEqual(p.written_slots, (3 * self.FS, 12 * self.FS))
        self.assertEqual(p.recompute_tokens, (18 * self.FS, 30 * self.FS))
        self.assertFalse(p.zero_kv_before)
        rec = RecacheRecord(profile_id="longlive1", switch_before_chunk=10, previous_segment=0, new_segment=1,
                            plan=p.to_dict(), context_timestep=0, cross_attention_reset=True)
        ep = rec.to_dict()["cache_epoch"]
        self.assertEqual(ep["rematerialized_latent_frames"], [21, 30])
        self.assertEqual(ep["rematerialized_tokens"], [21 * self.FS, 30 * self.FS])
        self.assertEqual(ep["segments"], [0, 1])
        self.assertTrue(ep["cause"].startswith("prompt_switch"))
        json.dumps(rec.to_dict())

    def test_without_global_sink_the_sink_is_recomputed(self):
        p = self.plan(30, global_sink=False)
        self.assertTrue(p.zero_kv_before)
        self.assertTrue(p.sink_recache)
        self.assertEqual(p.written_global_frames, (18, 30))
        self.assertEqual(p.discarded_global_frames, (18, 18))

    def test_early_switch_and_edge_cases(self):
        self.assertIsNone(self.plan(0))
        p = self.plan(6)  # after two chunks: recompute frames 0-5; current_start 0 -> not a "recompute"
        self.assertEqual((p.num_recache_frames, p.recache_start_frame), (6, 0))
        self.assertFalse(p.is_recompute)
        self.assertEqual(p.written_global_frames, (0, 6))  # upstream rewrites the sink too in this case
        p = self.plan(15)
        self.assertEqual((p.recache_start_frame, p.written_global_frames), (3, (6, 15)))
        with self.assertRaises(ValueError):
            plan_recache(current_start_frame=30, local_attn_frames=12, sink_frames=3, frame_seq=self.FS,
                         global_end_tokens=27 * self.FS, local_end_tokens=12 * self.FS, global_sink=True)
        with self.assertRaises(ValueError):
            self.plan(30, local_end_frames=6)


class LifecycleTests(unittest.TestCase):
    """Phase 2: decisions the profile stages make around the ReelCache bridge (resume, recache epochs, config)."""

    @staticmethod
    def report(chunk_sources: dict[int, str]) -> dict:
        spans = [{"chunk": c, "source": s, "stream": "cond"} for c, s in chunk_sources.items()]
        return {"restored_chunks": 10, "plans": {"cond": {"spans": spans}}}

    def test_resume_plan_without_switches(self):
        plan = lc.resume_plan(num_chunks=12, switch_chunks=[], resume_blocks=4)
        self.assertEqual((plan.segment, plan.restored_switches, plan.recache_before_first_chunk), (0, (), False))
        fresh = lc.resume_plan(num_chunks=13, switch_chunks=[10], resume_blocks=0)
        self.assertEqual((fresh.segment, fresh.restored_switches, fresh.recache_before_first_chunk), (0, (), False))
        with self.assertRaises(ValueError):
            lc.resume_plan(num_chunks=12, switch_chunks=[], resume_blocks=12)
        with self.assertRaises(ValueError):
            lc.resume_plan(num_chunks=12, switch_chunks=[0], resume_blocks=1)

    def test_resume_plan_around_a_prompt_switch(self):
        before = lc.resume_plan(num_chunks=13, switch_chunks=[10], resume_blocks=6)
        self.assertEqual((before.segment, before.restored_switches, before.recache_before_first_chunk), (0, (), False))
        after = lc.resume_plan(num_chunks=13, switch_chunks=[10], resume_blocks=11)
        self.assertEqual((after.segment, after.restored_switches, after.recache_before_first_chunk), (1, (10,), False))
        with self.assertRaises(ValueError):  # at the switch the caller must say whether its epoch was restored
            lc.resume_plan(num_chunks=13, switch_chunks=[10], resume_blocks=10)
        pending = lc.resume_plan(num_chunks=13, switch_chunks=[10], resume_blocks=10, switch_restored=False)
        self.assertEqual((pending.restored_switches, pending.recache_before_first_chunk), ((), True))
        self.assertEqual(pending.segment, 1)  # the first generated chunk runs under the new prompt
        done = lc.resume_plan(num_chunks=13, switch_chunks=[10], resume_blocks=10, switch_restored=True)
        self.assertEqual((done.restored_switches, done.recache_before_first_chunk), ((10,), False))
        two = lc.resume_plan(num_chunks=20, switch_chunks=[5, 10], resume_blocks=10, switch_restored=False)
        self.assertEqual((two.restored_switches, two.recache_before_first_chunk, two.segment), ((5,), True, 2))
        self.assertEqual([lc.segment_of(c, [5, 10]) for c in (0, 4, 5, 9, 10, 19)], [0, 0, 1, 1, 2, 2])

    def test_switch_recache_restored_reads_the_restored_plan(self):
        # LongLive 1.0 at 10 chunks: the next chunk attends to the sink (chunk 0) and chunks 8, 9.
        pre = self.report({0: "chunk", 8: "chunk", 9: "chunk"})
        post = self.report({0: "chunk", 8: "recache", 9: "recache"})
        self.assertIs(lc.switch_recache_restored(pre, 10), False)
        self.assertIs(lc.switch_recache_restored(post, 10), True)
        # consecutive switches: chunk 8 overridden by the previous switch's epoch, chunk 9 is the pre-switch chunk
        self.assertIs(lc.switch_recache_restored(self.report({0: "chunk", 8: "recache", 9: "chunk"}), 10), False)
        self.assertIsNone(lc.switch_recache_restored(None, 10))
        self.assertIsNone(lc.switch_recache_restored({"plans": {}}, 10))
        self.assertIsNone(lc.switch_recache_restored(self.report({0: "chunk"}), 10))
        mixed = {"plans": {"cond": {"spans": [{"chunk": 9, "source": "chunk"}]},
                           "uncond": {"spans": [{"chunk": 9, "source": "recache"}]}}}
        with self.assertRaises(ValueError):
            lc.switch_recache_restored(mixed, 10)

    def test_recache_epoch_identity_and_details(self):
        a, b = lc.recache_epoch_id(10, "prompt B"), lc.recache_epoch_id(10, "prompt C")
        self.assertNotEqual(a, b)
        self.assertEqual(a, lc.recache_epoch_id(10, "prompt B"))
        self.assertTrue(a.startswith("switch-c0010-") and "/" not in a)
        with self.assertRaises(ValueError):
            lc.recache_epoch_id(0, "prompt B")
        p = plan_recache(current_start_frame=30, local_attn_frames=12, sink_frames=3, frame_seq=1560,
                         global_end_tokens=30 * 1560, local_end_tokens=12 * 1560, global_sink=True)
        rec = RecacheRecord(profile_id="longlive1", switch_before_chunk=10, previous_segment=0, new_segment=1,
                            plan=p.to_dict(), context_timestep=0, cross_attention_reset=True)
        details = lc.recache_epoch_details(rec.cache_epoch_fields(), prompt="prompt B", segment=1,
                                           profile_id="longlive1")
        self.assertEqual(details["rematerialized_latent_frames"], [21, 30])  # c8-c10 only; the sink keeps e0
        self.assertEqual(details["kept_sink_slots"], [0, 3 * 1560])
        self.assertEqual(details["prompt_sha256_12"], lc.prompt_digest("prompt B"))
        self.assertEqual(json.loads(json.dumps(details)), details)
        self.assertIsNone(rec.to_dict()["cache_epoch_archive"])

    def test_raven_capture_dtypes_and_matmul_policy(self):
        self.assertEqual(pf.sglang_capture_dtypes(RAVEN), {"K": "float32", "V": "bfloat16"})
        self.assertIsNone(pf.sglang_capture_dtypes(CAUSVID))
        self.assertIsNone(pf.sglang_capture_dtypes(LONGLIVE1))
        self.assertEqual(pf.archive_matmul_precision(RAVEN), "fp32_verified_fp64")
        self.assertIsNone(pf.archive_matmul_precision(CAUSVID))
        plain_fields = pf.bridge_profile_fields(RAVEN)
        extended = pf.bridge_profile_fields(RAVEN, extended=True)
        self.assertNotIn("capture_dtypes", plain_fields)
        self.assertEqual((extended["capture_dtypes"], extended["matmul_precision"]),
                         ({"K": "float32", "V": "bfloat16"}, "fp32_verified_fp64"))
        self.assertEqual({k: v for k, v in extended.items() if k not in ("capture_dtypes", "matmul_precision")},
                         plain_fields)
        self.assertEqual(pf.bridge_profile_fields(CAUSVID, extended=True), pf.bridge_profile_fields(CAUSVID))

    def test_restored_head_kind_decides_the_switch(self):
        post = {**self.report({0: "chunk", 8: "recache", 9: "recache"}), "head_kind": "recache", "head_block": 10}
        pre = {**self.report({0: "chunk", 8: "chunk", 9: "chunk"}), "head_kind": "chunk", "head_block": 9}
        self.assertIs(lc.switch_recache_restored(post, 10), True)
        self.assertIs(lc.switch_recache_restored(pre, 10), False)
        self.assertIs(lc.switch_recache_restored({"head_kind": "recache", "head_block": 10}, 10), True)
        # a recache head whose epoch is only partly on the path (RAW intermediate node): chunk 9 is still pre-switch KV
        self.assertIs(lc.switch_recache_restored({**pre, "head_kind": "recache", "head_block": 10}, 10), False)
        with self.assertRaises(ValueError):  # restored epoch KV without a recache head is inconsistent
            lc.switch_recache_restored({**post, "head_kind": "chunk", "head_block": 9}, 10)

    def test_block_conditioning_ids_follow_the_prompt_segments(self):
        a, b, c = "prompt A", "prompt B", "prompt C"
        single = lc.block_conditioning_ids(a, [], 13)
        self.assertEqual(len(set(single)), 1)
        ab = lc.block_conditioning_ids(a, [{"chunk": 10, "prompt": b}], 13)
        ac = lc.block_conditioning_ids(a, [{"chunk": 10, "prompt": c}], 13)
        self.assertEqual(ab[:10], single[:10])  # the history before the switch is shared (fork from n10 under C)
        self.assertEqual(ab[:10], ac[:10])
        self.assertNotEqual(ab[10], ac[10])  # the switch epoch and later blocks belong to their own prompt
        self.assertEqual(len(set(ab[10:])), 1)
        aa = lc.block_conditioning_ids(a, [{"chunk": 10, "prompt": a}], 13)
        self.assertNotEqual(aa[10], single[10])  # a switch re-materializes KV even under the same prompt
        self.assertTrue(all(isinstance(x, str) and x and "/" not in x for x in ab))
        with self.assertRaises(ValueError):
            lc.block_conditioning_ids(a, [], 0)

    def test_request_cache_geometry_decides_cache_reuse(self):
        """The stage keeps its caches across requests and reuses them (resident session KV) only for the same
        geometry: full history follows the request length, sink + window policies never do."""
        fs = 1560

        def tokens(profile, frames, visibility=None):
            return lc.request_cache_tokens(profile, profile.visibility(visibility), total_frames=frames,
                                           tokens_per_frame=fs)

        self.assertEqual(tokens(CAUSVID, 36), {"cache_tokens": 36 * fs, "sink_tokens": 0,
                                               "attention_window_tokens": 36 * fs})
        self.assertEqual(tokens(CAUSVID, 3)["cache_tokens"], 6 * fs)  # >= 2 chunks: a 1-chunk request writes its KV
        self.assertEqual(tokens(CAUSVID, 15), tokens(CAUSVID, 15))
        self.assertNotEqual(tokens(CAUSVID, 15), tokens(CAUSVID, 36))  # a longer full-history request: new caches
        self.assertEqual(tokens(RAVEN, 15), tokens(CAUSVID, 15))
        self.assertEqual(tokens(LONGLIVE1, 30), {"cache_tokens": 12 * fs, "sink_tokens": 3 * fs,
                                                 "attention_window_tokens": 12 * fs})
        self.assertEqual(tokens(LONGLIVE1, 30), tokens(LONGLIVE1, 39))
        self.assertEqual(tokens(CAUSVID, 36, "window_21_frames"), tokens(CAUSVID, 12, "window_21_frames"))
        self.assertEqual(tokens(CAUSVID, 36, "window_21_frames")["cache_tokens"], 21 * fs)
        with self.assertRaises(ValueError):
            tokens(CAUSVID, 0)
        with self.assertRaises(ValueError):
            lc.request_cache_tokens(CAUSVID, CAUSVID.visibility(), total_frames=3, tokens_per_frame=0)

    def test_bridge_runtime_config_loads_in_the_bridge(self):
        from reelcache.integrations.sglang import ModelProfile as BridgeProfile
        from reelcache.integrations.sglang import ReelCacheRuntimeConfig

        for profile, visibility, window in ((CAUSVID, None, None), (CAUSVID, "window_21_frames", 21),
                                            (LONGLIVE1, None, 12), (RAVEN, None, None)):
            cfg = lc.bridge_runtime_config(profile, store_root="/nonexistent/store", visibility=visibility,
                                           representation="raw", verify_capture_rope=True)
            cfg = json.loads(json.dumps(cfg))
            runtime = ReelCacheRuntimeConfig.from_dict(cfg)
            bp = BridgeProfile.from_dict(runtime.profile_fields)
            self.assertEqual((bp.window_frames, bp.model_paths), (window, (profile.sglang.model_path,)))
            self.assertTrue(bp.matches({"transformer": "ReelCacheCausalWanTransformer3DModel",
                                        "model_path": profile.sglang.model_path}))
            self.assertFalse(bp.matches({"transformer": "ReelCacheCausalWanTransformer3DModel",
                                         "model_path": RAVEN.sglang.model_path if profile is not RAVEN
                                         else CAUSVID.sglang.model_path}))
        ll = BridgeProfile.from_dict(lc.bridge_runtime_config(LONGLIVE1, store_root="/x")["profile_fields"])
        self.assertEqual((ll.visibility, ll.sink_frames, ll.recache), ("sink_window", 3, "kv_recache_on_prompt_switch"))
        # the recache node stages one canvas per re-materialized chunk: the staging pool must hold them all
        lossy = ReelCacheRuntimeConfig.from_dict(lc.bridge_runtime_config(LONGLIVE1, store_root="/x"))
        self.assertEqual(lossy.max_staged_canvases, 4)
        self.assertNotIn("max_staged_canvases", lc.bridge_runtime_config(LONGLIVE1, store_root="/x", representation="raw"))
        self.assertNotIn("max_staged_canvases", lc.bridge_runtime_config(CAUSVID, store_root="/x"))
        self.assertEqual(lc.bridge_runtime_config(LONGLIVE1, store_root="/x", max_staged_canvases=8)["max_staged_canvases"], 8)
        with self.assertRaises(ValueError):
            lc.bridge_runtime_config(CAUSVID, store_root="/x", representation="lossless")
        with self.assertRaises(ValueError):
            lc.bridge_runtime_config(CAUSVID, store_root="/x", profile="causvid")


class ConversionTests(unittest.TestCase):
    @staticmethod
    def native_names(layers=30):
        top = ["patch_embedding.weight", "patch_embedding.bias", "text_embedding.0.weight", "text_embedding.0.bias",
               "text_embedding.2.weight", "text_embedding.2.bias", "time_embedding.0.weight", "time_embedding.0.bias",
               "time_embedding.2.weight", "time_embedding.2.bias", "time_projection.1.weight",
               "time_projection.1.bias", "head.head.weight", "head.head.bias", "head.modulation"]
        blk = ["modulation", "norm3.weight", "norm3.bias"]
        for a in ("self_attn", "cross_attn"):
            for m in "qkvo":
                blk += [f"{a}.{m}.weight", f"{a}.{m}.bias"]
            blk += [f"{a}.norm_q.weight", f"{a}.norm_k.weight"]
        blk += ["ffn.0.weight", "ffn.0.bias", "ffn.2.weight", "ffn.2.bias"]
        return top + [f"blocks.{i}.{n}" for i in range(layers) for n in blk]

    def test_key_mapping(self):
        names = self.native_names()
        self.assertEqual(len(names), 825)
        mapped = [cv.native_to_diffusers_key(n) for n in names]
        self.assertEqual(len(set(mapped)), 825)
        expect = {"blocks.3.self_attn.o.weight": "blocks.3.attn1.to_out.0.weight",
                  "blocks.3.cross_attn.norm_k.weight": "blocks.3.attn2.norm_k.weight",
                  "blocks.0.norm3.bias": "blocks.0.norm2.bias", "blocks.29.ffn.0.weight": "blocks.29.ffn.net.0.proj.weight",
                  "blocks.29.ffn.2.bias": "blocks.29.ffn.net.2.bias", "blocks.7.modulation": "blocks.7.scale_shift_table",
                  "head.modulation": "scale_shift_table", "head.head.weight": "proj_out.weight",
                  "time_projection.1.bias": "condition_embedder.time_proj.bias",
                  "text_embedding.2.weight": "condition_embedder.text_embedder.linear_2.weight"}
        for k, v in expect.items():
            self.assertEqual(cv.native_to_diffusers_key(k), v)
        for bad in ("blocks.0.self_attn.x.weight", "unknown.weight", "blocks.0.norm1.weight"):
            with self.assertRaises(KeyError):
                cv.native_to_diffusers_key(bad)
        self.assertEqual(cv.strip_prefix({"model.a": 1}, "model."), {"a": 1})
        with self.assertRaises(KeyError):
            cv.strip_prefix({"model.a": 1, "b": 2}, "model.")

    def test_lora_merge(self):
        g = torch.Generator().manual_seed(3)
        base, lora = {}, {}
        for t in cv.LORA_TARGETS:
            base[f"blocks.0.{t}.weight"] = torch.randn(6, 5, generator=g)
            lora[f"base_model.model.blocks.0.{t}.lora_A.weight"] = torch.randn(2, 5, generator=g)
            lora[f"base_model.model.blocks.0.{t}.lora_B.weight"] = torch.randn(6, 2, generator=g)
        base["blocks.0.modulation"] = torch.randn(1, 6, 5, generator=g)
        merged, info = cv.merge_lora(base, lora, prefix="base_model.model.", scale=0.5, num_layers=1)
        self.assertEqual(info["targets"], 10)
        for t in cv.LORA_TARGETS:
            w = base[f"blocks.0.{t}.weight"].double()
            a = lora[f"base_model.model.blocks.0.{t}.lora_A.weight"].double()
            b = lora[f"base_model.model.blocks.0.{t}.lora_B.weight"].double()
            self.assertTrue(torch.equal(merged[f"blocks.0.{t}.weight"], (w + 0.5 * (b @ a)).float()))
        self.assertIs(merged["blocks.0.modulation"], base["blocks.0.modulation"])
        bad = dict(lora)
        bad.pop(next(iter(bad)))
        with self.assertRaises(KeyError):
            cv.merge_lora(base, bad, prefix="base_model.model.", scale=1.0, num_layers=1)

    def test_build_and_verify_profile_dir_detects_tampering(self):
        g = torch.Generator().manual_seed(5)
        names = self.native_names(layers=1)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            state = {f"model.{n}": torch.randn(2, 3, generator=g) for n in names}
            src = tmp / "gen.pt"
            torch.save({"generator": state}, src)
            binding = pf.FileBinding("generator_checkpoint", str(src), cv.sha256_file(src), src.stat().st_size,
                                     state_dict_key="generator", key_prefix="model.")
            components = []  # stand-ins for the Diffusers snapshot's files (same roles; verified by size and sha256)
            for i, comp in enumerate(pf.DIFFUSERS_COMPONENTS):
                cfile = tmp / "components" / f"{i}.bin"
                cfile.parent.mkdir(exist_ok=True)
                cfile.write_bytes(comp.role.encode())
                components.append(dataclasses.replace(comp, path=str(cfile), sha256=cv.sha256_file(cfile),
                                                      bytes=cfile.stat().st_size))
            prof = dataclasses.replace(
                CAUSVID, files=(binding,), components=tuple(components),
                conversion=dataclasses.replace(CAUSVID.conversion, expected_tensors=len(names)),
                sglang=dataclasses.replace(CAUSVID.sglang, model_dir_name="unit-test-profile"))
            manifest = cv.build_profile_dir(prof, root=tmp / "models")
            root = tmp / "models" / "unit-test-profile"
            self.assertEqual(manifest["transformer"]["tensors"], len(names))
            self.assertEqual(json.loads((root / "model_index.json").read_text())["_class_name"],
                             "ReelCacheCausVidProfilePipeline")
            report = cv.verify_profile_dir(prof, root, full_hash=True)
            self.assertEqual(report["binding_fingerprint"], prof.binding_fingerprint())
            self.assertEqual(len(report["components"]), len(pf.DIFFUSERS_COMPONENTS))
            # a metadata-only binding change (DiT architecture name) is refreshed without re-conversion
            renamed = dataclasses.replace(prof, sglang=dataclasses.replace(prof.sglang, dit_class="OtherDiT"))
            with self.assertRaises(ValueError):
                cv.verify_profile_dir(renamed, root)
            refreshed = cv.refresh_profile_metadata(renamed, root=tmp / "models")
            self.assertEqual(refreshed["transformer"]["sha256"], manifest["transformer"]["sha256"])
            self.assertEqual(json.loads((root / "transformer/config.json").read_text())["_class_name"], "OtherDiT")
            cv.verify_profile_dir(renamed, root)
            cv.refresh_profile_metadata(prof, root=tmp / "models")
            cv.verify_profile_dir(prof, root)
            with self.assertRaises(FileExistsError):
                cv.build_profile_dir(prof, root=tmp / "models")
            # the source must match its bound hash
            bad_src = dataclasses.replace(prof, files=(dataclasses.replace(binding, sha256="0" * 64),))
            with self.assertRaises(ValueError):
                cv.build_profile_dir(bad_src, root=tmp / "other")
            # another profile's code fingerprint
            with self.assertRaises(ValueError):
                cv.verify_profile_dir(dataclasses.replace(prof, files=(dataclasses.replace(binding, bytes=binding.bytes + 1),)),
                                      root)
            # tampered converted weights
            wfile = root / "transformer" / cv.TRANSFORMER_FILE
            data = bytearray(wfile.read_bytes())
            data[-1] ^= 0xFF
            wfile.write_bytes(bytes(data))
            with self.assertRaises(ValueError):
                cv.verify_profile_dir(prof, root, full_hash=True)
            # a redirected component link
            (root / "vae").unlink()
            (root / "vae").symlink_to(tmp)
            with self.assertRaises(ValueError):
                cv.verify_profile_dir(prof, root, full_hash=False)


class IsolationTests(unittest.TestCase):
    def test_pure_modules_import_without_sglang(self):
        mods = ", ".join(f"reelcache_sglang_models.causal_wan.{m}" for m in PURE)
        code = ("import json, sys\n"
                "import reelcache_sglang_models.causal_wan as cw\n"
                f"import {mods}\n"
                "import reelcache_sglang_models.causal_wan.registration as reg\n"
                "print(json.dumps({'sglang': sorted(m for m in sys.modules if m == 'sglang' or m.startswith('sglang.')),"
                " 'side': sorted(m for m in sys.modules if m.startswith('reelcache_sglang_models.causal_wan.sglang_')),"
                " 'registered': reg.registered()}))\n")
        env = {k: v for k, v in os.environ.items() if k != "SGLANG_EXTERNAL_MODEL_PACKAGE"}
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), env.get("PYTHONPATH", "")) if p)
        env["CUDA_VISIBLE_DEVICES"] = ""
        r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env, capture_output=True, text=True,
                           timeout=900)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        info = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(info, {"sglang": [], "side": [], "registered": None})

    def test_static_import_layout(self):
        pkg = ROOT / "reelcache_sglang_models" / "causal_wan"
        for name in SGLANG_SIDE:
            self.assertRegex((pkg / f"{name}.py").read_text(), r"\nfrom sglang\.multimodal_gen", name)
        for name in PURE + ["__init__"]:
            src = (pkg / f"{name}.py").read_text()
            self.assertNotRegex(src, r"(^|\n)\s*(from|import) sglang", name)
            self.assertNotRegex(src, r"from \.sglang_|import \.sglang_", name)
        reg = (pkg / "registration.py").read_text()
        self.assertNotRegex(reg.split("def register_causal_wan_profiles", 1)[0], r"(from|import) sglang")
        init = (ROOT / "reelcache_sglang_models" / "__init__.py").read_text()
        self.assertIn("from .causal_wan.registration import register_causal_wan_profiles", init)
        for path in (ROOT / "reelcache").rglob("*.py"):
            self.assertIsNone(re.search(r"reelcache_sglang_models\.causal_wan", path.read_text(errors="replace")), str(path))


if __name__ == "__main__":
    unittest.main()
