"""CPU tests of the LongCat-Video SGLang port (``reelcache_sglang_models.longcat_video``).

They need neither SGLang nor a GPU. The real native LongCat-Video modules (commit 6b3f4b8, at
``REELCACHE_LONGCAT_VIDEO_SOURCE``, default the cluster checkout) are imported on the CPU and their source
files are hash-checked first, so every comparison below is against the exact native code:

* configuration / geometry / buckets / FFN sizing / weight binding;
* 3D RoPE angle tables and rotation (bitwise), segment-local positions;
* distilled sigma schedule, flow-shift timesteps and the Euler step (bitwise);
* prompt cleaning, frame preprocessing, latent (de)normalization, the native decode tail;
* LoRA parsing/application and the real ``cfg_step_lora`` target layout (header only);
* a tiny FP32 DiT: plain, conditioned, cache-build (pre-RoPE K) and cache-read forwards match the
  native ``LongCatVideoTransformer3DModel`` with torch SDPA attention;
* the conditioning-cache seam (epochs, layouts, registry, restore validation).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tests.assets import asset
from reelcache_sglang_models.longcat_video import (
    conditioning_cache as cc,
    dit_core,
    lora,
    rope,
    schedule,
    spec,
    text,
    video_io,
    weights,
)

ROOT = Path(__file__).resolve().parents[1]


def native_root() -> Path:
    """The native LongCat-Video source at commit 6b3f4b8 (a test asset)."""
    return asset("longcat", "source", override="REELCACHE_LONGCAT_VIDEO_SOURCE")


def checkpoint() -> Path:
    """The LongCat-Video checkpoint directory (a test asset)."""
    return asset("longcat", "weights", override="REELCACHE_LONGCAT_VIDEO_WEIGHTS")


NATIVE_SOURCE_SHA256 = {
    "longcat_video/pipeline_longcat_video.py": "fdcb1ff211744101b8f505830d8b17adb59bab98651c6ba8a44a1ff706ad54c1",
    "longcat_video/modules/attention.py": "d2d29f468c2b1e998adf39e740cf271e1d8a64b7fda7efb730fc8fcdbee3bd4b",
    "longcat_video/modules/rope_3d.py": "cca72c408d49bbb8abf9dc17784dfed4887c88a87874750a798a61ddeea6661c",
    "longcat_video/modules/blocks.py": "596f40ba74cb429a0d05987bc75beac9b43a096cbdc37474628d165cb6ccf152",
    "longcat_video/modules/longcat_video_dit.py": "cebb2ed2cfc754b4501a36bd4e57969c5fd47f4bf1ce352587dcfac092d4d1e5",
    "longcat_video/modules/lora_utils.py": "fba59cd7591b1597ebb2c0517b5000273562cf1574bf0893c7a723feeab849fd",
    "longcat_video/modules/scheduling_flow_match_euler_discrete.py":
        "322cef12d5efa660f839fc173e83eba68b913f031b3f06643d71e5181279be5a",
    "longcat_video/utils/bukcet_config.py": "9f2d2ae584863f9eaee9a8b6fcce80bed2c1415ef5570de62bb8c99f5547dd0b",
}
TINY = spec.LongCatVideoArch(hidden_size=64, depth=2, num_heads=4, caption_channels=32, adaln_tembed_dim=16,
                             frequency_embedding_size=32, mlp_ratio=4)

_NATIVE: dict = {}


def native():
    """Import the native modules once (after the source hash check)."""
    if not _NATIVE:
        NATIVE_ROOT = native_root()
        missing = [rel for rel in NATIVE_SOURCE_SHA256 if not (NATIVE_ROOT / rel).is_file()]
        if missing:
            raise AssertionError(f"native LongCat-Video source missing under {NATIVE_ROOT}: {missing}")
        for rel, digest in NATIVE_SOURCE_SHA256.items():
            actual = hashlib.sha256((NATIVE_ROOT / rel).read_bytes()).hexdigest()
            if actual != digest:
                raise AssertionError(f"native {rel} changed: {actual} != {digest}")
        if str(NATIVE_ROOT) not in sys.path:
            sys.path.insert(0, str(NATIVE_ROOT))
        from longcat_video import pipeline_longcat_video as pipe_mod
        from longcat_video.modules import attention, blocks, longcat_video_dit, lora_utils, rope_3d
        from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        from longcat_video.utils import bukcet_config

        _NATIVE.update(pipe=pipe_mod, attention=attention, blocks=blocks, dit=longcat_video_dit,
                       lora_utils=lora_utils, rope=rope_3d, scheduler=FlowMatchEulerDiscreteScheduler,
                       buckets=bukcet_config)
    return _NATIVE


def _native_pipeline_shell():
    pipe_cls = native()["pipe"].LongCatVideoPipeline
    shell = pipe_cls.__new__(pipe_cls)
    shell._num_timesteps = 1000
    shell._num_distill_sample_steps = 50
    shell.vae = types.SimpleNamespace(config=types.SimpleNamespace(
        latents_mean=list(spec.VAE_LATENTS_MEAN), latents_std=list(spec.VAE_LATENTS_STD), z_dim=spec.VAE_Z_DIM))
    return shell


def _install_native_sdpa(model):
    """Replace flash-attn calls by torch SDPA (as the M06 grid runtime does when flash-attn is unusable)."""
    attention = native()["attention"]

    def self_attn(self, q, k, v, shape):
        return F.scaled_dot_product_attention(q, k, v, scale=self.scale)

    def cross_attn(self, x, cond, kv_seqlen):
        b, n, c = x.shape
        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        outs, start = [], 0
        for i in range(b):
            qb = q[:, i * n:(i + 1) * n].transpose(1, 2)
            kb = k[:, start:start + kv_seqlen[i]].transpose(1, 2)
            vb = v[:, start:start + kv_seqlen[i]].transpose(1, 2)
            start += kv_seqlen[i]
            outs.append(F.scaled_dot_product_attention(qb, kb, vb).transpose(1, 2))
        return self.proj(torch.cat(outs, dim=1).reshape(b, -1, c))

    for module in model.modules():
        if isinstance(module, attention.Attention):
            module._process_attn = types.MethodType(self_attn, module)
        elif isinstance(module, attention.MultiHeadCrossAttention):
            module._process_cross_attn = types.MethodType(cross_attn, module)


def _tiny_native_and_port(seed: int = 0):
    torch.manual_seed(seed)
    nat = native()["dit"].LongCatVideoTransformer3DModel(
        in_channels=16, out_channels=16, hidden_size=TINY.hidden_size, depth=TINY.depth, num_heads=TINY.num_heads,
        caption_channels=TINY.caption_channels, mlp_ratio=TINY.mlp_ratio, adaln_tembed_dim=TINY.adaln_tembed_dim,
        frequency_embedding_size=TINY.frequency_embedding_size, patch_size=(1, 2, 2), cp_split_hw=[1, 1],
        text_tokens_zero_pad=True)
    with torch.no_grad():  # non-trivial norm weights/biases so every parameter matters
        for name, p in nat.named_parameters():
            p.copy_(torch.randn_like(p) * (0.2 if p.ndim > 1 else 0.5) + (1.0 if name.endswith("norm.weight") else 0))
    nat.eval()
    _install_native_sdpa(nat)
    port = dit_core.LongCatVideoDiTCore(TINY).eval()
    port.load_state_dict(nat.state_dict(), strict=True)
    return nat, port


def _synthetic_lora_state(model, rank: int = 4, seed: int = 1) -> dict:
    """Native-format LoRA state dict for the qkv/proj/cross/ffn linears of ``model`` (random values)."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for i, block in enumerate(model.blocks):
        targets = {"attn.qkv": 3, "attn.proj": 1, "cross_attn.q_linear": 1, "cross_attn.kv_linear": 2,
                   "ffn.w1": 1, "ffn.w2": 1, "ffn.w3": 1}
        for path, n_sep in targets.items():
            mod = block
            for part in path.split("."):
                mod = getattr(mod, part)
            name = "lora___lorahyphen___" + f"blocks.{i}.{path}".replace(".", "___lorahyphen___")
            out[f"{name}.lora_down.weight"] = torch.randn(n_sep * rank, mod.in_features, generator=gen) * 0.1
            if n_sep == 1:
                out[f"{name}.lora_up.weight"] = torch.randn(mod.out_features, rank, generator=gen) * 0.1
            else:
                for j in range(n_sep):
                    out[f"{name}.lora_up.blocks.{j}.weight"] = torch.randn(mod.out_features // n_sep, rank,
                                                                            generator=gen) * 0.1
            out[f"{name}.alpha_scale"] = torch.tensor(0.5)
    return out


class NativeBindingTests(unittest.TestCase):
    def test_native_sources_are_the_pinned_commit(self):
        mods = native()
        self.assertIn("LongCatVideoPipeline", dir(mods["pipe"]))
        self.assertEqual(spec.LONGCAT_VIDEO_COMMIT, "6b3f4b8582a8bc3f20f795735f5383716c4ba794")


class SpecGeometryTests(unittest.TestCase):
    def test_arch_matches_checkpoint_config(self):
        cfg = json.loads((checkpoint() / "dit" / "config.json").read_text())
        arch = spec.LongCatVideoArch.from_config_dict(cfg)
        self.assertEqual(arch, spec.LONGCAT_VIDEO_ARCH)
        self.assertEqual((arch.head_dim, arch.rope_axis_dims, arch.patch_numel), (128, (44, 42, 42), 4))
        with torch.device("meta"):
            ffn = native()["blocks"].FeedForwardSwiGLU(dim=4096, hidden_dim=int(4096 * 4))
        self.assertEqual(ffn.hidden_dim, arch.ffn_hidden_dim)
        self.assertEqual(arch.ffn_hidden_dim, 11008)
        with self.assertRaises(ValueError):
            spec.LongCatVideoArch.from_config_dict({**cfg, "enable_bsa": True})
        with self.assertRaises(ValueError):
            spec.LongCatVideoArch.from_config_dict({**cfg, "cp_split_hw": [1, 2]})
        with self.assertRaises(ValueError):
            spec.LongCatVideoArch.from_config_dict({**cfg, "unknown_key": 1})

    def test_bucket_tables_and_lookup_match_native(self):
        b = native()["buckets"]
        self.assertEqual(spec.ASPECT_RATIO_627, b.ASPECT_RATIO_627)
        self.assertEqual(spec.ASPECT_RATIO_960, b.ASPECT_RATIO_960)
        self.assertEqual(list(spec.ASPECT_RATIO_960), list(b.ASPECT_RATIO_960))  # order decides ties
        pipe_cls = native()["pipe"].LongCatVideoPipeline
        sizes = [(704, 1312), (720, 1280), (480, 832), (1080, 1920), (1312, 704), (512, 512), (300, 1600),
                 (1000, 999), (600, 1200)]
        for res in ("480p", "720p"):
            for h, w in sizes:
                frame = types.SimpleNamespace(height=h, width=w)
                expect = pipe_cls.get_condition_shape(None, [frame], res, scale_factor_spatial=16)
                key, th, tw = spec.bucket_for_condition(h, w, res)
                self.assertEqual((th, tw), tuple(expect), (res, h, w))
                self.assertEqual(tuple(spec.BUCKETS[res][key][0]), tuple(expect))

    def test_direct_720p_geometry(self):
        g = spec.DIRECT_720P
        self.assertEqual(spec.bucket_for_condition(704, 1312, "720p"), ("0.54", 704, 1312))
        self.assertEqual((g.latent_frames, g.cond_latent_frames, g.new_latent_frames), (24, 4, 20))
        self.assertEqual((g.latent_hw, g.token_hw, g.tokens_per_latent_frame), ((88, 164), (44, 82), 3608))
        self.assertEqual((g.cond_tokens, g.new_tokens, g.new_rgb_frames), (14432, 72160, 80))
        self.assertEqual(g.cache_bytes(), 11349786624)  # feasibility item 10: 11.35 GB BF16
        self.assertEqual(g.latent_shape, (1, 16, 24, 88, 164))
        self.assertEqual(g.cond_token_grid, (4, 44, 82))
        with self.assertRaises(ValueError):
            spec.ContinuationGeometry(num_frames=92)
        with self.assertRaises(ValueError):
            spec.ContinuationGeometry(height=700)
        with self.assertRaises(ValueError):
            spec.DistilledSampler(guidance_scale=4.0)
        with self.assertRaises(ValueError):
            spec.DistilledSampler(use_distill=False)


class WeightBindingTests(unittest.TestCase):
    def test_binding_matches_feasibility_receipt(self):
        table = asset("feasibility", "weights_sha256_table.csv")  # paths under weights.DEFAULT_CHECKPOINT_DIR
        import csv

        with table.open() as fh:
            rows = [r for r in csv.DictReader(fh) if r["model"] == "longcat720" and r["required"] == "yes"]
        receipt = {str(Path(r["path"]).relative_to(weights.DEFAULT_CHECKPOINT_DIR)): (int(r["size_bytes"]), r["sha256"])
                   for r in rows}
        bound = {f.relpath: (f.size, f.sha256) for f in weights.LONGCAT_VIDEO_FILES}
        self.assertEqual(bound, receipt)
        self.assertNotIn("lora/refinement_lora.safetensors", bound)
        self.assertEqual(weights.MODEL_FINGERPRINT, weights.binding_fingerprint())

    def test_stat_verification_of_real_checkpoint_and_tamper_detection(self):
        report = weights.verify_checkpoint(checkpoint(), mode="stat")
        self.assertEqual(report["files_checked"], len(weights.LONGCAT_VIDEO_FILES))
        self.assertTrue(report["weaker_stat_mode"])
        self.assertGreaterEqual(report["files_hashed"], 10)
        self.assertEqual(report["model_fingerprint"], weights.MODEL_FINGERPRINT)
        config_entry = tuple(f for f in weights.LONGCAT_VIDEO_FILES if f.relpath == "dit/config.json")
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp)
            (fake / "dit").mkdir()
            original = (checkpoint() / "dit" / "config.json").read_bytes()
            (fake / "dit" / "config.json").write_bytes(original)
            self.assertEqual(weights.verify_checkpoint(fake, mode="full", files=config_entry)["files_hashed"], 1)
            with self.assertRaisesRegex(weights.WeightBindingError, "missing"):
                weights.verify_checkpoint(fake, mode="stat")
            tampered = bytearray(original)
            tampered[10] ^= 1
            (fake / "dit" / "config.json").write_bytes(bytes(tampered))
            with self.assertRaisesRegex(weights.WeightBindingError, "sha256 mismatch dit/config.json"):
                weights.verify_checkpoint(fake, mode="full", files=config_entry)
            (fake / "dit" / "config.json").write_bytes(original + b" ")
            with self.assertRaisesRegex(weights.WeightBindingError, "size mismatch"):
                weights.verify_checkpoint(fake, mode="stat", files=config_entry)
        with self.assertRaises(weights.WeightBindingError):
            weights.resolve_verify_mode("fast")


class RopeTests(unittest.TestCase):
    def test_angle_tables_bitwise_equal_native(self):
        rope_mod = native()["rope"]
        for head_dim, grid in ((128, (4, 44, 82)), (128, (24, 44, 82)), (128, (3, 5, 7)), (64, (2, 3, 4))):
            nat = rope_mod.RotaryPositionalEmbedding(head_dim, cp_split_hw=[1, 1]).precompute_freqs_cis_3d(grid)
            ours = rope.rope_angles_3d(grid, head_dim)
            self.assertEqual(ours.dtype, torch.float32)
            self.assertTrue(torch.equal(ours, nat), (head_dim, grid))

    def test_rotation_bitwise_equal_native(self):
        rope_mod = native()["rope"]
        grid, head_dim, heads = (3, 4, 6), 128, 2
        n = grid[0] * grid[1] * grid[2]
        tables = rope.RopeTables(head_dim)
        cos, sin = tables.cos_sin(grid, "cpu")
        torch.manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            q = torch.randn(1, heads, n, head_dim).to(dtype)
            k = torch.randn(1, heads, n, head_dim).to(dtype)
            nq, nk = rope_mod.RotaryPositionalEmbedding(head_dim, cp_split_hw=[1, 1])(q, k, grid)
            oq = rope.apply_rope(q.transpose(1, 2), cos, sin).transpose(1, 2)
            ok = rope.apply_rope(k, cos, sin, seq_dim=2)
            self.assertTrue(torch.equal(oq, nq) and torch.equal(ok, nk), dtype)
            self.assertEqual(oq.dtype, dtype)

    def test_segment_local_positions(self):
        full = rope.rope_angles_3d((24, 44, 82))
        cache = rope.rope_angles_3d((4, 44, 82))
        self.assertTrue(torch.equal(full[: cache.shape[0]], cache))
        tables = rope.RopeTables()
        c_rows, s_rows = tables.rows((24, 44, 82), "cpu", 4, 20)
        c_full, _ = tables.cos_sin((24, 44, 82), "cpu")
        self.assertTrue(torch.equal(c_rows, c_full[4 * 3608:]))
        pos = rope.segment_positions(4, 20)
        self.assertEqual((pos["cache_frames"], pos["new_frames"][0], pos["rope_grid_frames"]), ([0, 1, 2, 3], 4, 24))
        with self.assertRaises(ValueError):
            tables.rows((24, 44, 82), "cpu", 20, 5)


class ScheduleTests(unittest.TestCase):
    def test_distill_sigmas_and_timesteps_bitwise_equal_native(self):
        shell = _native_pipeline_shell()
        cfg = json.loads((checkpoint() / "scheduler" / "scheduler_config.json").read_text())
        for steps in (16, 4, 50):
            nat_sigmas = shell.get_timesteps_sigmas(steps, use_distill=True)
            self.assertTrue(torch.equal(schedule.distill_sigmas(steps), nat_sigmas))
            sched = native()["scheduler"].from_config(cfg)
            sched.set_timesteps(steps, sigmas=nat_sigmas, device="cpu")
            ts, sg = schedule.continuation_schedule(steps, shift=float(cfg["shift"]))
            self.assertTrue(torch.equal(ts, sched.timesteps) and torch.equal(sg, sched.sigmas), steps)
        self.assertEqual(float(cfg["shift"]), spec.DISTILLED_SAMPLER.flow_shift)

    def test_euler_step_and_timestep_tensor_match_native(self):
        cfg = json.loads((checkpoint() / "scheduler" / "scheduler_config.json").read_text())
        sched = native()["scheduler"].from_config(cfg)
        sigmas = schedule.distill_sigmas(16)
        sched.set_timesteps(16, sigmas=sigmas, device="cpu")
        ts, sg = schedule.continuation_schedule(16)
        torch.manual_seed(0)
        sample = torch.randn(1, 16, 3, 4, 6).to(torch.bfloat16)
        for i, t in enumerate(sched.timesteps[:3]):
            out = -torch.randn(1, 16, 3, 4, 6)
            ref = sched.step(out, t, sample, return_dict=False)[0]
            ours = schedule.euler_step(out, sample, sg[i], sg[i + 1])
            self.assertTrue(torch.equal(ours, ref))
            self.assertEqual(ours.dtype, torch.float32)
            sample = ref
        t = ts[3]
        native_t = t.expand(1).to(torch.bfloat16).unsqueeze(-1).repeat(1, 20)
        self.assertTrue(torch.equal(schedule.dit_timestep(t, 1, 20, torch.bfloat16), native_t))


class TextAndVideoIOTests(unittest.TestCase):
    def test_prompt_clean_matches_native(self):
        pipe = native()["pipe"]
        for s in ["a boat accelerating to gain speed", "  A tranquil tableau\n of a  bunch of grapes ",
                  "caf&amp;eacute; &lt;b&gt;bold&lt;/b&gt;", "mojibake: cafÃ©   \t tabs", "ｆｕｌｌ width\u3000space"]:
            self.assertEqual(text.prompt_clean(s), pipe.prompt_clean(s), s)
        kw = text.tokenizer_kwargs()
        self.assertEqual((kw["max_length"], kw["padding"], kw["truncation"]), (512, "max_length", True))

    def test_frame_preprocessing_matches_diffusers_video_processor(self):
        from diffusers.video_processor import VideoProcessor
        from PIL import Image

        rng = np.random.default_rng(0)
        proc = VideoProcessor(vae_scale_factor=8)
        for (h, w), (th, tw) in (((32, 48), (32, 48)), ((30, 50), (32, 48))):
            frames = rng.integers(0, 256, size=(5, h, w, 3), dtype=np.uint8)
            ref = proc.preprocess_video([Image.fromarray(f) for f in frames], height=th, width=tw)
            ours = video_io.frames_to_vae_input(frames, th, tw)
            self.assertEqual(tuple(ours.shape), (1, 3, 5, th, tw))
            self.assertTrue(torch.equal(ours, ref), (h, w))
        with self.assertRaises(ValueError):
            video_io.load_condition_frames(np.zeros((2, 4, 4, 3), dtype=np.float32))

    def test_latent_normalization_and_decode_tail_match_native(self):
        shell = _native_pipeline_shell()
        torch.manual_seed(0)
        for dtype in (torch.bfloat16, torch.float32):
            lat = torch.randn(1, 16, 3, 4, 6).to(dtype)
            self.assertTrue(torch.equal(video_io.normalize_latents(lat), shell.normalize_latents(lat)))
            self.assertTrue(torch.equal(video_io.denormalize_latents(lat), shell.denormalize_latents(lat)))
        from diffusers.video_processor import VideoProcessor

        video = (torch.rand(1, 3, 5, 8, 12) * 2.2 - 1.1).clamp(-1, 1).to(torch.bfloat16)
        ref = VideoProcessor(vae_scale_factor=8).postprocess_video(video, output_type="np")
        ref_u8 = (ref[0] * 255).astype(np.uint8)
        ours = video_io.postprocess_decoded(video)
        self.assertTrue(np.array_equal(ours[0].numpy(), ref[0]))
        self.assertTrue(np.array_equal(video_io.to_uint8_native(ours[0]), ref_u8))
        self.assertTrue(np.array_equal((ours[0] * 255).to(torch.uint8).numpy(), ref_u8))


class LoRATests(unittest.TestCase):
    def test_parse_and_apply_match_native_lora(self):
        nat, port = _tiny_native_and_port()
        state = _synthetic_lora_state(port)
        with tempfile.TemporaryDirectory() as tmp:
            from safetensors.torch import save_file

            path = os.path.join(tmp, "lora.safetensors")
            save_file(state, path)
            nat.load_lora(path, "cfg_step_lora", multiplier=1.0, lora_network_dim=4, lora_network_alpha=2)
        nat.enable_loras(["cfg_step_lora"])
        report = lora.attach_lora(port, state, multiplier=1.0)
        self.assertEqual(report["modules"], 2 * 7)
        self.assertEqual(report["alpha_scales"], [0.5])
        torch.manual_seed(3)
        x = torch.randn(1, 16, 2, 4, 6)
        t = torch.tensor([700.0])
        y = torch.randn(1, 1, 7, TINY.caption_channels)
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])
        with torch.no_grad():
            self.assertTrue(torch.equal(port(x, t, y, mask), nat(x, t, y, mask)))
            lora.set_lora_enabled(port, False)
            nat.disable_all_loras()
            self.assertTrue(torch.equal(port(x, t, y, mask), nat(x, t, y, mask)))
        with self.assertRaises(ValueError):
            lora.parse_lora_state_dict({**state, "stray.weight": torch.zeros(1)})
        with self.assertRaises(ValueError):
            lora.attach_lora(port, state)  # already attached

    def test_real_cfg_step_lora_targets_fit_the_full_model(self):
        path = checkpoint() / weights.LORA_RELPATH
        with path.open("rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        header.pop("__metadata__", None)
        meta_state = {k: torch.empty(v["shape"], dtype=torch.float32, device="meta") for k, v in header.items()}
        entries = lora.parse_lora_state_dict(meta_state)
        self.assertEqual(len(entries), 48 * 7)
        with torch.device("meta"):
            full = dit_core.LongCatVideoDiTCore(spec.LONGCAT_VIDEO_ARCH)
        modules = dict(full.named_modules())
        splits = {}
        for name, entry in entries.items():
            target = modules[name]
            self.assertIsInstance(target, lora.LoRALinear)
            self.assertEqual(entry.down.shape[1], target.in_features)
            self.assertEqual(sum(u.shape[0] for u in entry.ups), target.out_features)
            splits[name.split(".", 2)[2]] = len(entry.ups)
        self.assertEqual(splits, {"attn.qkv": 3, "attn.proj": 1, "cross_attn.q_linear": 1, "cross_attn.kv_linear": 2,
                                  "ffn.w1": 1, "ffn.w2": 1, "ffn.w3": 1})


class OverlappedCacheTests(unittest.TestCase):
    """Layer-ahead reads of a conditioning cache that is still being written (``LayerReadiness``)."""

    def test_forward_waits_per_layer_and_matches_the_complete_cache(self):
        import threading
        import time as _time

        _, port = _tiny_native_and_port()
        torch.manual_seed(5)
        latents = torch.randn(1, 16, 6, 8, 12)
        text = torch.randn(1, 1, 9, TINY.caption_channels)
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0]])
        cond, new = latents[:, :, :2], latents[:, :, 2:]
        t = torch.tensor([931.0]).expand(1).unsqueeze(-1).repeat(1, 4)
        with torch.no_grad():
            keys, values = port.build_conditioning_kv(cond)
            reference = port.forward_with_conditioning(new, t, text, mask, list(zip(keys, values)), 2)
            pending_k = [torch.zeros_like(k) for k in keys]
            pending_v = [torch.zeros_like(v) for v in values]
            ready = cc.LayerReadiness(TINY.depth)
            order = []

            def producer():
                for i in range(TINY.depth):
                    _time.sleep(0.05)
                    pending_k[i].copy_(keys[i])
                    pending_v[i].copy_(values[i])
                    order.append(("written", i))
                    ready.mark(i)

            thread = threading.Thread(target=producer)
            ready.attach(thread)
            thread.start()
            out = port.forward_with_conditioning(new, t, text, mask, list(zip(pending_k, pending_v)), 2,
                                                 kv_cache_ready=ready)
            ready.join()
        self.assertTrue(torch.equal(out, reference))
        self.assertEqual(order, [("written", i) for i in range(TINY.depth)])

    def test_a_failed_restore_raises_instead_of_reading_unwritten_layers(self):
        _, port = _tiny_native_and_port()
        torch.manual_seed(6)
        latents = torch.randn(1, 16, 6, 8, 12)
        text = torch.randn(1, 1, 9, TINY.caption_channels)
        cond, new = latents[:, :, :2], latents[:, :, 2:]
        t = torch.tensor([500.0]).expand(1).unsqueeze(-1).repeat(1, 4)
        with torch.no_grad():
            keys, values = port.build_conditioning_kv(cond)
            ready = cc.LayerReadiness(TINY.depth)
            ready.mark(0)
            ready.fail(RuntimeError("decode failed at layer 1"))
            with self.assertRaisesRegex(cc.ConditioningCacheError, "overlapped conditioning-cache restore failed"):
                port.forward_with_conditioning(new, t, text, None, list(zip(keys, values)), 2, kv_cache_ready=ready)
            with self.assertRaises(cc.ConditioningCacheError):
                ready.join()
            slow = cc.LayerReadiness(TINY.depth, timeout=0.05)
            with self.assertRaisesRegex(cc.ConditioningCacheError, "was not restored within"):
                slow.wait(0)
            with self.assertRaises(ValueError):
                port.forward_with_conditioning(new, t, text, None, None, 2, kv_cache_ready=slow)

    def test_a_restore_that_found_damage_raises_a_typed_integrity_error(self):
        """A payload found corrupt after step 0 started (asynchronous verification, freeze D): waiters raise
        ``IntegrityError`` from the producer's failure (rollback and ``integrity`` code as in the decode-failure
        path), not ``ConditioningCacheError``."""
        from reelcache.errors import IntegrityError

        _, port = _tiny_native_and_port()
        torch.manual_seed(7)
        latents = torch.randn(1, 16, 6, 8, 12)
        text = torch.randn(1, 1, 9, TINY.caption_channels)
        cond, new = latents[:, :, :2], latents[:, :, 2:]
        t = torch.tensor([500.0]).expand(1).unsqueeze(-1).repeat(1, 4)
        damage = IntegrityError("immutable object hash or length mismatch: blob:" + "ab" * 32)
        with torch.no_grad():
            keys, values = port.build_conditioning_kv(cond)
            ready = cc.LayerReadiness(TINY.depth)
            ready.mark(0)  # step 0 already read layer 0
            ready.fail(damage)
            with self.assertRaises(IntegrityError) as caught:
                port.forward_with_conditioning(new, t, text, None, list(zip(keys, values)), 2, kv_cache_ready=ready)
            self.assertNotIsInstance(caught.exception, cc.ConditioningCacheError)
            self.assertIs(caught.exception.__cause__, damage)
            self.assertIn("overlapped conditioning-cache restore failed", str(caught.exception))
            self.assertIn("hash or length mismatch", str(caught.exception))
            with self.assertRaises(IntegrityError) as joined:
                ready.join()
            self.assertIs(joined.exception.__cause__, damage)


class DiTEquivalenceTests(unittest.TestCase):
    """Tiny FP32 model: the port and the native module produce identical tensors on CPU."""

    @classmethod
    def setUpClass(cls):
        cls.nat, cls.port = _tiny_native_and_port()
        torch.manual_seed(7)
        cls.latents = torch.randn(1, 16, 6, 8, 12)  # 2 conditioning + 4 new latent frames, 4x6 tokens/frame
        cls.text = torch.randn(1, 1, 9, TINY.caption_channels)
        cls.mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0]])

    def test_checkpoint_parameter_names_are_native(self):
        self.assertEqual(sorted(n for n, _ in self.port.named_parameters()),
                         sorted(n for n, _ in self.nat.named_parameters()))
        index = json.loads((checkpoint() / "dit" / "diffusion_pytorch_model.safetensors.index.json").read_text())
        with torch.device("meta"):
            full = dit_core.LongCatVideoDiTCore(spec.LONGCAT_VIDEO_ARCH)
        self.assertEqual({n for n, _ in full.named_parameters()}, set(index["weight_map"]))
        self.assertEqual({n: tuple(p.shape) for n, p in full.named_parameters()}["blocks.0.attn.qkv.weight"],
                         (12288, 4096))

    def test_plain_and_conditioned_forward_match_native(self):
        t = torch.tensor([640.0])
        with torch.no_grad():
            self.assertTrue(torch.equal(self.port(self.latents, t, self.text, self.mask),
                                        self.nat(self.latents, t, self.text, self.mask)))
            per_frame = torch.tensor([[0.0, 0.0, 640.0, 640.0, 640.0, 640.0]])
            ours = self.port(self.latents, per_frame, self.text, self.mask, num_cond_latents=2)
            ref = self.nat(self.latents, per_frame, self.text, self.mask, num_cond_latents=2)
            self.assertTrue(torch.equal(ours, ref))
            self.assertEqual(ours.dtype, torch.float32)

    def test_cache_build_is_native_pre_rope_kv(self):
        cond = self.latents[:, :, :2]
        empty = torch.zeros(1, 1, 9, TINY.caption_channels)
        with torch.no_grad():
            _, kv = self.nat(cond, torch.zeros(1, 2), empty, return_kv=True, skip_crs_attn=True)
            keys, values = self.port.build_conditioning_kv(cond)
        self.assertEqual(len(keys), TINY.depth)
        for i in range(TINY.depth):
            self.assertEqual(tuple(keys[i].shape), (1, 2 * 24, TINY.num_heads, TINY.head_dim))
            self.assertTrue(torch.equal(keys[i].permute(0, 2, 1, 3), kv[i][0]), i)
            self.assertTrue(torch.equal(values[i].permute(0, 2, 1, 3), kv[i][1]), i)
            self.assertTrue(keys[i].is_contiguous() and values[i].is_contiguous())
        # Layer 0 key is k_norm(qkv(modulated x)) with no rotation applied.
        blk = self.port.blocks[0]
        with torch.no_grad():
            x = self.port.x_embedder(cond)
            t_emb = self.port._embed_timestep(torch.zeros(1, 2), 1, 2, torch.float32)
            mod = dit_core.linear_fp32(blk.adaLN_modulation[1], F.silu(t_emb))
            shift, scale = mod.unsqueeze(2).chunk(6, dim=-1)[:2]
            xm = dit_core.modulate_fp32(blk.mod_norm_attn, x.view(1, 2, -1, TINY.hidden_size), shift, scale)
            _, k0, _ = blk.attn._qkv(xm.view(1, -1, TINY.hidden_size))
        self.assertTrue(torch.equal(k0, keys[0]))

    def test_cache_read_forward_matches_native(self):
        cond, new = self.latents[:, :, :2], self.latents[:, :, 2:]
        empty = torch.zeros(1, 1, 9, TINY.caption_channels)
        with torch.no_grad():
            _, kv = self.nat(cond, torch.zeros(1, 2), empty, return_kv=True, skip_crs_attn=True)
            keys, values = self.port.build_conditioning_kv(cond)
            t = torch.tensor([931.0]).expand(1).unsqueeze(-1).repeat(1, 4)
            ref = self.nat(new, t, self.text, self.mask, num_cond_latents=2, kv_cache_dict=kv)
            ours = self.port.forward_with_conditioning(new, t, self.text, self.mask, list(zip(keys, values)), 2)
            self.assertTrue(torch.equal(ours, ref))
            # a restored cache in native layout (e.g. from ReelCache) installs to identical outputs
            epoch = cc.ConditioningCacheEpoch(0, (2, 4, 6), "x", "y")
            restored = cc.LongCatConditioningCache.from_native_layout(
                epoch, [kv[i][0] for i in range(TINY.depth)], [kv[i][1] for i in range(TINY.depth)])
            again = self.port.forward_with_conditioning(new, t, self.text, self.mask,
                                                        list(zip(restored.keys_pre_rope, restored.values)), 2)
            self.assertTrue(torch.equal(again, ref))
            with self.assertRaises(ValueError):
                self.port.forward_with_conditioning(new, t, self.text, self.mask, list(zip(keys, values)), 3)

    def test_bf16_port_dtypes(self):
        port = dit_core.LongCatVideoDiTCore(TINY).to(torch.bfloat16).eval()
        with torch.no_grad():
            keys, values = port.build_conditioning_kv(self.latents[:, :, :2])
            out = port.forward_with_conditioning(self.latents[:, :, 2:], torch.tensor([500.0]), self.text,
                                                 self.mask, list(zip(keys, values)), 2)
        self.assertEqual((keys[0].dtype, values[0].dtype, out.dtype), (torch.bfloat16, torch.bfloat16, torch.float32))
        self.assertTrue(torch.isfinite(out).all())


class ConditioningCacheSeamTests(unittest.TestCase):
    def _epoch(self, **kw):
        base = dict(segment_index=1, token_grid=(2, 4, 6), cond_latents_sha256="abc",
                    model_fingerprint=weights.MODEL_FINGERPRINT, session_id="s", parent_segment_index=0)
        base.update(kw)
        return cc.ConditioningCacheEpoch(**base)

    def _cache(self, epoch, layers=2, heads=4, dim=16, dtype=torch.bfloat16):
        g = torch.Generator().manual_seed(0)
        ks = [torch.randn(1, epoch.tokens, heads, dim, generator=g).to(dtype) for _ in range(layers)]
        vs = [torch.randn(1, epoch.tokens, heads, dim, generator=g).to(dtype) for _ in range(layers)]
        return cc.LongCatConditioningCache(epoch=epoch, keys_pre_rope=ks, values=vs)

    def test_epoch_contract(self):
        e = self._epoch()
        self.assertEqual((e.tokens, e.tokens_per_frame, e.key_domain), (48, 24, "pre_rope"))
        self.assertEqual(e.as_dict()["cause"], "longcat_segment")
        for bad in (dict(segment_index=-1), dict(token_grid=(0, 4, 6)), dict(position_offset=4),
                    dict(key_domain="post_rope"), dict(parent_segment_index=3), dict(layout="TBHD")):
            with self.assertRaises(cc.ConditioningCacheError, msg=str(bad)):
                self._epoch(**bad)
        self.assertEqual(hash(e), hash(self._epoch()))

    def test_layouts_and_matrices(self):
        cache = self._cache(self._epoch())
        cache.validate(layers=2, heads=4, head_dim=16, dtype=torch.bfloat16)
        k_nat, v_nat = cache.layer_native(0)
        self.assertEqual(tuple(k_nat.shape), (1, 4, 48, 16))
        km, vm = cache.layer_matrices(1)
        self.assertEqual(tuple(km.shape), (48, 64))
        self.assertEqual(km.data_ptr(), cache.keys_pre_rope[1].data_ptr())  # BTHD matrices are views
        back = cc.LongCatConditioningCache.from_matrices(cache.epoch, [cache.layer_matrices(i)[0] for i in range(2)],
                                                         [cache.layer_matrices(i)[1] for i in range(2)], heads=4)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(back.keys_pre_rope, cache.keys_pre_rope)))
        native_cache = cc.LongCatConditioningCache.from_native_layout(
            cache.epoch, [cache.layer_native(i)[0] for i in range(2)], [cache.layer_native(i)[1] for i in range(2)])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(native_cache.values, cache.values)))
        self.assertEqual(cache.nbytes, 2 * 2 * 48 * 64 * 2)
        sha = cache.per_layer_sha256()[0]["K"]
        ref = hashlib.sha256(k_nat.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
        self.assertEqual(sha, ref)
        with self.assertRaises(cc.ConditioningCacheError):
            cache.validate(layers=3, heads=4, head_dim=16)
        with self.assertRaises(cc.ConditioningCacheError):
            cache.validate(layers=2, heads=4, head_dim=16, dtype=torch.float32)

    def test_backend_registry_and_restore_validation(self):
        epoch = self._epoch()
        good = self._cache(epoch)
        good.source = cc.SOURCE_RESTORED

        class Backend:
            def __init__(self):
                self.seen = []

            def restore(self, e):
                return good if e == epoch else None

            def on_cache_built(self, cache):
                self.seen.append(cache.epoch)

        backend = Backend()
        name = "unit-test-longcat-backend"
        cc.register_cache_backend(name, backend)
        try:
            self.assertIs(cc.get_cache_backend(name), backend)
            with self.assertRaises(cc.ConditioningCacheError):
                cc.register_cache_backend(name, backend)
            self.assertIn(name, cc.registered_cache_backends())
            ok = cc.accept_restored_cache(backend.restore(epoch), epoch, layers=2, heads=4, head_dim=16,
                                          dtype=torch.bfloat16, device="cpu")
            self.assertIs(ok, good)
            with self.assertRaises(cc.ConditioningCacheError):
                cc.accept_restored_cache(good, self._epoch(cond_latents_sha256="other"), layers=2, heads=4,
                                         head_dim=16, dtype=torch.bfloat16, device="cpu")
            built = self._cache(epoch)
            with self.assertRaises(cc.ConditioningCacheError):
                cc.accept_restored_cache(built, epoch, layers=2, heads=4, head_dim=16, dtype=torch.bfloat16,
                                         device="cpu")
            with self.assertRaises(cc.ConditioningCacheError):
                cc.accept_restored_cache({"k": 1}, epoch, layers=2, heads=4, head_dim=16, dtype=torch.bfloat16,
                                         device="cpu")
        finally:
            cc.unregister_cache_backend(name)
        with self.assertRaises(cc.ConditioningCacheError):
            cc.get_cache_backend(name)
        with self.assertRaises(cc.ConditioningCacheError):
            cc.register_cache_backend("x", object())


if __name__ == "__main__":
    unittest.main()
