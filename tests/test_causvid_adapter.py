"""CPU contract tests. These fakes are not evidence of native video parity."""

from __future__ import annotations

import copy
import gc
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from reelcache.adapters.causvid import CausVidAdapter, CausVidConfig, tensor_digest
from reelcache.errors import CapacityError, ConfigurationError, IntegrityError, StateError, UnsupportedError
from reelcache.manager import KVRuntime
from reelcache.types import CodecSpec, MatrixKey, ModelIdentity, TensorContext


CONFIG = CausVidConfig(
    frames_per_chunk=3, grid_height=1, grid_width=2, heads=2, head_dim=4, latent_channels=8,
)


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    frames, height, width = grid_sizes[0].tolist()
    positions = torch.arange(start_frame, start_frame + frames, device=x.device)
    positions = positions.repeat_interleave(height * width)
    angles = positions[:, None] * freqs[None, :].to(x.device)
    pairs = torch.view_as_complex(x.double().reshape(*x.shape[:-1], -1, 2))
    phase = torch.polar(torch.ones_like(angles), angles)[None, :, None, :]
    return torch.view_as_real(pairs * phase).flatten(-2).to(x.dtype)


class FakeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads, self.head_dim = config.heads, config.head_dim
        self.window_size = (-1, -1)
        self.q, self.k, self.v, self.o = [nn.Linear(config.channels, config.channels) for _ in range(4)]
        self.norm_k = nn.LayerNorm(config.channels)
        self.key_scale = 1
        self.seen_positions = []

    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask, kv_cache=None, current_start=0, current_end=0):
        shape = (1, x.shape[1], self.num_heads, self.head_dim)
        q, k, v = self.q(x).reshape(shape), self.norm_k(self.k(x)).reshape(shape), self.v(x).reshape(shape)
        frame_start = current_start // int(grid_sizes[0, 1:].prod())
        q = causal_rope_apply(q, grid_sizes, freqs, start_frame=frame_start)
        k = causal_rope_apply(k * self.key_scale, grid_sizes, freqs, start_frame=frame_start)
        kv_cache["k"][:, current_start:current_end] = k
        kv_cache["v"][:, current_start:current_end] = v
        keys, values = kv_cache["k"][:, :current_end], kv_cache["v"][:, :current_end]
        self.seen_positions.append((current_start, current_end, keys.shape[1], block_mask))
        result = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2),
        ).transpose(1, 2).flatten(2)
        return self.o(result)


def flash_attention(q, k, v, k_lens=None):
    return q + 17


class FakeCrossAttention(nn.Module):
    def forward(self, x, context, context_lens, crossattn_cache=None):
        return flash_attention(x, context, context, k_lens=context_lens)


class FakeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads, self.dim = config.heads, config.channels
        self.num_frame_per_block = config.frames_per_chunk
        self.num_layers, self.model_type = 2, "t2v"
        self.patch_size = (1, config.patch_height, config.patch_width)
        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            block = nn.Module()
            block.self_attn = FakeAttention(config)
            block.cross_attn = FakeCrossAttention()
            self.blocks.append(block)
        self.eval()


class FakeScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([800, 300])
        self.sigmas = torch.tensor([0.8, 0.3])

    def add_noise(self, clean, noise, timesteps):
        fraction = timesteps.to(clean.dtype).reshape(-1, 1, 1, 1) / 1000
        return clean * (1 - fraction) + noise * fraction


class FakeVAE:
    def __init__(self):
        self.clear_count = 0
        self.model = SimpleNamespace(clear_cache=self.clear_cache)

    def clear_cache(self):
        self.clear_count += 1

    def decode_to_pixel(self, latents):
        return latents[:, :, :3].tanh()


class FakeGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model, self.config = FakeModel(config), config
        self.seq_len, self.uniform_timestep = 200, False
        self.mask = object()

    def forward(self, noisy_image_or_video, conditional_dict, timestep, kv_cache,
                crossattn_cache, current_start, current_end):
        cfg = self.config
        pooled = torch.nn.functional.avg_pool2d(
            noisy_image_or_video.flatten(0, 1), (cfg.patch_height, cfg.patch_width),
        )
        x = pooled.unflatten(0, (1, cfg.frames_per_chunk)).permute(0, 1, 3, 4, 2)
        x = x.reshape(1, cfg.chunk_tokens, cfg.channels) + timestep.float().mean() / 5000
        for index, block in enumerate(self.model.blocks):
            text_cache = crossattn_cache[index]
            if not text_cache["is_init"]:
                text_cache["k"] = conditional_dict["prompt_embeds"].mean().clone()
                text_cache["v"] = text_cache["k"].clone()
                text_cache["is_init"] = True
            x = block.self_attn(
                x + text_cache["k"], torch.tensor([cfg.chunk_tokens]),
                torch.tensor([[cfg.frames_per_chunk, cfg.grid_height, cfg.grid_width]]),
                torch.tensor([0.3, 0.9], dtype=torch.float64), self.mask,
                kv_cache[index], current_start, current_end,
            )
        shift = x.reshape(1, cfg.frames_per_chunk, cfg.grid_height, cfg.grid_width, cfg.channels)
        shift = shift.mean((2, 3)).unsqueeze(-1).unsqueeze(-1)
        return noisy_image_or_video * 0.7 + shift * 0.2


class FakePipeline:
    def __init__(self, config=CONFIG):
        self.generator = FakeGenerator(config)
        self.scheduler, self.vae = FakeScheduler(), FakeVAE()
        self.denoising_step_list = torch.tensor([800, 300])
        self.num_transformer_blocks = 2
        self.num_frame_per_block, self.frame_seq_length = config.frames_per_chunk, config.frame_tokens
        self.kv_cache1 = None
        self.crossattn_cache = None


def initial_state(config=CONFIG, seed=11):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.empty((1, 9, *config.latent_tail)).normal_(generator=rng)
    return noise, {"prompt_embeds": torch.linspace(-0.5, 0.5, 24).reshape(1, 3, 8)}, rng


@torch.no_grad()
def dense_reference(pipe, noise, conditioning, rng, clean_prefix):
    cfg = pipe.generator.config
    pipe.kv_cache1 = [
        {name: torch.zeros(1, noise.shape[1] * cfg.frame_tokens, cfg.heads, cfg.head_dim)
         for name in ("k", "v")} for _ in range(pipe.num_transformer_blocks)
    ]
    pipe.crossattn_cache = [
        {"k": None, "v": None, "is_init": False} for _ in range(pipe.num_transformer_blocks)
    ]
    output = torch.zeros_like(noise)
    for start in range(0, noise.shape[1], cfg.frames_per_chunk):
        end = start + cfg.frames_per_chunk

        def forward(value, step):
            return pipe.generator(
                value, conditioning, torch.full((1, cfg.frames_per_chunk), step),
                pipe.kv_cache1, pipe.crossattn_cache,
                start * cfg.frame_tokens, end * cfg.frame_tokens,
            )

        if start == 0:
            clean = clean_prefix
        else:
            value = noise[:, start:end]
            for index, step in enumerate(pipe.denoising_step_list):
                clean = forward(value, step.item())
                if index + 1 < len(pipe.denoising_step_list):
                    value = pipe.scheduler.add_noise(
                        clean.flatten(0, 1), torch.empty_like(clean.flatten(0, 1)).normal_(generator=rng),
                        pipe.denoising_step_list[index + 1] * torch.ones(cfg.frames_per_chunk, dtype=torch.long),
                    ).unflatten(0, clean.shape[:2])
        output[:, start:end] = clean
        forward(clean, 0)
    return output


class CausVidAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.directory = Path(__file__).resolve().parents[1] / ".test-work" / f"causvid-{uuid.uuid4().hex}"
        self.directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.directory)
        torch.manual_seed(31)
        self.pipe = FakePipeline()
        self.model = ModelIdentity("causvid-cpu-fake", "fake-source-not-native-proof", 2, 8, "tokens_channels", "float32")
        self.runtime = KVRuntime(self.directory / "store", self.model, max_resident_bytes=1024 * 1024)
        self.addCleanup(self.runtime.close)
        self.noise, self.conditioning, self.rng = initial_state()
        self.prefix = self.noise[:, :3].clone() * 0.25

    def adapter(self, **kwargs):
        adapter = CausVidAdapter(self.pipe, self.runtime, "parent", config=CONFIG, **kwargs)
        adapter.install()
        self.addCleanup(adapter.close)
        adapter.start(self.noise, self.conditioning, self.rng, conditioning_id="fixed-prompt")
        return adapter

    def history_hashes(self, branch, end):
        rows = {}
        for layer in range(2):
            for target in ("K", "V"):
                key = MatrixKey(layer, target)
                with self.runtime.acquire_history(branch, key, token_end=end) as lease:
                    rows[key] = tensor_digest(lease.tensor)
        return rows

    def test_native_history_diagnostic_binds_real_packet_heads_and_post_rope_bytes(self):
        records, heads = [], []
        adapter = self.adapter(
            specs=CodecSpec("svd", rank=2),
            native_history_observer=lambda row, _tensor: records.append(row),
        )
        for _ in range(2):
            adapter.step()
            heads.append(self.runtime.branch_info("parent")["head_id"])
        commits = [row for row in records if row["event"] == "clean_publication"]
        self.assertEqual(len(commits), 8)
        for row in commits:
            chunk, key = row["chunk_id"], MatrixKey(row["layer"], row["target"])
            first, last = chunk * CONFIG.chunk_tokens, (chunk + 1) * CONFIG.chunk_tokens
            self.assertEqual(row["committed_head_id"], heads[chunk])
            with self.runtime.acquire_history(
                "parent", key, token_start=first, token_end=last, dtype=torch.float32,
            ) as lease:
                value = lease.tensor.reshape(1, CONFIG.chunk_tokens, CONFIG.heads, CONFIG.head_dim)
                if key.target == "K":
                    value = causal_rope_apply(
                        value, torch.tensor([[3, 1, 2]]), torch.tensor([0.3, 0.9], dtype=torch.float64),
                        start_frame=chunk * CONFIG.frames_per_chunk,
                    )
                self.assertEqual(tensor_digest(value), row["decoded_current_sha256"])
        self.assertEqual(self.runtime.active_leases, 0)
        self.assertEqual(self.runtime.resident_bytes, 0)
        adapter.close()
        self.assertFalse(adapter._native_committed_hashes)
        self.assertIsNone(self.pipe.kv_cache1)
        self.assertTrue(all(not block.self_attn._forward_hooks for block in self.pipe.generator.model.blocks))

    def test_raw_dense_resume_fork_and_absolute_positions(self):
        native = copy.deepcopy(self.pipe)
        rng = torch.Generator().set_state(self.rng.get_state())
        expected = dense_reference(native, self.noise, self.conditioning, rng, self.prefix)
        adapter = self.adapter()
        global_rng = torch.get_rng_state().clone()
        adapter.step(clean_prefix=self.prefix)
        snapshot = adapter.snapshot()
        adapter.step()
        adapter.step()
        self.assertTrue(torch.equal(expected, adapter.latent_state))
        self.assertTrue(torch.equal(rng.get_state(), adapter.rng_state))
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
        original_video = adapter.decode().clone()
        parent_hashes = self.history_hashes("parent", 3 * CONFIG.chunk_tokens)
        for key, digest in parent_hashes.items():
            self.assertEqual(digest, tensor_digest(native.kv_cache1[key.layer][key.target.lower()][0].flatten(1)))
        adapter.restore(snapshot, "resumed", conditioning_id="fixed-prompt")
        adapter.step()
        adapter.step()
        self.assertTrue(torch.equal(expected, adapter.latent_state))
        self.assertTrue(torch.equal(original_video, adapter.decode()))
        self.assertEqual([row["token_start"] for row in adapter.audit], [CONFIG.chunk_tokens, 2 * CONFIG.chunk_tokens])
        adapter.fork(snapshot, "child", seed=999)
        adapter.step()
        adapter.step()
        child = adapter.latent_state
        self.assertFalse(torch.equal(expected[:, 3:], child[:, 3:]))
        self.assertTrue(torch.equal(expected[:, :3], child[:, :3]))
        self.assertEqual(parent_hashes, self.history_hashes("parent", 3 * CONFIG.chunk_tokens))
        adapter.fork(snapshot, "child-replay", seed=999)
        adapter.step()
        adapter.step()
        self.assertTrue(torch.equal(child, adapter.latent_state))
        self.assertEqual(parent_hashes, self.history_hashes("parent", 3 * CONFIG.chunk_tokens))
        for block in self.pipe.generator.model.blocks:
            for start, end, visible, mask in block.self_attn.seen_positions:
                self.assertEqual(end, visible)
                self.assertEqual(end - start, CONFIG.chunk_tokens)
                self.assertIs(mask, self.pipe.generator.mask)

    def test_dense_and_history_references_released_per_layer(self):
        adapter = self.adapter()
        dense_refs, history_refs, active, maximum = [], [], [], [0]
        seal = self.runtime.seal_chunk
        acquire = self.runtime.acquire_history

        def observed_seal(*args, **kwargs):
            dense_refs.extend(weakref.ref(value) for value in adapter._mutable.values())
            self.assertGreater(adapter.memory_report()["live_mutable_kv_bytes"], 0)
            return seal(*args, **kwargs)

        class ObservedLease:
            def __init__(self, key, inner):
                self.key, self.inner = key, inner

            def __enter__(inner_self):
                inner_self.inner.__enter__()
                active.append(inner_self.key)
                maximum[0] = max(maximum[0], len(active))
                self.assertEqual(len({key.layer for key in active}), 1)
                history_refs.append(weakref.ref(inner_self.inner.tensor))
                return inner_self.inner

            def __exit__(inner_self, *args):
                active.remove(inner_self.key)
                return inner_self.inner.__exit__(*args)

        def observed_acquire(branch, key, **kwargs):
            return ObservedLease(key, acquire(branch, key, **kwargs))

        with patch.object(self.runtime, "seal_chunk", side_effect=observed_seal), patch.object(
            self.runtime, "acquire_history", side_effect=observed_acquire,
        ):
            adapter.step(clean_prefix=self.prefix)
            adapter.step()
            adapter.step()
        gc.collect()
        self.assertEqual(maximum[0], 2)
        self.assertEqual(active, [])
        self.assertTrue(dense_refs and all(ref() is None for ref in dense_refs))
        self.assertTrue(history_refs and all(ref() is None for ref in history_refs))
        report = adapter.memory_report()
        self.assertEqual(report["live_mutable_kv_bytes"], 0)
        self.assertEqual(report["live_attention_read_bytes"], 0)
        self.assertEqual(report["live_clean_capture_cpu_bytes"], 0)
        self.assertLess(report["peak_tracked_attention_read_bytes"], 20000)

    def test_clean_snapshot_guards_and_encoding_failure_rollback(self):
        adapter = self.adapter()
        with self.assertRaises(StateError):
            adapter.snapshot()
        phases = []

        def before_forward(_module, _args, kwargs):
            phases.append(bool((kwargs["timestep"] == 0).all()))
            with self.assertRaises(StateError):
                adapter.snapshot()

        handle = self.pipe.generator.register_forward_pre_hook(before_forward, with_kwargs=True)
        self.addCleanup(handle.remove)
        adapter.step(clean_prefix=self.prefix)
        before = adapter.snapshot()
        before_latents, before_rng = adapter.latent_state, adapter.rng_state
        real_seal = self.runtime.seal_chunk

        def fail_seal(*args, **kwargs):
            with self.assertRaises(StateError):
                adapter.snapshot()
            raise OSError("injected encode failure")

        with patch.object(self.runtime, "seal_chunk", side_effect=fail_seal):
            with self.assertRaisesRegex(OSError, "injected"):
                adapter.step()
        self.assertTrue(torch.equal(before_latents, adapter.latent_state))
        self.assertTrue(torch.equal(before_rng, adapter.rng_state))
        self.assertEqual(adapter.memory_report()["live_mutable_kv_bytes"], 0)
        self.assertEqual(adapter.snapshot(), before)
        self.assertIn(False, phases)
        self.assertIn(True, phases)
        with patch.object(self.runtime, "seal_chunk", wraps=real_seal):
            adapter.step()
        recovered = adapter.latent_state
        adapter.restore(before, "rollback-replay")
        adapter.step()
        self.assertTrue(torch.equal(recovered, adapter.latent_state))

    def test_partial_lease_failure_does_not_retain_k_or_advance_rng(self):
        adapter = self.adapter()
        adapter.step(clean_prefix=self.prefix)
        prior_rng = adapter.rng_state
        acquire = self.runtime.acquire_history

        def fail_v(branch, key, **kwargs):
            if key.target == "V":
                raise CapacityError("injected V admission failure")
            return acquire(branch, key, **kwargs)

        with patch.object(self.runtime, "acquire_history", side_effect=fail_v):
            with self.assertRaises(CapacityError):
                adapter.step()
        self.assertEqual(adapter.token_cursor, CONFIG.chunk_tokens)
        self.assertTrue(torch.equal(prior_rng, adapter.rng_state))
        self.assertEqual(adapter.memory_report()["live_attention_read_bytes"], 0)
        adapter.step()

    def test_hot_history_reuses_decoded_storage_and_matches_uncached_lossy(self):
        for kind in ("raw", "svd", "svd_hevc"):
            with self.subTest(kind=kind):
                specs = {
                    MatrixKey(layer, target): (
                        CodecSpec() if kind == "raw" else CodecSpec(kind, rank=3)
                    ) for layer in range(2) for target in ("K", "V")
                }
                with CausVidAdapter(
                    self.pipe, self.runtime, kind + "-cold", config=CONFIG, specs=specs,
                ) as cold:
                    cold.start(self.noise, self.conditioning, self.rng, conditioning_id="fresh")
                    cold.step(clean_prefix=self.prefix)
                    snapshot = cold.snapshot()
                    with patch.object(self.runtime, "acquire_history", wraps=self.runtime.acquire_history) as reads:
                        cold.step()
                        self.assertEqual(reads.call_count, 12)
                    expected, expected_rng = cold.latent_state, cold.rng_state
                    parent = self.runtime.branch_info(kind + "-cold")
                budget = 2 * 2 * CONFIG.chunk_tokens * CONFIG.channels * 4
                with CausVidAdapter(
                    self.pipe, self.runtime, kind + "-hot", config=CONFIG, specs=specs,
                    max_hot_history_bytes=budget,
                ) as hot:
                    hot.restore(snapshot, kind + "-hot")
                    seen, references = {}, []
                    original = hot._history_for

                    def observe(frame, key):
                        value = original(frame, key)
                        seen.setdefault(key, []).append(value.data_ptr())
                        references.append(weakref.ref(value))
                        report = hot.memory_report()
                        self.assertLessEqual(report["live_hot_history_bytes"], budget)
                        self.assertGreaterEqual(
                            report["live_attention_and_hot_unique_bytes"], report["live_hot_history_bytes"],
                        )
                        return value

                    seal = self.runtime.seal_chunk

                    def checked_seal(*args, **kwargs):
                        self.assertEqual(hot.memory_report()["live_hot_history_bytes"], 0)
                        return seal(*args, **kwargs)

                    with patch.object(hot, "_history_for", side_effect=observe), patch.object(
                        self.runtime, "acquire_history", wraps=self.runtime.acquire_history,
                    ) as reads, patch.object(
                        self.runtime, "seal_chunk", side_effect=checked_seal,
                    ):
                        hot.step()
                        self.assertEqual(reads.call_count, 4)
                    self.assertTrue(torch.equal(expected, hot.latent_state))
                    self.assertTrue(torch.equal(expected_rng, hot.rng_state))
                    self.assertTrue(all(len(values) == 3 and len(set(values)) == 1 for values in seen.values()))
                    self.assertTrue(all(reference() is None for reference in references))
                    self.assertEqual(hot.memory_report()["peak_hot_history_bytes"], budget)
                    self.assertEqual(hot.memory_report()["live_hot_history_bytes"], 0)
                    self.assertEqual(self.runtime.active_leases, 0)
                    self.assertEqual(parent, self.runtime.branch_info(kind + "-cold"))

    def test_hot_history_budget_epoch_mutation_and_failure_cleanup(self):
        for bad in (-1, True, 1.5):
            with self.assertRaises(ConfigurationError):
                CausVidAdapter(self.pipe, self.runtime, "invalid", config=CONFIG, max_hot_history_bytes=bad)
        budget = 2 * CONFIG.chunk_tokens * CONFIG.channels * 4
        adapter = self.adapter(max_hot_history_bytes=budget)
        adapter.step(clean_prefix=self.prefix)
        snapshot = adapter.snapshot()
        prior_rng, prior_latents = adapter.rng_state, adapter.latent_state
        events, refs = [], []
        calls = [0]

        class Event:
            synchronized = False

            def synchronize(self):
                self.synchronized = True

        def corrupt(_module, _args, _kwargs):
            calls[0] += 1
            if calls[0] == 2:
                self.assertEqual(len(adapter._hot_history), 2)
                for entry in adapter._hot_history.values():
                    entry.event = Event()
                    events.append(entry.event)
                    refs.append(weakref.ref(entry.tensor))
                next(iter(adapter._hot_history.values())).tensor.add_(1)

        handle = self.pipe.generator.register_forward_pre_hook(corrupt, with_kwargs=True)
        try:
            with self.assertRaisesRegex(IntegrityError, "mutated"):
                adapter.step()
        finally:
            handle.remove()
        self.assertTrue(all(event.synchronized for event in events))
        self.assertTrue(all(reference() is None for reference in refs))
        self.assertEqual(adapter.memory_report()["live_hot_history_bytes"], 0)
        self.assertEqual(self.runtime.active_leases, 0)
        self.assertTrue(torch.equal(prior_rng, adapter.rng_state))
        self.assertTrue(torch.equal(prior_latents, adapter.latent_state))
        self.assertEqual(snapshot, adapter.snapshot())
        adapter.step()
        self.assertEqual(adapter.memory_report()["peak_hot_history_bytes"], budget)
        self.assertEqual(adapter.memory_report()["hot_history_hits"], 4)
        info = self.runtime.branch_info
        parent_before = info("parent")
        calls[0] = 0

        def replaced_head(branch):
            calls[0] += 1
            value = info(branch)
            if adapter._hot_epoch is not None:
                value["head_id"] = "manifest:" + "0" * 64
            return value

        with patch.object(self.runtime, "branch_info", side_effect=replaced_head):
            with self.assertRaisesRegex(StateError, "epoch"):
                adapter.step()
        self.assertIsNone(adapter._hot_epoch)
        self.assertEqual(adapter.memory_report()["live_hot_history_bytes"], 0)
        self.assertEqual(adapter._phase, "poisoned")
        adapter.close()
        adapter.install()
        adapter.restore(snapshot, "isolated-hot-fork", seed=53)
        self.assertFalse(adapter._hot_history)
        self.assertIsNone(adapter._hot_epoch)
        adapter.step()
        self.assertEqual(info("parent")["token_end"], 2 * CONFIG.chunk_tokens)
        self.assertEqual(info("parent"), parent_before)

    def test_nonfinite_native_state_is_not_published(self):
        adapter = self.adapter()
        adapter.step(clean_prefix=self.prefix)
        snapshot, rng_before = adapter.snapshot(), adapter.rng_state

        def poison_output(_module, _args, kwargs, output):
            if bool((kwargs["timestep"] != 0).any()):
                return torch.full_like(output, float("nan"))
            return output

        def poison_clean_cache(_module, _args, kwargs, _output):
            if bool((kwargs["timestep"] == 0).all()):
                adapter._mutable[MatrixKey(0, "K")].fill_(float("nan"))

        for hook in (poison_output, poison_clean_cache):
            handle = self.pipe.generator.register_forward_hook(hook, with_kwargs=True)
            try:
                with self.assertRaises(IntegrityError):
                    adapter.step()
            finally:
                handle.remove()
            self.assertEqual(adapter.snapshot(), snapshot)
            self.assertTrue(torch.equal(adapter.rng_state, rng_before))
            self.assertEqual(adapter.memory_report()["live_mutable_kv_bytes"], 0)
            self.assertEqual(self.runtime.active_leases, 0)
        adapter.step()
        self.assertTrue(bool(torch.isfinite(adapter.latent_state).all()))

    def test_instance_local_reversible_hooks_and_cache_adoption_rejection(self):
        sibling = copy.deepcopy(self.pipe)
        original = self.pipe.generator.model.blocks[0].self_attn.forward.__func__
        sibling_original = sibling.generator.model.blocks[0].self_attn.forward.__func__
        adapter = self.adapter()
        self.assertIs(sibling.generator.model.blocks[0].self_attn.forward.__func__, sibling_original)
        adapter.step(clean_prefix=self.prefix)
        adapter.close()
        self.assertIs(self.pipe.generator.model.blocks[0].self_attn.forward.__func__, original)
        self.assertNotIn("forward", self.pipe.generator.model.blocks[0].self_attn.__dict__)
        self.assertIsNone(self.pipe.kv_cache1)
        self.pipe.kv_cache1 = [{"k": torch.ones(2), "v": torch.ones(2)}]
        foreign = self.pipe.kv_cache1
        with self.assertRaises(UnsupportedError):
            CausVidAdapter(self.pipe, self.runtime, "adopt", config=CONFIG).install()
        self.assertIs(self.pipe.kv_cache1, foreign)

    def test_wrong_config_and_conditioning_fail_before_branch_creation(self):
        wrong = CausVidConfig(frames_per_chunk=3, grid_height=1, grid_width=2, heads=1, head_dim=8, latent_channels=8)
        with self.assertRaises(ConfigurationError):
            CausVidAdapter(self.pipe, self.runtime, "wrong", config=wrong)
        with self.assertRaises(ConfigurationError):
            CausVidAdapter(self.pipe, self.runtime, "bad-reuse", config=CONFIG, reuse_threshold=float("nan"))
        adapter = self.adapter()
        adapter.step(clean_prefix=self.prefix)
        snapshot = adapter.snapshot()
        refs_before, rng_before = self.runtime.store.list_refs(), adapter.rng_state
        global_rng_before = torch.get_rng_state().clone()
        latents_before, noise_before = adapter.latent_state, adapter._noise.clone()
        cursor_before, phase_before = adapter.frame_cursor, adapter._phase
        steps = self.pipe.denoising_step_list
        for branch, kwargs, reuse, schedule in (
            ("wrong-prompt", {"conditioning_id": "other-prompt"}, None, steps),
            ("wrong-reuse-policy", {}, 0.1, steps),
            ("wrong-schedule", {}, None, torch.tensor([700, 300])),
        ):
            with self.subTest(branch=branch), patch.object(adapter, "reuse_threshold", reuse), patch.object(
                self.pipe, "denoising_step_list", schedule,
            ), patch.object(self.runtime, "fork", wraps=self.runtime.fork) as restore, patch(
                "reelcache.manager.deserialize_tensors", side_effect=AssertionError("snapshot tensors read too early"),
            ) as deserialize, patch.object(
                adapter, "_prepare_restore", side_effect=AssertionError("native preparation ran before rejection"),
            ) as prepare:
                with self.assertRaises(ConfigurationError):
                    adapter.restore(snapshot, branch, **kwargs)
                restore.assert_called_once()
                deserialize.assert_not_called()
                prepare.assert_not_called()
            self.assertEqual(self.runtime.store.list_refs(), refs_before)
            self.assertEqual(adapter.branch_id, "parent")
            self.assertEqual((adapter.frame_cursor, adapter._phase), (cursor_before, phase_before))
            self.assertTrue(torch.equal(adapter.rng_state, rng_before))
            self.assertTrue(torch.equal(torch.get_rng_state(), global_rng_before))
            self.assertTrue(torch.equal(adapter.latent_state, latents_before))
            self.assertTrue(torch.equal(adapter._noise, noise_before))
            self.assertEqual((self.runtime.active_leases, self.runtime.resident_bytes), (0, 0))
            self.assertFalse(self.runtime._pending)
        self.assertEqual(adapter.snapshot(), snapshot)

    def test_restore_and_seed_cannot_replace_existing_parent(self):
        adapter = self.adapter()
        adapter.step(clean_prefix=self.prefix)
        snapshot = adapter.snapshot()
        parent_before, rng_before = self.runtime.branch_info("parent"), adapter.rng_state
        for method in (adapter.restore, adapter.fork):
            with self.assertRaises(StateError):
                method(snapshot, "parent", seed=9)
            self.assertEqual(self.runtime.branch_info("parent"), parent_before)
            self.assertTrue(torch.equal(adapter.rng_state, rng_before))
            self.assertEqual(adapter.snapshot(), snapshot)

    def test_lossy_capture_is_direct_clean_normalized_pre_rope(self):
        specs = {
            MatrixKey(layer, target): CodecSpec("svd", rank=2 if target == "K" else 3)
            for layer in range(2) for target in ("K", "V")
        }
        adapter = self.adapter(specs=specs)
        captures = {}
        for layer, block in enumerate(self.pipe.generator.model.blocks):
            def observe(_module, _inputs, output, layer=layer):
                if adapter._phase == "clean_forward":
                    captures[layer] = output[0].detach().clone()
            handle = block.self_attn.norm_k.register_forward_hook(observe)
            self.addCleanup(handle.remove)
        seal = self.runtime.seal_chunk
        starts = []

        def inspect_seal(branch, chunk, matrices, contexts, requested, **kwargs):
            for layer in range(2):
                key = MatrixKey(layer, "K")
                self.assertTrue(torch.equal(captures[layer], matrices[key]))
                self.assertEqual(contexts[key].key_space, "pre_rope")
                if adapter.token_cursor:
                    self.assertFalse(torch.equal(
                        adapter._mutable[key][0].flatten(1), matrices[key],
                    ))
            starts.append(contexts[MatrixKey(0, "K")].token_start)
            return seal(branch, chunk, matrices, contexts, requested, **kwargs)

        with patch.object(self.runtime, "seal_chunk", side_effect=inspect_seal):
            adapter.step(clean_prefix=self.prefix)
            adapter.step()
            adapter.step()
        self.assertEqual(starts, [0, CONFIG.chunk_tokens, 2 * CONFIG.chunk_tokens])
        self.assertEqual(adapter.memory_report()["live_clean_capture_cpu_bytes"], 0)

    def test_unrecognized_key_processing_is_rejected_before_lossy_seal(self):
        self.pipe.generator.model.blocks[1].self_attn.key_scale = 2
        adapter = self.adapter(specs=CodecSpec("svd", rank=2))
        with self.assertRaisesRegex(IntegrityError, "pre-RoPE"):
            adapter.step(clean_prefix=self.prefix)
        self.assertEqual(adapter.frame_cursor, 0)
        self.assertEqual(self.runtime.branch_info("parent")["token_end"], 0)
        self.assertEqual(adapter.memory_report()["live_mutable_kv_bytes"], 0)
        self.assertEqual(adapter.memory_report()["live_clean_capture_cpu_bytes"], 0)

    def test_input_and_returned_state_have_no_mutable_parent_alias(self):
        rng_before = self.rng.get_state().clone()
        adapter = self.adapter()
        self.noise.zero_()
        self.conditioning["prompt_embeds"].zero_()
        adapter.step(clean_prefix=self.prefix)
        snapshot = adapter.snapshot()
        adapter.latent_state.zero_()
        adapter.rng_state.zero_()
        self.assertEqual(adapter.snapshot(), snapshot)
        self.assertTrue(torch.equal(self.rng.get_state(), rng_before))
        self.assertFalse(torch.equal(adapter._noise, self.noise))
        self.assertFalse(torch.equal(adapter._conditioning["prompt_embeds"], self.conditioning["prompt_embeds"]))

    def test_partial_hook_install_and_vae_failure_cleanup(self):
        adapter = CausVidAdapter(
            self.pipe, self.runtime, "parent", config=CONFIG, specs=CodecSpec("svd", rank=2),
        )
        target = self.pipe.generator.model.blocks[1].self_attn.norm_k
        with patch.object(target, "register_forward_hook", side_effect=RuntimeError("injected hook failure")):
            with self.assertRaisesRegex(RuntimeError, "injected hook"):
                adapter.install()
        for block in self.pipe.generator.model.blocks:
            self.assertNotIn("forward", block.self_attn.__dict__)
            self.assertFalse(block.self_attn.norm_k._forward_hooks)
        self.assertIsNone(self.pipe.kv_cache1)
        adapter = self.adapter()
        adapter.step(clean_prefix=self.prefix)
        snapshot, cleared = adapter.snapshot(), self.pipe.vae.clear_count
        with patch.object(self.pipe.vae, "decode_to_pixel", side_effect=RuntimeError("injected VAE failure")):
            with self.assertRaisesRegex(RuntimeError, "injected VAE"):
                adapter.decode()
        self.assertEqual(self.pipe.vae.clear_count, cleared + 2)
        self.assertEqual(adapter.snapshot(), snapshot)

    def test_bounded_attention_rejects_instead_of_truncating_history(self):
        adapter = self.adapter(max_attention_bytes=500)
        adapter.step(clean_prefix=self.prefix)
        prior = adapter.snapshot()
        with self.assertRaisesRegex(CapacityError, "Complete native"):
            adapter.step()
        self.assertEqual(adapter.snapshot(), prior)

    def test_new_conditioning_invalidates_native_cross_attention(self):
        adapter = self.adapter()
        adapter.step(clean_prefix=self.prefix)
        self.assertTrue(all(row["is_init"] for row in self.pipe.crossattn_cache))
        adapter.set_conditioning({"prompt_embeds": self.conditioning["prompt_embeds"] + 2}, conditioning_id="child-prompt")
        self.assertTrue(all(not row["is_init"] for row in self.pipe.crossattn_cache))
        adapter.step()
        self.assertTrue(all(torch.equal(row["k"], torch.tensor(2.0)) for row in self.pipe.crossattn_cache))

    def test_import_and_help_are_model_free_and_do_not_initialize_cuda(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        code = (
            "import sys, torch; import reelcache.adapters.causvid; "
            "assert not torch.cuda.is_initialized(); "
            "assert not any(k.startswith('third_party.causvid') for k in sys.modules)"
        )
        subprocess.run([sys.executable, "-B", "-c", code], cwd=root, env=env, check=True, capture_output=True)
        result = subprocess.run(
            [sys.executable, "-B", "examples/native_causvid.py", "--help"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        self.assertIn("--model-code-root", result.stdout)

    def test_native_driver_cross_attention_binding_is_local_and_reversible(self):
        path = Path(__file__).resolve().parents[1] / "examples/native_causvid.py"
        spec = importlib.util.spec_from_file_location("causvid_driver_cpu_test", path)
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        arguments = [
            "run", "--model-code-root", "unused", "--base-model", "unused",
            "--checkpoint", "unused", "--config", "unused",
            "--case-manifest", "unused", "--run-id", "cpu-parser-only",
        ]
        self.assertEqual(driver.parser().parse_args(arguments).arms, ["raw"])
        self.assertEqual(
            driver.parser().parse_args(arguments + ["--output-root", "artifacts/native_causvid"]).output_root,
            Path("artifacts/native_causvid"),
        )
        self.assertEqual(
            driver.parser().parse_args(arguments + ["--arms", "raw", "svd", "svd_hevc"]).arms,
            ["raw", "svd", "svd_hevc"],
        )
        source = self.directory / "source.py"
        source.write_text("x = 1\n")
        provenance = {"source_files": {"sample": {"path": str(source), "sha256": driver.sha256(source)}}}
        driver.verify_sources(provenance)
        source.write_text("x = 2\n")
        with self.assertRaisesRegex(RuntimeError, "source files changed"):
            driver.verify_sources(provenance)
        original = FakeCrossAttention.forward
        global_attention = original.__globals__["flash_attention"]
        independent = FakePipeline()
        query = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4) / 20
        with driver.instance_cross_attention_sdpa(self.pipe):
            self.assertIs(original.__globals__["flash_attention"], global_attention)
            self.assertIs(independent.generator.model.blocks[0].cross_attn.forward.__func__, original)
            output = self.pipe.generator.model.blocks[0].cross_attn(query, query, None)
            self.assertTrue(torch.equal(output, driver._full_sdpa(query, query, query)))
            with self.assertRaisesRegex(ValueError, "unmasked"):
                driver._full_sdpa(query, query, query, k_lens=torch.tensor([1]))
        for block in self.pipe.generator.model.blocks:
            self.assertIs(block.cross_attn.forward.__func__, original)
            self.assertNotIn("forward", block.cross_attn.__dict__)
        native = copy.deepcopy(self.pipe)
        dense_reference(
            native, self.noise, self.conditioning,
            torch.Generator().set_state(self.rng.get_state()), self.prefix,
        )
        expected = [
            {"layer": layer, "post_rope_K": tensor_digest(row["k"][:, :CONFIG.chunk_tokens]),
             "V": tensor_digest(row["v"][:, :CONFIG.chunk_tokens])}
            for layer, row in enumerate(native.kv_cache1)
        ]
        adapter = self.adapter()
        sealed = adapter.step(clean_prefix=self.prefix)
        driver.verify_raw_chunk(self.runtime, sealed, expected)
        expected[0]["post_rope_K"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "clean-cache parity"):
            driver.verify_raw_chunk(self.runtime, sealed, expected)
        with patch.object(self.runtime.store, "references", wraps=self.runtime.store.references) as references:
            sizes = driver.branch_bytes(self.runtime, "parent")
        self.assertTrue(all(call.args[0].startswith("manifest:") for call in references.call_args_list))
        self.assertEqual(sizes["payload_bytes"], 2 * 2 * CONFIG.chunk_tokens * CONFIG.channels * 4)
        self.assertEqual(sizes["full_canonical_basis_bytes"], 0)
        self.assertEqual(
            sizes["total_decoder_bytes"],
            sizes["payload_bytes"] + sizes["full_canonical_basis_bytes"] + sizes["metadata_bytes"],
        )
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_driver_counts_full_basis_leaves_once(self):
        path = Path(__file__).resolve().parents[1] / "examples/native_causvid.py"
        spec = importlib.util.spec_from_file_location("causvid_driver_basis_accounting_test", path)
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        generator = torch.Generator().manual_seed(91)
        matrices = {}
        contexts = {}
        for layer in range(2):
            for target in ("K", "V"):
                key = MatrixKey(layer, target)
                value = torch.randn(CONFIG.chunk_tokens, CONFIG.channels, generator=generator)
                matrices[key] = torch.nn.functional.layer_norm(value, (CONFIG.channels,)) if target == "K" else value
                contexts[key] = TensorContext(
                    self.model, key, 0, CONFIG.frames_per_chunk, CONFIG.grid_height, CONFIG.grid_width,
                    "pre_rope" if target == "K" else "none",
                )
        self.runtime.create_branch("basis-accounting")
        chunk = self.runtime.seal_chunk(
            "basis-accounting", 0, matrices, contexts, CodecSpec("svd", rank=2),
        )
        tensor_ids = [row["tensor"] for row in self.runtime.store.get_manifest(chunk)["matrices"]]
        basis_ids = {self.runtime.store.get_manifest(oid)["encoded"]["basis"] for oid in tensor_ids}
        expected = sum(self.runtime.store.object_size(oid) for oid in basis_ids)
        with patch.object(self.runtime.store, "references", wraps=self.runtime.store.references) as references:
            sizes = driver.branch_bytes(self.runtime, "basis-accounting")
        self.assertTrue(all(call.args[0].startswith("manifest:") for call in references.call_args_list))
        self.assertEqual(sizes["full_canonical_basis_bytes"], expected)
        self.assertGreater(expected, 0)
        self.assertGreater(sizes["payload_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
