"""Fake native seams exercise ownership without importing or loading MAGI."""

from contextlib import contextmanager, nullcontext
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import torch

from reelcache.adapters.magi import MagiAdapter, MagiGeometry, _tensor_hash
from reelcache.distributed import DistributedOwner, PeerFailure
from reelcache.errors import ConfigurationError, StateError
from reelcache.types import CodecSpec, MatrixKey, ModelIdentity, TensorContext


def fake_rotary(value, cos, sin):
    left, right = value.chunk(2, dim=-1)
    cosine = cos[None, :, None, :]
    sine = sin[None, :, None, :]
    return torch.cat((left * cosine - right * sine, right * cosine + left * sine), dim=-1)


class FakeAttention(torch.nn.Module):
    def __init__(self, index, geometry):
        super().__init__()
        self.layer_number = index + 10
        self.geometry = geometry
        self.k_layernorm = torch.nn.Identity()
        self.observed_meta = []

    def get_k(self, mixed, cos, sin):
        g = self.geometry
        value = mixed.float().reshape(-1, 1, g.kv_heads, g.head_dim)
        normalized = self.k_layernorm(value)
        return fake_rotary(normalized.transpose(0, 1), cos, sin).to(torch.bfloat16).reshape(
            -1, g.kv_heads, g.head_dim,
        )

    def get_v(self, mixed):
        return (mixed * 0.7 + 0.25).to(torch.bfloat16).reshape(
            -1, self.geometry.kv_heads, self.geometry.head_dim,
        )

    def adjust_key_and_value_for_inference(self, key_and_value, inference_params, meta_args):
        self.observed_meta.append(meta_args)
        start = meta_args.slice_point * meta_args.clip_token_nums
        cache = inference_params.key_value_memory_dict
        if self.layer_number not in cache:
            cache[self.layer_number] = torch.empty(
                inference_params.max_sequence_length, self.geometry.layout.local_heads,
                self.geometry.head_dim * 2, dtype=key_and_value.dtype,
            )
        history = cache[self.layer_number][:start]
        if inference_params.update_kv_cache:
            cache[self.layer_number][start:start + self.geometry.tokens] = key_and_value[:self.geometry.tokens]
        merged = torch.cat((history, key_and_value), dim=0)
        return tuple(part.contiguous() for part in merged.chunk(2, dim=-1))


class FakeModel:
    def __init__(self, geometry):
        self.videodit_blocks = SimpleNamespace(layers=[
            SimpleNamespace(self_attention=FakeAttention(i, geometry)) for i in range(geometry.layers)
        ])


class FakeRuntime:
    def __init__(self, identity):
        self.model = identity
        self.branches = {}
        self.snapshots = {}
        self.requested_columns = []
        self.resident_bytes = 0
        self.peak_bytes = 0
        self.fail_publication = False
        self.pins = 0
        self.store = SimpleNamespace(
            get_manifest=self._manifest, transaction=nullcontext,
            get_ref=lambda name: self.branch_info(name.removeprefix("branches/"))["head_id"],
            lease=self._lease,
        )

    def branch_info(self, branch):
        chunks = self.branches[branch]
        return {
            "head_id": f"{branch}:{len(chunks)}",
            "token_end": next(iter(chunks[-1]["contexts"].values())).token_end if chunks else 0,
        }

    def _lease(self, *args, **kwargs):
        self.pins += 1
        active = True

        def release():
            nonlocal active
            if active:
                self.pins -= 1
                active = False
        return SimpleNamespace(release=release)

    def _manifest(self, object_id):
        parts = object_id.split("/")
        snapshot = self.snapshots[parts[0]]
        if len(parts) == 1:
            return {"metadata": copy.deepcopy(snapshot["metadata"]), "head": f"{object_id}/head"}
        if parts[1:] == ["head"]:
            return {"chunks": [f"{parts[0]}/chunks/{index}" for index in range(len(snapshot["chunks"]))]}
        chunk = snapshot["chunks"][int(parts[2])]
        if len(parts) == 3:
            first = next(iter(chunk["contexts"].values()))
            return {
                "chunk_id": chunk["id"], "token_start": first.token_start, "token_end": first.token_end,
                "matrices": [{
                    "key": key.to_dict(), "tensor": f"{object_id}/matrices/{key.layer}:{key.target}",
                } for key in chunk["contexts"]],
            }
        layer, target = parts[4].split(":")
        key = MatrixKey(int(layer), target)
        return {"context": chunk["contexts"][key].to_dict(), "spec": chunk["specs"][key].to_dict()}

    def create_branch(self, branch):
        if branch in self.branches:
            raise StateError("Branch already exists")
        self.branches[branch] = []

    def seal_chunk(self, branch, chunk_id, matrices, contexts, specs, **kwargs):
        if self.fail_publication:
            raise OSError("simulated publication failure")
        self.branches[branch].append({
            "id": chunk_id, "values": {key: matrices[key].clone() for key in matrices},
            "contexts": contexts.copy(), "specs": specs.copy(), "reuse_threshold": kwargs.get("reuse_threshold"),
        })
        return chunk_id

    @contextmanager
    def acquire_history(self, branch, key, *, token_start, token_end, device, dtype, columns):
        self.requested_columns.append((key, columns.start, columns.stop))
        pieces = []
        for chunk in self.branches[branch]:
            context = chunk["contexts"][key]
            if context.token_start == token_start and context.token_end == token_end:
                pieces.append(chunk["values"][key][:, columns].to(dtype=dtype, device=device))
        value = torch.cat(pieces).contiguous()
        count = value.numel() * value.element_size()
        self.resident_bytes += count
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes)
        try:
            yield SimpleNamespace(tensor=value)
        finally:
            self.resident_bytes -= count

    def snapshot(self, branch, *, metadata, tensors):
        identity = f"snapshot-{len(self.snapshots)}"
        self.snapshots[identity] = {
            "metadata": json.loads(json.dumps(metadata)),
            "tensors": {key: value.clone() for key, value in tensors.items()},
            "chunks": list(self.branches[branch]),
        }
        return identity

    def load_snapshot(self, snapshot):
        state = self.snapshots[snapshot]
        return SimpleNamespace(
            snapshot_id=snapshot, metadata=copy.deepcopy(state["metadata"]),
            tensors={key: value.clone() for key, value in state["tensors"].items()},
        )

    def restore(self, snapshot, branch, *, metadata_validator=None, validator=None):
        if branch in self.branches:
            raise StateError("Branch already exists")
        if metadata_validator is not None:
            preview = SimpleNamespace(
                snapshot_id=snapshot, branch_id=branch,
                metadata=copy.deepcopy(self.snapshots[snapshot]["metadata"]),
                tensors=MappingProxyType({}), rank_policy=None, rank_state=None,
            )
            if metadata_validator(preview) is not None:
                raise ConfigurationError("Snapshot metadata validator must return None")
        saved = self.load_snapshot(snapshot)
        state = SimpleNamespace(
            snapshot_id=snapshot, branch_id=branch, metadata=saved.metadata, tensors=saved.tensors,
        )
        if validator is not None and validator(state) is not None:
            raise ConfigurationError("Snapshot validator must return None")
        self.create_branch(branch)
        self.branches[branch] = list(self.snapshots[snapshot]["chunks"])
        return state

    fork = restore


class MagiAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.root = Path(__file__).resolve().parents[1] / ".test-artifacts" / f"magi-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.geometry = MagiGeometry(layers=2, kv_heads=2, head_dim=4, frames=1, height=2, width=3, world_size=1)
        self.identity = ModelIdentity("fake_magi", "test_source_hash", 2, 8, "global_KV_channels", "bfloat16")
        self.model = FakeModel(self.geometry)
        self.cache = SimpleNamespace(max_batch_size=1, max_sequence_length=24,
                                     key_value_memory_dict={}, update_kv_cache=True)
        self.runtime = FakeRuntime(self.identity)
        self.adapter = self._adapter()

    def tearDown(self):
        if self.adapter._attached:
            self.adapter.detach()
        shutil.rmtree(self.root)

    def _adapter(self, *, specs=CodecSpec(), reuse_threshold=None, **options):
        return MagiAdapter(
            self.model, self.cache, self.runtime, model_identity=self.identity,
            branch_id="parent", staging_dir=self.root, geometry=self.geometry,
            coordinator=DistributedOwner(expected_world_size=1),
            rotary_apply=fake_rotary, specs=specs, reuse_threshold=reuse_threshold, **options,
        )

    def _forward(self, chunk, *, clean, near=False, guarded=True, native_timestep=None):
        g = self.geometry
        count = g.tokens * (2 if near else 1)
        mixed = torch.arange(count * g.channels, dtype=torch.float32).reshape(count, 1, g.channels) / 17
        mixed += chunk * 0.3
        angles = torch.arange(count, dtype=torch.float32)[:, None] * 0.03 + chunk * 0.07
        angles = angles.expand(count, g.head_dim // 2)
        cos, sin = angles.cos(), angles.sin()
        meta = SimpleNamespace(
            slice_point=chunk, clip_token_nums=g.tokens, start_chunk_id=chunk,
            end_chunk_id=chunk + 1 + int(near), denoising_range_num=1 + int(near),
            extract_prefix_video_feature=clean, distill_nearly_clean_chunk=near,
            core_attn_params={"visibility": [0, (chunk + 1) * g.tokens]},
            timestep=torch.full(
                (1, 1 + int(near)),
                native_timestep if native_timestep is not None else 0.9999 if clean else 0.31 if near else 0.0,
            ),
        )
        results = []

        def execute():
            for layer in self.model.videodit_blocks.layers:
                attention = layer.self_attention
                key, value = attention.get_k(mixed, cos, sin), attention.get_v(mixed)
                results.append(attention.adjust_key_and_value_for_inference(
                    torch.cat((key, value), -1), self.cache, meta,
                ))

        if guarded:
            with self.adapter.execution(chunk_index=chunk, clean=clean, clean_timestep=0.9999 if clean else None):
                execute()
        else:
            execute()
        return results, meta

    def _state(self):
        return {
            "metadata": {
                "conditioning_identity": "caption_hash",
                "execution_config": {"cfg": 1, "steps": 16, "near": [12, 13, 14, 15]},
                "vae_contract": {"mode": "stateless_chunk_decode", "temporal_tile_overlap": 0},
            },
            "tensors": {
                "latent": torch.arange(12, dtype=torch.float32),
                "conditioning": torch.arange(8, dtype=torch.float32),
                "conditioning_mask": torch.ones(8, dtype=torch.int32),
            },
        }

    def _restore_observation(self, generators):
        metadata, tensors = self.adapter._capture_rng(generators)
        native = {
            "branch": self.adapter.branch_id, "phase": self.adapter.phase,
            "cursor": self.adapter.cursor, "branch_created": self.adapter._branch_created,
            "specs": self.adapter.specs.copy(), "specs_id": id(self.adapter.specs),
            "reuse_threshold": self.adapter.reuse_threshold,
            "chunks": copy.deepcopy(self.adapter._chunks), "chunks_id": id(self.adapter._chunks),
            "stats": self.adapter._stats.copy(),
            "source_hashes": copy.deepcopy(self.adapter.canonical_source_hashes),
            "ropes": {name: [_tensor_hash(value) for value in pair]
                      for name, pair in self.adapter._ropes.items()},
            "pending_id": id(self.adapter._pending), "attached": self.adapter._attached,
            "cache_state": {name: value for name, value in vars(self.cache).items()
                            if name != "key_value_memory_dict"},
            "cache_id": id(self.cache.key_value_memory_dict),
            "cache_tensors": {key: (id(value), _tensor_hash(value))
                             for key, value in self.cache.key_value_memory_dict.items()},
        }
        return native, metadata, {name: _tensor_hash(value) for name, value in tensors.items()}

    def _assert_restore_rejected(self, snapshot, branch, *, operation="restore", generators=None,
                                 message="profile"):
        generators = generators or {}
        before = self._restore_observation(generators)
        with patch.object(self.runtime, "fork", wraps=self.runtime.fork) as publish, patch.object(
            self.adapter, "_restore_rng", wraps=self.adapter._restore_rng,
        ) as restore_rng:
            with self.assertRaisesRegex(PeerFailure, message):
                getattr(self.adapter, operation)(snapshot, branch, generators=generators)
            self.assertEqual(publish.call_count, 1)
            self.assertTrue(callable(publish.call_args.kwargs["validator"]))
            restore_rng.assert_not_called()
        self.assertEqual(self._restore_observation(generators), before)

    def test_restore_and_fork_reject_requested_profile_changes_before_publication(self):
        import numpy as np

        saved_spec = CodecSpec(kind="svd", rank=3, normalization="leading_columns", protected_fraction=0.5)
        self.adapter = self._adapter(specs=saved_spec, reuse_threshold=0.125)
        self.adapter.attach()
        self._forward(0, clean=True)
        self.adapter.seal_chunk()
        generator = torch.Generator().manual_seed(17)
        snapshot = self.adapter.snapshot(**self._state(), generators={"noise": generator})
        source_chunks = self.runtime.branches["parent"]
        source_values = {key: _tensor_hash(value) for key, value in source_chunks[0]["values"].items()}
        self.adapter.detach()
        variants = (
            ("codec_domain", CodecSpec(), 0.125),
            ("codec", replace(saved_spec, kind="svd_hevc"), 0.125),
            ("rank", replace(saved_spec, rank=2), 0.125),
            ("normalization", replace(saved_spec, normalization="global"), 0.125),
            ("protected_fraction", replace(saved_spec, protected_fraction=0.75), 0.125),
            ("pixel_format", replace(saved_spec, pix_fmt="yuv420p"), 0.125),
            ("qp", replace(saved_spec, qp=9), 0.125),
            ("preset", replace(saved_spec, preset="fast"), 0.125),
            ("reuse_disabled", saved_spec, None),
            ("reuse_threshold", saved_spec, 0.25),
        )
        for operation in ("restore", "fork"):
            for name, spec, threshold in variants:
                with self.subTest(operation=operation, mismatch=name):
                    self.adapter = self._adapter(specs=spec, reuse_threshold=threshold)
                    self.adapter.attach()
                    self.cache.sequence_len_offset = 7
                    self.cache.update_kv_cache = False
                    torch.manual_seed(211)
                    random.seed(311)
                    np.random.seed(411)
                    generator.manual_seed(511)
                    branch = f"rejected_{operation}_{name}"
                    self._assert_restore_rejected(
                        snapshot, branch, operation=operation, generators={"noise": generator},
                        message="reuse policy" if name.startswith("reuse") else "codec profile",
                    )
                    self.assertNotIn(branch, self.runtime.branches)
                    self.assertIs(self.runtime.branches["parent"], source_chunks)
                    self.assertEqual(
                        {key: _tensor_hash(value) for key, value in source_chunks[0]["values"].items()},
                        source_values,
                    )
                    self.adapter.detach()
        self.adapter = self._adapter(
            specs=replace(saved_spec, ffmpeg_path="/caller/ffmpeg"), reuse_threshold=0.125,
        )
        self.adapter.attach()
        requested = self.adapter.specs
        self.adapter.restore(snapshot, "same_profile", generators={"noise": generator})
        self.assertIs(self.adapter.specs, requested)
        self.assertTrue(all(spec.ffmpeg_path == "/caller/ffmpeg" for spec in self.adapter.specs.values()))
        self.assertEqual(self.adapter.reuse_threshold, 0.125)
        self.adapter.fork(snapshot, "same_profile_fork", generators={"noise": generator})
        self.assertIs(self.adapter.specs, requested)

    def test_configured_reuse_policy_and_one_commit_overrides_are_explicit(self):
        for threshold in (None, 0.125):
            with self.subTest(configured=threshold):
                self.adapter = self._adapter(
                    specs=CodecSpec(kind="svd", rank=3), reuse_threshold=threshold,
                )
                self.adapter.branch_id = f"reuse_{threshold}"
                self.adapter.attach()
                self._forward(0, clean=True)
                self.adapter.seal_chunk(reuse_threshold=0.5)
                self.assertEqual(self.runtime.branches[self.adapter.branch_id][0]["reuse_threshold"], 0.5)
                self.assertEqual(self.adapter.reuse_threshold, threshold)
                self._forward(1, clean=True)
                self.adapter.seal_chunk()
                self.assertEqual(self.runtime.branches[self.adapter.branch_id][1]["reuse_threshold"], threshold)
                snapshot = self.adapter.snapshot(**self._state())
                self.assertEqual(self.runtime.snapshots[snapshot]["metadata"]["reuse_threshold"], threshold)
                self.adapter.restore(snapshot, f"reuse_restored_{threshold}")
                self.assertEqual(self.adapter.reuse_threshold, threshold)
                self.adapter.detach()
        for invalid in (-1, float("nan"), float("inf"), True, "0.1"):
            with self.subTest(invalid=invalid), self.assertRaises(ConfigurationError):
                self._adapter(reuse_threshold=invalid)

    def test_restore_preserves_portable_profile_with_legacy_ignored_fraction(self):
        self.adapter.attach()
        self._forward(0, clean=True)
        self.adapter.seal_chunk()
        snapshot = self.adapter.snapshot(**self._state())
        metadata = self.runtime.snapshots[snapshot]["metadata"]
        metadata.pop("reuse_threshold")
        for spec in metadata["specs"].values():
            spec["protected_fraction"] = 0
            spec["ffmpeg_path"] = "/old/ffmpeg"
        self.adapter.detach()
        self.adapter = self._adapter(specs=CodecSpec(ffmpeg_path="/new/ffmpeg"))
        self.adapter.attach()
        requested = self.adapter.specs
        self.adapter.restore(snapshot, "portable_legacy_profile")
        self.assertIs(self.adapter.specs, requested)
        self.assertTrue(all(spec.ffmpeg_path == "/new/ffmpeg" for spec in requested.values()))
        self.assertEqual(self.runtime.snapshots[snapshot]["metadata"]["specs"]["0:K"]["protected_fraction"], 0)

    def test_profile_rejection_preserves_unsealed_native_staging(self):
        self.adapter = self._adapter(specs=CodecSpec(kind="svd", rank=3))
        self.adapter.attach()
        self._forward(0, clean=True)
        self.adapter.seal_chunk()
        generator = torch.Generator().manual_seed(61)
        snapshot = self.adapter.snapshot(**self._state(), generators={"noise": generator})
        self.adapter.specs = {key: CodecSpec(kind="svd", rank=2) for key in self.adapter.keys}
        self._forward(1, clean=True)
        pending = self.adapter._pending
        staged = {key: row["path"].read_bytes() for key, row in pending.entries.items()}
        generator.manual_seed(161)
        for operation in ("restore", "fork"):
            branch = f"reject_with_staging_{operation}"
            self._assert_restore_rejected(
                snapshot, branch, operation=operation, generators={"noise": generator},
                message="codec profile",
            )
            self.assertNotIn(branch, self.runtime.branches)
            self.assertIs(self.adapter._pending, pending)
            self.assertEqual(self.adapter.phase, "clean_unsealed")
            self.assertEqual(
                {key: row["path"].read_bytes() for key, row in pending.entries.items()}, staged,
            )
        self.adapter.seal_chunk()
        self.assertEqual(self.adapter.cursor, 2)

    def test_real_core_raw_to_factor_rejection_preserves_branches_native_state_and_rng(self):
        from reelcache.manager import KVRuntime

        self.runtime = KVRuntime(self.root / "profile_store", self.identity, max_resident_bytes=1024**2)
        self.adapter = self._adapter()
        generator = torch.Generator().manual_seed(71)
        try:
            self.adapter.attach()
            captured, _ = self._forward(0, clean=True)
            self.adapter.seal_chunk()
            snapshot = self.adapter.snapshot(**self._state(), generators={"noise": generator})
            parent_head = self.runtime.store.get_ref("branches/parent")
            self.runtime.create_branch("existing")
            existing_head = self.runtime.store.get_ref("branches/existing")
            self.adapter.detach()
            self.adapter = self._adapter(specs=CodecSpec(kind="svd", rank=3))
            self.adapter.attach()
            torch.manual_seed(191)
            generator.manual_seed(291)
            for operation in ("restore", "fork"):
                name = f"raw_into_factor_{operation}"
                self._assert_restore_rejected(
                    snapshot, name, operation=operation, generators={"noise": generator},
                    message="codec profile",
                )
                self.assertIsNone(self.runtime.store.get_ref(f"branches/{name}"))
                self.assertEqual(self.runtime.store.get_ref("branches/parent"), parent_head)
                self.assertEqual(self.runtime.store.get_ref("branches/existing"), existing_head)
            self.assertEqual(self.runtime.active_leases, 0)
            self.adapter.detach()
            self.adapter = self._adapter()
            self.adapter.attach()
            self.adapter.restore(snapshot, "same_raw_profile", generators={"noise": generator})
            restored, _ = self._forward(1, clean=False)
            for expected, actual in zip(captured, restored):
                self.assertTrue(torch.equal(expected[0], actual[0][:6]))
                self.assertTrue(torch.equal(expected[1], actual[1][:6]))
        finally:
            self.adapter.detach()
            self.runtime.close()

    def test_real_core_source_domain_mismatch_is_rejected_before_publication(self):
        from reelcache.manager import KVRuntime

        self.runtime = KVRuntime(self.root / "domain_store", self.identity, max_resident_bytes=1024**2)
        self.adapter = self._adapter()
        generator = torch.Generator().manual_seed(73)
        try:
            self.adapter.attach()
            captured, _ = self._forward(0, clean=True)
            self.adapter.seal_chunk()
            snapshot = self.adapter.snapshot(**self._state(), generators={"noise": generator})
            saved = self.runtime.load_snapshot(snapshot)
            parent_head = self.runtime.store.get_ref("branches/parent")
            for label, space, dtype in (
                ("key_space", "pre_rope", "bfloat16"),
                ("source_dtype", "post_rope", "float32"),
            ):
                matrices, contexts = {}, {}
                for layer, pair in enumerate(captured):
                    for index, target in enumerate(("K", "V")):
                        key = MatrixKey(layer, target)
                        source = dtype if target == "K" else "bfloat16"
                        matrices[key] = pair[index].reshape(6, 8).to(dtype=getattr(torch, source))
                        contexts[key] = TensorContext(
                            self.identity, key, 0, 1, 2, 3, space if target == "K" else "none",
                            source_dtype=source if source != "bfloat16" else None,
                        )
                self.runtime.create_branch(label)
                self.runtime.seal_chunk(label, "0", matrices, contexts, CodecSpec())
                incompatible = self.runtime.snapshot(label, metadata=saved.metadata, tensors=saved.tensors)
                source_head = self.runtime.store.get_ref(f"branches/{label}")
                torch.manual_seed(193)
                generator.manual_seed(293)
                for operation in ("restore", "fork"):
                    name = f"rejected_{label}_{operation}"
                    self._assert_restore_rejected(
                        incompatible, name, operation=operation, generators={"noise": generator},
                        message="source domain",
                    )
                    self.assertIsNone(self.runtime.store.get_ref(f"branches/{name}"))
                    self.assertEqual(self.runtime.store.get_ref("branches/parent"), parent_head)
                    self.assertEqual(self.runtime.store.get_ref(f"branches/{label}"), source_head)
            self.adapter.restore(snapshot, "valid_domain", generators={"noise": generator})
            self.assertEqual(self.adapter.cursor, 1)
            self.assertEqual(self.runtime.active_leases, 0)
        finally:
            self.adapter.detach()
            self.runtime.close()

    def test_native_driver_constructs_explicit_fixed_arm_profiles(self):
        from reelcache.adapters.magi_native import _arm_specs

        for arm in ("raw", "svd", "svd_hevc"):
            specs = _arm_specs(arm)
            self.assertEqual(len(specs), 96)
            for key, spec in specs.items():
                self.assertEqual(spec.kind, arm)
                self.assertEqual(spec.rank, None if arm == "raw" else 256 if key.target == "K" else 384)
                self.assertEqual(spec.normalization, "per_channel")
                self.assertEqual(spec.protected_fraction, 1)
        with self.assertRaises(ValueError):
            _arm_specs("implicit_saved_profile")

    def test_raw_noop_resume_is_exact_and_native_buffers_are_released(self):
        self._forward(0, clean=True, guarded=False)
        expected, _ = self._forward(1, clean=False, near=True, guarded=False)
        self.cache.key_value_memory_dict.clear()
        self.adapter.attach()
        self._forward(0, clean=True)
        self.adapter.seal_chunk()
        snapshot = self.adapter.snapshot(**self._state())
        metadata, state = self.adapter.restore(snapshot, "resumed")
        self.assertEqual(metadata["cursor"], 1)
        self.assertTrue(torch.equal(state["latent"], self._state()["tensors"]["latent"]))
        observed, meta = self._forward(1, clean=False, near=True)
        for reference, restored in zip(expected, observed):
            for left, right in zip(reference, restored):
                self.assertTrue(torch.equal(left, right))
        for layer in self.model.videodit_blocks.layers:
            self.assertIs(layer.self_attention.observed_meta[-1], meta)
        report = self.adapter.memory_report()
        self.assertEqual(report["live_native_cache_storage_bytes"], 0)
        self.assertEqual(report["native_cache_entries"], 0)
        self.assertEqual(self.runtime.resident_bytes, 0)
        self.assertFalse(report["query_head_expansion"])
        self.assertEqual(report["local_kv_heads"], 2)
        self.assertFalse(torch.cuda.is_initialized())

    def test_hooks_are_instance_local_and_reversible(self):
        other = FakeModel(self.geometry)
        attention = self.model.videodit_blocks.layers[0].self_attention
        original = attention.get_k.__func__
        untouched = other.videodit_blocks.layers[0].self_attention.get_k.__func__
        self.adapter.attach()
        self.assertNotIn("get_k", other.videodit_blocks.layers[0].self_attention.__dict__)
        self.assertIs(untouched, original)
        with self.assertRaises(StateError):
            self._adapter().attach()
        self.adapter.detach()
        self.assertNotIn("get_k", attention.__dict__)
        self.assertIs(attention.get_k.__func__, original)
        self.assertEqual(len(attention.k_layernorm._forward_hooks), 0)
        self.assertNotIn("_reelcache_magi_adapter", self.model.__dict__)

    def test_lossy_capture_is_global_pre_rope_not_native_post_rope(self):
        reference, _ = self._forward(0, clean=True, guarded=False)
        self.cache.key_value_memory_dict.clear()
        self.adapter = self._adapter(specs=CodecSpec(kind="svd", rank=3))
        self.adapter.attach()
        self._forward(0, clean=True)
        self.adapter.seal_chunk()
        chunk = self.runtime.branches["parent"][0]
        kkey = MatrixKey(0, "K")
        self.assertEqual(chunk["contexts"][kkey].key_space, "pre_rope")
        self.assertEqual(chunk["contexts"][kkey].source_dtype, "float32")
        self.assertEqual(chunk["contexts"][kkey].model, self.identity)
        self.assertEqual(self.identity.dtype, "bfloat16")
        self.assertEqual(chunk["values"][kkey].dtype, torch.float32)
        self.assertEqual(chunk["values"][kkey].shape, (6, 8))
        self.assertFalse(torch.equal(chunk["values"][kkey].to(torch.bfloat16),
                                     reference[0][0].reshape(6, 8)))
        snapshot = self.adapter.snapshot(**self._state())
        self.adapter.restore(snapshot, "restored_pre_rope")
        continued, _ = self._forward(1, clean=False)
        self.assertTrue(torch.equal(continued[0][0][:6], reference[0][0]))
        self.assertTrue(all(start == 0 and stop == 8
                            for _, start, stop in self.runtime.requested_columns))

    def test_noisy_and_near_clean_boundaries_cannot_be_snapshotted(self):
        self.adapter.attach()
        with self.assertRaises(PeerFailure):
            with self.adapter.execution(chunk_index=0, clean=True, clean_timestep=0.3):
                pass
        self._forward(0, clean=True)
        with self.assertRaises(PeerFailure):
            self.adapter.snapshot(**self._state())
        self.adapter.seal_chunk()
        self._forward(1, clean=False, near=True)
        with self.assertRaises(PeerFailure):
            self.adapter.snapshot(**self._state())
        with self.assertRaises(PeerFailure):
            self.adapter.seal_chunk()
        self.assertEqual(len(self.runtime.branches["parent"]), 1)

    def test_raw_fork_isolation_later_chunk_and_rng_state(self):
        self.adapter.attach()
        self._forward(0, clean=True)
        self.adapter.seal_chunk()
        generator = torch.Generator().manual_seed(103)
        snapshot = self.adapter.snapshot(**self._state(), generators={"noise": generator})
        expected = (torch.rand(3), torch.rand(3, generator=generator), random.random())
        _, state = self.adapter.fork(snapshot, "child", generators={"noise": generator})
        actual = (torch.rand(3), torch.rand(3, generator=generator), random.random())
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(expected[1], actual[1]))
        self.assertEqual(expected[2], actual[2])
        state["conditioning"].zero_()
        self.assertFalse(torch.equal(
            state["conditioning"], self.runtime.snapshots[snapshot]["tensors"]["state/conditioning"],
        ))
        self._forward(1, clean=False)
        self._forward(1, clean=True)
        self.adapter.seal_chunk()
        later, _ = self._forward(2, clean=False)
        self.assertEqual(later[0][0].shape, (18, 2, 4))
        self.assertEqual(len(self.runtime.branches["parent"]), 1)
        self.assertEqual(len(self.runtime.branches["child"]), 2)
        self.assertEqual(len(self.runtime.snapshots[snapshot]["chunks"]), 1)

    def test_interrupted_publication_keeps_staging_and_prior_snapshot(self):
        self.adapter.attach()
        self._forward(0, clean=True)
        self.adapter.seal_chunk()
        snapshot = self.adapter.snapshot(**self._state())
        self._forward(1, clean=True)
        self.runtime.fail_publication = True
        with self.assertRaises(PeerFailure):
            self.adapter.seal_chunk()
        self.assertEqual(self.adapter.phase, "clean_unsealed")
        self.assertGreater(self.adapter.memory_report()["transient_disk_staging_bytes"], 0)
        self.assertEqual(len(self.runtime.branches["parent"]), 1)
        self.adapter.restore(snapshot, "recovered")
        self.assertEqual(self.adapter.cursor, 1)
        self.assertEqual(self.adapter.memory_report()["transient_disk_staging_bytes"], 0)

    def test_vae_state_must_be_declared_and_complete(self):
        self.adapter.attach()
        self._forward(0, clean=True)
        self.adapter.seal_chunk()
        state = self._state()
        state["metadata"].pop("vae_contract")
        with self.assertRaises(PeerFailure):
            self.adapter.snapshot(**state)
        state["metadata"]["vae_contract"] = {"mode": "explicit_state", "tensor_names": ["vae_cache"]}
        with self.assertRaises(PeerFailure):
            self.adapter.snapshot(**state)
        state["tensors"]["vae_cache"] = torch.arange(3)
        snapshot = self.adapter.snapshot(**state)
        _, restored = self.adapter.restore(snapshot, "vae_restored")
        self.assertTrue(torch.equal(restored["vae_cache"], state["tensors"]["vae_cache"]))

    def test_caller_clean_label_cannot_override_actual_native_timestep(self):
        self.adapter.attach()
        self._forward(0, clean=True)
        self.adapter.seal_chunk()
        snapshot = self.adapter.snapshot(**self._state())
        with self.assertRaises(PeerFailure):
            self._forward(1, clean=True, native_timestep=0.3)
        self.assertEqual(self.adapter.phase, "failed")
        with self.assertRaises(PeerFailure):
            self.adapter.seal_chunk()
        self.assertEqual(len(self.runtime.branches["parent"]), 1)
        self.adapter.restore(snapshot, "valid_clean_boundary")
        self.assertEqual(self.adapter.cursor, 1)

    def test_inflight_snapshot_and_layer_budget_are_rejected(self):
        self.adapter.attach()
        self._forward(0, clean=True)
        self.adapter.seal_chunk()
        snapshot = self.adapter.snapshot(**self._state())
        with self.assertRaises(PeerFailure):
            with self.adapter.execution(chunk_index=1, clean=False):
                self.adapter.snapshot(**self._state())
        self.assertEqual(self.adapter.phase, "failed")
        self.adapter.restore(snapshot, "rollback")
        self.adapter.max_layer_buffer_bytes = 1
        with self.assertRaises(PeerFailure):
            self._forward(1, clean=False)
        self.assertEqual(self.runtime.resident_bytes, 0)
        self.assertEqual(self.adapter.memory_report()["live_native_cache_storage_bytes"], 0)
        self.assertEqual(len(self.runtime.branches["parent"]), 1)

    def test_real_core_raw_restart_restores_bounded_history(self):
        from reelcache.manager import KVRuntime

        self.runtime = KVRuntime(self.root / "store", self.identity, max_resident_bytes=1024**2)
        self.adapter = self._adapter()
        try:
            self.adapter.attach()
            captured, _ = self._forward(0, clean=True)
            self.adapter.seal_chunk()
            snapshot = self.adapter.snapshot(**self._state())
            self.adapter.detach()
            self.runtime.close()
            self.runtime = KVRuntime(self.root / "store", self.identity, max_resident_bytes=1024**2)
            self.adapter = self._adapter()
            self.adapter.attach()
            self.adapter.restore(snapshot, "after_restart")
            restored, _ = self._forward(1, clean=False)
            for original, actual in zip(captured, restored):
                self.assertTrue(torch.equal(original[0], actual[0][:6]))
                self.assertTrue(torch.equal(original[1], actual[1][:6]))
            self.assertEqual(self.runtime.resident_bytes, 0)
            self.assertEqual(self.runtime.active_leases, 0)
            self.assertGreater(self.runtime.residency.peak_reserved_bytes, 0)
            self.assertEqual(self.runtime.branch_info("parent")["chunk_ids"], ["0"])
            self.assertEqual(self.adapter.memory_report()["native_cache_entries"], 0)
        finally:
            self.adapter.detach()
            self.runtime.close()

    def test_real_core_svd_hevc_precision_and_shared_canonical_bases(self):
        from reelcache.adapters.magi_native import _precision_preflight, _storage
        from reelcache.manager import KVRuntime

        precision = _precision_preflight(["raw", "svd", "svd_hevc"])
        self.assertEqual(precision["pre_rope_source_dtype"], "float32")
        self.assertTrue(precision["native_identity_unchanged_across_representations"])
        self.runtime = KVRuntime(self.root / "native_store", self.identity, max_resident_bytes=1024**2)
        bases = {}
        snapshots = {}
        try:
            for kind in ("raw", "svd", "svd_hevc"):
                spec = CodecSpec() if kind == "raw" else CodecSpec(kind=kind, rank=3)
                self.adapter = self._adapter(specs=spec)
                self.adapter.branch_id = kind
                self.adapter.attach()
                self._forward(0, clean=True)
                self.adapter.seal_chunk()
                snapshot = self.adapter.snapshot(**self._state())
                snapshots[kind] = snapshot
                storage = _storage(self.runtime, kind, snapshot)
                self.assertEqual(storage["native_history_bytes_across_SP2"], 6 * 8 * 2 * 2 * 2)
                self.assertEqual(storage["canonical_matrix_input_bytes"], 6 * 8 * (4 if kind == "raw" else 6) * 2)
                self.assertEqual(
                    storage["ratio_native_history_to_unique_objects"],
                    storage["native_history_bytes_across_SP2"] / storage["reachable_unique_bytes"],
                )
                self.adapter.restore(snapshot, f"{kind}_restored")
                output, _ = self._forward(1, clean=False)
                self.assertTrue(all(torch.isfinite(value).all() for pair in output for value in pair))
                self.assertEqual(output[0][0].dtype, torch.bfloat16)
                self.assertEqual(self.runtime.resident_bytes, 0)
                with self.runtime.acquire_history(
                    kind, MatrixKey(0, "K"), columns=slice(0, 4),
                ) as lease:
                    self.assertEqual(lease.tensor.shape, (6, 4))
                    self.assertEqual(lease.tensor.dtype, torch.bfloat16 if kind == "raw" else torch.float32)
                chunk = self.runtime.store.get_manifest(self.runtime.branch_info(kind)["chunk_manifests"][0])
                packets = [self.runtime.store.get_manifest(item["tensor"]) for item in chunk["matrices"]]
                bases[kind] = [item["encoded"]["basis"] for item in packets]
                self.assertTrue(all(item["context"]["key_space"] == ("post_rope" if kind == "raw" else "pre_rope")
                                    for item in packets if item["context"]["key"]["target"] == "K"))
                for packet in packets:
                    context = packet["context"]
                    self.assertEqual(context["model"], self.identity.to_dict())
                    if kind != "raw" and context["key"]["target"] == "K":
                        self.assertEqual(context["source_dtype"], "float32")
                    else:
                        self.assertNotIn("source_dtype", context)
                self.adapter.detach()
            self.assertEqual(bases["svd"], bases["svd_hevc"])
            self.assertEqual(self.runtime.active_leases, 0)
            self.runtime.close()
            self.runtime = KVRuntime(self.root / "native_store", self.identity, max_resident_bytes=1024**2)
            self.adapter = self._adapter(specs=CodecSpec(kind="svd", rank=3))
            self.adapter.attach()
            self.adapter.restore(snapshots["svd"], "unified_after_restart")
            output, _ = self._forward(1, clean=False)
            self.assertTrue(torch.isfinite(output[0][0]).all())
            self.assertEqual(self.runtime.model.dtype, "bfloat16")
            self.assertEqual(self.runtime.branch_info("raw")["chunk_ids"], ["0"])
            self.assertFalse(torch.cuda.is_initialized())
        finally:
            self.adapter.detach()
            self.runtime.close()

    def test_context_source_override_preserves_legacy_declared_dtype_maps(self):
        from reelcache.adapters.magi import _tensor_context

        key = MatrixKey(0, "K")
        native_context = _tensor_context(self.identity, key, 0, self.geometry, "post_rope", "bfloat16")
        factor_context = _tensor_context(self.identity, key, 0, self.geometry, "pre_rope", "float32")
        legacy = ModelIdentity("fake_magi", "test_source_hash", 2, 8, "global_KV_channels",
                               {"K": "float32", "V": "bfloat16"})
        legacy_context = _tensor_context(legacy, key, 0, self.geometry, "pre_rope", "float32")
        self.assertNotIn("source_dtype", native_context.to_dict())
        self.assertNotIn("source_dtype", legacy_context.to_dict())
        self.assertEqual(factor_context.to_dict()["source_dtype"], "float32")
        self.assertEqual(native_context.model, factor_context.model)
        self.assertEqual(legacy_context.dtype, factor_context.dtype)

    def test_import_help_and_no_grant_never_import_native_model(self):
        project = Path(__file__).resolve().parents[1]
        environment = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "",
                       "TMPDIR": str(self.root), "PYTHONDONTWRITEBYTECODE": "1"}
        probe = """
import runpy, sys
runpy.run_path('examples/native_magi.py', run_name='import_probe')
assert 'torch' not in sys.modules
import reelcache.adapters.magi
import reelcache.adapters.magi_native
assert 'inference' not in sys.modules
assert 'transformers' not in sys.modules
assert 'torch' not in sys.modules or not sys.modules['torch'].cuda.is_initialized()
assert 'native_magi_sp_bench' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-B", "-c", probe], cwd=project,
                                env=environment, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        help_result = subprocess.run(
            [sys.executable, "-B", "examples/native_magi.py", "--help"],
            cwd=project, env=environment, text=True, capture_output=True, timeout=60,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        denied = subprocess.run(
            [sys.executable, "-B", "examples/native_magi.py", "run", "--output", str(self.root)],
            cwd=project, env=environment, text=True, capture_output=True, timeout=60,
        )
        self.assertNotEqual(denied.returncode, 0)
        self.assertIn("forbidden without an explicit parent grant", denied.stderr)


if __name__ == "__main__":
    unittest.main()
