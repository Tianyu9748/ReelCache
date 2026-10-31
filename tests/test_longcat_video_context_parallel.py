"""CPU tests of native LongCat-Video context parallelism (CP) in the port and of the per-rank cache seam.

No GPU and no SGLang (the SGLang side -- ``validate_server_args`` and CP resolved from SGLang's SP group -- is
checked by ``artifacts/.../step13_longcat/phase4/cpu_stage_cp.py`` in the sglang-cu132 env). The native LongCat-Video modules (hash-checked, see ``test_longcat_video_port``) run with their own
``context_parallel_util`` / ``ulysses_wrapper`` code on a Gloo group (torch SDPA as the attention kernel, inside the
native Ulysses wrapper), next to the port's :mod:`reelcache_sglang_models.longcat_video.context_parallel`:

* geometry: ``optimal_split`` == native ``get_optimal_split``; a rank's rows == native ``split_tensor_in_cp_2d`` ==
  ``reelcache.factor_sharding.NativeOwners.spatial_rows``; split/assemble round trip; CP RoPE rows == native CP RoPE;
* per-rank seam: ``RowShard`` epochs, local-token validation, installs from ``[T_local, C]`` rows and native
  ``[1, H, T_local, D]`` tensors, shard mismatches rejected, the single-GPU ReelCache backend refuses shards;
* 2 ranks (split 1x2) and 4 ranks (2x2) on Gloo: plain, conditioned, cache-build and cache-read forwards of the
  port under CP equal native CP bitwise; against one process the layer-0 K/V rows are bitwise equal and the rest
  agree within FP32 rounding; the Ulysses exchange equals native ``all_to_all``; installing every rank's rows
  through the seam reproduces the cache-read forward bitwise; a factor-sharded restore (``FactorShardDomain``
  owner fit over the gathered rows, ``restore_node(..., NativeOwners.spatial_rows(...), rope=None)``) installs
  through the seam and reproduces it within the FP16-basis rounding.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import sys
import types
import unittest
import uuid
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from reelcache_sglang_models.longcat_video import conditioning_cache as cc
from reelcache_sglang_models.longcat_video import context_parallel as lcp
from reelcache_sglang_models.longcat_video import dit_core, rope

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))
import test_longcat_video_port as base  # noqa: E402  (native-module loader with source hash check, tiny model)

TINY = base.TINY
GRID = (6, 4, 6)  # latents [1, 16, 6, 8, 12]: 2 conditioning + 4 new frames, 4 x 6 tokens per frame
COND_GRID = (2, 4, 6)


def _native_cp_module():
    base.native()
    from longcat_video.context_parallel import context_parallel_util, ulysses_wrapper

    return context_parallel_util, ulysses_wrapper


class _NativeCPState:
    """Set native ``context_parallel_util`` globals for a rank (``init_context_parallel`` needs a CUDA mesh)."""

    NAMES = ("cp_size", "cp_rank", "cp_group", "cp_ranks", "dp_size", "dp_rank")

    def __init__(self, size: int, rank: int, group=None):
        self.cpu, _ = _native_cp_module()
        self.values = {"cp_size": size, "cp_rank": rank, "cp_group": group,
                       "cp_ranks": list(range(size)), "dp_size": 1, "dp_rank": 0}

    def __enter__(self):
        self.saved = {name: getattr(self.cpu, name) for name in self.NAMES}
        for name, value in self.values.items():
            setattr(self.cpu, name, value)
        return self.cpu

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            setattr(self.cpu, name, value)


class GeometryTests(unittest.TestCase):
    def test_optimal_split_matches_native(self):
        cpu, _ = _native_cp_module()
        for size in range(1, 17):
            self.assertEqual(lcp.optimal_split(size), tuple(cpu.get_optimal_split(size)), size)
        self.assertEqual(lcp.optimal_split(2), (1, 2))

    def test_rows_match_native_split_and_native_owners(self):
        from reelcache.factor_sharding import NativeOwners

        for size, grid in ((2, (4, 44, 82)), (4, (2, 4, 6)), (8, (3, 4, 8))):
            split = lcp.optimal_split(size)
            index = torch.arange(grid[0] * grid[1] * grid[2]).reshape(grid)
            owners = NativeOwners.spatial_rows(frames=grid[0], height=grid[1], width=grid[2], split_h=split[0],
                                               split_w=split[1], kv_heads=32, head_dim=8)
            seen = []
            for rank in range(size):
                shard = lcp.RowShard(size, rank, split)
                rows = shard.rows(grid)
                with _NativeCPState(size, rank) as cpu:
                    native_rows = cpu.split_tensor_in_cp_2d(index, (1, 2), list(split)).reshape(-1)
                self.assertTrue(torch.equal(rows, native_rows), (size, rank))
                self.assertTrue(torch.equal(rows, owners.rows(rank)), (size, rank))
                kwargs = shard.native_owners_kwargs(grid, kv_heads=32, head_dim=8)
                self.assertTrue(torch.equal(NativeOwners.spatial_rows(**kwargs).rows(rank), rows))
                self.assertEqual(shard.local_tokens(grid), rows.numel())
                seen.append(rows)
            self.assertTrue(torch.equal(torch.sort(torch.cat(seen)).values, index.reshape(-1)))
        self.assertEqual(lcp.RowShard(2, 1, (1, 2)).local_grid((4, 44, 82)), (4, 44, 41))
        with self.assertRaises(lcp.ContextParallelError):
            lcp.RowShard(2, 0, (1, 2)).rows((4, 44, 81))
        with self.assertRaises(lcp.ContextParallelError):
            lcp.RowShard(2, 2, (1, 2))
        with self.assertRaises(lcp.ContextParallelError):
            lcp.RowShard(4, 0, (1, 2))

    def test_split_and_assemble_match_native(self):
        torch.manual_seed(3)
        x = torch.randn(1, 2 * 4 * 6, 5)
        for size in (2, 4):
            split = lcp.optimal_split(size)
            blocks = []
            for rank in range(size):
                shard = lcp.RowShard(size, rank, split)
                ours = lcp.split_tokens(x, (2, 4, 6), shard)
                with _NativeCPState(size, rank) as cpu:
                    ref = cpu.split_tensor_in_cp_2d(x.view(1, 2, 4, 6, 5), (2, 3), list(split)).reshape(1, -1, 5)
                self.assertTrue(torch.equal(ours, ref))
                self.assertTrue(torch.equal(ours[0], x[0].index_select(0, shard.rows((2, 4, 6)))))
                blocks.append(ours)
            self.assertTrue(torch.equal(lcp.assemble_tokens(blocks, (2, 4, 6), split), x))

    def test_cp_rope_rows_match_native_cp_rope(self):
        nat_rope = base.native()["rope"]
        tables = rope.RopeTables(TINY.head_dim)
        for size in (2, 4):
            split = lcp.optimal_split(size)
            for rank in range(size):
                with _NativeCPState(size, rank):
                    module = nat_rope.RotaryPositionalEmbedding(TINY.head_dim, cp_split_hw=list(split))
                    freqs = module.precompute_freqs_cis_3d(GRID)
                cos, sin = tables.cos_sin(GRID, "cpu", (split[0], split[1], rank))
                self.assertTrue(torch.equal(cos, freqs.float().cos()))
                self.assertTrue(torch.equal(sin, freqs.float().sin()))
                full_cos, _ = tables.cos_sin(GRID, "cpu")
                rows = lcp.RowShard(size, rank, split).rows(GRID)
                self.assertTrue(torch.allclose(cos, full_cos[rows], atol=1e-6, rtol=0))


class SeamTests(unittest.TestCase):
    def _epoch(self, shard=None):
        return cc.ConditioningCacheEpoch(0, COND_GRID, "a" * 64, "f" * 64, row_shard=shard)

    def test_row_shard_epoch_and_local_tokens(self):
        shard = lcp.RowShard(2, 1, (1, 2))
        epoch = self._epoch(shard)
        self.assertEqual(epoch.tokens, 48)
        self.assertEqual(epoch.local_tokens, 24)
        self.assertTrue(torch.equal(epoch.row_index(), shard.rows(COND_GRID)))
        self.assertEqual(epoch.as_dict()["row_shard"], shard.as_dict())
        self.assertEqual(epoch.full_epoch(), self._epoch())
        self.assertNotEqual(epoch, self._epoch(lcp.RowShard(2, 0, (1, 2))))
        with self.assertRaises(cc.ConditioningCacheError):
            cc.ConditioningCacheEpoch(0, (2, 4, 5), "a", "f", row_shard=shard)
        with self.assertRaises(cc.ConditioningCacheError):
            cc.ConditioningCacheEpoch(0, COND_GRID, "a", "f", row_shard=(2, 1))

    def test_installs_from_rows_and_native_layout(self):
        shard = lcp.RowShard(2, 0, (1, 2))
        epoch = self._epoch(shard)
        h, d = TINY.num_heads, TINY.head_dim
        gen = torch.Generator().manual_seed(5)
        rows_k = [torch.randn(24, h * d, generator=gen) for _ in range(2)]
        rows_v = [torch.randn(24, h * d, generator=gen) for _ in range(2)]
        cache = cc.LongCatConditioningCache.from_matrices(epoch, rows_k, rows_v, heads=h)
        accepted = cc.accept_restored_cache(cache, epoch, layers=2, heads=h, head_dim=d, dtype=torch.float32,
                                            device="cpu")
        self.assertEqual(tuple(accepted.keys_pre_rope[0].shape), (1, 24, h, d))
        for i in range(2):
            km, vm = accepted.layer_matrices(i)
            self.assertTrue(torch.equal(km, rows_k[i]) and torch.equal(vm, rows_v[i]))
        native = cc.LongCatConditioningCache.from_native_layout(
            epoch, [k.permute(0, 2, 1, 3) for k in accepted.keys_pre_rope],
            [v.permute(0, 2, 1, 3) for v in accepted.values])
        for i in range(2):
            self.assertTrue(torch.equal(native.keys_pre_rope[i], accepted.keys_pre_rope[i]))
        with self.assertRaises(cc.ConditioningCacheError):  # whole-cache rows for a shard epoch
            cc.LongCatConditioningCache.from_matrices(epoch, [torch.zeros(48, h * d)] * 2, [torch.zeros(48, h * d)] * 2,
                                                      heads=h)
        other = self._epoch(lcp.RowShard(2, 1, (1, 2)))
        with self.assertRaises(cc.ConditioningCacheError):  # another rank's shard
            cc.accept_restored_cache(cache, other, layers=2, heads=h, head_dim=d, dtype=torch.float32, device="cpu")
        whole = cc.LongCatConditioningCache.from_matrices(self._epoch(), [torch.zeros(48, h * d)] * 2,
                                                          [torch.zeros(48, h * d)] * 2, heads=h)
        with self.assertRaises(cc.ConditioningCacheError):  # a whole cache offered for a shard epoch
            cc.accept_restored_cache(whole, epoch, layers=2, heads=h, head_dim=d, dtype=torch.float32, device="cpu")

    def test_epoch_dict_roundtrip_and_row_seam(self):
        shard = lcp.RowShard(2, 1, (1, 2))
        epoch = self._epoch(shard)
        self.assertEqual(cc.ConditioningCacheEpoch.from_dict(epoch.as_dict()), epoch)
        self.assertEqual(cc.ConditioningCacheEpoch.from_dict(self._epoch().as_dict()), self._epoch())
        h, d = TINY.num_heads, TINY.head_dim
        gen = torch.Generator().manual_seed(9)
        built = cc.LongCatConditioningCache.from_matrices(
            epoch, [torch.randn(24, h * d, generator=gen) for _ in range(2)],
            [torch.randn(24, h * d, generator=gen) for _ in range(2)], heads=h, source=cc.SOURCE_NATIVE_BUILD)
        store = {}

        def capture(ep, keys, values):
            self.assertEqual([tuple(k.shape) for k in keys], [(24, h * d)] * 2)
            store[ep] = ([k.clone() for k in keys], [v.clone() for v in values])

        seam = cc.CacheRowSeam(heads=h, capture=capture, install=store.get)
        self.assertIsInstance(seam, cc.ConditioningCacheBackend)
        self.assertIsNone(seam.restore(epoch))  # nothing captured yet: the stage builds natively
        seam.on_cache_built(built)
        restored = cc.accept_restored_cache(seam.restore(epoch), epoch, layers=2, heads=h, head_dim=d,
                                            dtype=torch.float32, device="cpu")
        for i in range(2):
            self.assertTrue(torch.equal(restored.keys_pre_rope[i], built.keys_pre_rope[i]))
            self.assertTrue(torch.equal(restored.values[i], built.values[i]))
        self.assertIsNone(seam.restore(self._epoch(lcp.RowShard(2, 0, (1, 2)))))
        self.assertEqual([e["event"] for e in seam.events], ["captured", "installed"])
        with self.assertRaises(cc.ConditioningCacheError):
            cc.CacheRowSeam(heads=h)

    def test_single_gpu_reelcache_backend_refuses_row_shards(self):
        from reelcache_sglang_models.longcat_video import reelcache_backend as mb

        backend = object.__new__(mb.ReelCacheConditioningCacheBackend)
        backend.restore_representation = "svd_hevc"
        with self.assertRaisesRegex(cc.ConditioningCacheError, "row-sharded"):
            backend.restore(self._epoch(lcp.RowShard(2, 0, (1, 2))))
        with self.assertRaisesRegex(cc.ConditioningCacheError, "row-sharded"):
            mb._refuse_row_shard(self._epoch(lcp.RowShard(2, 1, (1, 2))))
        mb._refuse_row_shard(self._epoch())

    def test_set_context_parallel_is_type_checked(self):
        port = dit_core.LongCatVideoDiTCore(TINY)
        with self.assertRaises(lcp.ContextParallelError):
            port.set_context_parallel(object())
        self.assertIsNone(port.context_parallel())
        with self.assertRaises(lcp.ContextParallelError):  # no process group
            lcp.LongCatContextParallel(None)


# ---------------------------------------------------------------------------------------------------- Gloo
def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / max(float(b.double().norm()), 1e-30))


def _install_native_sdpa_cp(model):
    """Native SDPA attention (``test_longcat_video_port``) with the native Ulysses wrapper kept around it."""
    _, ulysses = _native_cp_module()
    attention = base.native()["attention"]
    base._install_native_sdpa(model)

    def self_attn(self, q, k, v, shape):
        return F.scaled_dot_product_attention(q, k, v, scale=self.scale)

    for module in model.modules():
        if isinstance(module, attention.Attention):
            module._process_attn = types.MethodType(ulysses.ulysses_wrapper(self_attn), module)


def _factor_sharded_restore(rank, world, root, cp, keys, values, out):
    """Owner-fit factor sharding of every layer's K/V rows; restore to the native spatial row owners."""
    from reelcache.distributed import DistributedOwner
    from reelcache.factor_sharding import FactorShardDomain, NativeOwners, OwnerFactorStore, fresh_fit
    from reelcache.store import ObjectStore
    from reelcache.types import MatrixKey, ModelIdentity, TensorContext

    channels = TINY.num_heads * TINY.head_dim
    data_group = dist.new_group(backend="gloo")
    store = OwnerFactorStore(ObjectStore(Path(root) / f"owner{rank}"), owner=rank, world_size=world)
    domain = FactorShardDomain(DistributedOwner(expected_world_size=world), data_group=data_group, store=store,
                               tile_tokens=16)
    kwargs = cp.row_shard().native_owners_kwargs(COND_GRID, kv_heads=TINY.num_heads, head_dim=TINY.head_dim)
    native = NativeOwners.spatial_rows(**kwargs)
    out["native_rows_equal_seam_rows"] = bool(torch.equal(native.rows(rank), cp.local_rows(COND_GRID)))
    model = ModelIdentity("tiny_longcat_cp", "step13-phase4-fixture", TINY.depth, channels, "global_KV_channels",
                          "float32")
    shards = {}
    for layer in range(TINY.depth):
        for target, mats in (("K", keys), ("V", values)):
            key = MatrixKey(layer, target)
            context = TensorContext(model, key, 0, *COND_GRID, "pre_rope" if target == "K" else "none")
            local = mats[layer][0].reshape(mats[layer].shape[1], channels).float()
            shards[key] = domain.fit_and_scatter(key, local, native, context,
                                                 fit=fresh_fit(min(channels, native.tokens)))
    node_map = domain.publish_node(domain.stage_node("longcat-seg0", shards, kind="svd"))
    restored = domain.restore_node(node_map, native, dtype=torch.float32, rope=None)
    store.store.close()
    return ([restored[MatrixKey(layer, "K")] for layer in range(TINY.depth)],
            [restored[MatrixKey(layer, "V")] for layer in range(TINY.depth)])


def _worker(rank: int, world: int, root: str, factor: bool) -> None:
    import traceback

    out: dict = {"rank": rank, "world": world}
    try:
        os.environ["TMPDIR"] = root
        torch.set_num_threads(1)
        dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(seconds=180),
                                init_method=f"file://{Path(root) / 'rendezvous'}")
        cpu, ulysses = _native_cp_module()
        split = lcp.optimal_split(world)
        nat1, port1 = base._tiny_native_and_port()
        state = nat1.state_dict()
        natcp = base.native()["dit"].LongCatVideoTransformer3DModel(
            in_channels=16, out_channels=16, hidden_size=TINY.hidden_size, depth=TINY.depth, num_heads=TINY.num_heads,
            caption_channels=TINY.caption_channels, mlp_ratio=TINY.mlp_ratio, adaln_tembed_dim=TINY.adaln_tembed_dim,
            frequency_embedding_size=TINY.frequency_embedding_size, patch_size=(1, 2, 2), cp_split_hw=list(split),
            text_tokens_zero_pad=True)
        natcp.load_state_dict(state, strict=True)
        natcp.eval()
        _install_native_sdpa_cp(natcp)
        portcp = dit_core.LongCatVideoDiTCore(TINY).eval()
        portcp.load_state_dict(state, strict=True)
        cp = lcp.LongCatContextParallel(None)
        portcp.set_context_parallel(cp)
        out["cp"] = cp.describe()
        torch.manual_seed(7)
        latents = torch.randn(1, 16, 6, 8, 12)
        text = torch.randn(1, 1, 9, TINY.caption_channels)
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0]])
        cond, new = latents[:, :, :2], latents[:, :, 2:]
        empty = torch.zeros(1, 1, 9, TINY.caption_channels)
        t_plain = torch.tensor([640.0])
        t_cond = torch.tensor([[0.0, 0.0, 640.0, 640.0, 640.0, 640.0]])
        t_new = torch.tensor([931.0]).expand(1).unsqueeze(-1).repeat(1, 4)
        rows = cp.local_rows(COND_GRID)
        with torch.no_grad(), _NativeCPState(world, rank, dist.group.WORLD):
            # 1. Ulysses exchange == native all_to_all (port layout [B, S, H, D], native [B, H, S, D])
            gen = torch.Generator().manual_seed(100 + rank)
            x = torch.randn(1, 12, TINY.num_heads, TINY.head_dim, generator=gen)
            ours_in = cp.ulysses_in(x)
            ref_in = ulysses.all_to_all(x.permute(0, 2, 1, 3), scatter_idx=1, gather_idx=2,
                                        group=cpu.get_cp_group()).permute(0, 2, 1, 3)
            ours_out = cp.ulysses_out(ours_in)
            ref_out = ulysses.all_to_all(ref_in.permute(0, 2, 1, 3), scatter_idx=2, gather_idx=1,
                                         group=cpu.get_cp_group()).permute(0, 2, 1, 3)
            out["a2a_in_bitwise_native"] = bool(torch.equal(ours_in, ref_in))
            out["a2a_out_bitwise_native"] = bool(torch.equal(ours_out, ref_out))
            out["a2a_roundtrip_identity"] = bool(torch.equal(ours_out, x))
            out["a2a_in_shape"] = list(ours_in.shape)
            # 2. forwards: port CP vs native CP (bitwise) and vs one process (port, no CP)
            pairs = {
                "plain": (portcp(latents, t_plain, text, mask), natcp(latents, t_plain, text, mask),
                          port1(latents, t_plain, text, mask)),
                "conditioned": (portcp(latents, t_cond, text, mask, num_cond_latents=2),
                                natcp(latents, t_cond, text, mask, num_cond_latents=2),
                                port1(latents, t_cond, text, mask, num_cond_latents=2)),
            }
            keys, values = portcp.build_conditioning_kv(cond)
            _, kv_nat = natcp(cond, torch.zeros(1, 2), empty, return_kv=True, skip_crs_attn=True)
            keys1, values1 = port1.build_conditioning_kv(cond)
            out["cache_local_shape"] = list(keys[0].shape)
            out["cache_bitwise_native_cp"] = all(
                torch.equal(keys[i].permute(0, 2, 1, 3), kv_nat[i][0]) and torch.equal(values[i].permute(0, 2, 1, 3),
                                                                                       kv_nat[i][1])
                for i in range(TINY.depth))
            out["cache_rows_vs_single"] = [
                {"K_bitwise": bool(torch.equal(keys[i], keys1[i][:, rows])),
                 "V_bitwise": bool(torch.equal(values[i], values1[i][:, rows])),
                 "K_rel": _rel(keys[i], keys1[i][:, rows]), "V_rel": _rel(values[i], values1[i][:, rows])}
                for i in range(TINY.depth)]
            pairs["cache_read"] = (
                portcp.forward_with_conditioning(new, t_new, text, mask, list(zip(keys, values)), 2),
                natcp(new, t_new, text, mask, num_cond_latents=2, kv_cache_dict=kv_nat),
                port1.forward_with_conditioning(new, t_new, text, mask, list(zip(keys1, values1)), 2))
            for name, (ours, nat, single) in pairs.items():
                out[f"{name}_bitwise_native_cp"] = bool(torch.equal(ours, nat))
                out[f"{name}_rel_vs_single"] = _rel(ours, single)
                out[f"{name}_shape_full"] = list(ours.shape) == list(single.shape)
            reference = pairs["cache_read"][0]
            # 3. per-rank seam: install this rank's rows ([T_local, C], the factor-sharded restore layout)
            epoch = cc.ConditioningCacheEpoch(0, COND_GRID, "c" * 64, "m" * 64, row_shard=cp.row_shard())
            channels = TINY.num_heads * TINY.head_dim
            installed = cc.LongCatConditioningCache.from_matrices(
                epoch, [k[0].reshape(-1, channels) for k in keys], [v[0].reshape(-1, channels) for v in values],
                heads=TINY.num_heads)
            installed = cc.accept_restored_cache(installed, epoch, layers=TINY.depth, heads=TINY.num_heads,
                                                 head_dim=TINY.head_dim, dtype=torch.float32, device="cpu")
            again = portcp.forward_with_conditioning(new, t_new, text, mask,
                                                     list(zip(installed.keys_pre_rope, installed.values)), 2)
            out["seam_install_rows_bitwise"] = bool(torch.equal(again, reference))
            native_layout = cc.LongCatConditioningCache.from_native_layout(
                epoch, [kv_nat[i][0] for i in range(TINY.depth)], [kv_nat[i][1] for i in range(TINY.depth)])
            again = portcp.forward_with_conditioning(new, t_new, text, mask,
                                                     list(zip(native_layout.keys_pre_rope, native_layout.values)), 2)
            out["seam_install_native_layout_bitwise"] = bool(torch.equal(again, reference))
            out["seam_rows_equal_epoch_rows"] = bool(torch.equal(epoch.row_index(), rows))
            from reelcache.factor_sharding import row_shard_gram

            # capture side: the fit owner's Gram is the sum of the per-rank row-shard Grams (layer 0: bitwise rows)
            gram = row_shard_gram(installed.layer_matrices(0)[1], layout="rows")
            dist.all_reduce(gram)
            full_v = values1[0][0].reshape(-1, channels).double()
            out["row_gram_sum_rel_vs_single_full"] = _rel(gram, full_v.T @ full_v)
        if factor:
            keys_r, values_r = _factor_sharded_restore(rank, world, root, cp, keys, values, out)
            with torch.no_grad():
                restored = cc.LongCatConditioningCache.from_matrices(epoch, keys_r, values_r, heads=TINY.num_heads)
                restored = cc.accept_restored_cache(restored, epoch, layers=TINY.depth, heads=TINY.num_heads,
                                                    head_dim=TINY.head_dim, dtype=torch.float32, device="cpu")
                out["factor_rows_rel"] = max(max(_rel(restored.keys_pre_rope[i], keys[i]),
                                                 _rel(restored.values[i], values[i])) for i in range(TINY.depth))
                again = portcp.forward_with_conditioning(new, t_new, text, mask,
                                                         list(zip(restored.keys_pre_rope, restored.values)), 2)
                out["factor_forward_rel"] = _rel(again, reference)
        dist.barrier()
        dist.destroy_process_group()
    except BaseException as exc:  # noqa: BLE001 - surfaced by the parent test
        out["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        (Path(root) / f"rank{rank}.json").write_text(json.dumps(out, default=str))
        os._exit(0)


class GlooContextParallelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base.native_root()  # the workers load the native LongCat-Video source (a test asset): skip here without it

    def setUp(self):
        self.root = ROOT / ".test-artifacts" / f"longcat-cp-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _run(self, world: int, factor: bool) -> list[dict]:
        context = multiprocessing.get_context("spawn")
        procs = [context.Process(target=_worker, args=(rank, world, str(self.root), factor)) for rank in range(world)]
        for proc in procs:
            proc.start()
        try:
            for proc in procs:
                proc.join(timeout=900)
                self.assertFalse(proc.is_alive(), "Gloo worker timed out")
            rows = [json.loads((self.root / f"rank{rank}.json").read_text()) for rank in range(world)]
        finally:
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=10)
        for row in rows:
            self.assertNotIn("error", row, row.get("error"))
        return rows

    def _check(self, rows: list[dict], world: int, local_tokens: int, factor: bool) -> None:
        for row in rows:
            rank = row["rank"]
            for name in ("a2a_in_bitwise_native", "a2a_out_bitwise_native", "a2a_roundtrip_identity",
                         "plain_bitwise_native_cp", "conditioned_bitwise_native_cp", "cache_read_bitwise_native_cp",
                         "cache_bitwise_native_cp", "seam_install_rows_bitwise", "seam_install_native_layout_bitwise",
                         "seam_rows_equal_epoch_rows", "plain_shape_full", "cache_read_shape_full"):
                self.assertTrue(row[name], f"rank {rank}: {name}")
            self.assertEqual(row["cp"]["cp_size"], world)
            self.assertEqual(row["cp"]["split_hw"], list(lcp.optimal_split(world)))
            self.assertEqual(row["cache_local_shape"], [1, local_tokens, TINY.num_heads, TINY.head_dim])
            self.assertEqual(row["a2a_in_shape"], [1, 12 * world, TINY.num_heads // world, TINY.head_dim])
            first = row["cache_rows_vs_single"][0]
            self.assertTrue(first["K_bitwise"] and first["V_bitwise"], f"rank {rank}: layer-0 rows")
            for layer in row["cache_rows_vs_single"]:
                self.assertLess(max(layer["K_rel"], layer["V_rel"]), 1e-5)
            for name in ("plain", "conditioned", "cache_read"):
                self.assertLess(row[f"{name}_rel_vs_single"], 1e-5, f"rank {rank}: {name}")
            self.assertLess(row["row_gram_sum_rel_vs_single_full"], 1e-12)
            if factor:
                self.assertTrue(row["native_rows_equal_seam_rows"])
                self.assertLess(row["factor_rows_rel"], 2e-3)  # FP16 basis rows, full-rank FP32 coefficients
                self.assertLess(row["factor_forward_rel"], 2e-3)

    def test_two_ranks_split_1x2_match_native_cp_and_restore_through_the_seam(self):
        self._check(self._run(2, factor=True), 2, 2 * 4 * 3, factor=True)

    def test_four_ranks_split_2x2_match_native_cp(self):
        self._check(self._run(4, factor=False), 4, 2 * 2 * 3, factor=False)


if __name__ == "__main__":
    unittest.main()
