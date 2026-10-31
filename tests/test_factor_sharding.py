"""Step 16: non-replicated factor ownership (FactorShardSpec, fragment packs, owner stores) and
2-/4-rank Gloo collectives (owner-fit reference, two-phase publication, tiled reduce-scatter restore,
growth/fork lineage, failure/cancellation, aggregate unique bytes). CPU only; artifacts stay in the checkout."""

import dataclasses
from datetime import timedelta
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import unittest
import uuid

import torch
import torch.distributed as dist

from reelcache.distributed import KVLayout
from reelcache.errors import ConfigurationError, IntegrityError
from reelcache.factor_sharding import (
    DEFAULT_MODE, PLAN_BLOCK_ROWS, FactorShardSpec, FragmentInput, NativeOwners, ShardedNodeMap,
    decode_fragment_pack, encode_fragment_pack, entry_label, parse_fragment_pack, reference_reconstruction,
    row_shard_gram, split_factors,
)
from reelcache.types import CodecSpec, MatrixKey, ModelIdentity, TensorContext

TOKENS, CHANNELS, HEADS, HEAD_DIM = 96, 64, 8, 8
FRAMES, HEIGHT, WIDTH = 2, 6, 8
RANKS = {(0, "K"): 6, (0, "V"): 40, (1, "K"): 64, (1, "V"): 33, (2, "K"): 20, (2, "V"): 50, (3, "K"): 32,
         (3, "V"): 1}


def model():
    return ModelIdentity("fake_factor_shard", "step16-fixture", 4, CHANNELS, "global_KV_channels", "float32")


def context(layer, target, node=0):
    return TensorContext(model(), MatrixKey(layer, target), node * TOKENS, FRAMES, HEIGHT, WIDTH,
                         "pre_rope" if target == "K" else "none")


def matrix(layer, target, node=0, rank=None):
    """Deterministic low-rank-plus-noise canonical [tokens, channels] FP32 matrix."""
    generator = torch.Generator().manual_seed(1000 * node + 10 * layer + (target == "V"))
    rank = rank or RANKS[(layer, target)]
    true = min(CHANNELS, rank + 4)
    scales = torch.logspace(0, -2, true)
    value = (torch.randn(TOKENS, true, generator=generator) * scales) @ torch.randn(true, CHANNELS, generator=generator)
    return (value + 1e-3 * torch.randn(TOKENS, CHANNELS, generator=generator)).float()


def rope(entry, values, tokens, columns):
    """Fake pre-RoPE rotation of channel pairs (FP32, per token position and absolute channel)."""
    start = columns.start or 0
    channel = torch.arange(start, start + values.shape[1], device=values.device)
    pair = (channel // 2).float()
    angle = tokens.float()[:, None] * 0.013 * (1.0 + pair[None, :] / 7.0)
    even = values[:, 0::2]
    odd = values[:, 1::2]
    cos, sin = angle[:, 0::2].cos(), angle[:, 0::2].sin()
    out = torch.empty_like(values)
    out[:, 0::2] = even * cos - odd * sin
    out[:, 1::2] = even * sin + odd * cos
    return out


class SpecTests(unittest.TestCase):
    def test_blocks_partition_rotation_and_uneven_owners(self):
        self.assertEqual((DEFAULT_MODE, PLAN_BLOCK_ROWS), ("factors", 32))
        for block_rows in (None, PLAN_BLOCK_ROWS):  # default whole-matrix owners; the plan's 32-row grid
            for world in (1, 2, 4):
                for rank in (1, 6, 31, 32, 33, 40, 64):
                    for layer in range(4):
                        for target in ("K", "V"):
                            spec = FactorShardSpec.fresh(MatrixKey(layer, target), channels=CHANNELS, rank=rank,
                                                         world_size=world, block_rows=block_rows)
                            spec.check_partition()
                            self.assertEqual(sum(spec.view_loads), rank)
                            self.assertEqual(spec.whole_matrix, block_rows is None)
                            self.assertEqual(spec.blocks, 1 if block_rows is None else math.ceil(rank / block_rows))
                            self.assertEqual(spec.block_owners[0], (layer + (target == "V")) % world)
                            if block_rows is None:
                                self.assertEqual(sorted(spec.view_loads)[-1], rank)  # one owner per matrix
            small = FactorShardSpec.fresh(MatrixKey(0, "K"), channels=CHANNELS, rank=6, world_size=2,
                                          block_rows=block_rows)
            self.assertEqual(small.view_loads, (6, 0))  # an explicit empty owner, never padding
            self.assertEqual(small.view_ranges(1), ())
            owners = {FactorShardSpec.fresh(MatrixKey(layer, target), channels=CHANNELS, rank=40, world_size=4,
                                            block_rows=block_rows).block_owners[0]
                      for layer in range(4) for target in ("K", "V")}
            self.assertEqual(owners, {0, 1, 2, 3})  # first blocks spread over every owner
        split = FactorShardSpec.fresh(MatrixKey(1, "V"), channels=CHANNELS, rank=40, world_size=2,
                                      block_rows=PLAN_BLOCK_ROWS)
        self.assertEqual(split.view_loads, (32, 8))  # the plan grid splits one matrix over owners
        whole = FactorShardSpec.fresh(MatrixKey(1, "V"), channels=CHANNELS, rank=40, world_size=2)
        self.assertEqual((whole.view_loads, whole.block_rows), ((40, 0), CHANNELS))

    def test_extension_keeps_owners_and_cuts_immutable_segments(self):
        whole = FactorShardSpec.fresh(MatrixKey(1, "V"), channels=CHANNELS, rank=40, world_size=2)
        whole_grown = whole.extend(20).extend(4, rank_view=50)
        whole_grown.check_stable(whole)
        self.assertEqual(whole_grown.block_owners, whole.block_owners)  # the matrix never changes owner
        self.assertEqual(whole_grown.segments(), ((0, 40, 0), (40, 60, 0), (60, 64, 0)))
        root = FactorShardSpec.fresh(MatrixKey(1, "V"), channels=CHANNELS, rank=40, world_size=2,
                                     block_rows=PLAN_BLOCK_ROWS)
        grown = root.extend(20)
        again = grown.extend(4, rank_view=50)
        for child, parent in ((grown, root), (again, grown), (again, root)):
            child.check_stable(parent)
        self.assertEqual(again.version_rows, (40, 60, 64))
        self.assertEqual([segment[:2] for segment in again.segments()], [(0, 32), (32, 40), (40, 60), (60, 64)])
        with self.assertRaises(ConfigurationError):
            again.extend(1)  # never beyond the channel count
        for start, stop, owner in again.segments():
            self.assertTrue(all(again.owner_of(row) == owner for row in range(start, stop)))
        self.assertEqual(again.view_rows(0) + again.view_rows(1), 50)
        trimmed = again.with_view(33)
        trimmed.check_partition()
        self.assertEqual(trimmed.block_owners, again.block_owners)
        self.assertEqual(FactorShardSpec.from_dict(again.to_dict()), again)
        moved = FactorShardSpec(key=again.key, channels=CHANNELS, world_size=2, version_rows=again.version_rows,
                                block_owners=(again.block_owners[1], again.block_owners[0]),
                                rank_view=64, rotation=again.rotation)
        with self.assertRaises(IntegrityError):
            moved.check_stable(root)
        for field, value in (("block_owners", [0, 1, 0]), ("version_rows", [40, 40]), ("rank_view", 65),
                             ("block_owners", [0, 2])):
            data = again.to_dict()
            data[field] = value
            with self.assertRaises(ConfigurationError):
                FactorShardSpec.from_dict(data)

    def test_runtime_basis_versions_give_stable_fork_owners(self):
        from reelcache.basis_versions import BasisVersionStore
        from reelcache.factor_sharding import spec_for_basis_version
        from reelcache.store import ObjectStore

        root = Path(__file__).resolve().parents[1] / ".test-artifacts" / f"factor-versions-{uuid.uuid4().hex}"
        try:
            with ObjectStore(root) as store:
                versions = BasisVersionStore(store, model())
                key = MatrixKey(2, "V")

                def rows(count, seed):
                    return torch.randn(count, CHANNELS, generator=torch.Generator().manual_seed(seed)).half()

                parent_ref, parent = versions.put_root(key, rows(40, 1))
                child_ref, _ = versions.put_extension(parent_ref, rows(20, 2))
                fork_a, _ = versions.put_extension(child_ref, rows(3, 3))
                fork_b, _ = versions.put_extension(child_ref, rows(4, 4))
                for block_rows in (None, PLAN_BLOCK_ROWS):
                    specs = {name: spec_for_basis_version(key, versions, ref, world_size=4, block_rows=block_rows)
                             for name, ref in (("root", parent_ref), ("child", child_ref), ("a", fork_a),
                                               ("b", fork_b))}
                    self.assertEqual(specs["a"].version_rows, (40, 60, 63))
                    self.assertEqual(specs["b"].version_rows, (40, 60, 64))
                    self.assertEqual(specs["a"].whole_matrix, block_rows is None)
                    for name in ("child", "a", "b"):
                        specs[name].check_stable(specs["root"])
                    specs["a"].check_stable(specs["child"])
                    specs["b"].check_stable(specs["child"])
                    self.assertEqual(specs["a"].basis_group_id, parent.group_id)
                    self.assertEqual(specs["a"].basis_version_ref, fork_a)
                    shared = [segment for segment in specs["a"].segments() if segment[1] <= 60]
                    self.assertEqual(shared, [segment for segment in specs["b"].segments() if segment[1] <= 60])
                    self.assertEqual(specs["a"].extend(1).block_owners[:2], specs["b"].block_owners[:2])
                with self.assertRaises(ConfigurationError):
                    spec_for_basis_version(MatrixKey(1, "V"), versions, fork_a, world_size=4)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_native_owners_and_row_only_gram_sums(self):
        layout = KVLayout(TOKENS, HEADS, HEAD_DIM, 2)
        heads = NativeOwners(layout, "heads")
        self.assertEqual(heads.shape(1), (TOKENS, 32))
        self.assertEqual(heads.columns(1), slice(32, 64))
        rows = NativeOwners(layout, "rows")
        self.assertEqual([int(rows.rows(r).numel()) for r in range(2)], [48, 48])
        spatial = NativeOwners.spatial_rows(frames=2, height=4, width=6, split_h=1, split_w=2, kv_heads=HEADS,
                                            head_dim=HEAD_DIM)
        self.assertEqual(spatial.rows(0)[:4].tolist(), [0, 1, 2, 6])
        with self.assertRaises(ConfigurationError):
            NativeOwners(layout, "rows", [range(0, 40), range(41, 96)])
        with self.assertRaises(ConfigurationError):
            NativeOwners(layout, "heads", [range(48), range(48, 96)])
        value = matrix(0, "V")
        full = value.double().T @ value.double()
        summed = sum(row_shard_gram(value[rows.rows(r)], layout="rows") for r in range(2))
        self.assertTrue(torch.allclose(summed, full, rtol=1e-12, atol=1e-9))
        columns = [value[:, heads.columns(r)] for r in range(2)]
        with self.assertRaises(ConfigurationError):
            row_shard_gram(columns[0], layout="heads")
        block = torch.block_diag(*[part.double().T @ part.double() for part in columns])
        self.assertGreater(float((block - full).abs().max()), 1.0)  # dropped cross-channel terms

    def test_split_reconstruction_equals_single_gemm_within_fp32_rounding(self):
        from reelcache.factorization import fit_basis

        for world, block_rows in ((2, None), (2, PLAN_BLOCK_ROWS), (4, None), (4, PLAN_BLOCK_ROWS)):
            spec = FactorShardSpec.fresh(MatrixKey(1, "V"), channels=CHANNELS, rank=33, world_size=world,
                                         block_rows=block_rows)
            factors = fit_basis(matrix(1, "V"), context(1, "V"), 33)
            parts = split_factors(spec, factors.coefficients.float(), factors.basis)
            summed = sum(p @ q.float() for p, q in parts.values() if p.shape[1])
            reference = reference_reconstruction(factors.coefficients, factors.basis)
            self.assertLess(float((summed - reference).abs().max()), 1e-4 * float(reference.abs().max()))
            self.assertEqual(sum(p.shape[1] for p, _ in parts.values()), 33)


class FragmentPackTests(unittest.TestCase):
    def setUp(self):
        from reelcache.factorization import fit_basis

        self.world = 2
        self.items = {}
        for layer, target in ((1, "V"), (2, "K"), (0, "K")):
            key = MatrixKey(layer, target)
            rank = RANKS[(layer, target)]
            # the plan grid splits 1:V over both owners, the harder case for independent fragment encodes
            spec = FactorShardSpec.fresh(key, channels=CHANNELS, rank=rank, world_size=self.world,
                                         block_rows=PLAN_BLOCK_ROWS)
            factors = fit_basis(matrix(layer, target), context(layer, target), rank)
            self.items[key] = (spec, factors.coefficients.float(), factors.basis)

    def fragments(self, owner):
        result, empty = [], []
        for key, (spec, coefficients, basis) in self.items.items():
            part = split_factors(spec, coefficients, basis)[owner]
            if part[0].shape[1]:
                result.append(FragmentInput(key, spec, context(key.layer, key.target), *part))
            else:
                empty.append(key)
        return result, empty

    def test_svd_pack_is_exact_and_integrity_checked(self):
        fragments, empty = self.fragments(1)
        pack = encode_fragment_pack("n0", 1, self.world, fragments, kind="svd", empty_entries=empty)
        decoded = decode_fragment_pack(pack)
        for item in fragments:
            self.assertTrue(torch.equal(decoded[item.entry], item.coefficients))
        self.assertEqual([entry["key"] for entry in pack.metadata["empty_entries"]],
                         [{"layer": 0, "target": "K"}, {"layer": 2, "target": "K"}])
        name = next(name for name in pack.payloads if name.startswith("coefficients"))
        tampered = dict(pack.payloads)
        tampered[name] = bytes([tampered[name][0] ^ 1]) + tampered[name][1:]
        from reelcache.types import EncodedTensor

        with self.assertRaises(IntegrityError):
            decode_fragment_pack(EncodedTensor(pack.metadata, tampered))
        broken = json.loads(json.dumps(pack.metadata))
        broken["tiles"][0]["ranges"] = [[30, 32]]  # rows no longer match the coefficient payload
        with self.assertRaises(IntegrityError):
            parse_fragment_pack(EncodedTensor(broken, pack.payloads))

    def test_hevc_fragments_are_independent_streams_with_random_access(self):
        codec = CodecSpec("svd_hevc", rank=64)
        packs = {}
        for owner in range(self.world):
            fragments, empty = self.fragments(owner)
            packs[owner] = (encode_fragment_pack("n0", owner, self.world, fragments, kind="svd_hevc", codec=codec,
                                                 validate=True, empty_entries=empty), fragments)
        full_fragments = []
        for key, (spec, coefficients, basis) in self.items.items():
            whole = FactorShardSpec.fresh(key, channels=CHANNELS, rank=spec.rank_view, world_size=1)
            full_fragments.append(FragmentInput(key, whole, context(key.layer, key.target), coefficients,
                                                basis[:spec.rank_view]))
        full = encode_fragment_pack("n0", 0, 1, full_fragments, kind="svd_hevc", codec=codec)
        full_stream = full.payloads["stream.hevc"]
        from reelcache import _hevc

        for owner, (pack, fragments) in packs.items():
            stream = pack.payloads["stream.hevc"]
            units = _hevc.all_intra_access_units(stream, pack.metadata["canvas"]["pictures"])
            self.assertTrue(all(unit["self_contained"] for unit in units))
            self.assertNotIn(stream, full_stream)  # its own encode, never a byte slice of the full stream
            decoded = decode_fragment_pack(pack)
            for item in fragments:
                error = float((decoded[item.entry] - item.coefficients).abs().max())
                spread = float(item.coefficients.max() - item.coefficients.min())
                self.assertLess(error, 0.05 * spread)
                single = decode_fragment_pack(pack, entries=[item.entry])
                self.assertTrue(torch.equal(single[item.entry], decoded[item.entry]))
        self.assertEqual([len(p.metadata["tiles"]) for p, _ in packs.values()], [3, 1])  # small matrices: one owner
        with self.assertRaises(ConfigurationError):
            encode_fragment_pack("n0", 0, self.world, packs[0][1], kind="svd_hevc",
                                 codec=CodecSpec("svd_hevc", rank=64, inter_prediction=True))
        with self.assertRaises(ConfigurationError):
            decode_fragment_pack(packs[0][0], device="cuda")  # no software fallback on CUDA


# ------------------------------------------------------------------------------------ Gloo workers

def _gather(value, world):
    rows = [None] * world
    dist.all_gather_object(rows, value)
    return rows


def _local(value, native, rank):
    if native.kind == "heads":
        return value[:, native.columns(rank)].contiguous()
    return value.index_select(0, native.rows(rank)).contiguous()


def _uneven_rows(world):
    owners = [min(row % (world + 1), world - 1) for row in range(TOKENS)]
    return [[row for row in range(TOKENS) if owners[row] == rank] for rank in range(world)]


def _reference(domain, node_map, world, rank, *, rotate):
    """Single-process FP32 reconstruction (one GEMM per matrix) from every owner's decoded fragments."""
    from reelcache.factor_sharding import ranges_index

    loaded = domain.load_node(node_map)
    mine = {entry_label(entry): (None if shard.coefficients is None else shard.coefficients,
                                 None if shard.coefficients is None else shard.view_basis())
            for entry, shard in loaded.items()}
    everyone = _gather(mine, world)
    result = {}
    for entry in node_map.entries:
        spec = node_map.spec(entry)
        coefficients = torch.zeros(TOKENS, spec.rank_view)
        basis = torch.zeros(spec.rank_view, CHANNELS, dtype=torch.float16)
        for owner in range(world):
            part = everyone[owner][entry_label(entry)]
            index = ranges_index(spec.view_ranges(owner))
            if index.numel():
                coefficients[:, index] = part[0]
                basis[index] = part[1]
        value = reference_reconstruction(coefficients, basis)
        if rotate and entry.target == "K":
            value = rope(entry, value, torch.arange(TOKENS), slice(0, CHANNELS))
        result[entry] = value
    return result


def _compare(outputs, reference, native, rank):
    worst = 0.0
    for entry, value in outputs.items():
        expected = _local(reference[entry], native, rank)
        scale = float(expected.abs().max()) or 1.0
        worst = max(worst, float((value.float() - expected).abs().max()) / scale)
    return worst


def _lifecycle(rank, world, root, domain, natives, output):
    from reelcache.factor_sharding import augment_fit, fresh_fit, reuse_fit, split_factors
    from reelcache.factorization import fit_basis

    heads = natives["heads"]
    keys = [MatrixKey(layer, target) for layer, target in sorted(RANKS)]
    codec = CodecSpec("svd_hevc", rank=CHANNELS)
    shards0, bitwise = {}, True
    for key in keys:
        value = matrix(key.layer, key.target)
        shard = domain.fit_and_scatter(key, _local(value, heads, rank), heads, context(key.layer, key.target),
                                       fit=fresh_fit(RANKS[(key.layer, key.target)]))
        shards0[key] = shard
        factors = fit_basis(value, context(key.layer, key.target), RANKS[(key.layer, key.target)])
        expected = split_factors(shard.spec, factors.coefficients.float(), factors.basis)[rank]
        if shard.coefficients is None:
            bitwise &= expected[0].shape[1] == 0
        else:
            bitwise &= torch.equal(shard.coefficients, expected[0]) and torch.equal(shard.view_basis(), expected[1])
    output["scatter_bitwise_vs_single_process_fit"] = bool(bitwise)
    output["policy"] = {"mode": domain.mode, "block_rows": domain.block_rows}
    output["blocks"] = {entry_label(key): shards0[key].spec.blocks for key in keys}
    output["whole_matrix"] = all(shards0[key].spec.whole_matrix for key in keys)
    output["owner_fit_reports"] = len(domain.counters["owner_fit"])
    output["fit_transient_bytes_max"] = max(item["transient_canonical_bytes"] for item in domain.counters["owner_fit"])
    output["view_loads"] = {entry_label(key): list(shards0[key].spec.view_loads) for key in keys}
    staged = domain.stage_node("n0", shards0, kind="svd_hevc", codec=codec, pack_groups=2, validate=True)
    output["n0_stage_error"] = staged["error"]
    map0 = domain.publish_node(staged)
    output["n0_published_everywhere"] = all(_gather(domain.store.published("n0") is not None, world))
    output["n0_record_chunks"] = math.ceil(len(json.dumps(staged, sort_keys=True, separators=(",", ":"))) / 2048)
    output["n0_digest_agreed"] = len(set(_gather(map0.digest, world))) == 1
    output["n0_map_roundtrip"] = ShardedNodeMap.from_dict(map0.to_dict()).digest == map0.digest
    errors, wire = {}, {}
    from reelcache import factor_sharding

    for name, native in natives.items():
        rotate = name == "heads"
        reference = _reference(domain, map0, world, rank, rotate=rotate)
        for mode in ("partials", "factors", "factors_chunked"):
            before = dict(domain.counters)
            bound = factor_sharding.FACTOR_EXCHANGE_BYTES
            if mode == "factors_chunked":  # several exchange rounds per matrix (tile-sized coefficient chunks)
                factor_sharding.FACTOR_EXCHANGE_BYTES = 1
            try:
                outputs = domain.restore_node(map0, native, dtype=torch.float32, rope=rope if rotate else None,
                                              mode=mode.split("_")[0])
            finally:
                factor_sharding.FACTOR_EXCHANGE_BYTES = bound
            errors[f"{name}/{mode}"] = _compare(outputs, reference, native, rank)
            wire[f"{name}/{mode}"] = {counter: domain.counters[counter] - before[counter] for counter in (
                "reduce_scatter_bytes_sent", "factor_exchange_bytes_sent", "reduce_scatter_calls",
                "factor_exchange_rounds")}
        output[f"n0_{name}_shapes"] = sorted({tuple(value.shape) for value in outputs.values()})
    output["n0_max_rel_error_vs_single_gemm"] = errors
    output["n0_wire_by_mode"] = wire
    output["n0_heads_reduce_scatter_calls"] = wire["heads/partials"]["reduce_scatter_calls"]
    bf16 = domain.restore_node(map0, heads, dtype=torch.bfloat16)
    reference = _reference(domain, map0, world, rank, rotate=False)
    mismatched = total = 0
    for entry, value in bf16.items():
        expected = _local(reference[entry], heads, rank).to(torch.bfloat16)
        mismatched += int((value.view(torch.int16) != expected.view(torch.int16)).sum())
        total += value.numel()
    output["n0_bf16_mismatch_fraction"] = mismatched / total
    output["n0_zero_partial_calls"] = domain.counters["zero_partial_calls"]
    output["n0_padding_bytes"] = domain.counters["reduce_scatter_padding_bytes"]
    # node 1: growth (augment / reuse / refit) with stable owners, exact FP32 coefficients
    fits = {MatrixKey(1, "V"): augment_fit(10), MatrixKey(0, "V"): augment_fit(20), MatrixKey(2, "K"): reuse_fit(13),
            MatrixKey(3, "K"): fresh_fit(16)}
    shards1 = {}
    for key in keys:
        value = matrix(key.layer, key.target, node=1)
        shards1[key] = domain.fit_and_scatter(key, _local(value, heads, rank), heads,
                                              context(key.layer, key.target, 1),
                                              fit=fits.get(key, reuse_fit()), parent=shards0[key])
    output["n1_version_rows"] = {entry_label(key): list(shards1[key].spec.version_rows) for key in keys}
    stable = True
    for key in keys:
        if key != MatrixKey(3, "K"):
            parent, child = shards0[key].spec, shards1[key].spec
            stable &= all(parent.owner_of(row) == child.owner_of(row) for row in range(parent.total_rank))
    output["n1_old_rows_kept_owners"] = bool(stable)
    staged1 = domain.stage_node("n1", shards1, kind="svd")
    map1 = domain.publish_node(staged1, parent_node_id="n0")
    shared = True
    for key in keys:
        if key == MatrixKey(3, "K"):
            continue
        old = {(a, b): ref for a, b, ref in map0.segment_refs(rank, key)}
        new = {(a, b): ref for a, b, ref in map1.segment_refs(rank, key)}
        shared &= all(new.get(span) == ref for span, ref in old.items())
    output["n1_parent_segments_shared"] = bool(shared)
    outputs = domain.restore_node(map1, natives["rows"], dtype=torch.float32)
    reference = _reference(domain, map1, world, rank, rotate=False)
    output["n1_rows_max_rel_error"] = _compare(outputs, reference, natives["rows"], rank)
    exact = True
    for key in keys:  # svd fragments are exact: decoded factors equal the scattered ones bitwise
        loaded = domain.load_node(map1, entries=[key])[key]
        if shards1[key].coefficients is not None:
            exact &= torch.equal(loaded.coefficients, shards1[key].coefficients)
        exact &= (loaded.basis is None and shards1[key].basis is None) or torch.equal(loaded.basis, shards1[key].basis)
    output["n1_store_roundtrip_bitwise"] = bool(exact)
    # forks: two branches extend n0's versions independently
    fork_keys = [MatrixKey(1, "V"), MatrixKey(2, "V")]
    maps = {}
    for branch, node in (("n2a", 2), ("n2b", 3)):
        shards = {key: domain.fit_and_scatter(key, _local(matrix(key.layer, key.target, node=node), heads, rank),
                                              heads, context(key.layer, key.target, node), fit=augment_fit(5),
                                              parent=shards0[key]) for key in fork_keys}
        staged_branch = domain.stage_node(branch, shards, kind="svd_hevc", codec=codec)
        maps[branch] = domain.publish_node(staged_branch, parent_node_id="n0")
        outputs = domain.restore_node(maps[branch], heads, dtype=torch.float32)
        output[f"{branch}_max_rel_error"] = _compare(outputs, _reference(domain, maps[branch], world, rank,
                                                                        rotate=False), heads, rank)
    fork_shared, fork_distinct = True, True
    for key in fork_keys:
        parent_rows = shards0[key].spec.total_rank
        a = {(s, e): ref for s, e, ref in maps["n2a"].segment_refs(rank, key)}
        b = {(s, e): ref for s, e, ref in maps["n2b"].segment_refs(rank, key)}
        base = {(s, e): ref for s, e, ref in map0.segment_refs(rank, key)}
        fork_shared &= all(a.get(span) == ref and b.get(span) == ref for span, ref in base.items())
        fork_distinct &= all(a[span] != b[span] for span in a if span[0] >= parent_rows and span in b)
        fork_shared &= maps["n2a"].spec(key).block_owners[:shards0[key].spec.blocks] == shards0[key].spec.block_owners
    output["fork_parent_segments_shared"] = bool(fork_shared)
    output["fork_extensions_distinct"] = bool(fork_distinct)
    # aggregate unique bytes: every segment stored once on exactly one owner
    storage = domain.aggregate_storage()
    unique = {}
    for node_map in (map0, map1, maps["n2a"], maps["n2b"]):
        for record in node_map.owners:
            for item in record["segments"]:
                unique[item["ref"]] = item["bytes"]
    output["storage_totals"] = storage["totals"]
    output["unique_segment_bytes"] = sum(unique.values())
    output["replicated_basis_bytes"] = world * sum(unique.values())  # every unique row on every rank
    output["map_storage_n0"] = map0.storage()
    output["counters"] = {name: value for name, value in domain.report().items() if name != "owner_fit"}


def _payload(domain):
    """Payload collectives issued so far in either reconstruction mode."""
    return domain.counters["reduce_scatter_calls"] + domain.counters["factor_exchange_rounds"]


def held_tensor_bytes(exc, suffixes=("factor_sharding.py",)):
    """Bytes of tensors that the frames of ``suffixes`` in a failure's traceback (and its chain) still reference
    through their own locals (parameters, i.e. caller-owned objects, and ``self`` are not counted), and the
    holders as ``["function.local=bytes", ...]``."""
    seen, total, chain, holders = set(), 0, [], []

    def visit(value, depth):
        nonlocal total
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            if storage.data_ptr() not in seen:
                seen.add(storage.data_ptr())
                total += storage.nbytes()
        elif depth and isinstance(value, dict):
            for item in value.values():
                visit(item, depth - 1)
        elif depth and isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item, depth - 1)
        elif depth and dataclasses.is_dataclass(value) and not isinstance(value, type):
            for item in dataclasses.fields(value):
                visit(getattr(value, item.name, None), depth - 1)

    while exc is not None and exc not in chain:
        chain.append(exc)
        tb = exc.__traceback__
        while tb is not None:
            code = tb.tb_frame.f_code
            if code.co_filename.endswith(suffixes):
                parameters = set(code.co_varnames[:code.co_argcount + code.co_kwonlyargcount]) | {"self"}
                for name, value in tb.tb_frame.f_locals.items():
                    if name not in parameters:
                        before = total
                        visit(value, 4)
                        if total > before:
                            holders.append(f"{code.co_name}.{name}={total - before}")
            tb = tb.tb_next
        exc = exc.__cause__ or exc.__context__
    return total, holders


def _failures(rank, world, root, domain, natives, output):
    from reelcache.distributed import CollectiveCancelled, PeerFailure
    from reelcache.factor_sharding import OwnerShard, fresh_fit

    heads = natives["heads"]
    keys = [MatrixKey(1, "V"), MatrixKey(2, "V"), MatrixKey(1, "K")]
    shards = {key: domain.fit_and_scatter(key, _local(matrix(key.layer, key.target), heads, rank), heads,
                                          context(key.layer, key.target),
                                          fit=fresh_fit(RANKS[(key.layer, key.target)])) for key in keys}
    map0 = domain.publish_node(domain.stage_node("f0", shards, kind="svd"))

    def expect(name, function, exception, text=None):
        try:
            function()
        except exception as exc:
            output[name] = text is None or text in str(exc)
            output[f"{name}_message"] = str(exc)[:300]
            output[f"{name}_held_bytes"], output[f"{name}_holders"] = held_tensor_bytes(exc)
        except BaseException as exc:  # noqa: BLE001 - recorded as a failed expectation
            output[name] = False
            output[f"{name}_message"] = f"unexpected {type(exc).__name__}: {exc}"[:300]
        else:
            output[name] = False

    staged = domain.stage_node("f1", shards, kind="svd_hevc", codec=CodecSpec("svd_hevc", rank=CHANNELS))
    expect("cancel_publish", lambda: domain.publish_node(staged, cancelled=rank == world - 1), CollectiveCancelled)
    output["cancel_publish_nothing_visible"] = all(_gather(domain.store.published("f1") is None, world))
    output["cancel_publish_staged_packs"] = len(staged["packs"])  # an owner of no matrix stages nothing
    output["cancel_publish_gc"] = domain.discard_staged(staged)["objects_removed"]
    altered = dict(shards)
    if rank == 1:
        key = MatrixKey(2, "V")
        shard = shards[key]
        spec = shard.spec.with_view(shard.spec.rank_view - 1)
        rows = spec.view_rows(rank)
        altered[key] = OwnerShard(key, spec, shard.context, rank,
                                  None if rows == 0 else shard.coefficients[:, :rows].contiguous(), shard.basis)
    staged = domain.stage_node("f2", altered, kind="svd")
    expect("disagreement", lambda: domain.publish_node(staged), PeerFailure, "disagree")
    output["disagreement_nothing_visible"] = all(_gather(domain.store.published("f2") is None, world))
    domain.discard_staged(staged)
    if rank == 1:
        domain.store.store.set_ref("factor_shard/f3", domain.store.published("f0"))
    staged = domain.stage_node("f3", shards, kind="svd")
    expect("commit_failure", lambda: domain.publish_node(staged), PeerFailure, "already published")
    visible = _gather(domain.store.published("f3"), world)
    output["commit_failure_rolled_back"] = all(ref is None for owner, ref in enumerate(visible) if owner != 1)
    output["commit_failure_foreign_ref_untouched"] = visible[1] == _gather(domain.store.published("f0"), world)[1]
    output["policy"] = {"mode": domain.mode, "block_rows": domain.block_rows}
    before = _payload(domain)
    victim = world - 1
    pack_ref = map0.owner_record(victim)["packs"][0]["ref"]
    path = domain.store.store.path_for(pack_ref)
    original = path.read_bytes() if rank == victim else None
    if rank == victim:
        os.chmod(path, 0o644)
        path.write_bytes(original[:-1] + bytes([original[-1] ^ 0xFF]))
    _gather(None, world)
    expect("corrupt_fragment", lambda: domain.restore_node(map0, heads, dtype=torch.float32), PeerFailure,
           "rank " + str(victim))
    output["corrupt_fragment_no_payload"] = _payload(domain) == before
    if rank == victim:
        path.write_bytes(original)
    _gather(None, world)
    expect("local_decode_failure", lambda: domain.restore_node(map0, heads, dtype=torch.float32,
                                                               pool="not-a-pool" if rank == 0 else None),
           PeerFailure, "rank 0")
    output["local_failure_no_payload"] = _payload(domain) == before
    expect("order_mismatch", lambda: domain.restore_node(map0, heads, dtype=torch.float32,
                                                         entries=keys if rank else keys[:1]), PeerFailure)
    domain.checkpoint_every = 1
    for mode in ("factors", "partials"):
        calls = {"count": 0}

        def cancel(calls=calls):
            calls["count"] += 1
            return rank == 0 and calls["count"] >= 3

        started = _payload(domain)
        expect(f"cancel_mid_restore_{mode}",
               lambda mode=mode, cancel=cancel: domain.restore_node(map0, heads, dtype=torch.float32, cancel=cancel,
                                                                    mode=mode), CollectiveCancelled)
        output[f"cancel_mid_restore_{mode}_partial_payload"] = _payload(domain) > started
    output["cancelled_operations"] = domain.counters["cancelled_operations"]
    domain.checkpoint_every = 0
    reference = _reference(domain, map0, world, rank, rotate=False)
    output["usable_after_failures"] = all(
        _compare(domain.restore_node(map0, heads, dtype=torch.float32, mode=mode), reference, heads, rank) < 1e-5
        for mode in ("factors", "partials"))
    output["broken"] = domain._broken


def _integrity(rank, world, root, domain, natives, output):
    """R1 / R2 (one address check per stored byte read): with a verifying store (verify_reads) the owner store skips
    its own SHA-256, without one it checks every pack and segment itself; packs are never re-hashed per payload on the
    read path. Corrupted stored bytes of a pack or a basis segment are detected either way."""
    import reelcache.factor_sharding as fs
    from reelcache.distributed import PeerFailure
    from reelcache.factor_sharding import fresh_fit

    heads = natives["heads"]
    keys = [MatrixKey(1, "V"), MatrixKey(2, "V"), MatrixKey(1, "K"), MatrixKey(0, "V")]
    shards = {key: domain.fit_and_scatter(key, _local(matrix(key.layer, key.target), heads, rank), heads,
                                          context(key.layer, key.target),
                                          fit=fresh_fit(RANKS[(key.layer, key.target)])) for key in keys}
    node_map = domain.publish_node(domain.stage_node("i0", shards, kind="svd_hevc",
                                                     codec=CodecSpec("svd_hevc", rank=CHANNELS)))
    output["verify_reads"] = domain.store.store.verify_reads
    reference = _reference(domain, node_map, world, rank, rotate=False)
    mine = node_map.owner_record(rank)
    expected_checks = len(mine["packs"]) + sum(len(node_map.segment_refs(rank, entry)) for entry in node_map.entries)
    output["owner_packs"], output["expected_explicit_checks"] = len(mine["packs"]), expected_checks
    calls = {"sha": 0}
    real_sha = fs._sha

    def counting_sha(data):
        calls["sha"] += 1
        return real_sha(data)

    fs._sha = counting_sha
    try:
        restored = domain.restore_node(node_map, heads, dtype=torch.float32)
    finally:
        fs._sha = real_sha
    output["explicit_sha_calls"] = calls["sha"]
    output["restore_matches_reference"] = _compare(restored, reference, heads, rank) < 1e-5
    first = restored
    victim = world - 1

    def corrupt_and_restore(name, ref):
        path = domain.store.store.path_for(ref) if rank == victim else None
        original = path.read_bytes() if rank == victim else None
        if rank == victim:
            os.chmod(path, 0o644)
            middle = len(original) // 2
            path.write_bytes(original[:middle] + bytes([original[middle] ^ 0x5A]) + original[middle + 1:])
        _gather(None, world)
        try:
            domain.restore_node(node_map, heads, dtype=torch.float32)
            output[name] = "NO ERROR"
        except PeerFailure as exc:
            output[name] = f"PeerFailure: {exc}"[:400]
        except Exception as exc:  # noqa: BLE001 - recorded as the wrong failure
            output[name] = f"WRONG {type(exc).__name__}: {exc}"[:400]
        if rank == victim:
            path.write_bytes(original)
        _gather(None, world)

    victim_record = node_map.owner_record(victim)
    corrupt_and_restore("corrupt_pack", victim_record["packs"][0]["ref"] if victim_record["packs"] else None)
    corrupt_and_restore("corrupt_segment", victim_record["segments"][0]["ref"])
    again = domain.restore_node(node_map, heads, dtype=torch.float32)
    output["restorable_after_corruption_bitwise"] = all(torch.equal(again[key], first[key]) for key in first)
    output["broken"] = domain._broken


def _peer_exit(rank, world, root, domain, natives, output):
    from reelcache.distributed import PeerFailure
    from reelcache.factor_sharding import fresh_fit

    heads = natives["heads"]
    key = MatrixKey(1, "V")
    shard = domain.fit_and_scatter(key, _local(matrix(1, "V"), heads, rank), heads, context(1, "V"),
                                   fit=fresh_fit(33))
    node_map = domain.publish_node(domain.stage_node("p0", {key: shard}, kind="svd"))
    dist.barrier()
    if rank == world - 1:
        dist.destroy_process_group()
        output["peer_exited"] = True
        return False
    try:
        domain.restore_node(node_map, heads, dtype=torch.float32)
    except PeerFailure as exc:
        output["failure_propagated"] = "failed" in str(exc)
    else:
        output["failure_propagated"] = False
    return True


def _worker(rank, world, root, scenario):
    import traceback

    root = Path(root)
    os.environ["TMPDIR"] = str(root)
    torch.set_num_threads(1)
    output = {}
    try:
        from reelcache.distributed import DistributedOwner
        from reelcache.factor_sharding import FactorShardDomain, OwnerFactorStore
        from reelcache.store import ObjectStore

        dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(seconds=90),
                                init_method=f"file://{root / 'rendezvous'}")
        data_group = dist.new_group(backend="gloo")
        coordinator = DistributedOwner(expected_world_size=world)
        store = OwnerFactorStore(ObjectStore(root / f"owner{rank}", verify_reads=scenario == "integrity_verified"),
                                 owner=rank, world_size=world)
        from reelcache.factor_sharding import PLAN_BLOCK_ROWS

        policy = {"mode": "partials", "block_rows": PLAN_BLOCK_ROWS} if scenario == "lifecycle_plan" else {}
        lifecycle = scenario.startswith("lifecycle")
        domain = FactorShardDomain(coordinator, data_group=data_group, store=store, tile_tokens=40,
                                   checkpoint_every=3 if lifecycle else 0,
                                   control_chunk=2048 if lifecycle else 192 * 1024, **policy)
        layout = KVLayout(TOKENS, HEADS, HEAD_DIM, world)
        natives = {"heads": NativeOwners(layout, "heads"), "rows": NativeOwners(layout, "rows"),
                   "rows_uneven": NativeOwners(layout, "rows", _uneven_rows(world))}
        keep = True
        if lifecycle:
            _lifecycle(rank, world, root, domain, natives, output)
        elif scenario == "failures":
            _failures(rank, world, root, domain, natives, output)
        elif scenario.startswith("integrity"):
            _integrity(rank, world, root, domain, natives, output)
        else:
            keep = _peer_exit(rank, world, root, domain, natives, output)
        output["cuda_initialized"] = torch.cuda.is_initialized()
        store.store.close()
        if keep:
            dist.destroy_process_group()
    except BaseException as exc:  # noqa: BLE001 - surfaced by the parent test
        output["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        (root / f"rank{rank}.json").write_text(json.dumps(output, default=str))
        os._exit(0)


class GlooFactorShardingTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / ".test-artifacts" / f"factor-shard-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _run(self, world, scenario):
        context_ = multiprocessing.get_context("spawn")
        processes = [context_.Process(target=_worker, args=(rank, world, str(self.root), scenario))
                     for rank in range(world)]
        for process in processes:
            process.start()
        try:
            for process in processes:
                process.join(timeout=600)
                self.assertFalse(process.is_alive(), "Gloo worker timed out")
            rows = [json.loads((self.root / f"rank{rank}.json").read_text()) for rank in range(world)]
            for row in rows:
                self.assertNotIn("error", row, row.get("error"))
            return rows
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)

    def _check_lifecycle(self, world, scenario="lifecycle"):
        rows = self._run(world, scenario)
        natives_names = ("heads", "rows", "rows_uneven")
        for rank, row in enumerate(rows):
            for name in ("scatter_bitwise_vs_single_process_fit", "n0_published_everywhere", "n0_digest_agreed",
                         "n0_map_roundtrip", "n1_old_rows_kept_owners", "n1_parent_segments_shared",
                         "n1_store_roundtrip_bitwise", "fork_parent_segments_shared", "fork_extensions_distinct"):
                self.assertTrue(row[name], f"rank {rank}: {name}")
            self.assertIsNone(row["n0_stage_error"])
            self.assertGreater(row["n0_record_chunks"], 1)  # the node map crossed several bounded control rounds
            self.assertEqual(len(row["n0_max_rel_error_vs_single_gemm"]), 9)
            for name, error in row["n0_max_rel_error_vs_single_gemm"].items():
                self.assertLess(error, 2e-6, f"{name}: FP32 reconstruction error")
            for name, moved in row["n0_wire_by_mode"].items():
                if "/factors" in name:  # no reduce-scatter in factors mode; partials mode never exchanges tiles
                    self.assertEqual(moved["reduce_scatter_bytes_sent"], 0)
                else:
                    self.assertEqual(moved["factor_exchange_bytes_sent"], 0)
                    self.assertGreater(moved["reduce_scatter_bytes_sent"], 0)
            for name in natives_names:  # one round per matrix by default; tile-sized chunks when forced
                one = row["n0_wire_by_mode"][f"{name}/factors"]
                chunked = row["n0_wire_by_mode"][f"{name}/factors_chunked"]
                # every native owner of "heads" needs all TOKENS (> tile_tokens) rows of every matrix
                compare = self.assertGreater if name == "heads" else self.assertGreaterEqual
                compare(chunked["factor_exchange_rounds"], one["factor_exchange_rounds"], name)
                self.assertEqual(chunked["factor_exchange_bytes_sent"], one["factor_exchange_bytes_sent"], name)
            self.assertLess(row["n1_rows_max_rel_error"], 2e-6)
            self.assertLess(row["n2a_max_rel_error"], 2e-6)
            self.assertLess(row["n2b_max_rel_error"], 2e-6)
            self.assertLess(row["n0_bf16_mismatch_fraction"], 0.01)
            self.assertEqual(row["n0_heads_shapes"], [[TOKENS, CHANNELS // world]])
            self.assertEqual(row["n1_version_rows"]["1:V"], [33, 43])
            self.assertEqual(row["n1_version_rows"]["0:V"], [40, 60])
            self.assertEqual(row["n1_version_rows"]["3:K"], [16])
            self.assertGreater(row["n0_heads_reduce_scatter_calls"], 0)
            self.assertFalse(row["cuda_initialized"])
            self.assertEqual(row["counters"]["retired_transfers"], 0)
        totals = rows[0]["storage_totals"]
        self.assertEqual(totals["basis_bytes"], rows[0]["unique_segment_bytes"])  # each row once, on one owner
        self.assertLess(rows[0]["unique_segment_bytes"], rows[0]["replicated_basis_bytes"])
        loads = rows[0]["view_loads"]
        self.assertEqual(loads["0:K"].count(0), world - 1)  # rank 6 < one block: a single owner
        self.assertEqual(sum(loads["1:K"]), 64)
        self.assertEqual(rows[0]["owner_fit_reports"], len(RANKS))
        self.assertGreater(rows[0]["n0_zero_partial_calls"], 0)
        for name in rows[0]["n0_wire_by_mode"]:  # aggregate over ranks: factor tiles < partial outputs
            if name.endswith("/factors"):
                moved = sum(row["n0_wire_by_mode"][name]["factor_exchange_bytes_sent"] for row in rows)
                partials = sum(row["n0_wire_by_mode"][name.replace("factors", "partials")]
                               ["reduce_scatter_bytes_sent"] for row in rows)
                self.assertLess(moved, partials, name)
        if scenario == "lifecycle":  # default policy: whole-matrix owners, factor-tile reconstruction
            self.assertEqual(rows[0]["policy"], {"mode": "factors", "block_rows": None})
            self.assertTrue(rows[0]["whole_matrix"])
            self.assertTrue(all(count == 1 for count in rows[0]["blocks"].values()))
            for label, owned in loads.items():
                self.assertEqual(max(owned), sum(owned), label)  # exactly one owner per matrix
        else:  # plan-literal ablation: 32-row blocks, FP32 partial-sum reduce-scatter
            self.assertEqual(rows[0]["policy"], {"mode": "partials", "block_rows": PLAN_BLOCK_ROWS})
            self.assertFalse(rows[0]["whole_matrix"])
            self.assertEqual(rows[0]["blocks"]["1:K"], 2)
            self.assertEqual(max(loads["1:K"]), 32)  # one matrix split over two owners
        return rows

    def test_two_rank_lifecycle_growth_fork_and_unique_bytes(self):
        self._check_lifecycle(2)

    def test_four_rank_lifecycle_growth_fork_and_unique_bytes(self):
        self._check_lifecycle(4)

    def test_two_rank_lifecycle_plan_ablation(self):
        self._check_lifecycle(2, "lifecycle_plan")

    def test_four_rank_lifecycle_plan_ablation(self):
        self._check_lifecycle(4, "lifecycle_plan")

    def _check_failures(self, world):
        rows = self._run(world, "failures")
        for rank, row in enumerate(rows):
            for name in ("cancel_publish", "cancel_publish_nothing_visible", "disagreement",
                         "disagreement_nothing_visible", "commit_failure", "commit_failure_rolled_back",
                         "commit_failure_foreign_ref_untouched", "corrupt_fragment", "corrupt_fragment_no_payload",
                         "local_decode_failure", "local_failure_no_payload", "order_mismatch",
                         "cancel_mid_restore_factors", "cancel_mid_restore_factors_partial_payload",
                         "cancel_mid_restore_partials", "cancel_mid_restore_partials_partial_payload",
                         "usable_after_failures"):
                self.assertTrue(row[name], f"rank {rank}: {name}: {row.get(name + '_message')}")
            # a failed or cancelled restore keeps none of its buffers (outputs, decoded factors) in its traceback
            for name in ("corrupt_fragment", "local_decode_failure", "order_mismatch", "cancel_mid_restore_factors",
                         "cancel_mid_restore_partials"):
                self.assertEqual(row[f"{name}_held_bytes"], 0, f"rank {rank}: {name}: {row[name + '_holders']}")
            # the cancelled node's packs (svd_hevc, unshared) are collected; shared segments of f0 survive
            self.assertGreaterEqual(row["cancel_publish_gc"], row["cancel_publish_staged_packs"])
            self.assertFalse(row["broken"])
            self.assertEqual(row["policy"], {"mode": "factors", "block_rows": None})
            self.assertGreaterEqual(row["cancelled_operations"], 3)
        self.assertGreater(sum(row["cancel_publish_gc"] for row in rows), 0)
        return rows

    def test_two_rank_failures_cancellation_and_integrity(self):
        self._check_failures(2)

    def test_four_rank_failures_cancellation_and_integrity(self):
        self._check_failures(4)

    def _check_integrity(self, scenario, verify_reads):
        rows = self._run(2, scenario)
        victim = 1
        for rank, row in enumerate(rows):
            self.assertIs(row["verify_reads"], verify_reads)
            self.assertTrue(row["restore_matches_reference"], f"rank {rank}")
            self.assertGreater(row["owner_packs"], 0)
            # R1: explicit address checks only without a verifying store; R2: no per-payload re-hash on reads
            self.assertEqual(row["explicit_sha_calls"], 0 if verify_reads else row["expected_explicit_checks"],
                             f"rank {rank}")
            for name, store_text, own_text in (
                    ("corrupt_pack", "hash or length mismatch",
                     "fragment pack bytes differ from their content address"),
                    ("corrupt_segment", "hash or length mismatch", "basis segment bytes differ from their address")):
                self.assertTrue(row[name].startswith("PeerFailure"), f"rank {rank}: {name}: {row[name]}")
                self.assertIn(f"rank {victim}: IntegrityError", row[name])
                self.assertIn(store_text if verify_reads else own_text, row[name])
            self.assertTrue(row["restorable_after_corruption_bitwise"])
            self.assertFalse(row["broken"])

    def test_reads_are_checked_once_without_a_verifying_store(self):
        self._check_integrity("integrity_unverified", False)

    def test_reads_are_checked_once_by_a_verifying_store(self):
        self._check_integrity("integrity_verified", True)

    def test_peer_exit_is_a_failure_not_a_result(self):
        rows = self._run(2, "peer_exit")
        self.assertTrue(rows[0]["failure_propagated"])
        self.assertTrue(rows[1]["peer_exited"])


if __name__ == "__main__":
    unittest.main()
