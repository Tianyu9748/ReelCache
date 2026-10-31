"""Step 16 integration: factor-sharded archive / restore of a context-parallel LongCat conditioning cache.

``reelcache.integrations.longcat_factor_shard`` driven through the SGLang port's own pure seam types
(``ConditioningCacheEpoch`` with a ``RowShard``, ``LongCatConditioningCache``, ``CacheRowSeam``) on 2- and 4-rank
Gloo groups: capture -> deferred flush (owner-fit K98/V95 energy fit, per-owner HEVC packs, two-phase publication;
RAW rows per rank) -> install through the seam (RAW bitwise; lossy rows equal to the single-process reconstruction
of every owner's decoded fragments), cancellation and a peer's failure during a sharded restore surface on every
rank and leave the archive restorable. CPU only; artifacts stay in the checkout."""

import dataclasses
from datetime import timedelta
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import unittest
import uuid

import torch
import torch.distributed as dist

from reelcache.errors import ConfigurationError, IntegrityError
from reelcache.integrations.longcat_factor_shard import (
    RowShardedCacheArchive, energy_fit, epoch_key, native_owners_for, pack_canvas_shapes, rel_fro,
)
from reelcache.types import MatrixKey, ModelIdentity, TensorContext

LAYERS, HEADS, HEAD_DIM = 3, 4, 8
CHANNELS = HEADS * HEAD_DIM
GRID = (2, 4, 6)
TOKENS = GRID[0] * GRID[1] * GRID[2]
ENERGY = {"K": 0.98, "V": 0.95}


def identity():
    return ModelIdentity("tiny_longcat_cp", "step16-integration-fixture", LAYERS, CHANNELS, "global_KV_channels",
                         "bfloat16")


def full_matrix(layer, target):
    """Deterministic low-rank-plus-noise canonical [tokens, channels] cache matrix (BF16, like the port)."""
    generator = torch.Generator().manual_seed(17 * layer + (target == "V"))
    true = 6 + 3 * layer
    scales = torch.logspace(0, -1.5, true)
    value = (torch.randn(TOKENS, true, generator=generator) * scales) @ torch.randn(true, CHANNELS, generator=generator)
    return (value + 2e-2 * torch.randn(TOKENS, CHANNELS, generator=generator)).to(torch.bfloat16)


def epoch_for(rank, world, session="s16-integration"):
    from reelcache_sglang_models.longcat_video.conditioning_cache import ConditioningCacheEpoch
    from reelcache_sglang_models.longcat_video.context_parallel import RowShard, optimal_split

    shard = None if world is None else RowShard(world, rank, optimal_split(world))
    return ConditioningCacheEpoch(segment_index=0, token_grid=GRID, cond_latents_sha256="ab" * 32,
                                  model_fingerprint="tiny-fp", session_id=session, row_shard=shard)


class EnergyFitTests(unittest.TestCase):
    def test_certified_prefix_meets_the_target_and_matches_fp64_eigh(self):
        for layer in range(LAYERS):
            for target in ("K", "V"):
                x = full_matrix(layer, target)
                context = TensorContext(identity(), MatrixKey(layer, target), 0, *GRID,
                                        "pre_rope" if target == "K" else "none", source_dtype="bfloat16")
                result = energy_fit(ENERGY[target])(x, context, None)
                metrics = result.metrics
                self.assertGreaterEqual(metrics["stored_energy"], ENERGY[target])
                self.assertGreaterEqual(result.rank_view, metrics["spectral_rank"])
                self.assertLess(result.rank_view, CHANNELS)
                evals = torch.linalg.eigvalsh(x.double().T @ x.double()).flip(0).clamp_min(0)
                cdf = torch.cumsum(evals, 0) / float((x.double() ** 2).sum())
                expected = int(torch.searchsorted(cdf, torch.tensor([ENERGY[target] - 1e-12], dtype=cdf.dtype))) + 1
                self.assertEqual(metrics["spectral_rank"], expected)
                self.assertEqual(result.fresh_basis.dtype, torch.float16)
                self.assertEqual(tuple(result.coefficients.shape), (TOKENS, result.rank_view))
                approx = result.coefficients @ result.fresh_basis.float()
                residual = rel_fro(approx, x.float())
                self.assertAlmostEqual(residual, metrics["projection_relative_residual"], places=5)
                if result.rank_view > metrics["spectral_rank"]:  # grown only while the stored energy fell short
                    with self.assertRaises(IntegrityError):
                        energy_fit(ENERGY[target], extra_rows=0)(x, context, None)

    def test_rejects_bad_targets_and_parents(self):
        with self.assertRaises(ConfigurationError):
            energy_fit(1.0)
        with self.assertRaises(ConfigurationError):
            energy_fit(0.98)(full_matrix(0, "K"), None, torch.zeros(1, CHANNELS, dtype=torch.float16))

    def test_epoch_identity_and_native_owners(self):
        whole = epoch_for(0, None)
        keys = {epoch_key(epoch_for(rank, 2)) for rank in range(2)} | {epoch_key(whole)}
        self.assertEqual(len(keys), 1)  # every rank of the group and the whole epoch share one identity
        self.assertNotEqual(epoch_key(epoch_for(0, 2, session="other")), keys.pop())
        for world in (2, 4):
            for rank in range(world):
                epoch = epoch_for(rank, world)
                native = native_owners_for(epoch, kv_heads=HEADS, head_dim=HEAD_DIM, world_size=world)
                self.assertTrue(torch.equal(native.rows(rank), epoch.row_index()))
        with self.assertRaises(ConfigurationError):
            native_owners_for(whole, kv_heads=HEADS, head_dim=HEAD_DIM, world_size=2)
        with self.assertRaises(ConfigurationError):
            native_owners_for(epoch_for(0, 2), kv_heads=HEADS, head_dim=HEAD_DIM, world_size=4)

    def test_archive_refuses_another_ranks_rows(self):
        from reelcache.factor_sharding import FactorShardDomain

        domain = FactorShardDomain.__new__(FactorShardDomain)  # rank/world only; no collectives are reached
        domain.rank, domain.world_size, domain.device = 0, 2, torch.device("cpu")
        archive = RowShardedCacheArchive.__new__(RowShardedCacheArchive)
        archive.domain, archive.heads, archive.head_dim = domain, HEADS, HEAD_DIM
        self.assertTrue(torch.equal(archive.native_owners(epoch_for(0, 2)).rows(0), epoch_for(0, 2).row_index()))
        with self.assertRaises(ConfigurationError):
            archive.native_owners(epoch_for(1, 2))


def _gather(value, world):
    rows = [None] * world
    dist.all_gather_object(rows, value)
    return rows


def held_tensor_bytes(exc, suffixes=("factor_sharding.py", "longcat_factor_shard.py")):
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


class RecordingPool:
    """Stands in for a DecoderPool's warm-up API (CPU tests decode with FFmpeg; a DecoderPool is CUDA only)."""

    max_sessions_per_worker = 16

    def __init__(self):
        self.calls = []

    def prewarm_shapes(self, shapes, timeout=None):
        self.calls.append([list(shape) for shape in shapes])
        return {"shapes": len(shapes), "warmed_shapes": len(shapes)}


class FailingPool(RecordingPool):
    def prewarm_shapes(self, shapes, timeout=None):
        super().prewarm_shapes(shapes, timeout)
        raise RuntimeError("injected: decoder session warm-up failed")


def _expect(output, name, function, error):
    try:
        function()
    except error as exc:
        output[name] = f"{type(exc).__name__}: {exc}"[:300]
        output[name + "_held_bytes"], output[name + "_holders"] = held_tensor_bytes(exc)
        return
    except Exception as exc:  # noqa: BLE001 - recorded as the wrong failure
        output[name] = f"WRONG {type(exc).__name__}: {exc}"[:300]
        return
    output[name] = "NO ERROR"


def _reference_rows(archive, epoch, world, rank):
    """Every owner decodes its fragments; the full FP32 reconstruction P @ Q is cast to BF16 (single process)."""
    from reelcache.integrations.longcat_factor_shard import epoch_key as key_of

    domain = archive.domain
    node_map = domain.store.node_map(archive.node_id(key_of(epoch), "svd_hevc"))
    shards = domain.load_node(node_map)
    mine = {}
    for entry, shard in shards.items():
        if shard.coefficients is not None:
            mine[entry] = (shard.coefficients.float() @ shard.view_basis().float()).to(torch.bfloat16)
    full = {}
    for row in _gather({str(k): v for k, v in mine.items()}, world):
        full.update(row)
    rows = epoch.row_index()
    return {entry: full[str(entry)].index_select(0, rows) for entry in archive.keys()}


def _worker(rank, world, root, scenario):
    root = Path(root)
    torch.set_num_threads(1)
    output = {"rank": rank}
    try:
        from reelcache.distributed import CollectiveCancelled, DistributedOwner, PeerFailure
        from reelcache.factor_sharding import FactorShardDomain, OwnerFactorStore, entry_from_dict, entry_label
        from reelcache.store import ObjectStore
        from reelcache_sglang_models.longcat_video import conditioning_cache as cc

        dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(seconds=120),
                                init_method=f"file://{root / 'rendezvous'}")
        data_group = dist.new_group(backend="gloo")
        store = OwnerFactorStore(ObjectStore(root / f"owner{rank}"), owner=rank, world_size=world)
        domain = FactorShardDomain(DistributedOwner(expected_world_size=world), data_group=data_group, store=store,
                                   tile_tokens=7, checkpoint_every=1)
        flush_pool = FailingPool() if scenario == "failures" else RecordingPool()
        archive = RowShardedCacheArchive(domain, identity=identity(), layers=LAYERS, heads=HEADS, head_dim=HEAD_DIM,
                                         energy=ENERGY, threads=1, pack_groups=2, pool=flush_pool)
        seam = cc.CacheRowSeam(heads=HEADS, capture=archive.capture, install=archive.install)
        epoch = epoch_for(rank, world)
        rows = epoch.row_index()
        source = {MatrixKey(layer, target): full_matrix(layer, target).index_select(0, rows)
                  for layer in range(LAYERS) for target in ("K", "V")}
        keys = [source[MatrixKey(i, "K")] for i in range(LAYERS)]
        values = [source[MatrixKey(i, "V")] for i in range(LAYERS)]
        cache = cc.LongCatConditioningCache.from_matrices(epoch, keys, values, heads=HEADS,
                                                          source=cc.SOURCE_NATIVE_BUILD)
        archive.arm(None)
        output["unarmed_install_is_none"] = seam.restore(epoch) is None
        seam.on_cache_built(cache)  # capture: host copy only
        output["captured_pending"] = len(archive.pending) == 1
        reports = archive.flush()
        archive.pool = None  # the CPU restores below decode with FFmpeg (a DecoderPool is CUDA only)
        report = reports[0]
        output["flush_prewarm_calls"] = flush_pool.calls
        output["flush_prewarm_error"] = report.get("canvas_prewarm", {}).get("error")
        output["flush_agreed_node_map"] = len(set(_gather(report["svd_hevc"]["node_map_sha256"], world))) == 1
        fits = report["svd_hevc"]["owner_fits"]
        output["fit_owners"] = sorted({row["fit_owner"] for row in fits})
        output["fits_meet_targets"] = all(
            row["metrics"]["stored_energy"] >= ENERGY[row["entry"].split(":")[-1]] for row in fits)
        output["fits_reported"] = len(fits)
        node_map = domain.store.node_map(report["svd_hevc"]["node_id"])
        tiles = sorted(entry_label(entry_from_dict(tile["entry"]))
                       for pack in node_map.owner_record(rank)["packs"] for tile in pack["tiles"])
        mine = sorted(entry_label(entry) for entry in node_map.entries if node_map.spec(entry).view_rows(rank))
        output["owner_tiles_are_its_own_rows"] = tiles == mine and len(mine) > 0
        output["owner_pack_bytes"] = report["svd_hevc"]["owner_pack_bytes"]
        output["owner_basis_bytes"] = report["svd_hevc"]["owner_basis_bytes"]
        output["raw_owner_bytes"] = report["raw"]["owner_bytes"]
        output["rank_sum"] = report["svd_hevc"]["rank_sum"]

        archive.arm("raw")
        restored = seam.restore(epoch)
        output["raw_install_bitwise"] = all(
            torch.equal(restored.layer_matrices(i)[0], source[MatrixKey(i, "K")])
            and torch.equal(restored.layer_matrices(i)[1], source[MatrixKey(i, "V")]) for i in range(LAYERS))
        output["raw_install_source"] = restored.source

        archive.arm("svd_hevc")
        lossy = seam.restore(epoch)
        mats = {MatrixKey(i, t): lossy.layer_matrices(i)[t == "V"] for i in range(LAYERS) for t in ("K", "V")}
        reference = _reference_rows(archive, epoch, world, rank)
        output["lossy_vs_reference_max_rel"] = max(rel_fro(mats[k].float(), reference[k].float()) for k in mats)
        output["lossy_vs_reference_bitwise"] = all(torch.equal(mats[k], reference[k]) for k in mats)
        output["lossy_vs_source_rel"] = {f"{k.target}{k.layer}": rel_fro(mats[k].float(), source[k].float())
                                         for k in mats}
        before = {k: v.clone() for k, v in mats.items()}

        # CR-17-5: the pack canvases derived from the node map are the encoded packs' canvases; warming them reads no
        # pack (attach: every published node; or the session's epochs) and the restore afterwards is bitwise the same
        derived = pack_canvas_shapes(node_map, rank)
        encoded = [domain.store.get_pack(pack["ref"]).metadata["canvas"]
                   for pack in node_map.owner_record(rank)["packs"]]
        output["canvas_shapes"] = [list(shape) for shape in derived]
        output["canvas_shapes_match_packs"] = derived == [(c["width"], c["height"], c["pix_fmt"]) for c in encoded]
        reads = {"packs": 0}
        real_get_pack = domain.store.get_pack

        def counting_get_pack(ref):
            reads["packs"] += 1
            return real_get_pack(ref)

        attach_pool = RecordingPool()
        domain.store.get_pack, archive.pool = counting_get_pack, attach_pool
        try:
            warmed = archive.prewarm_canvases()
            archive.prewarm_canvases([epoch])
        finally:
            archive.pool = None
            domain.store.__dict__.pop("get_pack", None)
        output["prewarm_calls"] = attach_pool.calls
        output["prewarm_node_ids"] = warmed["node_ids"]
        output["prewarm_read_no_pack"] = reads["packs"] == 0
        again = seam.restore(epoch)
        output["restore_after_prewarm_bitwise"] = all(
            torch.equal(again.layer_matrices(i)[t == "V"], before[MatrixKey(i, t)])
            for i in range(LAYERS) for t in ("K", "V"))

        if scenario == "failures":
            calls = {"n": 0}

            def cancel():
                calls["n"] += 1
                return rank == 0 and calls["n"] >= 3

            archive.cancel = cancel
            _expect(output, "cancel_mid_restore", lambda: seam.restore(epoch), CollectiveCancelled)
            archive.cancel = None
            original = domain.store.get_pack

            def broken(ref):
                raise OSError("injected unreadable fragment pack")

            if rank == world - 1:
                domain.store.get_pack = broken
            _expect(output, "peer_failure_during_restore", lambda: seam.restore(epoch), PeerFailure)
            domain.store.get_pack = original
            archive.arm("raw")
            archive.cancel = lambda: rank == 0
            _expect(output, "raw_cancel_after_load", lambda: seam.restore(epoch), CollectiveCancelled)
            archive.cancel = None
            if rank == 0:
                archive.domain.store.store.delete_ref("longcat_cc_raw/" + report["raw"]["node_id"])
            _expect(output, "raw_missing_on_one_owner", lambda: seam.restore(epoch), PeerFailure)
            archive.arm("svd_hevc")
            _expect(output, "unknown_epoch_armed", lambda: seam.restore(epoch_for(rank, world, session="never")),
                    PeerFailure)
            output["domain_not_broken"] = not domain._broken
            again = seam.restore(epoch)
            output["restorable_after_failures_bitwise"] = all(
                torch.equal(again.layer_matrices(i)[t == "V"], before[MatrixKey(i, t)])
                for i in range(LAYERS) for t in ("K", "V"))
        output["storage_totals"] = archive.storage()["totals"]
        output["cuda_initialized"] = torch.cuda.is_initialized()
        store.store.close()
        dist.destroy_process_group()
    except BaseException as exc:  # noqa: BLE001 - surfaced by the parent test
        import traceback

        output["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        (root / f"rank{rank}.json").write_text(json.dumps(output, default=str))
        os._exit(0)


class GlooIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / ".test-artifacts" / f"lc-factor-shard-{uuid.uuid4().hex}"
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

    def _check(self, world, scenario):
        rows = self._run(world, scenario)
        for row in rows:
            rank = row["rank"]
            for name in ("unarmed_install_is_none", "captured_pending", "flush_agreed_node_map", "fits_meet_targets",
                         "owner_tiles_are_its_own_rows", "raw_install_bitwise", "canvas_shapes_match_packs",
                         "prewarm_read_no_pack", "restore_after_prewarm_bitwise"):
                self.assertTrue(row[name], f"rank {rank}: {name}")
            # CR-17-5: flush warmed this owner's distinct canvases once; attach and the session's epochs warm the same
            shapes = []
            for shape in row["canvas_shapes"]:
                if shape not in shapes:
                    shapes.append(shape)
            self.assertTrue(shapes, f"rank {rank}: every rank owns packs here")
            self.assertEqual(row["flush_prewarm_calls"], [shapes])
            if scenario == "failures":  # a failed warm-up is recorded; the published archive still restores
                self.assertIn("injected: decoder session warm-up failed", row["flush_prewarm_error"])
            else:
                self.assertIsNone(row["flush_prewarm_error"])
            self.assertEqual(row["prewarm_calls"], [shapes, shapes])
            self.assertEqual(row["prewarm_node_ids"], [rows[0]["prewarm_node_ids"][0]])
            self.assertEqual(row["raw_install_source"], "restored")
            self.assertEqual(row["fits_reported"], 2 * LAYERS)  # every rank records every fit owner's report
            self.assertLess(row["lossy_vs_reference_max_rel"], 1e-6)  # one GEMM per native tile, BF16 cast
            for name, error in row["lossy_vs_source_rel"].items():
                limit = 0.2 if name[0] == "K" else 0.3  # sqrt(1 - energy) plus HEVC and BF16 error
                self.assertLess(error, limit, name)
            self.assertFalse(row["cuda_initialized"])
        self.assertEqual(sorted({owner for row in rows for owner in row["fit_owners"]}), list(range(world)))
        self.assertEqual(len({row["rank_sum"] for row in rows}), 1)
        totals = rows[0]["storage_totals"]
        self.assertEqual(totals["basis_bytes"], sum(row["owner_basis_bytes"] for row in rows))
        raw = sum(row["raw_owner_bytes"] for row in rows)
        self.assertEqual(raw, 2 * LAYERS * TOKENS * CHANNELS * 2)  # every row stored once, on its own rank
        return rows

    def test_two_rank_capture_flush_install(self):
        self._check(2, "lifecycle")

    def test_four_rank_capture_flush_install(self):
        self._check(4, "lifecycle")

    def test_two_rank_cancel_and_peer_failure_leave_the_archive_restorable(self):
        rows = self._check(2, "failures")
        for row in rows:
            for name in ("cancel_mid_restore", "raw_cancel_after_load"):
                self.assertTrue(row[name].startswith("CollectiveCancelled"), f"rank {row['rank']}: {name}: {row[name]}")
            for name in ("peer_failure_during_restore", "raw_missing_on_one_owner", "unknown_epoch_armed"):
                self.assertTrue(row[name].startswith("PeerFailure"), f"rank {row['rank']}: {name}: {row[name]}")
            # a failed or cancelled install keeps none of its buffers (rows, decoded factors) in its traceback
            for name in ("cancel_mid_restore", "raw_cancel_after_load", "peer_failure_during_restore",
                         "raw_missing_on_one_owner", "unknown_epoch_armed"):
                self.assertEqual(row[name + "_held_bytes"], 0, f"rank {row['rank']}: {name}: {row[name + '_holders']}")
            self.assertIn("injected unreadable fragment pack", rows[0]["peer_failure_during_restore"])
            self.assertTrue(row["domain_not_broken"])
            self.assertTrue(row["restorable_after_failures_bitwise"])


if __name__ == "__main__":
    unittest.main()
