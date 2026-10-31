"""Tiered SP history: native hot shards, distributed cold decode, and owner gather for compression."""

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

from reelcache.codecs import decode_tensor, encode_tensor
from reelcache.errors import ConfigurationError, IntegrityError
from reelcache.sp_kv import HotShards, pack_encoded, unpack_encoded
from reelcache.types import CodecSpec, EncodedTensor, MatrixKey
from tests.test_codecs import context


class WireAndHotTierTests(unittest.TestCase):
    def test_packet_wire_form_roundtrips_every_kind(self):
        source = torch.randn(24, 4, generator=torch.Generator().manual_seed(5))
        for spec in (CodecSpec(), CodecSpec("svd", rank=2), CodecSpec("svd_hevc", rank=3)):
            packet = encode_tensor(source, context(), spec)
            restored = unpack_encoded(pack_encoded(packet))
            self.assertEqual(restored.metadata, json.loads(json.dumps(packet.metadata)))
            self.assertEqual(dict(restored.payloads), dict(packet.payloads))
            self.assertEqual(restored.basis, packet.basis)
            self.assertTrue(torch.equal(decode_tensor(restored), decode_tensor(packet)))
        blob = pack_encoded(packet)
        for bad in (blob[:-1], blob[:4], blob + b"x"):
            with self.assertRaises(IntegrityError):
                unpack_encoded(bad)

    def test_hot_tier_evicts_oldest_chunks_only_and_is_deterministic(self):
        def pair(n):
            return torch.zeros(n, dtype=torch.uint8), torch.zeros(n, dtype=torch.uint8)

        hot = HotShards(100)
        for chunk in range(3):
            self.assertTrue(hot.admit((str(chunk), 0), *pair(20), priority=(chunk, 0), source="native"))
        self.assertEqual(hot.bytes, 80)
        self.assertEqual(sorted(hot.entries), [("1", 0), ("2", 0)])
        self.assertFalse(hot.admit(("0", 0), *pair(20), priority=(0, 0), source="decoded"))
        self.assertTrue(hot.admit(("3", 0), *pair(20), priority=(3, 0), source="native"))
        self.assertEqual(sorted(hot.entries), [("2", 0), ("3", 0)])
        self.assertFalse(hot.admit(("9", 0), *pair(60), priority=(9, 0), source="native"))
        self.assertEqual(hot.evictions, 2)
        other = HotShards(100)
        for chunk in (0, 1, 2, 3):
            other.admit((str(chunk), 0), *pair(20), priority=(chunk, 0), source="native")
        self.assertEqual(other.membership(), hot.membership())
        with self.assertRaises(ConfigurationError):
            HotShards(-1)


def _worker(rank, root):
    from types import SimpleNamespace
    from reelcache.adapters.magi import MagiAdapter, MagiGeometry
    from reelcache.codecs import decode_column_views
    from reelcache.distributed import DistributedOwner, PeerFailure
    from reelcache.manager import KVRuntime
    from reelcache.sp_kv import SPTieredHistory
    from reelcache.types import ModelIdentity
    from tests.test_magi_adapter import FakeModel, fake_rotary

    root = Path(root)
    os.environ["TMPDIR"] = str(root)
    torch.set_num_threads(1)
    output = {}
    try:
        dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=60),
                                init_method=f"file://{root / 'rendezvous'}")
        data_group = dist.new_group(backend="gloo")
        geometry = MagiGeometry(layers=3, kv_heads=4, head_dim=4, frames=1, height=2, width=3, world_size=2)
        identity = ModelIdentity("fake_sp_tiered", "known", 3, 16, "global_KV_channels", "bfloat16")
        generator = torch.Generator().manual_seed(11)
        mixed = [torch.randn(6, 1, 16, generator=generator) for _ in range(4)]
        angle = torch.arange(6, dtype=torch.float32)[:, None].expand(6, 2) * 0.07
        cosine, sine = angle.cos(), angle.sin()
        modes = {
            "runtime-svd": ("runtime", CodecSpec("svd", rank=4), None),
            "runtime-raw": ("runtime", CodecSpec(), None),
            "runtime-hevc": ("runtime", CodecSpec("svd_hevc", rank=4), None),
            "sp-cold-svd": ("sp_tiered", CodecSpec("svd", rank=4), {"hot_budget_bytes": 0}),
            "sp-cold-raw": ("sp_tiered", CodecSpec(), {"hot_budget_bytes": 0}),
            "sp-cold-hevc": ("sp_tiered", CodecSpec("svd_hevc", rank=4), {"hot_budget_bytes": 0}),
            "sp-sync-svd": ("sp_tiered", CodecSpec("svd", rank=4), {"hot_budget_bytes": 0, "overlap": False}),
            "sp-owner-svd": ("sp_tiered", CodecSpec("svd", rank=4),
                             {"hot_budget_bytes": 0, "decode_placement": "owner"}),
            "sp-hot-svd": ("sp_tiered", CodecSpec("svd", rank=4), {"hot_budget_bytes": 1 << 20}),
            "sp-promote-svd": ("sp_tiered", CodecSpec("svd", rank=4),
                               {"hot_budget_bytes": 3 * 2 * 6 * 8 * 2, "promote_decoded": True}),
            "sp-sync-hevc": ("sp_tiered", CodecSpec("svd_hevc", rank=4), {"hot_budget_bytes": 0, "overlap": False}),
            "sp-sync-hot": ("sp_tiered", CodecSpec("svd", rank=4), {"hot_budget_bytes": 1 << 20, "overlap": False}),
            "sp-fail-svd": ("sp_tiered", CodecSpec("svd", rank=4), {"hot_budget_bytes": 0}),
        }
        for mode, (cache_mode, spec, options) in modes.items():
            runtime = KVRuntime(root / f"store-{mode}", identity) if rank == 0 else None
            model = FakeModel(geometry)
            cache = SimpleNamespace(max_batch_size=1, max_sequence_length=30,
                                    key_value_memory_dict={}, update_kv_cache=True)
            owner = DistributedOwner(expected_world_size=2, max_wire_bytes=64)
            sp = None
            if options is not None:
                sp = SPTieredHistory(coordinator=owner, data_group=data_group, layout=geometry.layout,
                                     device="cpu", native_dtype="bfloat16", decoder=decode_tensor,
                                     column_decoder=decode_column_views, rotary_apply=fake_rotary, **options)
            adapter = MagiAdapter(model, cache, runtime, model_identity=identity, branch_id="main",
                                  staging_dir=root / f"staging-{mode}", geometry=geometry, coordinator=owner,
                                  specs=spec, rotary_apply=fake_rotary, cache_mode=cache_mode, sp_history=sp)
            adapter.attach()
            clean_native = {}

            def forward(chunk, clean):
                results = []
                meta = SimpleNamespace(
                    slice_point=chunk, clip_token_nums=6, start_chunk_id=chunk, end_chunk_id=chunk + 1,
                    denoising_range_num=1, extract_prefix_video_feature=clean, distill_nearly_clean_chunk=False,
                    timestep=torch.full((1, 1), 0.9999 if clean else 0.0),
                )
                with adapter.execution(chunk_index=chunk, clean=clean, clean_timestep=0.9999 if clean else None):
                    for index, layer in enumerate(model.videodit_blocks.layers):
                        attention = layer.self_attention
                        rows = geometry.layout.rows(rank)
                        source = mixed[chunk] + 0.1 * index
                        k = attention.get_k(source[rows], cosine[rows], sine[rows])
                        v = attention.get_v(source[rows])
                        packed = torch.cat((k, v), -1)
                        pieces = [torch.empty_like(packed) for _ in range(2)]
                        dist.all_gather(pieces, packed)
                        heads = torch.cat(pieces)[:, rank * 2:(rank + 1) * 2].contiguous()
                        if clean:
                            clean_native[(chunk, index)] = heads.clone()
                        results.append(attention.adjust_key_and_value_for_inference(heads, cache, meta))
                return results

            row = {}
            try:
                for chunk in range(3):
                    forward(chunk, False)
                    forward(chunk, True)
                    adapter.seal_chunk()
                    if mode == "sp-fail-svd" and chunk == 2 and rank == 1:
                        def broken(*args, **kwargs):
                            raise RuntimeError("injected rank-1 decode failure")
                        sp.column_decoder = broken
                try:
                    observed = forward(3, False)
                except PeerFailure as exc:
                    row["failure"] = str(exc)
                    observed = None
                if observed is not None:
                    row["history_digest"] = [
                        [float(part[:18].double().sum()), float(part[:18].double().abs().sum())]
                        for pair in observed for part in pair
                    ]
                    row["history_bytes"] = [part[:18].float().flatten().tolist() for pair in observed
                                            for part in pair]
                    row["native_history"] = [
                        torch.cat([clean_native[(c, i)] for c in range(3)])[:, :, :4].float().flatten().tolist()
                        + torch.cat([clean_native[(c, i)] for c in range(3)])[:, :, 4:].float().flatten().tolist()
                        for i in range(3)
                    ]
                if sp is not None:
                    row["sp"] = sp.report()
                if rank == 0:
                    packets = []
                    for chunk_object in runtime.branch_info("main")["chunk_manifests"]:
                        for entry in runtime.store.get_manifest(chunk_object)["matrices"]:
                            packet = runtime.store.get_manifest(entry["tensor"])
                            packet["encoded"].pop("metrics", None)
                            packets.append(json.dumps(packet, sort_keys=True))
                    row["packets"] = packets
            finally:
                if adapter.phase != "failed":
                    adapter.detach()
                else:
                    adapter._unpatch()
                if sp is not None:
                    sp.close()
                if runtime is not None:
                    runtime.close()
            output[mode] = row
        dist.destroy_process_group()
    except BaseException as exc:
        import traceback
        output["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        (root / f"rank{rank}.json").write_text(json.dumps(output))


class SPTieredTwoRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / ".test-artifacts" / f"sp-tiered-{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        try:
            context_ = multiprocessing.get_context("spawn")
            processes = [context_.Process(target=_worker, args=(rank, str(root))) for rank in range(2)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=300)
                if process.is_alive():
                    process.terminate()
                    raise AssertionError("Gloo worker timed out")
            cls.rows = [json.loads((root / f"rank{rank}.json").read_text()) for rank in range(2)]
        finally:
            shutil.rmtree(root, ignore_errors=True)
        for row in cls.rows:
            if "error" in row:
                raise AssertionError(row["error"])

    def test_cold_distributed_decode_matches_owner_restoration_bitwise(self):
        for kind in ("svd", "raw", "hevc"):
            for row in self.rows:
                self.assertEqual(row[f"sp-cold-{kind}"]["history_bytes"], row[f"runtime-{kind}"]["history_bytes"])
            self.assertEqual(self.rows[0][f"sp-cold-{kind}"]["packets"], self.rows[0][f"runtime-{kind}"]["packets"])
        for row in self.rows:
            self.assertEqual(row["sp-owner-svd"]["history_bytes"], row["runtime-svd"]["history_bytes"])

    def test_overlapped_and_synchronous_paths_are_identical(self):
        for row in self.rows:
            self.assertEqual(row["sp-sync-svd"]["history_bytes"], row["sp-cold-svd"]["history_bytes"])
            overlapped, synchronous = row["sp-cold-svd"]["sp"], row["sp-sync-svd"]["sp"]
            self.assertTrue(overlapped["overlap"])
            # Layers 1..2 of every forward are decoded one layer ahead. The last layer of a noisy
            # loop also prefetches layer 0 of the following clean forward (same history), so
            # hits = noisy forwards with 1, 2, 3 history chunks x 2 layers + clean forwards with
            # 1, 2 history chunks x 3 layers = 12 + 9.
            self.assertEqual(overlapped["prefetch_hits"], 2 * (1 + 2 + 3) + 3 * (1 + 2))
            self.assertEqual(overlapped["cold_loads"], 27)
            self.assertEqual(synchronous["prefetch_hits"], 0)
            self.assertEqual(synchronous["cold_loads"], overlapped["cold_loads"])

    def test_cold_chunks_are_decoded_concurrently_on_both_ranks(self):
        for rank, row in enumerate(self.rows):
            stats = row["sp-cold-svd"]["sp"]
            self.assertEqual(stats["hot_hits"], 0)
            self.assertGreater(stats["decoded_here"], 0)
            self.assertTrue(all(count > 0 for count in stats["cold_by_decoder_rank"]))
            self.assertGreater(stats["slice_bytes_received"], 0)
        self.assertGreater(self.rows[1]["sp-cold-svd"]["sp"]["packets_received"], 0)
        self.assertEqual(self.rows[0]["sp-cold-svd"]["sp"]["packets_received"], 0)
        # Sticky placement: a non-owner receives each packet once, then serves later loads from its cache.
        stats = self.rows[1]["sp-cold-svd"]["sp"]
        self.assertGreater(stats["packet_cache_hits"], 0)
        self.assertEqual(stats["packets_received"], 2 * 3)
        owner = [row["sp-owner-svd"]["sp"] for row in self.rows]
        self.assertEqual(owner[1]["decoded_here"], 0)
        self.assertEqual(owner[0]["cold_by_decoder_rank"][1], 0)

    def test_hot_native_shards_serve_later_generations_without_decoding(self):
        for row in self.rows:
            stats = row["sp-hot-svd"]["sp"]
            self.assertEqual(stats["cold_loads"], 0)
            self.assertEqual(stats["decoded_here"], 0)
            self.assertGreater(stats["hot_hits"], 0)
            self.assertEqual(stats["retained_native"], 9)
            self.assertEqual(row["sp-hot-svd"]["history_bytes"], row["runtime-raw"]["history_bytes"])
            self.assertNotEqual(row["sp-hot-svd"]["history_bytes"], row["runtime-svd"]["history_bytes"])
            flat = row["sp-hot-svd"]["history_bytes"]
            for layer in range(3):
                k, v = flat[2 * layer], flat[2 * layer + 1]
                self.assertEqual(k + v, row["sp-hot-svd"]["native_history"][layer])
        self.assertEqual(self.rows[0]["sp-hot-svd"]["packets"], self.rows[0]["runtime-svd"]["packets"])

    def test_decoded_promotion_and_budgeted_eviction(self):
        for row in self.rows:
            stats = row["sp-promote-svd"]["sp"]
            self.assertLessEqual(stats["hot_peak_bytes"], stats["hot_budget_bytes"])
            self.assertGreater(stats["hot_evictions"], 0)
            self.assertGreater(stats["hot_hits"], 0)
            self.assertGreater(stats["cold_loads"], 0)
        self.assertEqual(self.rows[0]["sp-promote-svd"]["sp"]["hot_entries"],
                         self.rows[1]["sp-promote-svd"]["sp"]["hot_entries"])

    def test_overlapped_pipeline_is_bitwise_identical_to_synchronous(self):
        for row in self.rows:
            for kind in ("svd", "hevc"):
                self.assertEqual(row[f"sp-cold-{kind}"]["history_bytes"], row[f"sp-sync-{kind}"]["history_bytes"])
            self.assertEqual(row["sp-hot-svd"]["history_bytes"], row["sp-sync-hot"]["history_bytes"])
            self.assertTrue(row["sp-cold-svd"]["sp"]["overlap"])
            self.assertFalse(row["sp-sync-svd"]["sp"]["overlap"])
            self.assertGreater(row["sp-cold-svd"]["sp"]["prefetched_layers"], 0)
            self.assertGreater(row["sp-cold-svd"]["sp"]["prefetch_hits"], 0)
            self.assertEqual(row["sp-sync-svd"]["sp"]["prefetched_layers"], 0)
        self.assertEqual(self.rows[0]["sp-cold-svd"]["packets"], self.rows[0]["sp-sync-svd"]["packets"])
        self.assertEqual(self.rows[0]["sp-sync-hot"]["packets"], self.rows[0]["runtime-svd"]["packets"])

    def test_one_rank_decode_failure_fails_both_without_hanging(self):
        for row in self.rows:
            self.assertIn("injected rank-1 decode failure", row["sp-fail-svd"]["failure"])


if __name__ == "__main__":
    unittest.main()
