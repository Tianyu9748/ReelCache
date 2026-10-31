"""Layer-streamed MAGI publication: equivalence with serial seal, failure cleanup, and SP=2 over Gloo."""

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

from reelcache.errors import ConfigurationError, UnsupportedError
from reelcache.execution import BoundedExecutor
from reelcache.types import CodecSpec, MatrixKey
from tests.test_magi_adapter import MagiAdapterTests


def portable(runtime, branch):
    rows = []
    for chunk_object in runtime.branch_info(branch)["chunk_manifests"]:
        chunk = runtime.store.get_manifest(chunk_object)
        for entry in chunk["matrices"]:
            packet = runtime.store.get_manifest(entry["tensor"])
            packet["encoded"].pop("metrics", None)
            rows.append((entry["key"], packet))
    return rows


class MagiLayerStreamingTests(unittest.TestCase):
    setUp = MagiAdapterTests.setUp
    tearDown = MagiAdapterTests.tearDown
    _adapter = MagiAdapterTests._adapter
    _forward = MagiAdapterTests._forward
    _state = MagiAdapterTests._state

    def _run(self, runtime, branch, spec, **options):
        self.runtime = runtime
        self.adapter = self._adapter(specs=spec, **options)
        self.adapter.branch_id = branch
        self.adapter.attach()
        try:
            for chunk in range(2):
                self._forward(chunk, clean=False)
                self._forward(chunk, clean=True)
                self.adapter.seal_chunk()
            snapshot = self.adapter.snapshot(**self._state())
            self.adapter.restore(snapshot, f"{branch}-restored")
            output, _ = self._forward(2, clean=False)
            return output, dict(self.adapter.last_stream_stats or {}), self.adapter.canonical_source_hashes
        finally:
            self.adapter.detach()

    def test_streamed_seal_matches_serial_packets_history_and_hashes(self):
        from reelcache.manager import KVRuntime

        with KVRuntime(self.root / "store", self.identity, max_resident_bytes=1 << 22) as runtime, \
                BoundedExecutor(max_workers=2, max_pending=2, max_staging_bytes=1 << 24) as pool:
            for kind in ("raw", "svd", "svd_hevc"):
                with self.subTest(kind=kind):
                    spec = CodecSpec() if kind == "raw" else CodecSpec(kind=kind, rank=3)
                    serial, _, serial_hashes = self._run(runtime, f"{kind}-serial", spec)
                    streamed, stats, hashes = self._run(runtime, f"{kind}-stream", spec, layer_streaming=True)
                    pooled, _, _ = self._run(runtime, f"{kind}-pool", spec, layer_streaming=True,
                                             stream_executor=pool)
                    self.assertEqual(portable(runtime, f"{kind}-serial"), portable(runtime, f"{kind}-stream"))
                    self.assertEqual(portable(runtime, f"{kind}-serial"), portable(runtime, f"{kind}-pool"))
                    self.assertEqual(serial_hashes, hashes)
                    for a, b, c in zip(serial, streamed, pooled):
                        for x, y, z in zip(a, b, c):
                            self.assertTrue(torch.equal(x, y) and torch.equal(x, z))
                    self.assertEqual(stats["outcome"], "complete")
                    self.assertEqual(stats["gathered"], 2 * self.geometry.layers)
                    self.assertEqual(stats["queued_bytes"], 0)
            self.assertEqual(runtime.active_leases, 0)

    def test_failed_clean_forward_aborts_stream_without_publishing(self):
        from reelcache.manager import KVRuntime

        with KVRuntime(self.root / "store", self.identity, max_resident_bytes=1 << 22) as runtime:
            self.runtime = runtime
            self.adapter = self._adapter(specs=CodecSpec(kind="svd", rank=3), layer_streaming=True)
            self.adapter.attach()
            self._forward(0, clean=True)
            self.adapter.seal_chunk()
            head = runtime.store.get_ref("branches/parent")
            objects = runtime.store.stats()["object_count"]
            original = self.model.videodit_blocks.layers[1].self_attention.get_v

            def broken(mixed):
                raise RuntimeError("injected native failure")

            self.adapter._patch(self.model.videodit_blocks.layers[1].self_attention, "get_v", broken)
            with self.assertRaises(Exception):
                self._forward(1, clean=True)
            self.assertEqual(self.adapter.last_stream_stats["outcome"], "aborted")
            self.assertEqual(runtime.store.get_ref("branches/parent"), head)
            self.assertEqual(runtime.store.stats()["object_count"], objects)
            self.assertEqual(runtime.branch_info("parent")["chunk_ids"], ["0"])
            del original

    def test_owner_encode_failure_fails_seal_and_publishes_nothing(self):
        from unittest import mock

        from reelcache.distributed import PeerFailure
        from reelcache.manager import KVRuntime

        with KVRuntime(self.root / "store", self.identity, max_resident_bytes=1 << 22) as runtime:
            self.runtime = runtime
            self.adapter = self._adapter(specs=CodecSpec(kind="svd", rank=3), layer_streaming=True)
            self.adapter.attach()
            self._forward(0, clean=True)
            self.adapter.seal_chunk()
            snapshot = self.adapter.snapshot(**self._state())
            head = runtime.store.get_ref("branches/parent")
            objects = runtime.store.stats()["object_count"]
            encoder = runtime.encoder

            def flaky(tensor, context, spec, **kwargs):
                if context.key == MatrixKey(1, "V"):
                    raise RuntimeError("injected encode failure")
                return encoder(tensor, context, spec, **kwargs)

            with mock.patch.object(runtime, "encoder", side_effect=flaky):
                self._forward(1, clean=True)
                with self.assertRaisesRegex(PeerFailure, "injected encode failure"):
                    self.adapter.seal_chunk()
            self.assertEqual(self.adapter.phase, "failed")
            self.assertEqual(self.adapter.last_stream_stats["outcome"], "failed")
            self.assertEqual(runtime.store.get_ref("branches/parent"), head)
            self.assertEqual(runtime.store.stats()["object_count"], objects)
            self.adapter.restore(snapshot, "recovered")
            self._forward(1, clean=True)
            self.adapter.seal_chunk()
            self.assertEqual(runtime.branch_info("recovered")["chunk_ids"], ["0", "1"])

    def test_streaming_rejects_unsupported_profiles(self):
        with self.assertRaises(UnsupportedError):
            self._adapter(specs=CodecSpec(kind="svd", rank=3), layer_streaming=True, reuse_threshold=0.1)
        for options in ({"stream_queue_depth": 0}, {"stream_thread_init": 3}, {"layer_streaming": 1}):
            with self.assertRaises(ConfigurationError):
                self._adapter(**{"layer_streaming": True, **options})


def _sp2_worker(rank, root):
    from types import SimpleNamespace
    from reelcache.adapters.magi import MagiAdapter, MagiGeometry
    from reelcache.distributed import DistributedOwner
    from reelcache.manager import KVRuntime
    from reelcache.types import ModelIdentity
    from tests.test_magi_adapter import FakeModel, fake_rotary

    root = Path(root)
    os.environ["TMPDIR"] = str(root)
    torch.set_num_threads(1)
    output = {}
    try:
        dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=60),
                                init_method=f"file://{root / 'rendezvous'}")
        stream_group = dist.new_group(backend="gloo")
        geometry = MagiGeometry(layers=3, kv_heads=4, head_dim=4, frames=1, height=2, width=3, world_size=2)
        identity = ModelIdentity("fake_sp2_stream", "known", 3, 16, "global_KV_channels", "bfloat16")
        for mode in ("serial", "stream"):
            full = torch.arange(6 * 16, dtype=torch.float32).reshape(6, 1, 16) / 31
            angle = torch.arange(6, dtype=torch.float32)[:, None].expand(6, 2) * 0.07
            cosine, sine = angle.cos(), angle.sin()
            runtime = KVRuntime(root / f"store-{mode}", identity) if rank == 0 else None
            model = FakeModel(geometry)
            cache = SimpleNamespace(max_batch_size=1, max_sequence_length=18,
                                    key_value_memory_dict={}, update_kv_cache=True)
            owner = DistributedOwner(expected_world_size=2, max_wire_bytes=64)
            adapter = MagiAdapter(
                model, cache, runtime, model_identity=identity, branch_id="main",
                staging_dir=root / f"staging-{mode}", geometry=geometry, coordinator=owner,
                specs=CodecSpec(kind="svd", rank=4), rotary_apply=fake_rotary,
                layer_streaming=mode == "stream", stream_group=stream_group if mode == "stream" else None,
            )
            adapter.attach()

            def forward(chunk, clean):
                results = []
                meta = SimpleNamespace(
                    slice_point=chunk, clip_token_nums=6, start_chunk_id=chunk, end_chunk_id=chunk + 1,
                    denoising_range_num=1, extract_prefix_video_feature=clean, distill_nearly_clean_chunk=False,
                    timestep=torch.full((1, 1), 0.9999 if clean else 0.0),
                )
                with adapter.execution(chunk_index=chunk, clean=clean, clean_timestep=0.9999 if clean else None):
                    for layer in model.videodit_blocks.layers:
                        attention = layer.self_attention
                        rows = geometry.layout.rows(rank)
                        k = attention.get_k(full[rows] + chunk, cosine[rows], sine[rows])
                        v = attention.get_v(full[rows] + chunk)
                        packed = torch.cat((k, v), -1)
                        pieces = [torch.empty_like(packed) for _ in range(2)]
                        dist.all_gather(pieces, packed)
                        heads = torch.cat(pieces)[:, rank * 2:(rank + 1) * 2].contiguous()
                        results.append(attention.adjust_key_and_value_for_inference(heads, cache, meta))
                return results

            forward(0, True)
            adapter.seal_chunk()
            observed = forward(1, False)
            output[f"{mode}_history"] = [[value[:6].float().sum().item() for value in pair] for pair in observed]
            output[f"{mode}_hashes"] = adapter.canonical_source_hashes
            output[f"{mode}_stats"] = adapter.last_stream_stats
            if rank == 0:
                output[f"{mode}_packets"] = [
                    json.dumps(p, sort_keys=True) for _, p in portable(runtime, "main")
                ]
            adapter.detach()
            if runtime is not None:
                runtime.close()
        dist.destroy_process_group()
    except BaseException as exc:
        import traceback
        output["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        (root / f"rank{rank}.json").write_text(json.dumps(output))


class MagiLayerStreamingSP2Tests(unittest.TestCase):
    def test_two_rank_streamed_publication_matches_serial(self):
        root = Path(__file__).resolve().parents[1] / ".test-artifacts" / f"stream-{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        try:
            context = multiprocessing.get_context("spawn")
            processes = [context.Process(target=_sp2_worker, args=(rank, str(root))) for rank in range(2)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=180)
                self.assertFalse(process.is_alive(), "Gloo worker timed out")
                self.assertEqual(process.exitcode, 0)
            rows = [json.loads((root / f"rank{rank}.json").read_text()) for rank in range(2)]
            for row in rows:
                self.assertNotIn("error", row, row.get("error"))
                self.assertEqual(row["serial_history"], row["stream_history"])
                self.assertEqual(row["serial_hashes"], row["stream_hashes"])
                self.assertEqual(row["stream_stats"]["outcome"], "complete")
                self.assertEqual(row["stream_stats"]["gathered"], 6)
            self.assertEqual(rows[0]["serial_packets"], rows[0]["stream_packets"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
