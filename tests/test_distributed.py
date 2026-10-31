"""CPU-only Gloo ownership tests; artifacts stay inside this checkout."""

from datetime import timedelta
from contextlib import nullcontext
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import threading
import unittest
import uuid

import torch
import torch.distributed as dist

from reelcache.distributed import CollectiveCancelled, DistributedOwner, KVLayout, PeerFailure
from reelcache.errors import ConfigurationError, StateError


def _worker(rank, root, fail_peer=False):
    root = Path(root)
    os.environ["TMPDIR"] = str(root)
    torch.set_num_threads(1)
    output = {}
    try:
        dist.init_process_group(
            "gloo", rank=rank, world_size=2, timeout=timedelta(seconds=60),
            init_method=f"file://{root / 'rendezvous'}",
        )
        owner = DistributedOwner(expected_world_size=2, max_wire_bytes=32)
        if fail_peer:
            dist.barrier()
            if rank == 1:
                dist.destroy_process_group()
                output["peer_exited"] = True
            else:
                try:
                    owner.checkpoint("peer_exit")
                except PeerFailure as exc:
                    output["failure_propagated"] = "transport failed" in str(exc)
            return
        layout = KVLayout(tokens=5, kv_heads=4, head_dim=2, world_size=2)
        full = torch.arange(40, dtype=torch.float32).reshape(5, 8)
        gathered_rows = owner.gather_matrix(
            full[layout.rows(rank)].clone(), layout, source_layout="rows",
        )
        gathered_heads = owner.gather_matrix(
            full[:, layout.columns(rank)].contiguous(), layout, source_layout="heads",
        )
        output["owner_only"] = (
            torch.equal(gathered_rows, full) and torch.equal(gathered_heads, full)
            if rank == 0 else gathered_rows is None and gathered_heads is None
        )
        output["gather_wire"] = owner.counters.to_dict()
        requested = []

        def load(columns):
            requested.append((columns.start, columns.stop))
            return full[:, columns].contiguous()

        local = owner.local_columns(layout, load, dtype=torch.float32)
        output["local_heads_exact"] = torch.equal(local, full[:, layout.columns(rank)])
        output["requested_columns"] = requested
        output["local_shape"] = list(local.shape)
        output["scatter_wire"] = owner.counters.to_dict()
        sent = torch.tensor([11, 29, 71], dtype=torch.int64) if rank == 1 else None
        state = owner.transfer_tensor(sent, source=1, destination=0)
        output["state_exact"] = torch.equal(state, torch.tensor([11, 29, 71])) if rank == 0 else state is None
        for name, operation, exception in (
            ("owner_failure",
             lambda: owner.owner_call("encode", lambda: (_ for _ in ()).throw(ValueError("bad packet"))),
             PeerFailure),
            ("cancel",
             lambda: owner.checkpoint("cancel", cancelled=rank == 1), CollectiveCancelled),
            ("order",
             lambda: owner.checkpoint("wrong" if rank else "expected"), PeerFailure),
            ("invalid_control",
             lambda: owner.checkpoint("nan", details=float("nan") if rank else None), PeerFailure),
            ("invalid_shape",
             lambda: owner.gather_matrix(
                 torch.zeros((1, 1)) if rank else full[layout.rows(rank)],
                 layout, source_layout="rows",
             ), PeerFailure),
        ):
            try:
                operation()
            except exception:
                output[name] = True
            else:
                output[name] = False
        owner.checkpoint("still_usable")
        errors = []

        def foreign_thread():
            try:
                owner.checkpoint("unordered_worker")
            except StateError:
                errors.append(True)

        worker = threading.Thread(target=foreign_thread)
        worker.start()
        worker.join()
        output["thread_guard"] = errors == [True]
        output["cuda_initialized"] = torch.cuda.is_initialized()
        dist.destroy_process_group()
    except BaseException as exc:
        import traceback

        output["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        (root / f"rank{rank}.json").write_text(json.dumps(output))


def _adapter_worker(rank, root, hot=False):
    from types import SimpleNamespace
    from reelcache.adapters.magi import MagiAdapter, MagiGeometry
    from reelcache.types import CodecSpec, MatrixKey, ModelIdentity
    from tests.test_magi_adapter import FakeModel, FakeRuntime, fake_rotary

    root = Path(root)
    os.environ["TMPDIR"] = str(root)
    torch.set_num_threads(1)
    output = {}
    try:
        dist.init_process_group(
            "gloo", rank=rank, world_size=2, timeout=timedelta(seconds=60),
            init_method=f"file://{root / 'rendezvous'}",
        )
        geometry = MagiGeometry(layers=2, kv_heads=4, head_dim=4, frames=1, height=2, width=3, world_size=2)
        full = torch.arange(6 * 16, dtype=torch.float32).reshape(6, 1, 16) / 31
        angle = torch.arange(6, dtype=torch.float32)[:, None].expand(6, 2) * 0.07
        cosine, sine = angle.cos(), angle.sin()
        expected_key = fake_rotary(full.reshape(1, 6, 4, 4), cosine, sine).to(torch.bfloat16)[0]
        expected_value = (full * 0.7 + 0.25).to(torch.bfloat16).reshape(6, 4, 4)
        state = {
            "metadata": {"conditioning_identity": "caption",
                         "execution_config": {"cfg": 1, "steps": 16},
                         "vae_contract": {"mode": "stateless_chunk_decode"}},
            "tensors": {"latent": full.clone(), "conditioning": torch.arange(8),
                        "conditioning_mask": torch.ones(8)},
        }
        for kind in ("raw", "svd"):
            identity = ModelIdentity(
                "fake_sp2", "known_source", 2, 16, "global_KV_channels",
                "bfloat16",
            )
            runtime = FakeRuntime(identity) if rank == 0 else None
            model = FakeModel(geometry)
            cache = SimpleNamespace(max_batch_size=1, max_sequence_length=18,
                                    key_value_memory_dict={}, update_kv_cache=True)
            owner = DistributedOwner(expected_world_size=2, max_wire_bytes=64)
            spec = CodecSpec() if kind == "raw" else CodecSpec(kind="svd", rank=4)
            adapter = MagiAdapter(
                model, cache, runtime, model_identity=identity, branch_id=kind,
                staging_dir=root / "staging", geometry=geometry, coordinator=owner,
                specs=spec, rotary_apply=fake_rotary,
                hot_window_bytes=384 if hot else 0, hot_window_require_full=hot,
                hot_window_verify_hits=1 if hot else 0,
            )
            adapter.attach()

            def forward(chunk, clean, guarded=True):
                results = []
                meta = SimpleNamespace(
                    slice_point=chunk, clip_token_nums=6, start_chunk_id=chunk,
                    end_chunk_id=chunk + 1, denoising_range_num=1,
                    extract_prefix_video_feature=clean, distill_nearly_clean_chunk=False,
                    timestep=torch.full((1, 1), 0.9999 if clean else 0.0),
                )
                with (adapter.execution(chunk_index=chunk, clean=clean,
                                        clean_timestep=0.9999 if clean else None) if guarded else nullcontext()):
                    for layer in model.videodit_blocks.layers:
                        attention = layer.self_attention
                        rows = geometry.layout.rows(rank)
                        k = attention.get_k(full[rows], cosine[rows], sine[rows])
                        v = attention.get_v(full[rows])
                        packed = torch.cat((k, v), -1)
                        pieces = [torch.empty_like(packed) for _ in range(2)]
                        dist.all_gather(pieces, packed)
                        heads = torch.cat(pieces)[:, rank * 2:(rank + 1) * 2].contiguous()
                        results.append(attention.adjust_key_and_value_for_inference(heads, cache, meta))
                return results

            forward(0, True)
            adapter.seal_chunk()
            torch.manual_seed(rank + 100)
            snapshot = adapter.snapshot(**state)
            expected_rng = torch.rand(4)
            original_specs = adapter.specs
            if rank == 1:
                adapter.specs = {key: CodecSpec(**{**value.to_dict(), "qp": value.qp + 1})
                                 for key, value in original_specs.items()}
            before_rng = torch.get_rng_state().clone()
            before_native = (adapter.branch_id, adapter.phase, adapter.cursor)
            try:
                adapter.fork(snapshot, f"{kind}-incompatible-ranks")
            except StateError:
                output[f"{kind}_profile_disagreement_rejected"] = True
            else:
                output[f"{kind}_profile_disagreement_rejected"] = False
            output[f"{kind}_rejection_rng_unchanged"] = torch.equal(before_rng, torch.get_rng_state())
            output[f"{kind}_rejection_native_unchanged"] = (
                before_native == (adapter.branch_id, adapter.phase, adapter.cursor)
            )
            if rank == 0:
                output[f"{kind}_rejection_branch_absent"] = f"{kind}-incompatible-ranks" not in runtime.branches
            adapter.specs = original_specs
            _, restored = adapter.fork(snapshot, f"{kind}-fork")
            output[f"{kind}_rng_exact"] = torch.equal(torch.rand(4), expected_rng)
            if hot:
                before_wire = owner.counters.payload_sent_bytes
                with adapter.execution(chunk_index=1, clean=False):
                    observed = forward(1, False, guarded=False)
                    second = forward(1, False, guarded=False)
                output[f"{kind}_hot_equal"] = all(
                    torch.equal(first, other) for a, b in zip(observed, second) for first, other in zip(a, b)
                )
                output[f"{kind}_hot_stats"] = adapter.memory_report()["decoded_hot_window"]
                output[f"{kind}_hot_wire"] = owner.counters.payload_sent_bytes - before_wire
                output[f"{kind}_hot_pin_released"] = runtime.pins == 0 if rank == 0 else True
            else:
                observed = forward(1, False)
            output[f"{kind}_K_exact"] = all(
                torch.equal(k[:6], expected_key[:, rank * 2:(rank + 1) * 2]) for k, _ in observed
            )
            output[f"{kind}_V_exact"] = all(
                torch.equal(v[:6], expected_value[:, rank * 2:(rank + 1) * 2]) for _, v in observed
            )
            output[f"{kind}_head_shape"] = list(observed[0][0].shape)
            output[f"{kind}_no_native_buffers"] = not cache.key_value_memory_dict
            if rank == 0:
                output[f"{kind}_full_domain"] = all(
                    tuple(value.shape) == (6, 16)
                    for value in runtime.branches[kind][0]["values"].values()
                )
                output[f"{kind}_local_column_requests"] = all(
                    stop - start == 8 for _, start, stop in runtime.requested_columns
                )
                output[f"{kind}_K_space"] = runtime.branches[kind][0]["contexts"][MatrixKey(0, "K")].key_space
            if hot:
                adapter.fork(snapshot, f"{kind}-hot-failure")
                try:
                    with adapter.execution(chunk_index=1, clean=False):
                        forward(1, False, guarded=False)
                        if rank == 1:
                            adapter._hot.clear()
                        forward(1, False, guarded=False)
                except PeerFailure as exc:
                    output[f"{kind}_membership_failure_propagated"] = "membership" in str(exc)
                    output[f"{kind}_membership_error"] = str(exc)
                output[f"{kind}_failure_released"] = (
                    adapter._hot_bytes == 0 and not adapter._hot and not cache.key_value_memory_dict
                    and (runtime.pins == 0 if rank == 0 else True)
                )
            adapter.detach()
        output["cuda_initialized"] = torch.cuda.is_initialized()
        dist.destroy_process_group()
    except BaseException as exc:
        import traceback
        output["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        (root / f"rank{rank}.json").write_text(json.dumps(output))


class DistributedTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / ".test-artifacts" / f"gloo-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root)

    def _run(self, *, fail_peer=False, adapter=False, hot=False):
        context = multiprocessing.get_context("spawn")
        processes = [
            context.Process(target=_adapter_worker if adapter else _worker,
                            args=(rank, str(self.root), hot) if adapter else (rank, str(self.root), fail_peer))
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        try:
            for process in processes:
                process.join(timeout=180)
                self.assertFalse(process.is_alive(), "Gloo worker timed out")
                self.assertEqual(process.exitcode, 0)
            rows = [json.loads((self.root / f"rank{rank}.json").read_text()) for rank in range(2)]
            for row in rows:
                self.assertNotIn("error", row, row.get("error"))
            return rows
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)

    def test_row_head_layouts_wire_accounting_and_failures(self):
        rows = self._run()
        for row in rows:
            for name in ("owner_only", "local_heads_exact", "state_exact", "owner_failure",
                         "cancel", "order", "invalid_control", "invalid_shape", "thread_guard"):
                self.assertTrue(row[name], name)
            self.assertEqual(row["local_shape"], [5, 4])
            self.assertFalse(row["cuda_initialized"])
        self.assertEqual(rows[0]["requested_columns"], [[0, 4], [4, 8]])
        self.assertEqual(rows[1]["requested_columns"], [])
        self.assertEqual(rows[0]["gather_wire"]["payload_received_bytes"], (2 * 8 + 5 * 4) * 4)
        self.assertEqual(rows[1]["gather_wire"]["payload_sent_bytes"], (2 * 8 + 5 * 4) * 4)
        self.assertEqual(rows[0]["gather_wire"]["payload_sent_bytes"], 0)
        self.assertEqual(rows[0]["scatter_wire"]["payload_sent_bytes"], 5 * 4 * 4)
        self.assertEqual(rows[1]["scatter_wire"]["payload_received_bytes"], 5 * 4 * 4)
        self.assertGreater(rows[0]["scatter_wire"]["control_sent_bytes"], 0)

    def test_peer_exit_is_not_a_success_shaped_result(self):
        rows = self._run(fail_peer=True)
        self.assertTrue(rows[0]["failure_propagated"])
        self.assertTrue(rows[1]["peer_exited"])

    def test_two_rank_adapter_raw_and_pre_rope_local_head_restoration(self):
        rows = self._run(adapter=True)
        for row in rows:
            for kind in ("raw", "svd"):
                for check in ("rng_exact", "K_exact", "V_exact", "no_native_buffers"):
                    self.assertTrue(row[f"{kind}_{check}"], f"{kind}_{check}: {row}")
                for check in ("profile_disagreement_rejected", "rejection_rng_unchanged",
                              "rejection_native_unchanged"):
                    self.assertTrue(row[f"{kind}_{check}"], f"{kind}_{check}")
                self.assertEqual(row[f"{kind}_head_shape"], [12, 2, 4])
            self.assertFalse(row["cuda_initialized"])
        for kind in ("raw", "svd"):
            self.assertTrue(rows[0][f"{kind}_rejection_branch_absent"])
            self.assertTrue(rows[0][f"{kind}_full_domain"])
            self.assertTrue(rows[0][f"{kind}_local_column_requests"])
        self.assertEqual(rows[0]["raw_K_space"], "post_rope")
        self.assertEqual(rows[0]["svd_K_space"], "pre_rope")

    def test_hot_window_sp2_exact_hits_wire_bytes_and_membership_failure(self):
        rows = self._run(adapter=True, hot=True)
        for row in rows:
            for kind in ("raw", "svd"):
                for check in ("hot_equal", "hot_pin_released", "membership_failure_propagated",
                              "failure_released", "K_exact", "V_exact"):
                    self.assertTrue(row[f"{kind}_{check}"], f"{kind}_{check}: {row}")
                stats = row[f"{kind}_hot_stats"]
                self.assertEqual(stats["hits"], 4)
                self.assertEqual(stats["misses"], 4)
                self.assertEqual(stats["verified_hits"], 1)
                self.assertEqual(stats["peak_bytes"], 384)
                self.assertEqual(stats["resident_bytes"], 0)
                self.assertEqual(row[f"{kind}_head_shape"], [12, 2, 4])
            self.assertFalse(row["cuda_initialized"])
        self.assertEqual(rows[0]["raw_hot_wire"], 480)
        self.assertEqual(rows[0]["svd_hot_wire"], 768)
        self.assertEqual(rows[1]["raw_hot_wire"], 0)
        self.assertEqual(rows[1]["svd_hot_wire"], 0)

    def test_layout_rejects_query_expansion_and_invalid_heads(self):
        layout = KVLayout(5, 8, 128, 2)
        self.assertEqual(layout.channels, 1024)
        self.assertEqual(layout.local_heads, 4)
        self.assertEqual(layout.shape(1, "heads"), (5, 512))
        with self.assertRaises(ConfigurationError):
            layout.shape(0, "query_heads")
        with self.assertRaises(ConfigurationError):
            KVLayout(5, 7, 128, 2)
        with self.assertRaises(ConfigurationError):
            layout.columns(2)


if __name__ == "__main__":
    unittest.main()
