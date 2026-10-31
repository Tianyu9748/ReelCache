import gc
from collections.abc import Mapping
import os
from pathlib import Path
import subprocess
import sys
import threading
from unittest.mock import Mock, patch
import weakref

import torch

from reelcache.errors import (
    CapacityError, ConfigurationError, IntegrityError, PendingError, StateError,
)
from reelcache.manager import KVRuntime, RestoredState, SnapshotState
from reelcache.serialization import tensor_bytes, tensor_from_bytes
from reelcache.types import CodecSpec, EncodedTensor, MatrixKey, TensorContext
from tests.core_support import (
    CoreWorkspaceTest, chunk_inputs, encoded_metadata, metadata, model, raw_decoder, raw_encoder,
)


class ManagerTests(CoreWorkspaceTest):
    def open_runtime(self, identity=None, **kwargs):
        identity = model() if identity is None else identity
        kwargs.setdefault("encoder", raw_encoder)
        kwargs.setdefault("decoder", raw_decoder)
        runtime = KVRuntime(self.workspace / "runtime", identity, **kwargs)
        self.addCleanup(runtime.close)
        return runtime

    def seed(self, runtime, *, branch="main", offset=0, key_space="pre_rope"):
        runtime.create_branch(branch)
        matrices, contexts = chunk_inputs(runtime.model, offset=offset, key_space=key_space)
        runtime.seal_chunk(branch, "c0", matrices, contexts)
        return matrices, contexts

    def state_tensors(self):
        return {
            "rng": torch.Generator().manual_seed(7).get_state(),
            "latent": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
            "vae": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "cursor": torch.tensor(4, dtype=torch.int64),
        }

    def test_complete_model_raw_history_and_partial_channel_leases(self):
        runtime = self.open_runtime()
        first, _ = self.seed(runtime)
        second, contexts = chunk_inputs(runtime.model, start=4, offset=1000)
        runtime.seal_chunk("main", "c1", second, contexts)
        for key in first:
            expected = torch.cat([first[key], second[key]])
            with runtime.acquire_history("main", key) as lease:
                self.assertTrue(torch.equal(lease.tensor, expected))
                self.assertEqual(runtime.resident_bytes, expected.numel() * expected.element_size())
            with runtime.acquire_history(
                "main", key, token_start=2, token_end=6, columns=slice(1, 3), dtype=torch.float64,
            ) as lease:
                self.assertTrue(torch.equal(lease.tensor, expected[2:6, 1:3].double()))
        self.assertEqual(runtime.resident_bytes, 0)
        self.assertEqual(runtime.active_leases, 0)
        self.assertEqual(runtime.branch_info("main")["token_end"], 8)

    def test_bfloat16_raw_bits_are_preserved(self):
        identity = model(dtype="bfloat16")
        runtime = self.open_runtime(identity)
        runtime.create_branch("main")
        matrices, contexts = chunk_inputs(identity)
        words = torch.tensor([0x0000, 0x8000, 0x7FC1, 0x7F81] * 4, dtype=torch.uint16)
        key = MatrixKey(0, "K")
        matrices[key] = words.view(torch.bfloat16).reshape(4, 4)
        runtime.seal_chunk("main", "c0", matrices, contexts)
        with runtime.acquire_history("main", key) as lease:
            self.assertEqual(lease.tensor.dtype, torch.bfloat16)
            self.assertEqual(tensor_bytes(lease.tensor), tensor_bytes(matrices[key]))

    def test_asymmetric_canonical_kv_domains_are_not_expanded_to_query_heads(self):
        runtime = self.open_runtime(model(layers=1, channels={"K": 4, "V": 6}))
        values, _ = self.seed(runtime)
        with runtime.acquire_history("main", MatrixKey(0, "K")) as lease:
            self.assertEqual(lease.tensor.shape, (4, 4))
        with runtime.acquire_history("main", MatrixKey(0, "V"), columns=slice(3, 6)) as lease:
            self.assertEqual(lease.tensor.shape, (4, 3))
            self.assertTrue(torch.equal(lease.tensor, values[MatrixKey(0, "V")][:, 3:6]))

    def test_mixed_native_k_float32_v_bfloat16_preserves_raw_dtype_and_bits(self):
        identity = model(layers=1, dtype={"K": "float32", "V": "bfloat16"})
        runtime = self.open_runtime(identity)
        runtime.create_branch("main")
        values, contexts = chunk_inputs(identity)
        values[MatrixKey(0, "K")] = torch.tensor(
            [0x7FC12345, 0x80000000, 0x3F800000, 0xFF800000] * 4, dtype=torch.uint32
        ).view(torch.float32).reshape(4, 4)
        values[MatrixKey(0, "V")] = torch.tensor(
            [0x7FC1, 0x8000, 0x3F80, 0xFF80] * 4, dtype=torch.uint16
        ).view(torch.bfloat16).reshape(4, 4)
        runtime.seal_chunk("main", "c0", values, contexts)
        runtime.close()
        restarted = self.open_runtime(identity)
        for key, expected in values.items():
            with restarted.acquire_history("main", key) as lease:
                self.assertEqual(lease.tensor.dtype, expected.dtype)
                self.assertEqual(tensor_bytes(lease.tensor), tensor_bytes(expected))
        with self.assertRaises(ConfigurationError):
            self.open_runtime(model(layers=1, dtype="bfloat16"))

    def test_mixed_raw_post_rope_and_pre_rope_layers_preserve_declared_dtype(self):
        identity = model(dtype={"0:K": "bfloat16", "0:V": "bfloat16",
                                "1:K": "float32", "1:V": "bfloat16"})
        runtime = self.open_runtime(identity)
        runtime.create_branch("main")
        values, contexts = chunk_inputs(identity)
        key = MatrixKey(0, "K")
        contexts[key] = TensorContext(identity, key, 0, 1, 2, 2, "post_rope")
        runtime.seal_chunk("main", "c0", values, contexts)
        for key, expected in values.items():
            with runtime.acquire_history("main", key) as lease:
                self.assertEqual(lease.tensor.dtype, expected.dtype)
                self.assertTrue(torch.equal(lease.tensor, expected))

    def test_missing_or_extra_layers_rejected_before_encoding(self):
        encode = Mock(side_effect=raw_encoder)
        runtime = self.open_runtime(encoder=encode)
        runtime.create_branch("main")
        matrices, contexts = chunk_inputs(runtime.model)
        del matrices[MatrixKey(1, "V")]
        with self.assertRaises(ConfigurationError):
            runtime.seal_chunk("main", "bad", matrices, contexts)
        encode.assert_not_called()
        self.assertEqual(runtime.branch_info("main")["token_end"], 0)
        self.assertEqual(runtime.store.stats()["blob_bytes"], 0)

    def test_lazy_matrix_mapping_streams_once_without_retaining_model_packets(self):
        tensor_references, packet_references, loaded_keys = [], [], []

        def encode(*args, **kwargs):
            packet = raw_encoder(*args, **kwargs)
            packet_references.append(weakref.ref(packet))
            return packet

        runtime = self.open_runtime(encoder=encode)
        runtime.create_branch("main")
        observer = self.open_runtime()
        _, contexts = chunk_inputs(runtime.model)
        del _
        keys = tuple(contexts)
        outer = self

        class LazyMatrices(Mapping):
            def __len__(self):
                return len(keys)

            def __iter__(self):
                return iter(keys)

            def __getitem__(self, key):
                gc.collect()
                outer.assertTrue(all(reference() is None for reference in tensor_references))
                outer.assertTrue(all(reference() is None for reference in packet_references))
                outer.assertEqual(observer.branch_info("main")["token_end"], 0)
                observer.gc()
                loaded_keys.append(key)
                tensor = torch.full(contexts[key].shape, float(keys.index(key)))
                tensor_references.append(weakref.ref(tensor))
                return tensor

        runtime.seal_chunk("main", "c0", LazyMatrices(), contexts)
        self.assertEqual(loaded_keys, list(keys))
        self.assertTrue(all(reference() is None for reference in tensor_references))
        self.assertTrue(all(reference() is None for reference in packet_references))
        self.assertEqual(runtime.branch_info("main")["token_end"], 4)
        self.assertEqual(runtime.store.stats()["leases"], 0)
        for index, key in enumerate(keys):
            with runtime.acquire_history("main", key) as lease:
                self.assertTrue(torch.equal(lease.tensor, torch.full(contexts[key].shape, float(index))))

    def test_identity_shape_dtype_context_and_spec_validation(self):
        runtime = self.open_runtime()
        runtime.create_branch("main")
        key = MatrixKey(0, "K")
        for kind in ("identity", "shape", "dtype", "geometry", "specs", "wrong-key"):
            matrices, contexts = chunk_inputs(runtime.model)
            specs = CodecSpec()
            if kind == "identity":
                contexts[key] = TensorContext(model(fingerprint="other"), key, 0, 1, 2, 2, "pre_rope")
            elif kind == "shape":
                matrices[key] = torch.zeros(2, 8)
            elif kind == "dtype":
                matrices[key] = matrices[key].double()
            elif kind == "geometry":
                contexts[key] = TensorContext(runtime.model, key, 0, 1, 1, 4, "pre_rope")
            elif kind == "specs":
                specs = {key: CodecSpec()}
            else:
                contexts[key] = contexts[MatrixKey(1, "K")]
            with self.subTest(kind=kind), self.assertRaises(ConfigurationError):
                runtime.seal_chunk("main", kind, matrices, contexts, specs)
        self.assertEqual(runtime.branch_info("main")["chunk_ids"], [])

    def test_positions_have_no_gaps_or_overlaps(self):
        runtime = self.open_runtime()
        self.seed(runtime)
        for start in (0, 3, 5, 8):
            matrices, contexts = chunk_inputs(runtime.model, start=start)
            with self.subTest(start=start), self.assertRaises(ConfigurationError):
                runtime.seal_chunk("main", f"invalid-{start}", matrices, contexts)
        self.assertEqual(runtime.branch_info("main")["token_end"], 4)

    def test_spatial_geometry_and_key_space_remain_stable(self):
        runtime = self.open_runtime()
        self.seed(runtime)
        for kwargs in ({"height": 1, "width": 4}, {"key_space": "post_rope"}):
            matrices, contexts = chunk_inputs(runtime.model, start=4, **kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigurationError):
                runtime.seal_chunk("main", "invalid", matrices, contexts)

    def test_post_rope_k_allows_raw_but_not_svd(self):
        runtime = self.open_runtime()
        self.seed(runtime, key_space="post_rope")
        matrices, contexts = chunk_inputs(runtime.model, start=4, key_space="post_rope")
        with self.assertRaisesRegex(ConfigurationError, "post-RoPE"):
            runtime.seal_chunk("main", "bad", matrices, contexts, CodecSpec("svd", 2))

    def test_raw_encoding_never_receives_a_projection_reuse_request(self):
        encoder = Mock(side_effect=raw_encoder)
        runtime = self.open_runtime(encoder=encoder)
        runtime.create_branch("main")
        values, contexts = chunk_inputs(runtime.model, key_space="post_rope")
        runtime.seal_chunk("main", "c0", values, contexts, reuse_threshold=.1)
        for call in encoder.call_args_list:
            self.assertIsNone(call.kwargs["reuse_basis"])
            self.assertIsNone(call.kwargs["reuse_threshold"])

    def test_mixed_layer_key_spaces_are_explicit_and_described_without_tensors(self):
        runtime = self.open_runtime()
        runtime.create_branch("main")
        values, contexts = chunk_inputs(runtime.model)
        key = MatrixKey(0, "K")
        contexts[key] = TensorContext(runtime.model, key, 0, 1, 2, 2, "post_rope")
        runtime.seal_chunk("main", "c0", values, contexts)
        self.assertEqual(runtime.history_contexts("main", key), (contexts[key],))
        self.assertEqual(runtime.history_specs("main", key), (CodecSpec(),))
        self.assertEqual(runtime.history_contexts("main", MatrixKey(1, "K"))[0].key_space, "pre_rope")
        self.assertEqual(runtime.resident_bytes, 0)
        with self.assertRaises(ConfigurationError):
            runtime.history_contexts("main", MatrixKey(2, "K"))
        runtime.close()
        reopened = self.open_runtime()
        self.assertEqual(reopened.history_contexts("main", key), (contexts[key],))

    def test_nonclean_duplicate_and_pending_states_are_rejected(self):
        runtime = self.open_runtime()
        first, contexts = self.seed(runtime)
        with self.assertRaises(PendingError):
            runtime.seal_chunk("main", "dirty", first, contexts, clean=False)
        with self.assertRaises(StateError):
            runtime.seal_chunk("main", "c0", first, contexts)
        with self.assertRaises(PendingError):
            runtime.snapshot("main", metadata={**metadata(), "pending": True})
        self.assertEqual(runtime.branch_info("main")["chunk_ids"], ["c0"])

    def test_snapshot_contract_is_verified_not_silently_defaulted(self):
        runtime = self.open_runtime()
        self.seed(runtime)
        invalid = (
            {}, {**metadata(), "clean_boundary": {"token_end": 5, "chunk_id": "c0"}},
            {**metadata(), "clean_boundary": {"token_end": 4, "chunk_id": "other"}},
            {**metadata(), "rng": None},
            {**metadata(), "conditioning": object()},
            {**metadata(), "clean_boundary": {"token_end": 4, "chunk_id": "c0", "clean": False}},
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                runtime.snapshot("main", metadata=value)
        self.assertEqual(runtime.store.list_refs("snapshots/"), {})

    def test_snapshot_restore_fork_copy_on_write_and_tensor_state(self):
        runtime = self.open_runtime()
        original, _ = self.seed(runtime)
        state = self.state_tensors()
        snapshot = runtime.snapshot("main", metadata=metadata(), tensors=state)
        before = runtime.store.stats()["object_bytes"]
        result = runtime.fork(snapshot, "child")
        self.assertIsInstance(result, RestoredState)
        self.assertEqual(runtime.store.stats()["object_bytes"], before)
        self.assertEqual(result.metadata, metadata())
        for name in state:
            self.assertEqual(tensor_bytes(result.tensors[name]), tensor_bytes(state[name]))
        result.tensors["latent"].zero_()
        result.metadata["conditioning"]["identity"] = "changed"
        child_values, child_contexts = chunk_inputs(runtime.model, start=4, offset=999)
        runtime.seal_chunk("child", "c1", child_values, child_contexts)
        self.assertEqual(runtime.branch_info("main")["token_end"], 4)
        self.assertEqual(runtime.branch_info("child")["token_end"], 8)
        for key, expected in original.items():
            with runtime.acquire_history("main", key) as lease:
                self.assertTrue(torch.equal(lease.tensor, expected))
        restored = runtime.restore(snapshot, "restored")
        self.assertEqual(restored.metadata, metadata())
        self.assertTrue(torch.equal(restored.tensors["latent"], state["latent"]))
        with self.assertRaises(StateError):
            runtime.fork(snapshot, "main")

    def test_restore_can_atomically_rewind_existing_branch(self):
        runtime = self.open_runtime()
        self.seed(runtime)
        snapshot = runtime.snapshot("main", metadata=metadata())
        values, contexts = chunk_inputs(runtime.model, start=4, offset=100)
        runtime.seal_chunk("main", "c1", values, contexts)
        runtime.restore(snapshot, "main")
        self.assertEqual(runtime.branch_info("main")["token_end"], 4)

    def test_load_snapshot_is_verified_read_only_and_retains_no_native_tensors(self):
        runtime = self.open_runtime()
        self.seed(runtime)
        state = self.state_tensors()
        snapshot = runtime.snapshot("main", metadata=metadata(), tensors=state)
        following, contexts = chunk_inputs(runtime.model, start=4)
        runtime.seal_chunk("main", "c1", following, contexts)
        before = runtime.branch_info("main")["head_id"]
        loaded = runtime.load_snapshot(snapshot)
        self.assertIsInstance(loaded, SnapshotState)
        self.assertEqual(loaded.snapshot_id, snapshot)
        self.assertEqual(loaded.metadata, metadata())
        self.assertEqual(runtime.branch_info("main")["head_id"], before)
        reference = weakref.ref(loaded.tensors["latent"])
        self.assertEqual(tensor_bytes(loaded.tensors["latent"]), tensor_bytes(state["latent"]))
        del loaded
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(runtime.resident_bytes, 0)

    def test_native_validation_precedes_fork_or_restore_publication(self):
        runtime = self.open_runtime()
        self.seed(runtime)
        snapshot = runtime.snapshot("main", metadata=metadata(), tensors=self.state_tensors())
        following, contexts = chunk_inputs(runtime.model, start=4)
        runtime.seal_chunk("main", "c1", following, contexts)
        before = runtime.branch_info("main")["head_id"]

        def incompatible(state):
            self.assertIsInstance(state, RestoredState)
            self.assertEqual(state.tensors["latent"].shape, (3, 4))
            raise ConfigurationError("conditioning or native execution differs")

        with self.assertRaises(ConfigurationError):
            runtime.fork(snapshot, "rejected", validator=incompatible)
        self.assertIsNone(runtime.store.get_ref("branches/rejected"))
        with self.assertRaises(ConfigurationError):
            runtime.restore(snapshot, "main", validator=incompatible)
        self.assertEqual(runtime.branch_info("main")["head_id"], before)
        with self.assertRaises(ConfigurationError):
            runtime.fork(snapshot, "false", validator=lambda state: False)
        self.assertIsNone(runtime.store.get_ref("branches/false"))

        def validate(state):
            self.assertIsNone(runtime.store.get_ref("branches/accepted"))
            self.assertEqual(state.branch_id, "accepted")
            self.assertEqual(state.metadata["conditioning"], metadata()["conditioning"])

        result = runtime.fork(snapshot, "accepted", validator=validate)
        self.assertEqual(result.branch_id, "accepted")
        self.assertEqual(runtime.branch_info("accepted")["token_end"], 4)
        with self.assertRaises(StateError):
            runtime.fork(snapshot, "broken", validator=Mock(side_effect=RuntimeError("bad native state")))
        self.assertIsNone(runtime.store.get_ref("branches/broken"))

    def test_restart_recovers_committed_raw_history_and_state(self):
        runtime = self.open_runtime()
        values, _ = self.seed(runtime)
        state = self.state_tensors()
        snapshot = runtime.snapshot("main", metadata=metadata(), tensors=state)
        runtime.close()
        restarted = self.open_runtime()
        result = restarted.restore(snapshot, "after-restart")
        for key in values:
            with restarted.acquire_history("after-restart", key) as lease:
                self.assertEqual(tensor_bytes(lease.tensor), tensor_bytes(values[key]))
        for name in state:
            self.assertEqual(tensor_bytes(result.tensors[name]), tensor_bytes(state[name]))

    def test_model_mismatch_rejected_on_reopen(self):
        runtime = self.open_runtime()
        self.seed(runtime)
        runtime.close()
        for identity in (
            model(fingerprint="other"), model(layers=1), model(channels=8),
            model(dtype="bfloat16"),
        ):
            with self.subTest(identity=identity), self.assertRaises(ConfigurationError):
                self.open_runtime(identity)

    def test_corrupt_or_missing_state_does_not_mutate_target_branch(self):
        runtime = self.open_runtime()
        self.seed(runtime)
        snapshot = runtime.snapshot("main", metadata=metadata(), tensors=self.state_tensors())
        old_head = runtime.branch_info("main")["head_id"]
        record = runtime.store.get_manifest(snapshot)
        state_path = runtime.store.path_for(record["tensor_state"])
        state_path.write_bytes(b"corrupt")
        with self.assertRaises(IntegrityError):
            runtime.restore(snapshot, "main")
        self.assertEqual(runtime.branch_info("main")["head_id"], old_head)
        with self.assertRaises(IntegrityError):
            runtime.fork(snapshot, "new")
        self.assertNotIn("branches/new", runtime.store.list_refs())
        state_path.unlink()
        with self.assertRaises(IntegrityError):
            runtime.restore(snapshot, "main")

    def test_missing_history_packet_is_not_filled_with_zeros(self):
        runtime = self.open_runtime()
        self.seed(runtime)
        head = runtime.store.get_manifest(runtime.branch_info("main")["head_id"])
        chunk = runtime.store.get_manifest(head["chunks"][0])
        tensor = runtime.store.get_manifest(chunk["matrices"][0]["tensor"])
        payload = next(iter(tensor["encoded"]["payloads"].values()))
        runtime.store.path_for(payload).unlink()
        with self.assertRaises(IntegrityError):
            runtime.acquire_history("main", MatrixKey(0, "K"))
        self.assertEqual(runtime.resident_bytes, 0)
        self.assertEqual(runtime.store.stats()["leases"], 0)

    def test_encode_failure_keeps_prior_committed_chunk_and_all_layers(self):
        calls = 0
        enabled = False

        def encode(*args, **kwargs):
            nonlocal calls
            calls += 1
            if enabled and args[1].key == MatrixKey(1, "K"):
                raise RuntimeError("injected codec failure")
            return raw_encoder(*args, **kwargs)

        runtime = self.open_runtime(encoder=encode)
        self.seed(runtime)
        old_head = runtime.branch_info("main")["head_id"]
        before_bytes = runtime.store.stats()["object_bytes"]
        enabled = True
        values, contexts = chunk_inputs(runtime.model, start=4, offset=100)
        with self.assertRaisesRegex(RuntimeError, "codec failure"):
            runtime.seal_chunk("main", "c1", values, contexts)
        self.assertEqual(runtime.branch_info("main")["head_id"], old_head)
        self.assertEqual(runtime.store.stats()["object_bytes"], before_bytes)
        self.assertEqual(runtime.store.stats()["leases"], 0)
        enabled = False
        runtime.seal_chunk("main", "c1", values, contexts)
        self.assertEqual(runtime.branch_info("main")["token_end"], 8)

    def test_interrupted_multilayer_publication_has_no_partial_root(self):
        runtime = self.open_runtime()
        self.seed(runtime)
        before = runtime.branch_info("main")
        values, contexts = chunk_inputs(runtime.model, start=4, offset=100)
        original = runtime._save_tensor
        calls = 0

        def save(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = original(*args, **kwargs)
            if calls == 3:
                raise RuntimeError("injected publication interruption")
            return result

        with patch.object(runtime, "_save_tensor", side_effect=save):
            with self.assertRaisesRegex(RuntimeError, "publication interruption"):
                runtime.seal_chunk("main", "c1", values, contexts)
        self.assertEqual(runtime.branch_info("main"), before)
        runtime.close()
        restarted = self.open_runtime()
        self.assertEqual(restarted.branch_info("main"), before)
        self.assertEqual(restarted.store.stats()["leases"], 0)

    def test_process_crash_recovers_and_reclaims_streamed_unpublished_packets(self):
        runtime = self.open_runtime()
        self.seed(runtime)
        runtime.gc()
        before = runtime.branch_info("main")
        before_bytes = runtime.store.stats()["object_bytes"]
        runtime.close()
        script = r'''
import os, sys
from reelcache.manager import KVRuntime
from reelcache.types import MatrixKey
from tests.core_support import model, chunk_inputs, raw_encoder, raw_decoder
def encode(tensor, context, spec, **kwargs):
    if context.key == MatrixKey(1, "K"):
        os._exit(0)
    return raw_encoder(tensor, context, spec, **kwargs)
runtime = KVRuntime(sys.argv[1], model(), encoder=encode, decoder=raw_decoder)
values, contexts = chunk_inputs(runtime.model, start=4, offset=500)
runtime.seal_chunk("main", "interrupted", values, contexts)
raise AssertionError("injected process exit did not occur")
'''
        subprocess.run(
            [sys.executable, "-c", script, str(self.workspace / "runtime")],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy(),
        )
        restarted = self.open_runtime()
        self.assertEqual(restarted.branch_info("main"), before)
        self.assertEqual(restarted.store.stats()["leases"], 0)
        self.assertEqual(restarted.store.stats()["object_bytes"], before_bytes)

    def test_pending_encode_blocks_snapshot_but_other_runtime_sees_old_complete_head(self):
        started, proceed = threading.Event(), threading.Event()
        enabled = False

        def encode(*args, **kwargs):
            if enabled and args[1].key == MatrixKey(0, "K"):
                started.set()
                if not proceed.wait(60):
                    raise RuntimeError("test synchronization timeout")
            return raw_encoder(*args, **kwargs)

        runtime = self.open_runtime(encoder=encode)
        values, _ = self.seed(runtime)
        reader = self.open_runtime()
        next_values, contexts = chunk_inputs(runtime.model, start=4, offset=100)
        errors = []
        enabled = True

        def publish():
            try:
                runtime.seal_chunk("main", "c1", next_values, contexts)
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=publish)
        thread.start()
        self.assertTrue(started.wait(60))
        try:
            with self.assertRaises(PendingError):
                runtime.snapshot("main", metadata=metadata())
            with self.assertRaises(PendingError):
                runtime.close()
            self.assertEqual(reader.branch_info("main")["token_end"], 4)
            with reader.acquire_history("main", MatrixKey(0, "K")) as lease:
                self.assertTrue(torch.equal(lease.tensor, values[MatrixKey(0, "K")]))
        finally:
            proceed.set()
            thread.join(60)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(reader.branch_info("main")["token_end"], 8)

    def test_lease_release_drops_real_history_and_pins_gc(self):
        runtime = self.open_runtime()
        self.seed(runtime)
        runtime.gc()
        head = runtime.store.get_manifest(runtime.branch_info("main")["head_id"])
        chunk = runtime.store.get_manifest(head["chunks"][0])
        leased_packet = chunk["matrices"][0]["tensor"]
        leased_payloads = runtime.store.get_manifest(leased_packet)["encoded"]["payloads"]
        lease = runtime.acquire_history("main", MatrixKey(0, "K"))
        reference = weakref.ref(lease.tensor)
        runtime.delete_branch("main")
        self.assertGreater(runtime.gc()["objects_removed"], 0)
        runtime.store.verify(leased_packet)
        for payload in leased_payloads.values():
            self.assertGreater(len(runtime.store.get(payload)), 0)
        lease.release()
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(runtime.resident_bytes, 0)
        self.assertGreater(runtime.gc()["objects_removed"], 0)
        self.assertEqual(runtime.store.stats()["blob_bytes"], 0)

    def test_committed_missing_layer_is_detected_on_restart(self):
        runtime = self.open_runtime()
        self.seed(runtime)
        head = runtime.store.get_manifest(runtime.branch_info("main")["head_id"])
        chunk = runtime.store.get_manifest(head["chunks"][0])
        chunk["matrices"] = chunk["matrices"][:-1]
        incomplete = runtime.store.put_manifest(
            chunk, references=[entry["tensor"] for entry in chunk["matrices"]]
        )
        invalid_head = runtime.store.put_manifest(
            {**head, "chunks": [incomplete]}, references=[incomplete]
        )
        runtime.store.set_ref("branches/main", invalid_head)
        runtime.close()
        with self.assertRaisesRegex(IntegrityError, "every layer"):
            self.open_runtime()

    def test_concurrent_append_compare_and_swap_does_not_lose_commits(self):
        started, proceed = threading.Event(), threading.Event()
        enabled = False

        def encode(*args, **kwargs):
            if enabled and args[1].key == MatrixKey(0, "K"):
                started.set()
                if not proceed.wait(60):
                    raise RuntimeError("test synchronization timeout")
            return raw_encoder(*args, **kwargs)

        first = self.open_runtime(encoder=encode)
        self.seed(first)
        second = self.open_runtime()
        pending_values, pending_contexts = chunk_inputs(first.model, start=4, offset=100)
        winner_values, winner_contexts = chunk_inputs(first.model, start=4, offset=200)
        failures = []
        enabled = True

        def publish():
            try:
                first.seal_chunk("main", "loser", pending_values, pending_contexts)
            except StateError:
                failures.append("changed")

        thread = threading.Thread(target=publish)
        thread.start()
        self.assertTrue(started.wait(60))
        try:
            second.seal_chunk("main", "winner", winner_values, winner_contexts)
        finally:
            proceed.set()
            thread.join(60)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, ["changed"])
        self.assertEqual(second.branch_info("main")["chunk_ids"], ["c0", "winner"])
        with second.acquire_history("main", MatrixKey(0, "V"), token_start=4) as lease:
            self.assertTrue(torch.equal(lease.tensor, winner_values[MatrixKey(0, "V")]))

    def test_capacity_failure_happens_before_decoding_and_releases_gc_pin(self):
        decoder = Mock(side_effect=raw_decoder)
        runtime = self.open_runtime(max_resident_bytes=63, decoder=decoder)
        self.seed(runtime)
        with patch.object(runtime.store, "_get", wraps=runtime.store._get) as reads:
            with self.assertRaises(CapacityError):
                runtime.acquire_history("main", MatrixKey(0, "K"))
            self.assertFalse(any(call.args[0].startswith(("blob:", "basis:")) for call in reads.call_args_list))
        decoder.assert_not_called()
        self.assertEqual(runtime.residency.reserved_bytes, 0)
        self.assertEqual(runtime.store.stats()["leases"], 0)

    def test_layer_lease_reads_only_selected_tensor_manifests_and_payload_once(self):
        runtime = self.open_runtime(model(layers=4))
        first, _ = self.seed(runtime)
        second, contexts = chunk_inputs(runtime.model, start=4, offset=500)
        runtime.seal_chunk("main", "c1", second, contexts)
        wanted = MatrixKey(2, "V")
        all_packets, selected_packets, selected_payloads = set(), set(), set()
        head = runtime.store.get_manifest(runtime.branch_info("main")["head_id"])
        for chunk_id in head["chunks"]:
            chunk = runtime.store.get_manifest(chunk_id)
            for record in chunk["matrices"]:
                all_packets.add(record["tensor"])
                if MatrixKey.from_dict(record["key"]) == wanted:
                    selected_packets.add(record["tensor"])
                    packet = runtime.store.get_manifest(record["tensor"])
                    selected_payloads.update(packet["encoded"]["payloads"].values())
        with patch.object(runtime.store, "_get", wraps=runtime.store._get) as reads:
            with runtime.acquire_history("main", wanted) as lease:
                self.assertTrue(torch.equal(lease.tensor, torch.cat((first[wanted], second[wanted]))))
            requested = [call.args[0] for call in reads.call_args_list]
        self.assertEqual(all_packets.intersection(requested), selected_packets)
        for payload in selected_payloads:
            self.assertEqual(requested.count(payload), 1)
        with patch.object(runtime.store, "_get", wraps=runtime.store._get) as reads:
            self.assertEqual(runtime.branch_info("main")["token_end"], 8)
            self.assertFalse(all_packets.intersection(call.args[0] for call in reads.call_args_list))

    def test_targeted_metadata_reads_preserve_requested_and_snapshot_corruption_checks(self):
        runtime = self.open_runtime()
        values, _ = self.seed(runtime)
        snapshot = runtime.snapshot("main", metadata=metadata())
        head = runtime.store.get_manifest(runtime.branch_info("main")["head_id"])
        chunk = runtime.store.get_manifest(head["chunks"][0])
        bad_key = MatrixKey(1, "V")
        bad_packet = next(record["tensor"] for record in chunk["matrices"]
                          if MatrixKey.from_dict(record["key"]) == bad_key)
        runtime.store.path_for(bad_packet).write_bytes(b"corrupt packet manifest")
        with runtime.acquire_history("main", MatrixKey(0, "K")) as lease:
            self.assertTrue(torch.equal(lease.tensor, values[MatrixKey(0, "K")]))
        with self.assertRaises(IntegrityError):
            runtime.acquire_history("main", bad_key)
        with self.assertRaises(IntegrityError):
            runtime.load_snapshot(snapshot)
        with self.assertRaises(IntegrityError):
            runtime.fork(snapshot, "bad-restore")
        self.assertIsNone(runtime.store.get_ref("branches/bad-restore"))

    def test_small_range_lease_never_reconstructs_or_caches_full_history(self):
        decoded_references = []

        def decode(*args, **kwargs):
            value = raw_decoder(*args, **kwargs)
            decoded_references.append(weakref.ref(value))
            return value

        runtime = self.open_runtime(model(layers=1), max_resident_bytes=192, decoder=decode)
        runtime.create_branch("main")
        for index in range(4):
            values, contexts = chunk_inputs(runtime.model, start=4 * index, offset=100 * index)
            runtime.seal_chunk("main", index, values, contexts)
        with self.assertRaises(CapacityError):
            runtime.acquire_history("main", MatrixKey(0, "K"))
        self.assertEqual(decoded_references, [])
        lease = runtime.acquire_history("main", MatrixKey(0, "K"), token_start=12)
        self.assertTrue(torch.equal(lease.tensor, values[MatrixKey(0, "K")]))
        self.assertEqual(runtime.resident_bytes, 64)
        self.assertEqual(len(decoded_references), 1)
        self.assertIsNone(decoded_references[0]())
        final_reference = weakref.ref(lease.tensor)
        lease.release()
        self.assertIsNone(final_reference())
        self.assertEqual(runtime.resident_bytes, 0)

    def test_lease_coverage_and_channel_bounds_are_explicit(self):
        runtime = self.open_runtime()
        self.seed(runtime)
        for kwargs in ({"token_start": -1}, {"token_end": 5}, {"token_start": 3, "token_end": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(StateError):
                runtime.acquire_history("main", MatrixKey(0, "K"), **kwargs)
        for columns in (slice(-1, 2), slice(0, 5), slice(0, 3, 2), (2, 2), [], [True], [4]):
            with self.subTest(columns=columns), self.assertRaises(ConfigurationError):
                runtime.acquire_history("main", MatrixKey(0, "K"), columns=columns)
        with runtime.acquire_history("main", MatrixKey(0, "K"), token_end=0) as lease:
            self.assertEqual(lease.tensor.shape, (0, 4))

    def test_store_budget_failure_cannot_publish_some_layers(self):
        runtime = self.open_runtime()
        self.seed(runtime)
        before = runtime.branch_info("main")
        runtime.store.max_bytes = runtime.store.stats()["object_bytes"] + 100
        values, contexts = chunk_inputs(runtime.model, start=4, offset=100)
        with self.assertRaises(CapacityError):
            runtime.seal_chunk("main", "c1", values, contexts)
        self.assertEqual(runtime.branch_info("main"), before)

    def test_false_provider_shape_is_detected_on_commit_or_decode(self):
        def bad_encode(tensor, context, spec, **kwargs):
            result = raw_encoder(tensor, context, spec, **kwargs)
            return EncodedTensor({**result.metadata, "shape": [2, 8]}, result.payloads)

        runtime = self.open_runtime(encoder=bad_encode)
        runtime.create_branch("main")
        values, contexts = chunk_inputs(runtime.model)
        with self.assertRaises(IntegrityError):
            runtime.seal_chunk("main", "c0", values, contexts)
        runtime.encoder = raw_encoder
        runtime.seal_chunk("main", "c0", values, contexts)
        runtime.decoder = lambda *args, **kwargs: torch.zeros((2, 8))
        with self.assertRaises(IntegrityError):
            runtime.acquire_history("main", MatrixKey(0, "K"))
        self.assertEqual(runtime.residency.reserved_bytes, 0)
        self.assertEqual(runtime.store.stats()["leases"], 0)

    def test_provider_context_spec_and_byte_descriptors_bind_to_request(self):
        runtime = self.open_runtime()
        runtime.create_branch("main")
        values, contexts = chunk_inputs(runtime.model)
        for mismatch in ("context", "codec", "kind", "payload_hash", "payload_length"):
            def encode(tensor, context, spec, **kwargs):
                packet = raw_encoder(tensor, context, spec, **kwargs)
                meta = packet.metadata
                if mismatch == "context":
                    meta["context"]["model"]["fingerprint"] = "another-model"
                elif mismatch == "codec":
                    meta["codec"]["normalization"] = "global"
                elif mismatch == "kind":
                    meta["kind"] = "svd"
                elif mismatch == "payload_hash":
                    meta["payloads"]["matrix"]["sha256"] = "0" * 64
                else:
                    meta["payloads"]["matrix"]["bytes"] += 1
                return EncodedTensor(meta, packet.payloads)

            runtime.encoder = encode
            with self.subTest(mismatch=mismatch), self.assertRaises(IntegrityError):
                runtime.seal_chunk("main", "bad", values, contexts)
            self.assertEqual(runtime.branch_info("main")["token_end"], 0)
            self.assertEqual(runtime.store.stats()["blob_bytes"], 0)

    def test_corrupt_provider_descriptor_is_rejected_before_restore_publication(self):
        runtime = self.open_runtime()
        self.seed(runtime)
        snapshot = runtime.snapshot("main", metadata=metadata())
        saved = runtime.store.get_manifest(snapshot)
        head = runtime.store.get_manifest(saved["head"])
        chunk = runtime.store.get_manifest(head["chunks"][0])
        packet_id = chunk["matrices"][0]["tensor"]
        packet = runtime.store.get_manifest(packet_id)
        packet["encoded"]["metadata"]["payloads"]["matrix"]["bytes"] += 1
        damaged_packet = runtime.store.put_manifest(packet, references=runtime.store.references(packet_id))
        chunk["matrices"][0]["tensor"] = damaged_packet
        damaged_chunk = runtime.store.put_manifest(
            chunk, references=[row["tensor"] for row in chunk["matrices"]]
        )
        damaged_head = runtime.store.put_manifest({**head, "chunks": [damaged_chunk]},
                                                   references=[damaged_chunk])
        damaged_snapshot = runtime.store.put_manifest(
            {**saved, "head": damaged_head}, references=[damaged_head, saved["tensor_state"]]
        )
        with self.assertRaises(IntegrityError):
            runtime.restore(damaged_snapshot, "invalid")
        self.assertIsNone(runtime.store.get_ref("branches/invalid"))
        self.assertEqual(runtime.branch_info("main")["token_end"], 4)

    def test_legacy_ignored_fraction_headers_remain_immutable_and_restartable(self):
        runtime = self.open_runtime()
        values, contexts = chunk_inputs(runtime.model)
        for mode, legacy_fraction in (("global", 1.0), ("per_channel", 0.0)):
            def encode(tensor, context, spec, **kwargs):
                packet = raw_encoder(tensor, context, spec, **kwargs)
                meta = packet.metadata
                meta["codec"]["protected_fraction"] = legacy_fraction
                return EncodedTensor(meta, packet.payloads)

            runtime.encoder = encode
            runtime.create_branch(mode)
            runtime.seal_chunk(mode, "c0", values, contexts, CodecSpec(normalization=mode))
            head = runtime.store.get_manifest(runtime.branch_info(mode)["head_id"])
            chunk = runtime.store.get_manifest(head["chunks"][0])
            packet = runtime.store.get_manifest(chunk["matrices"][0]["tensor"])
            self.assertEqual(packet["encoded"]["metadata"]["codec"]["protected_fraction"], legacy_fraction)
        runtime.close()
        reopened = self.open_runtime()
        for mode in ("global", "per_channel"):
            with reopened.acquire_history(mode, MatrixKey(0, "K")) as lease:
                self.assertEqual(tensor_bytes(lease.tensor), tensor_bytes(values[MatrixKey(0, "K")]))

    def test_actual_reused_full_bases_deduplicate_even_for_rank_views(self):
        reuse_arguments = []

        def encode(tensor, context, spec, *, reuse_basis=None, reuse_threshold=None):
            reuse_arguments.append(reuse_basis)
            basis = (torch.eye(context.channels, dtype=torch.float16)
                     if reuse_basis is None else reuse_basis)
            coefficients = tensor.float() @ basis[:spec.rank].float().T
            payloads = {"coefficients": tensor_bytes(coefficients)}
            return EncodedTensor(
                {**encoded_metadata(context, spec, payloads, basis=basis),
                 "coefficient_shape": list(coefficients.shape)},
                payloads, tensor_bytes(basis),
            )

        runtime = self.open_runtime(encoder=encode)
        runtime.create_branch("main")
        values, contexts = chunk_inputs(runtime.model)
        runtime.seal_chunk("main", "c0", values, contexts, CodecSpec("svd", 4))
        basis_bytes = runtime.store.stats()["basis_bytes"]
        self.assertEqual(basis_bytes, 4 * 4 * 2)
        following, next_contexts = chunk_inputs(runtime.model, start=4, offset=10)
        runtime.seal_chunk("main", "c1", following, next_contexts, CodecSpec("svd", 2),
                           reuse_threshold=1.0)
        self.assertEqual(runtime.store.stats()["basis_bytes"], basis_bytes)
        for tensor in reuse_arguments[4:]:
            self.assertEqual(tensor.shape, (4, 4))
            self.assertEqual(tensor.dtype, torch.float16)
        head = runtime.store.get_manifest(runtime.branch_info("main")["head_id"])
        last = runtime.store.get_manifest(head["chunks"][-1])
        record = runtime.store.get_manifest(last["matrices"][0]["tensor"])
        self.assertEqual(record["encoded"]["metadata"]["basis"]["view_rank"], 2)
        self.assertEqual(len(runtime.store.get(record["encoded"]["basis"])), 32)

    def test_owner_channel_slice_is_forwarded_before_local_factor_reconstruction(self):
        def encode(tensor, context, spec, **kwargs):
            basis = torch.eye(context.channels, dtype=torch.float16)
            payloads = {"coefficients": tensor_bytes(torch.zeros(context.token_count, spec.rank))}
            return EncodedTensor(
                encoded_metadata(context, spec, payloads, basis=basis), payloads,
                tensor_bytes(basis),
            )

        decoder = Mock(side_effect=lambda encoded, **kwargs: torch.zeros((8, 4), dtype=kwargs["dtype"]))
        runtime = self.open_runtime(
            model(layers=1, channels=8), max_resident_bytes=704, encoder=encode, decoder=decoder
        )
        runtime.create_branch("main")
        values, contexts = chunk_inputs(runtime.model, frames=2)
        runtime.seal_chunk("main", "c0", values, contexts, CodecSpec("svd", 2))
        with runtime.acquire_history("main", MatrixKey(0, "K"), columns=slice(4, 8)) as lease:
            self.assertEqual(lease.tensor.shape, (8, 4))
            self.assertEqual(runtime.resident_bytes, 128)
            self.assertEqual(runtime.residency.peak_reserved_bytes, 704)
        self.assertEqual(decoder.call_args.kwargs["columns"], slice(4, 8))
        with self.assertRaises(CapacityError):
            runtime.acquire_history("main", MatrixKey(0, "K"))
        self.assertEqual(runtime.store.stats()["basis_bytes"], 128)

    def test_executor_integration_uses_cpu_bounded_jobs_without_claimed_overlap(self):
        from reelcache.execution import BoundedExecutor
        executor = BoundedExecutor(max_workers=1, max_pending=1, max_staging_bytes=64)
        self.addCleanup(executor.close)
        runtime = self.open_runtime(executor=executor)
        self.seed(runtime)
        with runtime.acquire_history("main", MatrixKey(0, "K")) as lease:
            self.assertEqual(lease.tensor.shape, (4, 4))
        self.assertEqual(runtime.branch_info("main")["token_end"], 4)

    def test_snapshot_deletion_gc_keeps_branch_and_branch_deletion_keeps_snapshot(self):
        runtime = self.open_runtime()
        self.seed(runtime)
        snapshot = runtime.snapshot("main", metadata=metadata(), tensors=self.state_tensors())
        runtime.delete_branch("main")
        runtime.gc()
        runtime.restore(snapshot, "restored")
        runtime.delete_snapshot(snapshot)
        runtime.gc()
        with runtime.acquire_history("restored", MatrixKey(0, "V")) as lease:
            self.assertEqual(lease.tensor.shape, (4, 4))
        runtime.delete_branch("restored")
        runtime.gc()
        self.assertEqual(runtime.store.stats()["blob_bytes"], 0)

    def test_raw_resume_uses_real_provider_rng_latent_and_history_after_process_restart(self):
        from reelcache.codecs import decode_tensor, encode_tensor
        runtime = self.open_runtime(model(dtype="bfloat16"), encoder=encode_tensor, decoder=decode_tensor)
        initial, _ = self.seed(runtime)
        generator = torch.Generator().manual_seed(381)
        state = self.state_tensors()
        state["rng"] = generator.get_state()
        state["latent"] = torch.randn((3, 4), generator=generator).to(torch.bfloat16)
        state["rng"] = generator.get_state()
        snapshot = runtime.snapshot("main", metadata=metadata(), tensors=state)
        expected_noise = torch.randn((3, 4), generator=generator)
        expected = state["latent"].float() + expected_noise + initial[MatrixKey(0, "K")].float().sum(0)
        expected_bytes = tensor_bytes(expected).hex()
        runtime.close()
        script = r'''
import sys, torch
from reelcache.manager import KVRuntime
from reelcache.types import MatrixKey, ModelIdentity
from reelcache.serialization import tensor_bytes
identity=ModelIdentity("fixture","fixture-v1",2,4,"tokens_channels","bfloat16")
with KVRuntime(sys.argv[1],identity) as runtime:
    state=runtime.restore(sys.argv[2],"resumed-child")
    generator=torch.Generator()
    generator.set_state(state.tensors["rng"])
    noise=torch.randn((3,4),generator=generator)
    with runtime.acquire_history("resumed-child",MatrixKey(0,"K")) as lease:
        actual=state.tensors["latent"].float()+noise+lease.tensor.float().sum(0)
    assert tensor_bytes(actual).hex()==sys.argv[3], "RAW resume parity failed"
    assert runtime.resident_bytes==0
    assert runtime.branch_info("main")["token_end"]==4
print("real RAW resume passed")
'''
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.workspace / "runtime"), snapshot, expected_bytes],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=os.environ.copy(),
        )
        self.assertIn("real RAW resume passed", result.stdout)

    def test_real_provider_mixed_native_dtypes_and_owner_column_slices(self):
        from reelcache.codecs import decode_tensor, encode_tensor
        runtime = self.open_runtime(
            model(layers=1, channels=8, dtype={"K": "float32", "V": "bfloat16"}),
            encoder=encode_tensor, decoder=decode_tensor,
        )
        values, _ = self.seed(runtime)
        for key, value in values.items():
            for columns in (slice(2, 6), (2, 6), [7, 1], [0, 3, 5]):
                with runtime.acquire_history("main", key, columns=columns) as lease:
                    self.assertEqual(lease.tensor.dtype, value.dtype)
                    expected = value[:, slice(*columns) if isinstance(columns, tuple) else columns]
                    self.assertEqual(tensor_bytes(lease.tensor), tensor_bytes(expected))

    def test_real_pre_rope_fp32_and_raw_bf16_branches_share_native_identity(self):
        from reelcache.codecs import decode_tensor, encode_tensor
        identity = model(layers=1, dtype="bfloat16")
        observed = []

        def encode(tensor, context, spec, **kwargs):
            if spec.kind == "svd":
                observed.append((tensor.dtype, tensor_bytes(tensor)))
            return encode_tensor(tensor, context, spec, **kwargs)

        runtime = self.open_runtime(identity, encoder=encode, decoder=decode_tensor)
        raw, raw_contexts = chunk_inputs(identity, key_space="post_rope")
        k, v = MatrixKey(0, "K"), MatrixKey(0, "V")
        pre = {k: raw[k].float() + .003, v: raw[v]}
        pre_contexts = {
            k: TensorContext(identity, k, 0, 1, 2, 2, "pre_rope", source_dtype="float32"),
            v: raw_contexts[v],
        }
        self.assertNotEqual(tensor_bytes(pre[k]), tensor_bytes(pre[k].bfloat16().float()))
        for branch, values, contexts, specs in (
            ("raw-native", raw, raw_contexts, CodecSpec()),
            ("raw-pre", pre, pre_contexts, CodecSpec()),
            ("svd-pre", pre, pre_contexts, {k: CodecSpec("svd", 2), v: CodecSpec()}),
        ):
            runtime.create_branch(branch)
            runtime.seal_chunk(branch, "c0", values, contexts, specs)
            self.assertEqual(runtime.model, identity)
        self.assertEqual(observed, [(torch.float32, tensor_bytes(pre[k]))])
        with runtime.acquire_history("raw-native", k) as lease:
            self.assertEqual(lease.tensor.dtype, torch.bfloat16)
            self.assertEqual(tensor_bytes(lease.tensor), tensor_bytes(raw[k]))
        with runtime.acquire_history("raw-pre", k) as lease:
            self.assertEqual(lease.tensor.dtype, torch.float32)
            self.assertEqual(tensor_bytes(lease.tensor), tensor_bytes(pre[k]))
        with runtime.acquire_history("svd-pre", k) as lease:
            self.assertEqual(lease.tensor.dtype, torch.float32)
        with runtime.acquire_history("svd-pre", v) as lease:
            self.assertEqual(lease.tensor.dtype, torch.bfloat16)
            self.assertEqual(tensor_bytes(lease.tensor), tensor_bytes(pre[v]))
        runtime.close()
        reopened = self.open_runtime(identity, encoder=encode_tensor, decoder=decode_tensor)
        with reopened.acquire_history("raw-pre", k, token_end=0) as lease:
            self.assertEqual(lease.tensor.dtype, torch.float32)
        self.assertEqual(reopened.history_contexts("svd-pre", k)[0].source_dtype, "float32")

    def test_branch_source_dtype_must_remain_stable_and_input_is_never_implicitly_cast(self):
        identity = model(layers=1, dtype="bfloat16")
        runtime = self.open_runtime(identity)
        runtime.create_branch("main")
        values, contexts = chunk_inputs(identity)
        key = MatrixKey(0, "K")
        contexts[key] = TensorContext(identity, key, 0, 1, 2, 2, "pre_rope", source_dtype="float32")
        with self.assertRaises(ConfigurationError):
            runtime.seal_chunk("main", "bad-source", values, contexts)
        values[key] = values[key].float()
        runtime.seal_chunk("main", "c0", values, contexts)
        next_values, next_contexts = chunk_inputs(identity, start=4)
        with self.assertRaisesRegex(ConfigurationError, "source dtype"):
            runtime.seal_chunk("main", "changed-dtype", next_values, next_contexts)
        self.assertEqual(runtime.branch_info("main")["token_end"], 4)

    def test_real_hevc_normalization_endpoints_match_declared_masks(self):
        from reelcache.codecs import decode_tensor, encode_tensor
        runtime = self.open_runtime(model(layers=1), encoder=encode_tensor, decoder=decode_tensor)
        values, contexts = chunk_inputs(runtime.model, height=4, width=4)
        for index, (mode, fraction, protected) in enumerate((
            ("global", 0, 0), ("per_channel", 0, 2),
            ("leading_columns", 0, 0), ("leading_columns", 1, 2),
        )):
            branch = f"mode-{index}"
            runtime.create_branch(branch)
            runtime.seal_chunk(
                branch, "c0", values, contexts,
                CodecSpec("svd_hevc", rank=2, normalization=mode, protected_fraction=fraction),
            )
            head = runtime.store.get_manifest(runtime.branch_info(branch)["head_id"])
            chunk = runtime.store.get_manifest(head["chunks"][0])
            for record in chunk["matrices"]:
                meta = runtime.store.get_manifest(record["tensor"])["encoded"]["metadata"]
                effective_fraction = 0 if mode == "global" else 1 if mode == "per_channel" else fraction
                self.assertEqual(meta["codec"]["protected_fraction"], effective_fraction)
                self.assertEqual(meta["normalization"]["mode"], mode)
                self.assertEqual(meta["normalization"]["protected_count"], protected)
            with runtime.acquire_history(branch, MatrixKey(0, "K"), columns=slice(1, 3)) as lease:
                self.assertEqual(lease.tensor.shape, (16, 2))
                self.assertTrue(torch.isfinite(lease.tensor).all().item())

    def test_real_svd_reuses_full_basis_for_rank_view_without_decomposition(self):
        from reelcache.codecs import decode_tensor, encode_tensor
        runtime = self.open_runtime(model(layers=1), encoder=encode_tensor, decoder=decode_tensor)
        runtime.create_branch("main")
        values, contexts = chunk_inputs(runtime.model, frames=2)
        for key in values:
            values[key] = values[MatrixKey(0, "K")].clone()
        runtime.seal_chunk("main", "c0", values, contexts, CodecSpec("svd", 4))
        basis_bytes = runtime.store.stats()["basis_bytes"]
        self.assertEqual(basis_bytes, 4 * 4 * 2)
        following, next_contexts = chunk_inputs(runtime.model, start=8, frames=2)
        for key in following:
            following[key] = values[key].clone()
        with (
            patch("torch.linalg.svd", side_effect=AssertionError("reuse performed SVD")),
            patch("torch.linalg.eigh", side_effect=AssertionError("reuse performed EIGH")),
            patch("torch.linalg.qr", side_effect=AssertionError("reuse performed QR")),
        ):
            runtime.seal_chunk("main", "c1", following, next_contexts, CodecSpec("svd", 2),
                               reuse_threshold=.01)
        self.assertEqual(runtime.store.stats()["basis_bytes"], basis_bytes)
        head = runtime.store.get_manifest(runtime.branch_info("main")["head_id"])
        chunk = runtime.store.get_manifest(head["chunks"][-1])
        for record in chunk["matrices"]:
            metrics = runtime.store.get_manifest(record["tensor"])["encoded"]["metrics"]
            self.assertTrue(metrics["reuse_accepted"])
            self.assertEqual(metrics["decomposition_count"], 0)
        with runtime.acquire_history("main", MatrixKey(0, "K"), token_start=8) as lease:
            relative = torch.linalg.vector_norm(lease.tensor - following[MatrixKey(0, "K")])
            relative /= torch.linalg.vector_norm(following[MatrixKey(0, "K")])
            self.assertLess(relative.item(), .01)
