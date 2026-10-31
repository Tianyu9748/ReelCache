import dataclasses
from types import MappingProxyType
from unittest.mock import Mock, patch

import torch

from reelcache.errors import ConfigurationError, IntegrityError, StateError
from reelcache.manager import KVRuntime, RestoredState
from reelcache.serialization import deserialize_tensors
from tests.core_support import CoreWorkspaceTest, metadata, model
from tests.test_adaptive_manager import inputs, rank_policy


class RestoreMetadataTests(CoreWorkspaceTest):
    def setup_snapshot(self, *, adaptive=False, verify_reads=False):
        runtime = KVRuntime(self.workspace / "runtime", model(layers=1), verify_reads=verify_reads)
        self.addCleanup(runtime.close)
        runtime.create_branch("main")
        policy = rank_policy(runtime.model, solver="exact_oracle") if adaptive else None
        runtime.seal_chunk("main", "c0", *inputs(runtime.model), rank_policy=policy)
        native = {
            "rng": torch.Generator().manual_seed(19).get_state(),
            "latent": torch.arange(4, dtype=torch.float32),
        }
        snapshot = runtime.snapshot("main", metadata=metadata(), tensors=native)
        record = runtime.store.get_manifest(snapshot)
        return runtime, policy, snapshot, record["tensor_state"], native

    def reject_state_blob(self, runtime, state_blob):
        original = runtime.store._get

        def checked(object_id, **kwargs):
            self.assertNotEqual(object_id, state_blob, "metadata rejection read native tensor-state bytes")
            return original(object_id, **kwargs)

        return patch.object(runtime.store, "_get", side_effect=checked)

    def test_wrong_conditioning_rejects_before_blob_read_deserialization_or_full_validator(self):
        runtime, _, snapshot, state_blob, _ = self.setup_snapshot()
        runtime.seal_chunk("main", "c1", *inputs(runtime.model, start=4, rank=2))
        before = runtime.branch_info("main")
        rng = torch.get_rng_state().clone()
        full = Mock(side_effect=AssertionError("wrong conditioning reached full validator"))

        def reject(preview):
            self.assertIsInstance(preview, RestoredState)
            self.assertEqual(dict(preview.tensors), {})
            self.assertEqual(preview.metadata["conditioning"], metadata()["conditioning"])
            raise ConfigurationError("requested prompt differs from snapshot conditioning")

        for operation, branch in ((runtime.fork, "rejected"), (runtime.restore, "main")):
            with self.reject_state_blob(runtime, state_blob), \
                    patch("reelcache.manager.deserialize_tensors", side_effect=AssertionError("early deserialize")):
                with self.assertRaises(ConfigurationError):
                    operation(snapshot, branch, metadata_validator=reject, validator=full)
            self.assertEqual(runtime.branch_info("main"), before)
            self.assertIsNone(runtime.store.get_ref("branches/rejected"))
            self.assertEqual(runtime.store.stats()["leases"], 0)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        full.assert_not_called()

    def test_static_consumer_sees_actual_adaptive_head_before_native_state_is_read(self):
        runtime, policy, snapshot, state_blob, _ = self.setup_snapshot(adaptive=True)
        before = runtime.branch_info("main")
        scalar = runtime.history_rank_state("main").to_dict()
        rng = torch.get_rng_state().clone()

        def reject(preview):
            self.assertEqual(preview.rank_policy.fingerprint, policy.fingerprint)
            self.assertEqual(preview.rank_state.to_dict(), scalar)
            self.assertEqual(dict(preview.tensors), {})
            raise ConfigurationError("static consumer cannot adopt an adaptive snapshot")

        with self.reject_state_blob(runtime, state_blob), \
                patch("reelcache.manager.deserialize_tensors", side_effect=AssertionError("early deserialize")):
            with self.assertRaises(ConfigurationError):
                runtime.fork(snapshot, "static", metadata_validator=reject)
        self.assertIsNone(runtime.store.get_ref("branches/static"))
        self.assertEqual(runtime.branch_info("main"), before)
        self.assertEqual(runtime.history_rank_state("main").to_dict(), scalar)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(runtime.store.stats()["leases"], 0)

    def test_metadata_preview_is_detached_and_both_validators_precede_publication(self):
        runtime, policy, snapshot, state_blob, native = self.setup_snapshot(adaptive=True)
        events = []
        original_get = runtime.store._get

        def checked_get(object_id):
            if object_id == state_blob:
                self.assertIn("metadata", events)
            return original_get(object_id)

        def check_metadata(preview):
            self.assertIsNone(runtime.store.get_ref("branches/child"))
            self.assertEqual(preview.rank_policy.fingerprint, policy.fingerprint)
            self.assertEqual(preview.rank_state.completed_chunks, 1)
            self.assertIsInstance(preview.tensors, MappingProxyType)
            with self.assertRaises(TypeError):
                preview.tensors["unexpected"] = torch.zeros(1)
            preview.metadata["conditioning"]["identity"] = "isolated-preview-mutation"
            events.append("metadata")

        def decode(raw):
            events.append("deserialize")
            return deserialize_tensors(raw)

        def check_full(restored):
            self.assertEqual(events, ["metadata", "deserialize"])
            self.assertIsNone(runtime.store.get_ref("branches/child"))
            self.assertEqual(restored.metadata, metadata())
            self.assertTrue(torch.equal(restored.tensors["latent"], native["latent"]))
            events.append("full")

        with patch.object(runtime.store, "_get", side_effect=checked_get), \
                patch("reelcache.manager.deserialize_tensors", side_effect=decode):
            restored = runtime.fork(
                snapshot, "child", rank_policy=policy,
                metadata_validator=check_metadata, validator=check_full,
            )
        self.assertEqual(events, ["metadata", "deserialize", "full"])
        self.assertEqual(restored.metadata, metadata())
        self.assertEqual(runtime.branch_info("child")["head_id"], runtime.branch_info("main")["head_id"])
        restored.metadata["conditioning"]["identity"] = "caller-copy"
        self.assertEqual(runtime.load_snapshot(snapshot).metadata, metadata())

    def test_invalid_callback_and_wrong_policy_reject_before_state_reads(self):
        runtime, policy, snapshot, state_blob, _ = self.setup_snapshot(adaptive=True)
        cases = (
            (False, ConfigurationError),
            (lambda preview: False, ConfigurationError),
            (Mock(side_effect=RuntimeError("metadata failure")), StateError),
        )
        for callback, expected in cases:
            with self.reject_state_blob(runtime, state_blob), \
                    patch("reelcache.manager.deserialize_tensors", side_effect=AssertionError("early deserialize")):
                with self.assertRaises(expected):
                    runtime.fork(snapshot, "rejected", metadata_validator=callback)
            self.assertIsNone(runtime.store.get_ref("branches/rejected"))
        callback = Mock(side_effect=AssertionError("wrong policy reached metadata validator"))
        with self.reject_state_blob(runtime, state_blob), \
                patch("reelcache.manager.deserialize_tensors", side_effect=AssertionError("early deserialize")):
            with self.assertRaises(ConfigurationError):
                runtime.fork(
                    snapshot, "wrong-policy", rank_policy=dataclasses.replace(policy, seed=policy.seed + 1),
                    metadata_validator=callback,
                )
        callback.assert_not_called()
        self.assertIsNone(runtime.store.get_ref("branches/wrong-policy"))

    def test_payload_integrity_still_precedes_deserialization_and_full_validator(self):
        runtime, _, snapshot, state_blob, _ = self.setup_snapshot(verify_reads=True)
        runtime.store.path_for(state_blob).write_bytes(b"corrupt owned fixture")
        preview = Mock(return_value=None)
        full = Mock(side_effect=AssertionError("corrupt native state reached full validator"))
        with patch("reelcache.manager.deserialize_tensors", side_effect=AssertionError("corrupt deserialize")):
            with self.assertRaises(IntegrityError):
                runtime.fork(snapshot, "corrupt", metadata_validator=preview, validator=full)
        preview.assert_called_once()
        full.assert_not_called()
        self.assertIsNone(runtime.store.get_ref("branches/corrupt"))

    def test_fork_new_only_and_final_cas_remain_enforced(self):
        runtime, _, snapshot, state_blob, _ = self.setup_snapshot()
        callback = Mock(side_effect=AssertionError("existing branch reached metadata validator"))
        with self.reject_state_blob(runtime, state_blob):
            with self.assertRaises(StateError):
                runtime.fork(snapshot, "main", metadata_validator=callback)
        callback.assert_not_called()
        before = runtime.branch_info("main")

        def racing_callback(preview):
            runtime.create_branch(preview.branch_id)

        with self.assertRaises(StateError):
            runtime.fork(snapshot, "race", metadata_validator=racing_callback)
        self.assertIsNone(runtime.store.get_ref("branches/race"))
        self.assertEqual(runtime.branch_info("main"), before)
        self.assertEqual(runtime.store.stats()["leases"], 0)
