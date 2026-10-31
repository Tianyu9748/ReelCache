from collections.abc import Mapping
from concurrent.futures import CancelledError
import dataclasses
import gc
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch
import weakref

import torch

from reelcache.calibration import load_rank_policy, load_registry, save_rank_policy, save_registry
from reelcache.codecs import HEVCProvider, encode_tensor
from reelcache.errors import CapacityError, ConfigurationError, IntegrityError, StateError, UnsupportedError
from reelcache.execution import BoundedExecutor
from reelcache.manager import KVRuntime
from reelcache.online_rank import (
    RankAbstention, RankPolicy, RankPredictorProfile, RankPreparation, RankState,
)
from reelcache.policy import select_profile
from reelcache.types import CodecSpec, EncodedTensor, MatrixKey
from tests.core_support import CoreWorkspaceTest, chunk_inputs, metadata, model


def rank_policy(identity, **changes):
    keys = tuple(MatrixKey(layer, target) for layer in range(identity.layers) for target in ("K", "V"))
    fields = dict(
        policy_id="experimental-test", model=identity, geometry=(1, 2, 2),
        source_dtypes={key: identity.dtype_for(key) for key in keys},
        source_fingerprint="a" * 64, geometry_fingerprint="b" * 64,
        history_fingerprint="c" * 64, calibration_fingerprint="d" * 64,
        implementation_fingerprint="e" * 64, energy_target=0.95,
        template_ranks={key: 1 for key in keys},
        allowed_ranks={key: (1, 2, 3, 4) for key in keys},
        codec=CodecSpec("svd", 1), history_max=4, oversample=0,
        source_limit_bytes=4096, workspace_limit_bytes=1 << 20,
        solver_workspace_bytes=0, staging_limit_bytes=4096,
    )
    fields.update(changes)
    return RankPolicy(**fields)


def inputs(identity, *, start=0, rank=1):
    _, contexts = chunk_inputs(identity, start=start)
    values = {}
    for key, context in contexts.items():
        values[key] = torch.diag(torch.tensor(
            [float(key.layer + 2)] * rank + [0.0] * (4 - rank),
        )).to(getattr(torch, context.dtype))
    return values, contexts


class AdaptiveManagerTests(CoreWorkspaceTest):
    def runtime(self, name="runtime", identity=None, **kwargs):
        runtime = KVRuntime(self.workspace / name, model() if identity is None else identity, **kwargs)
        self.addCleanup(runtime.close)
        return runtime

    def seed(self, runtime, policy=None, *, rank=1):
        policy = rank_policy(runtime.model) if policy is None else policy
        runtime.create_branch("main")
        values, contexts = inputs(runtime.model, rank=rank)
        runtime.seal_chunk("main", "c0", values, contexts, rank_policy=policy)
        return policy, values, contexts

    def test_atomic_all_matrix_decisions_share_the_actual_factors(self):
        runtime = self.runtime()
        with patch("reelcache.codecs.factorize_tensor", side_effect=AssertionError("duplicate fit")):
            policy, values, _ = self.seed(runtime)
        head = runtime.branch_info("main")
        chunk = runtime.store.get_manifest(head["chunk_manifests"][0])
        self.assertEqual(chunk["version"], 2)
        self.assertEqual(chunk["rank_policy_identity"], policy.fingerprint)
        self.assertEqual(len(chunk["matrices"]), 2 * runtime.model.layers)
        self.assertEqual(runtime.history_rank_policy("main").to_dict(), policy.to_dict())
        state = runtime.history_rank_state("main")
        self.assertEqual(state.selected_ranks, {key: 1 for key in values})
        self.assertEqual(state.completed_chunks, 1)
        self.assertEqual(state.token_end, 4)
        for key, expected in values.items():
            decisions = runtime.history_rank_decisions("main", key)
            self.assertEqual(len(decisions), 1)
            self.assertEqual(decisions[0].rank, 1)
            self.assertEqual(decisions[0].context.key, key)
            with runtime.acquire_history("main", key) as lease:
                self.assertTrue(torch.equal(lease.tensor, expected))
        self.assertEqual(runtime.store.stats()["leases"], 0)
        runtime.gc()
        pending = list(runtime.store.list_refs().values())
        seen, totals = set(), {"blob": 0, "basis": 0, "manifest": 0}
        while pending:
            object_id = pending.pop()
            if object_id in seen:
                continue
            seen.add(object_id)
            raw = runtime.store.path_for(object_id).read_bytes()
            kind = object_id.split(":")[0]
            totals[kind] += len(raw)
            if kind == "manifest":
                pending.extend(json.loads(raw)["references"])
        stats = runtime.store.stats()
        self.assertEqual(sum(totals.values()), stats["object_bytes"])
        self.assertEqual(totals["basis"], stats["basis_bytes"])
        for entry in chunk["matrices"]:
            record = runtime.store.get_manifest(entry["tensor"])["encoded"]
            logical = len(json.dumps(record["metadata"], sort_keys=True, separators=(",", ":")).encode())
            logical += sum(runtime.store.object_size(oid) for oid in record["payloads"].values())
            logical += runtime.store.object_size(record["basis"])
            self.assertEqual(record["metrics"]["representation_bytes"], logical)

    def test_guard_tiling_is_admitted_persisted_and_checked_against_policy(self):
        identity = model(layers=1, channels=32)
        runtime = self.runtime(identity=identity)
        runtime.create_branch("main")
        _, contexts = chunk_inputs(identity, frames=1, height=1, width=129)
        keys = tuple(contexts)
        policy = rank_policy(
            identity, geometry=(1, 1, 129), tile_rows=2,
            template_ranks={key: 32 for key in keys},
            allowed_ranks={key: (32,) for key in keys},
            source_limit_bytes=1 << 20, staging_limit_bytes=1 << 20,
            workspace_limit_bytes=256 << 10,
        )
        values = {
            key: torch.randn(129, 32, generator=torch.Generator().manual_seed(101))
            for key in keys
        }
        runtime.seal_chunk("main", "c0", values, contexts, rank_policy=policy)
        before = runtime.branch_info("main")
        chunk = runtime.store.get_manifest(before["chunk_manifests"][0])
        for entry in chunk["matrices"]:
            self.assertEqual(entry["binding"]["schema"], "reelcache.prepared_factors.v2")
            self.assertEqual(entry["binding"]["guard_tile_rows"], 2)
            packet = runtime.store.get_manifest(entry["tensor"])["encoded"]
            verification = packet["metrics"]["prepared_validation"]
            self.assertEqual(verification["guard_tile_rows"], 2)
            self.assertLess(verification["workspace_estimate_bytes"], policy.workspace_limit_bytes)
            self.assertEqual(
                verification["projection_relative_residual"],
                entry["decision"]["guard_relative_residual"],
            )
        runtime.close()
        reopened = self.runtime(identity=identity)
        self.assertEqual(reopened.branch_info("main"), before)
        self.assertEqual(reopened.history_rank_decisions("main", keys[0])[0].metrics["tile_rows"], 2)
        entry = chunk["matrices"][0]
        entry["binding"]["guard_tile_rows"] = 1024
        packet = reopened.store.get_manifest(entry["tensor"])
        packet["encoded"]["metrics"]["prepared_binding"] = entry["binding"]
        entry["tensor"] = reopened.store.put_manifest(
            packet, references=list(packet["encoded"]["payloads"].values()) + [packet["encoded"]["basis"]],
        )
        bad_chunk = reopened.store.put_manifest(
            chunk, references=[record["tensor"] for record in chunk["matrices"]] + [chunk["parent_head"]],
        )
        head = reopened.store.get_manifest(before["head_id"])
        bad_head = reopened.store.put_manifest({**head, "chunks": [bad_chunk]}, references=[bad_chunk])
        reopened.store.set_ref("branches/main", bad_head)
        with self.assertRaises(IntegrityError):
            reopened.history_rank_decisions("main", keys[0])

    def test_restart_snapshot_fork_allows_new_ranks_only_under_the_same_policy(self):
        runtime = self.runtime()
        policy, _, _ = self.seed(runtime)
        initial_state = runtime.history_rank_state("main").to_dict()
        rng = torch.Generator().manual_seed(41).get_state()
        snapshot = runtime.snapshot("main", metadata=metadata(), tensors={"rng": rng})
        old_head = runtime.branch_info("main")["head_id"]
        runtime.close()
        reopened = self.runtime()
        loaded = reopened.load_snapshot(snapshot)
        self.assertEqual(loaded.rank_policy.fingerprint, policy.fingerprint)
        self.assertEqual(loaded.rank_state.to_dict(), initial_state)
        with self.assertRaises(ConfigurationError):
            reopened.fork(
                snapshot, "wrong-policy", rank_policy=dataclasses.replace(policy, seed=2),
                validator=lambda state: self.fail("wrong policy reached native validator"),
            )
        self.assertIsNone(reopened.store.get_ref("branches/wrong-policy"))
        restored = reopened.fork(snapshot, "child", rank_policy=policy)
        self.assertEqual(restored.rank_policy.fingerprint, policy.fingerprint)
        self.assertEqual(restored.rank_state.to_dict(), initial_state)
        self.assertTrue(torch.equal(restored.tensors["rng"], rng))
        self.assertEqual(reopened.history_rank_state("child").to_dict(), initial_state)
        values, contexts = inputs(reopened.model, start=4, rank=4)
        before_rng = torch.get_rng_state().clone()
        reopened.seal_chunk("child", "c1", values, contexts, rank_policy=policy)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
        self.assertEqual(reopened.branch_info("main")["head_id"], old_head)
        self.assertEqual(reopened.history_rank_state("main").to_dict(), initial_state)
        for key in values:
            self.assertEqual([spec.rank for spec in reopened.history_specs("child", key)], [1, 4])
        committed_child = reopened.branch_info("child")["head_id"]
        with self.assertRaises(ConfigurationError):
            reopened.restore(snapshot, "child", rank_policy=dataclasses.replace(policy, seed=2))
        self.assertEqual(reopened.branch_info("child")["head_id"], committed_child)
        with self.assertRaises(ConfigurationError):
            reopened.seal_chunk(
                "child", "c2", *inputs(reopened.model, start=8),
                rank_policy=dataclasses.replace(policy, seed=policy.seed + 1),
            )
        self.assertEqual(reopened.history_rank_state("child").completed_chunks, 2)

    def test_adaptive_evidence_is_deterministic_across_identical_snapshot_forks(self):
        identity = model(layers=1)
        runtime = self.runtime(identity=identity)
        policy, _, _ = self.seed(runtime)
        snapshot = runtime.snapshot("main", metadata=metadata())
        values, contexts = inputs(identity, start=4)
        generator = torch.Generator().manual_seed(99)
        values = {key: torch.randn(4, 4, generator=generator) for key in values}
        rng = torch.get_rng_state().clone()
        records = []
        for branch in ("a", "b"):
            runtime.fork(snapshot, branch, rank_policy=policy)
            chunk_id = runtime.seal_chunk(branch, "c1", values, contexts, rank_policy=policy)
            chunk = runtime.store.get_manifest(chunk_id)
            records.append((runtime.history_rank_state(branch).to_dict(), [
                (
                    entry["spec"], entry["binding"]["source_sha256"],
                    entry["binding"]["basis_sha256"], entry["binding"]["coefficients_sha256"],
                    entry["decision"]["rank"], entry["decision"]["prediction_sources"],
                )
                for entry in chunk["matrices"]
            ]))
        self.assertEqual(records[0], records[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_exact_oracle_supplies_its_own_basis_and_replays_its_declared_solver(self):
        identity = model(layers=1)
        runtime = self.runtime(identity=identity)
        policy = rank_policy(identity, solver="exact_oracle")
        with patch("reelcache.codecs.factorize_tensor", side_effect=AssertionError("duplicate fit")):
            self.seed(runtime, policy, rank=2)
        for key in policy.keys:
            decision = runtime.history_rank_decisions("main", key)[0]
            self.assertEqual(decision.solver, "exact_oracle")
            self.assertEqual(decision.exact_rank, 2)
            self.assertEqual(decision.rank, 2)
            self.assertEqual(decision.metrics["decomposition_count"], 1)
            self.assertEqual(decision.metrics["eigh_count"], 1)
            self.assertIsNone(decision.metrics["factorization_seed"])
        runtime.close()
        reopened = self.runtime(identity=identity)
        self.assertEqual(reopened.history_rank_policy("main").fingerprint, policy.fingerprint)

    def test_regime_specific_calibrated_predictions_replay_the_shared_policy_helper(self):
        runtime = self.runtime()
        profiles = {
            f"{regime}:{target}": RankPredictorProfile(
                cell="synthetic", target=target, tau=0.95, history_regime=regime,
                execution_mode="whole_clean_staged", layer_template=(1.0, 1.0),
                detector_baselines={"startup": (1.0, 1.0), "later": (2.0, 2.0)},
                anchor_layers=(1,) if regime == "startup" else (0,),
                rank_margin=0, rank_buckets=(1, 2, 3, 4), policy_kind="anchors",
                uniform_rank=None, training_groups=("calibration-group",),
            )
            for regime in ("startup", "later") for target in ("K", "V")
        }
        policy = rank_policy(
            runtime.model, predictor_profiles=profiles, anchor_mode="staged",
            anchors=tuple(MatrixKey(layer, target) for layer in range(2) for target in ("K", "V")),
        )
        self.seed(runtime, policy)
        runtime.seal_chunk("main", "c1", *inputs(runtime.model, start=4, rank=2), rank_policy=policy)
        for target in ("K", "V"):
            decisions = runtime.history_rank_decisions("main", MatrixKey(1, target))
            self.assertEqual(decisions[-1].predicted_rank, 1.0)
            self.assertEqual(decisions[-1].rank, 2)
            self.assertEqual(decisions[-1].prediction_sources[0]["detector_baseline"], 2.0)
            self.assertEqual(
                decisions[-1].metrics["predictor_profile_fingerprint"],
                profiles[f"later:{target}"].fingerprint,
            )
        runtime.close()
        reopened = self.runtime()
        self.assertEqual(reopened.history_rank_state("main").completed_chunks, 2)

    def test_independent_process_reopens_policy_state_and_forks_without_native_mutation(self):
        identity = model(layers=1)
        runtime = self.runtime(identity=identity)
        policy, _, _ = self.seed(runtime)
        snapshot = runtime.snapshot("main", metadata=metadata())
        runtime.close()
        script = """
import json, sys, torch
from reelcache.manager import KVRuntime
from reelcache.types import MatrixKey, ModelIdentity
identity = ModelIdentity.from_dict(json.loads(sys.argv[2]))
with KVRuntime(sys.argv[1], identity) as runtime:
    original = runtime.branch_info("main")["head_id"]
    policy = runtime.history_rank_policy("main")
    assert policy.fingerprint == sys.argv[4]
    restored = runtime.fork(sys.argv[3], "independent", rank_policy=policy)
    assert restored.rank_state.completed_chunks == 1
    assert restored.rank_state.selected_ranks[MatrixKey(0, "K")] == 1
    with runtime.acquire_history("independent", MatrixKey(0, "K")) as lease:
        assert torch.equal(lease.tensor, torch.diag(torch.tensor([2., 0., 0., 0.])))
    assert runtime.branch_info("main")["head_id"] == original
print("adaptive independent restart passed")
"""
        environment = dict(os.environ)
        environment.update(
            CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, str(self.workspace / "runtime"),
             json.dumps(identity.to_dict()), snapshot, policy.fingerprint],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
            timeout=600, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertIn(b"adaptive independent restart passed", result.stdout)

    def test_source_domain_and_effective_dtype_never_switch_to_native_raw(self):
        identity = model(dtype="bfloat16")
        runtime = self.runtime(identity=identity)
        keys = tuple(MatrixKey(layer, target) for layer in range(identity.layers) for target in ("K", "V"))
        dtypes = {key: "float32" if key.target == "K" else "bfloat16" for key in keys}
        policy = rank_policy(identity, source_dtypes=dtypes)
        runtime.create_branch("main")
        values, contexts = inputs(identity)
        for key in keys:
            contexts[key] = dataclasses.replace(contexts[key], source_dtype=dtypes[key])
            values[key] = values[key].to(getattr(torch, dtypes[key]))
        runtime.seal_chunk("main", "c0", values, contexts, rank_policy=policy)
        snapshot = runtime.snapshot("main", metadata=metadata())
        runtime.close()
        reopened = self.runtime(identity=identity)
        reopened.fork(snapshot, "child")
        old_head = reopened.branch_info("child")["head_id"]
        raw_values, raw_contexts = chunk_inputs(identity, start=4, key_space="post_rope")
        for kwargs in ({}, {"rank_policy": policy}):
            with self.assertRaises(ConfigurationError):
                reopened.seal_chunk("child", "c1", raw_values, raw_contexts, **kwargs)
        self.assertEqual(reopened.branch_info("child")["head_id"], old_head)
        self.assertEqual(reopened.history_contexts("child", MatrixKey(0, "K"))[0].dtype, "float32")

    def test_static_history_cannot_be_relabeled_as_adaptive(self):
        runtime = self.runtime()
        runtime.create_branch("main")
        runtime.seal_chunk("main", "c0", *inputs(runtime.model), CodecSpec("svd", 1))
        old_head = runtime.branch_info("main")["head_id"]
        self.assertIsNone(runtime.history_rank_policy("main"))
        self.assertIsNone(runtime.history_rank_state("main"))
        with self.assertRaises(ConfigurationError):
            runtime.seal_chunk(
                "main", "c1", *inputs(runtime.model, start=4),
                rank_policy=rank_policy(runtime.model),
            )
        self.assertEqual(runtime.branch_info("main")["head_id"], old_head)

    def test_policy_and_full_matrix_contract_fail_before_loading_any_source(self):
        runtime = self.runtime()
        runtime.create_branch("main")
        policy = rank_policy(runtime.model)
        values, contexts = inputs(runtime.model)
        old_head = runtime.branch_info("main")["head_id"]
        for changes in (
            {"contexts": {key: context for key, context in contexts.items() if key != MatrixKey(1, "V")}},
            {"rank_policy": dataclasses.replace(policy, model=model(fingerprint="wrong"))},
            {"specs": CodecSpec("svd", 1)},
            {"reuse_threshold": 0.1},
        ):
            kwargs = {"contexts": contexts, "rank_policy": policy, **changes}
            with self.subTest(changes=tuple(changes)), self.assertRaises(ConfigurationError):
                runtime.seal_chunk("main", "bad", values, **kwargs)
        self.assertEqual(runtime.branch_info("main")["head_id"], old_head)
        with self.assertRaises(ConfigurationError):
            rank_policy(runtime.model, allowed_ranks={key: (1, 5) for key in contexts})

    def test_lazy_staged_anchors_load_every_source_once_and_release_every_packet(self):
        tensor_refs, packet_refs, loaded = {}, [], []

        def encoder(*args, **kwargs):
            result = encode_tensor(*args, **kwargs)
            packet_refs.append(weakref.ref(result))
            return result

        runtime = self.runtime(encoder=encoder)
        runtime.create_branch("main")
        _, contexts = inputs(runtime.model)
        keys = tuple(contexts)
        policy = rank_policy(
            runtime.model, anchor_mode="staged", anchors=(MatrixKey(1, "K"), MatrixKey(1, "V")),
        )

        class LazySources(Mapping):
            def __len__(self):
                return len(keys)

            def __iter__(self):
                return iter(keys)

            def __getitem__(self, key):
                if key in loaded:
                    raise AssertionError("source was read twice")
                loaded.append(key)
                value = torch.diag(torch.tensor([5.0, 0, 0, 0]))
                tensor_refs[key] = weakref.ref(value)
                return value

        with patch("reelcache.codecs.factorize_tensor", side_effect=AssertionError("duplicate fit")):
            runtime.seal_chunk("main", "c0", LazySources(), contexts, rank_policy=policy)
        gc.collect()
        self.assertEqual(loaded, [MatrixKey(1, "K"), MatrixKey(1, "V"), MatrixKey(0, "K"), MatrixKey(0, "V")])
        self.assertTrue(all(reference() is None for reference in tensor_refs.values()))
        self.assertTrue(all(reference() is None for reference in packet_refs))
        for key in keys:
            self.assertEqual(runtime.history_rank_decisions("main", key)[0].metrics["decomposition_count"], 1)

    def test_missing_lazy_source_failure_reclaims_packets_without_advancing_state(self):
        runtime = self.runtime()
        policy, _, _ = self.seed(runtime)
        before_head = runtime.branch_info("main")["head_id"]
        before_state = runtime.history_rank_state("main").to_dict()
        before_bytes = runtime.store.stats()["object_bytes"]
        values, contexts = inputs(runtime.model, start=4, rank=4)

        class MissingSource(Mapping):
            def __len__(self):
                return len(values)

            def __iter__(self):
                return iter(values)

            def __getitem__(self, key):
                if key == MatrixKey(1, "V"):
                    raise KeyError(key)
                return values[key]

        with self.assertRaisesRegex(ConfigurationError, "disappeared"):
            runtime.seal_chunk("main", "failed", MissingSource(), contexts, rank_policy=policy)
        self.assertEqual(runtime.branch_info("main")["head_id"], before_head)
        self.assertEqual(runtime.history_rank_state("main").to_dict(), before_state)
        self.assertEqual(runtime.store.stats()["object_bytes"], before_bytes)
        self.assertEqual(runtime.store.stats()["leases"], 0)

    def test_codec_failure_and_forged_source_are_atomic(self):
        enabled = False

        def encoder(tensor, context, spec, **kwargs):
            if enabled and context.key == MatrixKey(1, "K"):
                return encode_tensor(tensor + 1, context, spec, **kwargs)
            return encode_tensor(tensor, context, spec, **kwargs)

        runtime = self.runtime(encoder=encoder)
        policy, _, _ = self.seed(runtime)
        before = runtime.branch_info("main")
        old_state = runtime.history_rank_state("main").to_dict()
        old_bytes = runtime.store.stats()["object_bytes"]
        enabled = True
        state = torch.get_rng_state().clone()
        with self.assertRaises(IntegrityError):
            runtime.seal_chunk("main", "failed", *inputs(runtime.model, start=4, rank=4), rank_policy=policy)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        self.assertEqual(runtime.branch_info("main"), before)
        self.assertEqual(runtime.history_rank_state("main").to_dict(), old_state)
        self.assertEqual(runtime.store.stats()["object_bytes"], old_bytes)

    def test_refit_failure_after_staging_survives_gc_and_preserves_committed_history(self):
        from reelcache.online_rank import fit_rank_directions
        runtime = self.runtime()
        policy, _, _ = self.seed(runtime)
        reader = self.runtime()
        old_head = reader.branch_info("main")["head_id"]
        old_state = reader.history_rank_state("main").to_dict()
        old_bytes = reader.store.stats()["object_bytes"]
        rng = torch.get_rng_state().clone()
        calls = 0

        def fit(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                reader.gc()
                self.assertEqual(reader.branch_info("main")["head_id"], old_head)
            if calls == 4:
                raise RuntimeError("injected rank refit failure")
            return fit_rank_directions(*args, **kwargs)

        with patch("reelcache.online_rank.fit_rank_directions", fit), \
                self.assertRaisesRegex(RuntimeError, "refit failure"):
            runtime.seal_chunk("main", "failed", *inputs(runtime.model, start=4, rank=4), rank_policy=policy)
        self.assertEqual(calls, 4)
        self.assertEqual(runtime.branch_info("main")["head_id"], old_head)
        self.assertEqual(runtime.history_rank_state("main").to_dict(), old_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(runtime.store.stats()["object_bytes"], old_bytes)
        self.assertEqual(runtime.store.stats()["leases"], 0)

    def test_forged_decision_is_rejected_before_provider_execution(self):
        runtime = self.runtime()
        runtime.create_branch("main")
        policy = rank_policy(runtime.model)
        original = RankPreparation.prepare_matrix

        def forged(preparation, *args, **kwargs):
            prepared = original(preparation, *args, **kwargs)
            decision = dataclasses.replace(prepared.decision, input_sha256="0" * 64)
            return dataclasses.replace(prepared, decision=decision)

        with patch.object(RankPreparation, "prepare_matrix", forged), \
                patch.object(runtime, "encoder", side_effect=AssertionError("must not encode")):
            with self.assertRaises(IntegrityError):
                runtime.seal_chunk("main", "bad", *inputs(runtime.model), rank_policy=policy)
        self.assertEqual(runtime.branch_info("main")["chunk_ids"], [])

    def test_unresolved_and_over_budget_attempts_leave_empty_history_and_rng_intact(self):
        for name, changes in (
            ("capacity", {"workspace_limit_bytes": 1}),
            ("precision", {"energy_target": 1.0}),
        ):
            runtime = self.runtime(name=name)
            runtime.create_branch("main")
            policy = rank_policy(runtime.model, **changes)
            values, contexts = inputs(runtime.model)
            if name == "precision":
                generator = torch.Generator().manual_seed(17)
                values = {key: torch.randn(4, 4, generator=generator) for key in contexts}
            rng = torch.get_rng_state().clone()
            before = runtime.branch_info("main")["head_id"]
            expected_error = CapacityError if name == "capacity" else RankAbstention
            with self.assertRaises(expected_error):
                runtime.seal_chunk("main", "unresolved", values, contexts, rank_policy=policy)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(runtime.branch_info("main")["head_id"], before)
            self.assertIsNone(runtime.history_rank_state("main"))
            self.assertEqual(runtime.store.stats()["leases"], 0)

    def test_executor_charges_source_basis_and_coefficients_and_cancel_is_atomic(self):
        def cancelled(*args, **kwargs):
            def result():
                raise CancelledError()
            return SimpleNamespace(result=result)

        runtime = self.runtime(name="cancel", executor=SimpleNamespace(submit=cancelled))
        runtime.create_branch("main")
        policy = rank_policy(runtime.model)
        with self.assertRaises(CancelledError):
            runtime.seal_chunk("main", "cancel", *inputs(runtime.model), rank_policy=policy)
        self.assertEqual(runtime.branch_info("main")["chunk_ids"], [])
        with BoundedExecutor(max_staging_bytes=64) as executor:
            runtime = self.runtime(name="capacity", executor=executor)
            runtime.create_branch("main")
            with self.assertRaises(CapacityError):
                runtime.seal_chunk("main", "too-big", *inputs(runtime.model), rank_policy=policy)
            self.assertEqual(executor.stats()["submitted"], 0)
            self.assertEqual(runtime.store.stats()["leases"], 0)
        with BoundedExecutor(max_staging_bytes=1024) as executor:
            runtime = self.runtime(name="executor", executor=executor)
            self.seed(runtime)
            self.assertGreater(executor.stats()["peak_staging_bytes"], 64)
            self.assertEqual(executor.stats()["succeeded"], 4)
            self.assertEqual(executor.stats()["staging_bytes"], 0)

    def test_real_queued_executor_cancellation_releases_all_input_staging(self):
        proceed = threading.Event()
        with BoundedExecutor(max_workers=1, max_pending=2, max_staging_bytes=1024) as executor:
            blocker = executor.submit(proceed.wait, 60)

            def submit(*args, **kwargs):
                job = executor.submit(*args, **kwargs)
                self.assertTrue(job.cancel())
                return job

            runtime = self.runtime(executor=SimpleNamespace(submit=submit))
            runtime.create_branch("main")
            try:
                with self.assertRaises(CancelledError):
                    runtime.seal_chunk(
                        "main", "cancelled", *inputs(runtime.model),
                        rank_policy=rank_policy(runtime.model),
                    )
            finally:
                proceed.set()
                blocker.result()
            self.assertEqual(executor.stats()["cancelled"], 1)
            self.assertEqual(executor.stats()["staging_bytes"], 0)
            self.assertEqual(runtime.branch_info("main")["chunk_ids"], [])
            self.assertIsNone(runtime.history_rank_state("main"))
            self.assertEqual(runtime.store.stats()["leases"], 0)

    def test_interrupted_running_job_is_drained_before_sources_are_released(self):
        calls = []

        class InterruptedJob:
            def __init__(self, function, args, kwargs):
                self.function, self.args, self.kwargs = function, args, kwargs
                self.completed = False
                self.interrupted = False

            def done(self):
                return self.completed

            def cancel(self):
                calls.append("cancel")
                return False

            def result(self):
                if not self.interrupted:
                    self.interrupted = True
                    raise KeyboardInterrupt()
                function, args, kwargs = self.function, self.args, self.kwargs
                self.function, self.args, self.kwargs = None, (), {}
                calls.append("drained")
                self.completed = True
                return function(*args, **kwargs)

        def submit(function, *args, **kwargs):
            kwargs.pop("staging_bytes")
            kwargs.pop("label")
            return InterruptedJob(function, args, kwargs)

        runtime = self.runtime(executor=SimpleNamespace(submit=submit))
        runtime.create_branch("main")
        with self.assertRaises(KeyboardInterrupt):
            runtime.seal_chunk(
                "main", "interrupted", *inputs(runtime.model), rank_policy=rank_policy(runtime.model),
            )
        self.assertEqual(calls, ["cancel", "drained"])
        self.assertEqual(runtime.branch_info("main")["chunk_ids"], [])
        self.assertEqual(runtime.store.stats()["leases"], 0)

    def test_real_adaptive_hevc_and_unverifiable_provider_rejection(self):
        identity = model(layers=1)
        runtime = self.runtime(identity=identity)
        policy = rank_policy(identity, codec=CodecSpec("svd_hevc", 1))
        _, values, _ = self.seed(runtime, policy)
        for key in values:
            with runtime.acquire_history("main", key) as lease:
                self.assertTrue(torch.isfinite(lease.tensor).all())
                self.assertEqual(lease.tensor.shape, values[key].shape)
        head = runtime.branch_info("main")
        chunk = runtime.store.get_manifest(head["chunk_manifests"][0])
        for entry in chunk["matrices"]:
            packet = runtime.store.get_manifest(entry["tensor"])["encoded"]
            self.assertEqual(packet["metadata"]["video"]["pix_fmt"], "yuv444p")
            self.assertEqual(packet["metadata"]["codec"]["qp"], 8)
            validation = packet["metrics"]["runtime_prepared_packet_validation"]
            self.assertEqual(validation["hevc_validation_decode_count"], 1)
            self.assertGreaterEqual(validation["post_hevc_relative_residual"], 0)
        wrapper = lambda *args, **kwargs: encode_tensor(*args, **kwargs)
        other = self.runtime(name="unverified", identity=identity, encoder=wrapper)
        other.create_branch("main")
        with self.assertRaises(UnsupportedError):
            other.seal_chunk("main", "bad", *inputs(identity), rank_policy=policy)
        self.assertEqual(other.branch_info("main")["chunk_ids"], [])
        provider = HEVCProvider()
        exact = self.runtime(
            name="bound-provider", identity=identity,
            encoder=provider.encode_tensor, decoder=provider.decode_tensor,
        )
        self.seed(exact, policy)

    def test_stale_head_publication_cannot_commit_its_candidate_state(self):
        started, proceed = threading.Event(), threading.Event()
        blocked = False

        def encoder(*args, **kwargs):
            if blocked and args[1].key == MatrixKey(0, "K"):
                started.set()
                if not proceed.wait(60):
                    raise RuntimeError("test synchronization timeout")
            return encode_tensor(*args, **kwargs)

        first = self.runtime(encoder=encoder)
        policy, _, _ = self.seed(first)
        second = self.runtime()
        old_state = second.history_rank_state("main").to_dict()
        failures = []
        blocked = True

        def seal():
            try:
                first.seal_chunk("main", "loser", *inputs(first.model, start=4, rank=2), rank_policy=policy)
            except StateError as exc:
                failures.append(exc)

        thread = threading.Thread(target=seal)
        thread.start()
        self.assertTrue(started.wait(60))
        try:
            self.assertEqual(second.history_rank_state("main").to_dict(), old_state)
            second.gc()
            second.seal_chunk("main", "winner", *inputs(second.model, start=4, rank=4), rank_policy=policy)
        finally:
            proceed.set()
            thread.join(60)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(failures), 1)
        self.assertEqual(second.branch_info("main")["chunk_ids"], ["c0", "winner"])
        state = second.history_rank_state("main")
        self.assertEqual(state.last_chunk_id, "winner")
        self.assertTrue(all(rank == 4 for rank in state.selected_ranks.values()))
        self.assertEqual(second.store.stats()["leases"], 0)

    def test_state_is_immutable_and_policy_registry_is_separate_from_static_profiles(self):
        runtime = self.runtime()
        policy, _, _ = self.seed(runtime)
        state = runtime.history_rank_state("main")
        with self.assertRaises(TypeError):
            state.selected_ranks[MatrixKey(0, "K")] = 4
        restored = RankState.from_dict(state.to_dict())
        self.assertEqual(restored.to_dict(), state.to_dict())
        path = self.workspace / "policy.json"
        digest = save_rank_policy(path, policy)
        self.assertEqual(save_rank_policy(path, policy), digest)
        self.assertEqual(load_rank_policy(path, expected_sha256=digest).fingerprint, policy.fingerprint)
        with self.assertRaises(IntegrityError):
            load_registry(path)
        with self.assertRaises(ConfigurationError):
            save_registry(self.workspace / "static.json", [policy])
        with self.assertRaises(ConfigurationError):
            select_profile(
                [policy], model=runtime.model, geometry=policy.geometry,
                history_chunks=1, features={}, raw_bytes=256,
            )
        with self.assertRaises(StateError):
            save_rank_policy(path, dataclasses.replace(policy, seed=2))
        payload = json.loads(path.read_text())
        payload["policy"]["seed"] += 1
        path.write_text(json.dumps(payload))
        with self.assertRaises(IntegrityError):
            load_rank_policy(path)

    def test_partial_or_inconsistent_adaptive_metadata_is_rejected_on_restart(self):
        mutations = {
            "policy": lambda chunk: chunk["rank_policy"].pop("source_fingerprint"),
            "identity": lambda chunk: chunk.update(rank_policy_identity="0" * 64),
            "state": lambda chunk: chunk["rank_state"]["state"]["selected_ranks"].pop("1:V"),
            "state-rank": lambda chunk: chunk["rank_state"]["state"]["selected_ranks"].update({"0:K": 4}),
            "state-bool": lambda chunk: chunk["rank_state"].update(committed_chunks=True),
            "decision": lambda chunk: chunk["matrices"][0]["decision"].update(input_sha256="0" * 64),
            "spec": lambda chunk: chunk["matrices"][0]["spec"].update(normalization="global"),
            "binding": lambda chunk: chunk["matrices"][0]["binding"].pop("source_sha256"),
            "partial": lambda chunk: chunk["matrices"][0].pop("decision"),
            "metrics-type": lambda chunk: chunk["matrices"][0].update(factor_metrics=[]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                runtime = self.runtime(name=name)
                self.seed(runtime)
                info = runtime.branch_info("main")
                old_head = runtime.store.get_manifest(info["head_id"])
                chunk = runtime.store.get_manifest(info["chunk_manifests"][0])
                mutate(chunk)
                bad_chunk = runtime.store.put_manifest(
                    chunk, references=[entry["tensor"] for entry in chunk["matrices"]] + [chunk["parent_head"]],
                )
                bad_head = runtime.store.put_manifest(
                    {**old_head, "chunks": [bad_chunk]}, references=[bad_chunk],
                )
                runtime.store.set_ref("branches/main", bad_head)
                runtime.close()
                with self.assertRaises(IntegrityError):
                    self.runtime(name=name)
