import copy
import json
from collections.abc import Mapping
from unittest.mock import patch

import torch

from reelcache.codecs import encode_tensor
from reelcache.errors import ConfigurationError, IntegrityError
from reelcache.manager import KVRuntime, _rank_digest
from reelcache.online_rank import RankAbstention
from reelcache.spectra import SpectrumNumericalError, SpectrumResult
from tests.core_support import CoreWorkspaceTest, metadata, model
from tests.test_adaptive_manager import inputs, rank_policy


def provenance_for(contexts):
    return {
        key: {
            "case_id": "paired-case", "prompt_group": "calibration-group",
            "split": "calibration", "seed": 7, "cell": "core-provenance-fixture",
            "layer": key.layer, "target": key.target, "boundary": 0,
            "trajectory_id": "exact-arm", "evidence_kind": "synthetic_fixture",
            "n_rows": context.token_count, "n_channels": context.channels,
            "n_layers": context.model.layers, "source_dtype": context.dtype,
            "source_domain": "normalized_pre_rope" if key.target == "K" else "direct",
            "tensor_context": context.to_dict(),
            "native_capture": {"request_id": "arm-specific-not-case-id"},
        }
        for key, context in contexts.items()
    }


class AdaptiveProvenanceTests(CoreWorkspaceTest):
    def runtime(self, **kwargs):
        runtime = KVRuntime(self.workspace / "runtime", model(layers=1), **kwargs)
        self.addCleanup(runtime.close)
        return runtime

    def test_same_bound_oracle_is_durable_and_exported_without_another_decomposition(self):
        from reelcache import rank_analysis
        from reelcache.study import StudyCell

        provenance = {}

        def encoder(tensor, context, spec, **kwargs):
            provenance[context.key]["case_id"] = "caller-mutated"
            provenance[context.key]["native_capture"]["request_id"] = "caller-mutated"
            return encode_tensor(tensor, context, spec, **kwargs)

        runtime = self.runtime(encoder=encoder)
        runtime.create_branch("main")
        policy = rank_policy(runtime.model, solver="exact_oracle")
        values, context_map = inputs(runtime.model, rank=2)
        provenance.update(provenance_for(context_map))
        cell = StudyCell(
            "core-provenance-fixture", "synthetic", 4, 4, 4, 0, 1, 2, 2,
            1, 4, "float32", "float32", 1,
        )
        with patch.object(rank_analysis, "CELLS", {**rank_analysis.CELLS, cell.cell_id: cell}):
            for key in policy.keys:
                provenance[key].update(rank_analysis.rank_observation_identity(
                    case_id="paired-case", group_id="calibration-group", split="calibration",
                    seed=7, cell_id=cell.cell_id, layer=key.layer, target=key.target,
                    boundary=0, trajectory_id="exact-arm", evidence_kind="synthetic_fixture",
                    source_bindings={"branch_id": "main", "parent_head": None, "chunk_id": "c0"},
                ))
        original = copy.deepcopy(provenance)
        rng = torch.get_rng_state().clone()
        with patch("torch.linalg.eigh", wraps=torch.linalg.eigh) as eigh, \
                patch("torch.linalg.svd", side_effect=AssertionError("duplicate SVD")):
            runtime.seal_chunk(
                "main", "c0", values, context_map, rank_policy=policy,
                source_provenance=provenance,
            )
        self.assertEqual(eigh.call_count, len(policy.keys))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        runtime.close()
        reopened = self.runtime()
        with patch.object(rank_analysis, "CELLS", {**rank_analysis.CELLS, cell.cell_id: cell}), \
                patch("torch.linalg.eigh", side_effect=AssertionError("writer duplicate eigh")), \
                patch("torch.linalg.svd", side_effect=AssertionError("writer duplicate SVD")):
            snapshot = reopened.snapshot("main", metadata=metadata(), tensors={"rng": rng})
            reopened.fork(snapshot, "child", rank_policy=policy)
            for key in policy.keys:
                decision = reopened.history_rank_decisions("child", key)[0]
                self.assertEqual(decision.source_provenance["case_id"], "paired-case")
                self.assertEqual(decision.source_provenance["branch_id"], "main")
                self.assertEqual(
                    decision.source_provenance["native_capture"]["request_id"],
                    "arm-specific-not-case-id",
                )
                observation = decision.to_dict()["metrics"]["oracle_observation"]
                with self.assertRaises(TypeError):
                    decision.oracle_observation["source"]["provenance"]["case_id"] = "rewritten"
                detached = decision.to_dict()
                detached["metrics"]["oracle_observation"]["source"]["provenance"]["case_id"] = "rewritten"
                self.assertEqual(
                    reopened.history_rank_decisions("child", key)[0].to_dict()["metrics"]["oracle_observation"],
                    observation,
                )
                numerical = SpectrumResult.from_dict(observation)
                self.assertEqual(numerical.source.provenance["case_id"], "paired-case")
                self.assertEqual(
                    numerical.source.provenance["original_bytes_sha256"], decision.input_sha256,
                )
                self.assertNotEqual(numerical.source.tensor_sha256, decision.input_sha256)
                spectrum_path = self.workspace / f"{key.layer}-{key.target}.spectrum.json"
                observation_path = self.workspace / f"{key.layer}-{key.target}.observation.json"
                receipt = rank_analysis.write_spectrum_observation(
                    observation, identity=original[key],
                    spectrum_path=spectrum_path, observation_path=observation_path,
                )
                self.assertTrue(observation_path.is_file())
                self.assertEqual(receipt["spectrum_sha256"], rank_analysis.file_sha256(spectrum_path))
                self.assertEqual(json.loads(spectrum_path.read_text()), observation)

    def test_invalid_provenance_fails_before_source_access_or_pending_state(self):
        runtime = self.runtime()
        runtime.create_branch("main")
        policy = rank_policy(runtime.model)
        values, contexts = inputs(runtime.model)
        first = policy.keys[0]
        before = runtime.branch_info("main")
        rng = torch.get_rng_state().clone()

        class UnreadableSources(Mapping):
            def __len__(self):
                return len(values)

            def __iter__(self):
                return iter(values)

            def __getitem__(self, key):
                raise AssertionError("invalid provenance accessed source values")

        mutations = (
            lambda rows: rows.pop(first),
            lambda rows: rows[first].update(case_id="mixed-case"),
            lambda rows: rows[first].update(source_dtype="bfloat16"),
            lambda rows: rows[first].update(branch_id="another-branch"),
            lambda rows: rows[first].pop("trajectory_id"),
        )
        for mutate in mutations:
            declared = provenance_for(contexts)
            mutate(declared)
            with self.assertRaises(ConfigurationError):
                runtime.seal_chunk(
                    "main", "bad", UnreadableSources(), contexts, rank_policy=policy,
                    source_provenance=declared,
                )
            self.assertEqual(runtime.branch_info("main"), before)
            self.assertEqual(runtime.store.stats()["leases"], 0)
        with self.assertRaises(ConfigurationError):
            runtime.seal_chunk(
                "main", "static", UnreadableSources(), contexts,
                source_provenance=provenance_for(contexts),
            )
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_rehashed_mixed_case_records_are_rejected_during_restart(self):
        runtime = self.runtime()
        runtime.create_branch("main")
        policy = rank_policy(runtime.model)
        values, contexts = inputs(runtime.model)
        runtime.seal_chunk(
            "main", "c0", values, contexts, rank_policy=policy,
            source_provenance=provenance_for(contexts),
        )
        before = runtime.branch_info("main")
        head = runtime.store.get_manifest(before["head_id"])
        chunk = runtime.store.get_manifest(before["chunk_manifests"][0])
        entry = chunk["matrices"][0]
        decision = entry["decision"]
        decision["source_provenance"]["case_id"] = "another-valid-case"
        decision["metrics"]["source_provenance_sha256"] = _rank_digest(decision["source_provenance"])
        entry["factor_metrics"] = copy.deepcopy(decision["metrics"])
        entry["binding"]["metrics_sha256"] = _rank_digest(entry["factor_metrics"])
        packet = runtime.store.get_manifest(entry["tensor"])
        packet["encoded"]["metrics"].update(copy.deepcopy(entry["factor_metrics"]))
        packet["encoded"]["metrics"]["prepared_binding"] = copy.deepcopy(entry["binding"])
        packet_refs = list(packet["encoded"]["payloads"].values()) + [packet["encoded"]["basis"]]
        entry["tensor"] = runtime.store.put_manifest(packet, references=packet_refs)
        chunk["rank_state"]["decisions_sha256"] = _rank_digest(
            [record["decision"] for record in chunk["matrices"]],
        )
        bad_chunk = runtime.store.put_manifest(
            chunk, references=[record["tensor"] for record in chunk["matrices"]] + [chunk["parent_head"]],
        )
        bad_head = runtime.store.put_manifest({**head, "chunks": [bad_chunk]}, references=[bad_chunk])
        runtime.store.set_ref("branches/main", bad_head)
        runtime.close()
        with self.assertRaises(IntegrityError):
            self.runtime()

    def test_failed_oracle_counts_remain_unknown_and_never_advance_history(self):
        from reelcache.spectra import exact_spectrum

        runtime = self.runtime()
        runtime.create_branch("main")
        policy = rank_policy(runtime.model, solver="exact_oracle")
        runtime.seal_chunk("main", "c0", *inputs(runtime.model), rank_policy=policy)
        before = runtime.branch_info("main")
        state = runtime.history_rank_state("main").to_dict()
        rng = torch.get_rng_state().clone()
        for after_solve in (False, True):
            def fail(*args, **kwargs):
                if after_solve:
                    exact_spectrum(*args, **kwargs)
                raise SpectrumNumericalError("injected numerics without stage telemetry")

            with patch("reelcache.spectra.exact_spectrum", fail), self.assertRaises(RankAbstention) as failed:
                runtime.seal_chunk(
                    "main", "failed", *inputs(runtime.model, start=4, rank=2), rank_policy=policy,
                )
            metrics = failed.exception.decision.metrics
            self.assertFalse(metrics["operation_counts_complete"])
            self.assertIsNone(metrics["decomposition_count"])
            self.assertIsNone(metrics["eigh_count"])
            self.assertIsNone(metrics["oracle_failure_metrics"])
            self.assertGreaterEqual(metrics["oracle_seconds"], 0)
            self.assertEqual(runtime.branch_info("main"), before)
            self.assertEqual(runtime.history_rank_state("main").to_dict(), state)
            self.assertEqual(runtime.store.stats()["leases"], 0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
