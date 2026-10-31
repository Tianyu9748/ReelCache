"""Step 6: stream-labeled node types, speculative registry views and batched re-orthogonalization."""

from __future__ import annotations

import hashlib
import unittest

import torch

from reelcache.basis_versions import rows_to_bytes
from reelcache.errors import ConfigurationError, IntegrityError, StateError
from reelcache.predictor_registry import RankPredictorRegistry
from reelcache.rank_method import LatestMethodPlanner, RankMethodConfig
from reelcache.types import (
    ActiveBasisBinding, BasisVersion, ChunkNode, CodecSpec, EnergyReport, EnergyTargets, FitObservation, MatrixKey,
    MatrixRecord, ModelIdentity, PredictorKey, TensorContext, TokenRangeRecords,
)

MODEL = ModelIdentity("step6-types", "step6-types-source", 2, 8, "tokens_channels", "bfloat16")
REF = "manifest:" + "1" * 64
REF2 = "manifest:" + "2" * 64
KEYS = [MatrixKey(layer, target) for layer in range(2) for target in ("K", "V")]


class StreamTypeTests(unittest.TestCase):
    def node(self, streams=(), records=None, bindings=None):
        entries = KEYS if not streams else [(key, stream) for key in KEYS for stream in streams]
        return ChunkNode("chunk", "c0", REF, "e0", REF2, 0, 6, 3, 1, 2, 6,
                         records if records is not None else {entry: REF for entry in entries},
                         bindings if bindings is not None else {entry: ActiveBasisBinding() for entry in entries},
                         None, streams)

    def test_single_stream_nodes_keep_their_v1_bytes(self):
        node = self.node()
        data = node.to_dict()
        self.assertEqual(data["schema"], "reelcache.chunk_node.v1")
        self.assertNotIn("streams", data)
        self.assertTrue(all(set(item) == {"key", "record"} for item in data["matrices"]))
        self.assertEqual(ChunkNode.from_dict(data), node)

    def test_cfg_streams_round_trip_and_are_complete(self):
        node = self.node(("cond", "uncond"))
        data = node.to_dict()
        self.assertEqual(data["schema"], "reelcache.chunk_node.v2")
        self.assertEqual(ChunkNode.from_dict(data), node)
        self.assertEqual(node.record_ref(KEYS[0], "uncond"), REF)
        self.assertIsNone(node.record_ref(KEYS[0]))
        partial = {(key, "cond"): REF for key in KEYS}
        partial.update({(key, "uncond"): REF for key in KEYS[:-1]})
        with self.assertRaises(ConfigurationError):
            self.node(("cond", "uncond"), records=partial)
        with self.assertRaises(ConfigurationError):
            self.node((), records={(KEYS[0], "cond"): REF})
        with self.assertRaises(ConfigurationError):
            self.node(("uncond", "cond"))
        data["schema"] = "reelcache.chunk_node.v1"
        with self.assertRaises(ConfigurationError):
            ChunkNode.from_dict(data)

    def test_range_records_and_matrix_records_optional_fields(self):
        streams = ("cond", "uncond")
        item = TokenRangeRecords("c0", 0, 6, 3, 1, 2, {(key, s): REF for key in KEYS for s in streams}, streams)
        self.assertEqual(TokenRangeRecords.from_dict(item.to_dict()), item)
        plain = TokenRangeRecords("c0", 0, 6, 3, 1, 2, {key: REF for key in KEYS})
        self.assertNotIn("streams", plain.to_dict())
        context = TensorContext(MODEL, KEYS[0], 0, 3, 1, 2, "pre_rope")
        spec = CodecSpec(kind="svd_hevc", rank=2, inter_prediction=False)
        energy = EnergyReport(0.9, 0.95, 0.94, None, 2, 32, 2)
        record = MatrixRecord(context, spec, REF, REF2, 2, energy, None, {"decision": "fresh"})
        self.assertNotIn("packet_tile", record.to_dict())
        self.assertNotIn("stream", record.to_dict())
        packed = MatrixRecord(context, spec, REF, REF2, 2, energy, None, {"decision": "fresh"}, packet_tile=3,
                              stream="cond")
        self.assertEqual(MatrixRecord.from_dict(packed.to_dict()), packed)
        self.assertEqual(packed.entry, (KEYS[0], "cond"))
        with self.assertRaises(ConfigurationError):
            MatrixRecord(context, CodecSpec(kind="svd", rank=2), REF, REF2, 2, energy, None, {}, packet_tile=0)


def observation(index, key=None, rank=64):
    key = key or PredictorKey("fp", 0, "K", 144, 64, "fp32_gram_eigh.v1", "det", "fresh")
    return FitObservation(key, hashlib.sha256(f"source-{index}".encode()).hexdigest(), 0.5, 0.98,
                          1.0 + index / 10, rank, 40.0, 0)


class SpeculativeRegistryTests(unittest.TestCase):
    def test_pin_with_equals_the_sequential_registry_and_mutates_nothing(self):
        shared = RankPredictorRegistry("r")
        sequential = RankPredictorRegistry("r")
        pending = [observation(index, rank=40 + index) for index in range(3)]
        view = shared.pin_with(pending)
        self.assertEqual(shared.sequence, 0)
        self.assertEqual(len(view.speculative), 3)
        sequential.observe(pending)
        self.assertEqual(view.digest, sequential.pin().digest)
        key = pending[0].key
        self.assertEqual(view.predict(key, 2.0).revision_ref, sequential.pin().predict(key, 2.0).revision_ref)
        shared.observe(pending[:2])
        later = shared.pin_with(pending)
        self.assertEqual(later.speculative, (pending[2].observation_id,))
        self.assertEqual(later.digest, view.digest)

    def test_conflicting_or_out_of_order_pending_observations_are_rejected(self):
        shared = RankPredictorRegistry("r")
        shared.observe([observation(0)])
        conflicting = FitObservation(observation(0).key, observation(0).source_id, 0.5, 0.98, 9.0, 99, 40.0, 0)
        with self.assertRaises(IntegrityError):
            shared.pin_with([conflicting])
        replay = RankPredictorRegistry("q", mode="replay",
                                       expected_order=[observation(1).observation_id, observation(2).observation_id])
        replay.pin_with([observation(1), observation(2)])
        with self.assertRaises(StateError):
            replay.pin_with([observation(2)])


class BatchedOrthogonalizationTests(unittest.TestCase):
    """Batched CholeskyQR2 re-orthogonalization makes the reference's decisions (real data: GPU receipt)."""

    def setUp(self):
        torch.manual_seed(1)
        self.model = ModelIdentity("reorth", "reorth-src", 2, 96, "tokens_channels", "bfloat16")
        self.keys = [MatrixKey(layer, target) for layer in range(2) for target in ("K", "V")]
        self.base = {key: torch.linalg.qr(torch.randn(96, 40)).Q.T for key in self.keys}
        self.method = RankMethodConfig(EnergyTargets("t", 0.95, 0.9, True),
                                       CodecSpec(kind="svd_hevc", rank=1, inter_prediction=False))

    def chunk(self, index, shift=0.3):
        torch.manual_seed(100 + index)
        matrices, contexts = {}, {}
        for key in self.keys:
            value = (torch.randn(240, 40) * torch.linspace(3, 0.05, 40)) @ self.base[key] + 0.02 * torch.randn(240, 96)
            if index:
                value = value + shift * torch.randn(240, 6) @ torch.randn(6, 96)
            matrices[key] = value.to(torch.bfloat16)
            contexts[key] = TensorContext(self.model, key, index * 240, 3, 8, 10,
                                          "pre_rope" if key.target == "K" else "none")
        return matrices, contexts

    def run_chain(self, orthogonalizer, chunks=3, **planner):
        registry = RankPredictorRegistry("r")
        bindings = {key: ActiveBasisBinding() for key in self.keys}
        parents = {key: None for key in self.keys}
        results = []
        for index in range(chunks):
            matrices, contexts = self.chunk(index)
            plan = LatestMethodPlanner(self.method, registry.pin(), orthogonalizer=orthogonalizer, **planner).plan(
                matrices, contexts, bindings, parents, chunk_id=f"c{index}",
                source_ids={key: f"s{index}{key}" for key in self.keys})
            results.append(plan)
            for key, item in plan.plans.items():
                if not item.sets_active or item.decision == "share":
                    continue
                rows = torch.cat((parents[key], item.new_rows)) if item.decision == "augment" else item.new_rows
                fresh = (bindings[key].fresh_rank, bindings[key].fresh_chunk_id) if item.decision == "augment" else (
                    item.rank_view, f"c{index}")
                bindings[key] = ActiveBasisBinding("manifest:" + hashlib.sha256(rows_to_bytes(rows)).hexdigest(),
                                                   int(rows.shape[0]),
                                                   BasisVersion.root_group_id("e", key, "basis:" + "0" * 64), *fresh)
                parents[key] = rows
            registry.observe([o for p in plan.plans.values() for o in p.observations])
        return results

    def test_decisions_ranks_and_energies_match_the_householder_reference(self):
        reference = self.run_chain("householder")
        batched = self.run_chain("batched")
        self.assertGreater(sum(plan.counters["batched_orthogonalizations"] for plan in batched), 0)
        for left, right in zip(reference, batched):
            for key in self.keys:
                self.assertEqual(left.plans[key].decision, right.plans[key].decision)
                self.assertEqual(left.plans[key].rank_view, right.plans[key].rank_view)
                self.assertAlmostEqual(left.plans[key].decision_energy, right.plans[key].decision_energy, delta=1e-7)
            self.assertEqual(right.orthogonalizer, "batched")

    def test_growth_past_the_batched_directions_falls_back_to_the_reference(self):
        import reelcache.rank_method as rank_method
        original = rank_method.LatestMethodPlanner._orthogonalize_group

        def tiny(self, members, channels, device):
            members = [(work, parent_rank, 1) for work, parent_rank, _ in members]
            return original(self, members, channels, device)

        reference = self.run_chain("householder")
        with unittest.mock.patch.object(rank_method.LatestMethodPlanner, "_orthogonalize_group", tiny):
            batched = self.run_chain("batched")
        fallbacks = sum(plan.counters["orthogonalization_fallbacks"] for plan in batched)
        self.assertGreater(fallbacks, 0)
        for left, right in zip(reference, batched):
            for key in self.keys:
                self.assertEqual((left.plans[key].decision, left.plans[key].rank_view),
                                 (right.plans[key].decision, right.plans[key].rank_view))

    def test_unknown_orthogonalizer_is_rejected(self):
        registry = RankPredictorRegistry("r")
        with self.assertRaises(ConfigurationError):
            LatestMethodPlanner(self.method, registry.pin(), orthogonalizer="gram_schmidt")


import unittest.mock  # noqa: E402

if __name__ == "__main__":
    unittest.main()
