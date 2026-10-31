"""Step 4: durable latest-method types, online ridge registry and its replay contract."""

import json
import math
import threading
import unittest
from pathlib import Path

import numpy as np

from reelcache.errors import ConfigurationError, IntegrityError, StateError
from reelcache.predictor_registry import RankPredictorRegistry, RegistryView
from reelcache.types import (
    ENERGY_TARGET_PROFILES, ActiveBasisBinding, BasisVersion, CacheEpoch, ChunkNode, CodecSpec,
    EnergyReport, EnergyTargets, FitObservation, MatrixKey, MatrixRecord, ModelIdentity, OnlineRidgeState,
    PredictorKey, TensorContext, TokenRangeRecords, canonical_digest, energy_targets,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "m07_causvid_rank_features.json"
REF = "manifest:" + "1" * 64
REF2 = "manifest:" + "2" * 64
BLOCK = "basis:" + "3" * 64
HEX = "4" * 64


def predictor_key(layer=0, target="K", regime="fresh", model="fixture-v1", rows=480):
    return PredictorKey(model, layer, target, rows, 96, "fp32_gram_eigh.v1", "sketch_qr_subspace.v1:m32:q2", regime)


def observation(key, source, x, label, anchor=0.5, target=0.98):
    return FitObservation(key, canonical_digest({"source": source}), anchor, target, x, label)


class EnergyTargetTests(unittest.TestCase):
    def test_builtin_profiles_are_explicit_and_causvid_is_final(self):
        causvid = energy_targets("causvid")
        self.assertEqual((causvid.key, causvid.value, causvid.provisional), (0.98, 0.95, False))
        for name in ("raven", "longlive1", "longlive2", "longcat_720p"):
            profile = energy_targets(name)
            self.assertEqual((profile.key, profile.value, profile.provisional), (0.98, 0.95, True))
            self.assertEqual(profile.for_target(MatrixKey(3, "K")), 0.98)
            self.assertEqual(profile.for_target("V"), 0.95)
        self.assertEqual(set(ENERGY_TARGET_PROFILES), {"causvid", "raven", "longlive1", "longlive2", "longcat_720p"})
        with self.assertRaises(ConfigurationError):
            energy_targets("magi")

    def test_fractions_and_diagnostic_target(self):
        for bad in (0, 0.0, 1.01, float("nan"), True, "0.9"):
            with self.assertRaises(ConfigurationError):
                EnergyTargets("x", bad, 0.95, True)
        diagnostic = EnergyTargets("x", 1.0, 0.95, True)
        self.assertTrue(diagnostic.is_diagnostic("K"))
        self.assertFalse(diagnostic.is_diagnostic("V"))
        self.assertEqual(EnergyTargets.from_dict(diagnostic.to_dict()), diagnostic)
        with self.assertRaises(ConfigurationError):
            EnergyTargets("x", 0.98, 0.95, 1)


class RecordTests(unittest.TestCase):
    def context(self, target="K"):
        model = ModelIdentity("fixture", "fixture-v1", 2, 96, "tokens_channels", "float32")
        return TensorContext(model, MatrixKey(1, target), 0, 2, 4, 6, "pre_rope" if target == "K" else "none")

    def test_energy_report_distinguishes_measurements(self):
        report = EnergyReport(0.98, 0.99, 0.985, None, 40, 64, 64)
        self.assertTrue(report.meets_target)
        self.assertIsNone(report.post_codec_relative_error)
        self.assertEqual(EnergyReport.from_dict(report.to_dict()), report)
        with self.assertRaises(ConfigurationError):
            EnergyReport(0.98, 0.99, 0.985, -0.1, 40, 64, 64)
        with self.assertRaises(ConfigurationError):
            EnergyReport(0.98, 0.9, 0.9, None, 0, 0, 1, zero_input=True)
        with self.assertRaises(ConfigurationError):
            EnergyReport(0.98, float("inf"), 0.9, None, 0, 0, 1)

    def test_basis_version_invariants_and_round_trip(self):
        key = MatrixKey(1, "K")
        root = BasisVersion(None, BLOCK, 32, 32, 96, BasisVersion.root_group_id("fixture-v1", key, BLOCK),
                            key, "fixture-v1", 0, HEX)
        self.assertEqual(root.domain, "normalized_pre_rope_k")
        self.assertEqual(BasisVersion.from_dict(root.to_dict()), root)
        child = BasisVersion(REF, "basis:" + "5" * 64, 32, 64, 96, root.group_id, key, "fixture-v1", 1, HEX)
        self.assertEqual(child.parent_rank, 32)
        for kwargs in ({"parent_ref": None, "depth": 1}, {"parent_ref": REF, "depth": 0},
                       {"total_rank": 128, "channels": 96}, {"dtype": "float32"}):
            fields = {**child.__dict__, **kwargs}
            with self.assertRaises(ConfigurationError):
                BasisVersion(**fields)
        data = root.to_dict()
        data["domain"] = "direct_v"
        with self.assertRaises(ConfigurationError):
            BasisVersion.from_dict(data)

    def test_matrix_record_contract_and_deep_immutability(self):
        context = self.context()
        report = EnergyReport(0.98, 0.99, 0.985, 0.1, 40, 64, 64)
        record = MatrixRecord(context, CodecSpec("svd", rank=64), REF, REF2, 64, report, "rev",
                              {"work": {"counters": [1, 2]}})
        self.assertEqual(MatrixRecord.from_dict(json.loads(json.dumps(record.to_dict()))), record)
        with self.assertRaises(TypeError):
            record.decision["work"]["x"] = 1
        self.assertIsInstance(record.decision["work"]["counters"], tuple)
        with self.assertRaises(ConfigurationError):
            MatrixRecord(context, CodecSpec("svd", rank=32), REF, REF2, 64, report, None, {})
        post = TensorContext(context.model, context.key, 0, 2, 4, 6, "post_rope")
        with self.assertRaises(ConfigurationError):
            MatrixRecord(post, CodecSpec("svd", rank=64), REF, REF2, 64, report, None, {})
        with self.assertRaises(ConfigurationError):
            MatrixRecord(context, CodecSpec(), REF, REF2, None, None, None, {})
        MatrixRecord(context, CodecSpec(), REF, None, None, None, None, {})

    def test_nodes_and_epochs(self):
        key_k, key_v = MatrixKey(0, "K"), MatrixKey(0, "V")
        bindings = ((key_k, ActiveBasisBinding()), (key_v, ActiveBasisBinding(REF2, 32, HEX, 32, "c1", 0.5, HEX)))
        node = ChunkNode("chunk", "c2", REF, "e0", REF2, 48, 96, 2, 4, 6, 96, ((key_v, REF), (key_k, REF2)), bindings)
        self.assertEqual(node.matrix_records[0][0], key_k)
        self.assertEqual(ChunkNode.from_dict(json.loads(json.dumps(node.to_dict()))), node)
        self.assertEqual(node.binding(key_v).fresh_rank, 32)
        for bad in (
            dict(node_kind="root", parent_ref=REF), dict(node_kind="recache", frames=2),
            dict(node_kind="chunk", matrix_records=()), dict(frontier=10),
            dict(node_kind="segment", token_start=48, token_end=96),
        ):
            with self.assertRaises(ConfigurationError):
                ChunkNode(**{**node.__dict__, **bad})
        span = TokenRangeRecords("c8", 48, 96, 2, 4, 6, ((key_k, REF), (key_v, REF2)))
        epoch = CacheEpoch("e1", REF, "kv_recache", "global", (span,), {"switch_before": "c11"})
        self.assertEqual(CacheEpoch.from_dict(json.loads(json.dumps(epoch.to_dict()))), epoch)
        with self.assertRaises(ConfigurationError):
            CacheEpoch("e0", None, "initial", "global", (span,), {})
        with self.assertRaises(ConfigurationError):
            CacheEpoch("e1", None, "kv_recache", "global", (span,), {})
        with self.assertRaises(ConfigurationError):
            CacheEpoch("s1", REF, "longcat_segment", "segment_local", (span,), {})
        overlapping = TokenRangeRecords("c9", 72, 120, 2, 4, 6, ((key_k, REF),))
        with self.assertRaises(ConfigurationError):
            CacheEpoch("e1", REF, "kv_recache", "global", (span, overlapping), {})
        with self.assertRaises(ConfigurationError):
            ActiveBasisBinding(REF, 0, HEX)
        with self.assertRaises(ConfigurationError):
            ActiveBasisBinding(None, 0, None, 32, None)


class RidgeStateTests(unittest.TestCase):
    def test_paper_initial_state_is_the_zero_data_rule(self):
        state = OnlineRidgeState.initial()
        self.assertEqual(state.m, (2.0, 0.0, 2.0))
        self.assertAlmostEqual(state.v[0], 2 * math.log(32))
        self.assertEqual(state.v[1], 2.0)
        a, b = state.coefficients()
        self.assertAlmostEqual(a, math.log(32))
        self.assertAlmostEqual(b, 1.0)
        anchor, target = 0.90, 0.98
        x = math.log((1 - anchor) / (1 - target))
        self.assertAlmostEqual(state.predict(x), 32 * (1 - anchor) / (1 - target))

    def test_update_matches_closed_form_ridge(self):
        rng = np.random.default_rng(3)
        xs = rng.uniform(0.2, 3.0, 25)
        labels = rng.integers(33, 900, 25)
        state = OnlineRidgeState.initial()
        for x, label in zip(xs, labels):
            state = state.updated(float(x), int(label))
        z = np.stack([np.ones_like(xs), xs], 1)
        m = 2 * np.eye(2) + z.T @ z
        v = 2 * np.array([math.log(32), 1.0]) + z.T @ np.log(labels)
        expected = np.linalg.solve(m, v)
        self.assertTrue(np.allclose(state.coefficients(), expected, rtol=1e-12, atol=1e-12))
        self.assertEqual((state.observations, state.revision), (25, 25))
        self.assertEqual(OnlineRidgeState.from_dict(state.to_dict()), state)
        self.assertEqual(state.predict(1e6), math.inf)
        with self.assertRaises(ConfigurationError):
            state.updated(float("nan"), 40)


class RegistryTests(unittest.TestCase):
    def test_idempotent_retries_and_conflicts(self):
        registry = RankPredictorRegistry("model-scope")
        key = predictor_key()
        first = observation(key, "chunk-1", 1.2, 80)
        self.assertEqual(registry.observe([first]), (True,))
        self.assertEqual(registry.observe([first]), (False,))
        self.assertEqual(registry.sequence, 1)
        conflict = FitObservation(key, first.source_id, 0.5, 0.98, 1.2, 81)
        with self.assertRaises(IntegrityError):
            registry.observe([conflict])
        self.assertEqual(registry.sequence, 1)
        self.assertEqual(registry.stats, {"applied": 1, "duplicates": 1, "rejected": 1, "divergent_duplicates": 0})
        with self.assertRaises(ConfigurationError):
            registry.observe([observation(key, "chunk-2", 1.0, 32)])

    def test_pinned_views_are_immutable_revisions(self):
        registry = RankPredictorRegistry("model-scope")
        key = predictor_key()
        view = registry.pin()
        before = view.predict(key, 1.5)
        registry.observe([observation(key, "a", 1.5, 300), observation(key, "b", 0.7, 90)])
        self.assertEqual(view.predict(key, 1.5).predicted_rank, before.predicted_rank)
        after = registry.pin().predict(key, 1.5)
        self.assertNotEqual(after.predicted_rank, before.predicted_rank)
        self.assertEqual(after.state.revision, 2)
        self.assertIn(":r2:", after.revision_ref)
        self.assertEqual(before.to_dict()["state"]["revision"], 0)
        self.assertIsInstance(view, RegistryView)

    def test_regime_model_and_geometry_isolation(self):
        registry = RankPredictorRegistry("model-scope")
        fresh, residual = predictor_key(regime="fresh"), predictor_key(regime="residual")
        other_model, other_rows = predictor_key(model="other-v1"), predictor_key(rows=960)
        baseline = {key: registry.pin().predict(key, 1.0).predicted_rank
                    for key in (residual, other_model, other_rows)}
        registry.observe([observation(fresh, f"s{index}", 1.0, 600) for index in range(5)])
        view = registry.pin()
        self.assertGreater(view.predict(fresh, 1.0).predicted_rank, 100)
        for key, value in baseline.items():
            self.assertEqual(view.predict(key, 1.0).predicted_rank, value)

    def test_replay_mode_pins_update_order(self):
        live = RankPredictorRegistry("live")
        key, other = predictor_key(), predictor_key(layer=1)
        batch = [observation(key, "a", 1.1, 70), observation(other, "b", 0.4, 40), observation(key, "c", 2.0, 300)]
        snapshot = live.to_dict()
        live.observe(batch)
        replay = RankPredictorRegistry.from_dict(snapshot, mode="replay", expected_order=live.log)
        with self.assertRaises(StateError):
            replay.observe([batch[2]])
        self.assertEqual(replay.sequence, 0)
        replay.observe(batch[:2])
        replay.observe(batch[2:])
        self.assertEqual(replay.to_dict()["learners"], live.to_dict()["learners"])
        with self.assertRaises(StateError):
            replay.observe([observation(key, "extra", 1.0, 50)])
        with self.assertRaises(ConfigurationError):
            RankPredictorRegistry("x", expected_order=[])

    def test_serialization_round_trip_and_tamper_detection(self):
        registry = RankPredictorRegistry("model-scope")
        registry.observe([observation(predictor_key(), "a", 1.3, 120), observation(predictor_key(target="V"), "b",
                                                                                  0.9, 64)])
        restored = RankPredictorRegistry.from_dict(json.loads(json.dumps(registry.to_dict())))
        for key in (predictor_key(), predictor_key(target="V")):
            self.assertEqual(restored.pin().predict(key, 0.8).predicted_rank,
                             registry.pin().predict(key, 0.8).predicted_rank)
        self.assertEqual(restored.observe([observation(predictor_key(), "a", 1.3, 120)]), (False,))
        data = registry.to_dict()
        data["sequence"] = 5
        with self.assertRaises(IntegrityError):
            RankPredictorRegistry.from_dict(data)

    def test_concurrent_updates_are_serialized(self):
        registry = RankPredictorRegistry("shared")
        keys = [predictor_key(layer=layer) for layer in range(4)]

        def worker(index):
            for step in range(25):
                registry.observe([observation(keys[step % 4], f"{index}-{step}", 0.5 + step / 10, 40 + step)])

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        self.assertEqual(registry.sequence, 100)
        self.assertEqual(sum(state["state"]["observations"] for state in registry.to_dict()["learners"]), 100)
        replay = RankPredictorRegistry.from_dict(RankPredictorRegistry("shared").to_dict(), mode="replay",
                                                 expected_order=registry.log)
        self.assertEqual(replay.sequence, 0)


class M07ReplayTests(unittest.TestCase):
    """Real-data replay: CausVid M07 sketch features and verify labels (36 prompts x 30 layers)."""

    def test_registry_reproduces_the_m07_online_per_layer_line(self):
        document = json.loads(FIXTURE.read_text())
        self.assertEqual(document["source_sha256"],
                         "e1d5da4642e5ddfc073cb8027bfb641b97caf111f6ef7274318aff7446993fdc")
        rows = document["rows"]
        self.assertEqual(len(rows), 2160)
        prompts = sorted({row[0] for row in rows})
        order = np.random.default_rng(0).permutation(len(prompts))
        by_prompt = {prompt: [row for row in rows if row[0] == prompt] for prompt in prompts}
        registry = RankPredictorRegistry("m07-replay")
        reference = {}
        correct = {"K": 0, "V": 0}
        total = {"K": 0, "V": 0}
        tail = 0

        def bucket(rank):
            return int(min(1536, max(32, math.ceil(max(rank, 1) / 32) * 32)))

        for index in order:
            for prompt, layer, target, energy, rho, insub, label, exact in by_prompt[prompts[index]]:
                key = PredictorKey("causvid-m07", layer, target, 4680, 1536, "m07_verify_label",
                                   "m07_sketch_k32_q2", "fresh")
                if rho <= 1 - energy:
                    predicted = float(insub)
                else:
                    tail += 1
                    x = math.log(rho) - math.log(1 - energy)
                    use = registry.pin().predict(key, x)
                    a_m, b_m = reference.setdefault((layer, target), [2 * np.eye(2), 2 * np.array([math.log(32), 1.0])])
                    expected = math.exp(float(np.linalg.solve(a_m, b_m) @ np.array([1.0, x])))
                    self.assertTrue(math.isclose(use.predicted_rank, expected, rel_tol=1e-9))
                    predicted = min(1536.0, max(1.0, use.predicted_rank))
                    if label > 32:
                        registry.observe([FitObservation(key, canonical_digest([prompt, layer, target]), 1 - rho,
                                                         energy, x, label, use.predicted_rank, use.state.revision)])
                        z = np.array([1.0, x])
                        reference[(layer, target)][0] += np.outer(z, z)
                        reference[(layer, target)][1] += z * math.log(label)
                correct[target] += bucket(predicted) == bucket(exact)
                total[target] += 1
        accuracy = {target: correct[target] / total[target] for target in total}
        # M07 reports 91.7% K / 82.7% V averaged over 30 random orders (stride 32).
        self.assertGreater(accuracy["K"], 0.88)
        self.assertGreater(accuracy["V"], 0.78)
        self.assertGreater(registry.sequence, 1000)
        self.assertGreater(tail, 1500)


if __name__ == "__main__":
    unittest.main()
