"""Step 4: paper-bound sketch, prediction, verify/grow/trim, certify/share, augment and refit."""

import math
import random
import unittest
from unittest import mock

import numpy as np
import torch

from reelcache.errors import ConfigurationError
from reelcache.predictor_registry import RankPredictorRegistry
from reelcache.rank_method import (
    LatestMethodPlanner, RankMethodConfig, RankTargetUnreachable, anchor_sketches, estimate_bytes, plan_chunk,
)
from reelcache.types import (
    ActiveBasisBinding, CodecSpec, EnergyTargets, MatrixKey, ModelIdentity, OnlineRidgeState, TensorContext,
    canonical_digest,
)

CHANNELS = 96
MODEL = ModelIdentity("fixture", "fixture-v1", 1, CHANNELS, "tokens_channels", "float32")
KEYS = (MatrixKey(0, "K"), MatrixKey(0, "V"))
SPECTRUM = torch.logspace(0, -1.5, CHANNELS, dtype=torch.float64)  # r98 = 54, r95 = 41, E32 = 0.903


def orthonormal(seed, size=CHANNELS):
    generator = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(size, size, generator=generator, dtype=torch.float64))[0].T


def matrix(seed, basis, spectrum=SPECTRUM, rows=480):
    generator = torch.Generator().manual_seed(seed)
    left = torch.linalg.qr(torch.randn(rows, CHANNELS, generator=generator, dtype=torch.float64))[0]
    return ((left * spectrum) @ basis).float()


def context(key, start=0, rows=480):
    return TensorContext(MODEL, key, start, 2, rows // 48, 24, "pre_rope" if key.target == "K" else "none")


def config(**kwargs):
    return RankMethodConfig(EnergyTargets("t", 0.98, 0.95, True), CodecSpec("svd", rank=1), **kwargs)


def exact_energy(source, rows16):
    q = rows16.float()
    p = source.float() @ q.T
    residual = source.double() - (p @ q).double()
    return 1 - float(residual.square().sum() / source.double().square().sum())


def exact_crossing(source, fraction):
    values = torch.linalg.svdvals(source.double()).square()
    cumulative = values.cumsum(0) / values.sum()
    return int(torch.nonzero(cumulative >= fraction)[0, 0]) + 1


def plan(sources, *, cfg=None, registry=None, bindings=None, parents=None, chunk="c0", start=0):
    cfg = cfg or config()
    registry = registry or RankPredictorRegistry("planner-test")
    keys = list(sources)
    return plan_chunk(
        cfg, registry.pin(), sources, {key: context(key, start, sources[key].shape[0]) for key in keys},
        bindings or {key: ActiveBasisBinding() for key in keys}, parents or {key: None for key in keys},
        chunk_id=chunk, source_ids={key: canonical_digest([chunk, key.to_dict()]) for key in keys},
    )


def active(plans, fresh_rank=None):
    bindings, parents = {}, {}
    for index, (key, item) in enumerate(plans.items()):
        rows = item.new_rows
        ref = "manifest:" + format(index + 1, "064x")
        bindings[key] = ActiveBasisBinding(ref, int(rows.shape[0]), "a" * 64, fresh_rank or int(rows.shape[0]), "c0")
        parents[key] = rows
    return bindings, parents


class SketchTests(unittest.TestCase):
    def test_qr_subspace_sketch_is_orthonormal_private_and_batch_invariant(self):
        sources = [matrix(seed, orthonormal(seed + 50)) for seed in range(3)]
        totals = [float(item.double().square().sum()) for item in sources]
        state = torch.get_rng_state()
        batched = anchor_sketches(sources, totals, [11, 12, 13], anchor_rank=32, power_iterations=2)
        single = anchor_sketches(sources[:1], totals[:1], [11], anchor_rank=32, power_iterations=2)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        sketch = batched[0]["sketch"]
        self.assertEqual(tuple(sketch.shape), (CHANNELS, 32))
        self.assertLess(float((sketch.T @ sketch - torch.eye(32)).abs().max()), 1e-5)
        self.assertAlmostEqual(batched[0]["captured_energy"], single[0]["captured_energy"], places=5)
        self.assertEqual((batched[0]["matmuls"], batched[0]["qr"]), (7, 3))
        for source, result in zip(sources, batched):
            values = torch.linalg.svdvals(source.double()).square()
            optimal = float(values[:32].sum() / values.sum())
            self.assertLessEqual(result["captured_energy"], optimal + 1e-6)
            # q = 2 subspace iteration: the sketch residual stays within 15% of the optimal rank-32 residual.
            self.assertLess(1 - result["captured_energy"], 1.15 * (1 - optimal))
        again = anchor_sketches(sources, totals, [11, 12, 13], anchor_rank=32, power_iterations=2)
        self.assertEqual(again[1]["captured_energy"], batched[1]["captured_energy"])

    def test_degenerate_widths_are_clamped(self):
        source = torch.randn(20, CHANNELS, generator=torch.Generator().manual_seed(1))
        result = anchor_sketches([source], [float(source.double().square().sum())], [1], anchor_rank=32,
                                 power_iterations=2)[0]
        self.assertEqual(result["width"], 20)
        self.assertAlmostEqual(result["captured_energy"], 1.0, places=5)


class FreshFitTests(unittest.TestCase):
    def test_verified_stored_factors_meet_target_and_label_is_in_subspace(self):
        sources = {key: matrix(index, orthonormal(10 + index)) for index, key in enumerate(KEYS)}
        rng_state, py_state, np_state = torch.get_rng_state(), random.getstate(), np.random.get_state()
        result = plan(sources)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_state))
        self.assertEqual(random.getstate(), py_state)
        self.assertEqual(np.random.get_state()[1].tolist(), np_state[1].tolist())
        for key, item in result.plans.items():
            target = 0.98 if key.target == "K" else 0.95
            self.assertEqual(item.decision, "fresh")
            self.assertEqual(item.rank_view % 32, 0)
            self.assertEqual(item.new_rows.dtype, torch.float16)
            self.assertGreaterEqual(exact_energy(sources[key], item.new_rows), target)
            exact = exact_crossing(sources[key], target)
            self.assertLessEqual(abs(item.observed_rank - exact), 1)
            self.assertLessEqual(item.observed_rank, item.rank_view)
            self.assertEqual(item.work["counters"]["eigh_count"], 1)
            self.assertEqual(item.work["counters"]["candidate_fit_count"], 1)
            self.assertLess(exact_energy(sources[key], item.new_rows[:item.rank_view - 32]), target)
            prediction = item.predictions[0]
            self.assertIn(prediction["kind"], ("log_linear", "log_linear_clamped"))
            self.assertEqual(prediction["predictor"]["state"]["revision"], 0)
            self.assertIn("reelcache-rank-predictor:planner-test:", prediction["predictor"]["revision_ref"])
            if item.observed_rank > 32:
                (observation,) = item.observations
                self.assertEqual(observation.key.regime, "fresh")
                self.assertEqual(observation.observed_rank, item.observed_rank)
                self.assertAlmostEqual(observation.x, math.log((1 - prediction["anchor_energy"]) / (1 - target)))
            else:
                self.assertEqual(item.observations, ())
        self.assertEqual(result.counters["decision_fresh"], 2)
        self.assertGreater(result.stages["eigh"], 0)

    def test_growth_and_trim_reuse_one_fit(self):
        source = {KEYS[0]: matrix(3, orthonormal(30))}
        good = plan(source).plans[KEYS[0]]
        for name, a in (("undershoot", 0.0), ("overshoot", 12.0)):
            registry = RankPredictorRegistry(name)
            key = config().predictor_key(context(KEYS[0]), "fresh")
            registry._states[key] = OnlineRidgeState((1e6, 0.0, 1e6), (1e6 * a, 0.0), 1, 1)
            item = plan(source, registry=registry).plans[KEYS[0]]
            self.assertEqual(item.rank_view, good.rank_view, name)
            self.assertEqual(item.observed_rank, good.observed_rank)
            counters = item.work["counters"]
            self.assertEqual(counters["eigh_count"], 1)
            if name == "undershoot":
                self.assertEqual(item.predictions[0]["initial_width"], 32)
                self.assertEqual(counters["growth_steps"], (good.rank_view - 32) // 32)
            else:
                self.assertEqual(item.predictions[0]["initial_width"], CHANNELS)
                self.assertEqual(counters["growth_steps"], 0)
                self.assertGreaterEqual(counters["trimmed_buckets"], 1)
                self.assertLessEqual(counters["trim_verifications"], 2)

    def test_sketch_bucket_zero_and_diagnostic_paths(self):
        low = torch.zeros(CHANNELS, dtype=torch.float64)
        low[:5] = torch.tensor([1.0, 0.8, 0.5, 0.3, 0.2], dtype=torch.float64)
        sources = {KEYS[0]: matrix(4, orthonormal(40), low), KEYS[1]: torch.zeros(480, CHANNELS)}
        result = plan(sources)
        sketch_item, zero_item = result.plans[KEYS[0]], result.plans[KEYS[1]]
        self.assertEqual(sketch_item.predictions[0]["kind"], "sketch_bucket")
        self.assertEqual(sketch_item.rank_view, 32)
        self.assertEqual(sketch_item.observations, ())
        self.assertEqual((zero_item.decision, zero_item.rank_view, zero_item.zero_input), ("zero_input", 1, True))
        self.assertEqual(zero_item.work["counters"]["candidate_fit_count"], 0)
        self.assertFalse(zero_item.sets_active)
        diagnostic = RankMethodConfig(EnergyTargets("diag", 1.0, 0.95, True), CodecSpec("svd", rank=1))
        item = plan({KEYS[0]: matrix(5, orthonormal(41))}, cfg=diagnostic).plans[KEYS[0]]
        self.assertEqual((item.decision, item.rank_view, item.diagnostic), ("diagnostic_full_width", CHANNELS, True))
        self.assertEqual((item.predictions, item.observations), ((), ()))
        self.assertLess(item.decision_energy, 1.0)
        unreachable = RankMethodConfig(EnergyTargets("x", 1 - 1e-12, 0.95, True), CodecSpec("svd", rank=1))
        with self.assertRaises(RankTargetUnreachable) as caught:
            plan({KEYS[0]: matrix(6, orthonormal(42))}, cfg=unreachable)
        self.assertGreaterEqual(caught.exception.work["counters"]["verification_count"], 1)

    def test_planner_input_contract(self):
        cfg = config()
        registry = RankPredictorRegistry("x")
        with self.assertRaises(ConfigurationError):
            LatestMethodPlanner(cfg, registry, device="cpu")
        post = TensorContext(MODEL, KEYS[0], 0, 2, 10, 24, "post_rope")
        with self.assertRaises(ConfigurationError):
            plan_chunk(cfg, registry.pin(), {KEYS[0]: matrix(1, orthonormal(1))}, {KEYS[0]: post},
                       {KEYS[0]: ActiveBasisBinding()}, {KEYS[0]: None}, chunk_id="c", source_ids={KEYS[0]: "x"})
        with self.assertRaises(ConfigurationError):
            RankMethodConfig(EnergyTargets("t", 0.98, 0.95, True), CodecSpec())
        with self.assertRaises(ConfigurationError):
            RankMethodConfig(EnergyTargets("t", 0.98, 0.95, True), CodecSpec("svd_hevc", rank=1, inter_prediction=None))


class PrecisionGuardTests(unittest.TestCase):
    def test_planning_accepts_any_flag_state_without_touching_global_flags(self):
        # Planning products are explicit IEEE FP32 (reelcache.exact_blas): the flags are neither refused nor toggled.
        from reelcache.rank_method import require_fp32_matmul
        require_fp32_matmul(torch.device("cpu"))
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            require_fp32_matmul(torch.device("cpu"))
            require_fp32_matmul(torch.device("cuda", 0))
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
        with mock.patch.object(torch, "get_float32_matmul_precision", return_value="high"):
            require_fp32_matmul(torch.device("cuda", 0))
        with self.assertRaises(ConfigurationError):
            require_fp32_matmul(torch.device("cuda", 0), "tf32")
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)


class ReuseTests(unittest.TestCase):
    def test_share_has_no_candidate_fit_and_keeps_a_prefix(self):
        basis = orthonormal(60)
        parent = plan({KEYS[0]: matrix(1, basis)})
        bindings, parents = active(parent.plans)
        steep = SPECTRUM.clone()
        steep[20:] *= 0.05
        child = plan({KEYS[0]: matrix(2, basis, steep)}, bindings=bindings, parents=parents, chunk="c1", start=480)
        item = child.plans[KEYS[0]]
        self.assertEqual(item.decision, "share")
        counters = item.work["counters"]
        self.assertEqual((counters["candidate_fit_count"], counters["eigh_count"], counters["sketch_count"]),
                         (0, 0, 0))
        self.assertEqual(counters["certify_projection_count"], 1)
        self.assertIsNone(item.new_rows)
        self.assertLess(item.rank_view, parent.plans[KEYS[0]].rank_view)
        prefix = parents[KEYS[0]][:item.rank_view]
        self.assertGreaterEqual(exact_energy(matrix(2, basis, steep), prefix), 0.98)
        self.assertEqual(item.parent_version_ref, bindings[KEYS[0]].version_ref)
        self.assertEqual((item.observations, item.predictions), ((), ()))

    def test_augment_appends_orthogonal_rows_and_meets_target(self):
        basis = orthonormal(70)
        parent = plan({KEYS[0]: matrix(1, basis)})
        bindings, parents = active(parent.plans)
        mixed = basis.clone()
        mixed[8:14] = orthonormal(71)[8:14]
        mixed = torch.linalg.qr(mixed.T)[0].T
        child_source = matrix(3, mixed)
        cheap = config(coefficient_bytes_proxy=0.01)
        item = plan({KEYS[0]: child_source}, cfg=cheap, bindings=bindings, parents=parents, chunk="c1").plans[KEYS[0]]
        self.assertEqual(item.decision, "augment")
        rows = parents[KEYS[0]]
        combined = torch.cat((rows, item.new_rows))
        self.assertEqual(item.rank_view, combined.shape[0])
        self.assertGreaterEqual(exact_energy(child_source, combined), 0.98)
        self.assertLess(float((rows.float() @ item.new_rows.float().T).abs().max()), 2e-3)
        self.assertEqual(item.new_rows.shape[0] % 32, 0)
        self.assertEqual(item.work["counters"]["residual_fit_count"], 1)
        self.assertEqual(item.estimate["decision"], "augment")
        self.assertEqual(item.estimate["augmented"]["estimate_type"], "factor_byte_proxy")
        self.assertLess(item.estimate["augmented"]["total_bytes"], item.estimate["fresh"]["total_bytes"])
        regimes = {observation.key.regime for observation in item.observations}
        self.assertLessEqual(regimes, {"residual"})
        self.assertEqual(item.predictions[0]["regime"], "residual")

    def test_refit_when_augmented_bytes_exceed_fresh(self):
        parent = plan({KEYS[0]: matrix(1, orthonormal(80))})
        bindings, parents = active(parent.plans)
        source = matrix(2, orthonormal(81))
        item = plan({KEYS[0]: source}, bindings=bindings, parents=parents, chunk="c1").plans[KEYS[0]]
        self.assertEqual(item.decision, "refit")
        self.assertIsNone(item.parent_version_ref)
        self.assertEqual(item.estimate["reason"], "augmented_bytes_exceed_fresh")
        self.assertGreater(item.estimate["augmented"]["total_bytes"], item.estimate["fresh"]["total_bytes"])
        self.assertEqual(item.fresh_rank, item.rank_view)
        self.assertEqual(item.work["counters"]["refit_count"], 1)
        self.assertEqual(item.work["counters"]["candidate_fit_count"], 2)
        self.assertGreaterEqual(exact_energy(source, item.new_rows), 0.98)
        no_refit = plan({KEYS[0]: source}, cfg=config(refit=False), bindings=bindings, parents=parents,
                        chunk="c1").plans[KEYS[0]]
        self.assertEqual(no_refit.decision, "augment")
        self.assertEqual(no_refit.estimate["reason"], "refit_disabled")

    def test_byte_estimator_prefers_measured_rates_for_matching_codecs(self):
        cfg = config()
        ctx = context(KEYS[0])
        from reelcache.rank_method import rate_identity
        identity = rate_identity(ctx, cfg.spec_for(32))
        measured = ActiveBasisBinding("manifest:" + "1" * 64, 32, "a" * 64, 32, "c0", 0.25, identity)
        estimate = estimate_bytes(cfg, ctx, 64, 32, measured)
        self.assertEqual(estimate["estimate_type"], "measured_coefficient_rate")
        self.assertEqual(estimate["coefficient_bytes"], 0.25 * 480 * 64)
        self.assertEqual(estimate["basis_bytes"], 2 * CHANNELS * 32)
        other = ActiveBasisBinding("manifest:" + "1" * 64, 32, "a" * 64, 32, "c0", 0.25, "b" * 64)
        self.assertEqual(estimate_bytes(cfg, ctx, 64, 32, other)["estimate_type"], "factor_byte_proxy")
        hevc = RankMethodConfig(EnergyTargets("t", 0.98, 0.95, True), CodecSpec("svd_hevc", rank=1))
        self.assertEqual(estimate_bytes(hevc, ctx, 64, 0, ActiveBasisBinding())["scale_bytes"], 2 * 64 * 4)


class WarmStartTests(unittest.TestCase):
    def test_explicit_warm_start_refits_from_the_parent_basis(self):
        parent = plan({KEYS[0]: matrix(1, orthonormal(90))})
        bindings, parents = active(parent.plans)
        source = matrix(2, orthonormal(91))
        warm = config(solver="warm_start")
        item = plan({KEYS[0]: source}, cfg=warm, bindings=bindings, parents=parents, chunk="c1").plans[KEYS[0]]
        self.assertEqual(item.decision, "refit")
        counters = item.work["counters"]
        self.assertEqual(counters["warm_start_count"], 1)
        self.assertGreaterEqual(exact_energy(source, item.new_rows), 0.98)
        fresh_keys = {observation.key.solver for observation in item.observations if observation.key.regime == "fresh"}
        self.assertLessEqual(fresh_keys, {"fp32_gram_warm_rr.v1:s1"})


if __name__ == "__main__":
    unittest.main()
