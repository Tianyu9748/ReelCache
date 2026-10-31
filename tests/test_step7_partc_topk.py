"""Step 7 Part C: top-k Rayleigh-Ritz planner directions (``RankMethodConfig(solver="topk_rr")``).

The opt-in solver replaces the full batched Gram eigensolve by blocked subspace iteration on the Gram and a
Rayleigh-Ritz solve over the predicted bucket plus a margin (warm-started from the anchor sketch subspace, and from
the parent basis for refits); growth past the fitted directions extends the subspace instead of refitting. The paper
semantics are unchanged (predicted bucket -> fit -> exact stored-factor energy verification -> grow/trim;
share/augment/refit), so every accepted rank still meets its target. These tests pin exactness where the subspace is
complete, the Ritz properties, extension, bucket alignment, and the planner's decisions on fixtures; the default
solver's fingerprints are unchanged. Real-capture evidence: artifacts/sglang_integration/runtime_gate/step7/
(scripts/solver_compare.py).
"""

from __future__ import annotations

import contextlib
import os
import unittest
from unittest import mock

import torch

from reelcache import rank_method
from reelcache.deferred_reads import BatchedCall, run_lockstep
from reelcache.errors import ConfigurationError
from reelcache.predictor_registry import RankPredictorRegistry
from reelcache.rank_method import (
    LatestMethodPlanner, RankMethodConfig, default_eigh, gram_directions, plan_chunk, topk_directions,
)
from reelcache.types import (
    ActiveBasisBinding, CodecSpec, EnergyTargets, MatrixKey, ModelIdentity, TensorContext, canonical_digest,
)

CHANNELS = 128
MODEL = ModelIdentity("fixture", "fixture-partc-c128", 1, CHANNELS, "tokens_channels", "float32")
KEYS = (MatrixKey(0, "K"), MatrixKey(0, "V"))
SPECTRUM = torch.logspace(0, -2.0, CHANNELS, dtype=torch.float64)  # r98 = 54, r95 = 42, E32 = 0.90


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


def plan(sources, *, cfg, bindings=None, parents=None, chunk="c0", orthogonalizer="householder", device="cpu",
         eigh=None):
    keys = list(sources)
    return plan_chunk(
        cfg, RankPredictorRegistry("partc-test").pin(), sources,
        {key: context(key, 0, sources[key].shape[0]) for key in keys},
        bindings or {key: ActiveBasisBinding() for key in keys}, parents or {key: None for key in keys},
        chunk_id=chunk, source_ids={key: canonical_digest([chunk, key.to_dict()]) for key in keys},
        orthogonalizer=orthogonalizer, device=device, eigh=eigh,
    )


def active(plans):
    bindings, parents = {}, {}
    for index, (key, item) in enumerate(plans.items()):
        rows = item.new_rows
        ref = "manifest:" + format(index + 1, "064x")
        bindings[key] = ActiveBasisBinding(ref, int(rows.shape[0]), "a" * 64, int(rows.shape[0]), "c0")
        parents[key] = rows
    return bindings, parents


def low_predictions(width):
    """Cap every predicted width (an under-predicting learner) so growth must pass the fitted top-k directions."""
    original = LatestMethodPlanner._predict

    def predict(self, work, regime, target, warm):
        original(self, work, regime, target, warm)
        record = work.prediction_state[regime]
        record["initial_width"] = min(record["initial_width"], width)
    return mock.patch.object(LatestMethodPlanner, "_predict", predict)


def stored_rows(item, parents):
    if item.decision == "share":
        return parents[item.key][:item.rank_view]
    return item.new_rows if item.decision != "augment" else torch.cat((parents[item.key], item.new_rows), 0)


def two_chunks():
    basis = orthonormal(5)
    first = {KEYS[0]: matrix(1, basis), KEYS[1]: matrix(2, basis)}
    # K moves to an unrelated subspace (a residual needing more than one bucket); V mixes in new directions.
    second = {KEYS[0]: matrix(3, orthonormal(7)), KEYS[1]: matrix(4, basis) + 0.6 * matrix(5, orthonormal(8))}
    return first, second


class TopkConfigTests(unittest.TestCase):
    def test_topk_is_the_default_and_the_explicit_exact_solver_keeps_its_fingerprint(self):
        default = config()
        self.assertEqual(default.solver, "topk_rr")  # user decision 2026-10-15 (Step 7 Part C)
        self.assertEqual(default.solver_id(), "fp32_gram_topk_rr.v1:m64:s3")
        self.assertEqual((default.to_dict()["solver"], default.to_dict()["topk_margin"],
                          default.to_dict()["topk_steps"]), ("topk_rr", 64, 3))
        exact = config(solver="gram_eigh")
        self.assertEqual(exact.solver_id(), "fp32_gram_eigh.v1")
        self.assertNotIn("topk_margin", exact.to_dict())
        self.assertNotIn("topk_steps", exact.to_dict())
        # The explicit exact solver with the paper refit rule (the earlier defaults; the paper rule is the default
        # again since 2026-10-17) has the earlier default's method fingerprint (stored methods read unchanged),
        # under this release's names: the digests of the same inputs with the renamed identifiers.
        self.assertEqual(config(solver="gram_eigh", refit_rule="paper").fingerprint,
                         "e2c9c0105d8e1a2591fee40e65fba4e0b8c610aabe5c242fd9d2623efcd56206")
        topk = config(solver="topk_rr", topk_margin=64, topk_steps=1)
        self.assertEqual(topk.solver_id(), "fp32_gram_topk_rr.v1:m64:s1")
        self.assertEqual((topk.to_dict()["topk_margin"], topk.to_dict()["topk_steps"]), (64, 1))
        # Separate learners: the predictor key carries the solver identity.
        self.assertNotEqual(exact.predictor_key(context(KEYS[0]), "fresh"),
                            topk.predictor_key(context(KEYS[0]), "fresh"))

    def test_rejects_invalid_topk_settings(self):
        for kwargs in ({"topk_margin": -1}, {"topk_margin": 4097}, {"topk_steps": 9}, {"topk_steps": True},
                       {"topk_margin": 1.5}, {"solver": "lobpcg"}):
            with self.subTest(**kwargs), self.assertRaises(ConfigurationError):
                config(**kwargs)


class TopkDirectionsTests(unittest.TestCase):
    def setUp(self):
        self.matrices = [matrix(10 + index, orthonormal(20 + index)) for index in range(3)]
        self.totals = [float(item.double().square().sum()) for item in self.matrices]
        self.exact = gram_directions(self.matrices, self.totals)

    def test_complete_when_the_subspace_covers_every_channel(self):
        topk = topk_directions(self.matrices, self.totals, [CHANNELS - 8] * 3, [None] * 3, [1, 2, 3], margin=32)
        for exact, item in zip(self.exact, topk):
            self.assertTrue(item.complete)
            self.assertIsNone(item.extender)
            self.assertTrue(torch.equal(item.rows, exact.rows))
            self.assertTrue(torch.equal(item.energies, exact.energies))

    def test_ritz_directions_are_orthonormal_ordered_and_interlaced(self):
        topk = topk_directions(self.matrices, self.totals, [32] * 3, [None] * 3, [1, 2, 3], margin=16, steps=2)
        for source, exact, item in zip(self.matrices, self.exact, topk):
            k = item.width
            self.assertEqual(k, 48)
            self.assertFalse(item.complete)
            self.assertLess(float((item.rows @ item.rows.T - torch.eye(k)).abs().max()), 1e-4)
            self.assertTrue(bool((item.energies[1:] <= item.energies[:-1]).all()))
            # Cauchy interlacing: Ritz values never exceed the exact eigenvalues.
            self.assertTrue(bool((item.energies <= exact.energies[:k] * (1 + 1e-5) + 1e-9).all()))
            leading = (exact.energies[:8] - item.energies[:8]).abs() / exact.energies[:8]
            self.assertLess(float(leading.max()), 1e-3)
            # The Ritz energies are the exact captured energies of the Ritz rows.
            captured = (source.double() @ item.rows[:32].double().T).square().sum()
            self.assertAlmostEqual(float(captured) / float(item.energies[:32].sum()), 1.0, places=4)

    def test_a_start_block_spanning_the_leading_subspace_is_exact(self):
        starts = [exact.rows[:40].T.contiguous() for exact in self.exact]
        topk = topk_directions(self.matrices, self.totals, [8] * 3, starts, [1, 2, 3], margin=32, steps=0)
        for exact, item in zip(self.exact, topk):
            self.assertEqual(item.width, 40)
            relative = (exact.energies[:40] - item.energies).abs() / exact.energies[:40]
            self.assertLess(float(relative.max()), 1e-4)

    def test_extension_doubles_the_subspace_then_completes_exactly(self):
        (item,) = topk_directions(self.matrices[:1], self.totals[:1], [16], [None], [5], margin=16, steps=1)
        self.assertEqual(item.width, 32)
        wider = item.extender(item, 32)
        self.assertEqual(wider.width, 64)
        self.assertFalse(wider.complete)
        self.assertIsNotNone(wider.extender)
        full = wider.extender(wider, 32)
        self.assertTrue(full.complete)
        self.assertTrue(torch.equal(full.rows, self.exact[0].rows))
        self.assertTrue(torch.equal(full.energies, self.exact[0].energies))

    def test_cholesky_breakdown_is_recomputed_with_householder_qr(self):
        # A rank-deficient start block (one column repeated) breaks the batched CholeskyQR2 down.
        column = torch.randn(CHANNELS, 1, generator=torch.Generator().manual_seed(3))
        starts = [column.repeat(1, 40), None, None]
        with mock.patch("reelcache.rank_method._householder_iteration",
                        wraps=rank_method._householder_iteration) as fallback:
            topk = topk_directions(self.matrices, self.totals, [16] * 3, starts, [1, 2, 3], margin=32, steps=2)
        self.assertEqual(fallback.call_count, 1)
        for exact, item in zip(self.exact, topk):
            self.assertLess(float((item.rows @ item.rows.T - torch.eye(item.width)).abs().max()), 1e-4)
            self.assertTrue(bool((item.energies <= exact.energies[:item.width] * (1 + 1e-5) + 1e-9).all()))
            self.assertLess(float((exact.energies[:4] - item.energies[:4]).abs().max() / exact.energies[0]), 1e-3)

    def test_subspace_width_is_rounded_up_to_the_bucket(self):
        topk = topk_directions(self.matrices[:2], self.totals[:2], [20, 40], [None, None], [1, 2], margin=10,
                               align=32)
        self.assertEqual([item.width for item in topk], [32, 64])

    def test_inputs_must_align(self):
        with self.assertRaises(ConfigurationError):
            topk_directions(self.matrices, self.totals, [32] * 2, [None] * 3, [1, 2, 3])


class TopkPlannerTests(unittest.TestCase):
    def check(self, cfg, *, orthogonalizer="householder", width=None):
        first, second = two_chunks()
        with low_predictions(width) if width is not None else contextlib.nullcontext():
            parent = plan(first, cfg=cfg, orthogonalizer=orthogonalizer)
            bindings, parents = active(parent.plans)
            child = plan(second, cfg=cfg, bindings=bindings, parents=parents, chunk="c1",
                         orthogonalizer=orthogonalizer)
        for key in KEYS:
            for item, source, inherited in ((parent.plans[key], first[key], None),
                                            (child.plans[key], second[key], parents)):
                rows = item.new_rows if inherited is None else stored_rows(item, inherited)
                self.assertGreaterEqual(exact_energy(source, rows), item.target - 1e-6)
                self.assertEqual(rows.shape[0], item.rank_view)
        return parent, child

    def test_decisions_and_ranks_match_the_exact_solver(self):
        for orthogonalizer in ("householder", "batched"):
            exact = self.check(config(refit=False, solver="gram_eigh"), orthogonalizer=orthogonalizer)
            for kwargs in ({}, {"topk_margin": 16}, {"topk_margin": 0, "topk_steps": 1}):
                with self.subTest(orthogonalizer=orthogonalizer, **kwargs):
                    topk = self.check(config(refit=False, solver="topk_rr", **kwargs), orthogonalizer=orthogonalizer)
                    for mine, ref in zip(topk, exact):
                        for key in KEYS:
                            self.assertEqual(mine.plans[key].decision, ref.plans[key].decision)
                            self.assertEqual(mine.plans[key].rank_view, ref.plans[key].rank_view)

    def test_augmentation_extends_the_residual_subspace_instead_of_refitting(self):
        for orthogonalizer in ("householder", "batched"):
            with self.subTest(orthogonalizer=orthogonalizer):
                exact = self.check(config(refit=False, solver="gram_eigh"), orthogonalizer=orthogonalizer,
                                   width=32)[1].plans[KEYS[0]]
                cfg = config(refit=False, solver="topk_rr", topk_margin=0, topk_steps=1)
                item = self.check(cfg, orthogonalizer=orthogonalizer, width=32)[1].plans[KEYS[0]]
                self.assertEqual(item.decision, "augment")
                self.assertEqual(item.reason, "residual_fit_appended")
                self.assertGreaterEqual(item.work["counters"]["subspace_extensions"], 1)
                self.assertEqual((item.decision, item.rank_view), (exact.decision, exact.rank_view))

    def test_refits_and_learners_use_the_topk_solver(self):
        cfg = config(solver="topk_rr")
        parent, child = self.check(cfg)
        decisions = {child.plans[key].decision for key in KEYS}
        self.assertLessEqual(decisions, {"share", "augment", "refit"})
        solvers = {observation.key.solver for result in (parent, child) for item in result.plans.values()
                   for observation in item.observations}
        self.assertLessEqual(solvers, {"fp32_gram_topk_rr.v1:m64:s3"})
        counters = [item.work["counters"] for item in parent.plans.values()]
        self.assertTrue(all(counter["eigh_count"] == 1 for counter in counters))

    def test_growth_past_the_fitted_directions_stays_on_the_bucket_grid(self):
        cfg = config(solver="topk_rr", topk_margin=0, topk_steps=1)
        parent, _ = self.check(cfg, width=32)
        exact, _ = self.check(config(solver="gram_eigh"), width=32)
        for key in KEYS:
            self.assertEqual(parent.plans[key].rank_view % cfg.bucket, 0)
            self.assertEqual(parent.plans[key].rank_view, exact.plans[key].rank_view)
        self.assertTrue(any(item.work["counters"]["subspace_extensions"] for item in parent.plans.values()))


class LockstepBatchedCallTests(unittest.TestCase):
    def test_calls_of_one_round_run_once_per_key_and_results_are_routed(self):
        calls = []

        def run(payloads):
            calls.append(list(payloads))
            return [payload * 10 for payload in payloads]

        def routine(index, key):
            got = yield BatchedCall(key, run, index)
            (value,) = yield (torch.tensor(float(got)),)
            return value + index

        results = run_lockstep([routine(0, "a"), routine(1, "a"), routine(2, "b"), routine(3, "a")])
        self.assertEqual(results, [0.0, 11.0, 22.0, 33.0])
        self.assertEqual(calls, [[0, 1, 3], [2]])

    def test_reads_and_calls_share_a_round(self):
        def reader():
            (value,) = yield (torch.tensor(2.0),)
            return value

        def caller():
            got = yield BatchedCall("k", lambda payloads: [payload + 1 for payload in payloads], 4)
            return got

        self.assertEqual(run_lockstep([reader(), caller()]), [2.0, 5])

    def test_a_failing_call_is_thrown_into_each_calling_routine(self):
        def run(_payloads):
            raise ValueError("boom")

        def catching():
            try:
                yield BatchedCall("k", run, 0)
            except ValueError:
                return "caught"

        def plain():
            yield BatchedCall("k", run, 1)
            return "unreachable"

        self.assertEqual(run_lockstep([catching(), catching()]), ["caught", "caught"])
        with self.assertRaisesRegex(ValueError, "boom"):
            run_lockstep([catching(), plain()])


class BatchedExtensionTests(unittest.TestCase):
    def test_extensions_requested_in_one_round_run_as_one_batch(self):
        first, _ = two_chunks()
        cfg = config(solver="topk_rr", topk_margin=0, topk_steps=1)
        with low_predictions(32), mock.patch("reelcache.rank_method._extend_topk_batch",
                                             wraps=rank_method._extend_topk_batch) as batch:
            result = plan(first, cfg=cfg)
        extensions = sum(item.work["counters"]["subspace_extensions"] for item in result.plans.values())
        sizes = [len(call.args[0]) for call in batch.call_args_list]
        self.assertEqual(sum(sizes), extensions)
        self.assertIn(2, sizes)  # K and V extend in the same verification round
        for key in KEYS:
            self.assertGreaterEqual(exact_energy(first[key], result.plans[key].new_rows), result.plans[key].target)

    def test_batched_and_inline_extensions_agree(self):
        matrices = [matrix(70 + index, orthonormal(80 + index)) for index in range(3)]
        totals = [float(item.double().square().sum()) for item in matrices]
        sets = topk_directions(matrices, totals, [16] * 3, [None] * 3, [1, 2, 3], margin=16, steps=2)
        batched = rank_method._extend_topk_batch([(item.extender, item, 32) for item in sets])
        for item, together in zip(sets, batched):
            alone = item.extender(item, 32)
            self.assertEqual(alone.width, together.width)
            self.assertTrue(torch.allclose(alone.energies, together.energies, rtol=1e-6, atol=0))


if os.environ.get("REELCACHE_STEP7_GPU_TESTS") == "1":
    class TopkCUDATests(unittest.TestCase):
        """CUDA top-k planner (inside GPU jobs only; collected only with REELCACHE_STEP7_GPU_TESTS=1)."""

        @classmethod
        def setUpClass(cls):
            if not torch.cuda.is_available():
                raise AssertionError("REELCACHE_STEP7_GPU_TESTS=1 requires a CUDA device")
            from reelcache.gpu_fitting import select_eigh_solver
            cls.label, cls.eigh = select_eigh_solver("auto")

        def test_complete_topk_equals_the_batched_gram_eigensolve(self):
            matrices = [matrix(30 + index, orthonormal(40 + index)).cuda() for index in range(4)]
            totals = [float(item.double().square().sum()) for item in matrices]
            exact = gram_directions(matrices, totals, eigh=self.eigh)
            topk = topk_directions(matrices, totals, [CHANNELS] * 4, [None] * 4, [1, 2, 3, 4], eigh=self.eigh)
            for left, right in zip(exact, topk):
                self.assertTrue(right.complete)
                self.assertTrue(torch.equal(left.rows, right.rows))
                self.assertTrue(torch.equal(left.energies, right.energies))

        def test_one_host_read_per_batch(self):
            import warnings
            matrices = [matrix(50 + index, orthonormal(60 + index)).cuda() for index in range(6)]
            totals = [float(item.double().square().sum()) for item in matrices]
            topk_directions(matrices, totals, [32] * 6, [None] * 6, list(range(6)), eigh=self.eigh)  # warm-up
            torch.cuda.synchronize()
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                torch.cuda.set_sync_debug_mode("warn")
                try:
                    topk_directions(matrices, totals, [32] * 6, [None] * 6, list(range(6)), eigh=self.eigh)
                finally:
                    torch.cuda.set_sync_debug_mode("default")
            # (set_sync_debug_mode itself warns that the mode is a prototype; only count synchronizing calls.)
            syncs = sum("called a synchronizing cuda operation" in str(item.message).lower() for item in records)
            # One read of the breakdown/finiteness flags plus at most the eigensolver's own convergence check.
            self.assertLessEqual(syncs, 2)

        def test_cuda_planner_meets_targets_with_matching_decisions(self):
            first, second = two_chunks()
            outcomes = {}
            for name, cfg in (("exact", config(refit=False, solver="gram_eigh")),
                              ("topk", config(refit=False, solver="topk_rr")),
                              ("topk_m0", config(refit=False, solver="topk_rr", topk_margin=0, topk_steps=1))):
                with low_predictions(32):
                    parent = plan({key: value.cuda() for key, value in first.items()}, cfg=cfg,
                                  orthogonalizer="batched", device="cuda", eigh=self.eigh)
                    bindings, parents = active(parent.plans)
                    parents = {key: value.cpu() for key, value in parents.items()}
                    child = plan({key: value.cuda() for key, value in second.items()}, cfg=cfg, bindings=bindings,
                                 parents=parents, chunk="c1", orthogonalizer="batched", device="cuda",
                                 eigh=self.eigh)
                for key in KEYS:
                    rows = stored_rows(child.plans[key], parents).cpu()
                    self.assertGreaterEqual(exact_energy(second[key], rows), child.plans[key].target - 1e-6)
                    self.assertGreaterEqual(exact_energy(first[key], parent.plans[key].new_rows.cpu()),
                                            parent.plans[key].target - 1e-6)
                outcomes[name] = [(result.plans[key].decision, result.plans[key].rank_view)
                                  for result in (parent, child) for key in KEYS]
            self.assertEqual(outcomes["topk"], outcomes["exact"])
            self.assertEqual(outcomes["topk_m0"], outcomes["exact"])


if __name__ == "__main__":
    unittest.main()
