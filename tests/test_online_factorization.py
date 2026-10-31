import math
import hashlib
import importlib.util
import pickle
import dataclasses
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from reelcache.errors import CapacityError, ConfigurationError
from reelcache.factorization import (
    fit_rank_directions,
    measure_source_energy,
    project_rank_directions,
    rank_fit_workspace_bytes,
    rank_projection_workspace_bytes,
    RankFitError,
    RankProjectionError,
    DetectorSpec,
    DetectorAcquisitionError,
    acquire_anchor_directions,
    tensor_sha256,
)
from reelcache.types import MatrixKey, ModelIdentity, TensorContext
from tests.core_support import CoreWorkspaceTest


def context(rows, channels, dtype="float32", target="K", key_space=None):
    model = ModelIdentity("rank-fixture", "synthetic", 1, channels, "tokens_channels", dtype)
    return TensorContext(
        model, MatrixKey(0, target), 0, 1, 1, rows,
        key_space or ("pre_rope" if target == "K" else "none"),
    )


def with_cuda_runtime_diagnostics(report):
    diagnostics = (
        "  - CUDA Runtime 12.8\n"
        "  - Built with CUDA Runtime 12.7\n"
        "  - NVCC architecture flags: -gencode;arch=compute_80,code=sm_80\n"
        "  - CuDNN 9.8.0  (built against CUDA 12.8)\n"
        "    - Built with CuDNN 9.7.0\n"
        "  - Magma 2.6.1\n"
    )
    return report.replace("  - Build settings: ", diagnostics + "  - Build settings: ", 1)


class OnlineFactorizationTests(unittest.TestCase):
    def test_finite_width_cannot_renormalize_observed_energy_to_one(self):
        matrix = torch.eye(8)
        fit = fit_rank_directions(matrix, context(8, 8), 2, tile_rows=3)
        self.assertIsNone(fit.required_rank(0.95))
        self.assertAlmostEqual(fit.energy_fraction(2), 0.25, places=5)
        self.assertEqual(fit.metrics["svd_count"], 1)
        self.assertEqual(fit.metrics["qr_count"], 5)
        self.assertEqual(fit.metrics["projection_count"], 0)

    def test_selected_prefix_projects_same_directions_without_another_svd(self):
        matrix = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
        with mock.patch("torch.linalg.svd", wraps=torch.linalg.svd) as svd:
            fit = fit_rank_directions(matrix, context(4, 4), 4, tile_rows=2)
            self.assertEqual(fit.required_rank(0.95), 3)
            factors = project_rank_directions(matrix, fit.directions, 3, tile_rows=2)
        self.assertEqual(svd.call_count, 1)
        self.assertEqual(tuple(factors.basis.shape), (3, 4))
        self.assertEqual(factors.basis.dtype, torch.float16)
        self.assertEqual(factors.coefficients.dtype, torch.float32)
        self.assertTrue(torch.equal(
            factors.coefficients, matrix @ factors.basis.float().T,
        ))
        self.assertAlmostEqual(
            factors.metrics["projection_relative_residual"], math.sqrt(1 / 30), places=5,
        )
        self.assertEqual(factors.metrics["basis_bytes"], 3 * 4 * 2)
        self.assertEqual(factors.metrics["coefficient_bytes"], 4 * 3 * 4)

    def test_private_generator_does_not_change_generation_rng(self):
        matrix = torch.randn(31, 12, generator=torch.Generator().manual_seed(311))
        before_torch = torch.get_rng_state().clone()
        before_python = random.getstate()
        before_numpy = np.random.get_state()
        first = fit_rank_directions(matrix, context(31, 12), 5, seed=99, tile_rows=7)
        second = fit_rank_directions(matrix, context(31, 12), 5, seed=99, tile_rows=7)
        self.assertTrue(torch.equal(first.directions, second.directions))
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
        self.assertEqual(before_python, random.getstate())
        after_numpy = np.random.get_state()
        self.assertEqual(before_numpy[0], after_numpy[0])
        np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
        self.assertEqual(before_numpy[2:], after_numpy[2:])

    def test_full_original_norm_and_guard_retain_source_precision(self):
        values = torch.tensor(
            [[1.003, 2.007, -4.011], [-2.019, 3.023, 6.029],
             [4.031, -1.037, 2.041], [3.043, 5.047, -7.053]],
        )
        for dtype in (torch.bfloat16, torch.float32, torch.float64):
            matrix = values.to(dtype)
            energy = measure_source_energy(matrix, tile_rows=1)
            self.assertAlmostEqual(energy.source_norm, float(matrix.double().norm()), places=12)
            fit = fit_rank_directions(
                matrix, context(4, 3, str(dtype).removeprefix("torch.")), 3, tile_rows=1,
            )
            result = project_rank_directions(matrix, fit.directions, 3, tile_rows=1)
            p = matrix.float() @ result.basis.float().T
            direct = matrix.double() - (p @ result.basis.float()).double()
            expected = float(direct.norm() / matrix.double().norm())
            self.assertAlmostEqual(result.metrics["projection_relative_residual"], expected, places=12)

    def test_nonzero_underflow_is_not_mathematical_zero(self):
        matrix = torch.full((4, 3), 1e-250, dtype=torch.float64)
        energy = measure_source_energy(matrix, tile_rows=1)
        self.assertFalse(energy.zero_input)
        self.assertGreater(energy.source_norm, 0)
        fit = fit_rank_directions(matrix, context(4, 3, "float64"), 3, tile_rows=2)
        self.assertEqual(fit.required_rank(0.95), 1)
        result = project_rank_directions(matrix, fit.directions, 1)
        self.assertFalse(result.metrics["zero_input"])
        self.assertEqual(result.metrics["projection_relative_residual"], 1.0)

    def test_zero_has_no_decomposition_and_a_deliberate_operational_prefix(self):
        matrix = torch.zeros(7, 4)
        with (
            mock.patch("torch.linalg.svd", side_effect=AssertionError("SVD on zero")),
            mock.patch("torch.linalg.eigh", side_effect=AssertionError("eigh on zero")),
            mock.patch("torch.linalg.qr", side_effect=AssertionError("QR on zero")),
            mock.patch("torch.randn", side_effect=AssertionError("RNG on zero")),
        ):
            fit = fit_rank_directions(matrix, context(7, 4), 3)
            factors = project_rank_directions(matrix, fit.directions, 1)
        self.assertEqual(fit.required_rank(0.99), 0)
        self.assertEqual(fit.metrics["decomposition_count"], 0)
        self.assertEqual(factors.rank, 1)
        self.assertEqual(factors.metrics["projection_relative_residual"], 0)

    def test_workspace_admission_precedes_factor_or_projection_work(self):
        matrix = torch.ones(8, 5)
        fit_bytes = rank_fit_workspace_bytes(matrix.shape, 3, tile_rows=2)
        projection_bytes = rank_projection_workspace_bytes(matrix.shape, 3, tile_rows=2)
        with (
            mock.patch("torch.linalg.svd", side_effect=AssertionError("unadmitted SVD")),
            mock.patch("torch.linalg.qr", side_effect=AssertionError("unadmitted QR")),
            mock.patch(
                "reelcache.factorization.project_onto_basis",
                side_effect=AssertionError("unadmitted projection"),
            ),
        ):
            with self.assertRaises(CapacityError):
                fit_rank_directions(
                    matrix, context(8, 5), 3, tile_rows=2,
                    max_workspace_bytes=fit_bytes - 1,
                )
            with self.assertRaises(CapacityError):
                project_rank_directions(
                    matrix, torch.eye(5), 3, tile_rows=2,
                    max_workspace_bytes=projection_bytes - 1,
                )

    def test_invalid_keys_widths_targets_and_precision_fail_closed(self):
        matrix = torch.ones(8, 5)
        with self.assertRaises(ConfigurationError):
            fit_rank_directions(matrix, context(8, 5, key_space="post_rope"), 3)
        for width in (0, True, 6):
            with self.assertRaises(ConfigurationError):
                fit_rank_directions(matrix, context(8, 5), width)
        fit = fit_rank_directions(matrix, context(8, 5), 3)
        for energy in (0, -1, 1.1, True, float("nan")):
            with self.assertRaises(ConfigurationError):
                fit.required_rank(energy)
        with self.assertRaises(ConfigurationError):
            project_rank_directions(matrix, fit.directions, 4)

    def test_source_and_factor_hashes_match_raw_codec_bytes_without_decomposition(self):
        matrix = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6).T
        expected = hashlib.sha256(matrix.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        with (
            mock.patch("torch.linalg.svd", side_effect=AssertionError("digest SVD")),
            mock.patch("torch.linalg.eigh", side_effect=AssertionError("digest eigh")),
            mock.patch("torch.linalg.qr", side_effect=AssertionError("digest QR")),
        ):
            for rows in (1, 2, 1024):
                self.assertEqual(tensor_sha256(matrix, tile_rows=rows), expected)

    def test_failed_decompositions_keep_the_counts_of_invoked_operations(self):
        matrix = torch.eye(4)
        with mock.patch("torch.linalg.qr", side_effect=RuntimeError("synthetic QR failure")):
            with self.assertRaises(RankFitError) as caught:
                fit_rank_directions(matrix, context(4, 4), 2)
        self.assertEqual(caught.exception.metrics["qr_count"], 1)
        self.assertEqual(caught.exception.metrics["svd_count"], 0)
        with mock.patch("torch.linalg.svd", side_effect=RuntimeError("synthetic SVD failure")):
            with self.assertRaises(RankFitError) as caught:
                fit_rank_directions(matrix, context(4, 4), 2)
        self.assertEqual(caught.exception.metrics["qr_count"], 5)
        self.assertEqual(caught.exception.metrics["svd_count"], 1)
        self.assertEqual(caught.exception.metrics["decomposition_count"], 1)

    def test_failed_projection_subphases_keep_attempt_counts_and_immutable_metrics(self):
        matrix = torch.eye(4)
        cases = (
            ("torch.linalg.cholesky_ex", "conditioning_cholesky", 1, 0, 1),
            ("torch.cholesky_inverse", "conditioning_inverse", 1, 1, 1),
            ("reelcache.factorization._canonical_signs", "basis_rounding", 0, 0, 0),
        )
        for target, phase, cholesky, inverse, projections in cases:
            with mock.patch(target, side_effect=RuntimeError("injected projection backend")):
                with self.assertRaises(RankProjectionError) as caught:
                    project_rank_directions(matrix, matrix, 4, tile_rows=2)
            error = caught.exception
            self.assertEqual(error.metrics["failure_phase"], phase)
            self.assertEqual(error.metrics["failure_kind"], "backend")
            self.assertEqual(error.metrics["projection_count"], projections)
            self.assertEqual(error.metrics["cholesky_count"], cholesky)
            self.assertEqual(error.metrics["cholesky_inverse_count"], inverse)
            self.assertEqual(error.metrics["coefficient_projection_count"], 0)
            self.assertEqual(error.metrics["residual_evaluation_count"], 0)
            self.assertEqual(error.metrics["svd_count"], 0)
            self.assertEqual(error.metrics["qr_count"], 0)
            self.assertTrue(error.metrics["operation_counts_complete"])
            self.assertGreater(error.metrics["basis_round_seconds"], 0)
            if projections:
                self.assertGreater(error.metrics["condition_seconds"], 0)
                self.assertGreater(error.metrics["projection_seconds"], 0)
            with self.assertRaises(TypeError):
                error.metrics["projection_count"] = 100
            restored = pickle.loads(pickle.dumps(error))
            self.assertEqual(dict(restored.metrics), dict(error.metrics))

    def test_failed_reconstruction_retains_partial_guard_attempts(self):
        matrix = torch.eye(4)
        real_matmul = torch.Tensor.__matmul__
        tiles = 0

        def fail_second_guard(left, right):
            nonlocal tiles
            if left.dtype == torch.float32 and tuple(left.shape) == (2, 4):
                tiles += 1
                if tiles == 2:
                    raise RuntimeError("second reconstruction tile")
            return real_matmul(left, right)

        with mock.patch.object(torch.Tensor, "__matmul__", new=fail_second_guard):
            with self.assertRaises(RankProjectionError) as caught:
                project_rank_directions(matrix, matrix, 4, tile_rows=2)
        metrics = caught.exception.metrics
        self.assertEqual(tiles, 2)
        self.assertEqual(metrics["failure_phase"], "residual_guard")
        self.assertEqual(metrics["matmul_count"], 4)
        self.assertEqual(metrics["projection_count"], 1)
        self.assertEqual(metrics["coefficient_projection_count"], 1)
        self.assertEqual(metrics["source_norm_count"], 1)
        self.assertEqual(metrics["residual_evaluation_count"], 1)
        self.assertEqual(metrics["residual_tile_count"], 2)
        self.assertGreater(metrics["projection_coefficient_seconds"], 0)
        self.assertGreater(metrics["projection_guard_seconds"], 0)
        self.assertTrue(metrics["operation_counts_complete"])

    def test_uninstrumented_projection_entry_does_not_claim_zero_work(self):
        matrix = torch.eye(4)
        with mock.patch(
            "reelcache.factorization.project_onto_basis", side_effect=RuntimeError("opaque projection entry"),
        ):
            with self.assertRaises(RankProjectionError) as caught:
                project_rank_directions(matrix, matrix, 4, tile_rows=2)
        metrics = caught.exception.metrics
        self.assertFalse(metrics["operation_counts_complete"])
        self.assertEqual(metrics["failure_phase"], "projection_entry")
        for name in ("projection_count", "matmul_count", "source_norm_count", "cholesky_count"):
            self.assertIsNone(metrics[name])
        self.assertGreater(metrics["projection_seconds"], 0)
        self.assertGreater(metrics["basis_round_seconds"], 0)
        self.assertEqual(metrics["svd_count"], 0)

    def test_failed_coefficient_backend_does_not_invent_a_residual_pass(self):
        matrix = torch.eye(4)
        real_matmul = torch.Tensor.__matmul__

        def fail_coefficients(left, right):
            if left.dtype == torch.float32:
                raise RuntimeError("coefficient backend")
            return real_matmul(left, right)

        with mock.patch.object(torch.Tensor, "__matmul__", new=fail_coefficients):
            with self.assertRaises(RankProjectionError) as caught:
                project_rank_directions(matrix, matrix, 4, tile_rows=2)
        metrics = caught.exception.metrics
        self.assertEqual(metrics["failure_phase"], "coefficient_projection")
        self.assertEqual(metrics["projection_count"], 1)
        self.assertEqual(metrics["matmul_count"], 2)
        self.assertEqual(metrics["cholesky_count"], 1)
        self.assertEqual(metrics["cholesky_inverse_count"], 1)
        self.assertEqual(metrics["coefficient_projection_count"], 1)
        self.assertEqual(metrics["source_norm_count"], 0)
        self.assertEqual(metrics["residual_evaluation_count"], 0)
        self.assertEqual(metrics["residual_tile_count"], 0)
        self.assertGreater(metrics["projection_coefficient_seconds"], 0)
        self.assertEqual(metrics["projection_guard_seconds"], 0)


class FixedDetectorTests(unittest.TestCase):
    def test_recipe_is_immutable_complete_and_shared_across_dimensions(self):
        spec = DetectorSpec()
        self.assertEqual(DetectorSpec.from_dict(spec.to_dict()), spec)
        self.assertEqual(DetectorSpec().fingerprint, spec.fingerprint)
        for change in ({"initial_width": 2}, {"base_seed": 9}, {"device": "cuda"}):
            with self.assertRaises(ConfigurationError):
                dataclasses.replace(spec, **change)
        for count in (True, False, 0, -1, 2.0, None):
            with self.assertRaises(ConfigurationError):
                dataclasses.replace(spec, cpu_threads=count)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.base_seed = 1
        payload = spec.to_dict()
        del payload["implementation_fingerprint"]
        with self.assertRaises(ConfigurationError):
            DetectorSpec.from_dict(payload)
        for changed in (
            dataclasses.replace(spec, implementation_fingerprint="0" * 64),
            dataclasses.replace(spec, backend_fingerprint="0" * 64),
        ):
            with self.assertRaises(ConfigurationError):
                acquire_anchor_directions(torch.eye(4), context(4, 4), spec=changed)
        with mock.patch("torch.get_num_threads", return_value=2):
            with self.assertRaises(ConfigurationError):
                acquire_anchor_directions(torch.eye(4), context(4, 4), spec=spec)

    def test_explicit_thread_recipes_match_runtime_without_setting_global_threads(self):
        matrix = torch.randn(64, 32, generator=torch.Generator().manual_seed(71))
        ctx = context(64, 32)
        previous_threads = torch.get_num_threads()
        rng = torch.get_rng_state().clone()
        fingerprints = {DetectorSpec().fingerprint}
        for count in (2, 8):
            with self.subTest(cpu_threads=count):
                spec = DetectorSpec(cpu_threads=count)
                fingerprints.add(spec.fingerprint)
                self.assertEqual(DetectorSpec.from_dict(spec.to_dict()), spec)
                self.assertEqual(spec.to_dict()["cpu_threads"], count)
                try:
                    torch.set_num_threads(count)
                    spec.validate_runtime()
                    with mock.patch("torch.set_num_threads", side_effect=AssertionError("helper changed threads")):
                        first = acquire_anchor_directions(matrix, ctx, spec=spec, solver_workspace_bytes=0)
                        second = acquire_anchor_directions(matrix, ctx, spec=spec, solver_workspace_bytes=0)
                    self.assertEqual(torch.get_num_threads(), count)
                    self.assertEqual(first.metrics["cpu_threads"], count)
                    self.assertEqual(first.metrics["detector_device"], "cpu")
                    self.assertEqual((first.r95, first.r99), (second.r95, second.r99))
                    self.assertTrue(torch.equal(first.directions, second.directions))
                    self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                    with self.assertRaises(ConfigurationError):
                        acquire_anchor_directions(matrix, ctx, spec=DetectorSpec(), solver_workspace_bytes=0)
                finally:
                    torch.set_num_threads(previous_threads)
        self.assertEqual(len(fingerprints), 3)
        self.assertEqual(torch.get_num_threads(), previous_threads)

    def test_both_features_share_final_expanded_fit_and_private_rng(self):
        matrix = torch.eye(160)
        before = torch.get_rng_state().clone()
        with mock.patch("torch.linalg.svd", wraps=torch.linalg.svd) as svd:
            result = acquire_anchor_directions(matrix, context(160, 160), solver_workspace_bytes=0)
        self.assertEqual(svd.call_count, 2)
        self.assertEqual([item["width"] for item in result.attempts], [128, 160])
        self.assertEqual((result.r95, result.r99), (result.fit.required_rank(.95), result.fit.required_rank(.99)))
        self.assertEqual(result.metrics["svd_count"], 2)
        self.assertEqual(result.metrics["qr_count"], 10)
        self.assertEqual(result.metrics["source_copy_count"], 1)
        self.assertEqual(result.metrics["source_copy_bytes"], matrix.numel() * 4)
        self.assertEqual(result.source_sha256, tensor_sha256(matrix))
        self.assertEqual(result.directions.device.type, "cpu")
        self.assertEqual(result.directions.dtype, torch.float32)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertNotIn("directions", result.to_dict())
        self.assertEqual(result.to_dict()["features"]["0.95"], result.r95)

    def test_private_seed_uses_geometry_position_key_not_model_labels(self):
        spec = DetectorSpec()
        ctx = context(12, 6)
        relabeled = dataclasses.replace(
            ctx, model=ModelIdentity("another-label", "another-revision", 1, 6, "tokens_channels", "float32"),
        )
        self.assertEqual(spec.seed_for(ctx), spec.seed_for(relabeled))
        self.assertNotEqual(spec.seed_for(ctx), spec.seed_for(dataclasses.replace(ctx, token_start=12)))
        self.assertNotEqual(spec.seed_for(ctx), spec.seed_for(context(12, 6, target="V")))
        matrix = torch.randn(12, 6, generator=torch.Generator().manual_seed(9))
        first = acquire_anchor_directions(matrix, ctx, spec=spec, solver_workspace_bytes=0)
        second = acquire_anchor_directions(matrix, relabeled, spec=spec, solver_workspace_bytes=0)
        self.assertTrue(torch.equal(first.directions, second.directions))
        self.assertEqual((first.r95, first.r99), (second.r95, second.r99))

    def test_cap_miss_remains_explicit_without_oracle_substitution(self):
        from reelcache.factorization import DirectionFit

        with (
            mock.patch.object(DirectionFit, "required_rank", return_value=None),
            mock.patch("torch.linalg.eigh", side_effect=AssertionError("detector oracle substitution")),
        ):
            result = acquire_anchor_directions(torch.eye(4), context(4, 4), solver_workspace_bytes=0)
        self.assertIsNone(result.r95)
        self.assertIsNone(result.r99)
        self.assertEqual(result.to_dict()["status"], "unresolved")
        self.assertEqual(result.fit.width, 4)
        self.assertEqual(result.metrics["svd_count"], 1)

    def test_zero_has_math_zero_features_without_solver_or_global_rng(self):
        with (
            mock.patch("torch.linalg.svd", side_effect=AssertionError("zero detector SVD")),
            mock.patch("torch.linalg.qr", side_effect=AssertionError("zero detector QR")),
            mock.patch("torch.randn", side_effect=AssertionError("zero detector RNG")),
        ):
            result = acquire_anchor_directions(torch.zeros(4, 4), context(4, 4), solver_workspace_bytes=0)
        self.assertEqual((result.r95, result.r99), (0, 0))
        self.assertEqual(result.metrics["svd_count"], 0)
        self.assertEqual(result.metrics["source_copy_count"], 1)

    def test_workspace_denial_keeps_completed_attempts_and_copy_charge(self):
        matrix = torch.eye(160)
        ctx = context(160, 160)
        allowance = matrix.numel() * matrix.element_size() + rank_fit_workspace_bytes(
            matrix.shape, 128, tile_rows=1024,
        )
        with self.assertRaises(DetectorAcquisitionError) as caught:
            acquire_anchor_directions(matrix, ctx, solver_workspace_bytes=0, max_workspace_bytes=allowance)
        work = caught.exception.metrics
        self.assertEqual(work["status"], "OVER_BUDGET")
        self.assertEqual(work["fit_attempts"], 1)
        self.assertEqual(work["svd_count"], 1)
        self.assertEqual(work["qr_count"], 5)
        self.assertEqual(work["source_copy_bytes"], matrix.numel() * 4)
        self.assertEqual(work["requested_fit_width"], 160)
        self.assertEqual(work["source_sha256"], tensor_sha256(matrix))
        self.assertGreater(work["elapsed_seconds"], 0)
        with self.assertRaises(DetectorAcquisitionError) as unadmitted:
            acquire_anchor_directions(matrix, ctx, solver_workspace_bytes=0, max_workspace_bytes=1)
        self.assertEqual(unadmitted.exception.metrics["source_copy_count"], 0)
        self.assertEqual(unadmitted.exception.metrics["svd_count"], 0)

    def test_failed_fixed_fit_keeps_actual_solver_counts(self):
        with mock.patch("torch.linalg.svd", side_effect=RuntimeError("fixed detector backend")):
            with self.assertRaises(DetectorAcquisitionError) as caught:
                acquire_anchor_directions(torch.eye(4), context(4, 4), solver_workspace_bytes=0)
        work = caught.exception.metrics
        self.assertEqual(work["svd_count"], 1)
        self.assertEqual(work["qr_count"], 5)
        self.assertTrue(work["operation_counts_complete"])
        self.assertEqual(work["source_copy_count"], 1)
        self.assertEqual(pickle.loads(pickle.dumps(caught.exception)).metrics, work)


class DetectorBackendIdentityTests(CoreWorkspaceTest):
    def setUp(self):
        super().setUp()
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(8)
        self.addCleanup(torch.set_num_threads, previous_threads)

    def test_cuda_visibility_report_does_not_change_cpu_acquisition(self):
        spec = DetectorSpec(cpu_threads=8)
        report = torch.__config__.show()
        augmented = with_cuda_runtime_diagnostics(report)
        self.assertNotEqual(report, augmented)
        matrix = torch.eye(160)
        before = torch.get_rng_state().clone()
        for target in ("K", "V"):
            with self.subTest(target=target):
                ctx = context(160, 160, target=target)
                offline = acquire_anchor_directions(matrix, ctx, spec=spec, solver_workspace_bytes=0)
                with (
                    mock.patch("torch.__config__.show", return_value=augmented),
                    mock.patch("torch.set_num_threads", side_effect=AssertionError("changed caller threads")),
                    mock.patch("torch.cuda._lazy_init", side_effect=AssertionError("initialized CUDA")),
                    mock.patch("torch.load", side_effect=AssertionError("loaded weights")),
                ):
                    self.assertEqual(DetectorSpec(cpu_threads=8).fingerprint, spec.fingerprint)
                    online = acquire_anchor_directions(matrix, ctx, spec=spec, solver_workspace_bytes=0)
                self.assertEqual([item["width"] for item in online.attempts], [128, 160])
                self.assertEqual((online.r95, online.r99), (offline.r95, offline.r99))
                self.assertEqual(online.source_sha256, offline.source_sha256)
                self.assertEqual(online.metrics["factorization_seed"], offline.metrics["factorization_seed"])
                self.assertEqual(online.metrics["svd_count"], 2)
                self.assertEqual(online.metrics["qr_count"], 10)
                self.assertTrue(torch.equal(online.directions, offline.directions))
                self.assertEqual(online.directions.device.type, "cpu")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(torch.get_num_threads(), 8)
        self.assertFalse(torch.cuda.is_initialized())

    def test_cpu_build_details_and_unknown_lines_remain_strictly_bound(self):
        spec = DetectorSpec(cpu_threads=8)
        report = with_cuda_runtime_diagnostics(torch.__config__.show())
        changes = (
            report.replace("  - Build settings: ", "  - Build settings: DIFFERENT_BUILD=ON, ", 1),
            report.replace("  - CPU capability usage: ", "  - CPU capability usage: changed-", 1),
            report + "  - Unknown backend setting: changed\n",
            report + "  - CUDA unknown setting: changed\n",
        )
        for changed in changes:
            with (
                self.subTest(report=changed),
                mock.patch("torch.__config__.show", return_value=changed),
                mock.patch("torch.set_num_threads", side_effect=AssertionError("changed caller threads")),
                self.assertRaisesRegex(ConfigurationError, "backend_fingerprint: expected"),
            ):
                spec.validate_runtime()
        for name, changed in (
            ("torch.__version__", "different-torch"),
            ("reelcache.factorization.platform.machine", lambda: "different-architecture"),
            ("torch.backends.cpu.get_cpu_capability", lambda: "different-cpu"),
        ):
            with self.subTest(component=name), mock.patch(name, new=changed):
                with self.assertRaisesRegex(ConfigurationError, "backend_fingerprint: expected"):
                    spec.validate_runtime()

    def test_runtime_mismatch_names_every_component_without_mutating_settings(self):
        spec = DetectorSpec(cpu_threads=8)
        changed = dataclasses.replace(
            spec, implementation_fingerprint="0" * 64, backend_fingerprint="1" * 64, cpu_threads=7,
        )
        with (
            mock.patch("torch.set_num_threads", side_effect=AssertionError("changed caller threads")),
            self.assertRaises(ConfigurationError) as caught,
        ):
            changed.validate_runtime()
        message = str(caught.exception)
        for name in ("implementation_fingerprint", "backend_fingerprint", "cpu_threads"):
            self.assertIn(f"{name}: expected", message)
        self.assertIn(spec.implementation_fingerprint, message)
        self.assertIn(spec.backend_fingerprint, message)
        self.assertIn("cpu_threads: expected 7, got 8", message)

    def test_source_relocation_preserves_bytes_binding_but_byte_changes_reject(self):
        from reelcache import factorization

        spec = DetectorSpec(cpu_threads=8)
        original = Path(factorization.__file__).read_bytes()
        for index, data in enumerate((original, original + b"\n")):
            with self.subTest(changed_bytes=bool(index)):
                path = self.workspace / f"factorization_{index}.py"
                path.write_bytes(data)
                name = f"reelcache._detector_relocated_{self.workspace.name}_{index}"
                module_spec = importlib.util.spec_from_file_location(name, path)
                relocated = importlib.util.module_from_spec(module_spec)
                with mock.patch.dict(sys.modules, {name: relocated}):
                    module_spec.loader.exec_module(relocated)
                    bound = relocated.DetectorSpec.from_dict(spec.to_dict())
                    if index:
                        with self.assertRaisesRegex(ConfigurationError, "implementation_fingerprint: expected"):
                            bound.validate_runtime()
                    else:
                        bound.validate_runtime()
                        self.assertEqual(relocated.DetectorSpec(cpu_threads=8).fingerprint, spec.fingerprint)


if __name__ == "__main__":
    unittest.main()
