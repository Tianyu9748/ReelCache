import math
import random
import unittest
from unittest import mock

import numpy as np
import torch

from reelcache.errors import ConfigurationError
from reelcache.factorization import (
    basis_condition_bound, fit_basis, project_onto_basis, validate_matrix,
)
from reelcache.types import MatrixKey, ModelIdentity, TensorContext


def context(rows, channels, *, target="K", key_space=None, dtype="float32"):
    model = ModelIdentity("factor-test", "fixture", 1, channels, "tokens_channels", dtype)
    return TensorContext(
        model, MatrixKey(0, target), 0, 1, 1, rows,
        key_space if key_space is not None else ("pre_rope" if target == "K" else "none"),
    )


class FactorizationTests(unittest.TestCase):
    def test_actual_rounded_basis_and_direct_reference(self):
        theta = 0.3
        original = torch.tensor([
            [math.cos(theta), math.sin(theta)],
            [-math.sin(theta), math.cos(theta)],
        ])
        basis = original.half()
        matrix = torch.tensor([[1.0, 2.0], [3.0, -2.0], [-1.0, 4.0]])
        result = project_onto_basis(matrix, basis, tile_rows=1)
        projection = matrix @ basis.float().T
        direct = matrix.double() - (projection @ basis.float()).double()
        reference = float(torch.linalg.vector_norm(direct) / matrix.double().norm())
        self.assertTrue(torch.equal(result.coefficients, projection))
        self.assertAlmostEqual(result.relative_residual, reference, places=7)
        naive = float((1 - projection.square().sum() / matrix.square().sum()).clamp_min(0).sqrt())
        self.assertGreater(result.relative_residual, 1e-4)
        self.assertEqual(naive, 0.0)

    def test_fitted_coefficients_use_stored_basis_not_unrounded_svd(self):
        generator = torch.Generator().manual_seed(41)
        matrix = torch.randn(24, 7, generator=generator)
        result = fit_basis(matrix, context(24, 7), 4)
        self.assertEqual(result.basis.dtype, torch.float16)
        self.assertEqual(result.coefficients.dtype, torch.float32)
        self.assertTrue(torch.equal(result.coefficients, matrix @ result.basis.float().T))
        self.assertEqual(result.metrics["svd_count"], 1)
        self.assertEqual(result.metrics["eigh_count"], 0)
        self.assertEqual(result.metrics["decomposition_count"], 1)
        self.assertEqual(result.metrics["factorization_method"], "exact_svd")
        self.assertLess(result.metrics["basis_condition_bound"], 1.01)

    def test_exact_and_randomized_are_deterministic_without_rng_changes(self):
        generator = torch.Generator().manual_seed(103)
        matrix = torch.randn(48, 24, generator=generator)
        for method in ("exact", "randomized"):
            with self.subTest(method=method):
                torch_state = torch.get_rng_state().clone()
                numpy_state = np.random.get_state()
                python_state = random.getstate()
                first = fit_basis(matrix, context(48, 24), 3, method=method, seed=923)
                second = fit_basis(matrix, context(48, 24), 3, method=method, seed=923)
                self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
                self.assertEqual(python_state, random.getstate())
                current_numpy = np.random.get_state()
                self.assertEqual(numpy_state[0], current_numpy[0])
                np.testing.assert_array_equal(numpy_state[1], current_numpy[1])
                self.assertEqual(numpy_state[2:], current_numpy[2:])
                self.assertTrue(torch.equal(first.basis, second.basis))
                self.assertTrue(torch.equal(first.coefficients, second.coefficients))
                self.assertEqual(first.metrics["svd_count"], 1)
                self.assertEqual(first.metrics["eigh_count"], 0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_projection_never_decomposes_or_draws_random_numbers(self):
        matrix = torch.arange(40, dtype=torch.float32).reshape(10, 4)
        basis = torch.eye(4, dtype=torch.float16)
        with (
            mock.patch("torch.linalg.svd", side_effect=AssertionError("SVD on projection")),
            mock.patch("torch.linalg.eigh", side_effect=AssertionError("eigh on projection")),
            mock.patch("torch.randn", side_effect=AssertionError("RNG on projection")),
        ):
            result = project_onto_basis(matrix, basis)
        self.assertEqual(result.relative_residual, 0)
        self.assertTrue(torch.equal(result.coefficients, matrix))

    def test_large_truncated_auto_path_uses_private_randomized_svd(self):
        generator = torch.Generator().manual_seed(27)
        matrix = (
            torch.randn(600, 8, generator=generator)
            @ torch.randn(8, 520, generator=generator)
        )
        before = torch.get_rng_state().clone()
        factors = fit_basis(matrix, context(600, 520), 8)
        self.assertEqual(factors.metrics["factorization_method"], "randomized_svd")
        self.assertEqual(factors.metrics["sketch_rank"], 16)
        self.assertEqual(factors.metrics["qr_count"], 5)
        self.assertEqual(factors.metrics["decomposition_count"], 1)
        self.assertLess(factors.metrics["projection_relative_residual"], 0.001)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_k_requires_pre_rope_and_does_not_renormalize(self):
        matrix = torch.tensor([[7.0, 0.0], [4.0, 0.0], [8.0, 0.0]])
        for key_space in ("post_rope", "none"):
            with self.subTest(key_space=key_space), self.assertRaises(ConfigurationError):
                fit_basis(matrix, context(3, 2, key_space=key_space), 1)
        factors = fit_basis(matrix, context(3, 2), 1)
        self.assertTrue(torch.equal(factors.coefficients @ factors.basis.float(), matrix))
        factors_v = fit_basis(matrix, context(3, 2, target="V"), 1)
        self.assertTrue(torch.equal(factors_v.coefficients @ factors_v.basis.float(), matrix))

    def test_zero_input_has_explicit_zero_residual(self):
        matrix = torch.zeros(12, 4)
        projection = project_onto_basis(matrix, torch.eye(4, dtype=torch.float16)[:2])
        self.assertTrue(projection.zero_input)
        self.assertEqual(projection.relative_residual, 0)
        self.assertEqual(projection.source_norm, 0)
        fitted = fit_basis(matrix, context(12, 4), 2)
        self.assertTrue(fitted.metrics["zero_input"])
        self.assertEqual(fitted.coefficients.count_nonzero().item(), 0)

    def test_tiny_float64_is_not_misclassified_as_zero(self):
        matrix = torch.full((5, 3), 1e-250, dtype=torch.float64)
        result = project_onto_basis(matrix, torch.eye(3, dtype=torch.float16))
        self.assertFalse(result.zero_input)
        self.assertEqual(result.relative_residual, 1.0)
        self.assertGreater(result.source_norm, 0)
        self.assertEqual(result.source_norm, result.residual_norm)

    def test_rank_dimensions_dtype_and_finite_checks(self):
        matrix = torch.ones(5, 3)
        for rank in (0, -1, 4, 1.5, True):
            with self.subTest(rank=rank), self.assertRaises(ConfigurationError):
                fit_basis(matrix, context(5, 3), rank)
        with self.assertRaises(ConfigurationError):
            fit_basis(matrix, context(6, 3), 2)
        with self.assertRaises(ConfigurationError):
            fit_basis(matrix.double(), context(5, 3), 2)
        for value in (float("nan"), float("inf")):
            bad = matrix.clone()
            bad[0, 0] = value
            with self.assertRaises(ConfigurationError):
                fit_basis(bad, context(5, 3), 2)
        huge = torch.full((5, 3), 1e100, dtype=torch.float64)
        with self.assertRaises(ConfigurationError):
            fit_basis(huge, context(5, 3, dtype="float64"), 2)
        self.assertIsNotNone(validate_matrix(
            huge, context(5, 3, dtype="float64"), lossy=False,
        ))

    def test_explicit_declared_matrix_dtype_remains_strict(self):
        model = ModelIdentity(
            "declared-dtype-fixture", "synthetic", 1, 4, "native_kv",
            {"K": "float32", "V": "bfloat16"},
        )
        values = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        for target, dtype, key_space in (
            ("K", torch.float32, "pre_rope"), ("V", torch.bfloat16, "none"),
        ):
            ctx = TensorContext(model, MatrixKey(0, target), 0, 1, 2, 3, key_space)
            matrix = values.to(dtype)
            factors = fit_basis(matrix, ctx, 2)
            self.assertTrue(torch.equal(
                factors.coefficients, matrix.float() @ factors.basis.float().T,
            ))
            with self.assertRaises(ConfigurationError):
                fit_basis(matrix.double(), ctx, 2)

    def test_basis_rank_geometry_precision_and_condition(self):
        matrix = torch.ones(5, 3)
        bad_bases = [
            torch.eye(3), torch.ones(2, 2, dtype=torch.float16),
            torch.zeros(2, 3, dtype=torch.float16),
            torch.ones(2, 3, dtype=torch.float16),
            torch.full((2, 3), float("nan"), dtype=torch.float16),
        ]
        for basis in bad_bases:
            with self.subTest(basis=basis), self.assertRaises(ConfigurationError):
                project_onto_basis(matrix, basis)
        near_singular = torch.tensor([[1, 1, 0], [1, 1.001, 0]], dtype=torch.float16)
        self.assertGreater(basis_condition_bound(near_singular), 1e6)
        with self.assertRaises(ConfigurationError):
            project_onto_basis(matrix, near_singular)
        with self.assertRaises(ConfigurationError):
            project_onto_basis(matrix, torch.eye(3, dtype=torch.float16)[:1], rank=2)
        for tile_rows in (0, -1, True, 1.5):
            with self.assertRaises(ConfigurationError):
                project_onto_basis(matrix, torch.eye(3, dtype=torch.float16), tile_rows=tile_rows)

    def test_noncontiguous_input_and_mixed_native_channel_counts(self):
        matrix = torch.arange(40, dtype=torch.float32).reshape(4, 10).T
        self.assertFalse(matrix.is_contiguous())
        model = ModelIdentity("gqa", "native", 1, {"K": 4, "V": 2}, "native_kv", "float32")
        key_context = TensorContext(model, MatrixKey(0, "K"), 8, 2, 1, 5, "pre_rope")
        result = fit_basis(matrix, key_context, 2)
        self.assertEqual(result.basis.shape, (2, 4))
        value_context = TensorContext(model, MatrixKey(0, "V"), 8, 2, 1, 5, "none")
        with self.assertRaises(ConfigurationError):
            fit_basis(matrix, value_context, 2)


if __name__ == "__main__":
    unittest.main()
