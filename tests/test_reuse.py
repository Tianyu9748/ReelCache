import unittest
from unittest import mock

import torch

from reelcache.codecs import decode_tensor, encode_tensor, encoded_nbytes
from reelcache.errors import ConfigurationError
from reelcache.factorization import fit_basis
from reelcache.reuse import factorize_tensor, try_reuse_basis
from reelcache.types import CodecSpec, MatrixKey, ModelIdentity, TensorContext


def context(rows=8, channels=4):
    return TensorContext(
        ModelIdentity("reuse-test", "fixture", 1, channels, "native_kv", "float32"),
        MatrixKey(0, "V"), 0, 1, 1, rows, "none",
    )


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.context = context()
        self.matrix = torch.zeros(8, 4)
        self.matrix[:, 0] = torch.arange(8)
        self.matrix[:, 1] = torch.arange(8).remainder(3)
        self.basis = torch.eye(4, dtype=torch.float16)

    def test_acceptance_reuses_projection_and_never_calls_svd_or_eigh(self):
        decision = try_reuse_basis(self.matrix, self.context, self.basis, 2, 0.01)
        self.assertTrue(decision.accepted)
        with (
            mock.patch("reelcache.reuse.try_reuse_basis", return_value=decision),
            mock.patch("reelcache.reuse.fit_basis", side_effect=AssertionError("refit on acceptance")),
            mock.patch("torch.linalg.svd", side_effect=AssertionError("SVD on acceptance")),
            mock.patch("torch.linalg.eigh", side_effect=AssertionError("eigh on acceptance")),
            mock.patch("torch.svd_lowrank", side_effect=AssertionError("lowrank SVD on acceptance")),
        ):
            factors = factorize_tensor(
                self.matrix, self.context, 2, reuse_basis=self.basis, reuse_threshold=0.01,
            )
        self.assertIs(factors.coefficients, decision.projection.coefficients)
        self.assertEqual(factors.metrics["reuse_attempts"], 1)
        self.assertTrue(factors.metrics["reuse_accepted"])
        self.assertEqual(factors.metrics["decomposition_count"], 0)
        self.assertEqual(factors.metrics["projection_count"], 1)
        self.assertEqual(factors.metrics["fit_seconds"], 0)

    def test_accepted_encoder_does_not_reproject(self):
        decision = try_reuse_basis(self.matrix, self.context, self.basis, 2, 0.01)
        with (
            mock.patch("reelcache.reuse.try_reuse_basis", return_value=decision),
            mock.patch("reelcache.factorization.project_onto_basis", side_effect=AssertionError("reprojection")),
            mock.patch("reelcache.reuse.project_onto_basis", side_effect=AssertionError("reprojection")),
            mock.patch("torch.linalg.svd", side_effect=AssertionError("unexpected decomposition")),
        ):
            encoded = encode_tensor(
                self.matrix, self.context, CodecSpec("svd", 2),
                reuse_basis=self.basis, reuse_threshold=0.01,
            )
        coefficients = torch.frombuffer(
            bytearray(encoded.payloads["coefficients.f32"]), dtype=torch.float32,
        ).reshape(8, 2)
        self.assertTrue(torch.equal(coefficients, decision.projection.coefficients))

    def test_rejection_refits_and_records_failed_projection_cost(self):
        basis = self.basis[[2, 3]]
        with mock.patch("reelcache.reuse.fit_basis", wraps=fit_basis) as fitted:
            result = factorize_tensor(
                self.matrix, self.context, 2, reuse_basis=basis, reuse_threshold=0.01,
            )
        fitted.assert_called_once()
        self.assertEqual(result.metrics["reuse_reason"], "residual_above_threshold")
        self.assertFalse(result.metrics["reuse_accepted"])
        self.assertEqual(result.metrics["reuse_rejections"], 1)
        self.assertEqual(result.metrics["reuse_attempts"], 1)
        self.assertEqual(result.metrics["decomposition_count"], 1)
        self.assertEqual(result.metrics["svd_count"], 1)
        self.assertEqual(result.metrics["eigh_count"], 0)
        self.assertEqual(result.metrics["projection_count"], 2)
        self.assertEqual(result.metrics["attempted_relative_residual"], 1.0)
        self.assertGreater(result.metrics["reuse_seconds"], 0)
        self.assertGreater(result.metrics["fit_seconds"], 0)
        self.assertLess(result.metrics["projection_relative_residual"], 1e-3)

    def test_smaller_rank_view_preserves_and_charges_whole_basis(self):
        original = self.basis.clone()
        packets = [
            encode_tensor(
                self.matrix, self.context, CodecSpec("svd", rank),
                reuse_basis=self.basis, reuse_threshold=0.01,
            )
            for rank in (4, 2)
        ]
        full, view = packets
        self.assertEqual(full.basis, view.basis)
        self.assertEqual(view.metadata["basis"]["shape"], [4, 4])
        self.assertEqual(view.metadata["basis"]["view_rank"], 2)
        self.assertEqual(view.metrics["basis_bytes"], 4 * 4 * 2)
        self.assertEqual(view.metrics["canonical_basis_rank"], 4)
        self.assertEqual(view.metadata["coefficient_shape"], [8, 2])
        self.assertEqual(view.metrics["representation_bytes"], encoded_nbytes(view))
        self.assertTrue(torch.equal(self.basis, original))
        self.assertTrue(torch.equal(decode_tensor(view), self.matrix))
        self.assertTrue(torch.equal(decode_tensor(view, columns=slice(1, 3)), self.matrix[:, 1:3]))
        self.basis.zero_()
        self.assertTrue(torch.equal(decode_tensor(view), self.matrix))

    def test_rank_increase_refits_never_zero_pads(self):
        result = factorize_tensor(
            self.matrix, self.context, 2, reuse_basis=self.basis[:1], reuse_threshold=1,
        )
        self.assertEqual(result.metrics["reuse_reason"], "rank_increase_requires_refit")
        self.assertEqual(result.metrics["decomposition_count"], 1)
        self.assertEqual(result.metrics["projection_count"], 1)
        self.assertEqual(result.basis.shape, (2, 4))
        self.assertGreater(result.basis[1].float().norm().item(), 0.99)
        self.assertEqual(result.metrics["view_rank"], 2)
        self.assertEqual(result.metrics["canonical_basis_rank"], 2)

    def test_nonfinite_and_ill_conditioned_candidates_refit_explicitly(self):
        cases = [
            (torch.full((2, 4), float("nan"), dtype=torch.float16), "nonfinite_basis"),
            (torch.zeros(2, 4, dtype=torch.float16), "ill_conditioned_basis"),
            (torch.ones(2, 4, dtype=torch.float16), "ill_conditioned_basis"),
            (torch.ones(2, 3, dtype=torch.float16), "incompatible_basis_geometry"),
        ]
        for basis, reason in cases:
            with self.subTest(reason=reason):
                result = factorize_tensor(
                    self.matrix, self.context, 2, reuse_basis=basis, reuse_threshold=0.5,
                )
                self.assertEqual(result.metrics["reuse_reason"], reason)
                self.assertEqual(result.metrics["reuse_attempts"], 1)
                self.assertEqual(result.metrics["reuse_rejections"], 1)
                self.assertEqual(result.metrics["decomposition_count"], 1)
                self.assertEqual(result.metrics["projection_count"], 1)

    def test_threshold_boundary_and_zero_cases_are_explicit(self):
        matrix = torch.zeros(8, 4)
        matrix[:, 0] = 3
        matrix[:, 2] = 4
        decision = try_reuse_basis(matrix, self.context, self.basis, 2, 0.8)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "threshold_boundary")
        self.assertTrue(try_reuse_basis(matrix, self.context, self.basis, 2, 0.81).accepted)
        exact = try_reuse_basis(self.matrix, self.context, self.basis, 2, 0)
        self.assertTrue(exact.accepted)
        self.assertEqual(exact.reason, "exact_projection")
        zero = try_reuse_basis(torch.zeros_like(matrix), self.context, self.basis, 2, 0)
        self.assertTrue(zero.accepted)
        self.assertEqual(zero.reason, "zero_input")

    def test_invalid_threshold_or_candidate_precision_is_not_silent(self):
        for threshold in (-0.1, 1.1, float("nan"), float("inf"), True, "0.1"):
            with self.subTest(threshold=threshold), self.assertRaises(ConfigurationError):
                factorize_tensor(
                    self.matrix, self.context, 2,
                    reuse_basis=self.basis, reuse_threshold=threshold,
                )
        with self.assertRaises(ConfigurationError):
            factorize_tensor(self.matrix, self.context, 2, reuse_basis=self.basis)
        with self.assertRaises(ConfigurationError):
            factorize_tensor(
                self.matrix, self.context, 2,
                reuse_basis=self.basis.float(), reuse_threshold=0.1,
            )

    def test_first_chunk_threshold_without_candidate_and_rng_preservation(self):
        before = torch.get_rng_state().clone()
        first = factorize_tensor(self.matrix, self.context, 2, reuse_threshold=0.1)
        accepted = factorize_tensor(
            self.matrix, self.context, 2, reuse_basis=self.basis, reuse_threshold=0.1,
        )
        rejected = factorize_tensor(
            self.matrix, self.context, 2, reuse_basis=self.basis[[2, 3]], reuse_threshold=0.1,
        )
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(first.metrics["reuse_attempts"], 0)
        self.assertEqual(first.metrics["reuse_reason"], "no_candidate")
        self.assertEqual(accepted.metrics["decomposition_count"], 0)
        self.assertEqual(rejected.metrics["decomposition_count"], 1)


if __name__ == "__main__":
    unittest.main()
