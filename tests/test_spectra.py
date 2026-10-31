import copy
import gc
import hashlib
import json
import math
import os
import pickle
import random
import time
import unittest
import weakref
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from reelcache.errors import CapacityError, ConfigurationError, IntegrityError, StateError
from reelcache.spectra import (
    HASH_FORMAT,
    SCHEMA_VERSION,
    SpectrumNumericalError,
    SpectrumNumerics,
    SpectrumResult,
    SpectrumRowShard,
    SpectrumSource,
    SpectrumTimings,
    StreamingGramAccumulator,
    direct_svd_crosscheck,
    exact_spectrum,
    spectrum_from_row_shards,
    write_spectrum_json,
)


def source(matrix, *, target="V", **kwargs):
    return SpectrumSource(
        identity="synthetic-original-v1",
        shape=tuple(matrix.shape),
        dtype=str(matrix.dtype).removeprefix("torch."),
        target=target,
        key_space="pre_rope" if target == "K" else "none",
        provenance={"fixture": True, "boundary": 0, "positions": [0, matrix.shape[0]]},
        **kwargs,
    )


def spectrum(matrix, **kwargs):
    return exact_spectrum(matrix, source=source(matrix), **kwargs)


def tie_matrix(tau):
    rows = 20 if tau == 0.95 else 100
    matrix = torch.zeros(rows, 2, dtype=torch.float64)
    matrix[:-1, 0] = 1.0
    matrix[-1, 1] = 1.0
    return matrix


class SpectrumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assertBracket(self, result, tau, expected):
        bracket = result.rank_at(tau)
        self.assertEqual((bracket.lower, bracket.upper), expected)
        if not result.zero_energy:
            self.assertLess(bracket.cdf_before_lower_bounds[1], tau)
            self.assertLessEqual(tau, bracket.cdf_at_upper_bounds[0])
        self.assertEqual(bracket.rank, expected[0] if expected[0] == expected[1] else None)
        return bracket

    def artifact_path(self):
        path = Path(f".test-spectra-{os.getpid()}-{time.monotonic_ns()}.json")
        self.assertFalse(path.exists())
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def assertFailureMetrics(self, error, *, eigh=0, eigvalsh=0, phase):
        metrics = error.metrics
        self.assertIsNotNone(metrics)
        self.assertTrue(metrics["operation_counts_complete"])
        self.assertEqual(metrics["decomposition_count"], eigh + eigvalsh)
        self.assertEqual(metrics["eigh_count"], eigh)
        self.assertEqual(metrics["eigvalsh_count"], eigvalsh)
        self.assertEqual(metrics["svd_count"], 0)
        self.assertEqual(metrics["failure_phase"], phase)
        for name, value in metrics.items():
            if name.endswith("_seconds"):
                self.assertTrue(math.isfinite(value))
                self.assertGreaterEqual(value, 0.0)
        self.assertEqual(
            metrics["total_seconds"],
            metrics["exclusive_pipeline_seconds"] + metrics["eigensolve_seconds"],
        )
        self.assertLessEqual(metrics["eigensolve_seconds"], metrics["finalization_seconds"])
        self.assertGreater(metrics["workspace_admission_bytes"], 0)
        self.assertLessEqual(metrics["workspace_admission_bytes"], metrics["workspace_limit_bytes"])
        self.assertFalse(any(isinstance(value, torch.Tensor) for value in metrics.values()))
        json.dumps(dict(metrics), allow_nan=False)
        with self.assertRaises(TypeError):
            metrics["decomposition_count"] = 99
        return metrics

    def test_analytical_rectangular_complete_squared_spectrum(self):
        matrix = torch.zeros(6, 3, dtype=torch.float64)
        matrix[:3] = torch.diag(torch.tensor([5.0, 2.0, 1.0], dtype=torch.float64))
        result = spectrum(matrix, tile_rows=2)
        self.assertEqual(len(result.eigenvalues_scaled), 3)
        self.assertEqual(result.energy, 30.0)
        np.testing.assert_allclose(
            np.array(result.eigenvalues_scaled) * 2.0 ** (2 * result.scale_exponent),
            [25.0, 4.0, 1.0], rtol=1e-14, atol=1e-14,
        )
        self.assertEqual((result.r95, result.r99), (2, 3))
        self.assertBracket(result, 0.95, (2, 2))
        self.assertBracket(result, 0.99, (3, 3))
        self.assertAlmostEqual(result.dc_energy_fraction, 1 / 6)
        self.assertEqual(result.solver.decomposition_count, 1)
        self.assertEqual(result.solver.algorithm, "torch.linalg.eigvalsh")
        self.assertIsNone(result.directions)
        # Singular-value energy is squared, not sum(sigma) or residual norm.
        self.assertAlmostEqual(math.sqrt(1 - result.rank_at(0.95).cdf_at_upper), math.sqrt(1 / 30))

    def test_exact_threshold_ties_are_not_silently_certified(self):
        for tau in (0.95, 0.99):
            with self.subTest(tau=tau):
                matrix = tie_matrix(tau)
                result = spectrum(matrix, tile_rows=7)
                bracket = self.assertBracket(result, tau, (1, 2))
                self.assertEqual(bracket.status, "interval")
                self.assertIsNone(bracket.rank)
                self.assertIn("straddles", bracket.diagnostic)
                comparison = direct_svd_crosscheck(matrix, result)
                direct = next(item for item in comparison.rank_brackets if item.tau == tau)
                self.assertLessEqual(direct.lower, 1)
                self.assertGreaterEqual(direct.upper, 2)

    def test_both_sides_of_threshold_and_one_ulp_ambiguity(self):
        for tau in (0.95, 0.99):
            for multiplier, expected in ((1 - 1e-4, 2), (1 + 1e-4, 1)):
                with self.subTest(tau=tau, multiplier=multiplier):
                    matrix = tie_matrix(tau)
                    matrix[:, 0] *= multiplier
                    result = spectrum(matrix, tile_rows=8)
                    bracket = self.assertBracket(result, tau, (expected, expected))
                    self.assertLess(bracket.cdf_before_lower, tau)
                    self.assertGreaterEqual(bracket.cdf_at_upper, tau)
            for direction in (0.0, math.inf):
                matrix = tie_matrix(tau)
                matrix[:-1, 0] = math.nextafter(1.0, direction)
                self.assertBracket(spectrum(matrix), tau, (1, 2))

    def test_zero_matrix_and_zero_dtype_flags(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                matrix = torch.zeros(3, 5, dtype=dtype)
                result = spectrum(matrix, return_directions=True, tile_rows=2)
                self.assertTrue(result.zero_energy)
                self.assertEqual(result.energy, 0.0)
                self.assertEqual(result.energy_scaled, 0.0)
                self.assertEqual(result.scale_exponent, 0)
                self.assertEqual(result.dc_energy_fraction, 0.0)
                self.assertEqual(result.eigenvalues_scaled, (0.0,) * 5)
                self.assertEqual((result.r95, result.r99), (0, 0))
                bracket = self.assertBracket(result, 0.95, (0, 0))
                self.assertEqual(bracket.status, "zero")
                self.assertIn("not a lossy CodecSpec", bracket.diagnostic)
                self.assertTrue(direct_svd_crosscheck(matrix, result).to_dict()["passed"])

    def test_rank_one_mean_dc_and_no_centering(self):
        matrix = torch.full((7, 4), 3.0, dtype=torch.float64)
        result = spectrum(matrix, tile_rows=2)
        self.assertEqual((result.r95, result.r99), (1, 1))
        self.assertAlmostEqual(result.dc_energy_fraction, 1.0)
        self.assertEqual(result.energy, 7 * 4 * 9)
        centered_fixture = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)
        centered = spectrum(centered_fixture)
        self.assertEqual(centered.r99, 1)
        self.assertEqual(centered.dc_energy_fraction, 0.0)

    def test_dc_consistency_includes_independent_norm_uncertainty(self):
        result = spectrum(torch.ones(8, 32, dtype=torch.float64), tile_rows=8)
        mean_error = result.diagnostics.mean_error_bound_scaled
        norm_error = result.diagnostics.norm_error_bound_scaled
        rounded = replace(
            result,
            mean_energy_scaled=result.energy_scaled + mean_error + 0.5 * norm_error,
        )
        self.assertGreater(rounded.dc_energy_fraction, 1.0)
        self.assertEqual(SpectrumResult.from_dict(rounded.to_dict()).to_dict(), rounded.to_dict())
        with self.assertRaises(SpectrumNumericalError):
            replace(
                result,
                mean_energy_scaled=result.energy_scaled + 2 * (mean_error + norm_error),
            )

    def test_flat_full_tail_not_renormalized(self):
        matrix = torch.eye(20, dtype=torch.float64)
        result = spectrum(matrix, tile_rows=3)
        self.assertEqual(len(result.eigenvalues_scaled), 20)
        self.assertEqual(result.eigenvalues_scaled, (1.0,) * 20)
        self.assertBracket(result, 0.95, (19, 20))
        self.assertBracket(result, 0.99, (20, 20))
        self.assertEqual(result.rank_at(0.99).cdf_before_lower, 0.95)
        self.assertEqual(result.rank_at(1.0).rank, 20)
        self.assertTrue(result.rank_at(1.0).complete_endpoint_used)
        with self.assertRaises(ConfigurationError):
            replace(result, eigenvalues_scaled=result.eigenvalues_scaled[:5])

    def test_wide_source_preserves_structural_zero_directions(self):
        matrix = torch.tensor(
            [[3.0, 0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        result = spectrum(matrix, return_directions=True, tile_rows=1)
        self.assertEqual(result.eigenvalues_scaled[2:], (0.0, 0.0, 0.0))
        self.assertEqual(result.directions.shape, (5, 5))
        self.assertEqual((result.r95, result.r99), (2, 2))
        for tau in (0.01, 0.95, 0.99, 1.0):
            self.assertLessEqual(result.rank_at(tau).upper, 2)
        direct = direct_svd_crosscheck(matrix, result)
        self.assertEqual(direct.eigenvalues_scaled[2:], (0.0, 0.0, 0.0))

    def test_near_degenerate_and_ill_conditioned_sources(self):
        for diagonal in (
            [1.0, 1.0 - 1e-12, 1.0 - 2e-12, 0.5],
            [1.0, 1e-5, 1e-10, 1e-100],
        ):
            matrix = torch.diag(torch.tensor(diagonal, dtype=torch.float64))
            result = spectrum(matrix, return_directions=True)
            direct = direct_svd_crosscheck(matrix, result)
            self.assertLess(direct.maximum_energy_fraction_difference, 1e-13)
            self.assertTrue(torch.allclose(
                result.directions @ result.directions.T, torch.eye(4, dtype=torch.float64),
                rtol=1e-13, atol=1e-13,
            ))
        self.assertIsNone(result.rank_at(1.0).rank)

    def test_extreme_finite_scale_overflow_underflow_and_rescaling(self):
        tiny = math.ulp(0.0)
        cases = [
            torch.diag(torch.tensor([1e308, 1e307], dtype=torch.float64)),
            torch.diag(torch.tensor([1e-300, 1e-301], dtype=torch.float64)),
            torch.diag(torch.tensor([tiny, tiny], dtype=torch.float64)),
            torch.tensor([[1e-308, 0.0], [1e308, 0.0]], dtype=torch.float64),
            torch.tensor([[torch.finfo(torch.float64).max, 0.0]], dtype=torch.float64),
        ]
        for matrix in cases:
            with self.subTest(matrix=matrix):
                original = matrix.clone()
                result = spectrum(matrix, tile_rows=1)
                self.assertFalse(result.zero_energy)
                self.assertIsNone(result.energy)
                self.assertTrue(math.isfinite(result.energy_scaled))
                self.assertTrue(all(math.isfinite(item) for item in result.eigenvalues_scaled))
                self.assertTrue(torch.equal(original, matrix))
                payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
                self.assertEqual(SpectrumResult.from_dict(payload).to_dict(), payload)
                self.assertTrue(direct_svd_crosscheck(matrix, result).to_dict()["passed"])
        result = spectrum(cases[3], tile_rows=1)
        self.assertEqual(result.diagnostics.rescalings, 1)
        self.assertEqual((result.r95, result.r99), (1, 1))

    def test_original_dtype_is_preserved_and_analysis_only_promotes(self):
        base = torch.tensor([[1.25, 2.125], [-2.0, 0.03125], [0.0, 3.5]])
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            matrix = base.to(dtype)
            before = matrix.clone()
            result = exact_spectrum(matrix, source=source(matrix, target="K"))
            self.assertEqual(result.source.dtype, str(dtype).removeprefix("torch."))
            self.assertEqual(result.solver.compute_dtype, "float64")
            self.assertTrue(torch.equal(matrix, before))
            self.assertTrue(direct_svd_crosscheck(matrix, result).to_dict()["passed"])
            wrong = replace(source(matrix), dtype="float32" if dtype != torch.float32 else "float64")
            with self.assertRaises(ConfigurationError):
                exact_spectrum(matrix, source=wrong)
        bad = torch.ones(2, 2, dtype=torch.int64)
        with self.assertRaises(ConfigurationError):
            source(bad)

    def test_noncontiguous_and_gradient_source_not_mutated(self):
        matrix = torch.arange(30, dtype=torch.float64).reshape(3, 10).T.requires_grad_()
        self.assertFalse(matrix.is_contiguous())
        before = matrix.detach().clone()
        result = spectrum(matrix, tile_rows=2, return_directions=True)
        contiguous = spectrum(matrix.detach().contiguous(), tile_rows=2)
        self.assertEqual(result.source.tensor_sha256, contiguous.source.tensor_sha256)
        np.testing.assert_allclose(
            result.eigenvalues_scaled, contiguous.eigenvalues_scaled, rtol=1e-13, atol=1e-13,
        )
        self.assertFalse(result.directions.requires_grad)
        self.assertIsNone(matrix.grad)
        self.assertTrue(torch.equal(before, matrix))

    def test_full_channel_row_shards_include_cross_head_covariance(self):
        # Two heads share correlated columns; head-local Grams would double rank.
        matrix = torch.tensor(
            [[1.0, 2.0, 1.0, 2.0], [2.0, 4.0, 2.0, 4.0],
             [-1.0, -2.0, -1.0, -2.0], [3.0, 6.0, 3.0, 6.0]],
            dtype=torch.float64,
        )
        self.assertGreater(float((matrix.T @ matrix)[:2, 2:].abs().sum()), 0.0)
        expected = spectrum(matrix, tile_rows=2)
        observed = spectrum_from_row_shards(
            (SpectrumRowShard(0, matrix[:2]), SpectrumRowShard(2, matrix[2:])),
            source=expected.source, tile_rows=2,
        )
        self.assertEqual(observed.to_dict(), expected.to_dict())
        self.assertEqual(observed.r99, 1)
        head_local = torch.block_diag(matrix[:, :2].T @ matrix[:, :2], matrix[:, 2:].T @ matrix[:, 2:])
        head_local_values = torch.linalg.eigvalsh(head_local).flip(0)
        incorrect_rank = int(torch.searchsorted(head_local_values.cumsum(0), 0.99 * head_local_values.sum())) + 1
        self.assertEqual(incorrect_rank, 2)

    def test_unequal_shards_rescalings_and_all_rows_equivalent(self):
        generator = torch.Generator().manual_seed(921)
        matrix = torch.randn(19, 7, dtype=torch.float64, generator=generator)
        matrix[:3] *= 1e-30
        matrix[11:] *= 1e30
        whole = spectrum(matrix, tile_rows=4)
        streamed = spectrum_from_row_shards(
            (matrix[:1], matrix[1:6], matrix[6:11], matrix[11:]),
            source=whole.source, tile_rows=3,
        )
        self.assertEqual(whole.source.tensor_sha256, streamed.source.tensor_sha256)
        self.assertEqual(whole.scale_exponent, streamed.scale_exponent)
        np.testing.assert_allclose(
            whole.eigenvalues_scaled, streamed.eigenvalues_scaled, rtol=1e-13, atol=1e-13,
        )
        self.assertAlmostEqual(whole.energy_scaled, streamed.energy_scaled, places=12)
        self.assertEqual((whole.r95, whole.r99), (streamed.r95, streamed.r99))
        self.assertTrue(direct_svd_crosscheck(matrix, streamed).to_dict()["passed"])

    def test_update_tiles_are_bounded_and_not_retained(self):
        import reelcache.spectra as module

        matrix = torch.arange(60, dtype=torch.float64).reshape(15, 4)
        seen = []
        native_tile = module._native_tile

        def observed(tile):
            seen.append(tile.shape[0])
            return native_tile(tile)

        with mock.patch.object(module, "_native_tile", side_effect=observed):
            accumulator = StreamingGramAccumulator(source(matrix), tile_rows=2)
            accumulator.update(matrix)
        self.assertEqual(max(seen), 2)
        reference = weakref.ref(matrix)
        del matrix
        gc.collect()
        self.assertIsNone(reference())
        result = accumulator.finalize()
        self.assertEqual(result.source.shape, (15, 4))
        self.assertIsNone(accumulator._gram)
        self.assertIsNone(accumulator._channel_sum)
        with self.assertRaises(StateError):
            accumulator.finalize()

    def test_directions_share_one_eigensolve_with_canonical_signs(self):
        generator = torch.Generator().manual_seed(7)
        matrix = torch.randn(9, 4, dtype=torch.float64, generator=generator)
        eigh = torch.linalg.eigh
        with (
            mock.patch("torch.linalg.eigh", wraps=eigh) as solve,
            mock.patch("torch.linalg.eigvalsh", side_effect=AssertionError("second eigen solve")),
            mock.patch("torch.linalg.svd", side_effect=AssertionError("second SVD")),
            mock.patch("torch.linalg.svdvals", side_effect=AssertionError("second SVD")),
        ):
            result = spectrum(matrix, return_directions=True, tile_rows=3)
        self.assertEqual(solve.call_count, 1)
        self.assertEqual(result.solver.algorithm, "torch.linalg.eigh")
        self.assertEqual(result.solver.decomposition_count, 1)
        self.assertEqual(result.solver.work_counts(), {
            "decomposition_count": 1, "eigh_count": 1, "eigvalsh_count": 0, "svd_count": 0,
        })
        self.assertEqual(result.work_counts(), {
            "decomposition_count": 1, "eigh_count": 1, "eigvalsh_count": 0, "svd_count": 0,
            "gram_build_count": 1, "source_hash_count": 1, "direction_verification_count": 1,
        })
        self.assertGreater(result.timings.direction_canonicalization_seconds, 0.0)
        self.assertGreater(result.timings.direction_verification_seconds, 0.0)
        self.assertLess(result.diagnostics.direction_orthogonality_error, 1e-13)
        self.assertLess(result.diagnostics.direction_residual_scaled, 1e-12)
        directions = result.directions
        pivots = directions.gather(1, directions.abs().argmax(1, keepdim=True))
        self.assertTrue(bool((pivots >= 0).all()))
        gram = matrix.T @ matrix / (2.0 ** (2 * result.scale_exponent))
        expected = torch.diag(torch.tensor(result.eigenvalues_scaled, dtype=torch.float64))
        self.assertTrue(torch.allclose(directions @ gram @ directions.T, expected, atol=1e-13, rtol=1e-13))
        decoded = SpectrumResult.from_dict(result.to_dict())
        self.assertIsNone(decoded.directions)
        self.assertTrue(decoded.solver.directions_requested)
        self.assertIsNone(decoded.timings)

    def test_timing_telemetry_is_separate_from_reproducible_observations(self):
        matrix = torch.eye(3, dtype=torch.float64)
        result = spectrum(matrix, tile_rows=1)
        self.assertIsInstance(result.timings, SpectrumTimings)
        telemetry = result.timings.to_dict()
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in telemetry.values()))
        self.assertLessEqual(telemetry["eigensolve_seconds"], telemetry["finalization_seconds"])
        self.assertLessEqual(
            telemetry["hash_update_seconds"] + telemetry["gram_matmul_add_seconds"],
            telemetry["accumulation_seconds"],
        )
        self.assertGreater(telemetry["hash_update_seconds"], 0.0)
        self.assertGreater(telemetry["gram_matmul_add_seconds"], 0.0)
        self.assertEqual(telemetry["direction_verification_seconds"], 0.0)
        self.assertEqual(result.work_counts()["direction_verification_count"], 0)
        self.assertEqual(telemetry["accounted_seconds"], sum(
            telemetry[name] for name in (
                "initialization_seconds", "accumulation_seconds", "finalization_seconds",
            )
        ))
        self.assertNotIn("timings", result.to_dict())
        self.assertEqual(result.solver.work_counts(), {
            "decomposition_count": 1, "eigh_count": 0, "eigvalsh_count": 1, "svd_count": 0,
        })
        direct = direct_svd_crosscheck(matrix, result)
        self.assertEqual(direct.solver.work_counts(), {
            "decomposition_count": 1, "eigh_count": 0, "eigvalsh_count": 0, "svd_count": 1,
        })
        with self.assertRaises(ConfigurationError):
            SpectrumTimings(0.0, 0.0, 2.0, 1.0)
        with self.assertRaises(ConfigurationError):
            SpectrumTimings(1e308, 1e308, 0.0, 0.0)
        with self.assertRaises(ConfigurationError):
            SpectrumTimings(0.0, 1.0, 0.0, 0.0, hash_update_seconds=2.0)
        with self.assertRaises(ConfigurationError):
            SpectrumTimings(0.0, 0.0, 0.0, 1.0, direction_verification_seconds=2.0)

    def test_no_global_rng_changes_and_reproducible_observation(self):
        generator = torch.Generator().manual_seed(412)
        matrix = torch.randn(13, 6, dtype=torch.float64, generator=generator)
        torch_before = torch.get_rng_state().clone()
        numpy_before = np.random.get_state()
        python_before = random.getstate()
        first = spectrum(matrix, tile_rows=4, return_directions=True)
        second = spectrum(matrix, tile_rows=4, return_directions=True)
        direct_svd_crosscheck(matrix, first)
        write_spectrum_json(first, self.artifact_path())
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertTrue(torch.equal(first.directions, second.directions))
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        self.assertEqual(python_before, random.getstate())
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])

    def test_serialization_roundtrip_and_source_retention_are_explicit(self):
        matrix = torch.eye(3, dtype=torch.float32)
        declaration = source(matrix, retention="spectrum-only")
        result = exact_spectrum(matrix, source=declaration)
        payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
        rebuilt = SpectrumResult.from_dict(payload)
        self.assertEqual(rebuilt.to_dict(), payload)
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(payload["hash_format"], HASH_FORMAT)
        self.assertEqual(payload["source"]["retention"], "spectrum-only")
        self.assertEqual(payload["source"]["representation"], "original")
        self.assertEqual(len(payload["source"]["tensor_sha256"]), 64)
        self.assertEqual(
            payload["unscale_energy_exponent"], 2 * payload["scale_exponent"],
        )
        with self.assertRaises(TypeError):
            declaration.provenance["fixture"] = False
        with self.assertRaises(TypeError):
            declaration.provenance["positions"][0] = 4

    def test_serialized_derived_values_and_spectrum_are_revalidated(self):
        payload = spectrum(torch.diag(torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64))).to_dict()
        corruptions = [
            lambda item: item.update(r95=1),
            lambda item: item.update(zero_energy=True),
            lambda item: item.update(energy=999.0),
            lambda item: item.update(unscale_energy_exponent=111),
            lambda item: item.update(unknown=True),
            lambda item: item.update(hash_format="unbound"),
            lambda item: item.update(eigenvalues_scaled=item["eigenvalues_scaled"][:2]),
            lambda item: item["eigenvalues_scaled"].reverse(),
            lambda item: item["eigenvalues_scaled"].__setitem__(0, float("nan")),
            lambda item: item["source"].update(dtype="int64"),
            lambda item: item["source"].update(tensor_sha256="no-hash"),
            lambda item: item["diagnostics"].update(eigenvalue_error_bound_scaled=1.0),
            lambda item: item["rank_brackets"]["0.95"].update(upper=1),
            lambda item: item["solver"].update(decomposition_count=2),
            lambda item: item.update(energy_scaled=True),
            lambda item: item.update(r99=True),
        ]
        for corrupt in corruptions:
            item = copy.deepcopy(payload)
            corrupt(item)
            with self.subTest(item=item), self.assertRaises((ConfigurationError, SpectrumNumericalError)):
                SpectrumResult.from_dict(item)
        with self.assertRaises(ConfigurationError):
            SpectrumResult.from_dict({"schema_version": SCHEMA_VERSION})

    def test_full_spectrum_file_is_canonical_finite_and_hash_verified(self):
        for matrix in (
            torch.eye(3, dtype=torch.float64),
            torch.zeros(2, 4, dtype=torch.bfloat16),
            torch.full((1, 1), 1e308, dtype=torch.float64),
        ):
            result = spectrum(matrix)
            path = self.artifact_path()
            digest = write_spectrum_json(result, path)
            content = path.read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), digest)
            self.assertEqual(content, json.dumps(
                result.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8"))
            self.assertEqual(
                SpectrumResult.from_dict(json.loads(content)).to_dict(), result.to_dict(),
            )
            self.assertEqual(list(path.parent.glob(f".{path.name}.part-*")), [])

    def test_spectrum_file_never_overwrites_existing_evidence(self):
        result = spectrum(torch.eye(2))
        path = self.artifact_path()
        first_hash = write_spectrum_json(result, path)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            write_spectrum_json(spectrum(torch.ones(2, 2)), path)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(hashlib.sha256(before).hexdigest(), first_hash)
        self.assertEqual(list(path.parent.glob(f".{path.name}.part-*")), [])
        alias = self.artifact_path()
        alias.symlink_to(path)
        with self.assertRaises(FileExistsError):
            write_spectrum_json(result, alias)
        self.assertTrue(alias.is_symlink())
        self.assertEqual(path.read_bytes(), before)

    def test_spectrum_file_failure_does_not_publish_partial_json(self):
        result = spectrum(torch.eye(2))
        for function in ("os.link", "os.fsync"):
            path = self.artifact_path()
            with mock.patch(function, side_effect=OSError("injected publication failure")):
                with self.assertRaises(OSError):
                    write_spectrum_json(result, path)
            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.glob(f".{path.name}.part-*")), [])
        published = self.artifact_path()
        with mock.patch("os.fsync", side_effect=[None, OSError("directory fsync failed")]):
            with self.assertRaises(OSError):
                write_spectrum_json(result, published)
        self.assertEqual(SpectrumResult.from_dict(json.loads(published.read_bytes())).to_dict(), result.to_dict())
        self.assertEqual(list(published.parent.glob(f".{published.name}.part-*")), [])
        for invalid in (None, "", True, "."):
            with self.subTest(invalid=invalid), self.assertRaises(ConfigurationError):
                write_spectrum_json(result, invalid)

    def test_threshold_types_ranges_and_numerics_cannot_disable_checks(self):
        result = spectrum(torch.eye(2))
        for tau in (True, False, None, "0.95", 0, -1, 1.01, math.inf, math.nan, [0.95]):
            with self.subTest(tau=tau), self.assertRaises(ConfigurationError):
                result.rank_at(tau)
        for factor in (False, 0.0, 0.5, 1025, math.inf, math.nan, "8"):
            with self.subTest(factor=factor), self.assertRaises(ConfigurationError):
                SpectrumNumerics(factor)
        self.assertEqual(result.rank_at(1).rank, 2)

    def test_invalid_source_geometry_domain_and_provenance(self):
        declaration = source(torch.ones(4, 3))
        for changes in (
            {"shape": (0, 3)}, {"shape": (4, True)}, {"shape": (4, 3, 1)},
            {"shape": (2**53 + 1, 3)}, {"dtype": "float8"},
            {"target": "Q"}, {"key_space": "post_rope"},
            {"target": "K", "key_space": "none"},
            {"retention": "recoverable-from-hash"}, {"representation": "P@B"},
            {"representation": "inverse_rope_post_k"}, {"identity": ""},
            {"provenance": {"score": math.nan}}, {"provenance": {1: "bad"}},
            {"provenance": {"tensor": torch.ones(1)}},
        ):
            with self.subTest(changes=changes), self.assertRaises(ConfigurationError):
                replace(declaration, **changes)
        for matrix in (
            torch.ones(5, 3), torch.ones(4, 2), torch.ones(12),
            torch.ones(4, 3, dtype=torch.int64),
            torch.ones(4, 3, dtype=torch.complex64),
            torch.ones(4, 3, device="meta"),
        ):
            with self.subTest(shape=matrix.shape), self.assertRaises(ConfigurationError):
                exact_spectrum(matrix, source=declaration)
        with self.assertRaises(ConfigurationError):
            exact_spectrum([[1.0]], source=declaration)

    def test_nonfinite_inputs_fail_and_poison_partial_stream(self):
        for invalid in (math.nan, math.inf, -math.inf):
            matrix = torch.ones(4, 3, dtype=torch.float64)
            matrix[-1, -1] = invalid
            accumulator = StreamingGramAccumulator(source(matrix), tile_rows=1)
            with self.subTest(invalid=invalid), self.assertRaises(ConfigurationError):
                accumulator.update(matrix)
            self.assertIsNone(accumulator._gram)
            with self.assertRaises(StateError):
                accumulator.finalize()

    def test_missing_duplicate_reordered_or_channel_shards_are_errors(self):
        matrix = torch.ones(6, 4)
        declaration = source(matrix)
        bad_shards = [
            [matrix[:3]],
            [matrix, matrix[:1]],
            [SpectrumRowShard(1, matrix[:3]), SpectrumRowShard(3, matrix[3:])],
            [SpectrumRowShard(0, matrix[:3]), SpectrumRowShard(0, matrix[3:])],
            [matrix[:, :2], matrix[:, 2:]],
            [matrix[:3], matrix[3:].double()],
            [],
            [torch.ones(0, 4)],
        ]
        for shards in bad_shards:
            with self.subTest(shards=shards), self.assertRaises(ConfigurationError):
                spectrum_from_row_shards(shards, source=declaration)
        matrix[1, 0] = 2.0
        observed = spectrum(matrix)
        with self.assertRaises(IntegrityError):
            spectrum_from_row_shards(
                [matrix[3:], matrix[:3]], source=observed.source,
            )

    def test_capacity_is_rejected_before_gram_or_svd_allocation(self):
        matrix = torch.ones(4, 3)
        with (
            mock.patch("torch.zeros", side_effect=AssertionError("allocated before admission")),
            mock.patch("torch.linalg.eigh") as eigh,
            mock.patch("torch.linalg.eigvalsh") as eigvalsh,
        ):
            with self.assertRaises(CapacityError):
                StreamingGramAccumulator(source(matrix), max_workspace_bytes=1)
            eigh.assert_not_called()
            eigvalsh.assert_not_called()
        result = spectrum(matrix)
        with mock.patch("torch.empty", side_effect=AssertionError("allocated before admission")):
            with self.assertRaises(CapacityError):
                direct_svd_crosscheck(matrix, result, max_workspace_bytes=1)
        for tile_rows in (True, 0, -1, 1.5):
            with self.subTest(tile_rows=tile_rows), self.assertRaises(ConfigurationError):
                spectrum(matrix, tile_rows=tile_rows)
        with self.assertRaises(ConfigurationError):
            spectrum(matrix, return_directions=1)

    def test_numerical_error_metrics_are_optional_readonly_and_pickleable(self):
        standalone = SpectrumNumericalError("outside accumulator")
        self.assertIsNone(standalone.metrics)
        self.assertEqual(str(standalone), "outside accumulator")
        original = {"nested": {"counts_available": False}}
        error = SpectrumNumericalError("fixture", metrics=original)
        original["nested"]["counts_available"] = True
        self.assertFalse(error.metrics["nested"]["counts_available"])
        with self.assertRaises(TypeError):
            error.metrics["nested"]["counts_available"] = True
        with self.assertRaises(AttributeError):
            error.metrics = {}
        restored = pickle.loads(pickle.dumps(error))
        self.assertEqual(str(restored), str(error))
        self.assertEqual(restored.metrics, error.metrics)

    def test_failure_before_eigensolve_reports_zero_invocations(self):
        matrix = torch.eye(2, dtype=torch.float64)
        accumulator = StreamingGramAccumulator(source(matrix))
        accumulator.update(matrix)
        accumulator._gram[0, 0] += 0.1
        with (
            mock.patch("torch.linalg.eigh") as eigh,
            mock.patch("torch.linalg.eigvalsh") as eigvalsh,
            self.assertRaises(SpectrumNumericalError) as failure,
        ):
            accumulator.finalize(return_directions=True)
        eigh.assert_not_called()
        eigvalsh.assert_not_called()
        metrics = self.assertFailureMetrics(failure.exception, phase="gram_validation")
        self.assertEqual(metrics["eigensolve_seconds"], 0.0)
        self.assertEqual(metrics["total_seconds"], metrics["exclusive_pipeline_seconds"])
        self.assertIsNone(accumulator._gram)
        self.assertIsNone(accumulator._channel_sum)
        self.assertEqual(
            pickle.loads(pickle.dumps(failure.exception)).metrics, failure.exception.metrics,
        )

    def test_failed_solver_invocation_is_counted_for_both_solver_paths(self):
        for directions in (False, True):
            algorithm = "eigh" if directions else "eigvalsh"
            with (
                mock.patch(f"torch.linalg.{algorithm}", side_effect=RuntimeError("solver failed")) as solve,
                self.assertRaises(SpectrumNumericalError) as failure,
            ):
                spectrum(torch.eye(2), return_directions=directions)
            self.assertEqual(solve.call_count, 1)
            metrics = self.assertFailureMetrics(
                failure.exception, eigh=int(directions), eigvalsh=int(not directions),
                phase="eigensolve",
            )
            self.assertGreater(metrics["eigensolve_seconds"], 0.0)
            self.assertEqual(metrics[f"{algorithm}_seconds"], metrics["eigensolve_seconds"])
            self.assertIsInstance(failure.exception.__cause__, RuntimeError)

    def test_postsolve_failures_keep_completed_solver_invocation(self):
        with (
            mock.patch(
                "torch.linalg.eigvalsh",
                return_value=torch.tensor([1.0, math.nan], dtype=torch.float64),
            ) as solve,
            self.assertRaises(SpectrumNumericalError) as failure,
        ):
            spectrum(torch.eye(2))
        self.assertEqual(solve.call_count, 1)
        self.assertFailureMetrics(failure.exception, eigvalsh=1, phase="spectrum_validation")
        eigh = torch.linalg.eigh
        with (
            mock.patch("torch.linalg.eigh", wraps=eigh) as solve,
            mock.patch(
                "reelcache.spectra._canonical_directions", return_value=torch.ones(3, 3, dtype=torch.float64),
            ),
            self.assertRaises(SpectrumNumericalError) as failure,
        ):
            spectrum(torch.eye(3), return_directions=True)
        self.assertEqual(solve.call_count, 1)
        self.assertFailureMetrics(failure.exception, eigh=1, phase="result_validation")

    def test_numeric_accumulation_failure_retains_prior_and_current_time(self):
        matrix = torch.ones(4, 2)
        accumulator = StreamingGramAccumulator(source(matrix), tile_rows=1)
        accumulator.update(matrix[:2], row_start=0)
        prior_seconds = accumulator._accumulation_seconds
        with (
            mock.patch("reelcache.spectra._hash_tile", side_effect=SpectrumNumericalError("injected failure")),
            self.assertRaises(SpectrumNumericalError) as failure,
        ):
            accumulator.update(matrix[2:], row_start=2)
        metrics = self.assertFailureMetrics(failure.exception, phase="accumulation")
        self.assertGreaterEqual(metrics["accumulation_seconds"], prior_seconds)
        self.assertEqual(metrics["finalization_seconds"], 0.0)
        self.assertEqual(metrics["eigensolve_seconds"], 0.0)
        self.assertIsNone(accumulator._gram)
        with self.assertRaises(StateError):
            accumulator.finalize()

    def test_material_symmetry_trace_and_negative_mass_fail(self):
        matrix = torch.eye(2, dtype=torch.float64)
        for fault in ("symmetry", "trace", "negative", "nonfinite"):
            accumulator = StreamingGramAccumulator(source(matrix))
            accumulator.update(matrix)
            if fault == "symmetry":
                accumulator._gram[0, 1] += 0.1
            elif fault == "trace":
                accumulator._gram[0, 0] += 0.1
            elif fault == "negative":
                accumulator._gram[0, 0] = -0.1
                accumulator._gram[1, 1] = 2.1
            else:
                accumulator._gram[0, 0] = math.inf
            with self.subTest(fault=fault), self.assertRaises(SpectrumNumericalError):
                accumulator.finalize()
            self.assertIsNone(accumulator._gram)

    def test_only_roundoff_negative_clipping_is_allowed_and_recorded(self):
        matrix = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
        with mock.patch(
            "torch.linalg.eigvalsh",
            return_value=torch.tensor([-1e-14, 1 + 1e-14], dtype=torch.float64),
        ):
            result = spectrum(matrix)
        self.assertEqual(result.eigenvalues_scaled[1], 0.0)
        self.assertAlmostEqual(result.diagnostics.negative_eigenvalue_mass_scaled, 1e-14)
        self.assertEqual(result.r99, 1)
        self.assertEqual(result.energy_scaled, 1.0)
        self.assertGreater(sum(result.eigenvalues_scaled), result.energy_scaled)
        self.assertEqual(SpectrumResult.from_dict(result.to_dict()).to_dict(), result.to_dict())
        with mock.patch(
            "torch.linalg.eigvalsh",
            return_value=torch.tensor([-1e-6, 1 + 1e-6], dtype=torch.float64),
        ), self.assertRaises(SpectrumNumericalError):
            spectrum(matrix)

    def test_inconsistent_cdf_is_not_made_exact_by_clamping(self):
        matrix = torch.diag(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64))
        budget = spectrum(matrix).diagnostics.eigenvalue_error_bound_scaled
        small = 0.9 * budget
        # Every negative is individually roundoff sized, but the leading
        # eigenvalue now violates the purported per-eigenvalue error interval.
        with mock.patch(
            "torch.linalg.eigvalsh",
            return_value=torch.tensor([-small, -small, -small, 1 + 3 * small], dtype=torch.float64),
        ), self.assertRaisesRegex(SpectrumNumericalError, "CDF lower bound"):
            spectrum(matrix)

    def test_invalid_eigenvector_directions_fail_numeric_checks(self):
        with mock.patch(
            "reelcache.spectra._canonical_directions",
            return_value=torch.ones(3, 3, dtype=torch.float64),
        ), self.assertRaisesRegex(SpectrumNumericalError, "not orthonormal"):
            spectrum(torch.eye(3), return_directions=True)

    def test_structural_roundoff_tail_clipping_and_material_tail_failure(self):
        matrix = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        for small in (1e-14, 1e-4):
            with mock.patch(
                "torch.linalg.eigvalsh",
                return_value=torch.tensor([0.0, small, 1.0 - small], dtype=torch.float64),
            ):
                if small > 1e-10:
                    with self.assertRaises(SpectrumNumericalError):
                        spectrum(matrix)
                else:
                    result = spectrum(matrix)
                    self.assertEqual(result.eigenvalues_scaled[1:], (0.0, 0.0))
                    self.assertEqual(result.diagnostics.structural_zero_tail_mass_scaled, small)

    def test_incomplete_nonfinite_and_failed_eigensolver(self):
        for output in (
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([1.0, math.nan], dtype=torch.float64),
            torch.tensor([1.0, math.inf], dtype=torch.float64),
        ):
            with mock.patch("torch.linalg.eigvalsh", return_value=output):
                with self.assertRaises(SpectrumNumericalError):
                    spectrum(torch.eye(2))
        with mock.patch("torch.linalg.eigvalsh", side_effect=RuntimeError("not converged")):
            with self.assertRaises(SpectrumNumericalError):
                spectrum(torch.eye(2))
        with mock.patch(
            "torch.linalg.eigvalsh", return_value=torch.tensor([0.1, 0.2], dtype=torch.float64),
        ), self.assertRaises(SpectrumNumericalError):
            spectrum(torch.eye(2))

    def test_direct_svd_source_hash_and_dtype_are_strict(self):
        matrix = torch.tensor([[3.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        result = spectrum(matrix)
        same_spectrum_wrong_bytes = matrix.flip(0)
        with self.assertRaises(IntegrityError):
            direct_svd_crosscheck(same_spectrum_wrong_bytes, result)
        with self.assertRaises(ConfigurationError):
            direct_svd_crosscheck(matrix.float(), result)
        with self.assertRaises(IntegrityError):
            exact_spectrum(matrix, source=replace(source(matrix), tensor_sha256="0" * 64))
        for value in (math.nan, math.inf, -1, True, 0.5):
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                direct_svd_crosscheck(matrix, result, absolute_energy_tolerance=value)

    def test_direct_svd_agreement_has_full_tail_and_one_separate_decomposition(self):
        generator = torch.Generator().manual_seed(134)
        for shape in ((11, 6), (4, 7), (1, 3)):
            matrix = torch.randn(*shape, dtype=torch.float64, generator=generator)
            result = spectrum(matrix, tile_rows=2)
            svdvals = torch.linalg.svdvals
            with (
                mock.patch("torch.linalg.svdvals", wraps=svdvals) as solve,
                mock.patch("torch.linalg.eigh", side_effect=AssertionError("Gram repeated")),
                mock.patch("torch.linalg.eigvalsh", side_effect=AssertionError("Gram repeated")),
            ):
                comparison = direct_svd_crosscheck(
                    matrix, result, absolute_energy_tolerance=1e-13, relative_tolerance=1e-12,
                )
            self.assertEqual(solve.call_count, 1)
            self.assertEqual(len(comparison.eigenvalues_scaled), shape[1])
            self.assertLess(comparison.maximum_energy_fraction_difference, 1e-13)
            self.assertEqual(comparison.solver.algorithm, "torch.linalg.svdvals")
            json.dumps(comparison.to_dict(), allow_nan=False)

    def test_direct_svd_detects_a_trace_preserving_wrong_spectrum(self):
        matrix = torch.diag(torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64))
        result = spectrum(matrix)
        values = list(result.eigenvalues_scaled)
        values[0] += 0.1
        values[1] -= 0.1
        wrong = replace(result, eigenvalues_scaled=tuple(values))
        with self.assertRaises(SpectrumNumericalError):
            direct_svd_crosscheck(matrix, wrong)
        for values in ([1.0], [1.0, math.nan, 0.0], [10.0, 2.0, 1.0]):
            with mock.patch(
                "torch.linalg.svdvals", return_value=torch.tensor(values, dtype=torch.float64),
            ), self.assertRaises(SpectrumNumericalError):
                direct_svd_crosscheck(matrix, result)


if __name__ == "__main__":
    unittest.main()
