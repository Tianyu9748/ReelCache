import dataclasses
import gc
import math
import os
import random
import time
import unittest
import weakref
from collections.abc import Mapping
from pathlib import Path
from unittest import mock

import torch

from reelcache.errors import CapacityError, ConfigurationError, IntegrityError, StateError, UnsupportedError
from reelcache.factorization import DetectorSpec, acquire_anchor_directions, rank_fit_workspace_bytes, tensor_sha256
from reelcache.online_rank import (
    PreparedRankMatrix, RankAbstention, RankDecision, RankPolicy, RankPreparation, RankState,
    RankPredictorProfile, bind_predictor_profiles, predict_rank, profile_builder, rank_predictor,
)
from reelcache.types import CodecSpec, MatrixKey, ModelIdentity, TensorContext
from tests.core_support import CoreWorkspaceTest


def policy(*, layers=1, channels=4, rows=4, **overrides):
    model = ModelIdentity("online-rank-test", "synthetic", layers, channels, "tokens_channels", "float32")
    keys = tuple(MatrixKey(layer, target) for layer in range(layers) for target in ("K", "V"))
    values = dict(
        policy_id="synthetic-only", model=model, geometry=(1, 1, rows),
        source_dtypes={"K": "float32", "V": "float32"},
        source_fingerprint="1" * 64, geometry_fingerprint="2" * 64,
        history_fingerprint="3" * 64, calibration_fingerprint="4" * 64,
        implementation_fingerprint="5" * 64, energy_target=0.95,
        template_ranks={key: 1.0 for key in keys},
        allowed_ranks={key: tuple(range(1, min(channels, rows) + 1)) for key in keys},
        oversample=0, tile_rows=2, solver_workspace_bytes=0,
        max_fit_attempts=4, max_guard_attempts=16,
    )
    values.update(overrides)
    return RankPolicy(**values)


def contexts(bound, token_start=0):
    return {
        key: TensorContext(
            bound.model, key, token_start, *bound.geometry,
            "pre_rope" if key.target == "K" else "none",
            source_dtype=bound.source_dtypes[key],
        )
        for key in bound.keys
    }


def rank_matrix(rank, channels=4, rows=4, dtype=torch.float32):
    result = torch.zeros(rows, channels, dtype=dtype)
    for index in range(rank):
        result[index, index] = 1
    return result


def complete(bound, matrices=None, *, prior_state=None, chunk_id="chunk-1", parent_head=None):
    start = 0 if prior_state is None else prior_state.token_end
    ctx = contexts(bound, start)
    matrices = {key: rank_matrix(1) for key in bound.keys} if matrices is None else matrices
    prep = RankPreparation(bound, prior_state, chunk_id=chunk_id, parent_head=parent_head, token_start=start)
    if bound.anchor_mode == "staged":
        prep.prepare_staged(matrices, ctx)
    results = {key: prep.prepare_matrix(matrices[key], ctx[key]) for key in bound.keys}
    return prep, results, prep.next_state()


class LazySources(Mapping):
    def __init__(self, bound, rank_for):
        self.bound = bound
        self.rank_for = rank_for
        self.reads = []
        self.references = []

    def __iter__(self):
        return iter(self.bound.keys)

    def __len__(self):
        return len(self.bound.keys)

    def __getitem__(self, key):
        self.reads.append(key)
        matrix = rank_matrix(self.rank_for.get(key, 1))
        self.references.append(weakref.ref(matrix))
        return matrix

    def values(self):
        raise AssertionError("retaining all lazy mapping values is forbidden")


class OnlineRankTests(unittest.TestCase):
    def test_policy_roundtrip_binds_all_execution_identities_and_is_deeply_frozen(self):
        bound = policy()
        restored = RankPolicy.from_dict(bound.to_dict())
        self.assertEqual(restored.fingerprint, bound.fingerprint)
        self.assertEqual(restored.to_dict(), bound.to_dict())
        for name in (
            "source_fingerprint", "geometry_fingerprint", "history_fingerprint",
            "calibration_fingerprint", "implementation_fingerprint",
        ):
            self.assertNotEqual(bound.fingerprint, dataclasses.replace(bound, **{name: "a" * 64}).fingerprint)
        self.assertNotEqual(bound.fingerprint, dataclasses.replace(bound, seed=13).fingerprint)
        self.assertNotEqual(bound.fingerprint, dataclasses.replace(bound, energy_target=0.99).fingerprint)
        self.assertNotEqual(
            bound.fingerprint,
            dataclasses.replace(bound, codec=CodecSpec("svd_hevc", 1)).fingerprint,
        )
        with self.assertRaises(TypeError):
            bound.template_ranks[MatrixKey(0, "K")] = 3
        with self.assertRaises(dataclasses.FrozenInstanceError):
            bound.energy_target = 0.99
        copy = bound.to_dict()
        copy["template_ranks"]["0:K"] = 4
        self.assertEqual(bound.template_ranks[MatrixKey(0, "K")], 1)
        del copy["schema_version"]
        with self.assertRaises(ConfigurationError):
            RankPolicy.from_dict(copy)

    def test_study_cell_source_bytes_include_actual_magi_pre_key_dtype(self):
        cells = [
            (30, 1536, (3, 30, 52), "bfloat16", 862_617_600),
            (48, 1024, (6, 45, 80), "float32", 6_370_099_200),
            (48, 1024, (6, 68, 120), "float32", 14_438_891_520),
        ]
        for layers, channels, geometry, key_dtype, expected in cells:
            keys = tuple(MatrixKey(layer, target) for layer in range(layers) for target in ("K", "V"))
            model = ModelIdentity("declared-cell", "synthetic-no-model", layers, channels, "native_kv", "bfloat16")
            bound = policy(
                layers=layers, channels=channels, rows=math.prod(geometry),
                model=model, geometry=geometry,
                source_dtypes={"K": key_dtype, "V": "bfloat16"},
                allowed_ranks={key: (1, 2) for key in keys},
                anchor_mode="staged", staging_limit_bytes=expected,
            )
            self.assertEqual(bound.source_bytes_per_chunk, expected)
            with self.assertRaises(CapacityError):
                dataclasses.replace(bound, staging_limit_bytes=expected - 1)

    def test_policy_rejects_raw_reuse_bad_ladders_and_anchor_budgets(self):
        with self.assertRaises(ConfigurationError):
            policy(codec=CodecSpec("raw"))
        with self.assertRaises(UnsupportedError):
            policy(reuse_enabled=True)
        with self.assertRaises(ConfigurationError):
            policy(allowed_ranks={MatrixKey(0, "K"): (1, 1), MatrixKey(0, "V"): (1, 2)})
        with self.assertRaises(ConfigurationError):
            policy(layers=3, anchors=tuple(MatrixKey(layer, "K") for layer in range(3)), anchor_mode="causal")
        with self.assertRaises(ConfigurationError):
            policy(anchors=(MatrixKey(0, "K"),))

    def test_prepared_seals_original_context_basis_and_actual_fp32_coefficients(self):
        bound = policy()
        ctx = contexts(bound)[MatrixKey(0, "K")]
        matrix = rank_matrix(2)
        prepared = RankPreparation(
            bound, chunk_id="c", parent_head=None, token_start=0,
        ).prepare_matrix(matrix, ctx)
        prepared.validate(matrix, ctx, prepared.spec, bound)
        self.assertEqual(prepared.decision.basis_sha256, tensor_sha256(prepared.factors.basis))
        self.assertEqual(prepared.decision.coefficients_sha256, tensor_sha256(prepared.factors.coefficients))
        self.assertTrue(torch.equal(
            prepared.factors.coefficients, matrix.float() @ prepared.factors.basis.float().T,
        ))
        with self.assertRaises(IntegrityError):
            prepared.validate(matrix + 1, ctx)
        with self.assertRaises(IntegrityError):
            prepared.validate(matrix, dataclasses.replace(ctx, token_start=4))
        with self.assertRaises(IntegrityError):
            prepared.validate(matrix, ctx, CodecSpec("svd", 4))
        prepared.factors.coefficients[0, 0] += 1
        with self.assertRaises(IntegrityError):
            prepared.validate(matrix, ctx)

    def test_mutated_basis_and_other_policy_cannot_use_prepared_proof(self):
        bound = policy()
        prep, results, _ = complete(bound)
        key = MatrixKey(0, "K")
        prepared = results[key]
        ctx = contexts(bound)[key]
        with self.assertRaises(IntegrityError):
            dataclasses.replace(bound, seed=8).validate_decision(prepared.decision)
        prepared.factors.basis[0, 0] += 0.25
        with self.assertRaises(IntegrityError):
            prepared.validate(rank_matrix(1), ctx)
        prep.close()

    def test_full_model_state_only_becomes_available_after_every_key_accepts(self):
        bound = policy()
        prep = RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0)
        ctx = contexts(bound)
        first = prep.prepare_matrix(rank_matrix(1), ctx[MatrixKey(0, "K")])
        self.assertIsInstance(first, PreparedRankMatrix)
        with self.assertRaises(StateError):
            prep.next_state()
        prep.prepare_matrix(rank_matrix(2), ctx[MatrixKey(0, "V")])
        state = prep.next_state()
        self.assertIs(state, prep.next_state())
        self.assertEqual(state.completed_chunks, 1)
        self.assertEqual(state.token_end, 4)
        self.assertEqual(state.last_chunk_id, "c")
        self.assertEqual(state.selected_ranks[MatrixKey(0, "V")], 2)
        self.assertEqual(RankState.from_dict(state.to_dict()).fingerprint, state.fingerprint)
        with self.assertRaises(TypeError):
            state.selected_ranks[MatrixKey(0, "K")] = 4
        prep.close()

    def test_failed_attempt_preserves_the_original_committed_scalar_state(self):
        bound = policy()
        first, _, state = complete(bound)
        original = state.to_dict()
        second = RankPreparation(
            bound, state, chunk_id="chunk-2", parent_head="committed-head-1", token_start=4,
        )
        ctx = contexts(bound, 4)
        second.prepare_matrix(rank_matrix(2), ctx[MatrixKey(0, "K")])
        invalid = rank_matrix(1)
        invalid[0, 0] = float("nan")
        with self.assertRaises(ConfigurationError):
            second.prepare_matrix(invalid, ctx[MatrixKey(0, "V")])
        with self.assertRaises(StateError):
            second.next_state()
        self.assertEqual(state.to_dict(), original)
        self.assertTrue(second.metrics["failed"])
        first.close()
        second.close()

    def test_finite_width_miss_is_unresolved_not_maximum_rank_success(self):
        keys = (MatrixKey(0, "K"), MatrixKey(0, "V"))
        bound = policy(allowed_ranks={key: (1, 2) for key in keys}, max_fit_width=2)
        prep = RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0)
        with self.assertRaises(RankAbstention) as caught:
            prep.prepare_matrix(torch.eye(4), contexts(bound)[keys[0]])
        decision = caught.exception.decision
        self.assertEqual(decision.status, "UNRESOLVED")
        self.assertIsNone(decision.rank)
        self.assertIsNone(decision.resolved_spec)
        self.assertFalse(decision.guard_passed)
        self.assertIsNone(decision.estimated_rank)
        self.assertEqual(decision.metrics["finite_width_misses"], 1)
        self.assertEqual(decision.metrics["svd_count"], 1)
        self.assertEqual(decision.metrics["projection_count"], 0)
        self.assertEqual(prep.failure_decision, decision)
        with self.assertRaises(StateError):
            prep.next_state()

    def test_expansion_refits_are_counted_and_do_not_renormalize_partial_energy(self):
        bound = policy(channels=8, rows=8)
        ctx = contexts(bound)[MatrixKey(0, "K")]
        with (
            mock.patch("torch.linalg.svd", wraps=torch.linalg.svd) as svd,
            mock.patch("torch.linalg.qr", wraps=torch.linalg.qr) as qr,
        ):
            result = RankPreparation(
                bound, chunk_id="c", parent_head=None, token_start=0,
            ).prepare_matrix(torch.eye(8), ctx)
        self.assertEqual(result.decision.rank, 8)
        self.assertEqual(result.decision.metrics["finite_width_misses"], 2)
        self.assertEqual(result.decision.metrics["refits"], 2)
        self.assertEqual(result.decision.metrics["svd_count"], svd.call_count)
        self.assertEqual(result.decision.metrics["qr_count"], qr.call_count)
        self.assertEqual(svd.call_count, 3)
        self.assertEqual(qr.call_count, 15)
        self.assertEqual(result.decision.metrics["projection_count"], 1)

    def test_exact_reference_uses_one_oracle_eigh_and_keeps_exact_integer_separate(self):
        keys = (MatrixKey(0, "K"), MatrixKey(0, "V"))
        bound = policy(
            solver="exact_oracle", allowed_ranks={key: (1, 4) for key in keys},
        )
        matrix = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
        with (
            mock.patch("torch.linalg.svd", side_effect=AssertionError("duplicate SVD")),
            mock.patch("torch.linalg.eigh", wraps=torch.linalg.eigh) as eigh,
        ):
            result = RankPreparation(
                bound, chunk_id="c", parent_head=None, token_start=0,
            ).prepare_matrix(matrix, contexts(bound)[keys[0]])
        self.assertEqual(eigh.call_count, 1)
        self.assertEqual(result.decision.exact_rank, 3)
        self.assertEqual(result.decision.rank, 4)
        self.assertEqual(result.decision.estimated_rank, 3)
        self.assertEqual(result.decision.metrics["eigh_count"], 1)
        self.assertEqual(result.decision.metrics["svd_count"], 0)
        self.assertEqual(result.decision.metrics["gram_build_count"], 1)
        self.assertEqual(result.decision.metrics["source_hash_count"], 2)
        self.assertEqual(result.decision.metrics["factor_hash_count"], 2)
        self.assertEqual(result.decision.metrics["direction_verification_count"], 1)
        self.assertEqual(result.decision.metrics["qr_count"], 0)
        self.assertEqual(result.decision.metrics["projection_count"], 1)
        observation = result.decision.oracle_observation
        self.assertEqual(len(observation["eigenvalues_scaled"]), 4)
        self.assertNotIn("directions", observation)
        self.assertEqual(
            observation["source"]["provenance"]["original_bytes_sha256"],
            result.decision.input_sha256,
        )
        self.assertEqual(observation["source"]["dtype"], "float32")
        self.assertEqual(observation["source"]["key_space"], "pre_rope")

    def test_full_rank_stored_precision_failure_is_explicit_abstention(self):
        bound = policy(solver="exact_oracle", energy_target=1.0)
        matrix = torch.randn(4, 4, generator=torch.Generator().manual_seed(334))
        with self.assertRaises(RankAbstention) as caught:
            RankPreparation(
                bound, chunk_id="c", parent_head=None, token_start=0,
            ).prepare_matrix(matrix, contexts(bound)[MatrixKey(0, "K")])
        failed = caught.exception.decision
        self.assertEqual(failed.reason, "stored_precision_guard_failed_at_full_rank")
        self.assertEqual(failed.exact_rank, 4)
        self.assertIsNone(failed.rank)
        self.assertIsNone(failed.resolved_spec)
        self.assertGreater(failed.guard_relative_residual, 0)
        self.assertEqual(failed.metrics["eigh_count"], 1)

    def test_zero_is_mathematical_rank_zero_and_positive_operational_rank(self):
        keys = (MatrixKey(0, "K"), MatrixKey(0, "V"))
        bound = policy(allowed_ranks={key: (2, 4) for key in keys})
        with (
            mock.patch("torch.linalg.svd", side_effect=AssertionError("zero SVD")),
            mock.patch("torch.linalg.qr", side_effect=AssertionError("zero QR")),
        ):
            result = RankPreparation(
                bound, chunk_id="c", parent_head=None, token_start=0,
            ).prepare_matrix(torch.zeros(4, 4), contexts(bound)[keys[0]])
        self.assertTrue(result.decision.zero_input)
        self.assertEqual(result.decision.exact_rank, 0)
        self.assertEqual(result.decision.estimated_rank, 0)
        self.assertEqual(result.decision.rank, 2)
        self.assertEqual(result.decision.guard_relative_residual, 0)
        self.assertEqual(result.decision.metrics["svd_count"], 0)

    def test_staged_late_anchor_fits_once_and_lazy_sources_are_not_retained(self):
        anchor = MatrixKey(2, "K")
        bound = policy(layers=3, anchors=(anchor,), anchor_mode="staged", staging_limit_bytes=384)
        sources = LazySources(bound, {anchor: 2})
        prep = RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0)
        ctx = contexts(bound)
        with mock.patch("torch.linalg.svd", wraps=torch.linalg.svd) as svd:
            prep.prepare_staged(sources, ctx)
            self.assertEqual(sources.reads, [anchor])
            gc.collect()
            self.assertTrue(all(reference() is None for reference in sources.references))
            self.assertEqual(svd.call_count, 1)
            first = prep.prepare_matrix(sources[MatrixKey(0, "K")], ctx[MatrixKey(0, "K")])
            self.assertEqual(first.decision.predicted_rank, 2)
            self.assertEqual(first.decision.prediction_sources[0]["availability"], "current")
            for key in bound.keys[1:]:
                prepared = prep.prepare_matrix(sources[key], ctx[key])
                if key == anchor:
                    self.assertTrue(prepared.decision.metrics["cached_anchor_consumed"])
            self.assertEqual(svd.call_count, len(bound.keys))
        self.assertEqual(prep.next_state().anchor_ranks[anchor], 2)
        self.assertEqual(prep.metrics["logical_staged_source_bytes"], 384)
        self.assertFalse(prep.metrics["streaming_memory_claim"])
        self.assertGreater(prep.metrics["peak_cached_factor_bytes"], 0)
        self.assertEqual(prep.metrics["cached_factor_bytes"], 0)
        prep.close()

    def test_staged_anchor_changed_after_fitting_fails_without_refit(self):
        anchor = MatrixKey(0, "K")
        bound = policy(anchors=(anchor,), anchor_mode="staged", staging_limit_bytes=128)
        matrices = {key: rank_matrix(1) for key in bound.keys}
        ctx = contexts(bound)
        prep = RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0)
        prep.prepare_staged(matrices, ctx)
        with mock.patch("torch.linalg.svd", side_effect=AssertionError("repeated anchor SVD")):
            with self.assertRaises(IntegrityError):
                prep.prepare_matrix(matrices[anchor] + 1, ctx[anchor])
        self.assertTrue(prep.metrics["failed"])
        self.assertEqual(prep.metrics["cached_factor_bytes"], 0)

    def test_causal_first_chunk_cannot_use_later_anchor_and_lagged_state_is_explicit(self):
        anchor = MatrixKey(2, "K")
        bound = policy(layers=3, anchors=(anchor,), anchor_mode="causal")
        matrices = {key: rank_matrix(2 if key == anchor else 1) for key in bound.keys}
        first, results, state = complete(bound, matrices)
        early = results[MatrixKey(0, "K")].decision
        self.assertEqual(early.predicted_rank, 1)
        self.assertFalse(early.prediction_sources)
        second, later, _ = complete(
            bound, matrices, prior_state=state, chunk_id="chunk-2", parent_head="head-1",
        )
        prior = later[MatrixKey(0, "K")].decision
        self.assertEqual(prior.predicted_rank, 2)
        self.assertEqual(prior.prediction_sources[0]["availability"], "prior_committed")
        self.assertEqual(prior.prediction_sources[0]["history_age"], 1)
        first.close()
        second.close()

    def test_earlier_current_anchor_guides_later_only_in_causal_not_lagged_mode(self):
        anchor = MatrixKey(0, "K")
        for mode, predicted in (("causal", 2), ("lagged", 1)):
            bound = policy(layers=2, anchors=(anchor,), anchor_mode=mode)
            matrices = {key: rank_matrix(2 if key == anchor else 1) for key in bound.keys}
            prep, results, _ = complete(bound, matrices)
            result = results[MatrixKey(1, "K")].decision
            self.assertEqual(result.predicted_rank, predicted)
            if mode == "causal":
                self.assertEqual(result.prediction_sources[0]["history_age"], 0)
            else:
                self.assertFalse(result.prediction_sources)
            prep.close()

    def test_causal_staging_control_counts_storage_but_does_not_prefit_future_anchors(self):
        bound = policy(
            layers=2, anchors=(MatrixKey(1, "K"),), anchor_mode="causal",
            staging_limit_bytes=256,
        )
        sources = LazySources(bound, {})
        prep = RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0)
        with mock.patch("torch.linalg.svd", side_effect=AssertionError("lookahead SVD")):
            prep.prepare_staged(sources, contexts(bound))
        self.assertFalse(sources.reads)
        self.assertEqual(prep.metrics["logical_staged_source_bytes"], 256)
        self.assertFalse(prep.metrics["streaming_memory_claim"])
        with self.assertRaises(StateError):
            prep.prepare_matrix(rank_matrix(1), contexts(bound)[MatrixKey(1, "K")])

    def test_hysteresis_cannot_hold_an_infeasible_prior_rank(self):
        bound = policy(hysteresis_fraction=1.0)
        first, _, state = complete(bound)
        matrices = {key: rank_matrix(2) for key in bound.keys}
        second, results, _ = complete(
            bound, matrices, prior_state=state, chunk_id="chunk-2", parent_head="head-1",
        )
        for prepared in results.values():
            self.assertEqual(prepared.decision.rank, 2)
            self.assertTrue(prepared.decision.guard_passed)
            self.assertNotIn("hysteresis_held_rank", prepared.decision.metrics)
        first.close()
        second.close()

    def test_guard_view_expands_same_fit_without_another_decomposition(self):
        bound = policy(solver="exact_oracle")
        matrix = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
        from reelcache import online_rank
        original = online_rank.project_rank_directions

        def reject_first(*args, **kwargs):
            result = original(*args, **kwargs)
            if result.rank == 3:
                metrics = dict(result.metrics)
                metrics["projection_relative_residual"] = 0.25
                return dataclasses.replace(result, metrics=metrics)
            return result

        with (
            mock.patch("reelcache.online_rank.project_rank_directions", side_effect=reject_first),
            mock.patch("torch.linalg.eigh", wraps=torch.linalg.eigh) as eigh,
        ):
            result = RankPreparation(
                bound, chunk_id="c", parent_head=None, token_start=0,
            ).prepare_matrix(matrix, contexts(bound)[MatrixKey(0, "K")])
        self.assertEqual(result.decision.exact_rank, 3)
        self.assertEqual(result.decision.rank, 4)
        self.assertEqual(eigh.call_count, 1)
        self.assertEqual(result.decision.metrics["projection_count"], 2)
        self.assertEqual(result.decision.metrics["view_expansions"], 1)

    def test_guard_attempt_limit_is_bounded_and_never_raw_fallback(self):
        bound = policy(solver="exact_oracle", max_guard_attempts=1)
        matrix = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
        from reelcache import online_rank
        original = online_rank.project_rank_directions

        def reject(*args, **kwargs):
            result = original(*args, **kwargs)
            return dataclasses.replace(result, metrics={
                **result.metrics, "projection_relative_residual": 1.0,
            })

        with mock.patch("reelcache.online_rank.project_rank_directions", side_effect=reject):
            with self.assertRaises(RankAbstention) as caught:
                RankPreparation(
                    bound, chunk_id="c", parent_head=None, token_start=0,
                ).prepare_matrix(matrix, contexts(bound)[MatrixKey(0, "K")])
        self.assertEqual(caught.exception.decision.reason, "guard_attempt_limit")
        self.assertEqual(caught.exception.decision.metrics["projection_count"], 1)
        self.assertIsNone(caught.exception.decision.resolved_spec)

    def test_policy_domain_and_state_mismatch_fail_before_decomposition(self):
        bound = policy()
        ctx = contexts(bound)[MatrixKey(0, "K")]
        with mock.patch("torch.linalg.svd", side_effect=AssertionError("invalid input SVD")):
            for bad in (
                dataclasses.replace(ctx, key_space="post_rope"),
                dataclasses.replace(ctx, source_dtype="bfloat16"),
                dataclasses.replace(ctx, token_start=4),
                dataclasses.replace(ctx, width=8),
            ):
                with self.assertRaises(ConfigurationError):
                    RankPreparation(
                        bound, chunk_id="c", parent_head=None, token_start=0,
                    ).prepare_matrix(rank_matrix(1), bad)
        _, _, state = complete(bound)
        with self.assertRaises(ConfigurationError):
            RankPreparation(dataclasses.replace(bound, seed=1), state, chunk_id="c2", parent_head="h", token_start=4)
        with self.assertRaises(ConfigurationError):
            RankPreparation(bound, state, chunk_id="c2", parent_head="h", token_start=8)
        with self.assertRaises(ConfigurationError):
            RankPreparation(bound, chunk_id="c", parent_head="existing-static-head", token_start=0)
        with self.assertRaises(StateError):
            RankPreparation(dataclasses.replace(bound, history_min=2), chunk_id="c", parent_head=None, token_start=0)

    def test_decision_roundtrip_nested_immutability_and_guard_invariant(self):
        bound = policy()
        prep, results, _ = complete(bound)
        decision = results[MatrixKey(0, "K")].decision
        self.assertEqual(RankDecision.from_dict(decision.to_dict()).to_dict(), decision.to_dict())
        with self.assertRaises(TypeError):
            decision.metrics["svd_count"] = 100
        with self.assertRaises(TypeError):
            decision.attempts[0]["fit_width"] = 100
        with self.assertRaises(ConfigurationError):
            dataclasses.replace(decision, guard_relative_residual=0.5)
        with self.assertRaises(ConfigurationError):
            dataclasses.replace(decision, guard_threshold=0.9)
        with self.assertRaises(ConfigurationError):
            dataclasses.replace(decision, basis_bytes=1)
        for missing in ("zero_input", "projection_relative_residual"):
            metrics = {key: value for key, value in decision.metrics.items() if key != missing}
            with self.assertRaises(ConfigurationError):
                dataclasses.replace(decision, metrics=metrics)
        prep.close()

    def test_source_dtype_and_generation_rng_remain_unchanged(self):
        bound = policy(source_dtypes={"K": "float32", "V": "bfloat16"})
        matrices = {key: rank_matrix(2, dtype=torch.float32 if key.target == "K" else torch.bfloat16)
                    for key in bound.keys}
        snapshots = {key: value.clone() for key, value in matrices.items()}
        before_torch = torch.get_rng_state().clone()
        before_python = random.getstate()
        prep, results, _ = complete(bound, matrices)
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
        self.assertEqual(before_python, random.getstate())
        for key, matrix in matrices.items():
            self.assertTrue(torch.equal(matrix, snapshots[key]))
            self.assertEqual(results[key].decision.context.dtype, bound.source_dtypes[key])
            self.assertEqual(results[key].decision.source_bytes, matrix.numel() * matrix.element_size())
            self.assertEqual(results[key].factors.coefficients.dtype, torch.float32)
        prep.close()

    def test_fit_attempt_limit_does_not_report_an_unfitted_larger_width(self):
        bound = policy(channels=8, rows=8, max_fit_attempts=1)
        with self.assertRaises(RankAbstention) as caught:
            RankPreparation(
                bound, chunk_id="c", parent_head=None, token_start=0,
            ).prepare_matrix(torch.eye(8), contexts(bound)[MatrixKey(0, "K")])
        self.assertEqual(caught.exception.decision.fit_width, 2)
        self.assertEqual(caught.exception.decision.metrics["fit_attempts"], 1)
        self.assertEqual(caught.exception.decision.metrics["svd_count"], 1)
        self.assertIsNone(caught.exception.decision.rank)

    def test_fit_and_projection_budgets_fail_before_the_unadmitted_operation(self):
        for allowance, expected_count, reason in (
            (500, 0, "fit_workspace_limit"),
            (600, 1, "projection_workspace_limit"),
        ):
            bound = policy(workspace_limit_bytes=allowance)
            with (
                mock.patch("torch.linalg.svd", wraps=torch.linalg.svd) as svd,
                mock.patch(
                    "reelcache.factorization.project_onto_basis",
                    side_effect=AssertionError("unadmitted projection"),
                ),
            ):
                with self.assertRaises(RankAbstention) as caught:
                    RankPreparation(
                        bound, chunk_id="c", parent_head=None, token_start=0,
                    ).prepare_matrix(rank_matrix(2), contexts(bound)[MatrixKey(0, "K")])
            self.assertEqual(caught.exception.decision.status, "OVER_BUDGET")
            self.assertEqual(caught.exception.decision.reason, reason)
            self.assertEqual(svd.call_count, expected_count)
            self.assertEqual(caught.exception.decision.metrics["projection_count"], 0)

    def test_failed_svd_work_is_retained_in_abstention_not_dropped_from_coverage(self):
        bound = policy()
        prep = RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0)
        with mock.patch("torch.linalg.svd", side_effect=RuntimeError("synthetic solver failure")):
            with self.assertRaises(RankAbstention) as caught:
                prep.prepare_matrix(rank_matrix(2), contexts(bound)[MatrixKey(0, "K")])
        self.assertEqual(caught.exception.decision.reason, "rank_decomposition_failed")
        self.assertEqual(caught.exception.decision.metrics["svd_count"], 1)
        self.assertEqual(caught.exception.decision.metrics["qr_count"], 5)
        self.assertEqual(prep.metrics["svd_count"], 1)
        self.assertEqual(prep.metrics["qr_count"], 5)
        with self.assertRaises(StateError):
            prep.next_state()

    def test_projection_backend_failure_keeps_completed_refits_and_attempted_work(self):
        bound = policy()
        prep = RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0)
        before_rng = torch.get_rng_state().clone()
        with (
            mock.patch("torch.linalg.svd", wraps=torch.linalg.svd) as svd,
            mock.patch("torch.linalg.qr", wraps=torch.linalg.qr) as qr,
            mock.patch("torch.linalg.cholesky_ex", side_effect=RuntimeError("conditioning backend fault")) as chol,
        ):
            with self.assertRaises(RankAbstention) as caught:
                prep.prepare_matrix(torch.eye(4), contexts(bound)[MatrixKey(0, "K")])
        decision = caught.exception.decision
        self.assertEqual((svd.call_count, qr.call_count, chol.call_count), (2, 10, 1))
        self.assertEqual(decision.reason, "projection_backend_failure")
        self.assertEqual(decision.metrics["svd_count"], 2)
        self.assertEqual(decision.metrics["qr_count"], 10)
        self.assertEqual(decision.metrics["projection_count"], 1)
        self.assertEqual(decision.metrics["cholesky_count"], 1)
        self.assertEqual(decision.metrics["coefficient_projection_count"], 0)
        self.assertEqual(decision.metrics["residual_evaluation_count"], 0)
        self.assertEqual(decision.metrics["matmul_count"], 25)
        self.assertEqual(decision.metrics["guard_attempts"], 1)
        self.assertEqual(decision.metrics["failed_projection_calls"], 1)
        self.assertEqual(decision.metrics["projection_failure_metrics"]["failure_phase"], "conditioning_cholesky")
        self.assertEqual(decision.attempts[-1]["failure_type"], "RuntimeError")
        self.assertGreater(decision.metrics["condition_seconds"], 0)
        self.assertGreater(decision.metrics["failed_projection_seconds"], 0)
        self.assertGreaterEqual(decision.metrics["decision_seconds"], decision.metrics["failed_projection_seconds"])
        self.assertTrue(decision.metrics["operation_counts_complete"])
        self.assertEqual(prep.failure_decision, decision)
        self.assertEqual(prep.metrics["svd_count"], 2)
        self.assertEqual(prep.metrics["qr_count"], 10)
        self.assertTrue(prep.metrics["failed"])
        self.assertEqual(prep.metrics["cached_factor_bytes"], 0)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
        with self.assertRaises(StateError):
            prep.next_state()
        with self.assertRaises(StateError):
            prep.prepare_matrix(torch.eye(4), contexts(bound)[MatrixKey(0, "V")])
        self.assertEqual(RankDecision.from_dict(decision.to_dict()).to_dict(), decision.to_dict())

    def test_uninstrumented_projection_failure_reports_unknown_not_zero_work(self):
        bound = policy()
        prep = RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0)
        with mock.patch("reelcache.online_rank.project_rank_directions", side_effect=RuntimeError("opaque backend")):
            with self.assertRaises(RankAbstention) as caught:
                prep.prepare_matrix(torch.eye(4), contexts(bound)[MatrixKey(0, "K")])
        decision = caught.exception.decision
        self.assertEqual(decision.metrics["svd_count"], 2)
        self.assertEqual(decision.metrics["qr_count"], 10)
        self.assertEqual(decision.metrics["guard_attempts"], 1)
        self.assertEqual(decision.metrics["known_work_before_failed_projection"]["matmul_count"], 24)
        for name in ("projection_count", "matmul_count", "source_norm_count", "cholesky_count"):
            self.assertIsNone(decision.metrics[name])
            self.assertIsNone(prep.metrics[name])
        self.assertFalse(decision.metrics["operation_counts_complete"])
        self.assertFalse(prep.metrics["operation_counts_complete"])
        self.assertIsNone(decision.metrics["projection_failure_metrics"])
        self.assertIsNone(decision.metrics["projection_guard_seconds"])
        self.assertGreater(decision.metrics["projection_seconds"], 0)
        self.assertGreater(decision.metrics["failed_projection_seconds"], 0)
        self.assertIs(prep.failure_decision, decision)
        self.assertEqual(RankDecision.from_dict(decision.to_dict()).to_dict(), decision.to_dict())
        with self.assertRaises(StateError):
            prep.next_state()

    def test_projection_failure_keeps_preceding_accepted_matrix_work(self):
        bound = policy()
        ctx = contexts(bound)
        prep = RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0)
        first = prep.prepare_matrix(rank_matrix(1), ctx[MatrixKey(0, "K")])
        with mock.patch("torch.linalg.cholesky_ex", side_effect=RuntimeError("later matrix backend")):
            with self.assertRaises(RankAbstention):
                prep.prepare_matrix(torch.eye(4), ctx[MatrixKey(0, "V")])
        self.assertEqual(prep.metrics["svd_count"], first.decision.metrics["svd_count"] + 2)
        self.assertEqual(prep.metrics["qr_count"], first.decision.metrics["qr_count"] + 10)
        self.assertEqual(prep.metrics["projection_count"], 2)
        self.assertEqual(prep.metrics["prepared_matrix_count"], 1)
        self.assertEqual(prep.failure_decision.key, MatrixKey(0, "V"))
        with self.assertRaises(StateError):
            prep.next_state()

    def test_missing_inner_projection_instrumentation_retains_the_known_fit_prefix(self):
        bound = policy()
        prep = RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0)
        with mock.patch(
            "reelcache.factorization.project_onto_basis", side_effect=RuntimeError("opaque lower backend"),
        ):
            with self.assertRaises(RankAbstention) as caught:
                prep.prepare_matrix(torch.eye(4), contexts(bound)[MatrixKey(0, "K")])
        metrics = caught.exception.decision.metrics
        self.assertEqual(metrics["svd_count"], 2)
        self.assertEqual(metrics["qr_count"], 10)
        self.assertEqual(metrics["known_work_before_failed_projection"]["matmul_count"], 24)
        self.assertIsNone(metrics["matmul_count"])
        self.assertIsNone(metrics["projection_count"])
        self.assertIsNone(metrics["condition_seconds"])
        self.assertEqual(metrics["projection_failure_metrics"]["failure_phase"], "projection_entry")
        self.assertFalse(prep.metrics["operation_counts_complete"])
        self.assertGreater(metrics["basis_round_seconds"], 0)
        self.assertEqual(RankDecision.from_dict(caught.exception.decision.to_dict()).to_dict(),
                         caught.exception.decision.to_dict())

    def test_staged_anchor_projection_failure_keeps_work_without_candidate_state(self):
        key = MatrixKey(0, "K")
        bound = policy(anchors=(key,), anchor_mode="staged", staging_limit_bytes=128)
        prep = RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0)
        with mock.patch("torch.linalg.cholesky_ex", side_effect=RuntimeError("anchor backend")):
            with self.assertRaises(RankAbstention):
                prep.prepare_staged({item: torch.eye(4) for item in bound.keys}, contexts(bound))
        self.assertEqual(prep.metrics["svd_count"], 1)
        self.assertEqual(prep.metrics["qr_count"], 5)
        self.assertEqual(prep.metrics["projection_count"], 1)
        self.assertEqual(prep.metrics["cached_factor_bytes"], 0)
        self.assertEqual(prep.failure_decision.key, key)
        with self.assertRaises(StateError):
            prep.next_state()

    def test_guard_recipe_and_tiling_are_bound_to_the_policy(self):
        bound = policy()
        prepared = RankPreparation(
            bound, chunk_id="c", parent_head=None, token_start=0,
        ).prepare_matrix(rank_matrix(1), contexts(bound)[MatrixKey(0, "K")])
        for name, value in (
            ("tile_rows", 1024), ("tile_rows", True),
            ("projection_convention", "tile_dependent_coefficients"),
            ("guard_domain", "rounded_source"), ("guard_is_pre_codec", False),
        ):
            altered = dataclasses.replace(
                prepared.decision, metrics={**prepared.decision.metrics, name: value},
            )
            with self.assertRaises(IntegrityError):
                bound.validate_decision(altered)

    def test_prepared_codec_repeats_the_actual_small_tile_guard(self):
        from reelcache.codecs import encode_tensor, factor_binding

        keys = (MatrixKey(0, "K"), MatrixKey(0, "V"))
        matrix = torch.randn(129, 32, generator=torch.Generator().manual_seed(101))
        for tile_rows in (1024, 2):
            with self.subTest(tile_rows=tile_rows):
                bound = policy(
                    channels=32, rows=129, tile_rows=tile_rows,
                    allowed_ranks={key: (32,) for key in keys},
                )
                ctx = contexts(bound)[keys[0]]
                prepared = RankPreparation(
                    bound, chunk_id="c", parent_head=None, token_start=0,
                ).prepare_matrix(matrix, ctx)
                self.assertEqual(prepared.decision.metrics["tile_rows"], tile_rows)
                with mock.patch("torch.linalg.svd", side_effect=AssertionError("duplicate codec SVD")):
                    binding = factor_binding(matrix, ctx, prepared.spec, prepared.factors, bound.fingerprint)
                    packet = encode_tensor(
                        matrix, ctx, prepared.spec,
                        prepared_factors=prepared.factors, prepared_binding=binding,
                    )
                self.assertEqual(packet.metadata["codec"]["rank"], prepared.spec.rank)

    def test_cached_anchor_factors_participate_in_workspace_admission(self):
        anchors = (MatrixKey(0, "K"), MatrixKey(1, "K"))
        bound = policy(
            layers=2, anchors=anchors, anchor_mode="staged",
            staging_limit_bytes=256, workspace_limit_bytes=1409,
        )
        ctx = contexts(bound)
        prep = RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0)
        sources = {key: rank_matrix(1) for key in bound.keys}
        with self.assertRaises(RankAbstention) as caught:
            prep.prepare_staged(sources, ctx)
        self.assertEqual(caught.exception.decision.status, "OVER_BUDGET")
        self.assertEqual(caught.exception.decision.metrics["cached_factor_bytes_before"], 24)
        self.assertEqual(prep.metrics["svd_count"], 1)
        self.assertEqual(prep.metrics["cached_factor_bytes"], 0)

    def test_core_context_set_and_state_replay_validation_helpers(self):
        bound = policy()
        self.assertEqual(bound.identity, bound.fingerprint)
        self.assertIsNone(bound.reuse_threshold)
        ctx = contexts(bound)
        bound.validate_contexts(ctx, history_chunks=1)
        with self.assertRaises(StateError):
            bound.validate_contexts(ctx, history_chunks=4)
        with self.assertRaises(ConfigurationError):
            bound.validate_contexts({MatrixKey(0, "K"): ctx[MatrixKey(0, "K")]})
        bad = dict(ctx)
        bad[MatrixKey(0, "V")] = dataclasses.replace(bad[MatrixKey(0, "V")], token_start=4)
        with self.assertRaises(ConfigurationError):
            bound.validate_contexts(bad)
        prep, results, state = complete(bound)
        self.assertEqual(state.policy_identity, bound.identity)
        state.validate(bound, last_chunk_id="chunk-1", parent_head=None, token_end=4, completed_chunks=1)
        for wrong in (
            {"last_chunk_id": "wrong"}, {"parent_head": "wrong"},
            {"token_end": 8}, {"completed_chunks": 2},
        ):
            with self.assertRaises(IntegrityError):
                state.validate(bound, **wrong)
        with self.assertRaises(IntegrityError):
            state.validate(dataclasses.replace(bound, seed=1))
        decision = results[MatrixKey(0, "K")].decision
        self.assertEqual(decision.policy_identity, bound.identity)
        self.assertEqual(decision.source_sha256, decision.input_sha256)
        self.assertEqual(decision.spec, decision.resolved_spec)
        prep.close()

    def test_core_codec_consumes_the_accepted_factors_without_a_second_svd(self):
        from reelcache.codecs import decode_tensor, encode_tensor, factor_binding

        bound = policy(energy_target=0.99)
        ctx = contexts(bound)[MatrixKey(0, "K")]
        matrices = [
            torch.zeros(4, 4),
            torch.randn(4, 4, generator=torch.Generator().manual_seed(943)),
        ]
        for matrix in matrices:
            prepared = RankPreparation(
                bound, chunk_id="c", parent_head=None, token_start=0,
            ).prepare_matrix(matrix, ctx)
            binding = factor_binding(matrix, ctx, prepared.spec, prepared.factors, bound.identity)
            with (
                mock.patch("torch.linalg.svd", side_effect=AssertionError("encoder refit SVD")),
                mock.patch("torch.linalg.eigh", side_effect=AssertionError("encoder refit eigh")),
            ):
                packet = encode_tensor(
                    matrix, ctx, prepared.spec,
                    prepared_factors=prepared.factors, prepared_binding=binding,
                )
            self.assertEqual(packet.metadata["basis"]["sha256"], prepared.decision.basis_sha256)
            self.assertEqual(
                packet.metadata["payloads"]["coefficients.f32"]["sha256"],
                prepared.decision.coefficients_sha256,
            )
            self.assertEqual(packet.metrics["prepared_validation"]["decomposition_count"], 0)
            self.assertEqual(packet.metrics["prepared_validation"]["projection_count"], 1)
            reconstructed = decode_tensor(packet, dtype=torch.float32)
            self.assertTrue(torch.equal(
                reconstructed, prepared.factors.coefficients @ prepared.factors.basis.float(),
            ))

    def test_uninstrumented_oracle_failure_is_not_reported_as_measured_zero_work(self):
        from reelcache.spectra import SpectrumNumericalError

        bound = policy(solver="exact_oracle")
        prep = RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0)
        with mock.patch(
            "reelcache.spectra.exact_spectrum",
            side_effect=SpectrumNumericalError("synthetic uninstrumented failure"),
        ):
            with self.assertRaises(RankAbstention) as caught:
                prep.prepare_matrix(rank_matrix(2), contexts(bound)[MatrixKey(0, "K")])
        self.assertEqual(caught.exception.decision.reason, "exact_oracle_numerical_failure")
        self.assertFalse(caught.exception.decision.metrics["operation_counts_complete"])
        self.assertEqual(caught.exception.decision.metrics["failed_oracle_calls"], 1)
        self.assertIsNone(caught.exception.decision.metrics["decomposition_count"])
        self.assertIsNone(caught.exception.decision.metrics["eigh_count"])
        self.assertIsNone(caught.exception.decision.metrics["oracle_failure_metrics"])
        self.assertFalse(prep.metrics["operation_counts_complete"])
        self.assertIsNone(prep.metrics["eigh_count"])
        self.assertIsNone(caught.exception.decision.resolved_spec)

    def test_actual_failed_oracle_eigh_keeps_attempted_operation_counts(self):
        bound = policy(solver="exact_oracle")
        prep = RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0)
        with mock.patch("torch.linalg.eigh", side_effect=RuntimeError("synthetic failed eigh")):
            with self.assertRaises(RankAbstention) as caught:
                prep.prepare_matrix(rank_matrix(2), contexts(bound)[MatrixKey(0, "K")])
        failed = caught.exception.decision
        self.assertEqual(failed.reason, "exact_oracle_numerical_failure")
        self.assertTrue(failed.metrics["operation_counts_complete"])
        self.assertEqual(failed.metrics["decomposition_count"], 1)
        self.assertEqual(failed.metrics["eigh_count"], 1)
        self.assertEqual(failed.metrics["svd_count"], 0)
        self.assertEqual(failed.metrics["qr_count"], 0)
        self.assertEqual(failed.metrics["projection_count"], 0)
        self.assertGreater(failed.metrics["oracle_failure_metrics"]["total_seconds"], 0)
        self.assertGreaterEqual(failed.metrics["oracle_seconds"], failed.metrics["decomposition_seconds"])
        self.assertEqual(prep.metrics["eigh_count"], 1)

    def test_pre_oracle_solve_numerical_failure_is_explicit_zero_decomposition(self):
        bound = policy(solver="exact_oracle")
        prep = RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0)
        with (
            mock.patch("torch.linalg.vector_norm", return_value=torch.tensor(1e100, dtype=torch.float64)),
            mock.patch("torch.linalg.eigh", side_effect=AssertionError("unreached eigensolve")),
        ):
            with self.assertRaises(RankAbstention) as caught:
                prep.prepare_matrix(rank_matrix(2), contexts(bound)[MatrixKey(0, "K")])
        self.assertTrue(caught.exception.decision.metrics["operation_counts_complete"])
        self.assertEqual(caught.exception.decision.metrics["eigh_count"], 0)
        self.assertEqual(caught.exception.decision.metrics["decomposition_count"], 0)
        self.assertGreater(caught.exception.decision.metrics["oracle_seconds"], 0)

    def test_exact_wide_matrix_keeps_full_channel_spectrum_but_legal_operational_rank(self):
        bound = policy(rows=3, channels=7, solver="exact_oracle")
        matrix = torch.randn(3, 7, generator=torch.Generator().manual_seed(844))
        key = MatrixKey(0, "K")
        prepared = RankPreparation(
            bound, chunk_id="c", parent_head=None, token_start=0,
        ).prepare_matrix(matrix, contexts(bound)[key])
        self.assertLessEqual(prepared.decision.rank, 3)
        self.assertEqual(len(prepared.decision.oracle_observation["eigenvalues_scaled"]), 7)
        self.assertEqual(
            tuple(prepared.decision.oracle_observation["eigenvalues_scaled"])[3:], (0, 0, 0, 0),
        )
        self.assertEqual(tuple(prepared.factors.basis.shape), (prepared.decision.rank, 7))
        self.assertEqual(prepared.decision.basis_bytes, 2 * prepared.decision.rank * 7)
        self.assertLess(prepared.decision.guard_relative_residual, prepared.decision.guard_threshold)


def predictor_parameters(**changes):
    values = {
        "cell": "synthetic", "target": "K", "tau": 0.95,
        "history_regime": "later", "execution_mode": "causal_lagged",
        "layer_template": [2.0] * 4,
        "detector_baselines": {"startup": [1.0] * 4, "later": [2.0] * 4},
        "anchor_layers": [3], "rank_margin": 0,
        "rank_buckets": [1, 2, 4, 8], "policy_kind": "anchors",
        "uniform_rank": None, "training_groups": ["cal-b", "cal-a"],
    }
    values.update(changes)
    return values


class FixedAnchorDeploymentTests(unittest.TestCase):
    @staticmethod
    def profiles(template=1.0, group="fold-a", detector_spec=None):
        spec = DetectorSpec() if detector_spec is None else detector_spec
        return {
            f"{regime}:{target}": RankPredictorProfile(
                cell="synthetic", target=target, tau=.95, history_regime=regime,
                execution_mode="causal_lagged", layer_template=(template,),
                detector_baselines={"startup": (1.0,), "later": (1.0,)},
                anchor_layers=(0,) if target == "K" else (), rank_margin=0,
                rank_buckets=(1, 2, 3, 4),
                policy_kind="anchors" if target == "K" else "fixed_layerwise",
                uniform_rank=None, training_groups=(group,),
                detector_fingerprint=spec.fingerprint,
            )
            for regime in ("startup", "later") for target in ("K", "V")
        }

    def test_fold_templates_profiles_policy_seed_and_metadata_do_not_change_acquisition(self):
        self.assert_template_independent_acquisition(DetectorSpec())

    def assert_template_independent_acquisition(self, detector_spec):
        from reelcache import online_rank

        matrix = rank_matrix(2)
        base = policy(detector_spec=detector_spec)
        ctx = contexts(base)[MatrixKey(0, "K")]
        offline = acquire_anchor_directions(matrix, ctx, spec=base.detector_spec, solver_workspace_bytes=0)
        captured = []
        project = online_rank.project_rank_directions

        def record_directions(value, directions, rank, **kwargs):
            captured.append(directions.clone())
            return project(value, directions, rank, **kwargs)

        decisions = []
        for template, seed, group in ((1.0, 7, "fold-a"), (3.0, 991, "fold-b")):
            bound = bind_predictor_profiles(
                dataclasses.replace(base, seed=seed),
                self.profiles(template, group, detector_spec=detector_spec),
            )
            provenance = {key: {"labels": {"fold": group, "generation_seed": seed}} for key in bound.keys}
            with (
                mock.patch("reelcache.online_rank.acquire_anchor_directions", wraps=acquire_anchor_directions) as acquire,
                mock.patch("reelcache.online_rank.project_rank_directions", side_effect=record_directions),
                mock.patch("torch.linalg.svd", wraps=torch.linalg.svd) as svd,
            ):
                result = RankPreparation(
                    bound, chunk_id="c", parent_head=None, token_start=0, source_provenance=provenance,
                ).prepare_matrix(matrix, ctx)
            self.assertEqual(acquire.call_count, 1)
            self.assertEqual(svd.call_count, offline.metrics["svd_count"])
            self.assertEqual(result.decision.estimated_rank, offline.r95)
            self.assertEqual(result.decision.metrics["anchor_acquisition"]["features"]["0.99"], offline.r99)
            self.assertEqual(result.decision.metrics["source_copy_count"], 1)
            self.assertEqual(result.decision.metrics["anchor_acquisition"]["source_sha256"], offline.source_sha256)
            self.assertTrue(torch.equal(result.factors.basis, offline.directions[:result.factors.rank].half()))
            decisions.append(result.decision)
        self.assertTrue(torch.equal(captured[0], offline.directions))
        self.assertTrue(torch.equal(captured[1], offline.directions))
        self.assertNotEqual(decisions[0].initial_rank, decisions[1].initial_rank)
        self.assertEqual(decisions[0].metrics["factorization_seed"], decisions[1].metrics["factorization_seed"])

    def test_frozen_thread_count_keeps_features_and_directions_profile_independent(self):
        previous_threads = torch.get_num_threads()
        for count in (2, 8):
            with self.subTest(cpu_threads=count):
                try:
                    torch.set_num_threads(count)
                    with mock.patch("torch.set_num_threads", side_effect=AssertionError("deployment changed threads")):
                        self.assert_template_independent_acquisition(DetectorSpec(cpu_threads=count))
                    self.assertEqual(torch.get_num_threads(), count)
                finally:
                    torch.set_num_threads(previous_threads)
        self.assertEqual(torch.get_num_threads(), previous_threads)

    def test_causal_and_staged_anchors_accept_only_gpu_report_changes_at_cpu8(self):
        from tests.test_online_factorization import with_cuda_runtime_diagnostics

        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(8)
            spec = DetectorSpec(cpu_threads=8)
            report = with_cuda_runtime_diagnostics(torch.__config__.show())
            matrix = rank_matrix(2)
            before = torch.get_rng_state().clone()
            for mode in ("causal", "staged"):
                with self.subTest(mode=mode):
                    bound = policy(
                        layers=2, anchors=(MatrixKey(1, "K"), MatrixKey(1, "V")),
                        anchor_mode=mode, detector_spec=spec, staging_limit_bytes=256,
                    )
                    matrices = {key: matrix for key in bound.keys}
                    _, offline, _ = complete(bound, matrices)
                    with (
                        mock.patch("torch.__config__.show", return_value=report),
                        mock.patch("torch.set_num_threads", side_effect=AssertionError("changed caller threads")),
                        mock.patch("torch.cuda._lazy_init", side_effect=AssertionError("initialized CUDA")),
                        mock.patch("torch.load", side_effect=AssertionError("loaded weights")),
                    ):
                        _, online, state = complete(bound, matrices)
                    for key in bound.anchors:
                        self.assertEqual(online[key].decision.estimated_rank, offline[key].decision.estimated_rank)
                        self.assertEqual(online[key].decision.metrics["detector_fingerprint"], spec.fingerprint)
                        self.assertTrue(torch.equal(online[key].factors.basis, offline[key].factors.basis))
                        self.assertEqual(state.anchor_ranks[key], online[key].decision.estimated_rank)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(torch.get_num_threads(), 8)
            self.assertFalse(torch.cuda.is_initialized())
        finally:
            torch.set_num_threads(previous_threads)

    def test_encoding_refit_cannot_rewrite_fixed_feature_or_committed_scalar_state(self):
        from reelcache import online_rank

        key = MatrixKey(0, "K")
        bound = policy(
            channels=160, rows=160, anchors=(key,), anchor_mode="causal",
            allowed_ranks={item: (8, 128, 160) for item in (key, MatrixKey(0, "V"))},
        )
        matrix = rank_matrix(1, channels=160, rows=160)
        ctx = contexts(bound)
        fit_refit = online_rank.fit_rank_directions
        project = online_rank.project_rank_directions

        def changed_encoder_estimate(*args, **kwargs):
            fitted = fit_refit(*args, **kwargs)
            spectrum = torch.zeros(fitted.width, dtype=torch.float64)
            spectrum[:2] = torch.tensor([.94, .06], dtype=torch.float64)
            return dataclasses.replace(fitted, squared_singular_values=spectrum)

        def require_refit(value, directions, rank, **kwargs):
            result = project(value, directions, rank, **kwargs)
            if directions.shape[0] == 128:
                return dataclasses.replace(result, metrics={
                    **result.metrics, "projection_relative_residual": 1.0,
                })
            return result

        prep = RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0)
        with (
            mock.patch("reelcache.online_rank.fit_rank_directions", side_effect=changed_encoder_estimate) as refit,
            mock.patch("reelcache.online_rank.project_rank_directions", side_effect=require_refit),
        ):
            result = prep.prepare_matrix(matrix, ctx[key])
        self.assertEqual(refit.call_count, 1)
        self.assertEqual(result.decision.metrics["svd_count"], 2)
        self.assertEqual(result.decision.metrics["encoding_estimated_rank"], 2)
        self.assertEqual(result.decision.metrics["anchor_feature_rank"], 1)
        self.assertEqual(result.decision.estimated_rank, 1)
        self.assertEqual(result.decision.metrics["anchor_feature_fit_width"], 128)
        self.assertEqual(result.decision.fit_width, 160)
        prep.prepare_matrix(matrix, ctx[MatrixKey(0, "V")])
        self.assertEqual(prep.next_state().anchor_ranks[key], 1)

    def test_zero_anchor_policy_never_calls_acquisition(self):
        bound = policy()
        with mock.patch("reelcache.online_rank.acquire_anchor_directions", side_effect=AssertionError("extra detector")):
            result = RankPreparation(
                bound, chunk_id="c", parent_head=None, token_start=0,
            ).prepare_matrix(rank_matrix(1), contexts(bound)[MatrixKey(0, "K")])
        self.assertFalse(result.decision.metrics["fixed_anchor_acquisition"])
        self.assertEqual(result.decision.metrics["source_copy_count"], 0)

    def test_profiles_and_decisions_cannot_swap_detector_recipe_or_feature(self):
        base = policy()
        profiles = self.profiles()
        profiles["startup:K"] = dataclasses.replace(profiles["startup:K"], detector_fingerprint="a" * 64)
        with self.assertRaises(ConfigurationError):
            bind_predictor_profiles(base, profiles)
        payload = self.profiles()["startup:K"].to_dict()
        del payload["detector_fingerprint"]
        with self.assertRaises(ConfigurationError):
            RankPredictorProfile.from_dict(payload)
        bound = bind_predictor_profiles(base, self.profiles())
        result = RankPreparation(
            bound, chunk_id="c", parent_head=None, token_start=0,
        ).prepare_matrix(rank_matrix(1), contexts(bound)[MatrixKey(0, "K")])
        changed = dataclasses.replace(result.decision, estimated_rank=2)
        with self.assertRaises(IntegrityError):
            bound.validate_decision(changed)
        self.assertEqual(bound.detector_spec.fingerprint, DetectorSpec().fingerprint)

    def test_fixed_acquisition_failure_retains_attempts_and_poisoned_state(self):
        key = MatrixKey(0, "K")
        bound = policy(anchors=(key,), anchor_mode="causal")
        prep = RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0)
        with mock.patch("torch.linalg.svd", side_effect=RuntimeError("fixed acquisition backend")):
            with self.assertRaises(RankAbstention) as caught:
                prep.prepare_matrix(torch.eye(4), contexts(bound)[key])
        self.assertEqual(caught.exception.decision.reason, "fixed_detector_fit_failed")
        self.assertEqual(prep.metrics["svd_count"], 1)
        self.assertEqual(prep.metrics["qr_count"], 5)
        self.assertEqual(prep.metrics["source_copy_count"], 1)
        self.assertGreater(caught.exception.decision.metrics["detector_seconds"], 0)
        with self.assertRaises(StateError):
            prep.next_state()

    def test_zero_anchors_grow_only_the_admitted_operational_basis(self):
        key = MatrixKey(0, "K")
        for rows, channels, rank in (
            (160, 160, 160), (160, 192, 160), (192, 160, 160),
            (192, 256, 160), (32, 160, 32), (160, 32, 32),
        ):
            for mode in ("causal", "staged"):
                with self.subTest(rows=rows, channels=channels, rank=rank, mode=mode):
                    bound = policy(
                        rows=rows, channels=channels, anchors=(key,), anchor_mode=mode,
                        allowed_ranks={item: (rank,) for item in (key, MatrixKey(0, "V"))},
                        staging_limit_bytes=2 * rows * channels * 4,
                    )
                    matrices = {item: torch.zeros(rows, channels) for item in bound.keys}
                    ctx = contexts(bound)
                    before_rng = torch.get_rng_state().clone()
                    before_python = random.getstate()
                    prep = RankPreparation(bound, chunk_id="zero", parent_head=None, token_start=0)
                    with (
                        mock.patch("torch.linalg.svd", side_effect=AssertionError("zero encoding SVD")),
                        mock.patch("torch.linalg.qr", side_effect=AssertionError("zero encoding QR")),
                        mock.patch("torch.linalg.eigh", side_effect=AssertionError("zero encoding oracle")),
                        mock.patch("torch.randn", side_effect=AssertionError("zero encoding RNG")),
                    ):
                        if mode == "staged":
                            prep.prepare_staged(matrices, ctx)
                        prepared = prep.prepare_matrix(matrices[key], ctx[key])
                        prep.prepare_matrix(matrices[MatrixKey(0, "V")], ctx[MatrixKey(0, "V")])
                        state = prep.next_state()
                    decision = prepared.decision
                    self.assertEqual(decision.rank, rank)
                    self.assertEqual(decision.fit_width, rank)
                    self.assertEqual(decision.estimated_rank, 0)
                    self.assertEqual(decision.metrics["anchor_acquisition"]["features"], {"0.95": 0, "0.99": 0})
                    self.assertEqual(decision.metrics["anchor_feature_fit_width"], min(128, rows, channels))
                    self.assertEqual(decision.metrics["fit_attempts"], 2 if rank > 128 else 1)
                    self.assertEqual(decision.metrics["factorization_method"], "zero_input_identity")
                    self.assertEqual(decision.metrics["guard_attempts"], 1)
                    self.assertEqual(decision.metrics["projection_count"], 1)
                    self.assertEqual(decision.metrics["decomposition_count"], 0)
                    self.assertEqual(decision.metrics["qr_count"], 0)
                    self.assertEqual(decision.metrics["source_norm_count"], 3 if rank > 128 else 2)
                    self.assertEqual(decision.metrics["source_copy_count"], 1)
                    self.assertEqual(decision.metrics["source_copy_bytes"], 4 * rows * channels)
                    self.assertEqual(decision.guard_relative_residual, 0)
                    self.assertTrue(decision.guard_passed)
                    self.assertEqual(decision.basis_bytes, 2 * rank * channels)
                    self.assertEqual(decision.coefficient_bytes, 4 * rows * rank)
                    self.assertEqual(tuple(prepared.factors.basis.shape), (rank, channels))
                    self.assertTrue(torch.equal(prepared.factors.basis, torch.eye(rank, channels).half()))
                    self.assertEqual(int(torch.count_nonzero(prepared.factors.coefficients)), 0)
                    self.assertEqual(state.anchor_ranks[key], 0)
                    self.assertEqual(state.selected_ranks[key], rank)
                    self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
                    self.assertEqual(random.getstate(), before_python)
                    prep.close()

    def test_zero_anchor_growth_capacity_is_checked_before_identity_allocation(self):
        key = MatrixKey(0, "K")
        rows = channels = rank = 160
        allowance = 4 * rows * channels + rank_fit_workspace_bytes(
            (rows, channels), 128, tile_rows=DetectorSpec().tile_rows,
        )
        bound = policy(
            rows=rows, channels=channels, anchors=(key,), anchor_mode="causal", tile_rows=1024,
            allowed_ranks={item: (rank,) for item in (key, MatrixKey(0, "V"))},
            workspace_limit_bytes=allowance,
        )
        before_rng = torch.get_rng_state().clone()
        prep = RankPreparation(bound, chunk_id="zero", parent_head=None, token_start=0)
        real_eye = torch.eye
        allocations = []

        def admitted_eye(size, *args, **kwargs):
            allocations.append(size)
            if size > 128:
                raise AssertionError("unadmitted operational identity")
            return real_eye(size, *args, **kwargs)

        with (
            mock.patch("torch.eye", side_effect=admitted_eye),
            mock.patch("reelcache.online_rank.project_rank_directions", side_effect=AssertionError("unadmitted guard")),
        ):
            with self.assertRaises(RankAbstention) as caught:
                prep.prepare_matrix(torch.zeros(rows, channels), contexts(bound)[key])
        decision = caught.exception.decision
        self.assertEqual(allocations, [128])
        self.assertEqual(decision.status, "OVER_BUDGET")
        self.assertEqual(decision.reason, "fit_workspace_limit")
        self.assertEqual(decision.metrics["anchor_acquisition"]["features"], {"0.95": 0, "0.99": 0})
        self.assertEqual(decision.metrics["projection_count"], 0)
        self.assertEqual(decision.metrics["decomposition_count"], 0)
        self.assertEqual(decision.metrics["source_norm_count"], 1)
        self.assertEqual(decision.metrics["guard_attempts"], 0)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
        with self.assertRaises(StateError):
            prep.next_state()


class CalibrationBridgeTests(unittest.TestCase):
    def test_profile_roundtrip_is_frozen_and_has_no_model_or_outcome_inputs(self):
        profile = profile_builder(predictor_parameters())
        restored = RankPredictorProfile.from_dict(profile.to_dict())
        self.assertEqual(restored.fingerprint, profile.fingerprint)
        self.assertEqual(profile.training_groups, ("cal-a", "cal-b"))
        with self.assertRaises(TypeError):
            profile.detector_baselines["startup"] = (8.0,) * 4
        copy = profile.to_dict()
        copy["layer_template"][0] = 8
        self.assertEqual(profile.layer_template[0], 2)
        with self.assertRaises(ConfigurationError):
            profile_builder({**predictor_parameters(), "future_quality": 0.9})

    def test_lagged_startup_and_later_baselines_are_not_interchanged(self):
        profile = profile_builder(predictor_parameters())
        kwargs = dict(layer=0, current_anchors={3: float("nan")}, previous_anchors={3: 2})
        self.assertEqual(rank_predictor(profile, **kwargs, previous_regime="startup"), 4)
        self.assertEqual(rank_predictor(profile, **kwargs, previous_regime="later"), 2)
        self.assertEqual(rank_predictor(
            profile, layer=0, current_anchors={3: 8}, previous_anchors={}, previous_regime=None,
        ), 2)
        with self.assertRaises(ConfigurationError):
            rank_predictor(profile, **kwargs, previous_regime=None)

    def test_current_late_anchor_is_only_available_after_the_staged_barrier(self):
        staged = profile_builder(predictor_parameters(execution_mode="whole_clean_staged"))
        causal = profile_builder(predictor_parameters())
        kwargs = dict(layer=0, current_anchors={3: 4}, previous_anchors={}, previous_regime=None)
        self.assertEqual(rank_predictor(staged, **kwargs), 4)
        self.assertEqual(rank_predictor(causal, **kwargs), 2)
        early = profile_builder(predictor_parameters(anchor_layers=[0]))
        self.assertEqual(rank_predictor(
            early, layer=2, current_anchors={0: 4}, previous_anchors={}, previous_regime=None,
        ), 4)

    def test_anchor_itself_uses_template_instead_of_circular_own_measurement(self):
        profile = profile_builder(predictor_parameters(execution_mode="whole_clean_staged"))
        self.assertEqual(rank_predictor(
            profile, layer=3, current_anchors={3: float("nan")},
            previous_anchors={3: 8}, previous_regime="startup",
        ), 2)

    def test_uniform_margin_is_not_double_applied_and_layerwise_margin_is_applied(self):
        uniform = profile_builder(predictor_parameters(
            anchor_layers=[], policy_kind="fixed_uniform", uniform_rank=4, rank_margin=3,
        ))
        layerwise = profile_builder(predictor_parameters(
            anchor_layers=[], policy_kind="fixed_layerwise", rank_margin=3,
        ))
        kwargs = dict(layer=0, current_anchors={}, previous_anchors={}, previous_regime=None)
        with (
            mock.patch("torch.linalg.svd", side_effect=AssertionError("predictor SVD")),
            mock.patch("torch.linalg.eigh", side_effect=AssertionError("predictor eigh")),
            mock.patch("torch.randn", side_effect=AssertionError("predictor RNG")),
        ):
            self.assertEqual(rank_predictor(uniform, **kwargs), 4)
            self.assertEqual(rank_predictor(layerwise, **kwargs), 8)

    def test_missing_lagged_baseline_falls_back_without_oracle_substitution(self):
        profile = profile_builder(predictor_parameters(detector_baselines={
            "startup": [None] * 4, "later": [2.0] * 4,
        }))
        self.assertEqual(rank_predictor(
            profile, layer=0, current_anchors={}, previous_anchors={3: 8}, previous_regime="startup",
        ), 2)
        with self.assertRaises(ConfigurationError):
            profile_builder(predictor_parameters(detector_baselines={
                "startup": [1.0] * 4, "later": [None] * 4,
            }))

    def test_bound_regime_profiles_share_the_actual_online_and_replay_predictor(self):
        profiles = {}
        for regime in ("startup", "later"):
            for target in ("K", "V"):
                anchors = ([0] if regime == "startup" else [1, 3]) if target == "K" else []
                profiles[f"{regime}:{target}"] = profile_builder(predictor_parameters(
                    target=target, history_regime=regime, anchor_layers=anchors,
                    policy_kind="anchors" if anchors else "fixed_layerwise",
                    layer_template=[1.0] * 4, rank_buckets=[1, 2, 3, 4],
                ))
        bound = bind_predictor_profiles(policy(layers=4), profiles)
        self.assertEqual(bound.active_anchors(1), (MatrixKey(0, "K"),))
        self.assertEqual(bound.active_anchors(2), (MatrixKey(1, "K"), MatrixKey(3, "K")))
        self.assertEqual(len(bound.anchors), 3)
        self.assertEqual(RankPolicy.from_dict(bound.to_dict()).fingerprint, bound.fingerprint)
        matrices = {
            key: rank_matrix(2 if key in {MatrixKey(1, "K"), MatrixKey(3, "K")} else 1)
            for key in bound.keys
        }
        first, _, state = complete(bound, matrices)
        self.assertEqual(set(state.anchor_ranks), set(bound.anchors))
        second, results, state2 = complete(
            bound, matrices, prior_state=state, chunk_id="chunk-2", parent_head="head-1",
        )
        for key, prepared in results.items():
            prediction = bound.predict_rank(
                key,
                current_anchors={anchor: result.decision.estimated_rank for anchor, result in results.items()},
                prior_state=state,
            )
            self.assertEqual(prediction.initial_rank, prepared.decision.initial_rank)
            self.assertEqual(prediction.prediction_sources, prepared.decision.prediction_sources)
        early = results[MatrixKey(0, "K")].decision
        callback = rank_predictor(
            profiles["later:K"], layer=0, current_anchors={},
            previous_anchors={1: 2, 3: 2}, previous_regime="startup",
        )
        self.assertEqual(early.initial_rank, callback)
        self.assertEqual(early.initial_rank, 2)
        self.assertEqual(early.prediction_sources[0]["baseline_regime"], "startup")
        self.assertEqual(early.prediction_sources[0]["detector_baseline"], 1)
        third_prediction = bound.predict_rank(
            MatrixKey(0, "K"), current_anchors={}, prior_state=state2,
        )
        self.assertEqual(third_prediction.initial_rank, 1)
        self.assertEqual(third_prediction.prediction_sources[0]["baseline_regime"], "later")
        first.close()
        second.close()

    def test_bound_profiles_require_complete_history_and_admitted_staging(self):
        parameters = predictor_parameters(
            anchor_layers=[], policy_kind="fixed_layerwise", rank_buckets=[1, 2, 3, 4],
            execution_mode="whole_clean_staged",
        )
        profiles = {
            f"{regime}:{target}": profile_builder({
                **parameters, "target": target, "history_regime": regime,
            })
            for regime in ("startup", "later") for target in ("K", "V")
        }
        with self.assertRaises(ConfigurationError):
            bind_predictor_profiles(policy(layers=4), {"startup:K": profiles["startup:K"]})
        with self.assertRaises(CapacityError):
            bind_predictor_profiles(policy(layers=4), profiles)
        bound = bind_predictor_profiles(policy(layers=4, staging_limit_bytes=512), profiles)
        self.assertEqual(bound.anchor_mode, "staged")
        self.assertFalse(bound.active_anchors(1))

    def test_public_pure_seam_is_used_by_runtime_and_calibration_without_fake_state(self):
        profile = profile_builder(predictor_parameters())
        bound = policy()
        from reelcache import online_rank
        with mock.patch("reelcache.online_rank.predict_rank", wraps=online_rank.predict_rank) as shared:
            with mock.patch("reelcache.online_rank.RankPreparation", side_effect=AssertionError("fake preparation")):
                rank_predictor(
                    profile, layer=0, current_anchors={}, previous_anchors={3: 2},
                    previous_regime="startup", chunk_ordinal=2,
                )
                bound.predict_rank(MatrixKey(0, "K"), current_anchors={})
            self.assertEqual(shared.call_count, 2)
        prediction = predict_rank(
            profile, MatrixKey(0, "K"), chunk_ordinal=5,
            current_anchors={MatrixKey(3, "K"): float("nan")},
            previous_anchors={MatrixKey(3, "K"): 2},
            previous_anchor_chunks={MatrixKey(3, "K"): 1},
        )
        self.assertEqual(prediction.initial_rank, 4)
        self.assertEqual(prediction.sources, prediction.prediction_sources)
        self.assertEqual(prediction.prediction_sources[0]["history_age"], 4)
        self.assertEqual(prediction.prediction_sources[0]["baseline_regime"], "startup")
        with self.assertRaises(ConfigurationError):
            predict_rank(
                profile, MatrixKey(0, "K"), chunk_ordinal=2, current_anchors={},
                previous_anchors={MatrixKey(3, "K"): 2},
                previous_anchor_chunks={MatrixKey(3, "K"): 2},
            )

    def test_per_target_energy_and_margins_remain_inside_one_policy_hash(self):
        profiles = {
            f"{regime}:{target}": profile_builder(predictor_parameters(
                target=target, tau=0.95 if target == "K" else 0.99,
                history_regime=regime, anchor_layers=[], policy_kind="fixed_layerwise",
                rank_margin=0 if target == "K" else 1, rank_buckets=[1, 2, 3, 4],
            ))
            for regime in ("startup", "later") for target in ("K", "V")
        }
        bound = bind_predictor_profiles(policy(layers=4), profiles)
        self.assertEqual(bound.energy_for("K"), 0.95)
        self.assertEqual(bound.energy_for("V"), 0.99)
        self.assertEqual(bound.predict_rank(MatrixKey(0, "K"), current_anchors={}).initial_rank, 2)
        self.assertEqual(bound.predict_rank(MatrixKey(0, "V"), current_anchors={}).initial_rank, 3)
        self.assertEqual(RankPolicy.from_dict(bound.to_dict()).fingerprint, bound.fingerprint)
        with self.assertRaises(TypeError):
            bound.energy_targets["K"] = 0.99
        wrong = dict(profiles)
        wrong["later:K"] = dataclasses.replace(profiles["later:K"], tau=0.99)
        with self.assertRaises(ConfigurationError):
            bind_predictor_profiles(policy(layers=4), wrong)

    def test_actual_guards_use_each_targets_energy_not_the_default_scalar(self):
        bound = policy(solver="exact_oracle", energy_targets={"K": 0.95, "V": 0.99})
        matrix = torch.diag(torch.tensor([8.0, 1.0, 1.0, 0.0]))
        prep, results, _ = complete(bound, {key: matrix for key in bound.keys})
        key_decision = results[MatrixKey(0, "K")].decision
        value_decision = results[MatrixKey(0, "V")].decision
        self.assertEqual(key_decision.rank, 1)
        self.assertEqual(value_decision.rank, 3)
        self.assertEqual(key_decision.energy_target, 0.95)
        self.assertEqual(value_decision.energy_target, 0.99)
        bound.validate_decision(key_decision)
        bound.validate_decision(value_decision)
        with self.assertRaises(IntegrityError):
            bound.validate_decision(dataclasses.replace(
                value_decision, energy_target=0.95, guard_threshold=math.sqrt(1 - 0.95),
            ))
        prep.close()

    def test_detector_statistic_is_explicit_and_oracle_labels_cannot_be_substituted(self):
        profile = profile_builder(predictor_parameters())
        self.assertEqual(profile.rank_statistic, "approximate_spectral_crossing")
        self.assertEqual(policy().anchor_statistic, "approximate_spectral_crossing")
        self.assertEqual(policy(solver="exact_oracle").anchor_statistic, "oracle_rank_upper")
        with self.assertRaises(ConfigurationError):
            profile_builder(predictor_parameters(rank_statistic="guarded_operational_rank"))
        with self.assertRaises(ConfigurationError):
            policy(anchor_statistic="oracle_rank_upper")


def study_provenance(bound, *, group="calibration-family", seed=123):
    ctx = contexts(bound)
    return {
        key: {
            "case_id": "paired-case", "prompt_group": group, "split": "calibration",
            "seed": seed, "cell": "synthetic-cpu-fixture",
            "layer": key.layer, "target": key.target, "boundary": 0,
            "trajectory_id": "exact-candidate-trajectory", "evidence_kind": "synthetic_fixture",
            "n_rows": bound.tokens, "n_channels": bound.model.channels_for(key),
            "n_layers": bound.model.layers, "source_dtype": ctx[key].dtype,
            "source_domain": "normalized_pre_rope" if key.target == "K" else "direct",
            "tensor_context": ctx[key].to_dict(),
            "native_capture": {"request_id": "not-assumed-to-be-case-id", "chunk_id": "chunk-000000"},
        }
        for key in bound.keys
    }


class SourceProvenanceTests(unittest.TestCase):
    def test_case_and_geometry_provenance_are_bound_before_the_single_oracle(self):
        from reelcache.spectra import SpectrumResult

        bound = policy(solver="exact_oracle")
        provenance = study_provenance(bound)
        key = MatrixKey(0, "K")
        prep = RankPreparation(
            bound, chunk_id="chunk-000000", parent_head=None, token_start=0,
            source_provenance=provenance,
        )
        provenance[key]["prompt_group"] = "mutated-after-construction"
        provenance[key]["native_capture"]["request_id"] = "mutated"
        with mock.patch("torch.linalg.eigh", wraps=torch.linalg.eigh) as eigh:
            prepared = prep.prepare_matrix(rank_matrix(2), contexts(bound)[key])
            observation = prepared.decision.to_dict()["metrics"]["oracle_observation"]
            restored = SpectrumResult.from_dict(observation)
        self.assertEqual(eigh.call_count, 1)
        self.assertEqual(restored.source.provenance["prompt_group"], "calibration-family")
        self.assertEqual(restored.source.provenance["case_id"], "paired-case")
        self.assertEqual(
            restored.source.provenance["native_capture"]["request_id"], "not-assumed-to-be-case-id",
        )
        self.assertEqual(prepared.decision.source_provenance["seed"], 123)
        self.assertEqual(prepared.decision.metrics["source_provenance_sha256"],
                         prepared.decision.source_provenance_sha256)
        self.assertTrue(prepared.decision.metrics["study_source_provenance_bound"])
        with self.assertRaises(TypeError):
            prepared.decision.source_provenance["seed"] = 456
        with self.assertRaises(TypeError):
            prepared.decision.source_provenance["native_capture"]["request_id"] = "different"
        prep.close()

    def test_wrong_geometry_partial_study_and_foreign_reserved_fields_fail_before_fit(self):
        bound = policy(solver="exact_oracle")
        key = MatrixKey(0, "K")
        for name, value in (
            ("source_dtype", "bfloat16"), ("n_rows", 8), ("layer", 1),
            ("target", "V"), ("parent_head", "foreign"), ("policy_hash", "9" * 64),
            ("source_fingerprint", "8" * 64), ("source_domain", "post_rope"),
            ("shape", [2, 4]), ("dtype", "bfloat16"), ("key_space", "post_rope"),
        ):
            provenance = study_provenance(bound)
            provenance[key][name] = value
            with mock.patch("torch.linalg.eigh", side_effect=AssertionError("invalid-provenance solve")):
                with self.assertRaises(ConfigurationError):
                    RankPreparation(
                        bound, chunk_id="c", parent_head=None, token_start=0,
                        source_provenance=provenance,
                    )
        provenance = study_provenance(bound)
        del provenance[key]["case_id"]
        with self.assertRaises(ConfigurationError):
            RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0,
                            source_provenance=provenance)
        with self.assertRaises(ConfigurationError):
            RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0,
                            source_provenance={key: {}})

    def test_complete_native_launch_binding_is_frozen_before_exact_decomposition(self):
        bound = policy(solver="exact_oracle")
        provenance = study_provenance(bound)
        for row in provenance.values():
            row.update(
                evidence_kind="native", protocol_id="a" * 64, launch_closure_id="b" * 64,
                cell_id=row["cell"], group_id=row["prompt_group"],
                fixture=False, native_execution=True, request_id="declared-request",
                arm_id="exact-arm", branch_id="request", parent_head=None,
            )
        prep = RankPreparation(
            bound, chunk_id="chunk-000000", parent_head=None, token_start=0,
            source_provenance=provenance,
        )
        key = MatrixKey(0, "K")
        provenance[key]["protocol_id"] = "c" * 64
        with (
            mock.patch("torch.linalg.eigh", wraps=torch.linalg.eigh) as eigh,
            mock.patch("torch.linalg.svd", side_effect=AssertionError("exact reference refit")),
        ):
            prepared = prep.prepare_matrix(rank_matrix(2), contexts(bound)[key])
        observed = prepared.decision.oracle_observation["source"]["provenance"]
        self.assertEqual(eigh.call_count, 1)
        self.assertEqual(observed["protocol_id"], "a" * 64)
        self.assertEqual(observed["launch_closure_id"], "b" * 64)
        self.assertEqual(observed["cell_id"], observed["cell"])
        self.assertEqual(observed["group_id"], observed["prompt_group"])
        self.assertIs(observed["fixture"], False)
        self.assertIs(observed["native_execution"], True)
        self.assertEqual(observed["request_id"], "declared-request")
        self.assertEqual(observed["case_id"], "paired-case")
        prep.close()

    def test_incomplete_native_and_reserved_measurements_fail_before_decomposition(self):
        bound = policy(solver="exact_oracle")
        provenance = study_provenance(bound)
        for row in provenance.values():
            row.update(
                evidence_kind="native", protocol_id="a" * 64, launch_closure_id="b" * 64,
                cell_id=row["cell"], group_id=row["prompt_group"], fixture=False,
                native_execution=True, request_id="request", arm_id="arm",
                branch_id="branch", parent_head=None,
            )
        key = MatrixKey(0, "K")
        for missing in (
            "protocol_id", "launch_closure_id", "cell_id", "group_id", "fixture",
            "native_execution", "request_id", "arm_id", "branch_id", "parent_head",
        ):
            incomplete = {item: dict(row) for item, row in provenance.items()}
            del incomplete[key][missing]
            with mock.patch("torch.linalg.eigh", side_effect=AssertionError("unbound native source solve")):
                with self.assertRaises(ConfigurationError):
                    RankPreparation(
                        bound, chunk_id="c", parent_head=None, token_start=0,
                        source_provenance=incomplete,
                    )
        for name, value in (
            ("protocol_id", "not-a-digest"), ("launch_closure_id", "not-a-digest"),
            ("fixture", True), ("native_execution", False), ("fixture", 0),
            ("cell_id", "other-cell"), ("group_id", "other-group"),
            ("basis_sha256", "a" * 64), ("coefficients_sha256", "b" * 64),
            ("oracle_observation", {}), ("energy_scaled", 1),
        ):
            wrong = {item: dict(row) for item, row in provenance.items()}
            wrong[key][name] = value
            with mock.patch("torch.linalg.eigh", side_effect=AssertionError("reserved source override solve")):
                with self.assertRaises(ConfigurationError):
                    RankPreparation(
                        bound, chunk_id="c", parent_head=None, token_start=0,
                        source_provenance=wrong,
                    )

    def test_one_atomic_chunk_cannot_mix_cases_or_partial_study_bindings(self):
        bound = policy()
        key = MatrixKey(0, "K")
        for changes in ({"case_id": "other"}, {"seed": 456}, {"boundary": 1}):
            provenance = study_provenance(bound)
            provenance[key].update(changes)
            with self.assertRaises(ConfigurationError):
                RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0,
                                source_provenance=provenance)
        provenance = study_provenance(bound)
        provenance[key] = {}
        with self.assertRaises(ConfigurationError):
            RankPreparation(bound, chunk_id="c", parent_head=None, token_start=0,
                            source_provenance=provenance)

    def test_case_labels_do_not_become_prediction_or_solver_rng_features(self):
        bound = policy()
        key = MatrixKey(0, "K")
        matrix = torch.randn(4, 4, generator=torch.Generator().manual_seed(1))
        outcomes = []
        for group, seed in (("family-a", 123), ("family-b", 456)):
            prep = RankPreparation(
                bound, chunk_id="c", parent_head=None, token_start=0,
                source_provenance=study_provenance(bound, group=group, seed=seed),
            )
            outcomes.append(prep.prepare_matrix(matrix, contexts(bound)[key]))
            prep.close()
        left, right = outcomes
        self.assertEqual(left.decision.initial_rank, right.decision.initial_rank)
        self.assertEqual(left.decision.metrics["factorization_seed"], right.decision.metrics["factorization_seed"])
        self.assertTrue(torch.equal(left.factors.basis, right.factors.basis))
        self.assertTrue(torch.equal(left.factors.coefficients, right.factors.coefficients))
        self.assertNotEqual(left.decision.source_provenance_sha256, right.decision.source_provenance_sha256)
        self.assertEqual(left.decision.policy_hash, right.decision.policy_hash)

    def test_completed_oracle_cannot_be_relabelled_after_capture(self):
        bound = policy(solver="exact_oracle")
        key = MatrixKey(0, "K")
        prepared = RankPreparation(
            bound, chunk_id="c", parent_head=None, token_start=0,
            source_provenance=study_provenance(bound),
        ).prepare_matrix(rank_matrix(2), contexts(bound)[key])
        payload = prepared.decision.to_dict()
        payload["source_provenance"]["prompt_group"] = "later-label"
        with self.assertRaises(ConfigurationError):
            RankDecision.from_dict(payload)
        payload = prepared.decision.to_dict()
        payload["metrics"]["oracle_observation"]["source"]["provenance"]["prompt_group"] = "later-label"
        with self.assertRaises(ConfigurationError):
            RankDecision.from_dict(payload)

    def test_same_exact_observation_reaches_the_paired_study_writer_without_another_solve(self):
        from reelcache import rank_analysis
        from reelcache.study import StudyCell

        bound = policy(solver="exact_oracle")
        cell = StudyCell(
            "synthetic-cpu-fixture", "synthetic", 4, 1, 1, 0, 1, 1, 4, 1, 4,
            "float32", "float32", 1,
        )
        stem = Path(".test-artifacts") / f"online-rank-{os.getpid()}-{time.monotonic_ns()}"
        spectrum_path = stem.with_suffix(".spectrum.json")
        observation_path = stem.with_suffix(".observation.json")
        bad_spectrum = stem.with_suffix(".bad-spectrum.json")
        bad_observation = stem.with_suffix(".bad-observation.json")
        owned_paths = (spectrum_path, observation_path, bad_spectrum, bad_observation)
        with mock.patch.object(rank_analysis, "CELLS", {**rank_analysis.CELLS, cell.cell_id: cell}):
            identities = {
                key: rank_analysis.rank_observation_identity(
                    case_id="parent-declared-paired-case", group_id="parent-declared-group",
                    split="calibration", seed=123, cell_id=cell.cell_id,
                    layer=key.layer, target=key.target, boundary=0,
                    trajectory_id="parent-declared-exact-arm", evidence_kind="synthetic_fixture",
                    source_bindings={"source_fingerprint": bound.source_fingerprint},
                )
                for key in bound.keys
            }
            prep = RankPreparation(
                bound, chunk_id="chunk-000000", parent_head=None, token_start=0,
                source_provenance=identities,
            )
            key = MatrixKey(0, "K")
            prepared = prep.prepare_matrix(rank_matrix(2), contexts(bound)[key])
            observation = prepared.decision.to_dict()["metrics"]["oracle_observation"]
            try:
                with (
                    mock.patch("torch.linalg.eigh", side_effect=AssertionError("writer duplicate eigh")),
                    mock.patch("torch.linalg.svd", side_effect=AssertionError("writer duplicate SVD")),
                ):
                    receipt = rank_analysis.write_spectrum_observation(
                        observation, identity=identities[key], spectrum_path=spectrum_path,
                        observation_path=observation_path,
                    )
                    self.assertTrue(spectrum_path.is_file())
                    self.assertTrue(observation_path.is_file())
                    self.assertEqual(receipt["spectrum_sha256"], rank_analysis.file_sha256(spectrum_path))
                    self.assertEqual(receipt["observation"]["case_id"], identities[key]["case_id"])
                    for changed in ({"case_id": "retroactive-case"}, {"n_channels": 8}):
                        with self.assertRaises(rank_analysis.AnalysisError):
                            rank_analysis.write_spectrum_observation(
                                observation, identity={**identities[key], **changed},
                                spectrum_path=bad_spectrum, observation_path=bad_observation,
                            )
                        self.assertFalse(bad_spectrum.exists())
                        self.assertFalse(bad_observation.exists())
            finally:
                prep.close()
                for path in owned_paths:
                    path.unlink(missing_ok=True)


class HistoryDecisionValidationTests(unittest.TestCase):
    def test_replay_validator_handles_causal_staged_lagged_and_exact_seeds(self):
        for mode in ("none", "causal", "lagged", "staged"):
            for solver in ("qr_q2", "exact_oracle"):
                bound = policy(
                    layers=2, solver=solver, anchor_mode=mode,
                    anchors=() if mode == "none" else (MatrixKey(1, "K"),),
                    staging_limit_bytes=256 if mode == "staged" else 0,
                )
                matrices = {key: rank_matrix(2 if key == MatrixKey(1, "K") else 1) for key in bound.keys}
                first, results, state = complete(bound, matrices)
                decisions = {key: result.decision for key, result in results.items()}
                bound.validate_history_decisions(decisions, RankState(bound.identity))
                second, next_results, _ = complete(
                    bound, matrices, prior_state=state, chunk_id="chunk-2", parent_head="head-1",
                )
                bound.validate_history_decisions(
                    {key: result.decision for key, result in next_results.items()}, state,
                )
                first.close()
                second.close()

    def test_counter_and_timing_types_fail_closed_in_serialized_decisions(self):
                prep, results, _ = complete(policy(solver="exact_oracle"))
                decision = results[MatrixKey(0, "K")].decision
                for name, value in (
                    ("svd_count", -1), ("eigh_count", True), ("qr_count", 0.0),
                    ("projection_count", "1"), ("fit_attempts", -1),
                    ("guard_attempts", True), ("decomposition_seconds", -0.1),
                    ("fit_attempts", None), ("qr_seconds", None),
                    ("source_hash_seconds", True), ("projection_guard_seconds", "0.01"),
                    ("workspace_admission_peak_bytes", -1),
                    ("basis_condition_bound", -1),
                ):
                    payload = decision.to_dict()
                    payload["metrics"][name] = value
                    with self.assertRaises(ConfigurationError):
                        RankDecision.from_dict(payload)
                for name in ("eigh_count", "projection_count"):
                    payload = decision.to_dict()
                    del payload["metrics"][name]
                    with self.assertRaises(ConfigurationError):
                        RankDecision.from_dict(payload)
                payload = decision.to_dict()
                payload["metrics"]["oracle_timings"]["eigensolve_seconds"] = -1
                with self.assertRaises(ConfigurationError):
                    RankDecision.from_dict(payload)
                payload = decision.to_dict()
                payload["metrics"]["oracle_solver"]["decomposition_count"] = True
                with self.assertRaises(ConfigurationError):
                    RankDecision.from_dict(payload)
                prep.close()

    def test_replay_validator_rejects_prediction_seed_state_and_availability_tampering(self):
        bound = policy(layers=2, anchors=(MatrixKey(1, "K"),), anchor_mode="causal")
        prep, results, _ = complete(bound)
        original = {key: result.decision for key, result in results.items()}
        key = MatrixKey(0, "K")
        decision = original[key]
        variants = [
            dataclasses.replace(decision, predicted_rank=2.0),
            dataclasses.replace(decision, initial_rank=2),
            dataclasses.replace(decision, parent_head="unbound-parent"),
            dataclasses.replace(decision, chunk_id="other-chunk"),
            dataclasses.replace(decision, prediction_sources=({
                "key": MatrixKey(1, "K").to_dict(), "estimated_rank": 1.0,
                "template_rank": 1.0, "ratio": 1.0, "availability": "current",
                "history_age": 0, "chunk_ordinal": 1,
            },)),
        ]
        for name, value in (
            ("factorization_seed", -1), ("chunk_ordinal", 2),
            ("prior_state_fingerprint", "0" * 64), ("prediction_kind", "oracle-label"),
            ("predictor_profile_fingerprint", "9" * 64),
        ):
            variants.append(dataclasses.replace(decision, metrics={**decision.metrics, name: value}))
        for altered in variants:
            with self.assertRaises(IntegrityError):
                bound.validate_history_decisions({**original, key: altered}, RankState(bound.identity))
        with self.assertRaises(IntegrityError):
            bound.validate_history_decisions({key: decision}, RankState(bound.identity))
        prep.close()

    def test_regime_bound_profile_history_validation_calls_the_same_predictor(self):
        profiles = {
            f"{regime}:{target}": profile_builder(predictor_parameters(
                target=target, history_regime=regime,
                anchor_layers=[0] if regime == "startup" and target == "K" else (
                    [3] if target == "K" else []
                ),
                policy_kind="anchors" if target == "K" else "fixed_layerwise",
                rank_buckets=[1, 2, 3, 4],
            ))
            for regime in ("startup", "later") for target in ("K", "V")
        }
        bound = bind_predictor_profiles(policy(layers=4), profiles)
        matrices = {key: rank_matrix(2) for key in bound.keys}
        first, first_results, state = complete(bound, matrices)
        second, results, _ = complete(
            bound, matrices, prior_state=state, chunk_id="chunk-2", parent_head="head-1",
        )
        bound.validate_history_decisions(
            {key: result.decision for key, result in first_results.items()}, RankState(bound.identity),
        )
        from reelcache import online_rank
        with mock.patch("reelcache.online_rank.predict_rank", wraps=online_rank.predict_rank) as shared:
            bound.validate_history_decisions({key: result.decision for key, result in results.items()}, state)
        self.assertEqual(shared.call_count, len(bound.keys))
        first.close()
        second.close()


class RuntimeSourceProvenanceTests(CoreWorkspaceTest):
    def runtime(self, bound):
        from reelcache.manager import KVRuntime

        runtime = KVRuntime(self.workspace / "store", bound.model, max_resident_bytes=1 << 20)
        self.addCleanup(runtime.close)
        runtime.create_branch("request")
        return runtime

    @staticmethod
    def provenance(bound, *, token_start=0, parent_head=None):
        provenance = study_provenance(bound)
        for key, row in provenance.items():
            row.update(
                evidence_kind="native", protocol_id="a" * 64, launch_closure_id="b" * 64,
                cell_id=row["cell"], group_id=row["prompt_group"], fixture=False,
                native_execution=True, request_id="declared-request", arm_id="exact-arm",
                branch_id="request", parent_head=parent_head,
                boundary=token_start // bound.tokens,
                tensor_context=contexts(bound, token_start)[key].to_dict(),
            )
        return provenance

    def test_core_publishes_the_same_prebound_oracle_once_per_matrix(self):
        bound = policy(solver="exact_oracle")
        runtime = self.runtime(bound)
        provenance = self.provenance(bound)
        values = {key: rank_matrix(2) for key in bound.keys}
        real_eigh = torch.linalg.eigh

        def mutate_caller_and_solve(*args, **kwargs):
            for row in provenance.values():
                row["protocol_id"] = "c" * 64
                row["native_capture"]["request_id"] = "mutated-caller"
            return real_eigh(*args, **kwargs)

        before_rng = torch.get_rng_state().clone()
        with (
            mock.patch("torch.linalg.eigh", side_effect=mutate_caller_and_solve) as eigh,
            mock.patch("torch.linalg.svd", side_effect=AssertionError("duplicate exact factorization")),
            mock.patch("reelcache.codecs.factorize_tensor", side_effect=AssertionError("encoder decomposition")),
        ):
            runtime.seal_chunk(
                "request", "chunk-000000", values, contexts(bound),
                rank_policy=bound, source_provenance=provenance,
            )
        self.assertEqual(eigh.call_count, len(bound.keys))
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
        self.assertEqual(runtime.history_rank_state("request").completed_chunks, 1)
        for key in bound.keys:
            decision = runtime.history_rank_decisions("request", key)[0]
            observed = decision.oracle_observation["source"]["provenance"]
            self.assertEqual(observed["protocol_id"], "a" * 64)
            self.assertEqual(observed["launch_closure_id"], "b" * 64)
            self.assertEqual(observed["case_id"], "paired-case")
            self.assertEqual(observed["native_capture"]["request_id"], "not-assumed-to-be-case-id")
            self.assertEqual(observed["target"], key.target)
            self.assertEqual(observed["layer"], key.layer)
            self.assertIs(observed["fixture"], False)
            self.assertIs(observed["native_execution"], True)
            self.assertEqual(decision.metrics["eigh_count"], 1)
            self.assertEqual(decision.metrics["svd_count"], 0)
        self.assertEqual(runtime.store.stats()["leases"], 0)

    def test_malformed_binding_does_not_change_existing_head_state_rng_or_leases(self):
        bound = policy(solver="exact_oracle")
        runtime = self.runtime(bound)
        values = {key: rank_matrix(1) for key in bound.keys}
        runtime.seal_chunk("request", "chunk-000000", values, contexts(bound), rank_policy=bound)
        before_head = runtime.branch_info("request")
        before_state = runtime.history_rank_state("request").to_dict()
        before_rng = torch.get_rng_state().clone()
        key = MatrixKey(0, "K")
        changes = (
            ("protocol_id", "bad-digest"), ("launch_closure_id", None),
            ("group_id", "other-group"), ("native_execution", False),
            ("basis_sha256", "d" * 64), ("layer", 9),
            ("branch_id", "other-branch"), ("parent_head", "other-head"),
        )
        for name, value in changes:
            provenance = self.provenance(bound, token_start=4, parent_head=before_head["head_id"])
            provenance[key][name] = value
            with (
                mock.patch("torch.linalg.eigh", side_effect=AssertionError("unbound eigensolve")),
                mock.patch("torch.linalg.svd", side_effect=AssertionError("unbound decomposition")),
            ):
                with self.assertRaises(ConfigurationError):
                    runtime.seal_chunk(
                        "request", "chunk-000001", values, contexts(bound, 4),
                        rank_policy=bound, source_provenance=provenance,
                    )
            self.assertEqual(runtime.branch_info("request"), before_head)
            self.assertEqual(runtime.history_rank_state("request").to_dict(), before_state)
            self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
            self.assertEqual(runtime.store.stats()["leases"], 0)
        with self.assertRaises(ConfigurationError):
            runtime.seal_chunk(
                "request", "chunk-000001", values, contexts(bound, 4), rank_policy=bound,
                source_provenance={key: {}},
            )
        self.assertEqual(runtime.branch_info("request"), before_head)

    def test_static_path_cannot_silently_discard_adaptive_source_provenance(self):
        bound = policy()
        runtime = self.runtime(bound)
        before = runtime.branch_info("request")
        with self.assertRaises(ConfigurationError):
            runtime.seal_chunk(
                "request", "static", {key: rank_matrix(1) for key in bound.keys},
                contexts(bound), source_provenance={key: {} for key in bound.keys},
            )
        self.assertEqual(runtime.branch_info("request"), before)

    def test_projection_backend_failure_does_not_publish_or_leak_a_lease(self):
        bound = policy()
        runtime = self.runtime(bound)
        before = runtime.branch_info("request")
        before_rng = torch.get_rng_state().clone()
        with mock.patch("torch.linalg.cholesky_ex", side_effect=RuntimeError("native backend fault fixture")):
            with self.assertRaises(RankAbstention) as caught:
                runtime.seal_chunk(
                    "request", "failed", {key: torch.eye(4) for key in bound.keys},
                    contexts(bound), rank_policy=bound,
                )
        self.assertEqual(caught.exception.decision.metrics["svd_count"], 2)
        self.assertEqual(caught.exception.decision.metrics["qr_count"], 10)
        self.assertEqual(caught.exception.decision.metrics["projection_count"], 1)
        self.assertEqual(runtime.branch_info("request"), before)
        self.assertIsNone(runtime.history_rank_state("request"))
        self.assertFalse(runtime.history_rank_decisions("request", MatrixKey(0, "K")))
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
        self.assertEqual(runtime.store.stats()["leases"], 0)


if __name__ == "__main__":
    unittest.main()
