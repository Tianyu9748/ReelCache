import copy
import dataclasses
import hashlib
import json
from unittest.mock import patch

import torch

from reelcache.codecs import (
    HEVCProvider, decode_tensor, encode_tensor, encoded_nbytes, factor_binding,
    prepared_validation_workspace_bytes, validate_prepared_packet,
)
from reelcache.errors import ConfigurationError, IntegrityError
from reelcache.factorization import Factors, fit_basis, project_onto_basis, rank_projection_workspace_bytes
from reelcache.online_rank import RankPreparation
from reelcache.types import CodecSpec, EncodedTensor, MatrixKey, TensorContext
from tests.core_support import CoreWorkspaceTest, model
from tests.test_adaptive_manager import rank_policy
from tests.test_codecs import context, raw_bytes


class PreparedCodecTests(CoreWorkspaceTest):
    def setUp(self):
        super().setUp()
        self.context = context()
        self.value = torch.randn(24, 4, generator=torch.Generator().manual_seed(143))
        self.spec = CodecSpec("svd", 3)
        self.factors = fit_basis(self.value, self.context, 3)
        self.binding = factor_binding(
            self.value, self.context, self.spec, self.factors, "a" * 64,
        )

    def encode(self, **kwargs):
        return encode_tensor(
            kwargs.get("value", self.value), kwargs.get("context", self.context),
            kwargs.get("spec", self.spec),
            prepared_factors=kwargs.get("factors", self.factors),
            prepared_binding=kwargs.get("binding", self.binding),
        )

    def test_prepared_projection_is_packed_once_without_another_svd(self):
        with patch("reelcache.codecs.factorize_tensor", side_effect=AssertionError("duplicate fit")), \
                patch("torch.linalg.svd", side_effect=AssertionError("duplicate SVD")):
            packet = self.encode()
        self.assertEqual(packet.basis, raw_bytes(self.factors.basis))
        self.assertEqual(packet.payloads["coefficients.f32"], raw_bytes(self.factors.coefficients))
        self.assertTrue(torch.equal(
            decode_tensor(packet), self.factors.coefficients @ self.factors.basis.float(),
        ))
        self.assertEqual(packet.metadata["schema"], "reelcache.tensor.v1")
        self.assertEqual(packet.metrics["decomposition_count"], 1)
        self.assertEqual(packet.metrics["prepared_validation"]["decomposition_count"], 0)
        self.assertEqual(packet.metrics["prepared_validation"]["projection_count"], 1)
        validate_prepared_packet(
            packet, self.context, self.spec, self.factors, self.binding,
        )

    def test_default_packet_bytes_and_accounting_are_unchanged(self):
        static = encode_tensor(self.value, self.context, self.spec)
        prepared = self.encode()
        self.assertEqual(static.metadata, prepared.metadata)
        self.assertEqual(static.payloads, prepared.payloads)
        self.assertEqual(static.basis, prepared.basis)
        expected = len(json.dumps(
            prepared.metadata, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()) + sum(map(len, prepared.payloads.values())) + len(prepared.basis)
        self.assertEqual(encoded_nbytes(prepared), expected)
        self.assertEqual(prepared.metrics["representation_bytes"], expected)
        self.assertGreater(encoded_nbytes(prepared, include_metrics=True), expected)

    def tiled_preparation(self, *, solver="qr_q2", tile_rows=2):
        identity = model(layers=1, channels=32)
        keys = (MatrixKey(0, "K"), MatrixKey(0, "V"))
        policy = rank_policy(
            identity, geometry=(1, 1, 129), tile_rows=tile_rows, solver=solver,
            template_ranks={key: 32 for key in keys},
            allowed_ranks={key: (32,) for key in keys},
            source_limit_bytes=1 << 20, staging_limit_bytes=1 << 20,
        )
        ctx = TensorContext(identity, keys[0], 0, 1, 1, 129, "pre_rope")
        value = torch.randn(129, 32, generator=torch.Generator().manual_seed(101))
        preparation = RankPreparation(policy, chunk_id="c0", parent_head=None, token_start=0)
        self.addCleanup(preparation.close)
        return value, ctx, policy, preparation.prepare_matrix(value, ctx)

    def test_nondefault_guard_tiling_replays_without_refitting_or_changing_factors(self):
        value, ctx, policy, prepared = self.tiled_preparation()
        binding = factor_binding(value, ctx, prepared.spec, prepared.factors, policy.identity)
        with patch("torch.linalg.svd", side_effect=AssertionError("encoder SVD")), \
                patch("torch.linalg.eigh", side_effect=AssertionError("encoder eigh")):
            packet = encode_tensor(
                value, ctx, prepared.spec,
                prepared_factors=prepared.factors, prepared_binding=binding,
            )
        self.assertEqual(
            packet.metrics["prepared_validation"]["projection_relative_residual"],
            prepared.decision.guard_relative_residual,
        )
        self.assertEqual(packet.basis, raw_bytes(prepared.factors.basis))
        self.assertEqual(packet.payloads["coefficients.f32"], raw_bytes(prepared.factors.coefficients))
        self.assertEqual(binding["guard_tile_rows"], 2)
        self.assertEqual(packet.metrics["prepared_validation"]["guard_tile_rows"], 2)
        self.assertEqual(
            packet.metrics["prepared_validation"]["workspace_estimate_bytes"],
            prepared_validation_workspace_bytes(ctx, prepared.spec, tile_rows=2),
        )

    def test_exact_and_qr_guards_match_for_small_and_oversized_tiles(self):
        for solver in ("qr_q2", "exact_oracle"):
            for tile_rows in (1, 2, 127, 2048):
                with self.subTest(solver=solver, tile_rows=tile_rows):
                    value, ctx, policy, prepared = self.tiled_preparation(
                        solver=solver, tile_rows=tile_rows,
                    )
                    binding = factor_binding(value, ctx, prepared.spec, prepared.factors, policy.identity)
                    with patch("torch.linalg.svd", side_effect=AssertionError("encoder SVD")), \
                            patch("torch.linalg.eigh", side_effect=AssertionError("encoder eigh")):
                        packet = encode_tensor(
                            value, ctx, prepared.spec,
                            prepared_factors=prepared.factors, prepared_binding=binding,
                        )
                    self.assertEqual(
                        packet.metrics["prepared_validation"]["projection_relative_residual"],
                        prepared.decision.guard_relative_residual,
                    )
                    self.assertEqual(packet.metrics["prepared_validation"]["guard_tile_rows"], tile_rows)

    def test_guard_tiling_and_accepted_factors_cannot_be_substituted(self):
        value, ctx, policy, prepared = self.tiled_preparation()
        binding = factor_binding(value, ctx, prepared.spec, prepared.factors, policy.identity)
        for invalid in (0, -1, True, 2.0, None):
            with self.subTest(tile_rows=invalid), self.assertRaises(ConfigurationError):
                factor_binding(
                    value, ctx, prepared.spec,
                    dataclasses.replace(
                        prepared.factors, metrics={**prepared.factors.metrics, "tile_rows": invalid},
                    ),
                )
        legacy = {name: item for name, item in binding.items() if name != "guard_tile_rows"}
        legacy["schema"] = "reelcache.prepared_factors.v1"
        packet = encode_tensor(
            value, ctx, prepared.spec, prepared_factors=prepared.factors, prepared_binding=legacy,
        )
        self.assertEqual(packet.metrics["prepared_validation"]["guard_tile_rows"], 2)
        validate_prepared_packet(packet, ctx, prepared.spec, prepared.factors, legacy)
        for changed in (
            {**binding, "guard_tile_rows": 1024},
            {name: item for name, item in binding.items() if name != "guard_tile_rows"},
        ):
            with self.assertRaises(IntegrityError):
                encode_tensor(
                    value, ctx, prepared.spec,
                    prepared_factors=prepared.factors, prepared_binding=changed,
                )
        changed_metrics = {**prepared.factors.metrics, "tile_rows": 1024}
        for factors in (
            dataclasses.replace(prepared.factors, metrics=changed_metrics),
            dataclasses.replace(prepared.factors, coefficients=prepared.factors.coefficients + 0.001),
            dataclasses.replace(prepared.factors, basis=-prepared.factors.basis),
        ):
            rebound = factor_binding(value, ctx, prepared.spec, factors, policy.identity)
            with self.assertRaises(IntegrityError):
                encode_tensor(
                    value, ctx, prepared.spec, prepared_factors=factors, prepared_binding=rebound,
                )

    def test_legacy_default_guard_binding_remains_supported(self):
        legacy = {name: item for name, item in self.binding.items() if name != "guard_tile_rows"}
        legacy["schema"] = "reelcache.prepared_factors.v1"
        packet = self.encode(binding=legacy)
        self.assertEqual(packet.metrics["prepared_validation"]["guard_tile_rows"], 1024)
        validate_prepared_packet(packet, self.context, self.spec, self.factors, legacy)
        self.assertEqual(packet.metadata, self.encode().metadata)
        self.assertEqual(packet.payloads, self.encode().payloads)

    def test_verification_workspace_tracks_small_and_large_guard_tiles(self):
        ctx = dataclasses.replace(self.context, frames=1, height=1, width=2049)
        default = prepared_validation_workspace_bytes(ctx, self.spec)
        small = prepared_validation_workspace_bytes(ctx, self.spec, tile_rows=2)
        large = prepared_validation_workspace_bytes(ctx, self.spec, tile_rows=2049)
        self.assertEqual(default - small, (1024 - 2) * ctx.channels * 64)
        self.assertEqual(large - default, (2049 - 1024) * ctx.channels * 64)
        for invalid in (0, -1, True, 2.0, None):
            with self.subTest(tile_rows=invalid), self.assertRaises(ConfigurationError):
                prepared_validation_workspace_bytes(ctx, self.spec, tile_rows=invalid)

    def test_verification_workspace_covers_wide_matrix_conditioning(self):
        identity = model(layers=1, channels=4096)
        ctx = TensorContext(identity, MatrixKey(0, "K"), 0, 1, 1, 129, "pre_rope")
        spec = CodecSpec("svd", 129)
        estimate = prepared_validation_workspace_bytes(ctx, spec, tile_rows=2)
        shared = rank_projection_workspace_bytes(ctx.shape, spec.rank, tile_rows=2)
        self.assertGreaterEqual(estimate, shared)

    def test_source_context_spec_and_factor_diagnostics_are_bound(self):
        altered_metrics = {**self.factors.metrics, "fit_seconds": 1000.0}
        for kwargs in (
            {"value": self.value + 1},
            {"context": dataclasses.replace(self.context, token_start=0)},
            {"spec": dataclasses.replace(self.spec, normalization="global")},
            {"factors": dataclasses.replace(self.factors, metrics=altered_metrics)},
        ):
            with self.subTest(kwargs=tuple(kwargs)), self.assertRaises(IntegrityError):
                self.encode(**kwargs)
        self.assertEqual(
            self.binding["source_sha256"], hashlib.sha256(raw_bytes(self.value)).hexdigest(),
        )

    def test_dtype_rank_geometry_finiteness_and_condition_are_not_trusted(self):
        cases = (
            dataclasses.replace(self.factors, basis=self.factors.basis.float()),
            dataclasses.replace(self.factors, coefficients=self.factors.coefficients.half()),
            dataclasses.replace(self.factors, rank=2),
            dataclasses.replace(self.factors, rank=True),
            dataclasses.replace(self.factors, basis=self.factors.basis[:, :2]),
            dataclasses.replace(self.factors, coefficients=self.factors.coefficients[:12]),
            dataclasses.replace(self.factors, basis=torch.full((3, 4), float("nan")).half()),
        )
        for factors in cases:
            with self.subTest(factors=(factors.rank, factors.basis.dtype)), \
                    self.assertRaises(ConfigurationError):
                self.encode(factors=factors)
        singular = dataclasses.replace(
            self.factors, basis=torch.ones(3, 4, dtype=torch.float16),
        )
        binding = factor_binding(self.value, self.context, self.spec, singular)
        with self.assertRaisesRegex(ConfigurationError, "ill-conditioned"):
            self.encode(factors=singular, binding=binding)
        with self.assertRaises(ConfigurationError):
            self.encode(spec=CodecSpec("svd", 25))

    def test_rebinding_forged_projection_or_metrics_does_not_bypass_validation(self):
        for factors in (
            dataclasses.replace(self.factors, coefficients=self.factors.coefficients + 1),
            dataclasses.replace(
                self.factors,
                metrics={**self.factors.metrics, "projection_relative_residual": 0.0},
            ),
            dataclasses.replace(self.factors, metrics={**self.factors.metrics, "zero_input": True}),
            dataclasses.replace(self.factors, metrics={**self.factors.metrics, "basis_condition_bound": 9.0}),
        ):
            binding = factor_binding(self.value, self.context, self.spec, factors)
            with self.assertRaises(IntegrityError):
                self.encode(factors=factors, binding=binding)

    def test_full_canonical_basis_is_charged_even_for_a_smaller_rank_view(self):
        fitted = fit_basis(self.value, self.context, 4)
        projection = project_onto_basis(self.value, fitted.basis, rank=2)
        factors = Factors(
            fitted.basis, projection.coefficients, 2,
            {
                **fitted.metrics, "projection_relative_residual": projection.relative_residual,
                "zero_input": projection.zero_input,
                "basis_condition_bound": projection.condition_bound,
            },
        )
        spec = CodecSpec("svd", 2)
        binding = factor_binding(self.value, self.context, spec, factors)
        packet = self.encode(spec=spec, factors=factors, binding=binding)
        self.assertEqual(packet.metadata["basis"]["shape"], [4, 4])
        self.assertEqual(packet.metadata["basis"]["view_rank"], 2)
        self.assertEqual(packet.metrics["basis_bytes"], 32)
        self.assertTrue(torch.equal(decode_tensor(packet), factors.coefficients @ factors.basis[:2].float()))

    def test_raw_and_post_rope_never_become_an_implicit_fallback(self):
        for kwargs in (
            {"spec": CodecSpec()},
            {"context": dataclasses.replace(self.context, key_space="post_rope")},
        ):
            with self.assertRaises(ConfigurationError):
                self.encode(**kwargs)
        with self.assertRaises(ConfigurationError):
            encode_tensor(
                self.value, self.context, self.spec,
                prepared_factors=self.factors, prepared_binding=self.binding,
                reuse_threshold=0.1,
            )
        for kwargs in (
            {"prepared_factors": self.factors}, {"prepared_binding": self.binding},
        ):
            with self.assertRaises(ConfigurationError):
                encode_tensor(self.value, self.context, self.spec, **kwargs)

    def test_provider_cannot_drop_or_substitute_the_prepared_binding(self):
        good = self.encode()
        for metrics in (
            None, {}, {**good.metrics, "prepared_binding": {}},
            {**good.metrics, "source_bytes": 1}, {**good.metrics, "representation_bytes": 1},
        ):
            packet = EncodedTensor(good.metadata, good.payloads, good.basis, metrics)
            with self.assertRaises(IntegrityError):
                validate_prepared_packet(
                    packet, self.context, self.spec, self.factors, self.binding,
                )
        bad_metadata = copy.deepcopy(good.metadata)
        bad_metadata["context"]["token_start"] += 24
        packet = EncodedTensor(bad_metadata, good.payloads, good.basis, good.metrics)
        with self.assertRaises(IntegrityError):
            validate_prepared_packet(
                packet, self.context, self.spec, self.factors, self.binding,
            )

    def test_real_hevc_uses_the_same_factors_and_independent_validation_decode(self):
        spec = CodecSpec("svd_hevc", 3)
        binding = factor_binding(self.value, self.context, spec, self.factors)
        with patch("reelcache.codecs.factorize_tensor", side_effect=AssertionError("duplicate fit")):
            packet = HEVCProvider().encode_tensor(
                self.value, self.context, spec,
                prepared_factors=self.factors, prepared_binding=binding,
            )
        validation = validate_prepared_packet(
            packet, self.context, spec, self.factors, binding, source=self.value,
        )
        restored = decode_tensor(EncodedTensor(packet.metadata, packet.payloads, packet.basis))
        self.assertEqual(restored.shape, self.value.shape)
        self.assertEqual(packet.metadata["video"]["pix_fmt"], "yuv444p")
        self.assertEqual(packet.metadata["codec"]["qp"], 8)
        self.assertTrue(packet.metrics["hevc_software_only"])
        self.assertGreaterEqual(packet.metrics["hevc_validation_decode_seconds"], 0)
        self.assertEqual(packet.basis, raw_bytes(self.factors.basis))
        self.assertEqual(validation["hevc_validation_decode_count"], 1)
        self.assertGreaterEqual(validation["post_hevc_relative_residual"], 0)
