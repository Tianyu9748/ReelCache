"""wl-base (workload-v2 W1c): the SVD-only and HEVC-only baselines, the uniform-rank ablation and ``reuse=False``.

* configuration: the arms' fields, rank rounding (C/6, C/4 to the nearest 32), clamping, illegal combinations, and
  unchanged content identities and method fingerprints of every existing configuration (golden values computed with
  the F6 tree ``w0_f6`` = 5be577ce83082926);
* the planner's fixed-rank and identity-basis policies; the identity finalization fast path equals the generic one;
* ``reelcache.factor_pack.v1`` (FP16 factors without HEVC): finalize, pack, store, validate, prepare, restore;
* every arm through the SGLang bridge on CPU with the toy causal model of ``tests.test_sglang_bridge`` (real
  FFmpeg/libx265 at QP8 for the HEVC arms): per-matrix bytes and relative Frobenius error vs RAW recorded, restores
  from the raw tier, prepared packs and cold packs, continuation; and an in-process A/B showing the new fields are
  inert at their defaults.
"""

import contextlib
import dataclasses
import json
import math
import unittest
from unittest import mock

import numpy as np
import torch

import tests.test_sglang_bridge as tsb
from reelcache import factor_packs as fp
from reelcache import rank_method as rm
from reelcache.codecs import FACTOR_PACK_SCHEMA, PACK_SCHEMA
from reelcache.deferred_reads import upload  # noqa: F401 - the finalize path's upload helper (imported by node_packets)
from reelcache.errors import ConfigurationError, IntegrityError, StateError
from reelcache.integrations.sglang import config as sglang_config
from reelcache.integrations.sglang.config import ReelCacheRuntimeConfig, fixed_rank
from reelcache.integrations.sglang.lineage_archive import factor_warmup_shapes, fixed_rank_fields
from reelcache.node_packets import (FinalizeItem, assemble_canvas, detach_pack_bases, encode_node_pack, finalize_tiles,
                                 pack_codec)
from reelcache.predictor_registry import RankPredictorRegistry
from reelcache.types import (CodecSpec, EnergyTargets, EncodedTensor, MatrixKey, ModelIdentity, TensorContext,
                          canonical_digest, energy_targets)

#: F6 (w0_f6, 5be577ce83082926) content identities of existing bridge configurations (canonical_digest of
#: ``content_dict()``) and RankMethodConfig fingerprints, computed with the base tree on 2026-10-19; values
#: under this release's names: the digests of the same inputs with the renamed identifiers.
F6_CONTENT = {
    "default": ({}, "6d971ed3d7fc0c0c720ee9add3ca2c15c65563cc22ce5c3c94a5cd1d2ae3f6ea"),
    "raw": ({"representation": "raw"}, "4dd6151ac9823d056a5160ec70bee7c35160f3a7baecb340320158dd23b02386"),
    "reuse_false": ({"reuse": False}, "f9cc1d37c3cc3ebfff6d375170658ed91b9cd3b9a8f30334a28de6ee54521e12"),
    "cumulative": ({"refit_rule": "cumulative"}, "144646dc5c0f7095e5cf15fc1d762300677bbea5141c919a5fe4ff88d0cf355d"),
    "energy": ({"energy_key": 0.97, "energy_value": 0.9},
               "751aa477df8baf1296c806e715be0fe6ee695b817e397b1adeff304ebe13ec8f"),
    "reuse_false_norefit": ({"reuse": False, "refit": False},
                            "3843a9c23aeb3e2c4e48198dab904395e5c14ed73e8671545abd5035a75ec597"),
    "qp": ({"hevc_qp": 8, "hevc_pix_fmt": "yuv420p"}, "7c2d75eebf449a44626bf016515bfb13373c107cef66c5d77e61fbcbd0f0ecc0"),
}
F6_METHODS = {
    "svd_hevc": ({"kind": "svd_hevc"}, {}, "a249ef6749997742d0dc0b996907c3a751801547fa3305e456142561efa55abf"),
    "svd_hevc_noreuse": ({"kind": "svd_hevc"}, {"reuse": False},
                         "75344838f975a12fd4a013fad709113f592217ba86cedfbeb29ff91ee9764383"),
    "svd": ({"kind": "svd"}, {}, "86af23d7627269f70a991ed5a3bf7b37862c03bb7ea7a940b635eed9c16b4fe7"),
    "svd_hevc_lag3_fp64": ({"kind": "svd_hevc"}, {"rate_lag": 3, "matmul_precision": "fp32_verified_fp64"},
                           "3a0e407ba01056268c8c9531770fe645a321da0ecd1bb9d28d4fccc5a6be746a"),
}
TOY_GRID, TOY_TPF = (6, 8), 48  # 96 tokens per toy chunk >= 64 channels (the identity basis needs N >= C)


def config(**values):
    return ReelCacheRuntimeConfig(store_root="/tmp/wl-base-config", **values)


class ConfigTests(unittest.TestCase):
    def test_svd_ranks_round_to_the_nearest_multiple_of_32(self):
        self.assertEqual([fixed_rank(1536, 1 / 6), fixed_rank(1536, 1 / 4)], [256, 384])
        self.assertEqual([fixed_rank(3072, 1 / 6), fixed_rank(3072, 1 / 4)], [512, 768])
        self.assertEqual(fixed_rank(1536, 0.1), 160)  # 153.6 -> 160
        self.assertEqual(fixed_rank(1536, 0.09), 128)  # 138.24 -> 128
        self.assertEqual(fixed_rank(1600, 0.01), 32)  # 16 = 0.5 bucket: ties go up; never below one bucket
        self.assertEqual(fixed_rank(1536, 1.0), 1536)
        self.assertEqual(fixed_rank(20, 0.5), 20)  # fewer channels than a bucket: clamped to C
        svd = config(representation="svd", reuse=False)
        self.assertEqual((svd.svd_rank_fraction_k, svd.svd_rank_fraction_v), (1 / 6, 1 / 4))
        self.assertEqual(svd.fixed_ranks({"K": 1536, "V": 1536}), {"K": 256, "V": 384})
        self.assertEqual(svd.fixed_ranks({"K": 3072, "V": 3072}), {"K": 512, "V": 768})
        self.assertEqual(sglang_config.SVD_RANK_FRACTIONS, {"K": 1 / 6, "V": 1 / 4})
        explicit = config(representation="svd", reuse=False, svd_rank_fraction_k=0.5, svd_rank_fraction_v=1)
        self.assertEqual(explicit.fixed_ranks({"K": 1536, "V": 1536}), {"K": 768, "V": 1536})
        self.assertIsInstance(explicit.svd_rank_fraction_v, float)
        self.assertIsNone(config().fixed_ranks({"K": 1536, "V": 1536}))
        self.assertIsNone(config(representation="hevc", reuse=False).fixed_ranks({"K": 1536, "V": 1536}))

    def test_uniform_rank_is_one_bucket_for_every_matrix_clamped_to_the_channels(self):
        for rank, channels, expected in ((512, 1536, 512), (1024, 3072, 1024), (288, 1536, 288), (736, 3072, 736),
                                         (2048, 1536, 1536), (4096, 3072, 3072)):
            uniform = config(rank_policy="fixed_uniform", uniform_rank=rank, reuse=False)
            self.assertEqual(uniform.fixed_ranks({"K": channels, "V": channels}), {"K": expected, "V": expected})
        model = ModelIdentity("toy", "fp", 2, 64, "tokens_channels", "bfloat16")
        uniform = config(rank_policy="fixed_uniform", uniform_rank=512, reuse=False)
        self.assertEqual(fixed_rank_fields(uniform, model), {"fixed_rank_k": 64, "fixed_rank_v": 64})
        self.assertEqual(fixed_rank_fields(config(representation="hevc", reuse=False), model),
                         {"fixed_basis": "identity"})
        self.assertEqual(fixed_rank_fields(config(representation="svd", reuse=False), model),
                         {"fixed_rank_k": 32, "fixed_rank_v": 32})
        self.assertEqual(fixed_rank_fields(config(), model), {})
        self.assertEqual(fixed_rank_fields(config(reuse=False), model), {})
        self.assertEqual(fixed_rank_fields(object(), model), {})  # a config object without the new fields

    def test_illegal_combinations_raise(self):
        bad = [
            {"representation": "svd"},  # basis sharing on (reuse defaults True)
            {"representation": "hevc"},
            {"rank_policy": "fixed_uniform", "uniform_rank": 512},
            {"rank_policy": "fixed_uniform", "reuse": False},  # no rank
            {"rank_policy": "fixed_uniform", "uniform_rank": 500, "reuse": False},  # not a bucket
            {"rank_policy": "fixed_uniform", "uniform_rank": 0, "reuse": False},
            {"rank_policy": "fixed_uniform", "uniform_rank": 16, "reuse": False},
            {"rank_policy": "fixed_uniform", "uniform_rank": 512.0, "reuse": False},
            {"rank_policy": "fixed_uniform", "uniform_rank": True, "reuse": False},
            {"rank_policy": "fixed_uniform", "uniform_rank": 65568, "reuse": False},
            {"rank_policy": "fixed_uniform", "uniform_rank": 512, "reuse": False, "representation": "svd"},
            {"rank_policy": "fixed_uniform", "uniform_rank": 512, "reuse": False, "representation": "hevc"},
            {"rank_policy": "fixed_uniform", "uniform_rank": 512, "reuse": False, "representation": "raw"},
            {"uniform_rank": 512},  # without the policy
            {"rank_policy": "uniform"},
            {"svd_rank_fraction_k": 0.5},  # fractions on svd_hevc
            {"representation": "hevc", "reuse": False, "svd_rank_fraction_v": 0.25},
            {"representation": "raw", "svd_rank_fraction_k": 0.25},
            {"representation": "svd", "reuse": False, "svd_rank_fraction_k": 0.0},
            {"representation": "svd", "reuse": False, "svd_rank_fraction_k": 1.5},
            {"representation": "svd", "reuse": False, "svd_rank_fraction_v": float("nan")},
            {"representation": "svd", "reuse": False, "svd_rank_fraction_v": True},
            {"representation": "svd", "reuse": False, "svd_rank_fraction_v": "0.25"},
            {"representation": "svd", "reuse": False, "hevc_inter_prediction": True},
            {"representation": "hevc", "reuse": False, "hevc_inter_prediction": True},
            {"representation": "latent"},
        ]
        for values in bad:
            with self.subTest(values=values), self.assertRaises(ConfigurationError):
                config(**values)
        for values in ({"representation": "svd", "reuse": False}, {"representation": "hevc", "reuse": False},
                       {"rank_policy": "fixed_uniform", "uniform_rank": 288, "reuse": False},
                       {"reuse": False}, {"reuse": False, "refit": False}):
            config(**values)  # legal

    def test_json_round_trip_and_content_fields(self):
        for values in ({"representation": "svd", "reuse": False},
                       {"representation": "hevc", "reuse": False, "hevc_qp": 8},
                       {"rank_policy": "fixed_uniform", "uniform_rank": 736, "reuse": False}):
            first = config(**values)
            again = ReelCacheRuntimeConfig.from_dict(json.loads(json.dumps(first.to_dict())))
            self.assertEqual(again, first)
            self.assertEqual(again.content_dict(), first.content_dict())
        svd = config(representation="svd", reuse=False).content_dict()
        self.assertEqual((svd["svd_rank_fraction_k"], svd["svd_rank_fraction_v"]), (1 / 6, 1 / 4))
        self.assertNotIn("rank_policy", svd)  # adaptive (the default) is omitted
        uniform = config(rank_policy="fixed_uniform", uniform_rank=288, reuse=False).content_dict()
        self.assertEqual((uniform["rank_policy"], uniform["uniform_rank"]), ("fixed_uniform", 288))
        self.assertNotIn("svd_rank_fraction_k", uniform)
        # every arm has its own session identity
        identities = {canonical_digest(config(**values).content_dict()) for values in (
            {}, {"reuse": False}, {"representation": "svd", "reuse": False},
            {"representation": "svd", "reuse": False, "svd_rank_fraction_k": 0.25},
            {"representation": "hevc", "reuse": False},
            {"rank_policy": "fixed_uniform", "uniform_rank": 288, "reuse": False},
            {"rank_policy": "fixed_uniform", "uniform_rank": 320, "reuse": False})}
        self.assertEqual(len(identities), 7)

    def test_existing_configurations_keep_their_f6_identities(self):
        """A/B against F6: content identities of existing configurations and the explicit defaults of the new fields
        equal the base tree's (golden values from w0_f6)."""
        for name, (values, digest) in F6_CONTENT.items():
            with self.subTest(name=name):
                self.assertEqual(canonical_digest(config(**values).content_dict()), digest)
                explicit = config(**values, rank_policy="adaptive", uniform_rank=None, svd_rank_fraction_k=None,
                                  svd_rank_fraction_v=None)
                self.assertEqual(canonical_digest(explicit.content_dict()), digest)
                for field in ("rank_policy", "uniform_rank", "svd_rank_fraction_k", "svd_rank_fraction_v"):
                    self.assertNotIn(field, explicit.content_dict())
        default = config()
        self.assertEqual(default.codec_template(), CodecSpec(kind="svd_hevc", rank=1))
        self.assertEqual(config(representation="hevc", reuse=False).codec_template(), default.codec_template())
        self.assertEqual(config(representation="svd", reuse=False).codec_template().kind, "svd")

    def test_rank_method_fixed_fields_and_unchanged_fingerprints(self):
        targets = energy_targets("causvid")
        for name, (codec, extra, digest) in F6_METHODS.items():
            with self.subTest(name=name):
                method = rm.RankMethodConfig(targets, CodecSpec(rank=1, **codec), **extra)
                self.assertEqual(method.fingerprint, digest)
                self.assertNotIn("fixed_ranks", method.to_dict())
                self.assertNotIn("fixed_basis", method.to_dict())
                self.assertIsNone(method.fixed_policy)
        hevc = CodecSpec(kind="svd_hevc", rank=1)
        fixed = rm.RankMethodConfig(targets, hevc, reuse=False, fixed_rank_k=288, fixed_rank_v=288)
        self.assertEqual(fixed.to_dict()["fixed_ranks"], {"K": 288, "V": 288})
        self.assertEqual((fixed.fixed_policy, fixed.fixed_rank_for("K"), fixed.fixed_rank_for("V")),
                         ("fixed_rank", 288, 288))
        identity = rm.RankMethodConfig(targets, hevc, reuse=False, fixed_basis="identity")
        self.assertEqual((identity.to_dict()["fixed_basis"], identity.fixed_policy), ("identity", "identity"))
        self.assertEqual(len({fixed.fingerprint, identity.fingerprint,
                              rm.RankMethodConfig(targets, hevc, reuse=False).fingerprint}), 3)
        for kwargs in ({"fixed_rank_k": 288}, {"fixed_rank_k": 0, "fixed_rank_v": 1},
                       {"fixed_rank_k": 32, "fixed_rank_v": 32.0}, {"fixed_basis": "eye"},
                       {"fixed_basis": "identity", "fixed_rank_k": 32, "fixed_rank_v": 32},
                       {"fixed_rank_k": 32, "fixed_rank_v": 32, "reuse": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigurationError):
                rm.RankMethodConfig(targets, hevc, **{"reuse": False, **kwargs})
        with self.assertRaises(ConfigurationError):  # the identity basis is HEVC-coded
            rm.RankMethodConfig(targets, CodecSpec(kind="svd", rank=1), reuse=False, fixed_basis="identity")


def _toy_model(layers=2, channels=64, dtype="bfloat16"):
    return ModelIdentity("wl-base-toy", "wl-base-v1", layers, channels, "tokens_channels", dtype)


def _toy_inputs(model, *, frames=2, grid=(6, 8), seed=0, decay=0.15):
    """Pre-RoPE K / V with a decaying spectrum (every K and V of every layer)."""
    generator = torch.Generator().manual_seed(seed)
    tokens, channels = frames * grid[0] * grid[1], model.channels_for("K")
    sources, contexts = {}, {}
    for layer in range(model.layers):
        for target in ("K", "V"):
            key = MatrixKey(layer, target)
            spectrum = torch.exp(-decay * torch.arange(channels, dtype=torch.float64))
            left = torch.randn(tokens, channels, generator=generator, dtype=torch.float64) * spectrum
            right = torch.linalg.qr(torch.randn(channels, channels, generator=generator, dtype=torch.float64))[0]
            sources[key] = (left @ right.T * 3).to(torch.bfloat16)
            contexts[key] = TensorContext(model, key, 0, frames, grid[0], grid[1],
                                          "pre_rope" if target == "K" else "none")
    return sources, contexts


def _plan(method, sources, contexts, **kwargs):
    registry = RankPredictorRegistry("wl-base-test", anchor_rank=method.anchor_rank)
    bindings = {key: rm.ActiveBasisBinding() for key in sources}
    rows = {key: None for key in sources}
    source_ids = {key: canonical_digest({"key": key.to_dict()}) for key in sources}
    planner = rm.LatestMethodPlanner(method, registry.pin(), device="cpu", **kwargs)
    return planner.plan(sources, contexts, bindings, rows, chunk_id="c0", source_ids=source_ids), registry


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.model = _toy_model()
        self.sources, self.contexts = _toy_inputs(self.model)
        self.targets = EnergyTargets("custom", 0.98, 0.95, True, "test")

    def test_fixed_rank_fits_exactly_the_configured_width_without_target_or_learner(self):
        for solver in ("topk_rr", "gram_eigh"):
            method = rm.RankMethodConfig(self.targets, CodecSpec(kind="svd_hevc", rank=1), reuse=False,
                                         fixed_rank_k=32, fixed_rank_v=64, solver=solver)
            plan, registry = _plan(method, self.sources, self.contexts)
            self.assertEqual(registry.pin().sequence, 0)
            for key, item in plan.plans.items():
                width = 32 if key.target == "K" else 64
                self.assertEqual((item.decision, item.reason, item.rank_view, item.diagnostic),
                                 ("fresh", rm.FIXED_RANK_REASON, width, True))
                self.assertEqual(tuple(item.new_rows.shape), (width, self.model.channels_for("K")))
                self.assertEqual((item.observations, item.predictions), ((), ()))
                self.assertGreater(item.decision_energy, 0.5)
            k_energy = plan.plans[MatrixKey(0, "K")].decision_energy
            self.assertLess(k_energy, 1.0)  # rank 32 of a decaying 64-channel spectrum: a fixed, unverified width
            # top-k Rayleigh-Ritz at a fixed width captures (almost) the exact truncated-SVD energy
            source = self.sources[MatrixKey(0, "K")].double()
            singular = torch.linalg.svdvals(source)
            optimum = float(singular[:32].square().sum() / singular.square().sum())
            self.assertGreater(k_energy, optimum - 2e-3)
            self.assertLessEqual(k_energy, optimum + 1e-6)

    def test_fixed_rank_is_clamped_to_the_matrix_width(self):
        method = rm.RankMethodConfig(self.targets, CodecSpec(kind="svd_hevc", rank=1), reuse=False,
                                     fixed_rank_k=512, fixed_rank_v=1024)
        plan, _ = _plan(method, self.sources, self.contexts)
        self.assertEqual({item.rank_view for item in plan.plans.values()}, {64})

    def test_identity_basis_plan_and_finalization_fast_path_equal_the_generic_path(self):
        method = rm.RankMethodConfig(self.targets, CodecSpec(kind="svd_hevc", rank=1), reuse=False,
                                     fixed_basis="identity")
        plan, _ = _plan(method, self.sources, self.contexts)
        shared = rm.identity_rows(self.model.channels_for("K"))
        for key, item in plan.plans.items():
            self.assertEqual((item.decision, item.reason, item.rank_view, item.decision_energy),
                             ("fresh", rm.IDENTITY_BASIS_REASON, 64, 1.0))
            self.assertIs(item.new_rows, shared)
        self.assertTrue(rm.is_identity_view(shared[:64].contiguous()))
        self.assertFalse(rm.is_identity_view(shared.clone()))
        self.assertFalse(rm.is_identity_view(shared[:32]))
        key = MatrixKey(1, "K")
        spec = method.spec_for(64)
        fast = finalize_tiles([FinalizeItem(self.sources[key], shared, self.contexts[key], spec, key, 0.98, True,
                                            shared[:64].contiguous())], tile_rows=40, keep_coefficients=True)[0]
        generic = finalize_tiles([FinalizeItem(self.sources[key], shared.clone(), self.contexts[key], spec, key,
                                               0.98, True, shared.clone())], tile_rows=40,
                                 keep_coefficients=True)[0]
        self.assertTrue(torch.equal(fast[1], generic[1]))  # identical pixels
        self.assertEqual(fast[0].affine, generic[0].affine)
        self.assertEqual(fast[0].basis, generic[0].basis)
        self.assertEqual((fast[0].stored_energy, fast[0].ideal_energy), (1.0, 1.0))
        self.assertAlmostEqual(generic[0].stored_energy, 1.0, places=12)
        self.assertTrue(np.array_equal(fast[0].coefficients, generic[0].coefficients))
        self.assertIsNone(fast[0].gram)
        self.assertIsNotNone(generic[0].gram)

    def test_identity_basis_needs_as_many_tokens_as_channels(self):
        model = _toy_model(layers=1, channels=128)
        sources, contexts = _toy_inputs(model, frames=2, grid=(6, 8))
        method = rm.RankMethodConfig(self.targets, CodecSpec(kind="svd_hevc", rank=1), reuse=False,
                                     fixed_basis="identity")
        with self.assertRaises(ConfigurationError):
            _plan(method, sources, contexts)


class FactorPackTests(unittest.TestCase):
    def setUp(self):
        self.model = _toy_model()
        self.sources, self.contexts = _toy_inputs(self.model)
        targets = EnergyTargets("custom", 0.98, 0.95, True, "test")
        self.method = rm.RankMethodConfig(targets, CodecSpec(kind="svd", rank=1), reuse=False,
                                          fixed_rank_k=32, fixed_rank_v=64)

    def _pack(self, sources=None):
        sources = self.sources if sources is None else sources
        plan, _ = _plan(self.method, sources, self.contexts)
        items = [FinalizeItem(sources[key], item.new_rows, self.contexts[key], self.method.spec_for(item.rank_view),
                              key, item.target, item.diagnostic, item.new_rows) for key, item in plan.plans.items()]
        tiles = finalize_tiles(items, tile_rows=40)
        canvas = assemble_canvas("node-0", tiles, codec=pack_codec(self.method.codec), ladder=None, channels=64)
        canvas.to_host()
        return encode_node_pack(canvas), plan

    def test_factor_pack_round_trip_records_exact_fp16_errors(self):
        encoded, plan = self._pack()
        self.assertEqual(encoded.metadata["schema"], FACTOR_PACK_SCHEMA)
        self.assertEqual(pack_codec(self.method.codec), {"kind": "svd", "coefficient_dtype": "float16"})
        coefficients = encoded.payloads[fp.COEFFICIENT_PAYLOAD]
        self.assertEqual(len(coefficients), 96 * (32 + 64) * 2 * 2)  # FP16 [N, r] per tile, K 32 + V 64, 2 layers
        self.assertEqual(sorted(encoded.metrics["tile_stream_bytes"]), sorted([96 * 32 * 2] * 2 + [96 * 64 * 2] * 2))
        decoded = fp.decode_factor_pack(encoded)
        for key, dense in decoded.items():
            source = self.sources[key].float()
            error = float((dense - source).norm() / source.norm())
            label = f"{key.layer}:{key.target}"
            recorded = encoded.metrics["post_codec_relative_error"][label]
            self.assertAlmostEqual(error, recorded, delta=1e-5)  # the recorded error is the exact stored error
            stored = plan.plans[key].decision_energy
            self.assertAlmostEqual(recorded, math.sqrt(1 - stored), delta=2e-3)
            if key.target == "V":  # full rank: exact up to the FP16 rounding of both factors
                self.assertLess(error, 2e-3)
        # bases detach like a tensor_pack.v2 pack's and re-attach with their checks
        metadata, payloads, bases = detach_pack_bases(encoded)
        self.assertEqual(set(payloads), {fp.COEFFICIENT_PAYLOAD})
        from reelcache.node_packets import attach_pack_bases
        again = attach_pack_bases(metadata, payloads, bases)
        prepared = fp.prepare_factor_packet(again, verify=True)
        self.assertEqual(prepared.schema, FACTOR_PACK_SCHEMA)
        self.assertEqual(prepared.stream_bytes, len(coefficients))
        self.assertEqual([tile.rank for tile in prepared.tiles], [32, 64, 32, 64])
        fp.validate_factor_pack(EncodedTensor(metadata, payloads, None, {}), external_bases=True)

    def test_tampered_or_mismatched_factor_packs_raise(self):
        encoded, _ = self._pack()
        payloads = dict(encoded.payloads)
        flipped = bytearray(payloads[fp.COEFFICIENT_PAYLOAD])
        flipped[7] ^= 1
        payloads[fp.COEFFICIENT_PAYLOAD] = bytes(flipped)
        with self.assertRaises(IntegrityError):
            fp.prepare_factor_packet(EncodedTensor(encoded.metadata, payloads, None, {}), verify=True)
        short = dict(encoded.payloads)
        short[fp.COEFFICIENT_PAYLOAD] = short[fp.COEFFICIENT_PAYLOAD][:-2]
        with self.assertRaises(IntegrityError):
            fp.prepare_factor_packet(EncodedTensor(encoded.metadata, short, None, {}))
        metadata = json.loads(json.dumps(encoded.metadata))
        metadata["tiles"][1]["byte_range"][0] += 2
        with self.assertRaises(IntegrityError):
            fp.validate_factor_pack(EncodedTensor(metadata, encoded.payloads, None, {}))
        metadata = json.loads(json.dumps(encoded.metadata))
        metadata["schema"] = PACK_SCHEMA
        with self.assertRaises(IntegrityError):
            fp.validate_factor_pack(EncodedTensor(metadata, encoded.payloads, None, {}))
        with self.assertRaises(Exception):  # no HEVC path accepts a factor pack
            from reelcache.packed_decode import prepare_packet
            prepare_packet(encoded)

    def test_fp16_coefficient_overflow_raises(self):
        sources = dict(self.sources)
        key = MatrixKey(0, "V")
        sources[key] = (sources[key].float() * 1e4).to(torch.bfloat16)
        with self.assertRaises(IntegrityError):
            self._pack(sources)


@contextlib.contextmanager
def _toy_grid():
    with mock.patch.multiple(tsb, GRID=TOY_GRID, TPF=TOY_TPF):
        yield


class BaselineBridgeTests(tsb.BridgeTestCase):
    """Every arm through the bridge (toy causal model, CPU planner, real libx265 at QP8 for the HEVC arms)."""

    representation = "svd_hevc"

    def setUp(self):
        super().setUp()
        patcher = mock.patch.multiple(tsb, GRID=TOY_GRID, TPF=TOY_TPF)
        patcher.start()
        self.addCleanup(patcher.stop)

    def reference(self, blocks=3):
        bridge = self.bridge("reference", representation="raw")
        _, latents, caches, _ = tsb.drive(bridge, tsb.request(tsb.handle("ref"), blocks=blocks))
        return latents, caches

    def errors(self, caches, reference, chunks):
        tokens = chunks * tsb.FRAMES * tsb.TPF
        result = {}
        for layer in range(tsb.LAYERS):
            for name in ("k", "v"):
                restored = getattr(caches["cond"][layer], name)[:, :tokens].double()
                native = getattr(reference["cond"][layer], name)[:, :tokens].double()
                result[(layer, name)] = float((restored - native).norm() / native.norm())
        return result

    def run_arm(self, store, **overrides):
        """Two archived chunks (durable pause), then restores: raw tier, prepared packs, cold packs."""
        bridge = self.bridge(store, verify_post_codec=True, **overrides)
        first, _, _, _ = tsb.drive(bridge, tsb.request(tsb.handle("s1", pause_after_chunks=2), blocks=3))
        commits = first["commits"]
        self.assertEqual(len(commits), 2)
        tiers = {}
        restores = {}
        # the CPU software restorer of the svd_hevc packs has no prepared cache: its compressed restores are cold
        cached = callable(getattr(bridge.archive.restorer, "drop_caches", None))
        for label in ("raw", "prepared", "cold") if cached else ("raw", "cold"):
            if label == ("prepared" if cached else "cold"):
                bridge.archive.drop_raw("s1")  # raw tier copies dropped: prepared packs (prepare-on-drop)
                bridge.archive.raw_tier.flush(30)
            if label == "cold" and cached:
                bridge.archive.restorer.drop_caches()  # nothing prepared, no resident basis: a cold restore
            bridge.drop_resident()
            fork = f"f-{label}"
            summary, latents, caches, _ = tsb.drive(bridge, tsb.request(tsb.handle(
                fork, "fork", from_session="s1", checkpoint=2), blocks=3))
            lease = summary["restore"]["lease"]
            tiers[label] = (lease.get("tiers") or {}).get("chunks")
            restores[label] = (summary, latents, caches)
        return bridge, commits, tiers, restores

    def check_tiers(self, tiers):
        self.assertTrue(set(tiers["raw"]) <= {"gpu_raw", "host_raw"}, tiers)  # the raw tier serves first
        self.assertEqual(sum(tiers["raw"].values()), 2)
        if "prepared" in tiers:
            self.assertEqual(tiers["prepared"], {"compressed_prepared": 2})
        self.assertEqual(tiers["cold"], {"compressed_cold": 2})

    def check_records(self, commits, representation, expect_rank=None):
        native = 2 * tsb.LAYERS * 2 * tsb.FRAMES * tsb.TPF * tsb.CHANNELS  # BF16 K and V per chunk
        for commit in commits:
            self.assertEqual(commit["representation"], representation)
            self.assertEqual(commit["native_bytes"], native)
            self.assertEqual(sum(row["incremental_bytes"] for row in commit["matrices"]), commit["incremental_bytes"])
            for row in commit["matrices"]:
                self.assertEqual(row["decision"], "fresh")  # no basis sharing in any arm here
                self.assertGreater(row["incremental_bytes"], 0)
                self.assertIsNotNone(row["post_codec_relative_error"])
                self.assertGreaterEqual(row["post_codec_relative_error"], 0.0)
                if expect_rank is not None:
                    self.assertEqual(row["rank_view"], expect_rank(row["key"]))
        return commits

    def test_svd_arm_fp16_factors_restore_from_every_tier(self):
        latents_ref, caches_ref = self.reference()
        bridge, commits, tiers, restores = self.run_arm("svd", representation="svd", reuse=False)
        self.check_tiers(tiers)
        self.check_records(commits, "svd", expect_rank=lambda key: 32)  # C=64: K 64/6 -> 32, V 64/4 = 16 -> 32
        method = commits[0]["method"]
        self.assertEqual((method["fixed_ranks"], method["reuse"], method["codec"]["kind"]),
                         ({"K": 32, "V": 32}, False, "svd"))
        node = bridge.archive.runtime.load_node(commits[0]["node"])
        record = bridge.archive.runtime.load_record(node.matrix_records[0][1])
        packet = bridge.archive.runtime.load_packet(record.packet_ref)
        self.assertEqual(packet["metadata"]["schema"], FACTOR_PACK_SCHEMA)
        self.assertNotIn("stream.hevc", packet["payloads"])
        actual = record.decision["actual_bytes"]
        self.assertEqual(actual["tile_stream_bytes"], 96 * 32 * 2)  # FP16 P [96, 32]
        self.assertEqual(actual["affine_bytes"], 0)
        self.assertEqual(actual["new_basis_block_bytes"], 32 * 64 * 2)  # FP16 Q [32, 64], a fresh basis per chunk
        self.assertEqual(actual["incremental_bytes"], 96 * 32 * 2 + 32 * 64 * 2)
        self.assertIsNone(bridge.archive.encoder_record["selected"])
        for label in ("prepared", "cold"):
            summary, latents, caches = restores[label]
            lease = summary["restore"]["lease"]
            self.assertEqual((lease["representation"], lease["decoder"]), ("svd", "factor_gemm"))
            errors = self.errors(caches, caches_ref, 2)
            recorded = {(int(row["key"].split(":")[0]), row["key"].split(":")[1].lower()):
                        row["post_codec_relative_error"] for row in commits[0]["matrices"]}
            for item, error in errors.items():  # restored error (incl. the BF16 cache cast) ~ the recorded error
                self.assertGreater(error, 0.0)
                self.assertLess(abs(error - recorded[item]), 0.02)
            self.assertLess(float((latents - latents_ref).norm() / latents_ref.norm()), 0.05)
        prepared, cold = restores["prepared"][2], restores["cold"][2]
        for layer in range(tsb.LAYERS):  # the same compressed bytes restore the same KV from either tier
            self.assertTrue(torch.equal(prepared["cond"][layer].k[:, :192], cold["cond"][layer].k[:, :192]))
        self.assertLifetimesRetired(bridge)

    def test_svd_full_rank_is_exact_up_to_fp16_rounding(self):
        _, caches_ref = self.reference()
        bridge, commits, tiers, restores = self.run_arm("svd-full", representation="svd", reuse=False,
                                                        svd_rank_fraction_k=1.0, svd_rank_fraction_v=1.0)
        self.check_tiers(tiers)
        self.check_records(commits, "svd", expect_rank=lambda key: 64)
        for row in commits[0]["matrices"]:
            self.assertAlmostEqual(row["stored_factor_energy"], 1.0, places=6)
            self.assertLess(row["post_codec_relative_error"], 1e-3)
        errors = self.errors(restores["cold"][2], caches_ref, 2)
        self.assertLess(max(errors.values()), 4e-3)  # FP16 factors + the BF16 cache cast

    def test_hevc_arm_identity_basis_through_ffmpeg_at_qp8(self):
        latents_ref, caches_ref = self.reference()
        bridge, commits, tiers, restores = self.run_arm("hevc", representation="hevc", reuse=False)
        self.check_tiers(tiers)
        self.check_records(commits, "hevc", expect_rank=lambda key: 64)  # the full width: the KV matrix itself
        method = commits[0]["method"]
        self.assertEqual((method["fixed_basis"], method["codec"]["qp"], method["codec"]["pix_fmt"],
                          method["codec"]["inter_prediction"]), ("identity", 8, "yuv444p", False))
        runtime = bridge.archive.runtime
        node = runtime.load_node(commits[1]["node"])
        records = [runtime.load_record(ref) for _, ref in node.matrix_records]
        packet = runtime.load_packet(records[0].packet_ref)
        self.assertEqual(packet["metadata"]["schema"], PACK_SCHEMA)
        self.assertEqual(packet["metadata"]["codec"]["qp"], 8)
        for record in records:
            actual = record.decision["actual_bytes"]
            self.assertEqual(actual["new_basis_block_bytes"], 0)  # the shared identity basis
            self.assertEqual(actual["incremental_bytes"], actual["tile_stream_bytes"] + actual["affine_bytes"])
            self.assertEqual(record.decision["reason"], rm.IDENTITY_BASIS_REASON)
            self.assertEqual(record.energy.stored_factor_energy, 1.0)
            self.assertLess(record.energy.post_codec_relative_error, 0.1)
            version = runtime.basis_versions.load(record.basis_version_ref)
            rows = runtime.basis_versions.assemble(record.basis_version_ref)
            self.assertTrue(torch.equal(rows, torch.eye(64, dtype=torch.float16)))
            self.assertEqual(version.total_rank, 64)
        blocks = {runtime.basis_versions.load(record.basis_version_ref).added_rows_ref for record in records}
        self.assertEqual(len(blocks), 1)  # one identity block per channel count in the store
        self.assertEqual(set(restores), {"raw", "cold"})  # CPU: the software decoder (FFmpeg) restores cold
        for label in ("cold",):
            summary, latents, caches = restores[label]
            self.assertEqual(summary["restore"]["lease"]["representation"], "hevc")
            errors = self.errors(caches, caches_ref, 2)
            self.assertLess(max(errors.values()), 0.1)  # 8-bit per-channel QP8 coding of the KV values
            self.assertGreater(min(errors.values()), 0.0)
            self.assertLess(float((latents - latents_ref).norm() / latents_ref.norm()), 0.05)
        self.assertLifetimesRetired(bridge)

    def test_uniform_rank_arm_one_rank_for_every_layer_and_k_v(self):
        _, caches_ref = self.reference()
        bridge, commits, tiers, restores = self.run_arm("uniform", rank_policy="fixed_uniform", uniform_rank=32,
                                                        reuse=False)
        self.check_tiers(tiers)
        self.check_records(commits, "svd_hevc", expect_rank=lambda key: 32)
        self.assertEqual(commits[0]["method"]["fixed_ranks"], {"K": 32, "V": 32})
        for row in commits[0]["matrices"]:  # no energy target: achieved energies recorded, not enforced
            self.assertLess(row["stored_factor_energy"], 1.0)
            self.assertIsNotNone(row["ideal_subspace_energy"])
        errors = self.errors(restores["cold"][2], caches_ref, 2)
        self.assertLess(max(errors.values()), 0.3)
        clamped = self.bridge("uniform-clamped", rank_policy="fixed_uniform", uniform_rank=512, reuse=False)
        self.assertEqual((clamped.archive.method.fixed_rank_k, clamped.archive.method.fixed_rank_v), (64, 64))

    def test_baselines_with_cfg_streams_and_a_recache_epoch(self):
        """CFG streams (cond + uncond tiles in one node pack) and a prompt-switch recache node of two ranges (one pack
        per range) in both baselines; the recached ranges restore from their epoch."""
        streams = ("cond", "uncond")
        reference = self.bridge("switch-reference", representation="raw")
        _, _, expected, _ = tsb.drive(reference, tsb.switch_request(tsb.handle("ref"), streams=streams))
        for representation, bound in (("svd", 0.3), ("hevc", 0.1)):
            with self.subTest(representation=representation):
                bridge = self.bridge(f"{representation}-switch", representation=representation, reuse=False,
                                     raw_restore="off")
                first, _, _, _ = tsb.drive(bridge, tsb.switch_request(tsb.handle("s1", pause_after_chunks=4),
                                                                      streams=streams))
                kinds = [node.kind for node in bridge.archive.path(first["head"])]
                self.assertEqual(kinds, ["root", "chunk", "chunk", "chunk", "recache", "chunk"])
                recache = [commit for commit in first["commits"] if commit.get("kind") == "recache"]
                self.assertEqual(len(recache), 1)
                self.assertEqual(recache[0]["representation"], representation)
                self.assertEqual(recache[0]["packets"], 2)  # one pack per re-materialized chunk range
                self.assertEqual(len(recache[0]["matrices"]), 2 * len(streams) * tsb.LAYERS * 2)
                self.assertEqual({row["stream"] for row in recache[0]["matrices"]}, set(streams))
                resumed, _, caches, _ = tsb.drive(bridge, tsb.switch_request(tsb.handle("s1", "resume"),
                                                                             streams=streams))
                plan = resumed["restore"]["plans"]["cond"]
                self.assertEqual([span["source"] for span in plan["spans"]], ["chunk", "recache", "recache", "chunk"])
                errors = self.errors(caches, expected, 4)
                self.assertLess(max(errors.values()), bound)
                self.assertLifetimesRetired(bridge, resumed)

    def test_reuse_false_keeps_the_adaptive_policy_without_basis_sharing(self):
        bridge = self.bridge("noreuse", reuse=False)
        first, _, _, _ = tsb.drive(bridge, tsb.request(tsb.handle("s1", pause_after_chunks=3), blocks=4))
        commits = first["commits"]
        self.assertEqual(len(commits), 3)
        for commit in commits:
            self.assertEqual(commit["representation"], "svd_hevc")
            self.assertFalse(commit["method"]["reuse"])
            self.assertNotIn("fixed_ranks", commit["method"])
            for row in commit["matrices"]:
                self.assertEqual(row["decision"], "fresh")
                self.assertGreaterEqual(row["stored_factor_energy"], row["target"])  # the energy targets hold
        runtime = bridge.archive.runtime
        reasons = {runtime.load_record(ref).decision["reason"]
                   for commit in commits[1:] for _, ref in runtime.load_node(commit["node"]).matrix_records}
        self.assertEqual(reasons, {"reuse_disabled"})
        shared = self.bridge("reuse", reuse=True)
        again, _, _, _ = tsb.drive(shared, tsb.request(tsb.handle("s1", pause_after_chunks=3), blocks=4))
        decisions = {row["decision"] for commit in again["commits"][1:] for row in commit["matrices"]}
        self.assertTrue(decisions - {"fresh"}, decisions)  # with sharing, later chunks inherit / augment / refit
        bridge.drop_resident()
        resumed, _, _, _ = tsb.drive(bridge, tsb.request(tsb.handle("s1", "resume"), blocks=4))
        self.assertEqual((resumed["resumed_from_chunks"], resumed["status"]), (3, "completed"))

    def test_new_fields_at_their_defaults_change_no_session_identity_or_byte(self):
        """In-process A/B: the default svd_hevc and raw bridges with the new fields omitted vs stated explicitly at
        their defaults archive identical node, record and packet ids."""
        heads = {}
        torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, torch_threads)
        explicit = {"rank_policy": "adaptive", "uniform_rank": None, "svd_rank_fraction_k": None,
                    "svd_rank_fraction_v": None}
        for representation in ("svd_hevc", "raw"):
            for label, extra in (("omitted", {}), ("explicit", explicit)):
                bridge = self.bridge(f"{representation}-{label}", representation=representation, **extra)
                tsb.drive(bridge, tsb.request(tsb.handle("s1", pause_after_chunks=2), blocks=2))
                bridge.drain_all()
                path = [node.ref for node in bridge.archive.path(bridge.archive.head("s1"))]
                records = []
                if representation == "svd_hevc":  # lineage nodes: their matrix records and packets too
                    runtime = bridge.archive.runtime
                    records = sorted((ref, runtime.load_record(ref).packet_ref) for ref_node in path[1:]
                                     for _, ref in runtime.load_node(ref_node).matrix_records)
                heads[(representation, label)] = (path, records)
            self.assertEqual(heads[(representation, "omitted")], heads[(representation, "explicit")])
            self.assertEqual(len(heads[(representation, "omitted")][0]), 3)


class WarmupShapeTests(unittest.TestCase):
    def test_factor_warmup_shapes_cover_the_fixed_ranks(self):
        targets = EnergyTargets("custom", 0.98, 0.95, True, "test")
        method = rm.RankMethodConfig(targets, CodecSpec(kind="svd", rank=1), reuse=False, fixed_rank_k=256,
                                     fixed_rank_v=384)
        model = ModelIdentity("toy", "fp", 30, 1536, "sglang_causal_wan_bthd", "bfloat16")
        profile = tsb.PROFILE
        fused = factor_warmup_shapes(profile, model, method)
        self.assertEqual(len(fused), 1)
        self.assertEqual((fused[0].source, fused[0].rope, fused[0].ranks, fused[0].source_dtype),
                         ("float", "fused", (256, 384), torch.float16))
        native = factor_warmup_shapes(profile, model, method, rope_mode="native")
        self.assertEqual((native[0].rope, native[0].native_rope_dtype), ("native", torch.bfloat16))


class FactorStagingTests(unittest.TestCase):
    """W1.2 (CONTRACT §10 reader check): a cold factor-pack history staged by the compressed arms' engine -- parallel
    range reads, the coefficient payload hashed while it is read, device bases staged through the pinned ring, the
    integrity gate -- restores bitwise like the store path; a corrupt payload fails the gate and is quarantined."""

    CHUNKS, LAYERS, HEADS, DIM, FRAMES, GRID = 3, 2, 4, 16, 2, (6, 8)

    def setUp(self):
        import shutil
        import tempfile

        self.root = tempfile.mkdtemp(prefix="wl-base-staging-")
        self.addCleanup(shutil.rmtree, self.root, True)
        threads = torch.get_num_threads()  # the CPU reference reconstruction is bitwise reproducible on one thread
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def archive(self):
        from reelcache.integrations.sglang.commit import matrix_contexts
        from reelcache.integrations.sglang.lineage_archive import LineageSessionArchive

        channels = self.HEADS * self.DIM
        model = ModelIdentity("wl-base-staging", "v1", self.LAYERS, channels, "sglang_causal_wan_bthd", "bfloat16")
        config = ReelCacheRuntimeConfig(store_root=self.root, representation="svd", reuse=False, fit_device="cpu",
                                     decoder="software", x265_threads=1, encode_workers=1)
        archive = LineageSessionArchive(self.root, model, config=config,
                                        targets=EnergyTargets("custom", 0.98, 0.95, True, "test"),
                                        device=torch.device("cpu"))
        self.addCleanup(archive.close)
        self.assertIsInstance(archive.restorer, fp.FactorLineageRestorer)
        self.assertIsNone(archive.restorer.staging_engine)  # off CUDA: the store path (the reference here)
        tokens = self.FRAMES * self.GRID[0] * self.GRID[1]
        generator = torch.Generator().manual_seed(11)
        archive.runtime.create_lineage("s", epoch_id="e0", streams=("cond",))
        for index in range(self.CHUNKS):
            matrices = {}
            for layer in range(self.LAYERS):
                for target in ("K", "V"):
                    spectrum = torch.arange(1, channels + 1, dtype=torch.float32).pow(-0.8 - 0.1 * layer)
                    left = torch.randn(tokens, channels, generator=generator) * spectrum
                    matrices[(MatrixKey(layer, target), "cond")] = (left @ torch.randn(channels, channels,
                                                                                    generator=generator)).bfloat16()
            contexts = matrix_contexts(model, token_start=index * tokens, frames=self.FRAMES, grid=self.GRID,
                                       streams=("cond",))
            archive.pipeline.submit("s", f"c{index:06d}", matrices, contexts, kind="chunk", timeout=None)
        return archive, archive.pipeline.drain("s"), tokens

    def restore(self, restorer, head, tokens):
        from reelcache import restore_kernels as rk

        buffers = {(MatrixKey(layer, target), "cond"): torch.zeros(1, self.CHUNKS * tokens, self.HEADS, self.DIM,
                                                                   dtype=torch.bfloat16)
                   for layer in range(self.LAYERS) for target in ("K", "V")}

        def destination(node_id, key, tile):
            return rk.NativeView(buffers[key if isinstance(key, tuple) else (key, "cond")], "BTHD",
                                 token_offset=tile.token_start)

        report = restorer.restore(head, destination, None, ranges=[(0, self.CHUNKS * tokens)])
        for handle in report.pop("handles", []):
            handle.release()
        return buffers, report

    def staged_restorer(self, archive, **engine):
        from reelcache.restore_staging import StagingEngine

        engine = StagingEngine("cpu", **{"threads": 3, "prepare_threads": 2, "pinned_bytes": 1 << 20, **engine})
        self.addCleanup(engine.close)
        return fp.FactorLineageRestorer(archive.runtime, "cpu", staging_engine=engine), engine

    def test_staged_cold_restore_equals_the_store_path_bitwise(self):
        from reelcache import restore_staging

        archive, head, tokens = self.archive()
        expected, _ = self.restore(archive.restorer, head, tokens)
        # the toy coefficient payloads (~25 KB) take the large-object path: parallel 1 KiB ranges, hashed while read
        with mock.patch.multiple(restore_staging, _PARALLEL_READ_BYTES=4096, _READ_CHUNK_BYTES=1024):
            restorer, engine = self.staged_restorer(archive)
            cold, report = self.restore(restorer, head, tokens)
            self.assertEqual(report["verification"]["mode"], "async")
            self.assertEqual(report["verification"]["verified_payloads"], self.CHUNKS)
            stats = engine.stats()
            self.assertEqual((stats["histories"], stats["packs"], stats["verify_streamed"]), (1, self.CHUNKS,
                                                                                             self.CHUNKS))
            self.assertEqual(stats["verify_failures"], 0)
            self.assertGreater(stats["payload_bytes"], 0)
            self.assertEqual(restorer.prepared.stats()["packets"], self.CHUNKS)  # staged packs adopted
            prepared, again = self.restore(restorer, head, tokens)  # prepared packs + resident bases: no staging
            self.assertEqual(engine.stats()["histories"], 1)
            self.assertEqual(again["verification"]["verified_payloads"], 0)
        for key in expected:
            self.assertTrue(torch.equal(cold[key], expected[key]), key)
            self.assertTrue(torch.equal(prepared[key], expected[key]), key)
        tile = restorer.prepared.get(restorer.history(head)[0]["packet"])[0].tiles[0]
        self.assertFalse(tile.coefficients.is_pinned())
        self.assertIsNone(tile.basis)  # stripped in the prepared cache; bases live in the device basis cache

    def test_synchronous_verification_and_small_payloads_take_the_plain_reads(self):
        archive, head, tokens = self.archive()
        expected, _ = self.restore(archive.restorer, head, tokens)
        restorer, engine = self.staged_restorer(archive, verify_threads=0)  # payloads under 8 MiB: one read task
        cold, report = self.restore(restorer, head, tokens)
        self.assertEqual(report["verification"]["mode"], "sync")
        self.assertEqual(engine.stats()["verify_streamed"], 0)
        for key in expected:
            self.assertTrue(torch.equal(cold[key], expected[key]), key)

    def test_a_corrupt_coefficient_payload_fails_the_gate_and_is_quarantined(self):
        from reelcache import restore_staging

        archive, head, tokens = self.archive()
        packet_ref = archive.restorer.history(head)[1]["packet"]
        blob = archive.runtime.load_packet(packet_ref)["payloads"][fp.COEFFICIENT_PAYLOAD]
        path = archive.runtime.store.path_for(blob)
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 0x10
        path.chmod(0o644)
        path.write_bytes(bytes(data))
        with mock.patch.multiple(restore_staging, _PARALLEL_READ_BYTES=4096, _READ_CHUNK_BYTES=1024):
            restorer, engine = self.staged_restorer(archive)
            with self.assertRaises(IntegrityError):
                self.restore(restorer, head, tokens)
        self.assertEqual(engine.stats()["verify_failures"], 1)
        self.assertEqual(restorer.prepared.stats()["packets"], 0)  # nothing adopted from a failed gate
        self.assertIn(blob, archive.runtime.store.quarantined_objects)

    def test_identity_basis_chains_share_one_staged_tip(self):
        """HEVC-only arm: every key's identity version is ONE content-addressed row block; staging reads, hashes and
        copies it once (one tip per channel width), the other versions alias it and keep their own checks."""
        from reelcache.integrations.sglang.commit import matrix_contexts
        from reelcache.integrations.sglang.lineage_archive import LineageSessionArchive
        from reelcache.packed_decode import prepare_packet
        from reelcache.restore_staging import StagingEngine

        channels = self.HEADS * self.DIM
        model = ModelIdentity("wl-base-identity", "v1", self.LAYERS, channels, "sglang_causal_wan_bthd", "bfloat16")
        config = ReelCacheRuntimeConfig(store_root=self.root, representation="hevc", reuse=False, fit_device="cpu",
                                     decoder="software", x265_threads=1, encode_workers=1)
        archive = LineageSessionArchive(self.root, model, config=config,
                                        targets=EnergyTargets("custom", 0.98, 0.95, True, "test"),
                                        device=torch.device("cpu"))
        self.addCleanup(archive.close)
        tokens = self.FRAMES * self.GRID[0] * self.GRID[1]
        generator = torch.Generator().manual_seed(5)
        archive.runtime.create_lineage("s", epoch_id="e0", streams=("cond",))
        for index in range(2):
            matrices = {(MatrixKey(layer, target), "cond"): torch.randn(tokens, channels, generator=generator).bfloat16()
                        for layer in range(self.LAYERS) for target in ("K", "V")}
            contexts = matrix_contexts(model, token_start=index * tokens, frames=self.FRAMES, grid=self.GRID,
                                       streams=("cond",))
            archive.pipeline.submit("s", f"c{index:06d}", matrices, contexts, kind="chunk", timeout=None)
        head = archive.pipeline.drain("s")
        runtime = archive.runtime
        history = archive.restorer.history(head)
        engine = StagingEngine("cpu", threads=3, prepare_threads=2, pinned_bytes=1 << 20)
        self.addCleanup(engine.close)
        staged = engine.stage(runtime, history)
        staged.wait()
        keys = self.LAYERS * 2
        self.assertEqual((staged.stats["tips"], staged.stats["shared_tips"]), (1, keys - 1))
        self.assertEqual(staged.stats["tip_bytes"], channels * channels * 2)  # one identity block staged
        self.assertEqual(len(staged.versions), keys)  # every key's complete version, all on the shared rows
        self.assertEqual(len({tensor.data_ptr() for _, _, tensor, _ in staged.versions}), 1)
        for item in history:
            expected = prepare_packet(runtime.node_pack(item["packet"]))
            for mine, theirs in zip(staged.prepared[item["packet"]].tiles, expected.tiles, strict=True):
                self.assertEqual((mine.key, mine.rank), (theirs.key, theirs.rank))
                self.assertTrue(torch.equal(mine.basis, theirs.basis))
        self.assertTrue(torch.equal(staged.versions[0][2], torch.eye(channels, dtype=torch.float16)))
        staged.release()


if __name__ == "__main__":
    unittest.main()
