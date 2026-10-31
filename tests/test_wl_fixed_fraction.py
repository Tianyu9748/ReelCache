"""wl-base (workload-v2; coordinator 2026-10-19 23:02): ``rank_policy="fixed_fraction"`` for ``svd_hevc``.

The ablation ladder's step (c), SVD + HEVC without basis sharing, at the ``svd`` arm's ranks (K = C/6, V = C/4 to the
nearest multiple of 32), so (a) SVD only -> (c) SVD + HEVC changes the codec alone. Configuration only: the planner
consumes ``ReelCacheRuntimeConfig.fixed_ranks`` through ``lineage_archive.fixed_rank_fields`` (unchanged).

* valid and invalid combinations (``reuse=False`` and ``uniform_rank=None`` required, ``svd_hevc`` only, the fraction
  checks of the ``svd`` arm);
* the ranks per channel count (1536 -> 256 / 384, 3072 -> 512 / 768) equal the ``svd`` arm's for every C;
* content identities: the F6 / W1.4 golden ids of the defaults and of every existing arm are unchanged (computed with
  the base tree ``freeze_w14`` cc59cacc52d97434), the new arm has its own id, explicit default fractions equal the
  resolved ones, JSON round trip;
* negative controls: mutations of the policy (no rank branch, policy not registered, sharing allowed) are caught.
"""

import json
import unittest
from unittest import mock

from reelcache.errors import ConfigurationError
from reelcache.integrations.sglang import config as sglang_config
from reelcache.integrations.sglang.config import ReelCacheRuntimeConfig, fixed_rank
from reelcache.integrations.sglang.lineage_archive import fixed_rank_fields
from reelcache.types import ModelIdentity, canonical_digest

#: canonical_digest(content_dict()) of existing configurations, computed with the base tree (freeze_w14,
#: cc59cacc52d97434) before the change; "default" / "reuse_false" are also F6's (tests/test_wl_baselines.F6_CONTENT);
#: values under this release's names: the digests of the same inputs with the renamed identifiers.
BASE_CONTENT = {
    "default": ({}, "6d971ed3d7fc0c0c720ee9add3ca2c15c65563cc22ce5c3c94a5cd1d2ae3f6ea"),
    "reuse_false": ({"reuse": False}, "f9cc1d37c3cc3ebfff6d375170658ed91b9cd3b9a8f30334a28de6ee54521e12"),
    "svd": ({"representation": "svd", "reuse": False},
            "ba682ce3f42ab07acd897008517c023085f818cf4084e85c8765486224c229a0"),
    "hevc": ({"representation": "hevc", "reuse": False},
             "79687ace69cc8a7e162bc8b735c415c9fa8f0b335fd36df57cf9f09b28d04817"),
    "fixed_uniform_320": ({"rank_policy": "fixed_uniform", "uniform_rank": 320, "reuse": False},
                          "cd30abb5234630f6349b1d3aba115cd3568cef4801b9c3b6530fec0b753812d0"),
    "fixed_uniform_736": ({"rank_policy": "fixed_uniform", "uniform_rank": 736, "reuse": False},
                          "1de8ee26d7fc47531ac8c8ec2006db33956e08196065591353e9336e9f7d09a2"),
}
#: the new arm's id at its defaults (golden from this change; a later change to it must be deliberate)
FIXED_FRACTION_CONTENT = "fe12c0d87d86ddd19448ee6319e1ec2e412733796dfae31f9f2476e89f5bdbd7"
FIXED_FRACTION = {"rank_policy": "fixed_fraction", "reuse": False}


def config(**values):
    return ReelCacheRuntimeConfig(store_root="/tmp/wl-base-fixed-fraction", **values)


def model(channels: int) -> ModelIdentity:
    return ModelIdentity("toy", "fp", 2, channels, "tokens_channels", "bfloat16")


class FixedFractionConfigTests(unittest.TestCase):
    def test_valid_combinations_resolve_the_svd_fractions(self):
        arm = config(**FIXED_FRACTION)
        self.assertEqual((arm.svd_rank_fraction_k, arm.svd_rank_fraction_v), (1 / 6, 1 / 4))
        self.assertIsNone(arm.uniform_rank)
        self.assertEqual(arm.codec_template().kind, "svd_hevc")  # HEVC-coded factors, unlike representation="svd"
        explicit = config(**FIXED_FRACTION, svd_rank_fraction_k=0.25, svd_rank_fraction_v=0.5)
        self.assertEqual(explicit.fixed_ranks({"K": 1536, "V": 1536}), {"K": 384, "V": 768})
        for extra in ({"refit": False}, {"hevc_qp": 8}, {"svd_rank_fraction_k": 1, "svd_rank_fraction_v": 1}):
            with self.subTest(extra=extra):
                config(**FIXED_FRACTION, **extra)  # legal

    def test_invalid_combinations_raise(self):
        bad = [
            {"rank_policy": "fixed_fraction"},  # reuse defaults True: basis sharing
            {"rank_policy": "fixed_fraction", "reuse": True},
            {**FIXED_FRACTION, "uniform_rank": 320},
            {**FIXED_FRACTION, "uniform_rank": 256},
            {**FIXED_FRACTION, "representation": "svd"},  # SVD only is representation="svd" itself
            {**FIXED_FRACTION, "representation": "hevc"},
            {**FIXED_FRACTION, "representation": "raw"},
            {**FIXED_FRACTION, "svd_rank_fraction_k": 0.0},
            {**FIXED_FRACTION, "svd_rank_fraction_k": 1.5},
            {**FIXED_FRACTION, "svd_rank_fraction_k": -0.25},
            {**FIXED_FRACTION, "svd_rank_fraction_v": float("nan")},
            {**FIXED_FRACTION, "svd_rank_fraction_v": True},
            {**FIXED_FRACTION, "svd_rank_fraction_v": "0.25"},
            {"svd_rank_fraction_k": 1 / 6, "reuse": False},  # fractions with the adaptive policy (unchanged rule)
            {"rank_policy": "fixed_uniform", "uniform_rank": 320, "reuse": False, "svd_rank_fraction_v": 0.25},
            {"rank_policy": "fixed-fraction", "reuse": False},
        ]
        for values in bad:
            with self.subTest(values=values), self.assertRaises(ConfigurationError):
                config(**values)

    def test_ranks_equal_the_svd_arms_for_every_channel_count(self):
        arm, svd = config(**FIXED_FRACTION), config(representation="svd", reuse=False)
        self.assertEqual(arm.fixed_ranks({"K": 1536, "V": 1536}), {"K": 256, "V": 384})
        self.assertEqual(arm.fixed_ranks({"K": 3072, "V": 3072}), {"K": 512, "V": 768})
        for channels in (20, 64, 1024, 1536, 1600, 3072, 5120):
            pair = {"K": channels, "V": channels}
            with self.subTest(channels=channels):
                self.assertEqual(arm.fixed_ranks(pair), svd.fixed_ranks(pair))
                self.assertEqual(arm.fixed_ranks(pair), {"K": fixed_rank(channels, 1 / 6),
                                                         "V": fixed_rank(channels, 1 / 4)})
        self.assertEqual(fixed_rank_fields(arm, model(1536)), {"fixed_rank_k": 256, "fixed_rank_v": 384})
        self.assertEqual(fixed_rank_fields(arm, model(3072)), fixed_rank_fields(svd, model(3072)))

    def test_existing_identities_unchanged_and_the_new_arm_distinct(self):
        for name, (values, digest) in BASE_CONTENT.items():
            with self.subTest(name=name):
                self.assertEqual(canonical_digest(config(**values).content_dict()), digest)
        content = config(**FIXED_FRACTION).content_dict()
        self.assertEqual(canonical_digest(content), FIXED_FRACTION_CONTENT)
        self.assertEqual((content["representation"], content["rank_policy"]), ("svd_hevc", "fixed_fraction"))
        self.assertEqual((content["svd_rank_fraction_k"], content["svd_rank_fraction_v"]), (1 / 6, 1 / 4))
        self.assertNotIn("uniform_rank", content)  # its default (None) stays out, as for every arm
        explicit = config(**FIXED_FRACTION, svd_rank_fraction_k=1 / 6, svd_rank_fraction_v=0.25)
        self.assertEqual(canonical_digest(explicit.content_dict()), FIXED_FRACTION_CONTENT)
        identities = {canonical_digest(config(**values).content_dict()) for values in (
            *(values for values, _ in BASE_CONTENT.values()), FIXED_FRACTION,
            {**FIXED_FRACTION, "svd_rank_fraction_k": 0.25})}
        self.assertEqual(len(identities), len(BASE_CONTENT) + 2)
        first = config(**FIXED_FRACTION)
        again = ReelCacheRuntimeConfig.from_dict(json.loads(json.dumps(first.to_dict())))
        self.assertEqual(again, first)
        self.assertEqual(again.content_dict(), first.content_dict())


class NegativeControlTests(unittest.TestCase):
    """Mutations of the policy that the checks above must catch."""

    def test_mutations_are_caught(self):
        # a uniform rank of the same rank sum is not the svd arm's ranks: the rank assertions discriminate
        uniform = config(rank_policy="fixed_uniform", uniform_rank=320, reuse=False)
        self.assertNotEqual(uniform.fixed_ranks({"K": 1536, "V": 1536}), {"K": 256, "V": 384})
        arm = config(**FIXED_FRACTION)
        # fixed_ranks without the fixed_fraction branch: the planner would fit adaptive ranks (no fixed fields)
        with mock.patch.object(ReelCacheRuntimeConfig, "fixed_ranks", lambda self, channels: None):
            self.assertEqual(fixed_rank_fields(arm, model(1536)), {})
        # the policy not registered: rejected by the validation (the base tree's behaviour)
        with mock.patch.object(sglang_config, "RANK_POLICIES", ("adaptive", "fixed_uniform")):
            with self.assertRaises(ConfigurationError):
                config(**FIXED_FRACTION)
        # basis sharing not enforced: the fixed-rank check is live for the new policy
        with mock.patch.object(sglang_config, "FIXED_RANK_POLICIES", ("fixed_uniform",)):
            config(rank_policy="fixed_fraction")  # the mutation accepts sharing ...
        with self.assertRaises(ConfigurationError):
            config(rank_policy="fixed_fraction")  # ... the real check rejects it


if __name__ == "__main__":
    unittest.main()
