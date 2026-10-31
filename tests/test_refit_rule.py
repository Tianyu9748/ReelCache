"""Freeze D: the cumulative ("ski-rental") refit rule with its hysteresis, and the paper rule (the default again since
the user decision of 2026-10-17; freezes D to F defaulted to the cumulative rule, which stays selectable).

Planner level (CPU, ``svd`` codec with a dyadic coefficient byte proxy, so every estimate is exact): a lineage whose
dominant directions move every chunk augments repeatedly under the paper rule while the cumulative rule refits once
the excess accrued since its last fresh fit reaches the refit's basis cost (hysteresis 1, the default; the mechanics
tests pin it) or a multiple of it (hysteresis 1.5, 2). Lineage level (``svd_hevc`` node packs,
the Step 6 fixture): the debt travels in the nodes' active bindings and in every record, so forks and reopened
runtimes continue with their parent's debt, and the paper rule writes no debt at all.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import replace
from unittest import mock

import torch

from reelcache.errors import ConfigurationError
from reelcache.manager import KVRuntime
from reelcache.predictor_registry import RankPredictorRegistry
from reelcache.rank_method import (
    REFIT_ESTIMATE_V1, REFIT_ESTIMATE_V2, REFIT_RULES, LatestMethodPlanner, RankMethodConfig, chosen_estimate_bytes,
    plan_chunk, refit_debt_test,
)
from reelcache.types import (
    ActiveBasisBinding, ChunkNode, CodecSpec, EnergyTargets, MatrixKey, ModelIdentity, TensorContext,
    canonical_digest,
)
from tests.step6_support import MODEL as STEP6_MODEL, RuntimeFactory, chunk, method, node_content
from tests.step6_support import registry as step6_registry

CHANNELS, ROWS = 256, 480
MODEL = ModelIdentity("refit-rule", "refit-rule-v1", 2, CHANNELS, "tokens_channels", "float32")
V0, V1 = MatrixKey(0, "V"), MatrixKey(1, "V")
TARGETS = EnergyTargets("refit-test", 0.98, 0.95, True)
# 0.375 bytes per coefficient: 180 bytes per coefficient rank at 480 rows; with 256 channels a fresh fit's basis costs
# 512 bytes per row, so every estimate below is an exact integer.
PROXY = 0.375
REF = "manifest:" + "1" * 64
HEX = "a" * 64

# RankMethodConfig(TARGETS, CodecSpec("svd", rank=1)).fingerprint and the svd_hevc template's, computed with freeze
# C3's tree (c430c1734048af33), whose methods had no refit rule: the paper rule keeps them. Values under this release's names: the digests of the same inputs with the renamed identifiers.
C3_PAPER_FINGERPRINTS = {
    "svd": "a3f007b03ece1394b29272de32f2c848e658307673e63c714e891d790967d92a",
    "svd_hevc": "306d1d4a5adcbc21242e03f581d4845990a758a8a6b778672e13c02c34c39f19",
}


def config(**kwargs):
    """The test method: the cumulative rule's plain ski-rental test (hysteresis 1) unless a test asks for another
    (the core default is the paper rule again, user decision 2026-10-17)."""
    kwargs.setdefault("refit_rule", "cumulative")
    kwargs.setdefault("coefficient_bytes_proxy", PROXY)
    kwargs.setdefault("refit_hysteresis", 1.0)
    return RankMethodConfig(TARGETS, CodecSpec("svd", rank=1), **kwargs)


def orthonormal(seed, size=CHANNELS):
    generator = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(size, size, generator=generator, dtype=torch.float64))[0].T


def drifting_sources(count, *, seed=5, moves=None, move=8):
    """Chunks whose top ``move`` directions are replaced in chunks 1..``moves`` (all chunks when None)."""
    spectrum = torch.logspace(0, -2.5, CHANNELS, dtype=torch.float64)
    basis = orthonormal(seed)
    sources = []
    for index in range(count):
        if index and (moves is None or index <= moves):
            basis = basis.clone()
            basis[:move] = orthonormal(seed + 100 + index)[:move]
            basis = torch.linalg.qr(basis.T)[0].T
        generator = torch.Generator().manual_seed(seed + 1000 + index)
        left = torch.linalg.qr(torch.randn(ROWS, CHANNELS, generator=generator, dtype=torch.float64))[0]
        sources.append(((left * spectrum) @ basis).float())
    return sources


def context(key, index):
    return TensorContext(MODEL, key, index * ROWS, 2, ROWS // 48, 24, "none")


class Chain:
    """A planner-level lineage of one or more V matrices: bindings and active rows advance like a commit's."""

    def __init__(self, cfg, keys=(V0,)):
        self.cfg, self.keys = cfg, tuple(keys)
        self.registry = RankPredictorRegistry("refit-chain")
        self.bindings = {key: ActiveBasisBinding() for key in self.keys}
        self.rows = {key: None for key in self.keys}
        self.index = 0

    def plan(self, sources, *, bindings=None, cfg=None):
        bindings = self.bindings if bindings is None else bindings
        return plan_chunk(cfg or self.cfg, self.registry.pin(), sources,
                          {key: context(key, self.index) for key in sources}, dict(bindings),
                          {key: self.rows[key] for key in sources}, chunk_id=f"c{self.index}",
                          source_ids={key: canonical_digest(["refit-chain", self.index, key.to_dict()])
                                      for key in sources})

    def advance(self, plan):
        for key, item in plan.plans.items():
            self.registry.observe(item.observations)
            binding = self.bindings[key]
            if item.decision in ("fresh", "refit"):
                rows, fresh = item.new_rows, (item.rank_view, f"c{self.index}")
            elif item.decision == "augment":
                rows, fresh = torch.cat((self.rows[key], item.new_rows)), (binding.fresh_rank, binding.fresh_chunk_id)
            else:
                rows, fresh = self.rows[key], (binding.fresh_rank, binding.fresh_chunk_id)
            ref = "manifest:" + canonical_digest(["version", self.index, key.to_dict(), item.decision])
            self.bindings[key] = ActiveBasisBinding(ref, int(rows.shape[0]), HEX, *fresh,
                                                    refit_debt=item.refit_debt or 0.0)
            self.rows[key] = rows
        self.index += 1

    def run(self, sources_per_chunk):
        items = []
        for sources in sources_per_chunk:
            plan = self.plan(sources)
            items.append(plan.plans)
            self.advance(plan)
        return items


def stored_energy(source, rows16):
    q = rows16.float()
    residual = source.double() - ((source.float() @ q.T) @ q).double()
    return 1 - float(residual.square().sum() / source.double().square().sum())


class RefitRuleConfigTests(unittest.TestCase):
    def test_default_validation_and_identity(self):
        default = RankMethodConfig(TARGETS, CodecSpec("svd", rank=1))
        self.assertEqual((default.refit_rule, default.refit_debt_on_share, default.refit_hysteresis),
                         ("paper", False, 1.0))  # the paper rule again (user decision 2026-10-17)
        self.assertNotIn("refit_rule", default.to_dict())  # paper-rule methods keep their pre-freeze-D fingerprints
        self.assertNotIn("refit_hysteresis", default.to_dict())
        self.assertEqual(REFIT_RULES, ("cumulative", "paper"))
        for bad in ("Paper", "", "ski", None, 1, ("paper",)):
            with self.subTest(refit_rule=bad), self.assertRaises(ConfigurationError):
                config(refit_rule=bad)
        for bad in (1, "yes", None):
            with self.subTest(refit_debt_on_share=bad), self.assertRaises(ConfigurationError):
                config(refit_debt_on_share=bad)
        with self.assertRaises(ConfigurationError):
            config(refit_rule="paper", refit_debt_on_share=True)
        paper, cumulative = config(refit_rule="paper").to_dict(), config().to_dict()
        on_share = config(refit_debt_on_share=True).to_dict()
        self.assertNotIn("refit_rule", paper)
        self.assertNotIn("refit_debt_on_share", paper)
        self.assertEqual(cumulative["refit_rule"], "cumulative")
        self.assertNotIn("refit_debt_on_share", cumulative)
        self.assertIs(on_share["refit_debt_on_share"], True)
        self.assertEqual({key: value for key, value in cumulative.items() if key != "refit_rule"}, paper)
        prints = {config(refit_rule="paper").fingerprint, config().fingerprint,
                  config(refit_debt_on_share=True).fingerprint}
        self.assertEqual(len(prints), 3)
        # A disabled refit keeps its rule in the identity (records still carry the cumulative debt).
        self.assertNotEqual(config(refit=False).fingerprint, config(refit=False, refit_rule="paper").fingerprint)

    def test_paper_rule_keeps_the_c3_method_fingerprints(self):
        svd = RankMethodConfig(TARGETS, CodecSpec("svd", rank=1), refit_rule="paper")
        hevc = RankMethodConfig(TARGETS, CodecSpec(kind="svd_hevc", rank=1, inter_prediction=False),
                                refit_rule="paper")
        self.assertEqual(svd.fingerprint, C3_PAPER_FINGERPRINTS["svd"])
        self.assertEqual(hevc.fingerprint, C3_PAPER_FINGERPRINTS["svd_hevc"])

    def test_hysteresis_option(self):
        for bad in (0.5, 0.0, 17, math.nan, math.inf, True, "2", None):
            with self.subTest(refit_hysteresis=bad), self.assertRaises(ConfigurationError):
                config(refit_hysteresis=bad)
        self.assertIsInstance(config(refit_hysteresis=2).refit_hysteresis, float)
        # Serialized away from 1 under the cumulative rule: hysteresis 1 has the plain ski-rental identity, and the
        # paper rule ignores it (its fingerprints stay C3's).
        self.assertNotIn("refit_hysteresis", config().to_dict())
        self.assertEqual(config(refit_hysteresis=1.5).to_dict()["refit_hysteresis"], 1.5)
        self.assertEqual(len({config().fingerprint, config(refit_hysteresis=1.5).fingerprint,
                              config(refit_hysteresis=2).fingerprint}), 3)
        for value in (1.0, 2.0, 3.0):
            paper = RankMethodConfig(TARGETS, CodecSpec("svd", rank=1), refit_rule="paper", refit_hysteresis=value)
            self.assertEqual(paper.fingerprint, C3_PAPER_FINGERPRINTS["svd"])


class BindingDebtTests(unittest.TestCase):
    def test_debt_validation_serialization_and_old_records(self):
        plain = ActiveBasisBinding(REF, 32, HEX, 32, "c0", 0.5, HEX)
        self.assertEqual(plain.refit_debt, 0.0)
        self.assertNotIn("refit_debt", plain.to_dict())
        self.assertEqual(set(plain.to_dict()), {"version_ref", "total_rank", "group_id", "fresh_rank",
                                                "fresh_chunk_id", "coefficient_rate", "rate_identity"})
        self.assertEqual(ActiveBasisBinding.from_dict(plain.to_dict()), plain)  # archives without a debt: 0
        owed = ActiveBasisBinding(REF, 32, HEX, 32, "c0", 0.5, HEX, 117234)
        self.assertIsInstance(owed.refit_debt, float)
        self.assertEqual(owed.to_dict()["refit_debt"], 117234.0)
        self.assertEqual(ActiveBasisBinding.from_dict(owed.to_dict()), owed)
        self.assertEqual(ActiveBasisBinding.from_dict(owed.to_dict()).to_dict(), owed.to_dict())
        for bad in (-1.0, math.nan, math.inf, True, "1", None):
            with self.subTest(debt=bad), self.assertRaises(ConfigurationError):
                ActiveBasisBinding(REF, 32, HEX, 32, "c0", refit_debt=bad)
        with self.assertRaises(ConfigurationError):
            ActiveBasisBinding(refit_debt=1.0)  # no active basis
        with self.assertRaises(ConfigurationError):
            ActiveBasisBinding(REF, 32, HEX, refit_debt=1.0)  # no fresh rank to owe against
        with self.assertRaises(ConfigurationError):
            ActiveBasisBinding.from_dict({**plain.to_dict(), "refit_debt": 0.0})  # a zero debt is omitted
        self.assertEqual(ActiveBasisBinding(REF, 32, HEX, 32, "c0", refit_debt=-0.0).to_dict(),
                         ActiveBasisBinding(REF, 32, HEX, 32, "c0").to_dict())

    def test_chunk_nodes_round_trip_their_debts(self):
        key_k, key_v = MatrixKey(0, "K"), MatrixKey(0, "V")
        bindings = ((key_k, ActiveBasisBinding(REF, 32, HEX, 32, "c0")),
                    (key_v, ActiveBasisBinding(REF, 64, HEX, 32, "c0", 0.5, HEX, 5760.5)))
        node = ChunkNode("recache", "r1", REF, "e1", REF, 0, 0, 0, 0, 0, 0, (), bindings)
        data = node.to_dict()
        self.assertNotIn("refit_debt", data["active_basis"][0]["binding"])
        self.assertEqual(data["active_basis"][1]["binding"]["refit_debt"], 5760.5)
        again = ChunkNode.from_dict(data)
        self.assertEqual(again, node)
        self.assertEqual(again.binding(key_v).refit_debt, 5760.5)
        self.assertEqual(canonical_digest(again.to_dict()), canonical_digest(data))


class RefitDebtTestFunctionTests(unittest.TestCase):
    @staticmethod
    def terms(coefficients, basis, scales=0, metadata=4096):
        return {"coefficient_bytes": float(coefficients), "basis_bytes": basis, "scale_bytes": scales,
                "metadata_bytes": metadata, "total_bytes": float(coefficients) + basis + scales + metadata}

    def test_thresholds_and_ties(self):
        fresh = self.terms(17280, 49152, 768)
        candidate = self.terms(17280 + 15360, 16384, 1024)  # excess 15616, cost 32768
        self.assertEqual(refit_debt_test(0.0, candidate, fresh), (False, 15616.0, 32768))
        # With no debt the test is the paper's B_candidate > B_fresh: an exact tie keeps the candidate ...
        tie = self.terms(17280 + 32768 - 256, 16384, 1024)
        self.assertEqual(tie["total_bytes"], fresh["total_bytes"])
        self.assertEqual(refit_debt_test(0.0, tie, fresh), (False, 32768.0, 32768))
        above = self.terms(17280 + 32768 - 255, 16384, 1024)
        self.assertTrue(refit_debt_test(0.0, above, fresh)[0])
        # ... with a debt the chunk refits as soon as debt + excess reaches the cost.
        self.assertTrue(refit_debt_test(32768.0 - 15616.0, candidate, fresh)[0])
        self.assertFalse(refit_debt_test(32768.0 - 15616.0 - 0.5, candidate, fresh)[0])
        self.assertTrue(refit_debt_test(1.0, tie, fresh)[0])
        # A candidate cheaper per chunk than the fresh fit has a negative excess (the planner accrues none).
        small = self.terms(10000, 16384, 512)
        refit, excess, cost = refit_debt_test(5.0, small, fresh)
        self.assertEqual((refit, excess, cost), (False, 10512.0 - 18048.0, 32768))

    def test_chosen_bytes_read_v1_and_v2(self):
        augmented, fresh = self.terms(100, 10), self.terms(50, 40)
        self.assertEqual(chosen_estimate_bytes({"schema": REFIT_ESTIMATE_V1, "decision": "augment",
                                                "augmented": augmented, "fresh": fresh}), augmented["total_bytes"])
        self.assertEqual(chosen_estimate_bytes({"schema": REFIT_ESTIMATE_V1, "decision": "refit",
                                                "augmented": augmented, "fresh": fresh}), fresh["total_bytes"])
        self.assertIsNone(chosen_estimate_bytes({"decision": "refit", "augmented": augmented, "fresh": None}))
        self.assertEqual(chosen_estimate_bytes({"schema": REFIT_ESTIMATE_V2, "decision": "share",
                                                "shared": augmented, "fresh": fresh}), augmented["total_bytes"])

    def test_raised_thresholds(self):
        fresh = self.terms(17280, 49152, 768)
        candidate = self.terms(17280 + 15360, 16384, 1024)  # excess 15616, cost 32768
        # The debt must reach hysteresis * cost - excess; no debt keeps the paper test.
        for hysteresis in (1.5, 2.0, 3.0):
            threshold = hysteresis * 32768
            with self.subTest(hysteresis=hysteresis):
                self.assertTrue(refit_debt_test(threshold - 15616.0, candidate, fresh, hysteresis=hysteresis)[0])
                self.assertFalse(refit_debt_test(threshold - 15616.5, candidate, fresh, hysteresis=hysteresis)[0])
                self.assertFalse(refit_debt_test(0.0, candidate, fresh, hysteresis=hysteresis)[0])
        # A raised threshold never drops a refit the chunk's own bytes call for (the paper test), with or without debt.
        above = self.terms(17280 + 32768 - 255, 16384, 1024)
        for debt in (0.0, 1.0):
            self.assertTrue(refit_debt_test(debt, above, fresh, hysteresis=16.0)[0])
        # Hysteresis 1 is the plain cumulative test.
        for debt in (1.0, 17152.0, 17151.5):
            self.assertEqual(refit_debt_test(debt, candidate, fresh, hysteresis=1.0),
                             refit_debt_test(debt, candidate, fresh))


class CumulativePlannerTests(unittest.TestCase):
    """Planner-level lineages (one or two V matrices) under both rules."""

    def chain(self, cfg, count=8, **kwargs):
        sources = drifting_sources(count, **kwargs)
        return Chain(cfg).run([{V0: source} for source in sources])

    def test_cumulative_refits_where_the_paper_rule_keeps_augmenting(self):
        paper = [items[V0] for items in self.chain(config(refit_rule="paper"))]
        cumulative = [items[V0] for items in self.chain(config())]
        self.assertEqual([item.decision for item in paper],
                         ["fresh", "augment", "augment", "augment", "augment", "share", "share", "share"])
        self.assertEqual([item.decision for item in cumulative],
                         ["fresh", "augment", "augment", "refit", "augment", "augment", "refit", "augment"])
        self.assertEqual([item.rank_view for item in paper], [96, 128, 160, 192, 224, 224, 224, 224])
        self.assertEqual([item.rank_view for item in cumulative], [96, 128, 160, 96, 128, 160, 96, 128])
        for item in paper:
            self.assertIsNone(item.refit_debt)
            self.assertNotIn("refit_debt", item.summary())
            if item.estimate is not None:
                self.assertEqual(item.estimate["schema"], REFIT_ESTIMATE_V1)
                self.assertEqual(item.estimate["decision"], "augment")
        # Exact debts: 180 bytes per coefficient rank, a 32-row augmentation of a 96-row fresh fit costs 32768.
        debts = []
        for item in cumulative[1:]:
            estimate = item.estimate
            self.assertEqual(estimate["schema"], REFIT_ESTIMATE_V2)
            self.assertEqual(estimate["rule"], "cumulative")
            self.assertEqual(estimate["cost"], 2 * CHANNELS * (estimate["r_fresh"] - estimate["r_added"]))
            self.assertEqual(estimate["excess"], 180.0 * (estimate["r_parent"] + estimate["r_added"]
                                                          - estimate["r_fresh"]))
            reaches = estimate["debt_before"] + estimate["excess"] >= estimate["cost"]
            self.assertEqual(item.decision == "refit", reaches)
            self.assertEqual(estimate["debt_after"], 0.0 if reaches else estimate["debt_before"]
                             + max(0.0, estimate["excess"]))
            self.assertEqual(item.refit_debt, estimate["debt_after"])
            self.assertEqual(item.summary()["refit_debt"], estimate["debt_after"])
            debts.append((estimate["debt_before"], estimate["debt_after"], estimate["reason"]))
        self.assertEqual(debts[:3], [(0.0, 5760.0, "augmented_bytes_within_fresh"),
                                     (5760.0, 17280.0, "refit_debt_below_cost"),
                                     (17280.0, 0.0, "refit_debt_reaches_cost")])
        self.assertEqual(cumulative[3].reason, "refit_debt_reaches_cost")
        self.assertEqual(cumulative[3].fresh_rank, cumulative[3].rank_view)
        self.assertEqual(cumulative[3].work["counters"]["refit_count"], 1)
        self.assertEqual(cumulative[3].work["counters"]["residual_fit_count"], 1)  # the verified augmentation
        self.assertEqual(cumulative[0].refit_debt, 0.0)
        # The paper's own estimate at chunk 3 still prefers the augmentation (55040 < 70528 bytes).
        self.assertLess(cumulative[3].estimate["augmented"]["total_bytes"],
                        cumulative[3].estimate["fresh"]["total_bytes"])
        self.assertEqual(chosen_estimate_bytes(cumulative[3].estimate), cumulative[3].estimate["fresh"]["total_bytes"])

    def test_no_debt_reproduces_the_paper_decision_bitwise(self):
        sources = drifting_sources(3)
        for refit in (True, False):
            chains = {rule: Chain(config(refit_rule=rule, refit=refit)) for rule in REFIT_RULES}
            for index, source in enumerate(sources):
                plans = {rule: chain.plan({V0: source}).plans[V0] for rule, chain in chains.items()}
                paper, cumulative = plans["paper"], plans["cumulative"]
                with self.subTest(refit=refit, chunk=index):
                    for name in ("decision", "reason", "rank_view", "bucket_rank", "observed_rank",
                                 "decision_energy", "observations", "predictions", "fresh_rank"):
                        self.assertEqual(getattr(paper, name), getattr(cumulative, name), name)
                    self.assertEqual(paper.work, cumulative.work)
                    if paper.new_rows is None:
                        self.assertIsNone(cumulative.new_rows)
                    else:
                        self.assertTrue(torch.equal(paper.new_rows, cumulative.new_rows))
                    if paper.estimate is None:
                        self.assertIsNone(cumulative.estimate)
                    else:
                        for name in ("r_parent", "r_added", "r_fresh", "fresh_rank_source", "augmented", "fresh",
                                     "decision", "reason"):
                            self.assertEqual(paper.estimate[name], cumulative.estimate[name], name)
                        self.assertEqual(cumulative.estimate["debt_before"], 0.0)
                for chain in chains.values():
                    chain.advance(type("Plan", (), {"plans": {V0: plans["paper"]}})())  # identical states

    def test_step4_refit_and_augment_cases_agree_at_zero_debt(self):
        from tests import test_step4_rank_method as s4
        parent = s4.plan({s4.KEYS[0]: s4.matrix(1, s4.orthonormal(80))})
        bindings, parents = s4.active(parent.plans)
        source = s4.matrix(2, s4.orthonormal(81))
        for kwargs in ({}, {"coefficient_bytes_proxy": 0.01}):
            plans = {rule: s4.plan({s4.KEYS[0]: source}, cfg=s4.config(refit_rule=rule, **kwargs), bindings=bindings,
                                   parents=parents, chunk="c1").plans[s4.KEYS[0]] for rule in REFIT_RULES}
            with self.subTest(**kwargs):
                self.assertEqual(plans["paper"].decision, plans["cumulative"].decision)
                self.assertEqual(plans["paper"].reason, plans["cumulative"].reason)
                self.assertEqual(plans["paper"].rank_view, plans["cumulative"].rank_view)
                self.assertTrue(torch.equal(plans["paper"].new_rows, plans["cumulative"].new_rows))
        self.assertEqual(plans["paper"].decision, "augment")  # the cheap proxy augments; the default refits

    def chain_at(self, cfg, chunks, **kwargs):
        """A chain advanced through ``chunks`` chunks, and the next chunk's source."""
        sources = drifting_sources(chunks + 1, **kwargs)
        chain = Chain(cfg)
        chain.run([{V0: source} for source in sources[:chunks]])
        return chain, sources[chunks]

    def test_threshold_edges_with_an_inherited_debt(self):
        chain, source = self.chain_at(config(), 3, moves=None)
        # Chunk 3 inherits the paper-equivalent state of chunk 2 with its own debt replaced below.
        binding = chain.bindings[V0]
        self.assertEqual(binding.refit_debt, 17280.0)
        excess, cost = 180.0 * (160 + 32 - 96), 32768
        cases = ((cost - excess, "refit", "refit_debt_reaches_cost", 0.0),
                 (cost - excess - 1, "augment", "refit_debt_below_cost", cost - 1),
                 (cost - excess - 0.25, "augment", "refit_debt_below_cost", cost - 0.25),
                 (cost + 1.0, "refit", "refit_debt_reaches_cost", 0.0))
        for debt, decision, reason, after in cases:
            item = chain.plan({V0: source}, bindings={V0: replace(binding, refit_debt=debt)}).plans[V0]
            with self.subTest(debt=debt):
                self.assertEqual((item.decision, item.estimate["reason"], item.refit_debt), (decision, reason, after))
                self.assertEqual((item.estimate["excess"], item.estimate["cost"]), (excess, cost))
                self.assertEqual(item.estimate["debt_before"], debt)
        disabled = chain.plan({V0: source}, bindings={V0: replace(binding, refit_debt=cost)},
                              cfg=config(refit=False)).plans[V0]
        self.assertEqual((disabled.decision, disabled.estimate["reason"]), ("augment", "refit_disabled"))
        self.assertEqual(disabled.refit_debt, cost + excess)  # a disabled refit still accrues

    def test_hysteresis_raises_the_threshold(self):
        chain, source = self.chain_at(config(), 3, moves=None)
        binding = chain.bindings[V0]
        excess, cost = 180.0 * (160 + 32 - 96), 32768
        plain = chain.plan({V0: source}).plans[V0]
        for name in ("hysteresis", "threshold"):
            self.assertNotIn(name, plain.estimate)  # hysteresis 1 documents are the plain rule's
        for hysteresis in (1.5, 2.0):
            threshold = hysteresis * cost
            for debt, decision in ((threshold - excess, "refit"), (threshold - excess - 1, "augment")):
                item = chain.plan({V0: source}, bindings={V0: replace(binding, refit_debt=debt)},
                                  cfg=config(refit_hysteresis=hysteresis)).plans[V0]
                with self.subTest(hysteresis=hysteresis, debt=debt):
                    self.assertEqual(item.decision, decision)
                    self.assertEqual((item.estimate["hysteresis"], item.estimate["threshold"]), (hysteresis, threshold))
                    self.assertEqual(item.refit_debt, 0.0 if decision == "refit" else debt + excess)
                    self.assertEqual(item.estimate["reason"], "refit_debt_reaches_cost" if decision == "refit"
                                     else "refit_debt_below_cost")
        # Over a longer lineage a higher hysteresis refits later and less often (the default is 1).
        sources = [{V0: item} for item in drifting_sources(12)]
        runs = {value: [plan[V0].decision for plan in Chain(config(refit_hysteresis=value)).run(sources)]
                for value in (1.0, 1.5, 2.0)}
        self.assertEqual({value: decisions.index("refit") for value, decisions in runs.items()},
                         {1.0: 3, 1.5: 4, 2.0: 8})
        self.assertEqual([runs[value].count("refit") for value in (1.0, 1.5, 2.0)], [3, 2, 1])
        default = RankMethodConfig(TARGETS, CodecSpec("svd", rank=1), coefficient_bytes_proxy=PROXY,
                                   refit_rule="cumulative")  # the cumulative rule's default hysteresis
        self.assertEqual([plan[V0].decision for plan in Chain(default).run(sources)], runs[1.0])

    def test_missing_fresh_rank_behaves_as_today(self):
        chain, source = self.chain_at(config(), 1)
        binding = replace(chain.bindings[V0], fresh_rank=None, fresh_chunk_id=None)
        items = {rule: chain.plan({V0: source}, bindings={V0: binding}, cfg=config(refit_rule=rule)).plans[V0]
                 for rule in REFIT_RULES}
        for rule, item in items.items():
            with self.subTest(rule=rule):
                self.assertEqual((item.decision, item.estimate["reason"]), ("augment", "no_fresh_rank_estimate"))
                self.assertIsNone(item.estimate["fresh"])
        self.assertTrue(torch.equal(items["paper"].new_rows, items["cumulative"].new_rows))
        estimate = items["cumulative"].estimate
        self.assertEqual((estimate["excess"], estimate["cost"], estimate["debt_after"]), (None, None, 0.0))

    def test_shares_keep_the_debt_and_forced_refits_reset_it(self):
        items = [item[V0] for item in self.chain(config(), 7, moves=4)]
        self.assertEqual([item.decision for item in items],
                         ["fresh", "augment", "augment", "refit", "augment", "share", "share"])
        self.assertEqual([item.refit_debt for item in items], [0.0, 5760.0, 17280.0, 0.0, 5760.0, 5760.0, 5760.0])
        for item in items[5:]:
            self.assertIsNone(item.estimate)
            self.assertEqual(item.summary()["refit_debt"], 5760.0)
        # A refit forced by a failed augmentation (here: residual directions inside the inherited span) resets the
        # debt like any refit; it has no augmentation estimate.
        chain, source = self.chain_at(config(), 2, moves=None)
        self.assertEqual(chain.bindings[V0].refit_debt, 5760.0)
        original = LatestMethodPlanner._orthogonalize_reference

        def inside(planner, work, parent, directions, limit):
            return original(planner, work, parent, directions, limit)[0], 0

        with mock.patch.object(LatestMethodPlanner, "_orthogonalize_reference", inside):
            item = chain.plan({V0: source}).plans[V0]
        self.assertEqual((item.decision, item.reason, item.refit_debt, item.estimate),
                         ("refit", "residual_directions_inside_inherited_span", 0.0, None))

    def test_debt_on_share_accrues_and_refits_a_shared_chunk(self):
        cfg = config(refit_debt_on_share=True)
        chain, source = self.chain_at(cfg, 5, moves=4)
        binding = chain.bindings[V0]
        self.assertEqual((binding.total_rank, binding.fresh_rank, binding.refit_debt), (128, 96, 5760.0))
        item = chain.plan({V0: source}).plans[V0]
        self.assertEqual((item.decision, item.rank_view), ("share", 128))
        estimate = item.estimate
        self.assertEqual(estimate["schema"], REFIT_ESTIMATE_V2)
        self.assertEqual((estimate["r_view"], estimate["r_total"], estimate["excess"], estimate["cost"]),
                         (128, 128, 5760.0, 2 * CHANNELS * 96))
        self.assertEqual((estimate["debt_before"], estimate["debt_after"], item.refit_debt),
                         (5760.0, 11520.0, 11520.0))
        self.assertEqual(chosen_estimate_bytes(estimate), estimate["shared"]["total_bytes"])
        self.assertEqual(estimate["reason"], "refit_debt_below_cost")
        # Without the option the share leaves the debt unchanged.
        plain = chain.plan({V0: source}, cfg=config()).plans[V0]
        self.assertEqual((plain.decision, plain.refit_debt, plain.estimate), ("share", 5760.0, None))
        # At the threshold the shared chunk refits: one fresh fit, no residual fit, a verified fresh basis.
        cost = 2 * CHANNELS * 96
        for debt, decision in ((cost - 5760.0, "refit"), (cost - 5760.5, "share")):
            item = chain.plan({V0: source}, bindings={V0: replace(binding, refit_debt=debt)}).plans[V0]
            with self.subTest(debt=debt):
                self.assertEqual(item.decision, decision)
        self.assertEqual(item.refit_debt, cost - 0.5)
        item = chain.plan({V0: source}, bindings={V0: replace(binding, refit_debt=cost - 5760.0)}).plans[V0]
        counters = item.work["counters"]
        self.assertEqual((item.reason, item.estimate["decision"], item.refit_debt),
                         ("refit_debt_reaches_cost", "refit", 0.0))
        self.assertEqual((counters["refit_count"], counters["candidate_fit_count"], counters["residual_fit_count"]),
                         (1, 1, 0))
        self.assertIsNone(item.parent_version_ref)
        self.assertEqual(item.fresh_rank, item.rank_view)
        self.assertGreaterEqual(stored_energy(source, item.new_rows), 0.95)
        self.assertEqual(chosen_estimate_bytes(item.estimate), item.estimate["fresh"]["total_bytes"])

    def test_shared_and_augmented_refits_are_fitted_in_one_batch(self):
        cfg = config(refit_debt_on_share=True)
        sources = drifting_sources(6, moves=4)
        chain = Chain(cfg, keys=(V0, V1))
        # V1 drifts every chunk (augment / refit); V0 stops drifting after chunk 4 (shares from chunk 5).
        moving = drifting_sources(6, moves=None)  # chunks 0-4 equal V0's
        chain.run([{V0: sources[index], V1: moving[index]} for index in range(5)])
        cost = 2 * CHANNELS * 96
        bindings = dict(chain.bindings)
        bindings[V0] = replace(bindings[V0], refit_debt=cost)
        bindings[V1] = replace(bindings[V1], refit_debt=cost)
        calls = []
        original = LatestMethodPlanner._fit_jobs

        def record(planner, clock, fresh, residual, chunk_id, *, refit):
            calls.append((refit, [work.key for work in fresh], [work.key for work in residual]))
            return original(planner, clock, fresh, residual, chunk_id, refit=refit)

        with mock.patch.object(LatestMethodPlanner, "_fit_jobs", record):
            plan = chain.plan({V0: sources[5], V1: moving[5]}, bindings=bindings)
        self.assertEqual({key: item.decision for key, item in plan.plans.items()}, {V0: "refit", V1: "refit"})
        self.assertEqual(plan.plans[V0].estimate["r_view"], 128)
        self.assertIn("r_added", plan.plans[V1].estimate)
        self.assertEqual(calls, [(False, [], [V0, V1]), (True, [V0, V1], [])])
        self.assertEqual(plan.counters["decision_refit"], 2)
        self.assertEqual(plan.counters["refit_count"], 2)


class CumulativeLineageTests(unittest.TestCase):
    """Node-packed lineages (Step 6 fixture): bindings, records, forks and reopened runtimes."""

    def setUp(self):
        self.runtimes = RuntimeFactory(self)

    def commit_chain(self, runtime, start, count, reg, *, branch="main", cfg=None):
        """Commits under ``cfg``, by default the cumulative rule (the core default is the paper rule again)."""
        refs = []
        for index in range(start, start + count):
            matrices, contexts = chunk(index)
            refs.append(runtime.commit_node(branch, f"chunk-{index:03d}", matrices, contexts,
                                            method=cfg or replace(method(), refit_rule="cumulative"),
                                            registry=reg).node_ref)
        return refs

    def check_records(self, runtime, refs):
        """Every record states the debt its active binding carries; returns the largest debt seen."""
        largest = 0.0
        for parent_ref, ref in zip(refs, refs[1:]):
            parent, node = runtime.load_node(parent_ref), runtime.load_node(ref)
            for entry, record_ref in node.matrix_records:
                decision = runtime.load_record(record_ref).decision
                before, after = parent.binding(entry), node.binding(entry)
                self.assertEqual(after.refit_debt, decision["refit_debt"])
                estimate = decision["estimate"]
                if decision["decision"] in ("fresh", "refit"):
                    self.assertEqual(after.refit_debt, 0.0)
                elif decision["decision"] == "share":
                    self.assertEqual(after.refit_debt, before.refit_debt)
                if estimate is not None:
                    self.assertEqual(estimate["schema"], REFIT_ESTIMATE_V2)
                    self.assertEqual(estimate["debt_before"], before.refit_debt)
                    self.assertEqual(estimate["debt_after"], after.refit_debt)
                largest = max(largest, after.refit_debt)
        return largest

    def test_bindings_and_records_carry_the_debt(self):
        runtime = self.runtimes("debt")
        runtime.create_lineage("main")
        root = runtime.lineage_head("main")
        refs = [root] + self.commit_chain(runtime, 0, 4, step6_registry())
        self.assertGreater(self.check_records(runtime, refs), 0.0)
        head = runtime.load_node(refs[-1])
        owed = [entry for entry, binding in head.active_basis if binding.refit_debt]
        self.assertTrue(owed)
        stored = head.to_dict()["active_basis"]
        self.assertTrue(any("refit_debt" in item["binding"] for item in stored))
        self.assertTrue(any("refit_debt" not in item["binding"] for item in stored))

    def test_forks_and_reopened_runtimes_continue_with_the_parent_debt(self):
        straight = self.runtimes("straight")
        straight.create_lineage("main")
        expected = [node_content(straight, ref) for ref in self.commit_chain(straight, 0, 5, step6_registry())]
        runtime = self.runtimes("forked")
        runtime.create_lineage("main")
        reg = step6_registry()
        head = self.commit_chain(runtime, 0, 3, reg)[-1]
        debts = {entry: binding.refit_debt for entry, binding in runtime.load_node(head).active_basis}
        self.assertTrue(any(debts.values()))
        root = runtime.store.root
        runtime.close()
        reopened = KVRuntime(root, STEP6_MODEL)
        self.addCleanup(reopened.close)
        reopened.fork_lineage(head, "fork")
        refs = self.commit_chain(reopened, 3, 2, reg, branch="fork")
        self.assertEqual([node_content(reopened, ref) for ref in refs], expected[3:])
        node = reopened.load_node(refs[0])
        for entry, record_ref in node.matrix_records:
            decision = reopened.load_record(record_ref).decision
            if decision["decision"] == "share":
                self.assertEqual(decision["refit_debt"], debts[entry])
            if decision["estimate"] is not None:
                self.assertEqual(decision["estimate"]["debt_before"], debts[entry])

    def test_paper_lineages_write_no_debt(self):
        runtime = self.runtimes("paper")
        runtime.create_lineage("main")
        cfg = replace(method(), refit_rule="paper")
        refs = self.commit_chain(runtime, 0, 3, step6_registry(), cfg=cfg)
        cumulative = self.runtimes("cumulative")
        cumulative.create_lineage("main")
        others = self.commit_chain(cumulative, 0, 3, step6_registry())
        for ref, other in zip(refs, others):
            node = runtime.load_node(ref)
            for _, binding in node.active_basis:
                self.assertEqual(binding.refit_debt, 0.0)
                self.assertNotIn("refit_debt", binding.to_dict())
            for entry, record_ref in node.matrix_records:
                decision = runtime.load_record(record_ref).decision
                self.assertNotIn("refit_debt", decision)
                if decision["estimate"] is not None:
                    self.assertEqual(decision["estimate"]["schema"], REFIT_ESTIMATE_V1)
            # Same decisions, ranks and packet bytes as the cumulative rule on this short lineage (no debt is due).
            mine, theirs = node_content(runtime, ref), node_content(cumulative, other)
            for entry, row in mine["rows"].items():
                self.assertEqual(row[:2], theirs["rows"][entry][:2])
                self.assertEqual(row[5], theirs["rows"][entry][5])

    def test_archives_without_debts_continue_from_zero(self):
        """Nodes written without a debt (freeze C3 and earlier, the paper rule) read as debt 0 when the cumulative rule
        continues their lineage."""
        runtime = self.runtimes("old")
        runtime.create_lineage("main")
        reg = step6_registry()
        refs = self.commit_chain(runtime, 0, 3, reg, cfg=replace(method(), refit_rule="paper"))
        head = runtime.load_node(refs[-1])
        self.assertFalse(any(binding.refit_debt for _, binding in head.active_basis))
        self.assertFalse(any("refit_debt" in item["binding"] for item in head.to_dict()["active_basis"]))
        more = self.commit_chain(runtime, 3, 2, reg)
        node = runtime.load_node(more[0])
        for entry, record_ref in node.matrix_records:
            decision = runtime.load_record(record_ref).decision
            if decision["estimate"] is not None:
                self.assertEqual(decision["estimate"]["debt_before"], 0.0)
            if decision["decision"] == "share":
                self.assertEqual(decision["refit_debt"], 0.0)
        self.check_records(runtime, refs[-1:] + more)


if __name__ == "__main__":
    unittest.main()
