"""Session-tree workload generator (workload v2, ``reelcache.session_workloads``): CPU tests on a fixture dataset.

Covers plan section 0.7 "Generator CPU tests": request count = new nodes; k=0 is the latest node by generation
index (not depth); age across old branches (the A-F example); uniform session choice (chi-square); infeasible
settings raise and nothing falls back or snaps; equal seeds give byte-identical bundles; trees do not change with
rate, burstiness, session count or arrival mode; legal-support conditioning with min_switch_depth / max_depth;
closed-loop determinism; up-front grid validation; content binding and bank validation; the example CLI.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from reelcache import session_workloads as sw
from reelcache.errors import ConfigurationError
from reelcache.synthetic_workloads import HISTORY_LENGTHS

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("forest wildlife", "heritage pottery", "museum geology", "garden pond observation", "tea_preparation",
              "sports", "fantasy magic", "transit")
# chi-square critical values at alpha = 0.001
CHI2_CRIT_001 = {7: 24.322, 9: 27.877}
# S per model from the coordinator's Step 19 RTX hint (03:30): tests use the CausVid table unless they test S itself
CAUSVID_S = {"by_depth": sw.SERVICE_TIME_HINTS_S["causvid"]}


def fixture_session(index: int) -> dict:
    """Explicitly templated fixture (never benchmark content): 20 main prompts of 21-30 words, five forks."""
    sid = f"s{index:04d}"
    extra = " ".join(["softly"] * (index % 10))
    prompts = [f"Fixture {sid} main prompt {p}: a copper toy bird in a miniature paper garden unfolds its wings "
               f"beside the violet bench {extra}".strip() for p in range(1, 21)]
    forks = [{"history_prompt_count": h, "change_type": "action",
              "prompts": [f"Fixture {sid} fork {h} step {i}: the copper toy bird inspects a fallen feather beside "
                          "the violet bench in the paper garden" for i in range(2)]} for h in HISTORY_LENGTHS]
    return {"session_id": sid, "title": f"Fixture title {index}", "category": CATEGORIES[index % len(CATEGORIES)],
            "prompts": prompts, "forks": forks}


def make_dataset(root: Path, sessions: int = 50, development: int = 10) -> str:
    root.mkdir(parents=True)
    with open(root / "sessions.jsonl", "w") as handle:
        for index in range(sessions):
            handle.write(json.dumps(fixture_session(index)) + "\n")
    ids = [f"s{i:04d}" for i in range(sessions)]
    (root / "splits.json").write_text(json.dumps({"development": ids[:development],
                                                  "evaluation_candidate": ids[development:]}))
    return str(root)


def valid_text(key: str, salt: str = "") -> str:
    """A 22-word authored-looking fixture continuation, unique per key."""
    return (f"Authored fixture continuation {key[:12]} {salt}: the copper toy bird lifts a paper lantern while "
            "the violet bench glows in the quiet garden").replace("  ", " ")


def plan_signature(workload: sw.Workload) -> dict:
    """Per-session tree content of a workload (no timing fields)."""
    fields = ("ordinal", "source_generation_index", "k", "output_generation_index", "prompt_id", "path_prompt_ids",
              "legal_ages", "support", "child_ordinal")
    out: dict = {}
    for request in workload.requests:
        out.setdefault(request["session_id"], []).append(tuple(json.dumps(request[f]) for f in fields))
    return {sid: sorted(rows) for sid, rows in out.items()}


class SessionWorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dataset = make_dataset(Path(cls.tmp.name) / "dataset")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setting(self, **overrides) -> sw.WorkloadSetting:
        values = {"dataset": self.dataset, "service_time_s": CAUSVID_S, "target_model": "causvid", "n_sessions": 4,
                  "budget": 10, **overrides}
        return sw.WorkloadSetting(**values)

    def build(self, **overrides) -> sw.Workload:
        return sw.build_workload(self.setting(**overrides))

    # ---------------------------------------------------------------------------------------------- counting
    def test_request_count_equals_new_nodes(self):
        for p in (0.0, 0.5, 1.0):
            with self.subTest(p_historical=p):
                workload = self.build(p_historical=p, n_sessions=5, budget=7)
                coverage = workload.coverage
                self.assertEqual(len(workload.requests), 35)
                self.assertEqual(coverage["count_equality"], {"requests": 35, "new_nodes": 35, "equal": True,
                                                              "expected": 35})
                for sid, tree in workload.trees.items():
                    self.assertEqual(len(tree.nodes), 8 + 7)
                    outputs = sorted(r["output_generation_index"] for r in workload.requests if r["session_id"] == sid)
                    self.assertEqual(outputs, list(range(9, 16)))
                    for request in workload.requests:
                        if request["session_id"] == sid:
                            node = tree.nodes[request["output_generation_index"]]
                            self.assertEqual(node.parent, request["source_generation_index"])
                            self.assertEqual(node.depth, request["target_depth"] + 1)
                            self.assertEqual(request["total_chunks"], request["target_depth"] + 1)
                            self.assertEqual(request["new_chunks"], 1)

    # -------------------------------------------------------------------------------------------- node age
    def test_a_to_f_example_age_across_old_branches(self):
        tree = sw.SessionTree("s0001", 20)
        for gen in range(1, 5):  # A[1] -> B[2] -> C[3] -> D[4]
            tree.append(gen - 1 if gen > 1 else None, "setup")
        args = {"min_switch_depth": 1, "max_depth": None}
        self.assertEqual(tree.legal_ages(1, None), [1, 2, 3])
        target, e = sw.rewind_and_generate(tree, 2, created_by="r1", **args)  # k=2 from latest D targets B
        self.assertEqual((target.generation_index, e.generation_index, e.parent), (2, 5, 2))
        self.assertEqual(tree.latest, 5)  # E is latest although D is deeper
        self.assertGreater(tree.nodes[4].depth, e.depth)
        self.assertEqual(tree.legal_ages(1, None), [1, 2, 3, 4])
        target, f = sw.rewind_and_generate(tree, 1, created_by="r2", **args)  # k=1 from latest E targets D
        self.assertEqual((target.generation_index, f.generation_index, f.parent), (4, 6, 4))
        self.assertFalse(tree.is_ancestor_or_self(4, 5))  # D is not an ancestor of E: an old-branch target
        target, g = sw.rewind_and_generate(tree, 0, created_by="r3", **args)  # k=0 is F, the newest node
        self.assertEqual((target.generation_index, g.parent), (6, 6))
        self.assertEqual(sorted(tree.nodes), list(range(1, 8)))  # every node retained
        self.assertEqual(tree.nodes[2].children, [3, 5])
        self.assertEqual(tree.nodes[4].children, [6])
        # E is B's second child: an authored branch continuation keyed by B's path and ordinal 2
        self.assertEqual((e.prompt_source, e.prompt_kind), ("bank", "branch"))
        self.assertEqual(e.prompt_key, sw.slot_key("s0001", tree.nodes[2].path_prompt_ids, 2))
        with self.assertRaises(sw.InfeasibleError):
            sw.rewind_and_generate(tree, 7, created_by="r4", **args)  # no node of age 7

    def test_k0_is_latest_by_generation_index_not_depth(self):
        workload = self.build(p_historical=0.5, n_sessions=6, budget=16)
        found = 0
        for sid, tree in workload.trees.items():
            for request in (r for r in workload.requests if r["session_id"] == sid):
                n = request["latest_generation_index"]
                if request["k"] == 0:
                    self.assertEqual(request["source_generation_index"], n)
                    self.assertEqual(request["planned_action"], "resume")
                    deepest = max(tree.nodes[g].depth for g in range(1, n + 1))
                    if tree.nodes[n].depth < deepest:
                        found += 1  # latest is shallower than an older node, and k=0 still targets latest
                else:
                    self.assertEqual(request["source_generation_index"], n - request["k"])
                    self.assertEqual(request["planned_action"], "rewind")
        self.assertGreater(found, 0)

    # ------------------------------------------------------------------------------------------ uniformity
    def test_session_choice_uniform_chi_square(self):
        sessions = [f"s{i:04d}" for i in range(8)]
        times = [float(i) for i in range(1, 8001)]
        draws = sw.Uniforms(11, "session_choice")
        assignment = sw.assign_open_loop(times, sessions, 10 ** 6, lambda s, r: 0.0, draws.next)
        counts = [sum(1 for sid, _ in assignment if sid == s) for s in sessions]
        expected = len(times) / len(sessions)
        chi2 = sum((c - expected) ** 2 / expected for c in counts)
        self.assertLess(chi2, CHI2_CRIT_001[7], counts)
        # ordinals are consecutive per session
        seen: dict = {}
        for sid, ordinal in assignment:
            seen[sid] = seen.get(sid, 0) + 1
            self.assertEqual(ordinal, seen[sid])

    def test_uniform_gap_profile_chi_square_over_legal_support(self):
        support, probabilities = sw.gap_distribution("uniform", range(1, 11))
        self.assertEqual(support, list(range(1, 11)))
        counts = {age: 0 for age in support}
        for index in range(10000):
            counts[sw.choose_age(support, probabilities, sw.Uniforms(5, "targeting", 1, index).next())] += 1
        chi2 = sum((c - 1000) ** 2 / 1000 for c in counts.values())
        self.assertLess(chi2, CHI2_CRIT_001[9], counts)

    def test_closed_loop_schedule_determinism(self):
        closed = self.build(arrival_mode="closed", think_time_mult=4.0, p_historical=0.5)
        again = self.build(arrival_mode="closed", think_time_mult=4.0, p_historical=0.5)
        self.assertEqual(sw.canonical_json(closed.requests), sw.canonical_json(again.requests))
        other_load = self.build(arrival_mode="closed", think_time_mult=4.0, p_historical=0.5, rho=0.3, arrival_cv=4)
        self.assertEqual([r["think_s"] for r in closed.requests], [r["think_s"] for r in other_load.requests])
        think_mean = closed.config["derived"]["think_mean_s"]
        self.assertAlmostEqual(think_mean, 4.0 * closed.config["derived"]["service_time_s"], places=12)
        for request in closed.requests:
            self.assertIsNone(request["t_intended_s"])
            self.assertEqual(request["think_s"], sw.think_seconds(0, request["session_id"], request["ordinal"],
                                                                  think_mean, 1.0))
            self.assertGreater(request["think_s"], 0)
        # nested cohort: the same (session, ordinal) keeps its keyed think draw. With a depth-dependent S the scale
        # (think_time_mult x S, S = mean S(h) over the bundle's target depths) follows the cohort; a numeric S keeps
        # the think times themselves.
        bigger = self.build(arrival_mode="closed", think_time_mult=4.0, p_historical=0.5, n_sessions=7)
        scale = bigger.config["derived"]["think_mean_s"] / think_mean
        think = {r["request_id"]: r["think_s"] for r in bigger.requests}
        for request in closed.requests:
            self.assertAlmostEqual(think[request["request_id"]], request["think_s"] * scale, places=9)
        fixed = [self.build(arrival_mode="closed", think_time_mult=4.0, p_historical=0.5, n_sessions=n,
                            service_time_s=3.3) for n in (4, 7)]
        think = {r["request_id"]: r["think_s"] for r in fixed[1].requests}
        for request in fixed[0].requests:
            self.assertEqual(think[request["request_id"]], request["think_s"])


class FixtureCase(unittest.TestCase):
    """Shared fixture dataset and helpers (no tests of its own)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dataset = make_dataset(Path(cls.tmp.name) / "dataset")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    setting = SessionWorkloadTests.setting
    build = SessionWorkloadTests.build


class FeasibilityTests(FixtureCase):
    def assertInfeasible(self, code, **overrides):
        with self.assertRaises(sw.InfeasibleError) as caught:
            self.build(**overrides)
        self.assertEqual(caught.exception.code, code, str(caught.exception))
        return caught.exception

    def test_infeasible_settings_raise(self):
        self.assertInfeasible("empty_support", d0=5, p_historical=1.0)  # LongLive 1.0: only depth >= 5 is legal
        self.assertInfeasible("latest_illegal", d0=4)
        self.assertInfeasible("fixed_k_illegal", d0=8, p_historical=0.5, gap_profile="fixed", gap_k=4)
        self.assertInfeasible("think_ge_spacing", think_time_mult=100.0)  # cooldown 200 s >= N / lambda = 10 s
        self.assertInfeasible("not_enough_sessions", n_sessions=41)
        bounded = {**sw.DEFAULT_MODELS, "longlive2": {"min_switch_depth": 1, "max_depth": 12}}
        self.assertInfeasible("max_depth", models=bounded, budget=5)  # latest-only reaches depth 13 > 12
        with self.assertRaises(sw.InfeasibleError) as caught:
            sw.assign_open_loop([1.0, 2.0, 3.0], ["s0001", "s0002"], 5, lambda s, r: 100.0,
                                sw.Uniforms(0, "session_choice").next)
        self.assertEqual((caught.exception.code, caught.exception.details["arrival_index"]),
                         ("no_eligible_session", 3))
        with self.assertRaises(ConfigurationError):
            self.setting(gap_profile="fixed")  # fixed needs gap_k
        with self.assertRaises(ConfigurationError):
            self.setting(p_historical=1.5)

    def test_no_fallback_or_snapping(self):
        fixed = self.build(d0=8, p_historical=1.0, gap_profile="fixed", gap_k=2)
        self.assertEqual({r["k"] for r in fixed.requests}, {2})
        far = self.build(p_historical=1.0, gap_profile="far", budget=12)
        recent = self.build(p_historical=1.0, gap_profile="recent", budget=12)
        for request in far.requests:
            legal = request["legal_ages"]
            self.assertEqual(request["support"], legal[len(legal) // 2:])
            self.assertIn(request["k"], request["support"])
        for request in recent.requests:
            self.assertEqual(request["support"], request["legal_ages"])
            weights = [0.5 ** (a - 1) for a in request["support"]]
            for p, w in zip(request["support_probabilities"], weights):
                self.assertAlmostEqual(p, w / sum(weights), places=12)
        with self.assertRaises(sw.InfeasibleError) as caught:
            sw.gap_distribution("fixed", [1, 3], gap_k=2)  # never snapped to 1 or 3
        self.assertEqual(caught.exception.code, "fixed_k_illegal")
        with self.assertRaises(sw.InfeasibleError):
            sw.gap_distribution("uniform", [])
        # a fixed k that becomes illegal mid-run (max_depth) raises at that request instead of choosing another age
        bounded = {**sw.DEFAULT_MODELS, "raven": {"min_switch_depth": 1, "max_depth": 10}}
        error = self.assertInfeasible("fixed_k_illegal", models=bounded, p_historical=1.0, gap_profile="fixed",
                                      gap_k=1, budget=8)
        self.assertTrue(error.details["request_id"].endswith("-r006"), error.details)

    def test_legal_support_conditioning_with_min_switch_depth(self):
        workload = self.build(d0=8, p_historical=1.0, budget=12)
        for sid, tree in workload.trees.items():
            requests = [r for r in workload.requests if r["session_id"] == sid]
            self.assertEqual(requests[0]["legal_ages"], [1, 2, 3])  # nodes 7, 6, 5: depth >= 5
            for request in requests:
                n = request["latest_generation_index"]
                expected = [a for a in range(1, n) if tree.nodes[n - a].depth >= 5]
                self.assertEqual(request["legal_ages"], expected)
                self.assertTrue(set(request["support"]) <= set(expected))
                self.assertGreaterEqual(request["target_depth"], 5)
                self.assertAlmostEqual(sum(request["support_probabilities"]), 1.0, places=12)
        loose = {name: {"min_switch_depth": 1, "max_depth": None} for name in sw.DEFAULT_MODELS}
        workload = self.build(d0=8, p_historical=1.0, budget=3, models=loose)
        self.assertEqual(workload.requests[0]["legal_ages"], list(range(1, 8)))
        strict = self.setting(models={"x": {"min_switch_depth": 2, "max_depth": 40},
                                      "y": {"min_switch_depth": 4, "max_depth": 30}})
        record = strict.constraint_record()
        self.assertEqual((strict.min_switch_depth, strict.max_depth), (4, 30))
        self.assertEqual((record["min_switch_depth_from"], record["max_depth_from"]), (["y"], ["y"]))

    def test_trees_invariant_to_load_burstiness_sessions_and_mode(self):
        base = self.build(p_historical=0.5, n_sessions=6, budget=12)
        signature = plan_signature(base)
        variants = {"rho": {"rho": 0.3}, "rho_high": {"rho": 0.95}, "cv0": {"arrival_cv": 0.0},
                    "cv4": {"arrival_cv": 4.0}, "service": {"service_time_s": 9.0},
                    "fewer": {"n_sessions": 3}, "more": {"n_sessions": 10},
                    "closed": {"arrival_mode": "closed", "think_time_mult": 4.0},
                    "cooldown": {"think_time_mult": 0.2}}
        for name, overrides in variants.items():
            with self.subTest(variant=name):
                other = self.build(**{"p_historical": 0.5, "n_sessions": 6, "budget": 12, **overrides})
                theirs = plan_signature(other)
                common = set(signature) & set(theirs)
                self.assertEqual(len(common), min(len(signature), len(theirs)))
                for sid in common:
                    self.assertEqual(signature[sid], theirs[sid], (name, sid))
                count = min(6, other.setting.n_sessions)
                self.assertEqual(other.cohort["sessions"][:count], base.cohort["sessions"][:count])
        changed = plan_signature(self.build(p_historical=0.5, n_sessions=6, budget=12, seed=1))
        self.assertNotEqual(changed, signature)

    def test_open_loop_schedule_and_cooldown(self):
        workload = self.build(p_historical=0.0, n_sessions=4, budget=10, arrival_cv=0.0, rho=0.5)
        times = [r["t_intended_s"] for r in workload.requests]
        self.assertEqual([r["arrival_index"] for r in workload.requests], list(range(1, 41)))
        spacing = workload.config["derived"]["mean_interarrival_s"]
        self.assertAlmostEqual(spacing, workload.config["derived"]["service_time_s"] / 0.5, places=12)
        for index, t in enumerate(times, 1):
            self.assertAlmostEqual(t, index * spacing, places=9)  # CV 0: constant spacing S / rho
        # constant spacing 4 s and a constant 1 s cooldown: always feasible (a Poisson process with a cooldown can
        # be infeasible at the tail, when few sessions have budget left; that raises, see FeasibilityTests)
        cooled = self.build(p_historical=0.0, n_sessions=4, budget=10, think_time_mult=0.5, think_cv=0.0,
                            arrival_cv=0.0, rho=0.5)
        last: dict = {}
        for request in cooled.requests:
            sid = request["session_id"]
            if sid in last:
                self.assertGreaterEqual(request["t_intended_s"], last[sid] + request["cooldown_s"])
            last[sid] = request["t_intended_s"]
            self.assertEqual(request["ordinal"], sum(1 for r in cooled.requests[:request["arrival_index"]]
                                                     if r["session_id"] == sid))


class ServiceTimeTests(FixtureCase):
    def test_depth_dependent_service_time(self):
        hints = sw.SERVICE_TIME_HINTS_S
        causvid = self.setting(service_time_s={"by_depth": hints["causvid"]})
        for depth, expected in ((1, 0.9), (8, 3.3), (12, 4.3), (16, 5.3), (24, 5.8), (32, 6.3), (40, 6.3)):
            self.assertAlmostEqual(causvid.service_time_at(depth), expected, places=12, msg=depth)
        longlive1 = self.setting(service_time_s={"by_depth": hints["longlive1"]})
        self.assertAlmostEqual(longlive1.service_time_at(30), 4.7, places=12)  # saturates at h >= 16
        longlive2 = self.setting(service_time_s={"by_depth": hints["longlive2"]})
        self.assertAlmostEqual(longlive2.service_time_at(3), 5.5 + 9.0 * 2 / 5, places=12)
        self.assertAlmostEqual(longlive2.service_time_at(10), 14.5, places=12)  # h >= 6
        workload = sw.build_workload(causvid)
        depths = [r["target_depth"] for r in workload.requests]
        expected = sum(causvid.service_time_at(d) for d in depths) / len(depths)
        derived = workload.config["derived"]
        self.assertAlmostEqual(derived["service_time_s"], expected, places=12)
        self.assertAlmostEqual(derived["lambda_req_per_s"], 0.8 / expected, places=12)
        self.assertEqual(sum(derived["target_depth_histogram"].values()), len(workload.requests))
        self.assertEqual(sw.WorkloadSetting(dataset=self.dataset, service_time_s=3.3).rates().lambda_req, 0.8 / 3.3)
        with self.assertRaises(ConfigurationError):
            causvid.lambda_req  # depth-dependent S needs the bundle's target depths
        for bad in ({"by_depth": {}}, {"by_depth": {"0": 1.0}}, {"by_depth": {"8": -1.0}}, {"by_depth": {"x": 1.0}},
                    {"table": {"8": 1.0}}, 0.0, -2.0):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                self.setting(service_time_s=bad)
        # trees and prompts do not depend on S; arrival times do
        number = sw.build_workload(self.setting(service_time_s=3.3))
        self.assertEqual(plan_signature(number), plan_signature(workload))
        self.assertNotEqual([r["t_intended_s"] for r in number.requests],
                            [r["t_intended_s"] for r in workload.requests])


class BundleAndContentTests(FixtureCase):
    def write(self, workload: sw.Workload, name: str) -> dict:
        root = Path(self.tmp.name) / "bundles"
        return sw.write_bundle(workload, root / name)

    def test_equal_seeds_give_identical_bundles(self):
        first = self.write(self.build(seed=3, p_historical=0.5), "seed3a")
        second = self.write(self.build(seed=3, p_historical=0.5), "seed3b")
        third = self.write(self.build(seed=4, p_historical=0.5), "seed4")
        self.assertEqual(first["bundle_id"], second["bundle_id"])
        self.assertEqual(first["files"], second["files"])
        for name in sw.BUNDLE_FILES:
            self.assertEqual(Path(first["path"], name).read_bytes(), Path(second["path"], name).read_bytes())
        self.assertNotEqual(first["bundle_id"], third["bundle_id"])
        loaded = sw.load_bundle(first["path"])
        self.assertEqual(len(loaded["requests"]), 40)
        self.assertEqual(loaded["bundle_id"], first["bundle_id"])
        self.assertFalse(os.access(Path(first["path"], "requests.jsonl"), os.W_OK))
        with self.assertRaises(ConfigurationError):
            sw.write_bundle(self.build(seed=3), first["path"])  # immutable
        tampered = Path(self.tmp.name) / "tampered"
        shutil.copytree(first["path"], tampered)
        tampered.chmod(0o755)
        (tampered / "requests.jsonl").chmod(0o644)
        with open(tampered / "requests.jsonl", "a") as handle:
            handle.write("\n")
        with self.assertRaises(ConfigurationError):
            sw.load_bundle(tampered)

    def test_setup_prefix_and_main_storyline_binding(self):
        workload = self.build(p_historical=0.0, d0=8, budget=20, n_sessions=2)
        for setup in workload.setup:
            sid = setup["session_id"]
            expected = [f"{sid}:main:1"] * 5 + [f"{sid}:main:{j - 3}" for j in range(5, 8)]
            self.assertEqual(setup["path_prompt_ids"], expected)
            self.assertEqual([node["generation_index"] for node in setup["nodes"]], list(range(1, 9)))
            mine = [r for r in workload.requests if r["session_id"] == sid]
            # 16 main prompts remain (5..20), then main-path extensions 21..24 need authored text
            self.assertEqual([r["prompt_id"] for r in mine[:16]], [f"{sid}:main:{m}" for m in range(5, 21)])
            self.assertEqual([(r["prompt_kind"], r["main_index"]) for r in mine[16:]],
                             [("main_extension", m) for m in range(21, 25)])
        self.assertEqual(workload.coverage["bindings"]["slots"], {"main_extension": {"slots": 8, "unbound": 8}})
        self.assertFalse(workload.frozen)
        frozen = self.build(p_historical=0.0, d0=8, budget=16, n_sessions=2)
        self.assertTrue(frozen.frozen)
        self.assertEqual(frozen.slots, [])

    def test_content_slots_bank_binding_and_validation(self):
        workload = self.build(p_historical=0.5, n_sessions=3, budget=8)
        self.assertFalse(workload.frozen)
        slots = workload.slots
        self.assertTrue(slots)
        for slot in slots:
            tree = workload.trees[slot["session_id"]]
            parent = tree.nodes[slot["parent_generation_index"]]
            self.assertEqual(slot["key"], sw.slot_key(slot["session_id"], parent.path_prompt_ids,
                                                      slot["child_ordinal"]))
            self.assertEqual(sum(s["chunks"] for s in slot["context"]), parent.depth)
            self.assertEqual({s["generation_index"] for s in slot["siblings"]},
                             set(parent.children) - {slot["generation_index"]})
        bank = {slot["key"]: valid_text(slot["key"]) for slot in slots}
        report = sw.validate_bank(bank, slots, dataset=sw.load_dataset(self.dataset))
        self.assertTrue(report["ok"] and report["complete"], report["errors"][:3])
        bound = sw.build_workload(self.setting(p_historical=0.5, n_sessions=3, budget=8), bank=bank)
        self.assertTrue(bound.frozen)
        self.assertEqual(bound.coverage["bindings"]["requests_unbound"], 0)
        texts = {b["prompt_id"]: b for b in bound.bindings}
        for request in bound.requests:
            self.assertTrue(texts[request["prompt_id"]]["bound"])
        # each check fires
        slot = next(s for s in slots if any(sib["text"] for sib in s["siblings"]))
        sibling = next(sib["text"] for sib in slot["siblings"] if sib["text"])
        other = next(s for s in slots if s["key"] != slot["key"])
        cases = {
            "word_count": {slot["key"]: "Too short to be a continuation."},
            "equals_sibling": {slot["key"]: sibling},
            "copies_parent": {slot["key"]: slot["parent_text"]} if slot["parent_text"] else None,
            "duplicate": {slot["key"]: valid_text("same"), other["key"]: valid_text("same")},
            "contains_parent": ({slot["key"]: slot["parent_text"] + " and then waves"}
                                if slot["parent_text"] and len((slot["parent_text"] + " and then waves").split()) <= 35
                                else None),
        }
        for check, entries in cases.items():
            if entries is None:
                continue
            with self.subTest(check=check):
                result = sw.validate_bank(entries, slots)
                self.assertIn(check, {e["check"] for e in result["errors"]}, result["errors"])
                self.assertFalse(result["ok"])
        unknown = sw.validate_bank({"0" * 64: valid_text("0" * 64)}, slots)
        self.assertEqual(unknown["warnings"][0]["check"], "unknown_key")
        bad = dict(bank)
        bad[slot["key"]] = sibling
        self.assertFalse(sw.build_workload(self.setting(p_historical=0.5, n_sessions=3, budget=8), bank=bad).frozen)

    def test_sweep_expansion_and_up_front_grid_validation(self):
        self.assertEqual(sw.expand_values([1, 2]), [1, 2])
        self.assertEqual(sw.expand_values({"values": [3]}), [3])
        self.assertEqual(sw.expand_values({"start": 0.1, "stop": 0.5, "step": 0.2}), [0.1, 0.3, 0.5])
        self.assertEqual(sw.expand_values({"start": 2, "stop": 8, "step": 3}), [2, 5, 8])
        self.assertEqual(sw.expand_values(0.8), [0.8])
        base = {"dataset": self.dataset, "service_time_s": CAUSVID_S, "target_model": "causvid", "budget": 6}
        document = {"schema": sw.SWEEP_SCHEMA, "name": "t", "base": base,
                    "anchor": {"rho": 0.8, "arrival_cv": 1.0, "p_historical": 0.5, "gap_profile": "uniform",
                               "n_sessions": 4, "d0": 8, "arrival_mode": "open", "think_time_mult": 0.0},
                    "sweep": {"rho": [0.3, 0.5, 0.7, 0.9, 0.95], "arrival_cv": [0.0, 0.5, 2.0, 4.0],
                              "p_historical": [0.0, 0.1, 0.25, 0.75, 1.0], "gap_profile": ["recent", "far"],
                              "n_sessions": [2, 4, 16, 32], "d0": [12, 20]},
                    "settings": [{"id": "think", "arrival_mode": "closed", "think_time_mult": [1.0, 4.0, 16.0]}]}
        expanded = sw.expand_sweep(document)
        self.assertEqual(len(expanded), 25)  # 26 of plan 0.6 minus n_sessions=4 (= the anchor here)
        self.assertEqual(expanded[0]["setting_id"], "anchor")
        self.assertIn("think:think_time_mult=16", [e["setting_id"] for e in expanded])
        bad = dict(document, anchor={**document["anchor"], "arrival_cv": 0.0, "think_cv": 0.0},
                   sweep={"d0": [4, 12], "think_time_mult": [0.5, 100.0], "n_sessions": [2, 41]})
        report = sw.check_settings(sw.expand_sweep(bad))
        self.assertEqual(report["region"], {"d0": {"4": False, "12": True},
                                            "think_time_mult": {"0.5": True, "100": False},
                                            "n_sessions": {"2": True, "41": False}})
        codes = {row["setting_id"]: row["problems"][0]["code"] for row in report["settings"] if not row["feasible"]}
        self.assertEqual(codes, {"d0=4": "latest_illegal", "think_time_mult=100": "think_ge_spacing",
                                 "n_sessions=41": "not_enough_sessions"})
        with self.assertRaises(ConfigurationError):
            sw.expand_sweep(dict(document, anchor={**document["anchor"], "rho": [0.5, 0.8]}))


class RngAndCliTests(FixtureCase):
    def test_variates(self):
        draws = sw.Uniforms(1, "arrivals")
        self.assertEqual(draws.interarrival(2.5, 0.0), 2.5)
        for cv in (0.5, 1.0, 2.0):
            values = [draws.interarrival(1.0, cv) for _ in range(40000)]
            mean = sum(values) / len(values)
            sd = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
            self.assertAlmostEqual(mean, 1.0, delta=0.05 * max(1.0, cv))
            self.assertAlmostEqual(sd / mean, cv, delta=0.08 * cv)
        think = sw.Uniforms(1, "think", 3, 4)
        values = [think.lognormal(4.0, 1.0) for _ in range(40000)]
        mean = sum(values) / len(values)
        self.assertAlmostEqual(mean, 4.0, delta=0.25)
        self.assertEqual(sw.Uniforms(7, "targeting", 1, 2).next(), sw.Uniforms(7, "targeting", 1, 2).next())
        self.assertNotEqual(sw.Uniforms(7, "targeting", 1, 2).next(), sw.Uniforms(7, "think", 1, 2).next())
        with self.assertRaises(ConfigurationError):
            sw.Uniforms(7, "noise")

    def test_cohort_is_stratified_nested_and_deterministic(self):
        dataset = sw.load_dataset(self.dataset)
        big = sw.select_cohort(dataset, "evaluation_candidate", 40, 0)
        self.assertEqual(sorted(big["sessions"]), sorted(dataset.splits["evaluation_candidate"]))
        for n in (2, 8, 17):
            cohort = sw.select_cohort(dataset, "evaluation", n, 0)
            self.assertEqual(cohort["sessions"], big["sessions"][:n])
            self.assertEqual(cohort, sw.select_cohort(dataset, "evaluation_candidate", n, 0))
        cohort = sw.select_cohort(dataset, "evaluation_candidate", 20, 0)
        for stratum in cohort["strata"]:  # every prefix is close to proportional
            self.assertLessEqual(abs(stratum["selected"] - 20 * stratum["candidates"] / 40), 1.5, stratum)
        self.assertNotEqual(sw.select_cohort(dataset, "evaluation_candidate", 8, 1)["sessions"],
                            sw.select_cohort(dataset, "evaluation_candidate", 8, 0)["sessions"])
        self.assertEqual(sw.coarse_category("forest_amphibian_development"), "nature_wildlife")
        self.assertEqual(sw.coarse_category("heritage pottery"), "craft_art")
        self.assertEqual(sw.coarse_category("sports"), "people_places")

    def test_example_cli(self):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
        script = str(ROOT / "examples" / "benchmark_interactive_sessions.py")
        sweep = Path(self.tmp.name) / "sweep.json"
        sweep.write_text(json.dumps({
            "schema": sw.SWEEP_SCHEMA, "name": "cli",
            "base": {"dataset": self.dataset, "service_time_s": {"by_depth": {"1": 0.9, "8": 3.3, "16": 5.3,
                                                                              "32": 6.3}},
                     "target_model": "causvid", "budget": 4, "n_sessions": 2, "split": "development"},
            "anchor": {"p_historical": 0.0}, "sweep": {"rho": [0.5], "d0": [4]}}))

        def run(*args):
            return subprocess.run([sys.executable, "-B", script, *args], cwd=ROOT, env=env, capture_output=True,
                                  text=True, timeout=300)

        self.assertIn("usage:", run("--help").stdout)
        checked = run("check", str(sweep))
        self.assertEqual(checked.returncode, 3, checked.stderr)  # d0=4 is infeasible (min_switch_depth 5)
        self.assertEqual(json.loads(checked.stdout)["infeasible"], ["d0=4"])
        out = Path(self.tmp.name) / "cli_bundles"
        refused = run("generate", str(sweep), "--out", str(out))
        self.assertEqual(refused.returncode, 3)
        self.assertFalse(out.exists())  # every combination is checked before anything is written
        written = run("generate", str(sweep), "--out", str(out), "--feasible-only")
        self.assertEqual(written.returncode, 0, written.stderr)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["anchor", "feasibility.json", "rho=0.5"])
        bundle = sw.load_bundle(out / "anchor")
        self.assertTrue(bundle["frozen"])
        self.assertEqual(len(bundle["requests"]), 8)


if __name__ == "__main__":
    unittest.main()
