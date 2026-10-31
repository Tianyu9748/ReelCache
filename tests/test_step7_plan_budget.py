"""SGLang Step 7 Part C: the planner's working set under a memory budget (LongCat planning on the GPU).

``LatestMethodPlanner(max_working_bytes=...)`` / ``ArchiveOptions(plan_memory_bytes=...)`` plan a chunk's matrices in
consecutive groups whose planning tensors fit the budget and release each group's tensors before the next; every
matrix's products are per matrix, so bounded plans equal unbounded plans bit for bit -- and nodes keep their content
addresses (the budget and the eigensolver splits are unhashed run telemetry) -- whenever a matrix's eigensolves do
not depend on the batch they share: on the CPU, and on CUDA with a fixed eigensolver sub-batch (one matrix per call
from n = 4096 by default, e.g. LongCat). cuSOLVER's batched eigenpairs depend on the call's batch size class, which
the size-dependent mock solver reproduces on the CPU. The CUDA test (``REELCACHE_STEP7_GPU_TESTS=1``) also checks the
measured peak; LongCat-size evidence: runtime_gate/step7 (scripts/longcat_plan_budget.py).
"""

from __future__ import annotations

import math
import os
import unittest

import torch

from reelcache.errors import ConfigurationError
from reelcache.lineage_commit import ArchiveOptions
from reelcache.rank_method import LatestMethodPlanner, RankMethodConfig
from reelcache.types import ActiveBasisBinding
from tests import step6_support as s6


def _plans(method, *, budget, device="cpu", eigh=None, eigh_label=None, orthogonalizer="batched"):
    """Plan four chunks of a lineage (fresh, then share/augment/refit) with the unbounded or bounded planner."""
    registry = s6.registry()
    bindings = {key: ActiveBasisBinding() for key in s6.KEYS}
    parents = {key: None for key in s6.KEYS}
    results = []
    for index in range(4):
        matrices, contexts = s6.chunk(index)
        matrices = {key: value.to(device) for key, value in matrices.items()}
        planner = LatestMethodPlanner(method, registry.pin(), device=device, eigh=eigh, eigh_label=eigh_label,
                                      orthogonalizer=orthogonalizer, max_working_bytes=budget)
        plan = planner.plan(matrices, contexts, bindings, parents, chunk_id=f"c{index}",
                            source_ids={key: f"id-{index}-{key.layer}{key.target}" for key in s6.KEYS})
        results.append(plan)
        registry.observe([observation for item in plan.plans.values() for observation in item.observations])
        for key, item in plan.plans.items():
            if item.decision in ("fresh", "refit", "diagnostic_full_width"):
                rows = item.new_rows
            elif item.decision == "augment":
                rows = torch.cat((parents[key], item.new_rows), 0)
            else:
                rows = parents[key]
            parents[key] = rows
            bindings[key] = ActiveBasisBinding("manifest:" + format(index * 10 + key.layer * 2 + (key.target == "V"),
                                                                    "064x"),
                                               int(rows.shape[0]), "a" * 64, int(rows.shape[0]), f"c{index}")
    return results


def _size_dependent_eigh(grams):
    """Ascending values and row vectors whose two leading vectors turn with the call's batch size (like cuSOLVER's
    batch-size classes: a matrix's result depends on how many matrices share the call, not on which)."""
    values, columns = torch.linalg.eigh(grams)
    rows = columns.transpose(1, 2).clone()
    angle = 1e-3 * grams.shape[0]
    top, second = rows[:, -1].clone(), rows[:, -2].clone()
    rows[:, -1] = math.cos(angle) * top + math.sin(angle) * second
    rows[:, -2] = -math.sin(angle) * top + math.cos(angle) * second
    return values, rows


def _per_matrix_bytes():
    return 3 * 4 * s6.TOKENS * s6.CHANNELS + 3 * 4 * s6.CHANNELS ** 2 + 3 * 8 * s6.TOKENS * s6.CHANNELS


def _same(test, left, right):
    for a, b in zip(left, right):
        test.assertEqual(set(a.plans), set(b.plans))
        for key in a.plans:
            x, y = a.plans[key], b.plans[key]
            test.assertEqual((x.decision, x.reason, x.rank_view, x.bucket_rank, x.observed_rank),
                             (y.decision, y.reason, y.rank_view, y.bucket_rank, y.observed_rank))
            test.assertEqual(x.decision_energy, y.decision_energy)
            test.assertEqual(x.new_rows is None, y.new_rows is None)
            if x.new_rows is not None:
                test.assertTrue(torch.equal(x.new_rows.cpu(), y.new_rows.cpu()))
            test.assertEqual([o.to_dict() for o in x.observations], [o.to_dict() for o in y.observations])


class PlannerBudgetTests(unittest.TestCase):
    def test_bounded_plans_equal_unbounded_plans(self):
        base = s6.method()
        exact = s6.exact_method()
        per_matrix = 3 * 4 * s6.TOKENS * s6.CHANNELS + 3 * 4 * s6.CHANNELS ** 2 + 3 * 8 * s6.TOKENS * s6.CHANNELS
        for name, method, orthogonalizer in (("exact", exact, "batched"), ("reference", exact, "householder"),
                                             ("topk", RankMethodConfig(base.targets, base.codec, solver="topk_rr"),
                                              "batched")):
            with self.subTest(name=name):
                unbounded = _plans(method, budget=None, orthogonalizer=orthogonalizer)
                self.assertTrue(all(plan.memory is None for plan in unbounded))
                for groups_of in (1, 3):
                    bounded = _plans(method, budget=2 * groups_of * per_matrix, orthogonalizer=orthogonalizer)
                    self.assertEqual(bounded[0].memory["group_size"], groups_of)
                    self.assertEqual(bounded[0].memory["groups"], -(-len(s6.KEYS) // groups_of))
                    _same(self, bounded, unbounded)
                    self.assertEqual([plan.counters for plan in bounded], [plan.counters for plan in unbounded])

    def test_fixed_eigh_sub_batches_keep_bounded_plans_identical_under_a_size_dependent_solver(self):
        from reelcache.gpu_fitting import BoundedEigh
        base = s6.method()
        for name, method in (("exact", s6.exact_method()),
                             ("topk", RankMethodConfig(base.targets, base.codec, solver="topk_rr"))):
            with self.subTest(name=name):
                plans = {}
                for policy, size in (("fixed", 4), ("whole", None)):
                    for budget in (None, 2 * _per_matrix_bytes()):
                        eigh = BoundedEigh("size-dependent", _size_dependent_eigh, workspace=lambda b, n, d: 0,
                                           total_memory=1 << 40, sub_batch_size=lambda n, size=size: size)
                        plans[policy, budget is None] = _plans(method, budget=budget, eigh=eigh,
                                                               eigh_label="size-dependent")
                _same(self, plans["fixed", False], plans["fixed", True])
                # Control: the same solver called on whole batches gives group-dependent rows.
                differs = any(not torch.equal(a.plans[key].new_rows, b.plans[key].new_rows)
                              for a, b in zip(plans["whole", False], plans["whole", True]) for key in a.plans
                              if a.plans[key].new_rows is not None and b.plans[key].new_rows is not None)
                self.assertTrue(differs)

    def test_budget_is_execution_telemetry_and_validated(self):
        with self.assertRaises(ConfigurationError):
            ArchiveOptions(plan_memory_bytes=0)
        with self.assertRaises(ConfigurationError):
            LatestMethodPlanner(s6.method(), s6.registry().pin(), max_working_bytes=-5)
        self.assertNotIn("plan_memory_bytes", ArchiveOptions().to_dict())
        self.assertNotIn("plan_memory_bytes", ArchiveOptions(plan_memory_bytes=1 << 20).output_identity())
        refs = {}
        for name, options in (("unbounded", ArchiveOptions(encode_threads=1)),
                              ("bounded", ArchiveOptions(encode_threads=1, plan_memory_bytes=1 << 20))):
            runtime = s6.RuntimeFactory(self)(f"budget-{name}")
            runtime.create_lineage("main")
            registry = s6.registry()
            refs[name] = [runtime.commit_node("main", f"chunk-{index:03d}", *s6.chunk(index), method=s6.method(),
                                              registry=registry, options=options).node_ref for index in range(3)]
            report = runtime.node_commit_report(refs[name][-1])
            memory = report["plans"][0].get("plan_memory")
            self.assertEqual(memory is None, name == "unbounded")
            # Finalization also runs in bounded groups (one matrix per group at this budget) with identical tiles.
            finalize = report.get("finalize_memory")
            self.assertEqual(finalize is None, name == "unbounded")
            if finalize is not None:
                self.assertEqual((finalize[0]["group_size"], finalize[0]["groups"]), (1, len(s6.KEYS)))
                self.assertIsNone(runtime.store.get_manifest(refs[name][-1]).get("commit", {}).get(
                    "finalize_memory"))
        # Same content addresses: plans are identical and the budget is unhashed run telemetry.
        self.assertEqual(refs["bounded"], refs["unbounded"])


if os.environ.get("REELCACHE_STEP7_GPU_TESTS") == "1":
    class PlannerBudgetCUDATests(unittest.TestCase):
        """Inside GPU jobs only (collected only with REELCACHE_STEP7_GPU_TESTS=1)."""

        def test_cuda_bounded_plans_equal_unbounded_plans(self):
            from reelcache.gpu_fitting import BoundedEigh, select_eigh_solver
            if not torch.cuda.is_available():
                raise AssertionError("REELCACHE_STEP7_GPU_TESTS=1 requires a CUDA device")
            base = s6.method()
            budget = 2 * _per_matrix_bytes()
            for name, method in (("exact", s6.exact_method()), ("topk", RankMethodConfig(base.targets, base.codec,
                                                                                           solver="topk_rr"))):
                with self.subTest(name=name):
                    # One matrix per eigensolver call (the default from n = 4096): bitwise identical plans.
                    label, eigh = select_eigh_solver("auto")
                    alone = BoundedEigh(label, eigh.solver, sub_batch_size=lambda n: 1)
                    unbounded = _plans(method, budget=None, device="cuda", eigh=alone, eigh_label=label)
                    bounded = _plans(method, budget=budget, device="cuda", eigh=alone, eigh_label=label)
                    _same(self, bounded, unbounded)
                    self.assertEqual(bounded[0].memory["group_size"], 1)
                    # Default policy below n = 4096 (whole-batch calls): a matrix's eigenpairs follow its call's
                    # cuSOLVER size class; within near-degenerate eigenvalue clusters that turns the fitted directions,
                    # so plans may differ (both verified against the targets). Recorded, not asserted equal.
                    label, eigh = select_eigh_solver("auto")
                    whole = _plans(method, budget=None, device="cuda", eigh=eigh, eigh_label=label)
                    label, eigh = select_eigh_solver("auto")
                    grouped = _plans(method, budget=budget, device="cuda", eigh=eigh, eigh_label=label)
                    changed, largest = 0, 0.0
                    for a, b in zip(whole, grouped):
                        for key in a.plans:
                            x, y = a.plans[key], b.plans[key]
                            changed += (x.decision, x.rank_view) != (y.decision, y.rank_view)
                            largest = max(largest, abs(x.decision_energy - y.decision_energy))
                            for plan in (x, y):
                                self.assertGreaterEqual(plan.decision_energy, plan.target)
                    print(f"\n  [{name}] default whole-batch eigh, bounded vs unbounded: {changed} decision/rank "
                          f"changes, max |decision energy difference| {largest:.2e}")

        def test_cuda_planner_peak_follows_the_budget(self):
            """Eight [8192, 512] matrices: the bounded working set stays near its budget, far below unbounded."""
            from reelcache.gpu_fitting import select_eigh_solver
            from reelcache.predictor_registry import RankPredictorRegistry
            from reelcache.types import MatrixKey, ModelIdentity, TensorContext
            frames, height, width, channels = 2, 64, 64, 512
            tokens = frames * height * width
            model = ModelIdentity("step7-budget", "step7-budget-peak", 4, channels, "tokens_channels", "bfloat16")
            generator = torch.Generator(device="cuda").manual_seed(11)
            matrices, contexts = {}, {}
            for layer in range(4):
                for target in ("K", "V"):
                    key = MatrixKey(layer, target)
                    value = torch.randn(tokens, 96, device="cuda", generator=generator) @ torch.randn(
                        96, channels, device="cuda", generator=generator)
                    value = value + 0.05 * torch.randn(tokens, channels, device="cuda", generator=generator)
                    if target == "K":
                        value = torch.nn.functional.layer_norm(value, (channels,))
                    matrices[key] = value.to(torch.bfloat16)
                    contexts[key] = TensorContext(model, key, 0, frames, height, width,
                                                  "pre_rope" if target == "K" else "none")
            per_matrix = 3 * 4 * tokens * channels + 3 * 4 * channels ** 2 + 3 * 8 * min(tokens, 1024) * channels
            peaks, plans = {}, {}
            for budget in (None, 2 * per_matrix):
                label, eigh = select_eigh_solver("auto")
                planner = LatestMethodPlanner(s6.method(), RankPredictorRegistry("step7-budget-peak").pin(),
                                              device="cuda", eigh=eigh, eigh_label=label, orthogonalizer="batched",
                                              max_working_bytes=budget)
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                plans[budget] = planner.plan(matrices, contexts, {key: ActiveBasisBinding() for key in matrices},
                                             {key: None for key in matrices}, chunk_id="c0",
                                             source_ids={key: f"peak-{key.layer}{key.target}" for key in matrices})
                torch.cuda.synchronize()
                peaks[budget] = torch.cuda.max_memory_allocated() - before
            bounded = peaks[2 * per_matrix]
            self.assertEqual(plans[2 * per_matrix].memory["group_size"], 1)
            self.assertLess(bounded, 2 * per_matrix + (32 << 20), peaks)
            self.assertLess(2 * bounded, peaks[None], peaks)
            for key in matrices:
                x, y = plans[None].plans[key], plans[2 * per_matrix].plans[key]
                self.assertEqual((x.decision, x.rank_view), (y.decision, y.rank_view))


if __name__ == "__main__":
    unittest.main()
