"""Tests of the LongCat group-level planning gate (CR-17-6 follow-up, package g7).

:class:`~reelcache_sglang_models.longcat_video.planning_gate.GatedPlanner` plans exactly like
:class:`reelcache.rank_method.LatestMethodPlanner`, while a chunk's host sources go to the planner's device one group at
a time. Every group is a unit of the worker's ``PlanningGate``, so a memory-critical section (denoising, VAE
decoding) waits for one group instead of a node. On the CPU the planner's "device copies" are accounted by a ledger
(placement returns registered clones). The CUDA checks (bitwise plans with real placement; the planner holding one
group) run inside GPU jobs with ``REELCACHE_LONGCAT_G7_GPU_TESTS=1``.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import shutil
import tempfile
import threading
import time
import types
import unittest
import weakref
from pathlib import Path
from unittest import mock

import torch

from reelcache.errors import CapacityError, StateError
from reelcache.gpu_fitting import GPUFitter
from reelcache.rank_method import LatestMethodPlanner, RankMethodConfig
from reelcache.types import ActiveBasisBinding
from reelcache_sglang_models.longcat_video import reelcache_backend as mb
from reelcache_sglang_models.longcat_video import planning_gate as pg
from reelcache_sglang_models.longcat_video import session_provider as sp
from tests import step6_support as s6
from tests.test_longcat_video_planning_gate import stage_request
from tests.test_longcat_video_session_provider import ARCH, FINGERPRINT, HEADS, initial_frames
from tests.test_step7_plan_budget import _same

# Source of the core internals GatedPlanner / GatedGPUFitter build on (Step 7 dev20c = freeze B2). A change there
# needs a re-validation of the g7 planner (bitwise plans, placement) before these pins are updated. The
# GPUFitter.plan_chunk pin is that of the same source under this release's names (only identifiers renamed).
CORE_SOURCES = {
    "LatestMethodPlanner.plan": (LatestMethodPlanner.plan, "801dd888b27d9520"),
    "LatestMethodPlanner._admit": (LatestMethodPlanner._admit, "fe1a666839c7c411"),
    "LatestMethodPlanner._plan_group": (LatestMethodPlanner._plan_group, "800c1e3f4a3e67a4"),
    "LatestMethodPlanner._working_groups": (LatestMethodPlanner._working_groups, "e56dcc7a53fae5f5"),
    "GPUFitter.plan_chunk": (GPUFitter.plan_chunk, "6742e5dd8a5529a3"),
}


class PlacementLedger:
    """The planner's device copies, on the CPU: placement returns clones registered here (weak references)."""

    def __init__(self, unit_seconds: float = 0.0):
        self.lock = threading.Lock()
        self.refs: list = []
        self.placed = 0
        self.max_alive = 0
        self.violations: list = []
        self.unit_seconds = unit_seconds

    def place(self, tensors: list) -> list:
        clones = [tensor.clone() for tensor in tensors]
        with self.lock:
            self.refs.extend((weakref.ref(clone), clone.numel() * clone.element_size()) for clone in clones)
            self.placed += len(clones)
            self.max_alive = max(self.max_alive, sum(1 for ref, _ in self.refs if ref() is not None))
        if self.unit_seconds:
            time.sleep(self.unit_seconds)  # a group takes a while on the planner
        return clones

    def alive_bytes(self) -> int:
        with self.lock:
            return sum(nbytes for ref, nbytes in self.refs if ref() is not None)

    def probe(self, where: str) -> None:
        alive = self.alive_bytes()
        if alive:
            self.violations.append((where, alive))


class LedgerPlanner(pg.GatedPlanner):
    ledger: PlacementLedger

    def _place_matrices(self, tensors: list) -> list:
        return self.ledger.place(tensors)

    def _place_rows(self, tensors: list) -> list:
        return self.ledger.place(tensors)


def ledger_planner(ledger: PlacementLedger, gate: pg.PlanningGate):
    def build(method, view, **kwargs):
        planner = LedgerPlanner(method, view, device="cpu", gate=gate, **kwargs)
        planner.ledger = ledger
        return planner
    return build


def base_planner(method, view, **kwargs):
    return LatestMethodPlanner(method, view, device="cpu", **kwargs)


def lineage_plans(build, method, *, budget, orthogonalizer="batched", after=None):
    """Four chunks of a lineage (fresh, then share / augment / refit), as tests/test_step7_plan_budget.py plans them."""
    registry = s6.registry()
    bindings = {key: ActiveBasisBinding() for key in s6.KEYS}
    parents = {key: None for key in s6.KEYS}
    results = []
    for index in range(4):
        matrices, contexts = s6.chunk(index)
        planner = build(method, registry.pin(), orthogonalizer=orthogonalizer, max_working_bytes=budget)
        plan = planner.plan(matrices, contexts, bindings, parents, chunk_id=f"c{index}",
                            source_ids={key: f"id-{index}-{key.layer}{key.target}" for key in s6.KEYS})
        if after is not None:
            after(planner, plan)
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


def per_matrix_bytes() -> int:
    return 3 * 4 * s6.TOKENS * s6.CHANNELS + 3 * 4 * s6.CHANNELS ** 2 + 3 * 8 * s6.TOKENS * s6.CHANNELS


def run_in_thread(target, *args):
    errors = []

    def body():
        try:
            target(*args)
        except BaseException as exc:  # noqa: BLE001 - reported by the test thread
            errors.append(exc)

    thread = threading.Thread(target=body, daemon=True)
    thread.start()
    return thread, errors


class PlanningGateTests(unittest.TestCase):
    def gate(self) -> pg.PlanningGate:
        gate = pg.PlanningGate()
        gate.configure(enabled=True)
        return gate

    def test_a_section_waits_only_for_the_unit_in_progress(self):
        gate = self.gate()
        started = threading.Event()

        def planner():
            for _ in range(20):  # a long node: twenty groups back to back
                with gate.unit("group"):
                    started.set()
                    time.sleep(0.1)

        thread, errors = run_in_thread(planner)
        self.assertTrue(started.wait(5))
        time.sleep(0.03)
        clock = time.perf_counter()
        with gate.section():
            waited = time.perf_counter() - clock
            self.assertFalse(gate.busy)  # no unit runs inside a section
            units = gate.stats()["units"]
            time.sleep(0.25)
            self.assertEqual(gate.stats()["units"], units)  # and none starts
            self.assertFalse(gate.busy)
        self.assertLess(waited, 0.2)  # the current group, not the remaining node
        thread.join(10)
        self.assertEqual(errors, [])
        stats = gate.stats()
        self.assertEqual(stats["units"], 20)
        self.assertGreaterEqual(stats["units_deferred"], 1)
        self.assertGreater(stats["unit_wait_seconds"], 0.2)
        self.assertEqual((stats["sections"], stats["open_sections"]), (1, 0))

    def test_units_wait_for_open_and_waiting_sections(self):
        gate = self.gate()
        ran = threading.Event()
        with gate.section():
            thread, errors = run_in_thread(self._unit_then, gate, ran)
            time.sleep(0.2)
            self.assertFalse(ran.is_set())
            with gate.section():  # sections nest and overlap
                time.sleep(0.05)
            self.assertFalse(ran.is_set())
        self.assertTrue(ran.wait(5))
        thread.join(5)

    @staticmethod
    def _unit_then(gate, ran):
        with gate.unit("group"):
            ran.set()

    def test_a_disabled_gate_does_not_gate(self):
        gate = pg.PlanningGate()
        inside = threading.Event()
        release = threading.Event()

        def planner():
            with gate.unit("group"):
                inside.set()
                release.wait(5)

        thread, errors = run_in_thread(planner)
        self.assertTrue(inside.wait(5))
        clock = time.perf_counter()
        with gate.section():
            self.assertLess(time.perf_counter() - clock, 0.05)
        release.set()
        thread.join(5)
        self.assertEqual(gate.stats()["sections"], 0)

    def test_section_timeout_and_units_do_not_nest(self):
        gate = self.gate()
        inside = threading.Event()

        def planner():
            with gate.unit("group"):
                inside.set()
                time.sleep(0.4)

        thread, errors = run_in_thread(planner)
        self.assertTrue(inside.wait(5))
        with self.assertRaises(CapacityError):
            with gate.section(timeout=0.05):
                pass
        thread.join(5)
        self.assertEqual(gate.stats()["section_timeouts"], 1)
        with gate.unit("group"):  # the timed-out section left nothing waiting
            with self.assertRaises(StateError):
                with gate.unit("group"):
                    pass
        with self.assertRaises(StateError):
            gate.configure(enabled=1)


class GatedPlannerTests(unittest.TestCase):
    def test_gated_plans_equal_the_planner_and_sources_are_placed_one_group_at_a_time(self):
        base = s6.method()
        per_matrix = per_matrix_bytes()
        for name, method, orthogonalizer in (("exact", s6.exact_method(), "batched"),
                                             ("reference", s6.exact_method(), "householder"),
                                             ("topk", RankMethodConfig(base.targets, base.codec, solver="topk_rr"),
                                              "batched")):
            for groups_of in (None, 1, 3):
                with self.subTest(name=name, groups_of=groups_of):
                    budget = None if groups_of is None else 2 * groups_of * per_matrix
                    expected = lineage_plans(base_planner, method, budget=budget, orthogonalizer=orthogonalizer)
                    ledger, gate = PlacementLedger(), pg.PlanningGate()
                    residencies = []

                    def after(planner, plan):
                        self.assertEqual(ledger.alive_bytes(), 0)  # nothing placed survives the chunk
                        residencies.append(dict(planner.residency))

                    gated = lineage_plans(ledger_planner(ledger, gate), method, budget=budget,
                                          orthogonalizer=orthogonalizer, after=after)
                    _same(self, gated, expected)
                    size = len(s6.KEYS) if groups_of is None else groups_of
                    self.assertLessEqual(ledger.max_alive, 2 * size)  # one group's matrices (+ its inherited rows)
                    matrices = sum(1 for _ in s6.KEYS) * 4
                    self.assertGreaterEqual(ledger.placed, matrices)  # every matrix placed once, rows when inherited
                    self.assertTrue(all(item["lazy_sources"] for item in residencies))
                    groups = -(-len(s6.KEYS) // size)
                    self.assertEqual([item["groups"] for item in residencies], [groups] * 4)
                    self.assertEqual(gate.stats()["units"], 4 * groups)

    def test_lazy_sources_off_keeps_the_planners_admission(self):
        method, per_matrix = s6.exact_method(), per_matrix_bytes()
        gate = pg.PlanningGate()
        expected = lineage_plans(base_planner, method, budget=2 * per_matrix)
        gated = lineage_plans(lambda m, v, **kw: pg.GatedPlanner(m, v, device="cpu", gate=gate, lazy_sources=False,
                                                                  **kw), method, budget=2 * per_matrix)
        _same(self, gated, expected)
        self.assertEqual(gate.stats()["units"], 4 * (1 + len(s6.KEYS)))  # admission + one unit per group

    def test_core_internals_are_the_validated_ones(self):
        changed = {name: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]
                   for name, (fn, pin) in CORE_SOURCES.items()
                   if hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16] != pin}
        self.assertEqual(changed, {}, "core planner internals changed: re-validate GatedPlanner / GatedGPUFitter "
                                      "(bitwise plans, one group placed at a time) before updating CORE_SOURCES")

    def test_backend_selects_the_gate_for_host_sources(self):
        gate = pg.PlanningGate()
        prop = mb.ReelCacheConditioningCacheBackend.planning_gate.fget

        def backend(device, deferred_sources, fitter=True):
            pipeline = types.SimpleNamespace(device=device, fitter=types.SimpleNamespace(gate=gate) if fitter else None)
            return types.SimpleNamespace(pipeline=pipeline, deferred_sources=deferred_sources)

        self.assertIs(prop(backend("cuda:0", "host")), gate)
        self.assertIs(prop(backend("cpu", "device")), gate)  # a CPU pipeline always plans host sources
        self.assertIsNone(prop(backend("cuda:0", "device")))  # device sources: finalization on the GPU (g6 gate)
        self.assertIsNone(prop(backend("cuda:0", "host", fitter=False)))
        self.assertIsNone(prop(types.SimpleNamespace(pipeline=None, deferred_sources="host")))


class CPUGatedFitter:
    """A CPU stand-in for GatedGPUFitter in an ArchivePipeline (the planner's placement accounted by a ledger)."""

    def __init__(self, gate: pg.PlanningGate, ledger: PlacementLedger):
        self.gate, self.ledger = gate, ledger
        self.last_residency = None

    def adopt_current_thread(self):
        pass

    def plan_chunk(self, config, view, sources, contexts, bindings, parent_rows, *, chunk_id, source_ids,
                   orthogonalizer="householder", max_working_bytes=None):
        planner = LedgerPlanner(config, view, device="cpu", orthogonalizer=orthogonalizer,
                                max_working_bytes=max_working_bytes, gate=self.gate)
        planner.ledger = self.ledger
        plan = planner.plan(sources, contexts, bindings, parent_rows, chunk_id=chunk_id, source_ids=source_ids)
        self.last_residency = dict(planner.residency)
        return plan


class GatedProviderTests(unittest.TestCase):
    """Two sessions on one worker: g7 sections wait for one planner group; the g6 node-level gate (control) waits for
    the node's host stages too; identical archives."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="longcat-g7-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.addCleanup(sp._ACTIVE.clear)

    def provider(self, name: str, *, gated: bool, ledger: PlacementLedger | None = None):
        from reelcache.archive_pipeline import ArchivePipeline
        from reelcache.archive_restore import SoftwareLineageRestorer
        from reelcache.lineage_commit import ArchiveOptions
        from reelcache.predictor_registry import RankPredictorRegistry

        store = self.root / name
        # one matrix per planner group (4 groups per node): 2 x a test matrix's planning working set
        cfg = sp.LongCatSessionConfig.from_dict({"store_root": str(store), "durability": "never",
                                                 "plan_memory_bytes": 2 * 638976})
        runtime = mb.open_runtime(store, mb.model_identity(FINGERPRINT, ARCH), **cfg.integrity())
        gate = pg.PlanningGate() if gated else None
        fitter = CPUGatedFitter(gate, ledger or PlacementLedger()) if gated else None
        pipeline = ArchivePipeline(runtime, method=mb.default_method(), registry=RankPredictorRegistry("s14-longcat"),
                                   options=ArchiveOptions(encode_threads=1, plan_memory_bytes=cfg.plan_memory_bytes),
                                   device="cpu", encode_workers=1, fitter=fitter)
        decision = sp.resolve_schedule(cfg.schedule, representation=cfg.representation,
                                       defer_planning=cfg.defer_planning, plan_device="cpu",
                                       plan_memory_bytes=cfg.plan_memory_bytes, resident_bytes=0, total_bytes=0)
        backend = mb.ReelCacheConditioningCacheBackend(runtime, pipeline=pipeline,
                                                    restorer=SoftwareLineageRestorer(runtime),
                                                    archive=(cfg.representation,), heads=HEADS, pin_memory=False,
                                                    capture_slots=cfg.c3_slots(), schedule=decision["chosen"],
                                                    deferred_sources="host")
        provider = sp.LongCatSessionProvider(cfg, backend=backend, owned=(sp._Closer(pipeline, runtime),),
                                             startup={"schedule": decision})
        self.addCleanup(self._close, provider)
        # the gate as create_provider sets it where the eager-overlap check fails (its CUDA check bypassed)
        backend.defer_planning = True
        if gate is not None:
            gate.configure(enabled=True)
        sp._ACTIVE[:] = [provider]
        return provider

    @staticmethod
    def _close(provider):
        try:
            provider.close()
        except Exception:  # noqa: BLE001
            pass

    def two_sessions(self, provider, ledger):
        """Session A's non-durable request ends (its node plans behind it), session B starts at once."""
        from reelcache.lineage_commit import CommitJob

        real_finalize = CommitJob.finalize

        def slow_finalize(job, *args, **kwargs):  # the node's host stage (LongCat: ~45 s on the A100 host)
            time.sleep(0.8)
            return real_finalize(job, *args, **kwargs)

        busy = {"seen": False}
        with mock.patch.object(CommitJob, "finalize", slow_finalize):
            stage_request(provider, {"session_id": "a", "action": "generate"}, ledger, frames=initial_frames(31))
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not provider.backend.pipeline.stats()["submitted"]:
                time.sleep(0.005)
            time.sleep(0.05)  # the fitter is inside A's node (planning its groups)
            busy["seen"] = provider.backend.pending() > 0
            second = stage_request(provider, {"session_id": "b", "action": "generate"}, ledger,
                                   frames=initial_frames(32), denoise=0.3)
            provider.backend.flush(wait=True)
        waits = [segment["planning_deferral_wait_seconds"] for segment in second["segments"]]
        return waits, busy["seen"]

    def node_refs(self, provider):
        return sorted((record.epoch.session_id, record.epoch.segment_index, record.lossy.get("node_ref"))
                      for record in provider.backend._index.values())

    def test_sections_wait_for_one_group_and_archives_are_identical(self):
        ledger = PlacementLedger(unit_seconds=0.08)
        gated = self.provider("gated", gated=True, ledger=ledger)
        waits, busy = self.two_sessions(gated, ledger)
        self.assertTrue(busy)
        gate = gated.backend.planning_gate
        stats = gate.stats()
        self.assertLess(waits[0], 0.3, stats)  # at most the planner's current group (0.08 s) + scheduling
        self.assertLess(stats["max_section_wait_seconds"], 0.3, stats)
        self.assertEqual(ledger.violations, [])  # no placed source alive inside any section
        self.assertEqual(ledger.alive_bytes(), 0)
        self.assertEqual(stats["units"], 2 * 4)  # 2 nodes x 4 groups, every one a unit
        self.assertGreaterEqual(stats["sections"], 4)
        refs = self.node_refs(gated)
        self.assertTrue(all(ref for _, _, ref in refs))

        sp._ACTIVE.clear()
        control = self.provider("node", gated=False)  # g6: the core's node-level gate
        control_waits, _ = self.two_sessions(control, PlacementLedger())
        self.assertGreater(control_waits[0], 0.5)  # B waited for A's whole node (incl. the 0.8 s host stage)
        self.assertEqual(self.node_refs(control), refs)  # bitwise: the same content-addressed nodes


if os.environ.get("REELCACHE_LONGCAT_G7_GPU_TESTS") == "1":
    class GatedPlannerCUDATests(unittest.TestCase):
        """Inside GPU jobs only (REELCACHE_LONGCAT_G7_GPU_TESTS=1)."""

        def test_cuda_gated_plans_equal_the_planner_with_host_sources(self):
            from reelcache.gpu_fitting import BoundedEigh, select_eigh_solver
            if not torch.cuda.is_available():
                raise AssertionError("REELCACHE_LONGCAT_G7_GPU_TESTS=1 requires a CUDA device")
            base = s6.method()
            for name, method in (("exact", s6.exact_method()),
                                 ("topk", RankMethodConfig(base.targets, base.codec, solver="topk_rr"))):
                for groups_of in (None, 1):
                    with self.subTest(name=name, groups_of=groups_of):
                        budget = None if groups_of is None else 2 * groups_of * per_matrix_bytes()

                        def build(cls, **extra):
                            def make(method, view, **kwargs):
                                label, eigh = select_eigh_solver("auto")
                                alone = BoundedEigh(label, eigh.solver, sub_batch_size=lambda n: 1)
                                return cls(method, view, device="cuda", eigh=alone, eigh_label=label, **kwargs,
                                           **extra)
                            return make

                        expected = lineage_plans(build(LatestMethodPlanner), method, budget=budget)
                        gate = pg.PlanningGate()
                        gated = lineage_plans(build(pg.GatedPlanner, gate=gate), method, budget=budget)
                        _same(self, gated, expected)

        def test_cuda_gated_planner_holds_one_group(self):
            """Eight [8192, 512] BF16 host matrices, groups of one: the planner's peak is one group, nothing stays."""
            from reelcache.gpu_fitting import select_eigh_solver
            from reelcache.predictor_registry import RankPredictorRegistry
            from reelcache.types import MatrixKey, ModelIdentity, TensorContext
            frames, height, width, channels = 2, 64, 64, 512
            tokens = frames * height * width
            model = ModelIdentity("g7", "g7-gate", 4, channels, "tokens_channels", "bfloat16")
            generator = torch.Generator().manual_seed(11)
            matrices, contexts = {}, {}
            for layer in range(4):
                for target in ("K", "V"):
                    key = MatrixKey(layer, target)
                    value = torch.randn(tokens, 96, generator=generator) @ torch.randn(96, channels, generator=generator)
                    value = value + 0.05 * torch.randn(tokens, channels, generator=generator)
                    if target == "K":
                        value = torch.nn.functional.layer_norm(value, (channels,))
                    matrices[key] = value.to(torch.bfloat16).pin_memory()
                    contexts[key] = TensorContext(model, key, 0, frames, height, width,
                                                  "pre_rope" if target == "K" else "none")
            sources = sum(m.numel() * m.element_size() for m in matrices.values())
            per_matrix = 3 * 4 * tokens * channels + 3 * 4 * channels ** 2 + 3 * 8 * min(tokens, 1024) * channels
            peaks, plans = {}, {}
            for cls in (LatestMethodPlanner, pg.GatedPlanner):
                label, eigh = select_eigh_solver("auto")
                extra = {"gate": pg.PlanningGate()} if cls is pg.GatedPlanner else {}
                planner = cls(s6.method(), RankPredictorRegistry("g7-gate").pin(), device="cuda", eigh=eigh,
                              eigh_label=label, orthogonalizer="batched", max_working_bytes=2 * per_matrix, **extra)
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                plans[cls] = planner.plan(matrices, contexts, {key: ActiveBasisBinding() for key in matrices},
                                          {key: None for key in matrices}, chunk_id="c0",
                                          source_ids={key: f"g7-{key.layer}{key.target}" for key in matrices})
                torch.cuda.synchronize()
                peaks[cls] = torch.cuda.max_memory_allocated() - before
                if cls is pg.GatedPlanner:
                    self.assertTrue(planner.residency["lazy_sources"])
                    self.assertEqual(planner.residency["held_bytes_max"], 0)
            self.assertLess(peaks[pg.GatedPlanner], peaks[LatestMethodPlanner] - sources // 2, peaks)
            for key in matrices:
                x, y = plans[LatestMethodPlanner].plans[key], plans[pg.GatedPlanner].plans[key]
                self.assertEqual((x.decision, x.rank_view, x.decision_energy), (y.decision, y.rank_view,
                                                                                 y.decision_energy))
                self.assertTrue(torch.equal(x.new_rows, y.new_rows))


if __name__ == "__main__":
    unittest.main()
