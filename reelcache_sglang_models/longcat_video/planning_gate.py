"""Group-level GPU-planning gate for LongCat archival next to the served model (CR-17-6 follow-up, package g7).

g6 (freeze B2) gates GPU planning with the core's ``ArchivePipeline.defer_gpu_work``: a memory-critical section
(denoising, VAE decoding) waits until the archive fitter leaves its *node*. For LongCat the fitter's node is mostly
host work. The sources are the pinned host copies of the captured epoch (``deferred_sources="host"``), so the node's
source fingerprints and its finalization run on the host, and only the planner's matrix groups use the GPU. Yet a
section waited for the whole node: 47-105 s on A100-80GB.

Here the GPU part itself is gated, one planner *group* at a time:

* :class:`GatedPlanner` plans exactly like :class:`reelcache.rank_method.LatestMethodPlanner` (same groups, kernels and
  per-matrix arithmetic). Each group's sources (and inherited basis rows) are copied to the device when the group
  starts (the same ``tensor.to(device)`` / :func:`reelcache.deferred_reads.upload` the planner's admission used for
  the whole chunk) and dropped when it ends, together with the group's working tensors and new rows (downloaded,
  as the planner does). Between groups the planner holds nothing on the device. The sources stay in the
  pinned capture slots on the host, so nothing has to be offloaded when a section opens. LongCat's groups are
  independent (one matrix per eigensolver call from n = 4096: bounded and unbounded plans are bitwise identical),
  so the plans, and therefore the archive bytes and node ids, do not change. Admission (validation and the
  finiteness check) runs on the host copies, which costs about 1.4 s of host time per 96-matrix node.
* :class:`PlanningGate` is the worker's gate: every planner group is a *unit*. A unit starts only while no
  section is open or waiting, and a section waits only for the unit in progress (a fraction of a second), then
  returns the planner's cached device blocks (``release_cached_memory``) and runs with the planner off the device.
  Sections have priority, units wait. The fitter keeps taking nodes and running their host stages during sections.
* :class:`GatedGPUFitter` is :class:`reelcache.gpu_fitting.GPUFitter` with :class:`GatedPlanner` (same device checks,
  solver, stream and lock).

The planner subclass relies on ``LatestMethodPlanner._admit`` / ``_plan_group`` and on ``GPUFitter.plan_chunk``.
``tests/test_longcat_video_planning_gate_g7.py`` pins their source (a core change there fails the test and asks for
re-validation). The natural home of this behaviour is the core planner (Step 7); see DESIGN.md section 16.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any

import torch

from reelcache.deferred_reads import download, upload
from reelcache.errors import CapacityError, StateError
from reelcache.gpu_fitting import GPUFitter, select_eigh_solver
from reelcache.rank_method import LatestMethodPlanner

# The _KeyWork attributes that may hold device tensors after the group's planning (the planner itself clears
# scaled / residual / certify_projection / directions and downloads new rows); none is read after the group.
_TRANSIENT_ATTRIBUTES = ("sketches", "augment_result", "batched_orthogonal")


class PlanningGate:
    """The GPU-planning gate of one worker: planner units (groups) never run inside a memory-critical section.

    ``section()`` is the request side: it registers a waiting section (no new unit starts), waits for the unit in
    progress (bounded by ``timeout``: ``CapacityError``), optionally releases the CUDA caching allocator's unoccupied
    blocks (``release_cached_memory``), and runs. Sections nest and overlap across threads. ``unit(kind)`` is the
    planner side (one thread). It waits while any section is open or waiting, then holds the gate until the unit
    ends. A disabled gate (``configure(enabled=False)``) lets units run freely and sections enter at once.
    ``stats()`` reports the gate's counters.
    """

    def __init__(self, device: Any = None):
        self.device = None if device is None else torch.device(device)
        self._cond = threading.Condition()
        self.enabled = False
        self.release_cached_memory = False
        self._sections = 0
        self._waiting = 0
        self._unit: tuple[str, float] | None = None
        self._stats = {"units": 0, "unit_seconds": 0.0, "max_unit_seconds": 0.0, "max_unit_kind": None,
                       "units_deferred": 0, "unit_wait_seconds": 0.0, "sections": 0, "section_wait_seconds": 0.0,
                       "max_section_wait_seconds": 0.0, "sections_waited": 0, "released_bytes": 0,
                       "section_timeouts": 0, "resident_bytes_max": 0}

    def configure(self, *, enabled: bool, release_cached_memory: bool = False) -> None:
        if type(enabled) is not bool or type(release_cached_memory) is not bool:
            raise StateError("enabled and release_cached_memory must be boolean")
        with self._cond:
            self.enabled = enabled
            self.release_cached_memory = enabled and release_cached_memory
            self._cond.notify_all()

    @property
    def busy(self) -> bool:
        """A planner unit is in progress (the planner may hold device memory)."""
        return self._unit is not None

    def stats(self) -> dict:
        with self._cond:
            return {**self._stats, "enabled": self.enabled, "release_cached_memory": self.release_cached_memory,
                    "open_sections": self._sections, "busy": self._unit is not None,
                    "unit_kind": None if self._unit is None else self._unit[0]}

    def note_resident(self, nbytes: int) -> None:
        with self._cond:
            self._stats["resident_bytes_max"] = max(self._stats["resident_bytes_max"], int(nbytes))

    @contextlib.contextmanager
    def section(self, *, timeout: float | None = None):
        if not self.enabled:
            yield self
            return
        started = time.perf_counter()
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            self._waiting += 1
            try:
                while self._unit is not None:
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        self._stats["section_timeouts"] += 1
                        raise CapacityError("the archive planner did not finish its current group within the "
                                            "section's timeout")
                    self._cond.wait(remaining)
            finally:
                self._waiting -= 1
            self._sections += 1
            waited = time.perf_counter() - started
            self._stats["sections"] += 1
            self._stats["section_wait_seconds"] += waited
            self._stats["max_section_wait_seconds"] = max(self._stats["max_section_wait_seconds"], waited)
            self._stats["sections_waited"] += int(waited > 1e-3)
        try:
            if self.release_cached_memory and torch.cuda.is_available():
                device = self.device if self.device is not None else torch.device("cuda", torch.cuda.current_device())
                with torch.cuda.device(device):
                    before = torch.cuda.memory_reserved(device)
                    torch.cuda.empty_cache()
                    released = max(0, before - torch.cuda.memory_reserved(device))
                with self._cond:
                    self._stats["released_bytes"] += released
            yield self
        finally:
            with self._cond:
                self._sections -= 1
                self._cond.notify_all()

    @contextlib.contextmanager
    def unit(self, kind: str):
        started = time.perf_counter()
        with self._cond:
            if self._unit is not None:
                raise StateError("planner units do not nest")
            deferred = False
            while self.enabled and (self._sections or self._waiting):
                deferred = True
                self._cond.wait()
            began = time.perf_counter()
            self._unit = (kind, began)
            if deferred:
                self._stats["units_deferred"] += 1
                self._stats["unit_wait_seconds"] += began - started
        try:
            yield self
        finally:
            with self._cond:
                seconds = time.perf_counter() - began
                self._unit = None
                self._stats["units"] += 1
                self._stats["unit_seconds"] += seconds
                if seconds > self._stats["max_unit_seconds"]:
                    self._stats["max_unit_seconds"], self._stats["max_unit_kind"] = seconds, kind
                self._cond.notify_all()


def _cuda_storages(value, found: dict, depth: int = 0) -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            storage = value.untyped_storage()
            found[storage.data_ptr()] = storage.nbytes()
        return
    if depth >= 3:
        return
    if isinstance(value, dict):
        for item in value.values():
            _cuda_storages(item, found, depth + 1)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _cuda_storages(item, found, depth + 1)
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        for item in vars(value).values():
            _cuda_storages(item, found, depth + 1)


def device_bytes_held(works) -> int:
    """Device bytes the planner's per-matrix states reference (distinct storages; diagnostics)."""
    found: dict = {}
    for work in works:
        _cuda_storages(vars(work), found)
    return sum(found.values())


class GatedPlanner(LatestMethodPlanner):
    """:class:`LatestMethodPlanner` whose groups are :class:`PlanningGate` units and whose sources are on the device
    only while their group plans (module docstring). ``lazy_sources=False`` keeps the planner's admission (the whole
    chunk uploaded at once) and only gates the groups. Host sources are required for lazy placement; a chunk with
    any source already on the device is planned as the base planner does (e.g. the synthetic warm-up chunk).
    ``residency`` reports the groups, the bytes placed per chunk and the device bytes the per-matrix states still
    held after each group (largest)."""

    def __init__(self, *args, gate: PlanningGate, lazy_sources: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(gate, PlanningGate):
            raise StateError("GatedPlanner needs a PlanningGate")
        self.gate = gate
        self.lazy_sources = bool(lazy_sources)
        self._lazy = False
        self._works: list = []
        self.residency = {"lazy_sources": False, "groups": 0, "placed_bytes": 0, "held_bytes_max": 0}

    # -- placement (tests substitute these to account the device copies) ---------------------------------------
    def _place_matrices(self, tensors: list) -> list:
        return [tensor.to(self.device) for tensor in tensors]

    def _place_rows(self, tensors: list) -> list:
        return upload(tensors, self.device)

    def _admit(self, keys, sources, contexts, bindings, parent_rows, source_ids):
        lazy = self.lazy_sources and all(isinstance(sources[key], torch.Tensor) and sources[key].device.type == "cpu"
                                         for key in keys)
        if not lazy:
            with self.gate.unit("admit"):
                works = super()._admit(keys, sources, contexts, bindings, parent_rows, source_ids)
        else:
            # The planner's admission on the host copies: identical validation, finiteness and error order; the
            # device copies are made per group (_plan_group).
            device, self.device = self.device, torch.device("cpu")
            try:
                works = super()._admit(keys, sources, contexts, bindings, parent_rows, source_ids)
            finally:
                self.device = device
        self._lazy = lazy
        self._works = works
        self.residency["lazy_sources"] = lazy
        return works

    def _plan_group(self, clock, works, chunk_id: str) -> None:
        with self.gate.unit("group"):
            if not self._lazy:
                super()._plan_group(clock, works, chunk_id)
                self._finish_group(works)
            else:
                hosts = [(work.matrix, work.parent_rows16) for work in works]
                try:
                    placed = self._place_matrices([work.matrix for work in works])
                    for work, matrix in zip(works, placed):
                        work.matrix = matrix
                    inheriting = [work for work in works if work.parent_rows16 is not None]
                    for work, rows in zip(inheriting, self._place_rows([work.parent_rows16 for work in inheriting])):
                        work.parent_rows16 = rows
                    self.residency["placed_bytes"] += sum(t.numel() * t.element_size() for t in placed)
                    del placed
                    super()._plan_group(clock, works, chunk_id)
                    self._finish_group(works)
                finally:
                    for work, (matrix, rows) in zip(works, hosts):
                        work.matrix, work.parent_rows16 = matrix, rows
                        for name in _TRANSIENT_ATTRIBUTES:
                            value = getattr(work, name, None)
                            if isinstance(value, dict):
                                setattr(work, name, {})
                            elif value is not None:
                                setattr(work, name, None)
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            self.residency["groups"] += 1
            held = device_bytes_held(self._works)
            self.residency["held_bytes_max"] = max(self.residency["held_bytes_max"], held)
            if self._lazy:
                self.gate.note_resident(held)

    @staticmethod
    def _finish_group(works) -> None:
        """What ``LatestMethodPlanner.plan`` does after each group (new rows to the host in one transfer, working
        tensors released), done inside the unit; the planner's own pass then finds nothing left to do."""
        pending = [work.plan for work in works
                   if work.plan.new_rows is not None and work.plan.new_rows.device.type != "cpu"]
        if pending:
            for plan, rows in zip(pending, download([plan.new_rows for plan in pending])):
                plan.new_rows = rows
        for work in works:
            work.scaled = work.residual = work.certify_projection = None
            work.directions.clear()


class GatedGPUFitter(GPUFitter):
    """:class:`reelcache.gpu_fitting.GPUFitter` planning with :class:`GatedPlanner` under ``gate`` (module docstring).
    ``last_residency``: the latest chunk's :attr:`GatedPlanner.residency`."""

    def __init__(self, *, gate: PlanningGate | None = None, lazy_sources: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.gate = PlanningGate(device=torch.device("cuda", self.gpu_id)) if gate is None else gate
        self.lazy_sources = bool(lazy_sources)
        self.last_residency: dict | None = None

    def plan_chunk(self, config, view, sources, contexts, bindings, parent_rows, *, chunk_id, source_ids,
                   orthogonalizer="householder", max_working_bytes=None):
        """``GPUFitter.plan_chunk`` with :class:`GatedPlanner` (the same owner check, device, solver, stream, lock)."""
        self._check_owner()
        device = self._device()
        label, solver = select_eigh_solver(self.eigh_solver)
        planner = GatedPlanner(config, view, device=device, eigh=solver, eigh_label=label,
                               orthogonalizer=orthogonalizer, max_working_bytes=max_working_bytes, gate=self.gate,
                               lazy_sources=self.lazy_sources)
        if self.side_stream and self._stream is None:
            self._stream = torch.cuda.Stream(device)
        scope = self._lock if self.shared else contextlib.nullcontext()
        with scope, torch.cuda.device(device), (
            torch.cuda.stream(self._stream) if self.side_stream else contextlib.nullcontext()
        ):
            plan = planner.plan(sources, contexts, bindings, parent_rows, chunk_id=chunk_id, source_ids=source_ids)
        self.prefit_stats["solver"] = label
        self.last_residency = dict(planner.residency)
        return plan


__all__ = ["GatedGPUFitter", "GatedPlanner", "PlanningGate", "device_bytes_held"]
