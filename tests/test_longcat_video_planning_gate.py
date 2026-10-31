"""CPU tests of LongCat-Video archival planning gated across sessions (CR-17-6; no SGLang, no GPU).

On a device where the eager-overlap memory check does not fit (A100-80GB), ``schedule: auto`` gives ``deferred`` and
``defer_planning`` is forced on: one gated planner per worker never plans while any request -- another session's or a
plain one -- is inside a memory-critical section (denoising, VAE decoding), and the planner's cached device memory is
released when a section opens. A100 job 4410238 ran out of memory without it: a non-durable request's epoch was
planned (its node sources on the device) while the next session denoised.

These tests drive the provider like the port's stages (restore or build the cache outside the sections; denoise and
decode inside ``session_provider.memory_critical_section``) with a CPU archive pipeline whose fitter is gated the same
way, and account the planner's per-node device memory (sources + workspace, held from the start of planning to the
end of finalization -- what the core frees when a node leaves the GPU) with a mocked ledger.
"""

from __future__ import annotations

import contextlib
import threading
import time
import unittest
from unittest import mock

import torch

from reelcache_sglang_models.longcat_video import conditioning_cache as cc
from reelcache_sglang_models.longcat_video import session_provider as sp
from reelcache_sglang_models.longcat_video.segment_resume import generator_state_hex, set_generator_state_hex
from tests.test_longcat_video_session_provider import (
    FINGERPRINT,
    GEOMETRY,
    GRID,
    conditioning_cache,
    frames_digest,
    initial_frames,
    make_provider,
    next_frames,
)

GIB = 1 << 30


class PlannerLedger:
    """Mocked device memory of the planner: a node's sources from the start of planning to the end of finalization."""

    def __init__(self, plan_seconds: float):
        from reelcache.lineage_commit import CommitJob

        self.plan_seconds = plan_seconds
        self.lock = threading.Lock()
        self.bytes = 0
        self.planned = 0
        self.in_section_max = 0
        self.violations: list = []
        real_plan, real_finalize = CommitJob.plan, CommitJob.finalize
        ledger = self

        def plan(job, *args, **kwargs):
            nbytes = sum(t.numel() * t.element_size() for item in job.ranges for t in item.sources.values())
            with ledger.lock:
                ledger.bytes += nbytes
                job._ledger_bytes = nbytes
            time.sleep(ledger.plan_seconds)  # a node takes a while on the planner
            return real_plan(job, *args, **kwargs)

        def finalize(job, *args, **kwargs):
            try:
                return real_finalize(job, *args, **kwargs)
            finally:
                with ledger.lock:
                    ledger.bytes -= getattr(job, "_ledger_bytes", 0)
                    ledger.planned += 1

        self.patches = [mock.patch.object(CommitJob, "plan", plan), mock.patch.object(CommitJob, "finalize", finalize)]

    def __enter__(self):
        for patch in self.patches:
            patch.start()
        return self

    def __exit__(self, *exc):
        for patch in self.patches:
            patch.stop()

    def probe(self, where: str) -> None:
        with self.lock:
            self.in_section_max = max(self.in_section_max, self.bytes)
            if self.bytes:
                self.violations.append((where, self.bytes))

    def wait_busy(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.bytes:
                    return True
            time.sleep(0.005)
        return False


class OtherBackend:
    """A request naming another registered cache backend that has its own section."""

    def __init__(self):
        self.entered = 0

    @contextlib.contextmanager
    def _section(self):
        self.entered += 1
        yield

    def generation_section(self):
        return self._section()


def stage_request(provider, handle: dict | None, ledger: PlannerLedger, *, frames=None, prompt: str = "a cat",
                  seed: int = 0, num_segments: int = 1, denoise: float = 0.05, decode: float = 0.02,
                  slots: list | None = None, backend=None) -> dict:
    """One request as the port's stages run it: cache restore/build outside the sections, denoising and VAE
    decoding inside ``memory_critical_section`` (the request's cache backend's and the worker's)."""
    if handle is None:  # a plain request on a worker with sessions enabled: no session run
        waits = []
        for seconds in (denoise, decode):
            started = time.perf_counter()
            with sp.memory_critical_section(backend):
                waits.append(time.perf_counter() - started)
                ledger.probe("plain:enter")
                time.sleep(seconds)
                ledger.probe("plain:exit")
        return {"planning_deferral_wait_seconds": sum(waits)}
    fields = {"request_id": None, "prompt": prompt, "seed": seed, "num_segments": num_segments, "num_frames": 93,
              "session_id": None, "start_segment_index": 0, "generator_state": None,
              "conditioning_cache_backend": None, "has_client_frames": frames is not None, "injected": []}
    run = provider.begin_request(fields, handle)
    generator = torch.Generator().manual_seed(seed)
    if run.plan.generator_state is not None:
        set_generator_state_hex(generator, run.plan.generator_state)
    cond = frames if run.plan.frames is None else run.plan.frames
    run.bind_request(frames=cond, generator_state=generator_state_hex(generator), geometry=GEOMETRY)
    error = None
    try:
        for seg in range(run.plan.start_segment, run.plan.start_segment + run.plan.num_segments):
            epoch = cc.ConditioningCacheEpoch(segment_index=seg, token_grid=GRID, cond_latents_sha256=frames_digest(cond),
                                              model_fingerprint=FINGERPRINT, session_id=run.session.session_id,
                                              parent_segment_index=seg - 1 if seg else None)
            cache = provider.cache_backend.restore(epoch)
            if cache is None:
                cache = conditioning_cache(cond, seg, run.session.session_id)
                provider.cache_backend.on_cache_built(cache)
            elif cache.ready is not None:  # an overlapped restore (d5 default): joined as the stage joins it
                for layer in range(cache.ready.layers):
                    cache.ready.wait(layer)
                cache.ready.join()
                cache.ready = None
            if slots is not None:
                slots.append(len(provider.backend._pending))
            rec = {"planning_deferral_wait_seconds": 0.0}
            for name, seconds in (("denoise", denoise), ("decode", decode)):
                started = time.perf_counter()
                with sp.memory_critical_section(provider.cache_backend):
                    rec["planning_deferral_wait_seconds"] += time.perf_counter() - started
                    ledger.probe(f"{run.session.session_id}:{seg}:{name}:enter")
                    time.sleep(seconds)
                    ledger.probe(f"{run.session.session_id}:{seg}:{name}:exit")
            cond = next_frames(cond, prompt, generator)
            record = {"epoch": epoch.as_dict(), "cache_source": cc.SOURCE_NATIVE_BUILD, "generator_state": "start-state",
                      **rec}
            if run.end_segment(segment_index=seg, record=record, next_frames=cond,
                               generator_state=generator_state_hex(generator), prompt=prompt):
                break
    except BaseException as exc:  # noqa: BLE001 - the stage reports it, then re-raises
        error = exc
        raise
    finally:
        summary = run.finish(error=error)
    return summary


class PlanningGateTests(unittest.TestCase):
    def setUp(self):
        import shutil
        import tempfile
        from pathlib import Path

        self.root = Path(tempfile.mkdtemp(prefix="longcat-gate-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def provider(self, **config) -> sp.LongCatSessionProvider:
        provider = make_provider(self.root / "store", durability="never", **config)
        # the gate as create_provider sets it where the eager-overlap check fails (its CUDA check bypassed: the gate
        # itself is device-free), and the provider registered as the worker's (plain requests use its section)
        provider.backend.defer_planning = True
        sp._ACTIVE[:] = [provider]
        self.addCleanup(sp._ACTIVE.clear)
        self.addCleanup(self._close, provider)
        return provider

    @staticmethod
    def _close(provider):
        try:
            provider.close()
        except Exception:  # noqa: BLE001
            pass

    def test_schedule_forces_gating_where_eager_overlap_does_not_fit(self):
        from reelcache.errors import ConfigurationError

        rtx, a100, resident = int(94.97 * GIB), int(79.25 * GIB), int(37.4 * GIB)

        def decide(requested, total, *, device="cuda", defer=False, representation="svd_hevc"):
            return sp.resolve_schedule(requested, defer_planning=defer, plan_device=device, plan_memory_bytes=8 * GIB,
                                       resident_bytes=resident, total_bytes=total, representation=representation)

        a100_auto = decide("auto", a100)
        self.assertEqual((a100_auto["chosen"], a100_auto["defer_planning"], a100_auto["release_planner_memory"]),
                         ("deferred", True, True))
        self.assertIn("never overlaps any request's denoising or decoding", a100_auto["defer_planning_reason"])
        self.assertIn("never during any request's denoising or decoding", a100_auto["reason"])
        explicit = decide("deferred", a100)  # an explicit deferred schedule is gated there too
        self.assertEqual((explicit["chosen"], explicit["defer_planning"]), ("deferred", True))
        self.assertEqual(decide("auto", a100, defer=True)["chosen"], "eager")  # eager + defer, memory released
        self.assertTrue(decide("auto", a100, defer=True)["release_planner_memory"])
        with self.assertRaisesRegex(ConfigurationError, "defer_planning"):
            decide("eager", a100)  # explicit eager without the gate stays refused
        for decision in (decide("auto", rtx), decide("deferred", rtx), decide("auto", a100, device="cpu"),
                         decide("auto", a100, representation="raw")):
            self.assertEqual((decision["defer_planning"], decision["release_planner_memory"]), (False, False))
        self.assertEqual(decide("auto", rtx)["chosen"], "eager")
        self.assertEqual(decide("auto", rtx, defer=True)["defer_planning"], True)

    def test_planning_never_overlaps_another_sessions_sections(self):
        provider = self.provider()
        with PlannerLedger(plan_seconds=0.4) as ledger:
            # session A ends a non-durable request: its epoch is submitted and planned behind generation
            first = stage_request(provider, {"session_id": "a", "action": "generate"}, ledger,
                                  frames=initial_frames(31))
            self.assertEqual(first["archival"], "behind generation")
            self.assertTrue(ledger.wait_busy())  # the planner holds A's node on the "device" now
            # session B starts at once: its sections wait for A's node, then run with the planner idle
            second = stage_request(provider, {"session_id": "b", "action": "generate"}, ledger,
                                   frames=initial_frames(32), denoise=0.3)
            waits = [segment["planning_deferral_wait_seconds"] for segment in second["segments"]]
            self.assertGreater(waits[0], 0.05)  # the second session's added latency: A's node finished first
            # a plain request (no session) is gated by the worker's provider as well
            stage_request(provider, {"session_id": "c", "action": "generate"}, ledger, frames=initial_frames(33))
            self.assertTrue(ledger.wait_busy())
            plain = stage_request(provider, None, ledger, denoise=0.2)
            self.assertGreater(plain["planning_deferral_wait_seconds"], 0.0)
            # a request naming another backend with its own section: that section and the worker's gate
            stage_request(provider, {"session_id": "d", "action": "generate"}, ledger, frames=initial_frames(34))
            self.assertTrue(ledger.wait_busy())
            other = OtherBackend()
            named = stage_request(provider, None, ledger, denoise=0.2, backend=other)
            self.assertEqual(other.entered, 2)
            self.assertGreater(named["planning_deferral_wait_seconds"], 0.0)
            provider.backend.flush(wait=True)
        self.assertEqual(ledger.violations, [])  # nothing of the planner resident inside any section
        self.assertEqual(ledger.in_section_max, 0)
        self.assertEqual(ledger.bytes, 0)
        self.assertEqual(ledger.planned, 4)
        stats = provider.backend.pipeline.stats()
        self.assertGreaterEqual(stats["deferrals"], 8)
        self.assertGreater(stats["deferral_wait_seconds"], 0.0)
        for sid in ("a", "b", "c", "d"):
            for segment in provider.control("describe", {"session_id": sid})["segments"]:
                self.assertEqual(segment["cache_epoch_now"], "published")
                self.assertIn("planning_deferral_wait_seconds", segment)

    def test_sections_without_a_worker_provider(self):
        sp._ACTIVE.clear()
        self.assertIsNone(sp.memory_critical_section(None))
        self.assertIsNone(sp.memory_critical_section(object()))  # a backend without a section
        other = OtherBackend()
        with sp.memory_critical_section(other):
            pass
        self.assertEqual(other.entered, 1)
        provider = self.provider()  # the worker's provider: a session request enters its own section once
        before = provider.backend.pipeline.stats()["deferrals"]
        with sp.memory_critical_section(provider.cache_backend):
            pass
        self.assertEqual(provider.backend.pipeline.stats()["deferrals"] - before, 1)

    def test_without_the_gate_the_planner_overlaps_another_sessions_sections(self):
        """The A100 4410238 failure mode (control): ungated, a non-durable request's planning overlaps the next
        session's denoising."""
        provider = self.provider()
        provider.backend.defer_planning = False
        with PlannerLedger(plan_seconds=0.4) as ledger:
            stage_request(provider, {"session_id": "a", "action": "generate"}, ledger, frames=initial_frames(31))
            self.assertTrue(ledger.wait_busy())
            stage_request(provider, {"session_id": "b", "action": "generate"}, ledger, frames=initial_frames(32),
                          denoise=0.3)
            provider.backend.flush(wait=True)
        self.assertTrue(ledger.violations)
        self.assertGreater(ledger.in_section_max, 0)

    def test_back_to_back_sessions_progress_with_a_bounded_backlog(self):
        slots_bound = 2
        provider = self.provider(capture_slots=slots_bound)
        results, errors, pending = [], [], []

        def serve():
            try:
                with PlannerLedger(plan_seconds=0.15) as ledger:
                    results.append(ledger)
                    for round_index in range(2):
                        for index, sid in enumerate(("p", "q", "r")):
                            handle = ({"session_id": sid, "action": "generate"} if round_index == 0
                                      else {"session_id": sid, "action": "resume"})
                            frames = initial_frames(40 + index) if round_index == 0 else None
                            results.append(stage_request(provider, handle, ledger, frames=frames, num_segments=2,
                                                         slots=pending))
                    results.append({"planned_before_final_drain": ledger.planned})
                    provider.backend.flush(wait=True)
            except BaseException as exc:  # noqa: BLE001 - reported by the test thread
                errors.append(exc)

        thread = threading.Thread(target=serve, name="longcat-gate-serve", daemon=True)
        thread.start()
        thread.join(timeout=180)
        self.assertFalse(thread.is_alive(), "back-to-back requests deadlocked")
        self.assertEqual(errors, [])
        ledger, summaries, tail = results[0], results[1:-1], results[-1]
        self.assertEqual(len(summaries), 6)
        self.assertEqual(ledger.violations, [])
        self.assertLessEqual(max(pending), slots_bound)  # the archival backlog never exceeds the capture slots
        self.assertGreater(sum(summary["backpressure_seconds"] for summary in summaries), 0.0)  # it back-pressured
        self.assertGreater(tail["planned_before_final_drain"], 0)  # planning progressed between requests/sections
        self.assertEqual(ledger.planned, 12)  # 3 sessions x 2 requests x 2 segments: the planner completed
        self.assertEqual(provider.backend.pending(), 0)
        for sid in ("p", "q", "r"):
            described = provider.control("describe", {"session_id": sid})
            self.assertEqual(described["generated_segments"], 4)
            self.assertEqual({segment["cache_epoch_now"] for segment in described["segments"]}, {"published"})


if __name__ == "__main__":
    unittest.main()
