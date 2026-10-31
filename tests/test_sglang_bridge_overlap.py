"""Layer-overlapped restore in the SGLang bridge (freeze F5: Step 11's bridge on Step 5c's O3; default on).

The toy stage emulates SGLang's seam: every self-attention cache carries ``restore_pending`` and waits on it once,
before its first read or write (``update_and_get_attention_kv``). Each new chunk runs a denoising forward (step 0)
that reads every layer's history before its clean forward.

* The overlap is the default (``ReelCacheRuntimeConfig.overlap_restore``, on when the config has no such field). A
  request's session handle ``overlap_restore`` (bool) turns it off or on for that request. RAW reference archives
  never overlap.
* An overlapped restore gives bitwise the latents and caches of the plain restore. Step 0 waits per layer. The run
  joins after the first clean forward, before anything is committed.
* An unprepared history falls back to the plain call. A failed overlapped restore fails the request, typed, and leaves
  no pending wait behind. A request failing during step 0 stops its restore (O3 ``quiesce``).
* LongLive 1.0: a resume at a prompt switch overlaps through its recache forward, and a restored recache epoch
  restores bitwise. CFG streams restore in chained calls, cond first.
"""

import dataclasses
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from reelcache.errors import ConfigurationError, IntegrityError
from reelcache.integrations.sglang.config import ReelCacheRuntimeConfig
from tests.test_sglang_bridge import (FRAMES, GRID, LAYERS, TPF, BridgeTestCase, Capture, Chunk, FullHistoryCache,
                                     Recache, handle, request, rope_cos_sin, rotate_key, switch_of, timeline_request,
                                     toy_kv)
from tests.test_sglang_bridge import drive as seamless_drive


class PendingCache(FullHistoryCache):
    """The toy cache with SGLang's seam: a pending background restore is awaited once before any access."""

    def __init__(self, capacity):
        super().__init__(capacity)
        self.restore_pending = None
        self.awaited = 0

    def await_restored_history(self):
        pending, self.restore_pending = self.restore_pending, None
        if pending is not None:
            self.awaited += 1
            pending.wait()

    def update(self, key, value, start):
        self.await_restored_history()
        super().update(key, value, start)

    def overwrite(self, start, key, value):  # a recache's recompute write (SGLang: the same update method)
        self.await_restored_history()
        super().overwrite(start, key, value)


class OverlapStage:
    """The stage keeps its caches across requests and resets only their cursors when a request starts."""

    def __init__(self):
        self.caches = None

    def existing(self):
        return self.caches

    def caches_for(self, streams):
        if self.caches is None or tuple(self.caches) != tuple(streams):
            self.caches = {stream: [PendingCache(8 * FRAMES * TPF) for _ in range(LAYERS)] for stream in streams}
        else:
            for caches in self.caches.values():
                for cache in caches:
                    cache.reset_indices()
        return self.caches


def toy_rope(frames, height, width, *, start_frame, device):
    """The engine RoPE as SGLang's transformer exposes it (``causal_rotary_cos_sin``)."""
    return rope_cos_sin(start_frame, frames)


def recache(run, req, caches, at, *, written=None):
    """The toy LongLive 1.0 prompt switch before block ``at`` (``tests.test_sglang_bridge.recache``), with the
    overlap seam: each layer waits for its restored history before its recompute reads it. This is SGLang's
    ``update_and_get_attention_kv``, the recache forward's attention."""
    switch = switch_of(req)
    first = at - 2
    window = Chunk(-1, first * FRAMES, 2 * FRAMES, first * FRAMES * TPF, 2 * FRAMES * TPF, float(first * FRAMES),
                   False, 0)
    if written is None:
        written = ((window.token_start, window.num_tokens),)
    cos, sin = rope_cos_sin(window.rope_start_frame, window.num_frames)
    for stream in req.streams:
        capture = Capture(window, run.stream_of(caches[stream]), "kv_recache")
        run.begin_clean_forward(capture)
        for layer in range(LAYERS):
            caches[stream][layer].await_restored_history()
            key, value = toy_kv(switch["prompt"], window, layer, stream, caches[stream][layer])
            run.capture_layer(capture, layer, key, value, (cos, sin))
            caches[stream][layer].overwrite(window.token_start, rotate_key(key, cos, sin), value)
        run.end_clean_forward(capture)
    return run.end_recache(Recache(window, tuple(written), f"switch-{at}", details={"prompt": switch["prompt"]}))


def drive(bridge, req, stage, *, fail_in_step0=False):
    """One request like SGLang's worker and causal stage with the overlap seam. Each new chunk runs a denoising
    step 0 that reads every layer, then the clean forward. Returns (summary, latents, caches, None)."""
    req.dispatched_at = time.perf_counter()  # the worker's dispatch (first-new-chunk accounting)
    if True:
        latents = torch.zeros(1, 4, sum(req.block_sizes), *GRID)
        generator = torch.Generator().manual_seed(req.conditioning["seed"])
        torch.randn(3, generator=generator)  # latent preparation draws
        caches = stage.caches_for(req.streams)
        run = bridge.begin_request(req)
        target = SimpleNamespace(caches=caches, latents=latents, generator=generator, rope_cos_sin=rope_cos_sin,
                                 rotate_key=rotate_key, device=torch.device("cpu"))
        try:
            resume = run.restore(target)
            start = 0
            for index, frames in enumerate(req.block_sizes):
                if index < resume:
                    start += frames
                    continue
                switch = switch_of(req)
                if switch is not None and index == switch["at"]:
                    # the restored head says whether the switch's recache epoch was restored (rerun it otherwise)
                    if not (index == resume and run.restore_report.get("head_kind") == "recache"):
                        recache(run, req, caches, index, written=switch.get("written"))
                chunk = Chunk(index, start, frames, start * TPF, frames * TPF, float(start), False, 0)
                step0 = []
                for stream in req.streams:  # step 0: every attention layer reads its history first (cond, uncond)
                    for layer in range(LAYERS):
                        if fail_in_step0 and layer == 1:
                            raise RuntimeError("denoising step failed")
                        cache = caches[stream][layer]
                        cache.await_restored_history()
                        step0.append(float(cache.k[:, :cache.end].float().sum()))
                history = caches["cond"][0].k[:, :caches["cond"][0].end].float().mean() if caches["cond"][0].end \
                    else 0.0
                noise = torch.randn(1, 4, frames, *GRID, generator=generator)
                latents[:, :, start:start + frames] = noise + float(history) + 1e-6 * sum(step0)
                cos, sin = rope_cos_sin(chunk.rope_start_frame, frames)
                prompt = req.conditioning["prompt"] if switch is None or index < switch["at"] else switch["prompt"]
                for stream in req.streams:
                    capture = Capture(chunk, run.stream_of(caches[stream]))
                    run.begin_clean_forward(capture)
                    for layer in range(LAYERS):
                        key, value = toy_kv(prompt, chunk, layer, stream, caches[stream][layer])
                        run.capture_layer(capture, layer, key, value, (cos, sin))
                        caches[stream][layer].update(rotate_key(key, cos, sin), value, chunk.token_start)
                    run.end_clean_forward(capture)
                start += frames
                if run.end_chunk(chunk, latents[:, :, start - frames:start].clone()):
                    break
            return run.finish(error=None), latents, caches, None
        except BaseException as exc:
            run.finish(error=exc)
            raise


class OverlapTestCase(BridgeTestCase):
    representation = "svd_hevc"

    def bridge(self, store="archive", profile=None, **overrides):
        overrides.setdefault("raw_restore", "off")  # the compressed restore (prepared packs on GPU)
        bridge = super().bridge(store, **({} if profile is None else {"profile": profile}), **overrides)
        bridge.overlap_require_prepared = False  # the CPU archive's software restorer has no prepared-pack cache
        return bridge

    def session_with_history(self, bridge, stage, chunks=3):
        summary, _, _, _ = drive(bridge, request(handle("L", pause_after_chunks=chunks)), stage)
        bridge.drain_all()
        return summary

    def fork(self, bridge, stage, name, *, overlap=None, checkpoint=2, **kwargs):
        extra = {} if overlap is None else {"overlap_restore": overlap}
        req = request(handle(name, "fork", from_session="L", checkpoint=checkpoint,
                             pause_after_chunks=checkpoint + 1, **extra))
        return drive(bridge, req, stage, **kwargs)

    def assertSameCaches(self, left, right):
        for stream in right:
            for a, b in zip(left[stream], right[stream]):
                self.assertEqual(a.end, b.end)
                self.assertTrue(torch.equal(a.k[:, :a.end], b.k[:, :b.end]))
                self.assertTrue(torch.equal(a.v[:, :a.end], b.v[:, :b.end]))

    def assertNothingPending(self, caches):
        self.assertTrue(all(cache.restore_pending is None for items in caches.values() for cache in items))

    @staticmethod
    def snapshot(latents, caches):
        return latents.clone(), {stream: [SimpleNamespace(end=c.end, k=c.k.clone(), v=c.v.clone()) for c in items]
                                 for stream, items in caches.items()}


class OverlappedRestoreTests(OverlapTestCase):
    def test_first_forward_overlap_is_bitwise_the_plain_restore(self):
        bridge, stage = self.bridge(), OverlapStage()
        self.session_with_history(bridge, stage)
        plain, plain_latents, plain_caches, _ = self.fork(bridge, stage, "p", overlap=False)
        expected = {"latents": plain_latents.clone(), "caches": {"cond": [SimpleNamespace(
            end=cache.end, k=cache.k.clone(), v=cache.v.clone()) for cache in plain_caches["cond"]]}}
        self.assertIsNone(plain["restore"]["overlap"])
        summary, latents, caches, _ = self.fork(bridge, stage, "o", overlap=True)
        self.assertTrue(torch.equal(latents, expected["latents"]))
        self.assertSameCaches(caches, expected["caches"])
        record = summary["restore"]["overlap"]
        self.assertEqual((record["mode"], record["calls"], record["fallback"], record["error"]),
                         ("first_forward", 1, None, None))
        self.assertEqual(record["layers_waited"], LAYERS)  # step 0 waited on every layer, in order
        self.assertEqual([item["layer"] for item in record["waits"]], list(range(LAYERS)))
        self.assertIsNotNone(record["joined_at_seconds"])
        self.assertEqual(summary["restore"]["lease"]["overlap"]["calls"], 1)
        self.assertTrue(summary["restore"]["lease"]["closed"])
        self.assertIsNotNone(summary["restore"]["lease"]["verification"])
        self.assertTrue(all(cache.awaited == 1 for cache in caches["cond"]))
        self.assertNothingPending(caches)
        self.assertEqual(summary["chunk_digests"], plain["chunk_digests"])
        self.assertLifetimesRetired(bridge, summary)

    def test_a_layer_ordered_restorer_releases_early_layers_first_and_is_bitwise(self):
        """5c's O3 with the production restorer class (``LineageRestorer``, CPU decoder pool): prepared packs, layers
        released in order while later ones are reconstructed; bitwise the plain restore through the same restorer."""
        from reelcache.archive_restore import LineageRestorer
        from tests.test_packed_nvdec import SoftwarePool

        bridge, stage = self.bridge(), OverlapStage()
        bridge.overlap_require_prepared = True
        self.session_with_history(bridge, stage)
        restorer = LineageRestorer(bridge.archive.runtime, SoftwarePool(), arena_bytes=1 << 30)
        self.addCleanup(restorer.close)
        bridge.archive.restorer = restorer
        head = bridge.controller._resolve_checkpoint(bridge.controller.get("L"), 2)
        restorer.prewarm(restorer.history(head))
        _, plain_latents, plain_caches, _ = self.fork(bridge, stage, "p", overlap=False)
        expected_latents = plain_latents.clone()
        expected = {"cond": [SimpleNamespace(end=c.end, k=c.k.clone(), v=c.v.clone()) for c in plain_caches["cond"]]}
        summary, latents, caches, _ = self.fork(bridge, stage, "o", overlap=True)
        record = summary["restore"]["overlap"]
        self.assertEqual((record["calls"], record["fallback"], record["error"]), (1, None, None))
        [producer] = record["producer"]
        self.assertTrue(producer["progressive"])
        [timeline] = record["timelines"]
        marks = timeline["marked_at"]
        self.assertTrue(all(mark is not None for mark in marks))
        self.assertEqual(marks, sorted(marks))  # layer order
        self.assertTrue(torch.equal(latents, expected_latents))
        self.assertSameCaches(caches, expected)
        self.assertNothingPending(caches)
        self.assertLifetimesRetired(bridge, summary)

    def test_the_restore_thread_writes_inference_tensor_caches_in_the_callers_mode(self):
        """SGLang serves under torch.inference_mode: its caches are inference tensors. Step 5c's O3 runs the restore
        thread in the caller's mode, so the overlapped restore writes them as the plain restore does (P1 needed a
        wrapper for this)."""
        bridge, stage = self.bridge(), OverlapStage()
        with torch.inference_mode():
            self.session_with_history(bridge, stage)
            _, plain_latents, _, _ = self.fork(bridge, stage, "p", overlap=False)
            expected = plain_latents.clone()
            summary, latents, caches, _ = self.fork(bridge, stage, "o", overlap=True)
        self.assertTrue(caches["cond"][0].k.is_inference())
        record = summary["restore"]["overlap"]
        self.assertEqual((record["calls"], record["error"], record["layers_waited"]), (1, None, LAYERS))
        self.assertTrue(torch.equal(latents, expected))
        self.assertLifetimesRetired(bridge, summary)

    def test_a_request_failing_during_step0_stops_the_restore_without_waiting_for_it(self):
        """O3 ``quiesce``: a failed request cancels its overlapped restore at the next layer, joins the thread and
        leaves no pending wait; the worker continues."""
        bridge, stage = self.bridge(), OverlapStage()
        self.session_with_history(bridge, stage)
        with self.assertRaisesRegex(RuntimeError, "denoising step failed"):
            self.fork(bridge, stage, "o", overlap=True, fail_in_step0=True)
        failed = bridge.request_summary(bridge.requests[-1]["request_key"])
        record = failed["restore"]["overlap"]
        self.assertIn("request failed", record["aborted"])
        self.assertTrue(failed["restore"]["lease"]["closed"])
        self.assertNothingPending(stage.caches)
        self.assertFalse(bridge.active_request)
        summary, _, _, _ = self.fork(bridge, stage, "again", overlap=True)
        self.assertEqual(summary["status"], "paused")
        self.assertEqual(set(summary["restore"]["overlap"]["setup_seconds"]),
                         {"section_entry", "install_cursors", "rope_tables_and_pool_snapshot", "archive_restore_start"})
        self.assertLifetimesRetired(bridge, summary)

    def test_an_unprepared_history_falls_back_to_the_plain_call(self):
        bridge, stage = self.bridge(), OverlapStage()
        bridge.overlap_require_prepared = True  # the software restorer has no prepared packs: StateError
        self.session_with_history(bridge, stage)
        _, plain_latents, _, _ = self.fork(bridge, stage, "p", overlap=False)
        expected = plain_latents.clone()
        summary, latents, caches, _ = self.fork(bridge, stage, "o", overlap=True)
        record = summary["restore"]["overlap"]
        self.assertIn("prepared", record["fallback"])
        self.assertEqual((record["calls"], record["layers_waited"]), (0, 0))
        self.assertTrue(torch.equal(latents, expected))
        self.assertNothingPending(caches)
        self.assertLifetimesRetired(bridge, summary)

    def test_a_failed_overlapped_restore_fails_the_request_typed(self):
        bridge, stage = self.bridge(), OverlapStage()
        self.session_with_history(bridge, stage)
        restorer = bridge.archive.restorer
        real = restorer.restore

        def failing(node_ref, destination, rope=None, **kwargs):
            raise IntegrityError("payload hash mismatch (test)")

        leases = []
        archive_restore = bridge.archive.restore

        def capture(**kwargs):
            lease = archive_restore(**kwargs)
            leases.append((lease, list(getattr(lease, "readiness", None) or ())))
            return lease

        with mock.patch.object(restorer, "restore", failing), mock.patch.object(bridge.archive, "restore", capture):
            with self.assertRaises(IntegrityError):
                self.fork(bridge, stage, "o", overlap=True)
        restorer.restore = real
        failed = bridge.request_summary(bridge.requests[-1]["request_key"])
        self.assertIn("IntegrityError", failed["error"])
        self.assertIn("IntegrityError", failed["restore"]["overlap"]["error"])
        self.assertNothingPending(stage.caches)
        # nothing outlives the lease but plain records: no readiness, no archive closure (a lease <-> closure
        # cycle), and the stored producer failure's frames hold no locals (they can pin device memory)
        lease, calls = leases[-1]
        self.assertTrue(lease.closed and calls)
        self.assertEqual((lease.readiness, "complete" in vars(lease)), ([], False))
        self.assertIs(lease.complete(), lease)
        self.assertEqual(lease.stats["overlap"]["errors"][0]["type"], "IntegrityError")
        self.assertEqual(failed["restore"]["overlap"]["timelines"], lease.stats["overlap"]["timelines"])
        tb = calls[0][1].error.__traceback__
        self.assertIsNotNone(tb)
        while tb is not None:
            self.assertEqual(dict(tb.tb_frame.f_locals), {})
            tb = tb.tb_next
        self.assertFalse(bridge.active_request)
        summary, _, _, _ = self.fork(bridge, stage, "again", overlap=True)  # the worker continues
        self.assertEqual(summary["status"], "paused")


class LongLiveOverlapTests(OverlapTestCase):
    """LongLive 1.0 (prompt switch = recache epoch) and CFG streams (LongLive 2.0 with guidance) under the
    first-forward overlap. LongLive 2.0's sink, pinned shots and window are the SGLang cache's: they are covered by
    the seam's unit tests (``test_causal_kv_overlap.py``) with the real cache class."""

    def timeline(self, name, prompts, *, at=3, **session):
        return timeline_request(handle(name, **session), prompts, at=at)

    def test_a_resume_at_a_prompt_switch_overlaps_through_its_recache_forward(self):
        """A single-prompt session forked at the switch block under a new prompt. The recache (not restored)
        reruns, and its forward is the first to read the restored history: it waits per layer. The run joins after
        it, before the epoch is committed. Bitwise the plain restore."""
        bridge, stage = self.bridge(), OverlapStage()
        drive(bridge, self.timeline("L", ("A", "A"), pause_after_chunks=3), stage)
        bridge.drain_all()
        plain, latents_p, caches_p, _ = drive(bridge, self.timeline(
            "p", ("A", "C"), action="fork", from_session="L", checkpoint=3, pause_after_chunks=4,
            overlap_restore=False), stage)
        expected = self.snapshot(latents_p, caches_p)
        self.assertEqual(plain["restore"]["head_kind"], "chunk")  # the switch's epoch is not in the history
        summary, latents, caches, _ = drive(bridge, self.timeline(
            "o", ("A", "C"), action="fork", from_session="L", checkpoint=3, pause_after_chunks=4,
            overlap_restore=True), stage)
        record = summary["restore"]["overlap"]
        self.assertEqual((record["mode"], record["calls"], record["error"]), ("first_forward", 1, None))
        self.assertEqual(record["layers_waited"], LAYERS)  # waited inside the recache forward
        self.assertEqual([item.get("recache") for item in summary["commits"]].count("switch-3"), 1)
        self.assertTrue(torch.equal(latents, expected[0]))
        self.assertSameCaches(caches, expected[1])
        self.assertEqual(summary["chunk_digests"], plain["chunk_digests"])
        self.assertNothingPending(caches)
        self.assertLifetimesRetired(bridge, summary)

    def test_a_history_with_a_recache_epoch_restores_overlapped_bitwise(self):
        """A session past its prompt switch: the restore serves the switch's epoch (the recomputed chunks beside
        the sink) and the chunks after it, overlapped, bitwise the plain restore."""
        bridge, stage = self.bridge(), OverlapStage()
        drive(bridge, self.timeline("L", ("A", "C"), pause_after_chunks=4), stage)
        bridge.drain_all()
        plain, latents_p, caches_p, _ = drive(bridge, self.timeline(
            "p", ("A", "C"), action="fork", from_session="L", checkpoint=4, pause_after_chunks=5,
            overlap_restore=False), stage)
        expected = self.snapshot(latents_p, caches_p)
        sources = [span["source"] for span in plain["restore"]["plans"]["cond"]["spans"]]
        self.assertIn("recache", sources)
        summary, latents, caches, _ = drive(bridge, self.timeline(
            "o", ("A", "C"), action="fork", from_session="L", checkpoint=4, pause_after_chunks=5,
            overlap_restore=True), stage)
        record = summary["restore"]["overlap"]
        self.assertEqual((record["calls"], record["error"], record["layers_waited"]), (1, None, LAYERS))
        self.assertEqual([span["source"] for span in summary["restore"]["plans"]["cond"]["spans"]], sources)
        self.assertTrue(torch.equal(latents, expected[0]))
        self.assertSameCaches(caches, expected[1])
        self.assertLifetimesRetired(bridge, summary)

    def test_cfg_streams_restore_in_chained_calls_cond_first(self):
        """Two CFG streams: one overlapped call per stream, uncond chained after cond (``after=``), so the cond
        forward never waits for uncond work. Every layer of both streams is waited once; bitwise the plain restore."""
        bridge, stage = self.bridge(), OverlapStage()
        streams = ("cond", "uncond")
        drive(bridge, request(handle("L", pause_after_chunks=3), streams=streams), stage)
        bridge.drain_all()
        _, latents_p, caches_p, _ = drive(bridge, request(handle("p", "fork", from_session="L", checkpoint=2,
                                                                 pause_after_chunks=3, overlap_restore=False),
                                                          streams=streams), stage)
        expected = self.snapshot(latents_p, caches_p)
        summary, latents, caches, _ = drive(bridge, request(handle("o", "fork", from_session="L", checkpoint=2,
                                                                   pause_after_chunks=3,
                                                                   overlap_restore=True),
                                                            streams=streams), stage)
        record = summary["restore"]["overlap"]
        self.assertEqual((record["calls"], record["error"]), (2, None))
        self.assertEqual(record["layers_waited"], 2 * LAYERS)
        self.assertEqual([item["stream"] for item in record["waits"]], ["cond"] * LAYERS + ["uncond"] * LAYERS)
        self.assertTrue(torch.equal(latents, expected[0]))
        self.assertSameCaches(caches, expected[1])
        self.assertNothingPending(caches)
        self.assertLifetimesRetired(bridge, summary)


class OverlapSwitchTests(OverlapTestCase):
    """F5: default on (``ReelCacheRuntimeConfig.overlap_restore``; True when the config has no such field), per-request
    override through the session handle, reported in ``capabilities``."""

    def test_the_overlap_is_the_default_and_the_handle_turns_it_off(self):
        bridge, stage = self.bridge(), OverlapStage()
        self.assertTrue(bridge.overlap_restore_default)
        self.session_with_history(bridge, stage)
        default, latents_d, _, _ = self.fork(bridge, stage, "d")  # no override: overlapped
        expected = latents_d.clone()
        self.assertTrue(default["restore"]["overlap_requested"])
        self.assertEqual(default["restore"]["overlap"]["mode"], "first_forward")
        off, latents_o, _, _ = self.fork(bridge, stage, "o", overlap=False)
        self.assertFalse(off["restore"]["overlap_requested"])
        self.assertIsNone(off["restore"]["overlap"])
        self.assertTrue(torch.equal(latents_o, expected))
        caps = bridge.control("capabilities")["overlap_restore"]
        self.assertEqual((caps["default"], caps["available"]), (True, True))
        self.assertLifetimesRetired(bridge, off)

    def test_a_config_default_off_still_lets_a_request_overlap(self):
        bridge, stage = self.bridge(), OverlapStage()
        bridge.overlap_restore_default = False  # as ReelCacheRuntimeConfig(overlap_restore=False) sets it
        self.session_with_history(bridge, stage)
        plain, _, _, _ = self.fork(bridge, stage, "p")
        self.assertIsNone(plain["restore"]["overlap"])
        forced, _, _, _ = self.fork(bridge, stage, "f", overlap=True)
        self.assertEqual(forced["restore"]["overlap"]["mode"], "first_forward")
        self.assertFalse(bridge.control("capabilities")["overlap_restore"]["default"])

    def test_the_config_field_sets_the_default_when_the_config_has_it(self):
        has_field = "overlap_restore" in {field.name for field in dataclasses.fields(ReelCacheRuntimeConfig)}
        bridge = self.bridge()
        self.assertTrue(bridge.overlap_restore_default)  # the field's default, or the fallback without the field
        self.assertEqual(bridge.control("capabilities")["overlap_restore"]["config_field"], has_field)
        if has_field:  # Step 7's config.py: validated and serialized like every runtime option
            self.assertIs(bridge.startup["config"]["overlap_restore"], True)
            self.assertFalse(self.bridge("off", overlap_restore=False).overlap_restore_default)
            self.assertIs(ReelCacheRuntimeConfig.from_dict({"store_root": "a", "overlap_restore": False}).to_dict()[
                "overlap_restore"], False)
            for bad in ("yes", 1, None):
                with self.assertRaisesRegex(ConfigurationError, "overlap_restore"):
                    self.bridge("bad", overlap_restore=bad)

    def test_a_handle_value_that_is_not_a_bool_is_rejected(self):
        bridge = self.bridge()
        for bad in ("off", "first_forward", 1, 0):
            with self.assertRaisesRegex(ValueError, "overlap_restore must be a bool"):
                bridge.overlap_requested({"overlap_restore": bad})
        self.assertEqual((bridge.overlap_requested({"overlap_restore": False}), bridge.overlap_requested({})),
                         (False, True))

    def test_a_stage_without_the_seam_restores_plain(self):
        """On by default, but the stage's caches never wait for ``restore_pending`` (an SGLang without the
        layer-overlap commits): the restore runs plain (recorded), bitwise the explicit plain restore."""
        bridge = self.bridge()
        seamless_drive(bridge, request(handle("L", pause_after_chunks=3)))
        bridge.drain_all()
        summary, latents, caches, _ = seamless_drive(bridge, request(handle("f", "fork", from_session="L",
                                                                            checkpoint=2, pause_after_chunks=3)))
        restore = summary["restore"]
        self.assertTrue(restore["overlap_requested"])
        self.assertIsNone(restore["overlap"])
        self.assertIn("no restore_pending seam", restore["overlap_unsupported"])
        self.assertTrue(all(getattr(cache, "restore_pending", None) is None
                            for items in caches.values() for cache in items))
        expected = latents.clone()
        off, latents_off, _, _ = seamless_drive(bridge, request(handle("g", "fork", from_session="L", checkpoint=2,
                                                                       pause_after_chunks=3, overlap_restore=False)))
        self.assertTrue(torch.equal(latents_off, expected))
        self.assertIsNone(off["restore"]["overlap_unsupported"])
        self.assertLifetimesRetired(bridge, off)

    def test_raw_reference_archives_never_overlap(self):
        bridge, stage = self.bridge("raw", representation="raw", raw_restore="prefer"), OverlapStage()
        drive(bridge, request(handle("L", pause_after_chunks=3)), stage)
        summary, _, _, _ = drive(bridge, request(handle("r", "fork", from_session="L", checkpoint=2,
                                                        pause_after_chunks=3, overlap_restore=True)), stage)
        self.assertFalse(summary["restore"]["overlap_requested"])
        self.assertFalse(bridge.control("capabilities")["overlap_restore"]["available"])


if __name__ == "__main__":
    unittest.main()
