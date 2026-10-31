"""CPU tests of the ReelCache conditioning-cache backend of the LongCat-Video port (Step 13 phase 3).

``reelcache_sglang_models.longcat_video.reelcache_backend`` archives each segment's conditioning cache as a ReelCache
``segment`` node (``longcat_segment`` cache epoch, latest method at the LongCat K98/V95 targets, all-intra node packs)
and/or a RAW reference branch, and restores an archived epoch on resume/rewind. These tests drive the backend with
small synthetic caches through a CPU ArchivePipeline and the software restorer (no SGLang, no GPU): epoch structure,
RAW identity, lossy restore error vs the energy targets, explicit errors (no silent rebuild, no dropped capture,
foreign caches), deduplication, the rewind fork, restart re-indexing, and the segment-resume generator helpers.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from reelcache_sglang_models.longcat_video import conditioning_cache as cc
from reelcache_sglang_models.longcat_video import reelcache_backend as mb
from reelcache_sglang_models.longcat_video import segment_resume
from reelcache_sglang_models.longcat_video.spec import LongCatVideoArch

LAYERS, HEADS, HEAD_DIM = 2, 2, 64
CHANNELS = HEADS * HEAD_DIM
GRID = (2, 6, 8)
TOKENS = GRID[0] * GRID[1] * GRID[2]
FINGERPRINT = "test-longcat-fingerprint"
ARCH = LongCatVideoArch(hidden_size=CHANNELS, depth=LAYERS, num_heads=HEADS)


def synthetic_cache(seed: int, segment: int, *, session: str = "session", cond_sha: str | None = None,
                    fingerprint: str = FINGERPRINT, source: str = cc.SOURCE_NATIVE_BUILD) -> cc.LongCatConditioningCache:
    """A compressible cache: rank-32 content plus a small full-rank floor (K row-normalized like k_norm)."""
    generator = torch.Generator().manual_seed(seed)
    keys, values = [], []
    for _layer in range(LAYERS):
        out = []
        for target in ("K", "V"):
            basis = torch.linalg.qr(torch.randn(CHANNELS, 32, generator=generator)).Q.T
            value = (torch.randn(TOKENS, 32, generator=generator) * torch.linspace(2, 0.02, 32)) @ basis
            value = value + 0.01 * torch.randn(TOKENS, CHANNELS, generator=generator)
            if target == "K":
                value = torch.nn.functional.layer_norm(value, (CHANNELS,))
            out.append(value.to(torch.bfloat16).reshape(1, TOKENS, HEADS, HEAD_DIM).contiguous())
        keys.append(out[0])
        values.append(out[1])
    cond = torch.randn(1, 16, GRID[0], 3, 4, generator=generator).to(torch.bfloat16)
    epoch = cc.ConditioningCacheEpoch(
        segment_index=segment, token_grid=GRID, cond_latents_sha256=cond_sha or cc.tensor_sha256(cond),
        model_fingerprint=fingerprint, session_id=session, parent_segment_index=segment - 1 if segment else None)
    state = segment_resume.generator_state_hex(torch.Generator().manual_seed(1000 + segment))
    return cc.LongCatConditioningCache(epoch=epoch, keys_pre_rope=keys, values=values, source=source,
                                       metadata={"build_seconds": 0.0, "segment_generator_state": state},
                                       cond_latents=cond)


class ReelCacheBackendTests(unittest.TestCase):
    def setUp(self):
        from reelcache.archive_pipeline import ArchivePipeline
        from reelcache.archive_restore import SoftwareLineageRestorer
        from reelcache.lineage_commit import ArchiveOptions
        from reelcache.manager import KVRuntime
        from reelcache.predictor_registry import RankPredictorRegistry

        self.root = Path(tempfile.mkdtemp(prefix="longcat-backend-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.model = mb.model_identity(FINGERPRINT, ARCH)
        self.runtime = KVRuntime(self.root / "store", self.model)
        self.pipeline = ArchivePipeline(self.runtime, method=mb.default_method(),
                                        registry=RankPredictorRegistry("longcat-backend-test"),
                                        options=ArchiveOptions(encode_threads=1), device="cpu", encode_workers=1)
        self.backend = mb.ReelCacheConditioningCacheBackend(
            self.runtime, pipeline=self.pipeline, restorer=SoftwareLineageRestorer(self.runtime),
            archive=("svd_hevc", "raw"), heads=HEADS, pin_memory=False)
        self.addCleanup(self._close)

    def _close(self):
        try:
            if not self.pipeline._closed:
                self.pipeline.close(drain=False)
        finally:
            if not self.runtime._closed:
                self.runtime.close()

    @staticmethod
    def equal(a: cc.LongCatConditioningCache, b: cc.LongCatConditioningCache) -> bool:
        return all(torch.equal(x, y) for x, y in zip(a.keys_pre_rope + a.values, b.keys_pre_rope + b.values))

    def test_segments_publish_longcat_epochs_and_restore(self):
        caches = [synthetic_cache(1, 0), synthetic_cache(2, 1)]
        for cache in caches:
            self.backend.on_cache_built(cache)
        self.assertEqual(self.backend.pending(), 2)
        result = self.backend.flush()
        self.assertEqual(len(result["published"]), 2)
        self.assertEqual(self.backend.pending(), 0)
        refs = [self.backend.node_ref(cache.epoch) for cache in caches]
        path = self.runtime.node_path(refs[1])
        self.assertEqual([node.node_kind for _, node, _ in path], ["root", "segment", "segment"])
        epochs = [epoch for _, _, epoch in path]
        self.assertEqual([epoch.cause for epoch in epochs], ["initial", "longcat_segment", "longcat_segment"])
        self.assertEqual(path[2][2].parent_epoch_ref, path[1][1].epoch_ref)
        for _, node, epoch in path[1:]:
            self.assertEqual([(r.token_start, r.token_end) for r in epoch.ranges], [(0, TOKENS)])
            self.assertEqual(epoch.position_space, "segment_local")
            self.assertIsNotNone(node.execution_state_ref)
        self.assertEqual(path[2][2].details["port_epoch"]["segment_index"], 1)
        self.assertEqual([p.entry.source for p in self.runtime.resolve_history(refs[1])], ["epoch"])
        stats = mb.node_bytes(self.runtime, refs[0])
        self.assertEqual(stats["records"], 2 * LAYERS)
        self.assertGreater(stats["payload_bytes"], 0)
        self.assertGreaterEqual(min(stats["min_stored_energy"]["K"], 1.0), 0.98)
        self.assertGreaterEqual(min(stats["min_stored_energy"]["V"], 1.0), 0.95)
        for cache in caches:
            self.backend.arm_restore("raw")
            raw = self.backend.restore(cache.epoch)
            self.assertTrue(self.equal(raw, cache))
            cc.accept_restored_cache(raw, cache.epoch, layers=LAYERS, heads=HEADS, head_dim=HEAD_DIM,
                                     dtype=torch.bfloat16, device="cpu")
            self.backend.arm_restore("svd_hevc")
            lossy = self.backend.restore(cache.epoch)
            cc.accept_restored_cache(lossy, cache.epoch, layers=LAYERS, heads=HEADS, head_dim=HEAD_DIM,
                                     dtype=torch.bfloat16, device="cpu")
            self.assertEqual(lossy.metadata["node_ref"], self.backend.node_ref(cache.epoch))
            for name, restored, source, target in (
                    ("K", lossy.keys_pre_rope, cache.keys_pre_rope, 0.98), ("V", lossy.values, cache.values, 0.95)):
                for a, b in zip(restored, source):
                    error = float((a.double() - b.double()).norm() / b.double().norm())
                    # stored-factor energy >= target bounds the pre-codec error; QP8 adds a little
                    self.assertLess(error, (1 - target) ** 0.5 + 0.03, name)
        self.backend.arm_restore(None)
        self.assertIsNone(self.backend.restore(caches[0].epoch))
        warm = self.backend.warmup()
        self.assertEqual((warm["representation"], warm["segment_index"]), ("svd_hevc", 1))
        self.assertEqual(self.backend.events[-1]["event"], "warmup_restore")
        self.assertEqual(self.backend.warmup(caches[0].epoch, "raw")["segment_index"], 0)

    def test_process_prewarm_and_session_staging_are_separate(self):
        with self.assertRaisesRegex(cc.ConditioningCacheError, "prepare_epochs needs archived"):
            self.backend.prepare_epochs()
        process = self.backend.prewarm_process(((GRID[0], GRID[1], GRID[2]),), max_rank=CHANNELS)
        self.assertEqual(process["level"], "process")
        self.assertEqual(process["kernels"]["resolved_backend"], "reference")  # CPU: nothing to compile
        self.assertIn("no hardware decoder sessions", process["decoders"])
        self.assertIs(self.backend.process_warm, process)
        with self.assertRaises(cc.ConditioningCacheError):
            self.backend.prewarm_process(max_rank=CHANNELS + 1)
        caches = [synthetic_cache(11, 0), synthetic_cache(12, 1)]
        for cache in caches:
            self.backend.on_cache_built(cache)
        self.backend.flush()
        self.backend.arm_restore("svd_hevc")
        before = self.backend.restore(caches[1].epoch)
        with torch.inference_mode():  # SGLang opens sessions inside its executor
            session = self.backend.prepare_epochs()
        self.assertEqual((session["level"], session["epochs"]), ("session (CPU packing -> prepared state)", [0, 1]))
        self.assertIn("keeps no prepared packs", session["staging"])  # the CPU software restorer
        self.assertEqual(self.backend.events[-1]["event"], "prepared_epochs")
        both = self.backend.prewarm([caches[1].epoch], token_grids=((GRID[0], GRID[1], GRID[2]),), max_rank=CHANNELS)
        self.assertEqual((both["session"]["epochs"], both["process"]["level"]), ([1], "process"))
        self.assertTrue(self.equal(before, self.backend.restore(caches[1].epoch)))  # warm-ups change no value
        # the GPU stack's decoder pool holds a session for every canonical canvas of the real LongCat geometry
        from reelcache.packed_decode import canonical_canvas_shapes
        shapes = canonical_canvas_shapes(4, 44, 82, mb.default_method().codec.pix_fmt, channels=4096)
        self.assertEqual(len(shapes), 18)
        self.assertGreaterEqual(mb.LONGCAT_DECODER_SESSIONS, len(shapes))

    def test_overlapped_restore_completes_layer_by_layer_to_the_same_cache(self):
        caches = [synthetic_cache(13, 0), synthetic_cache(14, 1)]
        for cache in caches:
            self.backend.on_cache_built(cache)
        self.backend.flush()
        self.backend.arm_restore("svd_hevc")
        reference = self.backend.restore(caches[1].epoch)
        self.backend.set_overlap(True)
        with torch.inference_mode():
            overlapped = self.backend.restore(caches[1].epoch)
        self.assertIsNotNone(overlapped.ready)
        for layer in range(LAYERS):
            overlapped.ready.wait(layer)
        info = overlapped.ready.join()
        self.assertTrue(info["overlapped"])
        self.assertFalse(info["progressive_marks"])  # the software restorer marks every layer at the end
        self.assertEqual(info["tier"], "compressed_cold")  # d5: labelled at completion (no prepared cache here)
        self.assertEqual(info["consumer_waits"]["count"], LAYERS)
        self.assertTrue(self.equal(overlapped, reference))
        self.assertFalse(overlapped.keys_pre_rope[0].is_inference())
        cc.accept_restored_cache(overlapped, caches[1].epoch, layers=LAYERS, heads=HEADS, head_dim=HEAD_DIM,
                                 dtype=torch.bfloat16, device="cpu")
        with self.assertRaises(cc.ConditioningCacheError):
            self.backend.set_overlap("yes")
        self.backend.set_overlap(False)
        self.assertIsNone(self.backend.restore(caches[1].epoch).ready)

    def test_overlapped_restore_labels_its_tier_when_it_completes(self):
        """d5: the tier of an overlapped restore is labelled on its thread when it completes -- d4 labelled it at
        issue, before a cold restore's store read -- and the per-call choice overrides the backend's default."""
        import threading

        caches = [synthetic_cache(15, 0), synthetic_cache(16, 1)]
        for cache in caches:
            self.backend.on_cache_built(cache)
        self.backend.flush()
        self.backend.arm_restore("svd_hevc")
        reference = self.backend.restore(caches[1].epoch)

        class CountingRestorer:
            """The software restorer with a prepared-pack miss counter; a cold restore counts its miss late."""

            def __init__(self, inner):
                self.inner, self.misses, self.cold, self.last_staging = inner, 0, False, None
                self.release = threading.Event()

            def stats(self):
                return {"prepared": {"misses": self.misses}}

            def restore(self, *args, **kwargs):
                if self.cold:
                    self.release.wait(30)  # the caller has the cache before the pack is read
                    self.misses += 1
                    self.cold = False
                return self.inner.restore(*args, **kwargs)

        counting = CountingRestorer(self.backend.restorer)
        self.backend.restorer = counting
        before = {tier: row["count"] for tier, row in self.backend._restore_tiers.items()}
        try:
            counting.cold = True
            cold = self.backend.restore(caches[1].epoch, overlapped=True)
            self.assertIsNotNone(cold.ready)
            pending = self.backend.last_restore_tier
            self.assertEqual((pending["tier"], pending["pending"], pending["overlapped"]), (None, True, True))
            counting.release.set()
            for layer in range(LAYERS):
                cold.ready.wait(layer)
            info = cold.ready.join()
            self.assertEqual(info["tier"], "compressed_cold")  # d4: compressed_prepared (labelled before the read)
            self.assertEqual(info["bytes"], cold.nbytes)
            self.assertGreater(info["seconds"], 0.0)
            self.assertEqual(info["consumer_waits"]["count"], LAYERS)
            self.assertEqual((self.backend.last_restore_tier["tier"], self.backend.last_restore_tier["overlapped"]),
                             ("compressed_cold", True))
            self.assertTrue(self.equal(cold, reference))
            prepared = self.backend.restore(caches[1].epoch, overlapped=True)
            self.assertEqual(prepared.ready.join()["tier"], "compressed_prepared")
            self.assertTrue(self.equal(prepared, reference))
            self.backend.set_overlap(True)  # per call: overlapped=False is plain whatever the default
            self.assertIsNone(self.backend.restore(caches[1].epoch, overlapped=False).ready)
            self.assertEqual(self.backend.last_restore_tier["tier"], "compressed_prepared")
            self.backend.set_overlap(False)
            per_call = self.backend.restore(caches[1].epoch, overlapped=True)
            self.assertIsNotNone(per_call.ready)
            per_call.ready.join()  # values are read after the join
            self.assertTrue(self.equal(per_call, reference))
            self.assertIsNone(self.backend.restore(caches[1].epoch).ready)  # explicitly off
            self.backend.set_overlap(None)
            self.assertIsNone(self.backend.overlap)
            self.assertIsNone(self.backend.restore(caches[1].epoch).ready)  # unset: plain for direct calls
            with self.assertRaises(cc.ConditioningCacheError):
                self.backend.restore(caches[1].epoch, overlapped="yes")
            with self.assertRaises(cc.ConditioningCacheError):
                self.backend.set_overlap(1)
            counts = {tier: row["count"] - before.get(tier, 0) for tier, row in self.backend._restore_tiers.items()}
            self.assertEqual((counts["compressed_cold"], counts["compressed_prepared"]), (1, 5))
        finally:
            self.backend.restorer = counting.inner
            self.backend.arm_restore(None)

    def test_diagnostic_method_reaches_explicit_energy_targets(self):
        from reelcache.archive_pipeline import ArchivePipeline
        from reelcache.lineage_commit import ArchiveOptions
        from reelcache.manager import KVRuntime
        from reelcache.predictor_registry import RankPredictorRegistry

        method = mb.diagnostic_method(0.999, 0.99)
        self.assertEqual((method.targets.key, method.targets.value), (0.999, 0.99))
        self.assertTrue(method.targets.provisional)
        self.assertNotEqual(method.fingerprint, mb.default_method().fingerprint)
        runtime = KVRuntime(self.root / "diag_store", self.model)
        pipeline = ArchivePipeline(runtime, method=method, registry=RankPredictorRegistry("longcat-backend-diag"),
                                   options=ArchiveOptions(encode_threads=1), device="cpu", encode_workers=1)
        try:
            backend = mb.ReelCacheConditioningCacheBackend(runtime, pipeline=pipeline, archive=("svd_hevc",), heads=HEADS,
                                                        pin_memory=False)
            cache = synthetic_cache(21, 0, session="diag")
            backend.on_cache_built(cache)
            backend.flush()
            stats = mb.node_bytes(runtime, backend.node_ref(cache.epoch))
            self.assertGreaterEqual(min(stats["min_stored_energy"]["K"], 1.0), 0.999)
            self.assertGreaterEqual(min(stats["min_stored_energy"]["V"], 1.0), 0.99)
        finally:
            pipeline.close()
            runtime.close()

    def test_restores_inside_inference_mode_return_normal_tensors(self):
        # SGLang runs stages under torch.inference_mode(); the GPU restorer's decoder threads (mode off) write
        # canvases and the kernels write the fresh destinations in place, so restore buffers must not be inference
        # tensors (an in-place write to one outside InferenceMode raises).
        import threading

        cache = synthetic_cache(13, 0, session="inference")
        self.backend.on_cache_built(cache)
        self.backend.flush()
        errors = []
        for representation in ("svd_hevc", "raw"):
            self.backend.arm_restore(representation)
            with torch.inference_mode():
                restored = self.backend.restore(cache.epoch)
            tensors = restored.keys_pre_rope + restored.values
            self.assertFalse(any(t.is_inference() for t in tensors), representation)

            def write(tensor=tensors[0]):
                try:
                    tensor.add_(0)
                except RuntimeError as exc:  # pragma: no cover - the regression this test guards
                    errors.append(exc)
            thread = threading.Thread(target=write)
            thread.start()
            thread.join()
        self.assertEqual(errors, [])

    def test_explicit_errors_never_rebuild_or_drop_silently(self):
        cache = synthetic_cache(3, 0)
        with self.assertRaisesRegex(cc.ConditioningCacheError, "warmup needs an archived"):
            self.backend.warmup()
        self.backend.arm_restore("svd_hevc")
        with self.assertRaisesRegex(cc.ConditioningCacheError, "refusing to rebuild silently"):
            self.backend.restore(cache.epoch)
        self.backend.on_cache_built(cache)
        with self.assertRaisesRegex(cc.ConditioningCacheError, "call flush"):
            self.backend.restore(cache.epoch)
        with self.assertRaisesRegex(cc.ConditioningCacheError, "never submitted"):
            self.backend.drain()
        self.backend.on_cache_built(synthetic_cache(4, 1))
        with self.assertRaisesRegex(cc.ConditioningCacheError, "capture slots"):
            self.backend.on_cache_built(synthetic_cache(5, 2))
        with self.assertRaisesRegex(cc.ConditioningCacheError, "never archived"):
            self.backend.close()
        self.backend.flush()
        self.backend.close()
        with self.assertRaisesRegex(cc.ConditioningCacheError, "fingerprint"):
            self.backend.on_cache_built(synthetic_cache(6, 2, fingerprint="another-model"))
        with self.assertRaisesRegex(cc.ConditioningCacheError, "natively built"):
            self.backend.on_cache_built(synthetic_cache(6, 2, source=cc.SOURCE_RESTORED))
        with self.assertRaisesRegex(cc.ConditioningCacheError, "ends at segment"):
            self.backend.on_cache_built(synthetic_cache(7, 3))
            self.backend.flush()

    def test_dedup_rewind_fork_and_restart_reindex(self):
        from reelcache.archive_restore import SoftwareLineageRestorer
        from reelcache.manager import KVRuntime

        seg0, seg1 = synthetic_cache(8, 0), synthetic_cache(9, 1)
        for cache in (seg0, seg1):
            self.backend.on_cache_built(cache)
        self.backend.flush()
        head = self.runtime.lineage_head(mb.lineage_branch("session"))
        # the same conditioning latents again (e.g. a resume that rebuilt): nothing new is archived
        self.backend.on_cache_built(synthetic_cache(9, 1))
        self.assertEqual(self.backend.pending(), 0)
        self.assertEqual(self.backend.events[-1]["event"], "capture_deduplicated")
        # rewind: segment 1 regenerated from other conditioning latents forks the lineage after segment 0
        self.backend.set_archive(("svd_hevc",))
        regenerated = synthetic_cache(10, 1)
        self.backend.on_cache_built(regenerated)
        self.backend.flush()
        fork = self.runtime.node_path(self.backend.node_ref(regenerated.epoch))
        original = self.runtime.node_path(head)
        self.assertEqual([ref for ref, _, _ in fork][:2], [ref for ref, _, _ in original][:2])
        self.assertNotEqual(fork[2][0], original[2][0])
        self.assertEqual(self.runtime.lineage_head(mb.lineage_branch("session")), head)
        self.assertIn(mb.lineage_branch("session") + "~r1", self.runtime.list_lineages())
        self.backend.arm_restore("svd_hevc")
        before = self.backend.restore(seg1.epoch)
        self.pipeline.close()
        self.runtime.close()
        runtime = KVRuntime(self.root / "store", self.model)
        self.addCleanup(runtime.close)
        backend = mb.ReelCacheConditioningCacheBackend(runtime, restorer=SoftwareLineageRestorer(runtime), archive=(),
                                                    heads=HEADS, pin_memory=False)
        indexed = backend.reload_index()
        backend.arm_restore("svd_hevc")
        self.assertTrue(self.equal(backend.restore(seg1.epoch), before))
        self.assertEqual(backend.last_reload["svd_hevc"], 3)
        self.assertGreaterEqual(backend.last_reload["raw"], 1)  # RAW branches of the reindexed epochs (restart path)
        self.assertEqual(indexed, 3 + backend.last_reload["raw"])  # (epoch, representation) entries
        backend.arm_restore("raw")
        self.assertTrue(self.equal(backend.restore(seg1.epoch), seg1))
        with self.assertRaisesRegex(cc.ConditioningCacheError, "needs an ArchivePipeline"):
            backend.set_archive(("svd_hevc",))

    def test_deferred_archival_keeps_every_segment_on_its_capture_time_parent(self):
        """A rewind captured before the previous request's epochs were archived (deferred schedule): each node still
        gets the parent its segment had when it was captured; a seeded timeline parents a resumed segment."""
        from reelcache.archive_restore import SoftwareLineageRestorer

        backend = mb.ReelCacheConditioningCacheBackend(
            self.runtime, pipeline=self.pipeline, restorer=SoftwareLineageRestorer(self.runtime), archive=("svd_hevc",),
            heads=HEADS, pin_memory=False, capture_slots=4)
        first = [synthetic_cache(20 + i, i, session="t") for i in range(3)]
        for cache in first:
            backend.on_cache_built(cache)
        rewound = synthetic_cache(30, 1, session="t")  # segment 1 again, other conditioning, before any flush
        backend.on_cache_built(rewound)
        backend.flush()
        refs = [backend.node_ref(cache.epoch) for cache in first]
        path = [ref for ref, _, _ in self.runtime.node_path(refs[2])]
        self.assertEqual(path[1:], refs)
        fork = [ref for ref, _, _ in self.runtime.node_path(backend.node_ref(rewound.epoch))]
        self.assertEqual(fork[1], refs[0])
        self.assertEqual(self.runtime.lineage_head(mb.lineage_branch("t")), refs[2])
        # a new process resumes the rewound timeline at segment 2: seed_timeline gives its node the right parent
        backend.forget_session("t")
        backend.seed_timeline("t", {0: first[0].epoch, 1: rewound.epoch})
        resumed = synthetic_cache(31, 2, session="t")
        backend.on_cache_built(resumed)
        backend.flush()
        resumed_path = [ref for ref, _, _ in self.runtime.node_path(backend.node_ref(resumed.epoch))]
        self.assertEqual(resumed_path[1:3], fork[1:3])
        self.assertEqual(backend.archive_state(resumed.epoch, "svd_hevc"), "published")
        self.assertIsNone(backend.archive_state(synthetic_cache(32, 3, session="t").epoch, "svd_hevc"))
        self.assertGreaterEqual(len(backend.session_branches("t")), 2)

    def test_bounded_planning_archives_the_unbounded_node_and_defers_on_request(self):
        """ArchiveOptions(plan_memory_bytes=B) (the GPU-planning budget of a serving worker) stores the node of
        unbounded planning (CPU: batch-independent solves); eager archival is allowed for a bounded GPU pipeline and
        refused for an unbounded one; a generation section defers the fitter until it closes."""
        import time as _time
        from types import SimpleNamespace

        from reelcache.archive_pipeline import ArchivePipeline
        from reelcache.archive_restore import SoftwareLineageRestorer
        from reelcache.lineage_commit import ArchiveOptions
        from reelcache.manager import KVRuntime
        from reelcache.predictor_registry import RankPredictorRegistry

        caches = [synthetic_cache(40, 0, session="b"), synthetic_cache(41, 1, session="b")]
        refs, rows_by_label = {}, {}
        for label, budget in (("unbounded", None), ("bounded", 1 << 20)):
            runtime = KVRuntime(self.root / f"store-{label}", self.model)
            pipeline = ArchivePipeline(runtime, method=mb.default_method(), registry=RankPredictorRegistry("budget"),
                                       options=ArchiveOptions(encode_threads=1, plan_memory_bytes=budget),
                                       device="cpu", encode_workers=1)
            backend = mb.ReelCacheConditioningCacheBackend(runtime, pipeline=pipeline,
                                                        restorer=SoftwareLineageRestorer(runtime),
                                                        archive=("svd_hevc",), heads=HEADS, pin_memory=False)
            for cache in caches:
                backend.on_cache_built(cache)
            backend.flush()
            refs[label] = [backend.node_ref(cache.epoch) for cache in caches]
            rows_by_label[label] = [mb.node_matrix_rows(runtime, ref) for ref in refs[label]]
            if budget is not None:
                plans = runtime.node_commit_report(refs[label][0])["plans"][0]
                self.assertGreater(plans["plan_memory"]["groups"], 1)  # really planned in groups
            pipeline.close()
            runtime.close()
        self.assertEqual(refs["bounded"], refs["unbounded"])  # identical node ids = identical stored content
        for unbounded_ref, bounded_ref in zip(rows_by_label["unbounded"], rows_by_label["bounded"]):
            comparison = mb.compare_node_rows(unbounded_ref, bounded_ref)
            self.assertTrue(comparison["within_planner_tolerance"], comparison)
            self.assertEqual(comparison["max_abs_stored_energy_delta"], 0.0)
            self.assertEqual(comparison["matrices"], 2 * LAYERS)
            for key, row in bounded_ref.items():
                self.assertEqual(row["target"], 0.98 if key.endswith(":K") else 0.95)
                self.assertGreaterEqual(row["stored_energy"], row["target"])
        changed = {key: dict(row) for key, row in rows_by_label["bounded"][0].items()}
        first = sorted(changed)[0]
        changed[first]["rank"] += 1
        changed[first]["stored_energy"] = changed[first]["target"] - 1e-3
        comparison = mb.compare_node_rows(rows_by_label["unbounded"][0], changed)
        self.assertEqual((comparison["rank_differences"], comparison["below_target"]), ([first], [first]))
        self.assertFalse(comparison["within_planner_tolerance"])
        with self.assertRaisesRegex(cc.ConditioningCacheError, "different matrices"):
            mb.compare_node_rows(rows_by_label["unbounded"][0], {first: changed[first]})
        gpu_like = SimpleNamespace(runtime=self.runtime, device="cuda:0", options=ArchiveOptions())
        with self.assertRaisesRegex(cc.ConditioningCacheError, "bounded planning"):
            mb.ReelCacheConditioningCacheBackend(self.runtime, pipeline=gpu_like, archive=(), heads=HEADS,
                                              schedule="eager")
        bounded_gpu = SimpleNamespace(runtime=self.runtime, device="cuda:0",
                                      options=ArchiveOptions(plan_memory_bytes=8 << 30))
        eager = mb.ReelCacheConditioningCacheBackend(self.runtime, pipeline=bounded_gpu, archive=(), heads=HEADS,
                                                  schedule="eager")
        self.assertEqual(eager.schedule, "eager")
        with self.assertRaisesRegex(cc.ConditioningCacheError, "CUDA archive pipeline"):
            self.backend.set_defer_planning(True)
        with self.backend.generation_section():  # off: a no-op
            pass
        # the deferral itself (CPU pipeline, section semantics of ArchivePipeline.defer_gpu_work)
        deferring = mb.ReelCacheConditioningCacheBackend(
            self.runtime, pipeline=self.pipeline, restorer=SoftwareLineageRestorer(self.runtime),
            archive=("svd_hevc",), heads=HEADS, pin_memory=False, schedule="eager")
        deferring.defer_planning = True  # the CUDA check is bypassed: the section semantics are device-independent
        cache = synthetic_cache(42, 0, session="deferred")
        with deferring.generation_section():
            deferring.on_cache_built(cache)  # eager: submitted at capture
            ticket = deferring._pending[0].ticket
            _time.sleep(1.0)
            self.assertFalse(ticket.done())  # the fitter does not start inside the section
            self.assertTrue(self.pipeline.stats()["deferred"])
        self.assertIsNotNone(ticket.result(600))  # planned once the section closed
        deferring.flush()

    def test_gpu_planner_receives_host_copies_one_node_at_a_time(self):
        """``deferred_sources="host"`` (the serving setting): a CUDA planner receives the pinned host copies at flush
        and uploads each node itself when it plans it (no re-upload of every captured epoch at flush); eager
        archival hands the same host copies over at capture. A recording stand-in for the CUDA pipeline (no GPU)."""
        from types import SimpleNamespace
        from unittest import mock

        from reelcache.lineage_commit import ArchiveOptions

        class RecordingPipeline:
            device = "cuda:0"
            options = ArchiveOptions(plan_memory_bytes=8 << 30)
            method = mb.default_method()

            def __init__(self, runtime):
                self.runtime, self.submitted = runtime, []

            def submit(self, branch, node, matrices, contexts, **kwargs):
                self.submitted.append({"node": node, "devices": {value.device.type for value in matrices.values()},
                                       "pointers": {key: value.data_ptr() for key, value in matrices.items()}})
                kwargs["release"](None)
                return SimpleNamespace(result=lambda timeout=None, node=node: f"ref-{node}", done=lambda: True)

        with self.assertRaisesRegex(cc.ConditioningCacheError, "deferred_sources"):
            mb.ReelCacheConditioningCacheBackend(self.runtime, pipeline=RecordingPipeline(self.runtime), archive=(),
                                              heads=HEADS, deferred_sources="gpu")
        for schedule in ("deferred", "eager"):
            pipeline = RecordingPipeline(self.runtime)
            backend = mb.ReelCacheConditioningCacheBackend(self.runtime, pipeline=pipeline, archive=("svd_hevc",),
                                                        heads=HEADS, pin_memory=False, schedule=schedule,
                                                        deferred_sources="host")
            caches = [synthetic_cache(60, 0, session=f"host-{schedule}"), synthetic_cache(61, 1, session=f"host-{schedule}")]
            hosts = []
            for cache in caches:
                backend.on_cache_built(cache)
                hosts.append({key: value.data_ptr() for key, value in backend._pending[-1].host.items()})
            self.assertEqual(len(pipeline.submitted), 2 if schedule == "eager" else 0)
            with mock.patch.object(mb, "node_bytes", return_value={}):
                backend.flush()
            self.assertEqual([item["devices"] for item in pipeline.submitted], [{"cpu"}, {"cpu"}])
            self.assertEqual([item["pointers"] for item in pipeline.submitted], hosts)  # the slots themselves
            self.assertEqual([backend.node_ref(cache.epoch) for cache in caches],
                             [f"ref-{mb.node_id(cache.epoch)}" for cache in caches])
            self.assertEqual(backend.pending(), 0)

    def test_eager_schedule_archives_behind_generation_on_the_cpu(self):
        from reelcache.archive_restore import SoftwareLineageRestorer

        backend = mb.ReelCacheConditioningCacheBackend(
            self.runtime, pipeline=self.pipeline, restorer=SoftwareLineageRestorer(self.runtime), archive=("svd_hevc",),
            schedule="eager", heads=HEADS, pin_memory=False)
        cache = synthetic_cache(12, 0, session="eager")
        backend.on_cache_built(cache)
        self.assertTrue(backend.events[-1]["submitted_now"])
        ticket = backend._pending[0].ticket
        self.assertIsNotNone(ticket)  # submitted at capture, before any flush
        ref = ticket.result(600)
        backend.flush()
        self.assertEqual(backend.node_ref(cache.epoch), ref)
        backend.arm_restore("svd_hevc")
        restored = backend.restore(cache.epoch)
        error = float((restored.values[0].double() - cache.values[0].double()).norm() / cache.values[0].double().norm())
        self.assertLess(error, 0.05 ** 0.5 + 0.03)
        with self.assertRaisesRegex(cc.ConditioningCacheError, "CPU archive pipeline"):
            mb.ReelCacheConditioningCacheBackend(self.runtime, archive=(), schedule="eager", heads=HEADS)

    def test_configuration_is_validated(self):
        with self.assertRaises(cc.ConditioningCacheError):
            mb.ReelCacheConditioningCacheBackend(object())
        with self.assertRaises(cc.ConditioningCacheError):
            mb.ReelCacheConditioningCacheBackend(self.runtime, pipeline=self.pipeline, schedule="overlap", heads=HEADS)
        with self.assertRaises(cc.ConditioningCacheError):
            mb.ReelCacheConditioningCacheBackend(self.runtime, pipeline=self.pipeline, archive=("svd_hevc", "svd_hevc"),
                                              heads=HEADS)
        with self.assertRaises(cc.ConditioningCacheError):
            mb.ReelCacheConditioningCacheBackend(self.runtime, archive=("svd_hevc",), heads=HEADS)
        with self.assertRaises(cc.ConditioningCacheError):
            mb.ReelCacheConditioningCacheBackend(self.runtime, pipeline=self.pipeline)  # 128 channels: heads required
        with self.assertRaises(cc.ConditioningCacheError):
            self.backend.arm_restore("svd")
        identity = mb.model_identity("fp")
        self.assertEqual((identity.layers, identity.channels_for("K"), identity.dtype_for("V")), (48, 4096, "bfloat16"))
        method = mb.default_method()
        self.assertEqual((method.targets.key, method.targets.value, method.targets.profile), (0.98, 0.95, "longcat_720p"))
        self.assertFalse(method.codec.inter_prediction)
        self.assertNotIn("/", mb.lineage_branch("a/b c"))
        self.assertTrue(isinstance(cc.LongCatConditioningCache, type))


class SegmentResumeTests(unittest.TestCase):
    def test_generator_state_round_trip_reproduces_draws(self):
        generator = torch.Generator().manual_seed(5)
        torch.randn(7, generator=generator)
        state = segment_resume.generator_state_hex(generator)
        expected = torch.randn(11, generator=generator)
        other = torch.Generator().manual_seed(99)
        segment_resume.set_generator_state_hex(other, state)
        self.assertTrue(torch.equal(torch.randn(11, generator=other), expected))
        self.assertEqual(len(segment_resume.generator_state_sha256(state)), 64)
        with self.assertRaises(ValueError):
            segment_resume.set_generator_state_hex(other, "zz")
        with self.assertRaises(ValueError):
            segment_resume.set_generator_state_hex(other, "00" * 16)

    def test_cache_carries_validated_cond_latents(self):
        cache = synthetic_cache(11, 0)
        self.assertEqual(tuple(cache.cond_latents.shape), (1, 16, GRID[0], 3, 4))
        self.assertNotIn("cond_latents", cache.summary())
        with self.assertRaises(cc.ConditioningCacheError):
            cc.LongCatConditioningCache(epoch=cache.epoch, keys_pre_rope=cache.keys_pre_rope, values=cache.values,
                                        cond_latents=torch.zeros(1, 16, GRID[0] + 1, 3, 4))


if __name__ == "__main__":
    unittest.main()
