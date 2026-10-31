"""Step 7 runtime gate: RAW identity lineages, explicit NVENC archival, size-scaled encode deadlines, prewarm,
and content addresses that depend only on content (run telemetry in an unhashed store sidecar).

CPU contracts with the fake CausVid pipeline (tests.test_causvid_adapter) and the software NVENC stand-in
(tests.test_nvenc_provider); native evidence is in artifacts/sglang_integration/runtime_gate/step7/.
"""

from __future__ import annotations

import functools
import os
import shutil
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

import torch

from reelcache import _hevc, nvenc
from reelcache import restore_kernels as rk
from reelcache.adapters.causvid import CausVidArchiveNativeAdapter
from reelcache.archive_pipeline import ArchivePipeline, RawArchive
from reelcache.archive_restore import RawLineageRestorer, SoftwareLineageRestorer
from reelcache.codecs import decode_packed
from reelcache.errors import ConfigurationError, PendingError, UnsupportedError
from reelcache.lineage_commit import ArchiveOptions
from reelcache.manager import KVRuntime
from reelcache.predictor_registry import RankPredictorRegistry
from reelcache.rank_method import RankMethodConfig
from reelcache.types import CodecSpec, EnergyTargets, ModelIdentity
from tests import step6_support as s6
from tests import test_causvid_adapter as _causvid_adapter_tests
from tests.test_causvid_adapter import CONFIG, FakePipeline, causal_rope_apply, initial_state
from tests.test_nvenc_provider import DEVICE, FakeAPI

MODEL = ModelIdentity("causvid-cpu-fake", "fake-source-not-native-proof", 2, 8, "tokens_channels", "float32")
METHOD = RankMethodConfig(EnergyTargets("t", 0.98, 0.95, True),
                          CodecSpec(kind="svd_hevc", rank=1, inter_prediction=False))
FREQS = torch.tensor([0.3, 0.9], dtype=torch.float64)


def fake_rope_table(total_frames):
    frames = torch.arange(total_frames).repeat_interleave(CONFIG.frame_tokens).double()
    angles = frames[:, None] * FREQS[None]
    return rk.RopeTable(torch.cos(angles), torch.sin(angles), None)


class RawIdentityLifecycleTests(unittest.TestCase):
    """RAW lineages restore the exact native cache: resume, fork, rewind and restart are bitwise."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / ".test-work" / f"step7-raw-{uuid.uuid4().hex}"
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def adapter(self, pipe, runtime, *, archive=None, restorer=None, branch="main"):
        adapter = CausVidArchiveNativeAdapter(pipe, runtime, branch, config=CONFIG, archive=archive,
                                              restorer=restorer, rope=causal_rope_apply)
        adapter.install()
        self.addCleanup(lambda: adapter.close(drain=False) if adapter._installed else None)
        return adapter

    def build(self, name, *, raw=True):
        torch.manual_seed(31)
        pipe = FakePipeline()
        pipe.generator.model.freqs = FREQS
        runtime = KVRuntime(self.root / name, MODEL)
        self.addCleanup(lambda: runtime.close() if not runtime._closed else None)
        archive = RawArchive(runtime) if raw else None
        adapter = self.adapter(pipe, runtime, archive=archive, restorer=RawLineageRestorer(runtime))
        noise, conditioning, rng = initial_state()
        adapter.start(noise, conditioning, rng, conditioning_id="fixed-prompt")
        return pipe, runtime, adapter

    def test_raw_resume_fork_rewind_and_restart_are_bitwise(self):
        _, _, reference = self.build("reference", raw=False)
        for _ in range(3):
            reference.step()
        expected, native_k, native_v = reference.latents, [k.clone() for k in reference._k], [
            v.clone() for v in reference._v]
        pipe, runtime, adapter = self.build("raw")
        tickets = [adapter.step() for _ in range(3)]
        self.assertTrue(all(ticket.done() for ticket in tickets))
        self.assertTrue(torch.equal(adapter.latents, expected))
        head = adapter.snapshot()["node_ref"]
        path = runtime.node_path(head)
        node1 = path[2][0]
        record = runtime.load_record(runtime.load_node(node1).matrix_records[0][1])
        self.assertEqual(record.codec.kind, "raw")

        # Resume after chunk 1: the restored cache equals the native cache bit for bit (K re-roped natively).
        result = adapter.restore(node1, "resume", rope_table=fake_rope_table(9))
        tokens = 2 * CONFIG.chunk_tokens
        self.assertEqual(result["entries"], 2 * 4)
        for layer in range(2):
            self.assertTrue(torch.equal(adapter._k[layer][:, :tokens], native_k[layer][:, :tokens]))
            self.assertTrue(torch.equal(adapter._v[layer][:, :tokens], native_v[layer][:, :tokens]))
        adapter.step()
        self.assertTrue(torch.equal(adapter.latents, expected))

        # Fork with another seed: replayable, and the parent lineage is untouched.
        forks = []
        for name in ("fork-a", "fork-b"):
            adapter.restore(node1, name, rope_table=fake_rope_table(9))
            adapter._rng.manual_seed(1234)
            adapter.step()
            forks.append(adapter.latents)
        self.assertTrue(torch.equal(forks[0], forks[1]))
        self.assertFalse(torch.equal(forks[0], expected))
        self.assertEqual(runtime.lineage_head("main"), head)
        self.assertEqual(runtime.load_node(runtime.lineage_head("fork-a")).parent_ref, node1)

        # Rewind the main lineage to node 1: resuming its head reproduces the original continuation.
        runtime.rewind_lineage("main", node1)
        adapter.restore(runtime.lineage_head("main"), "rewound", rope_table=fake_rope_table(9))
        adapter.step()
        self.assertTrue(torch.equal(adapter.latents, expected))

        # Restart: a new runtime over the same store (all caches dropped) restores the stored node exactly.
        adapter.close(drain=False)
        runtime.close()
        reopened = KVRuntime(self.root / "raw", MODEL)
        self.addCleanup(lambda: reopened.close() if not reopened._closed else None)
        restarted = self.adapter(pipe, reopened, restorer=RawLineageRestorer(reopened), branch="after-restart")
        restarted.restore(node1, "after-restart", rope_table=fake_rope_table(9))
        restarted.step()
        self.assertTrue(torch.equal(restarted.latents, expected))

    def test_raw_restore_requires_native_rope_and_raw_lineages(self):
        _, runtime, adapter = self.build("rules")
        adapter.step()
        head = adapter.snapshot()["node_ref"]
        restorer = RawLineageRestorer(runtime)
        with self.assertRaises(ConfigurationError):
            restorer.restore(head, lambda *a: None, lambda *a: None)
        lossy = KVRuntime(self.root / "lossy", s6.MODEL)
        self.addCleanup(lossy.close)
        lossy.create_lineage("main")
        matrices, contexts = s6.chunk(0)
        lossy.commit_node("main", "chunk-000000", matrices, contexts, method=s6.method(),
                          registry=s6.registry())
        later, later_contexts = s6.chunk(1)
        with self.assertRaises(UnsupportedError):
            lossy.commit_raw_node("main", "chunk-000001", later, later_contexts)
        with self.assertRaises(UnsupportedError):
            RawArchive(runtime).submit("main", "c", {}, {}, kind="recache")

    def test_raw_history_restores_exact_sources_through_acquire_node_history(self):
        runtime = KVRuntime(self.root / "history", s6.MODEL)
        self.addCleanup(runtime.close)
        runtime.create_lineage("main")
        chunks = [s6.chunk(index) for index in range(2)]
        for index, (matrices, contexts) in enumerate(chunks):
            commit = runtime.commit_raw_node("main", f"chunk-{index:06d}", matrices, contexts,
                                             execution_state={"marker": torch.tensor([index])})
        self.assertEqual(commit.report["representation"], "raw")
        for key in s6.KEYS:
            with runtime.acquire_node_history(commit.node_ref, key) as lease:
                expected = torch.cat([chunks[0][0][key], chunks[1][0][key]])
                self.assertTrue(torch.equal(lease.tensor, expected))


class NVENCArchivePipelineTests(unittest.TestCase):
    """The explicit NVENC encoder option (software stand-in on CPU; RTX PRO 6000 in the GPU receipts)."""

    def setUp(self):
        self.factory = s6.RuntimeFactory(self)
        self.encoder = nvenc.NVENCEncoder(api=FakeAPI(), device=DEVICE)

    def method(self, **kwargs):
        return RankMethodConfig(EnergyTargets("step7-nvenc", 0.95, 0.9, True), nvenc.nvenc_spec(1, **kwargs))

    def test_pipeline_encodes_nvenc_specifications_with_the_explicit_encoder(self):
        runtime = self.factory("nvenc")
        runtime.create_lineage("main")
        pipeline = ArchivePipeline(runtime, method=self.method(), registry=s6.registry(),
                                   options=ArchiveOptions(encode_threads=1), encoder=self.encoder)
        self.addCleanup(lambda: pipeline.close(drain=False, timeout=60))
        chunks = [s6.chunk(index) for index in range(2)]
        for index, (matrices, contexts) in enumerate(chunks):
            pipeline.submit("main", f"chunk-{index:06d}", matrices, contexts)
        head = pipeline.drain("main", timeout=120)
        node = runtime.load_node(head)
        record = runtime.load_record(node.matrix_records[0][1])
        packet = runtime.load_packet(record.packet_ref)
        self.assertEqual(packet["metadata"]["hevc"]["encoder"], "nvenc")
        self.assertEqual(packet["metadata"]["codec"]["preset"], "nvenc-p1-hq")
        self.assertEqual(record.codec.preset, "nvenc-p1-hq")
        self.assertGreaterEqual(self.encoder.stats["streams"], 2)
        decoded = decode_packed(runtime.node_pack(record.packet_ref), dtype=torch.float32)
        for key, value in decoded.items():
            source = chunks[1][0][key].float()
            error = float((value - source).norm() / source.norm())
            self.assertLess(error, 0.4, key)
        restored = SoftwareLineageRestorer(runtime).history(head)
        self.assertEqual(len(restored), 2)

    def test_encoders_never_substitute(self):
        runtime = self.factory("rules")
        with self.assertRaises(UnsupportedError):
            ArchivePipeline(runtime, method=self.method(), registry=s6.registry())
        with self.assertRaises(ConfigurationError):
            ArchivePipeline(runtime, method=s6.method(), registry=s6.registry(), encoder=self.encoder)
        with self.assertRaises(ConfigurationError):
            ArchivePipeline(runtime, method=self.method(qp=9), registry=s6.registry(), encoder=self.encoder)
        with self.assertRaises(ConfigurationError):
            ArchivePipeline(runtime, method=self.method(), registry=s6.registry(), encoder=object())


class EncodeDeadlineTests(unittest.TestCase):
    def test_deadline_scales_with_picture_samples(self):
        self.assertEqual(_hevc.size_scaled_timeout(0), 300.0)
        self.assertEqual(_hevc.size_scaled_timeout(180 * 3 * 576 * 320), 300.0)  # CausVid node: the floor
        self.assertEqual(_hevc.size_scaled_timeout(1_400_000_000), 1400.0)  # LongCat-size node
        self.assertEqual(_hevc.size_scaled_timeout(1_000_000_000, seconds_per_megasample=0.25), 300.0)
        with self.assertRaises(ConfigurationError):
            _hevc.size_scaled_timeout(-1)

    def test_encodes_pass_the_scaled_deadline_to_ffmpeg(self):
        pictures = torch.zeros((2, 3, 144, 144), dtype=torch.uint8).numpy()
        seen = []
        original = _hevc._run

        def spy(command, **kwargs):
            seen.append(kwargs.get("timeout"))
            return original(command, **kwargs)

        with mock.patch.object(_hevc, "_run", spy):
            _hevc.encode_pictures(pictures, "yuv444p")
        self.assertIn(_hevc.size_scaled_timeout(pictures.size), seen)


class PrewarmTests(unittest.TestCase):
    def test_install_prewarm_prepares_stored_history_through_the_restorer(self):
        calls = []

        class Restorer:
            def prewarm_lineages(self, branch_ids=None, *, node_refs=None):
                calls.append((branch_ids, node_refs))
                return {"heads": 1, "packets": 2, "seconds": 0.0}

        root = Path(__file__).resolve().parents[1] / ".test-work" / f"step7-prewarm-{uuid.uuid4().hex}"
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        runtime = KVRuntime(root, MODEL)
        self.addCleanup(runtime.close)
        adapter = CausVidArchiveNativeAdapter(FakePipeline(), runtime, "main", config=CONFIG, restorer=Restorer(),
                                              rope=causal_rope_apply)
        adapter.install(prewarm=True)
        self.addCleanup(adapter.close)
        self.assertEqual(calls, [(None, None)])
        self.assertEqual(adapter.prewarm(node_refs=["x"])["packets"], 2)
        self.assertEqual(calls[-1], (None, ["x"]))
        self.assertEqual(len(adapter.stats_["prewarm"]), 2)
        with self.assertRaises(ConfigurationError):
            CausVidArchiveNativeAdapter(FakePipeline(), runtime, "other", config=CONFIG).prewarm()


class ArchiveWarmupTests(unittest.TestCase):
    """ArchivePipeline.prewarm pays the cold start on the fitter thread without side effects on outputs."""

    def setUp(self):
        self.runtimes = s6.RuntimeFactory(self)

    def pipeline(self, runtime, reg):
        pipeline = ArchivePipeline(runtime, method=s6.method(1), registry=reg, options=ArchiveOptions(encode_threads=1),
                                   encode_workers=2, max_source_chunks=2, max_unpublished=3)
        self.addCleanup(lambda: pipeline.close(drain=False, timeout=60))
        return pipeline

    def test_prewarm_has_no_side_effects_on_store_registry_rng_or_content_addresses(self):
        reference = ContentAddressTests.chain(self, self.runtimes("ref"), "pipeline")
        runtime = self.runtimes("warm")
        runtime.create_lineage("main")
        reg = s6.registry()
        pipeline = self.pipeline(runtime, reg)
        before = (runtime.store.stats()["object_count"], reg.to_dict(), torch.get_rng_state())
        report = pipeline.prewarm(s6.chunk(0)[1])
        self.assertEqual(len(report["plan_seconds"]), 2)
        self.assertEqual(report["keys"], len(s6.KEYS))
        self.assertEqual(report["decisions"][0], {"fresh": len(s6.KEYS)})
        self.assertTrue(set(report["decisions"][1]) & {"share", "augment", "refit"})
        self.assertGreater(report["encode_seconds"], 0.0)
        self.assertEqual(pipeline.stats()["prewarm"][0]["keys"], len(s6.KEYS))
        self.assertEqual(runtime.store.stats()["object_count"], before[0])
        self.assertEqual(reg.to_dict(), before[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        small = pipeline.prewarm(s6.chunk(0)[1], encode=False, max_frames=1)
        self.assertNotIn("encode_seconds", small)
        tickets = [pipeline.submit("main", f"chunk-{i:03d}", *s6.chunk(i)) for i in range(3)]
        pipeline.drain("main", timeout=300)
        self.assertEqual([ticket.result() for ticket in tickets], reference)
        with self.assertRaises(ConfigurationError):
            pipeline.prewarm({})

    def test_adapter_prewarm_archive_warms_the_chunk_geometry(self):
        seen = []

        class Archive:
            max_source_chunks = 2

            def prewarm(self, contexts, *, encode=True, max_frames=None):
                seen.append((contexts, encode, max_frames))
                return {"seconds": 0.0}

        root = Path(__file__).resolve().parents[1] / ".test-work" / f"step7-warm-{uuid.uuid4().hex}"
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        runtime = KVRuntime(root, MODEL)
        self.addCleanup(runtime.close)
        adapter = CausVidArchiveNativeAdapter(FakePipeline(), runtime, "main", config=CONFIG, archive=Archive(),
                                              rope=causal_rope_apply)
        adapter.install(prewarm_archive=True)
        self.addCleanup(lambda: adapter.close(drain=False))
        contexts, encode, max_frames = seen[0]
        self.assertTrue(encode)
        self.assertIsNone(max_frames)
        self.assertEqual(set(contexts), set(adapter.keys))
        context = contexts[adapter.keys[0]]
        self.assertEqual((context.token_start, context.frames, context.height, context.width),
                         (0, CONFIG.frames_per_chunk, CONFIG.grid_height, CONFIG.grid_width))
        self.assertEqual(adapter.prewarm_archive(encode=False, max_frames=1), {"seconds": 0.0})
        self.assertEqual(seen[-1][1:], (False, 1))
        self.assertEqual(len(adapter.stats_["prewarm_archive"]), 2)


class MixedSourceDtypeTests(unittest.TestCase):
    """A chunk may mix FP32 pre-RoPE K with BF16 V (RAVEN's native capture); each matrix keeps its own dtype."""

    def setUp(self):
        self.runtimes = s6.RuntimeFactory(self)

    @staticmethod
    def mixed(index):
        import dataclasses
        matrices, contexts = s6.chunk(index)
        for key in list(matrices):
            if key.target == "K":
                matrices[key] = matrices[key].float() + 1e-3 * torch.randn(matrices[key].shape,
                                                                             generator=torch.Generator().manual_seed(index))
                contexts[key] = dataclasses.replace(contexts[key], source_dtype="float32")
        return matrices, contexts

    def test_fingerprints_cover_each_matrix_dtype(self):
        from reelcache.lineage_commit import source_fingerprint, source_fingerprints
        matrices, _ = self.mixed(0)
        values = list(matrices.values())
        fingerprints = source_fingerprints(values)
        self.assertEqual(fingerprints, [source_fingerprint(value) for value in values])
        self.assertEqual({text.split(":")[2] for text in fingerprints}, {"torch.float32", "torch.bfloat16"})
        changed = values[0].clone()
        changed.view(torch.int32)[0, 0] ^= 1  # the low mantissa bit of an FP32 source changes its fingerprint
        self.assertNotEqual(source_fingerprint(changed), fingerprints[0])

    def test_fingerprints_stream_row_tiles_with_unchanged_values(self):
        """Row-tiled fingerprints (bounded int64 temporaries) equal the whole-matrix formula bit for bit."""
        from reelcache import lineage_commit as lc

        def whole_matrix(tensor):
            size = tensor.element_size()
            bits = tensor.contiguous().view(torch.int16 if size == 2 else torch.int32).to(torch.int64)
            parts = [bits] if size == 2 else [bits & 0xFFFF, bits >> 16]
            column = torch.arange(tensor.shape[1], dtype=torch.int64)
            row = torch.arange(tensor.shape[0], dtype=torch.int64)
            values = []
            for part in parts:
                for a, b, square in ((37, 101, False), (613, 211, False), (1, 17, True)):
                    per_row = ((part * part if square else part) * ((column * a + 7) % 1021 + 1)).sum(1)
                    per_row = per_row.remainder(lc._FINGERPRINT_PRIME)
                    values.append((per_row * ((row * b + 3) % 1019 + 1)).sum().remainder(lc._FINGERPRINT_PRIME))
            return torch.stack(values)
        generator = torch.Generator().manual_seed(4)
        rows = 2 * (lc.FINGERPRINT_TILE_ELEMENTS // 2048) + 333
        for dtype in (torch.bfloat16, torch.float32, torch.float16):
            value = (1e3 * torch.randn(rows, 2048, generator=generator)).to(dtype)
            self.assertTrue(torch.equal(lc._fingerprint_terms(value), whole_matrix(value)), dtype)
            short = value[:17]
            self.assertTrue(torch.equal(lc._fingerprint_terms(short), whole_matrix(short)), dtype)

    def test_mixed_dtype_chunks_commit_pipeline_and_restore(self):
        runtime = self.runtimes("mixed")
        runtime.create_lineage("main")
        reg = s6.registry()
        refs = [runtime.commit_node("main", f"chunk-{i:03d}", *self.mixed(i), method=s6.method(), registry=reg,
                                    options=ArchiveOptions(encode_threads=1)).node_ref for i in range(2)]
        dtypes = {runtime.load_record(ref).context.key.target: runtime.load_record(ref).context.dtype
                  for _, ref in runtime.load_node(refs[-1]).matrix_records}
        self.assertEqual(dtypes, {"K": "float32", "V": "bfloat16"})
        pipe_runtime = self.runtimes("mixed-pipe")
        pipe_runtime.create_lineage("main")
        pipeline = ArchivePipeline(pipe_runtime, method=s6.method(), registry=s6.registry(),
                                   options=ArchiveOptions(encode_threads=1))
        self.addCleanup(lambda: pipeline.close(drain=False, timeout=60))
        tickets = [pipeline.submit("main", f"chunk-{i:03d}", *self.mixed(i)) for i in range(2)]
        pipeline.drain("main", timeout=300)
        self.assertEqual([ticket.result() for ticket in tickets], refs)
        key = next(key for key in s6.KEYS if key.target == "K")
        with runtime.acquire_node_history(refs[-1], key, dtype=torch.float32) as lease:
            self.assertEqual(tuple(lease.tensor.shape), (2 * s6.TOKENS, 128))


def _rows_eigh(grams):
    values, columns = torch.linalg.eigh(grams)
    return values, columns.transpose(1, 2).contiguous()


class EighSubBatchTests(unittest.TestCase):
    """Batched eigensolves split automatically (same solver) when cuSOLVER rejects the size or memory is short."""

    @staticmethod
    def grams(batch, n, seed=3):
        x = torch.randn(batch, 3 * n, n, generator=torch.Generator().manual_seed(seed))
        return x.transpose(1, 2) @ x

    def test_rejected_batch_sizes_split_into_sub_batches_of_the_same_solver(self):
        from reelcache.gpu_fitting import BoundedEigh
        seen = []

        def solver(grams):
            seen.append(int(grams.shape[0]))
            return _rows_eigh(grams)
        eigh = BoundedEigh("test", solver, workspace=lambda b, n, d: None if b > 32 else 0, total_memory=1 << 40)
        grams = self.grams(60, 8)
        values, vectors = eigh(grams)
        expected = _rows_eigh(grams)
        self.assertTrue(torch.equal(values, expected[0]) and torch.equal(vectors, expected[1]))
        self.assertEqual(seen, [30, 30])
        self.assertEqual(eigh.last, {"batch": 60, "n": 8, "sub_batch": 30, "solver_calls": 2,
                                     "reason": "solver_rejects_batch_size"})
        eigh(self.grams(20, 8))
        self.assertEqual(eigh.last["sub_batch"], 20)
        self.assertEqual(eigh.last["reason"], "whole_batch")
        with self.assertRaises(UnsupportedError):
            BoundedEigh("test", solver, workspace=lambda b, n, d: None, total_memory=1 << 40)(grams)

    def test_memory_bound_uses_the_largest_admissible_halving(self):
        from reelcache.gpu_fitting import BoundedEigh
        n = 8
        per_matrix = 8 * n * n + 4 * n
        eigh = BoundedEigh("test", _rows_eigh, memory_fraction=0.125, workspace=lambda b, n, d: 0,
                           total_memory=8 * 10 * per_matrix)
        grams = self.grams(60, n)
        values, vectors = eigh(grams)
        self.assertEqual((eigh.last["sub_batch"], eigh.last["solver_calls"], eigh.last["reason"]),
                         (8, 8, "memory_bound"))
        expected = _rows_eigh(grams)
        self.assertTrue(torch.equal(values, expected[0]) and torch.equal(vectors, expected[1]))
        # select_eigh_solver wraps every batched solver; the explicit per-matrix loop stays unwrapped.
        from reelcache import gpu_fitting
        with mock.patch.object(gpu_fitting, "torch_batched_eigh_is_xsyevbatched", return_value=True):
            label, solver = gpu_fitting.select_eigh_solver("auto")
            self.assertIsInstance(solver, BoundedEigh)
            self.assertEqual(solver.label, label)
        self.assertNotIsInstance(gpu_fitting.select_eigh_solver("torch_loop")[1], BoundedEigh)

    def test_planner_records_sub_batches_and_decisions_do_not_change(self):
        from reelcache.gpu_fitting import BoundedEigh
        from reelcache.rank_method import LatestMethodPlanner, anchor_sketches
        from reelcache.types import ActiveBasisBinding
        matrices, contexts = s6.chunk(0)
        bindings = {key: ActiveBasisBinding() for key in matrices}
        rows = {key: None for key in matrices}
        ids = {key: f"id-{key.layer}-{key.target}" for key in matrices}
        plans = []
        for eigh in (_rows_eigh, BoundedEigh("test", _rows_eigh, workspace=lambda b, n, d: None if b > 2 else 0,
                                             total_memory=1 << 40)):
            planner = LatestMethodPlanner(s6.exact_method(), s6.registry().pin(), device="cpu", eigh=eigh,
                                          eigh_label="test")
            plans.append(planner.plan(matrices, contexts, bindings, rows, chunk_id="c0", source_ids=ids))
        whole, split = plans
        self.assertEqual(whole.batches, [])
        self.assertEqual({entry["stage"] for entry in split.batches}, {"eigh"})
        self.assertTrue(all(entry["sub_batch"] == 2 for entry in split.batches))
        for key in matrices:
            a, b = whole.plans[key], split.plans[key]
            self.assertEqual((a.decision, a.rank_view), (b.decision, b.rank_view))
            self.assertTrue(torch.equal(a.new_rows, b.new_rows))
        scaled = [value.float() for value in matrices.values()]
        totals = [float(value.square().sum()) for value in scaled]
        seeds = list(range(len(scaled)))
        # Sketches are computed per matrix (no stacked batch; Part C): a stack bound never splits anything and
        # each item's sketch is the same alone or with the others.
        record = []
        one = anchor_sketches(scaled, totals, seeds, anchor_rank=16, power_iterations=1)
        parts = anchor_sketches(scaled, totals, seeds, anchor_rank=16, power_iterations=1,
                                max_stack_bytes=2 * 4 * (s6.TOKENS * 128 + s6.TOKENS * 16 + 2 * 128 * 16),
                                record=record)
        alone = [anchor_sketches([matrix], [total], [seed], anchor_rank=16, power_iterations=1)[0]
                 for matrix, total, seed in zip(scaled, totals, seeds)]
        self.assertEqual(record, [])
        for a, b, c in zip(one, parts, alone):
            self.assertEqual(a["captured_energy"], b["captured_energy"])
            self.assertEqual(a["captured_energy"], c["captured_energy"])
            self.assertTrue(torch.equal(a["sketch"], b["sketch"]))
            self.assertTrue(torch.equal(a["sketch"], c["sketch"]))


def _size_dependent_eigh(grams):
    """Like cusolverDnXsyevBatched: a matrix's eigenpairs depend on the call's batch size (not on the others)."""
    values, vectors = _rows_eigh(grams)
    return values * (1 + 1e-6 * grams.shape[0]), vectors


class FixedEighSubBatchTests(unittest.TestCase):
    """A fixed eigensolver sub-batch (padded) makes a matrix's eigenpairs independent of what it is solved with."""

    grams = staticmethod(EighSubBatchTests.grams)

    def test_fixed_sub_batches_pad_with_identity_and_are_composition_independent(self):
        from reelcache.gpu_fitting import BoundedEigh
        seen = []

        def solver(grams):
            seen.append((int(grams.shape[0]), bool(torch.equal(grams[-1], torch.eye(grams.shape[1])))))
            return _size_dependent_eigh(grams)
        eigh = BoundedEigh("test", solver, sub_batch_size=lambda n: 4)
        grams = self.grams(10, 8)
        values, vectors = eigh(grams)
        self.assertEqual(seen, [(4, False), (4, False), (4, True)])
        self.assertEqual(eigh.last, {"batch": 10, "n": 8, "sub_batch": 4, "solver_calls": 3,
                                     "reason": "fixed_sub_batch", "padded": 2})
        self.assertEqual(tuple(values.shape), (10, 8))
        for start, stop in ((0, 1), (2, 7), (9, 10), (0, 10)):
            part = eigh(grams[start:stop])
            self.assertTrue(torch.equal(part[0], values[start:stop]) and torch.equal(part[1], vectors[start:stop]))
        # The raw size-dependent solver is not composition independent: the fixed sub-batch is what makes it so.
        self.assertFalse(torch.equal(solver(grams[:1])[0][0], solver(grams[:4])[0][0]))
        alone = BoundedEigh("test", _size_dependent_eigh, sub_batch_size=lambda n: 1)
        self.assertEqual(alone(grams[3:4])[0].tolist(), alone(grams)[0][3:4].tolist())
        self.assertEqual(alone.last["reason"], "per_matrix")

    def test_default_policy_solves_large_matrices_one_per_call(self):
        from reelcache.gpu_fitting import BoundedEigh, PER_MATRIX_EIGH_MIN_N, default_eigh_sub_batch
        self.assertEqual(PER_MATRIX_EIGH_MIN_N, 4096)
        self.assertEqual([default_eigh_sub_batch(n) for n in (8, 1536, 3072, 4095, 4096, 5120)],
                         [None, None, None, None, 1, 1])
        eigh = BoundedEigh("test", _rows_eigh, workspace=lambda b, n, d: 0, total_memory=1 << 40)
        self.assertEqual(eigh.sub_batch(60, 4096, "cpu"), (1, "per_matrix"))
        self.assertEqual(eigh.sub_batch(60, 3072, "cpu"), (60, "whole_batch"))
        derived = eigh.derive(memory_fraction=1.0, total_memory=1 << 30)
        self.assertIs(derived.sub_batch_size, eigh.sub_batch_size)
        self.assertEqual((derived.memory_fraction, derived.total_memory), (1.0, 1 << 30))

    def test_a_fixed_sub_batch_is_never_changed_to_fit(self):
        from reelcache.gpu_fitting import BoundedEigh
        n = 8
        per_matrix = 8 * n * n + 4 * n
        fits = BoundedEigh("test", _rows_eigh, memory_fraction=1.0, workspace=lambda b, n, d: 0,
                           total_memory=4 * per_matrix, sub_batch_size=lambda n: 4)
        self.assertEqual(fits.sub_batch(9, n, "cpu"), (4, "fixed_sub_batch"))
        fits(self.grams(9, n))
        self.assertNotIn("over_budget", fits.last)
        tight = BoundedEigh("test", _rows_eigh, memory_fraction=1.0, workspace=lambda b, n, d: 0,
                            total_memory=4 * per_matrix - 1, sub_batch_size=lambda n: 4)
        values, _ = tight(self.grams(9, n))
        self.assertEqual((tight.last["sub_batch"], tight.last["solver_calls"], tight.last["over_budget"]),
                         (4, 3, True))
        self.assertTrue(torch.equal(values, fits(self.grams(9, n))[0]))
        rejected = BoundedEigh("test", _rows_eigh, workspace=lambda b, n, d: None if b > 2 else 0,
                               total_memory=1 << 40, sub_batch_size=lambda n: 4)
        with self.assertRaises(UnsupportedError):
            rejected(self.grams(9, n))
        with self.assertRaises(ConfigurationError):
            BoundedEigh("test", _rows_eigh, sub_batch_size=lambda n: 0)(self.grams(2, n))
        with self.assertRaises(ConfigurationError):
            BoundedEigh("test", _rows_eigh, sub_batch_size=4)

    def test_sub_batches_use_unchecked_calls_and_check_real_slots_once(self):
        from reelcache.gpu_fitting import BoundedEigh

        def unchecked(grams):
            values, vectors = _rows_eigh(grams)
            return values, vectors, torch.zeros(grams.shape[0], dtype=torch.int32)

        def solver(grams):
            raise AssertionError("the checked call is not used for sub-batches")
        solver.unchecked = unchecked
        eigh = BoundedEigh("test", solver, sub_batch_size=lambda n: 2)
        values, _ = eigh(self.grams(5, 8))
        self.assertEqual(eigh.last["solver_calls"], 3)
        self.assertTrue(torch.equal(values, _rows_eigh(self.grams(5, 8))[0]))

        def failing(grams):
            values, vectors = _rows_eigh(grams)
            info = torch.zeros(grams.shape[0], dtype=torch.int32)
            info[-1] = 1 if grams.shape[0] == 2 and not torch.equal(grams[-1], torch.eye(grams.shape[1])) else 0
            return values, vectors, info
        solver.unchecked = failing
        with self.assertRaises(ConfigurationError):
            BoundedEigh("test", solver, sub_batch_size=lambda n: 2)(self.grams(5, 8))
        # A padded slot's flag is ignored (its results are dropped).
        flags = []

        def padded_fails(grams):
            values, vectors = _rows_eigh(grams)
            info = torch.zeros(grams.shape[0], dtype=torch.int32)
            if torch.equal(grams[-1], torch.eye(grams.shape[1])):
                info[-1] = 1
            flags.append(int(info.sum()))
            return values, vectors, info
        solver.unchecked = padded_fails
        BoundedEigh("test", solver, sub_batch_size=lambda n: 2)(self.grams(3, 8))
        self.assertEqual(flags, [0, 1])


class LayerAheadRestoreTests(unittest.TestCase):
    """LineageRestorer.restore(layer_ahead=d) bounds the canvas arena to ~d+1 layers with identical results."""

    def test_bounded_layer_ahead_restores_identically_within_one_layer_of_canvases(self):
        from reelcache.archive_restore import LineageRestorer
        from reelcache.errors import CapacityError
        from reelcache.types import MatrixKey
        from tests.test_packed_nvdec import SoftwarePool
        runtime = s6.RuntimeFactory(self)("layer-ahead")
        runtime.create_lineage("main")
        reg = s6.registry()
        for index in range(3):
            head = runtime.commit_node("main", f"chunk-{index:03d}", *s6.chunk(index), method=s6.method(),
                                       registry=reg, options=ArchiveOptions(encode_threads=1)).node_ref

        def destination_factory():
            caches = {}

            def destination(node_id, key, tile):
                key = key if isinstance(key, MatrixKey) else key[0]
                if key not in caches:
                    caches[key] = torch.zeros((1, 3 * s6.TOKENS, 1, 128), dtype=torch.float32)
                return rk.NativeView(caches[key], "BTHD", token_offset=tile.token_start)
            return caches, destination

        whole = LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 30)
        reference, destination = destination_factory()
        report = whole.restore(head, destination)
        self.assertEqual((report["layer_ahead"], report["layer_ahead_auto"]), (None, False))
        refs, pins = whole.node_refs(whole.history(head))
        batch = whole.restorer.plan(refs)
        per_layer = {}
        for job in batch.jobs:
            size = job.pictures * job.packet.planes * job.packet.height * job.packet.width
            per_layer[job.first_layer] = per_layer.get(job.first_layer, 0) + size
        batch.close()
        whole.finish(pins)
        self.assertGreater(len(per_layer), 1)
        budget = max(per_layer.values())
        self.assertLess(budget, sum(per_layer.values()))
        bounded = LineageRestorer(runtime, SoftwarePool(), arena_bytes=budget)
        bounded.restorer.arena.wait_seconds = 0.2
        # The whole history does not fit the arena: the default selects the bounded mode itself (before Part C:
        # CapacityError) and restores identically.
        caches, destination = destination_factory()
        report = bounded.restore(head, destination)
        self.assertEqual((report["layer_ahead"], report["layer_ahead_auto"]), (0, True))
        for key, value in reference.items():
            self.assertTrue(torch.equal(caches[key], value), key)
        caches, destination = destination_factory()
        report = bounded.restore(head, destination, layer_ahead=0)
        self.assertEqual((report["layer_ahead"], report["layer_ahead_auto"]), (0, False))
        self.assertEqual(set(caches), set(reference))
        for key, value in reference.items():
            self.assertTrue(torch.equal(caches[key], value), key)
        self.assertLessEqual(bounded.restorer.arena.report()["peak_bytes"], budget)
        with self.assertRaises(ConfigurationError):
            bounded.restore(head, destination, layer_ahead=-1)
        # A depth that cannot fit still raises (explicit modes are never changed).
        tiny = LineageRestorer(runtime, SoftwarePool(), arena_bytes=budget)
        tiny.restorer.arena.budget_bytes = min(per_layer.values()) - 1
        tiny.restorer.arena.wait_seconds = 0.2
        with self.assertRaises(CapacityError):
            tiny.restore(head, destination_factory()[1], layer_ahead=0)

    def test_automatic_depth_is_the_deepest_window_that_fits(self):
        from reelcache.archive_restore import LineageRestorer

        class Job:
            def __init__(self, layer, size):
                self.first_layer, self.pictures = layer, size
                self.packet = type("Packet", (), {"planes": 1, "height": 1, "width": 1})()

        class Batch:
            def __init__(self, sizes):
                self.jobs = [Job(layer, size) for layer, size in enumerate(sizes)]

        restorer = LineageRestorer.__new__(LineageRestorer)
        restorer.restorer = type("Restorer", (), {"arena": type("Arena", (), {"budget_bytes": 100})()})()
        depth = lambda sizes: restorer._fitting_layer_ahead(Batch(sizes), list(range(len(sizes))))
        self.assertIsNone(depth([10] * 10))           # the whole history fits: submit everything
        self.assertEqual(depth([10] * 30), 8)         # windows of d + 2 = 10 layers
        self.assertEqual(depth([60] * 5), 0)          # even two layers exceed the budget: one layer ahead
        self.assertEqual(depth([10, 10, 90, 10, 10]), 0)


class MatmulPrecisionPolicyTests(unittest.TestCase):
    """TF32 policy: flags are never toggled; 'fp32_verified_fp64' measures every acceptance in FP64 products."""

    def test_policy_validation_fingerprint_and_refusal(self):
        from reelcache.rank_method import require_fp32_matmul
        default = s6.method()
        declared = RankMethodConfig(default.targets, default.codec, matmul_precision="fp32_verified_fp64")
        self.assertEqual(RankMethodConfig(default.targets, default.codec, matmul_precision="fp32").fingerprint,
                         RankMethodConfig(default.targets, default.codec).fingerprint)
        self.assertNotIn("matmul_precision", default.to_dict())
        self.assertEqual(declared.to_dict()["matmul_precision"], "fp32_verified_fp64")
        self.assertNotEqual(declared.fingerprint, default.fingerprint)
        self.assertEqual((default.measurement_products, declared.measurement_products), ("fp32", "fp64"))
        with self.assertRaises(ConfigurationError):
            RankMethodConfig(default.targets, default.codec, matmul_precision="tf32")
        with mock.patch.object(torch, "get_float32_matmul_precision", return_value="high"):
            # Planning products never read the flags (reelcache.exact_blas): no flag state is refused.
            require_fp32_matmul(torch.device("cuda", 0))
            require_fp32_matmul(torch.device("cuda", 0), "fp32_verified_fp64")
            require_fp32_matmul(torch.device("cpu"))
        with self.assertRaises(ConfigurationError):
            require_fp32_matmul(torch.device("cpu"), "tf32")

    def test_fp64_products_measure_the_stored_factors_exactly(self):
        from reelcache.deferred_reads import projection_routine, run_one
        matrices, _ = s6.chunk(1)
        matrix = next(iter(matrices.values()))
        basis = torch.linalg.qr(matrix.float().T).Q.T[:24].to(torch.float16)
        reference = run_one(projection_routine(matrix, basis, tile_rows=64))
        exact = run_one(projection_routine(matrix, basis, tile_rows=64, products="fp64"))
        coefficients = (matrix.double() @ basis.double().T).float()
        self.assertTrue(torch.equal(exact.coefficients, coefficients))
        residual = matrix.double() - coefficients.double() @ basis.double()
        relative = float(residual.square().sum().sqrt() / matrix.double().square().sum().sqrt())
        self.assertAlmostEqual(exact.relative_residual, relative, places=12)
        self.assertAlmostEqual(exact.relative_residual, reference.relative_residual, places=5)
        with self.assertRaises(ConfigurationError):
            run_one(projection_routine(matrix, basis, products="tf32"))

    def test_declared_policy_plans_and_commits_meet_targets(self):
        default = s6.method()
        declared = RankMethodConfig(default.targets, default.codec, matmul_precision="fp32_verified_fp64")
        runtime = s6.RuntimeFactory(self)("tf32-policy")
        runtime.create_lineage("main")
        reg = s6.registry()
        for index in range(3):
            commit = runtime.commit_node("main", f"chunk-{index:03d}", *s6.chunk(index), method=declared,
                                         registry=reg, options=ArchiveOptions(encode_threads=1))
            for row in commit.report["records"]:
                self.assertGreaterEqual(row["energy"]["stored_factor_energy"], row["energy"]["target"])
        self.assertEqual(runtime.node_commit_report(commit.node_ref)["method"]["matmul_precision"],
                         "fp32_verified_fp64")


class ArchiveStagingTests(unittest.TestCase):
    """CR-12p2-8: a node with more ranges than host staging slots streams them through; never a deadlock."""

    TIMEOUT = 300.0

    def setUp(self):
        self.runtimes = s6.RuntimeFactory(self)

    def archive(self, name, slots, workers=1):
        runtime = self.runtimes(name)
        runtime.create_lineage("main")
        pipeline = ArchivePipeline(runtime, method=s6.method(), registry=s6.registry(),
                                   options=ArchiveOptions(encode_threads=1), encode_workers=workers,
                                   max_source_chunks=1, max_staged_canvases=slots, max_unpublished=2)
        self.addCleanup(lambda: pipeline.close(drain=False, timeout=60))
        for index in range(3):
            pipeline.submit("main", f"chunk-{index:06d}", *s6.chunk(index))
        # A LongLive 1.0-style prompt-switch epoch: one node re-materializing three chunks (three canvases).
        ranges = [(f"chunk-{index:06d}", *s6.chunk(20 + index, token_start=index * s6.TOKENS)) for index in range(3)]
        ticket = pipeline.submit("main", "switch", {}, {}, kind="recache", ranges=ranges, epoch_id="e1")
        later = pipeline.submit("main", "chunk-000003", *s6.chunk(3))
        # Timeout-guarded: a stalled pipeline raises PendingError here instead of hanging the suite.
        head = pipeline.drain("main", timeout=self.TIMEOUT)
        return runtime, pipeline, ticket, later, head

    def test_more_ranges_than_staging_slots_stream_and_publish_identical_nodes(self):
        runtime, reference, ticket, later, head = self.archive("slots-4", 4)
        self.assertEqual(reference.stats()["streamed_ranges"], 0)
        recache = ticket.result(timeout=0)
        self.assertEqual(runtime.load_node(recache).node_kind, "recache")
        self.assertEqual(later.result(timeout=0), head)
        for slots, workers in ((1, 1), (2, 1), (2, 2)):
            with self.subTest(slots=slots, workers=workers):
                _, pipeline, streamed, after, tail = self.archive(f"slots-{slots}-{workers}", slots, workers)
                self.assertEqual(streamed.result(timeout=0), recache)  # same content addresses
                self.assertEqual(after.result(timeout=0), head)
                self.assertEqual(tail, head)
                self.assertGreater(pipeline.stats()["streamed_ranges"], 0)
                self.assertLessEqual(pipeline.stats()["staging"]["peak_in_use"], slots)

    def test_staging_never_hangs_when_encodes_fail_mid_node(self):
        runtime = self.runtimes("failing")
        runtime.create_lineage("main")
        pipeline = ArchivePipeline(runtime, method=s6.method(), registry=s6.registry(),
                                   options=ArchiveOptions(encode_threads=1), encode_workers=1,
                                   max_source_chunks=1, max_staged_canvases=1, max_unpublished=2)
        self.addCleanup(lambda: pipeline.close(drain=False, timeout=60))
        pipeline.submit("main", "chunk-000000", *s6.chunk(0))
        pipeline.submit("main", "chunk-000001", *s6.chunk(1))
        pipeline.drain("main", timeout=self.TIMEOUT)
        ranges = [(f"chunk-{index:06d}", *s6.chunk(20 + index, token_start=index * s6.TOKENS)) for index in range(2)]
        from reelcache import lineage_commit
        with mock.patch.object(lineage_commit, "encode_node_pack", side_effect=RuntimeError("encoder died")):
            ticket = pipeline.submit("main", "switch", {}, {}, kind="recache", ranges=ranges, epoch_id="e1")
            with self.assertRaises(Exception) as caught:
                pipeline.drain("main", timeout=self.TIMEOUT)
        self.assertNotIsInstance(caught.exception, PendingError)
        self.assertTrue(ticket.done())
        self.assertEqual(pipeline.stats()["staging"]["peak_in_use"], 1)
        pipeline.reset("main", timeout=60)
        self.assertEqual(pipeline._staging._in_use, 0)


class InferenceModeTests(unittest.TestCase):
    """Archive and restore inside torch.inference_mode (SGLang serves requests in it) and across modes."""

    def setUp(self):
        self.runtimes = s6.RuntimeFactory(self)

    @staticmethod
    def inference_chunk(index):
        with torch.inference_mode():
            matrices, contexts = s6.chunk(index)
            matrices = {key: value.clone() for key, value in matrices.items()}
        assert all(value.is_inference() for value in matrices.values())
        return matrices, contexts

    def test_archive_inside_inference_mode_publishes_the_normal_mode_content(self):
        reference = ContentAddressTests.chain(self, self.runtimes("normal"), "pipeline")
        runtime = self.runtimes("inference")
        runtime.create_lineage("main")
        pipeline = ArchivePipeline(runtime, method=s6.method(1), registry=s6.registry(),
                                   options=ArchiveOptions(encode_threads=1), encode_workers=2, max_source_chunks=2,
                                   max_unpublished=3)
        self.addCleanup(lambda: pipeline.close(drain=False, timeout=60))
        with torch.inference_mode():
            tickets = [pipeline.submit("main", f"chunk-{i:03d}", *self.inference_chunk(i)) for i in range(3)]
            head = pipeline.drain("main", timeout=300)
            frontiers = pipeline.frontiers("main")
        self.assertEqual([ticket.result() for ticket in tickets], reference)
        self.assertEqual(head, reference[-1])
        self.assertEqual(frontiers["generated_frontier"], frontiers["published_frontier"])
        sync = self.runtimes("inference-sync")
        sync.create_lineage("main")
        reg = s6.registry()
        with torch.inference_mode():
            refs = [sync.commit_node("main", f"chunk-{i:03d}", *self.inference_chunk(i), method=s6.method(1),
                                     registry=reg, options=ArchiveOptions(encode_threads=1)).node_ref
                    for i in range(3)]
            with sync.acquire_node_history(refs[-1], s6.KEYS[0], dtype=torch.float32) as lease:
                inside = lease.tensor.clone()
        self.assertEqual(refs, reference)
        with sync.acquire_node_history(refs[-1], s6.KEYS[0], dtype=torch.float32) as lease:
            self.assertTrue(torch.equal(lease.tensor, inside))

    def test_packed_restore_state_is_reusable_across_inference_and_normal_mode(self):
        from reelcache.archive_restore import LineageRestorer
        from reelcache.types import MatrixKey
        from tests.test_packed_nvdec import SoftwarePool
        runtime = self.runtimes("restore")
        runtime.create_lineage("main")
        reg = s6.registry()
        for index in range(2):
            head = runtime.commit_node("main", f"chunk-{index:03d}", *s6.chunk(index), method=s6.method(),
                                       registry=reg, options=ArchiveOptions(encode_threads=1)).node_ref

        def restore(restorer, inference):
            caches = {}

            def destination(node_id, key, tile):
                key = key if isinstance(key, MatrixKey) else key[0]
                if key not in caches:
                    caches[key] = torch.zeros((1, 2 * s6.TOKENS, 1, 128), dtype=torch.float32)
                return rk.NativeView(caches[key], "BTHD", token_offset=tile.token_start)
            with torch.inference_mode(inference):
                restorer.restore(head, destination)
            return caches

        reference = restore(LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 30), False)
        for first in (True, False):
            restorer = LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 30)
            with torch.inference_mode(first):
                restorer.prewarm_lineages()
            for inference in (first, not first, first):
                caches = restore(restorer, inference)
                self.assertEqual(set(caches), set(reference))
                for key, value in reference.items():
                    self.assertTrue(torch.equal(caches[key], value), (first, inference, key))
            stats = restorer.stats()
            self.assertGreater(stats["bases"]["hits"] + stats["bases"]["prefix_hits"], 0)
            for entry in restorer.bases._entries.values():
                self.assertFalse(entry["tensor"].is_inference())


def _inside_inference_mode(case):
    """A subclass of ``case`` whose every test runs inside ``torch.inference_mode`` (SGLang's request mode)."""
    namespace = {"__doc__": f"{case.__name__}, every test inside torch.inference_mode."}
    for name in dir(case):
        if name.startswith("test_"):
            original = getattr(case, name)

            def run(self, original=original):
                with torch.inference_mode():
                    return original(self)
            namespace[name] = functools.wraps(original)(run)
    return type(f"{case.__name__}InferenceMode", (case,), namespace)


# The native CausVid adapters inside inference mode: owned history/staging tensors are normal tensors whatever the
# caller's mode (their version counters keep guarding them); borrowed inference tensors are treated as immutable and
# keep their content-digest audits (adapters.causvid._version_stamp).
CausVidAdapterInferenceModeTests = _inside_inference_mode(_causvid_adapter_tests.CausVidAdapterTests)
RawIdentityLifecycleInferenceModeTests = _inside_inference_mode(RawIdentityLifecycleTests)


class _ScaledClock:
    """Every duration measured with time.perf_counter differs from an unpatched run."""

    def __enter__(self):
        self.real = real = time.perf_counter
        time.perf_counter = lambda: 3.7 * real() + 12345.0
        return self

    def __exit__(self, *exc):
        time.perf_counter = self.real


class ContentAddressTests(unittest.TestCase):
    """Node, record and packet ids depend only on content; run telemetry lives in an unhashed sidecar."""

    def setUp(self):
        self.runtimes = s6.RuntimeFactory(self)

    def chain(self, runtime, mode):
        if mode == "raw":
            runtime.create_lineage("main")
            return [runtime.commit_raw_node("main", f"chunk-{i:03d}", *s6.chunk(i)).node_ref for i in range(2)]
        streams = ("cond", "uncond") if mode == "streams" else ()
        runtime.create_lineage("main", streams=streams)
        reg = s6.registry()
        lag = 1 if mode in ("lag1", "pipeline") else 0
        if mode == "pipeline":
            pipeline = ArchivePipeline(runtime, method=s6.method(lag), registry=reg,
                                       options=ArchiveOptions(encode_threads=1), encode_workers=2,
                                       max_source_chunks=2, max_unpublished=3)
            try:
                tickets = [pipeline.submit("main", f"chunk-{i:03d}", *s6.chunk(i)) for i in range(3)]
                pipeline.drain("main", timeout=300)
                return [ticket.result() for ticket in tickets]
            finally:
                pipeline.close(drain=False, timeout=60)
        options = (ArchiveOptions(packing="per_matrix", orthogonalizer="householder") if mode == "per_matrix"
                   else ArchiveOptions(measure_post_codec=True, encode_threads=2) if mode == "validated"
                   else ArchiveOptions(encode_threads=1))
        refs = [runtime.commit_node("main", f"chunk-{i:03d}", *s6.chunk(i, streams=streams), method=s6.method(lag),
                                    registry=reg, options=options).node_ref for i in range(3)]
        if mode == "recache":
            refs.append(runtime.commit_recache("main", "recache-1", [("chunk-001", *s6.chunk(1))],
                                               method=s6.method(), registry=reg, epoch_id="e1").node_ref)
            runtime.create_lineage("segments")
            refs.append(runtime.commit_segment("segments", "seg-0", *s6.chunk(0), method=s6.method(), registry=reg,
                                               epoch_id="s0").node_ref)
        return refs

    @staticmethod
    def objects(runtime, refs):
        seen, pending = set(), list(refs)
        while pending:
            ref = pending.pop()
            if ref not in seen:
                seen.add(ref)
                if ref.startswith("manifest:"):
                    pending.extend(runtime.store.references(ref))
        return seen

    def test_identical_commits_produce_identical_node_record_and_packet_ids(self):
        for mode in ("lag0", "lag1", "validated", "per_matrix", "raw", "streams", "recache", "pipeline"):
            with self.subTest(mode=mode):
                first = self.runtimes(f"{mode}-a")
                refs = self.chain(first, mode)
                second = self.runtimes(f"{mode}-b")
                with _ScaledClock():
                    again = self.chain(second, mode)
                self.assertEqual(again, refs)
                self.assertEqual(self.objects(second, again), self.objects(first, refs))
                # The timings moved to the sidecar; they are not lost and they did differ between the runs.
                report_a, report_b = first.node_commit_report(refs[-1]), second.node_commit_report(refs[-1])
                self.assertIn("seconds_before_publication", report_a)
                self.assertNotEqual(report_a["seconds_before_publication"], report_b["seconds_before_publication"])
                self.assertNotIn("seconds_before_publication", first.store.get_manifest(refs[-1])["commit"])
                node = first.load_node(refs[-1])
                if node.matrix_records:
                    record_ref = node.matrix_records[0][1]
                    packet_ref = first.load_record(record_ref).packet_ref
                    stored = first.load_packet(packet_ref)["metrics"]
                    self.assertFalse([name for name in stored if "seconds" in name])
                    telemetry = first.run_telemetry(packet_ref)
                    self.assertIsNotNone(telemetry)
                    self.assertTrue(any(name.endswith("_seconds") for name in telemetry["metrics"]))

    def test_pipeline_publishes_the_synchronous_content_addresses(self):
        sync = self.chain(self.runtimes("sync"), "lag1")
        self.assertEqual(self.chain(self.runtimes("pipe"), "pipeline"), sync)

    def test_manifests_with_inline_telemetry_still_read_and_extend(self):
        """Stores written before the split (timings hashed inline) read, restore and extend unchanged."""
        from reelcache import lineage, lineage_commit
        inline = (lambda value: (value, None))
        runtime = self.runtimes("legacy")
        runtime.create_lineage("main")
        reg = s6.registry()
        with mock.patch.object(lineage, "split_run_telemetry", inline), \
                mock.patch.object(lineage_commit, "split_run_telemetry", inline):
            legacy = [runtime.commit_node("main", f"chunk-{i:03d}", *s6.chunk(i), method=s6.method(), registry=reg,
                                          options=ArchiveOptions(encode_threads=1)).node_ref for i in range(2)]
        head = legacy[-1]
        self.assertIn("seconds_before_publication", runtime.store.get_manifest(head)["commit"])
        self.assertIsNone(runtime.run_telemetry(head))
        report = runtime.node_commit_report(head)
        self.assertIn("stage_seconds", report)
        record = runtime.load_record(runtime.load_node(head).matrix_records[0][1])
        self.assertIn("hevc_encode_seconds", runtime.load_packet(record.packet_ref)["metrics"])
        with runtime.acquire_node_history(head, s6.KEYS[0], dtype=torch.float32) as lease:
            self.assertEqual(tuple(lease.tensor.shape), (2 * s6.TOKENS, 128))
        child = runtime.commit_node("main", "chunk-002", *s6.chunk(2), method=s6.method(), registry=reg,
                                    options=ArchiveOptions(encode_threads=1))
        self.assertNotIn("seconds_before_publication", runtime.store.get_manifest(child.node_ref)["commit"])
        self.assertIn("seconds_before_publication", runtime.node_commit_report(child.node_ref))

    def test_telemetry_sidecar_is_collected_with_its_object_and_old_indexes_upgrade(self):
        import sqlite3
        runtime = self.runtimes("gc")
        refs = self.chain(runtime, "lag0")
        runtime.create_lineage("keep")
        kept = runtime.commit_node("keep", "chunk-000", *s6.chunk(0, scale=2.0), method=s6.method(),
                                   registry=s6.registry(), options=ArchiveOptions(encode_threads=1)).node_ref
        self.assertIsNotNone(runtime.store.telemetry(refs[-1]))
        runtime.delete_lineage("main")
        runtime.store.collect_garbage()
        self.assertIsNone(runtime.store.telemetry(refs[-1]))
        orphans = runtime.store._db.execute(
            "SELECT COUNT(*) FROM telemetry WHERE object_id NOT IN (SELECT id FROM objects)").fetchone()[0]
        self.assertEqual(orphans, 0)
        self.assertIsNotNone(runtime.run_telemetry(kept))
        root = runtime.store.root
        runtime.close()
        with sqlite3.connect(root / "index.sqlite3") as db:  # an index written before telemetry sidecars existed
            db.execute("DROP TABLE telemetry")
        reopened = KVRuntime(root, s6.MODEL)
        try:
            self.assertIsNone(reopened.run_telemetry(kept))
            self.assertIn("decisions", reopened.node_commit_report(kept))
            reopened.create_lineage("again")
            commit = reopened.commit_node("again", "chunk-000", *s6.chunk(0), method=s6.method(),
                                          registry=s6.registry(), options=ArchiveOptions(encode_threads=1))
            self.assertIsNotNone(reopened.run_telemetry(commit.node_ref))
        finally:
            reopened.close()


if os.environ.get("REELCACHE_STEP7_GPU_TESTS") == "1":
    class CUDAArchiveWarmupTests(unittest.TestCase):
        """CUDA fitter warm-up (inside GPU jobs only; collected only with REELCACHE_STEP7_GPU_TESTS=1)."""

        def setUp(self):
            if not torch.cuda.is_available():
                raise AssertionError("REELCACHE_STEP7_GPU_TESTS=1 requires a CUDA device")
            self.runtimes = s6.RuntimeFactory(self)

        def chain(self, prewarm, *, fitter=False):
            from reelcache.gpu_fitting import GPUFitter
            runtime = self.runtimes("cuda-warm" if prewarm else "cuda-cold")
            runtime.create_lineage("main")
            reg = s6.registry()
            pipeline = ArchivePipeline(runtime, method=s6.method(1), registry=reg,
                                       options=ArchiveOptions(encode_threads=1), device="cuda",
                                       fitter=GPUFitter(gpu_id=0, gpu_uuid="GPU-" + str(
                                           torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-"),
                                           method="gram_batched", eigh_solver="auto")
                                       if fitter else None,
                                       encode_workers=2, max_source_chunks=2, max_unpublished=3)
            try:
                report = None
                if prewarm:
                    rng = torch.cuda.get_rng_state()
                    report = pipeline.prewarm(s6.chunk(0)[1])
                    self.assertTrue(torch.equal(torch.cuda.get_rng_state(), rng))
                    self.assertEqual(report["device"].split(":")[0], "cuda")
                    self.assertEqual(len(report["plan_seconds"]), 2)
                    self.assertGreaterEqual(report["pinned_staging_buffers"], 1)
                tickets = []
                for index in range(3):
                    matrices, contexts = s6.chunk(index)
                    matrices = {key: value.cuda() for key, value in matrices.items()}
                    event = torch.cuda.Event()
                    event.record()
                    tickets.append(pipeline.submit("main", f"chunk-{index:03d}", matrices, contexts,
                                                   ready_event=event))
                pipeline.drain("main", timeout=300)
                return [ticket.result() for ticket in tickets], report
            finally:
                pipeline.close(drain=False, timeout=60)

        def test_cuda_prewarm_leaves_content_addresses_unchanged(self):
            cold, _ = self.chain(False)
            warm, report = self.chain(True)
            self.assertEqual(warm, cold)
            self.assertIsNotNone(report["eigh_solver"])

        def test_cuda_eigh_sub_batches_are_bitwise_identical_and_longcat_size_splits(self):
            from reelcache import cusolver_eigh
            from reelcache.gpu_fitting import BoundedEigh, select_eigh_solver, xsyevbatched_workspace_bytes
            label, eigh = select_eigh_solver("auto")
            # Halving (n < 4096 default: whole batch unless rejected): records the split; at n = 256 batch sizes
            # 2..64 share one cuSOLVER size class, so these sub-batches are bitwise equal to the whole batch.
            x = torch.randn(12, 768, 256, device="cuda", generator=torch.Generator(device="cuda").manual_seed(5))
            grams = (x.transpose(1, 2) @ x).contiguous()
            whole = eigh.solver(grams)
            forced = BoundedEigh(label, eigh.solver, workspace=lambda b, n, d: None if b > 3 else 0)
            parts = forced(grams)
            self.assertEqual(forced.last["sub_batch"], 3)
            self.assertTrue(torch.equal(whole[0], parts[0]) and torch.equal(whole[1], parts[1]))
            # A fixed sub-batch makes a matrix's eigenpairs independent of what it is solved with, also at n = 1536
            # where cuSOLVER's size classes are {1} {2} {3..8} {9} {10, 11} {12..30} ... (identity padding).
            x = torch.randn(20, 1600, 1536, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7))
            grams = (x.transpose(1, 2) @ x).contiguous()
            del x
            fixed = BoundedEigh(label, eigh.solver, sub_batch_size=lambda n: 8)
            reference = fixed(grams)
            self.assertEqual((fixed.last["solver_calls"], fixed.last["padded"]), (3, 4))
            for members in ((0, 1), (0, 5), (0, 13), (3, 4), (16, 20)):
                values, vectors = fixed(grams[members[0]:members[1]])
                self.assertTrue(torch.equal(values, reference[0][members[0]:members[1]]), members)
                self.assertTrue(torch.equal(vectors, reference[1][members[0]:members[1]]), members)
            del grams, reference, values, vectors
            if cusolver_eigh.available():
                # CUDA 12.x cuSOLVER rejects 60 x 4096^2 (status 3); CUDA 13 accepts it with a ~16 GB workspace.
                whole = xsyevbatched_workspace_bytes(60, 4096, "cuda")
                self.assertTrue(whole is None or whole > 0)
                self.assertIsInstance(xsyevbatched_workspace_bytes(32, 4096, "cuda"), int)
            big = torch.randn(60, 4096, device="cuda", generator=torch.Generator(device="cuda").manual_seed(6))
            grams = torch.diag_embed(big.square()) + 1e-3  # 60 x 4096^2 (LongCat's Gram batch shape)
            values, vectors = eigh(grams)
            # From n = 4096 every matrix is solved alone by default (as fast as batched calls; batch-independent).
            self.assertEqual((eigh.last["sub_batch"], eigh.last["solver_calls"], eigh.last["reason"]),
                             (1, 60, "per_matrix"))
            self.assertEqual(tuple(values.shape), (60, 4096))
            self.assertTrue(bool(torch.isfinite(values).all()))
            alone = eigh(grams[7:8].contiguous())
            self.assertTrue(torch.equal(alone[0][0], values[7]) and torch.equal(alone[1][0], vectors[7]))
            del grams, values, vectors, big, alone

        def test_cuda_fp64_measurements_do_not_depend_on_the_tf32_flag(self):
            from reelcache.deferred_reads import projection_routine, run_one
            from reelcache.rank_method import LatestMethodPlanner, require_fp32_matmul
            from reelcache.types import ActiveBasisBinding
            matrix = torch.randn(4680, 1536, device="cuda", generator=torch.Generator(device="cuda").manual_seed(8))
            matrix = (matrix[:, :96] @ torch.randn(96, 1536, device="cuda") + 0.01 * matrix).to(torch.bfloat16)
            basis = torch.linalg.qr(matrix.float().T).Q.T[:128].to(torch.float16)
            previous = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                ieee = run_one(projection_routine(matrix, basis, products="fp64"))
                torch.backends.cuda.matmul.allow_tf32 = True
                tf32 = run_one(projection_routine(matrix, basis, products="fp64"))
                self.assertTrue(torch.equal(ieee.coefficients, tf32.coefficients))
                self.assertEqual(ieee.relative_residual, tf32.relative_residual)
                require_fp32_matmul(torch.device("cuda", 0))  # flag-independent products: nothing to refuse
                default = s6.method()
                declared = RankMethodConfig(default.targets, default.codec, matmul_precision="fp32_verified_fp64")
                matrices, contexts = s6.chunk(0)
                matrices = {key: value.cuda() for key, value in matrices.items()}
                plan = LatestMethodPlanner(declared, s6.registry().pin(), device="cuda").plan(
                    matrices, contexts, {key: ActiveBasisBinding() for key in matrices},
                    {key: None for key in matrices}, chunk_id="c0",
                    source_ids={key: f"id-{key.layer}{key.target}" for key in matrices})
                self.assertNotIn("tf32_allowed_during_plan", plan.counters)  # content never records flag state
                for item in plan.plans.values():
                    self.assertGreaterEqual(item.decision_energy, item.target)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous

        def test_cuda_prewarm_on_the_fitter_thread_with_a_gpu_fitter(self):
            # A GPUFitter records its own eigensolver label in the commit report: compare fitter runs only.
            cold, _ = self.chain(False, fitter=True)
            warm, report = self.chain(True, fitter=True)
            self.assertEqual(warm, cold)
            self.assertIsNotNone(report["eigh_solver"])


if __name__ == "__main__":
    unittest.main()
