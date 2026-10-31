"""Step 6: archive_native CausVid adapter (fake native pipeline; CPU contracts, not native evidence)."""

from __future__ import annotations

import shutil
import threading
import time
import unittest
import uuid
from pathlib import Path

import torch

from reelcache import restore_kernels as rk
from reelcache.adapters.causvid import CausVidArchiveNativeAdapter
from reelcache.archive_pipeline import ArchiveFailure, ArchivePipeline
from reelcache.archive_restore import SoftwareLineageRestorer
from reelcache.errors import CapacityError, ConfigurationError, IntegrityError, StateError
from reelcache.lineage_commit import ArchiveOptions
from reelcache.manager import KVRuntime
from reelcache.predictor_registry import RankPredictorRegistry
from reelcache.rank_method import RankMethodConfig
from reelcache.types import CodecSpec, EnergyTargets, ModelIdentity
from tests.test_causvid_adapter import CONFIG, FakePipeline, causal_rope_apply, initial_state

MODEL = ModelIdentity("causvid-cpu-fake", "fake-source-not-native-proof", 2, 8, "tokens_channels", "float32")
METHOD = RankMethodConfig(EnergyTargets("t", 0.98, 0.95, True),
                          CodecSpec(kind="svd_hevc", rank=1, inter_prediction=False))


def fake_rope_table(total_frames):
    """Dense per-token table equal to the fake model's rotation (angle = frame * freqs)."""
    freqs = torch.tensor([0.3, 0.9], dtype=torch.float64)
    frames = torch.arange(total_frames).repeat_interleave(CONFIG.frame_tokens).double()
    angles = frames[:, None] * freqs[None]
    return rk.RopeTable(torch.cos(angles), torch.sin(angles), None)


class ArchiveNativeAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / ".test-work" / f"archive-native-{uuid.uuid4().hex}"
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def build(self, *, archival=True, hooks=None, slots=2, slot_timeout=None, name="store"):
        torch.manual_seed(31)
        pipe = FakePipeline()
        runtime = KVRuntime(self.root / name, MODEL)
        self.addCleanup(runtime.close)
        archive = None
        if archival:
            archive = ArchivePipeline(runtime, method=METHOD, registry=RankPredictorRegistry("r"),
                                      options=ArchiveOptions(), hooks=hooks, max_source_chunks=2, max_unpublished=3)
            self.addCleanup(lambda: archive.close(drain=False, timeout=60))
        adapter = CausVidArchiveNativeAdapter(pipe, runtime, "main", config=CONFIG, archive=archive,
                                              restorer=SoftwareLineageRestorer(runtime), capture_slots=slots,
                                              slot_timeout=slot_timeout, verify_pre_rope=True,
                                              rope=causal_rope_apply)
        adapter.install()
        self.addCleanup(lambda: adapter.close(drain=False) if adapter._installed else None)
        noise, conditioning, rng = initial_state()
        adapter.start(noise, conditioning, rng, conditioning_id="fixed-prompt")
        return pipe, runtime, archive, adapter

    def test_native_active_generation_is_bitwise_identical_with_archival_on_and_off(self):
        _, _, _, reference = self.build(archival=False, name="off")
        for _ in range(3):
            self.assertIsNone(reference.step())
        expected = reference.latents
        _, runtime, archive, adapter = self.build(name="on")
        tickets = [adapter.step() for _ in range(3)]
        self.assertTrue(torch.equal(adapter.latents, expected))
        snapshot = adapter.snapshot()
        self.assertEqual(snapshot["node_ref"], tickets[-1].result())
        self.assertEqual(runtime.load_node(snapshot["node_ref"]).frontier, 3 * CONFIG.chunk_tokens)
        with self.assertRaises(StateError):
            reference.drain()

    def test_steps_do_not_wait_for_publication(self):
        entered, release = threading.Event(), threading.Event()

        def hooks(stage, node_id):
            if stage == "encode" and node_id == "chunk-000000":
                entered.set()
                if not release.wait(30):
                    raise RuntimeError("gate not released")

        _, runtime, archive, adapter = self.build(hooks=hooks)
        first = adapter.step()
        self.assertTrue(entered.wait(30))
        second = adapter.step()  # generated frontier advances while chunk 0 is still encoding
        self.assertFalse(first.done())
        self.assertFalse(second.done())
        frontiers = archive.frontiers("main")
        self.assertEqual(frontiers["generated_frontier"], 2 * CONFIG.chunk_tokens)
        self.assertEqual(frontiers["published_frontier"], 0)
        self.assertEqual(adapter.frame_cursor, 2 * CONFIG.frames_per_chunk)
        release.set()
        head = adapter.drain(timeout=60)
        self.assertEqual(head, second.result())

    def test_capture_slot_exhaustion_is_explicit_backpressure(self):
        entered, release = threading.Event(), threading.Event()

        def hooks(stage, node_id):
            if stage == "plan" and node_id == "chunk-000000":
                entered.set()
                if not release.wait(30):
                    raise RuntimeError("gate not released")

        _, _, archive, adapter = self.build(hooks=hooks, slots=1, slot_timeout=0.05)
        adapter.step()
        self.assertTrue(entered.wait(30))
        cursor, latents = adapter.frame_cursor, adapter.latents
        with self.assertRaises(CapacityError):
            adapter.step()
        self.assertEqual(adapter.frame_cursor, cursor)  # nothing ran; the step can be retried
        self.assertTrue(torch.equal(adapter.latents, latents))
        release.set()
        adapter.slot_timeout = 30
        adapter.step()
        adapter.drain(timeout=60)
        self.assertGreaterEqual(adapter.stats_["slot_waits"], 1)

    def test_restore_forks_a_published_node_and_continues_natively(self):
        _, runtime, archive, adapter = self.build()
        for _ in range(3):
            adapter.step()
        snapshot = adapter.snapshot()
        native_k = [tensor.clone() for tensor in adapter._k]
        native_v = [tensor.clone() for tensor in adapter._v]
        latents = adapter.latents
        node1 = runtime.node_path(snapshot["node_ref"])[2][0]
        result = adapter.restore(node1, "fork", rope_table=fake_rope_table(9))
        tokens = 2 * CONFIG.chunk_tokens
        self.assertEqual(result["frame_cursor"], 2 * CONFIG.frames_per_chunk)
        self.assertEqual(result["entries"], 2 * 4)
        self.assertTrue(torch.equal(adapter.latents, latents[:, :6]))
        for layer in range(2):
            for restored, original in ((adapter._k[layer], native_k[layer]), (adapter._v[layer], native_v[layer])):
                error = (restored[:, :tokens] - original[:, :tokens]).norm() / original[:, :tokens].norm()
                self.assertLess(float(error), 0.25)
        ticket = adapter.step()
        self.assertEqual(adapter.drain(timeout=60), ticket.result())
        self.assertEqual(runtime.load_node(ticket.result()).parent_ref, node1)
        self.assertEqual(runtime.lineage_head("main"), snapshot["node_ref"])  # the source lineage is untouched
        with self.assertRaises(StateError):
            adapter.restore(snapshot["node_ref"], "fork")

    def test_archive_failure_surfaces_at_the_next_step(self):
        def hooks(stage, node_id):
            if stage == "encode" and node_id == "chunk-000001":
                raise RuntimeError("injected archive failure")

        _, runtime, archive, adapter = self.build(hooks=hooks)
        adapter.step()
        failing = adapter.step()
        with self.assertRaises(ArchiveFailure):
            failing.result(timeout=60)
        with self.assertRaises(ArchiveFailure):
            adapter.step()
        self.assertEqual(adapter._phase, "poisoned")
        head = archive.reset("main", timeout=60)
        self.assertEqual(runtime.load_node(head).frontier, CONFIG.chunk_tokens)

    def test_configuration_errors(self):
        pipe = FakePipeline()
        runtime = KVRuntime(self.root / "config", MODEL)
        self.addCleanup(runtime.close)
        archive = ArchivePipeline(runtime, method=METHOD, registry=RankPredictorRegistry("r"), max_source_chunks=1,
                                  max_unpublished=1)
        self.addCleanup(lambda: archive.close(drain=False, timeout=60))
        with self.assertRaises(ConfigurationError):
            CausVidArchiveNativeAdapter(pipe, runtime, "main", config=CONFIG, archive=archive, capture_slots=2)
        with self.assertRaises(ConfigurationError):
            CausVidArchiveNativeAdapter(pipe, runtime, "bad/branch", config=CONFIG)
        adapter = CausVidArchiveNativeAdapter(pipe, runtime, "main", config=CONFIG)
        with self.assertRaises(StateError):
            adapter.step()


if __name__ == "__main__":
    unittest.main()
