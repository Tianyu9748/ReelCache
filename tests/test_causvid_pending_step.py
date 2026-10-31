from concurrent.futures import CancelledError
import threading
import time
import unittest
from unittest import mock

import torch

from reelcache.codecs import encode_tensor
from reelcache.errors import ConfigurationError, PendingError, StateError
from reelcache.execution import BoundedExecutor
from reelcache.types import CodecSpec
from tests import test_causvid_adapter as fixtures


class CausVidPendingStepTests(unittest.TestCase):
    setUp = fixtures.CausVidAdapterTests.setUp
    adapter = fixtures.CausVidAdapterTests.adapter

    def coordinator(self):
        value = BoundedExecutor(max_staging_bytes=1 << 20)
        self.addCleanup(value.close)
        return value

    def test_raw_svd_hevc_deferred_matches_same_profile_and_reuse(self):
        coordinator = self.coordinator()
        for kind in ("raw", "svd", "svd_hevc"):
            with self.subTest(kind=kind):
                adapter = fixtures.CausVidAdapter(
                    self.pipe, self.runtime, kind, config=fixtures.CONFIG, commit_executor=coordinator,
                    specs=CodecSpec(kind=kind, rank=None if kind == "raw" else 3),
                    reuse_threshold=None if kind == "raw" else 0.9,
                )
                with adapter:
                    adapter.start(self.noise, self.conditioning, self.rng, conditioning_id="fixed-prompt")
                    adapter.step(clean_prefix=self.prefix)
                    snapshot = adapter.snapshot()
                    adapter.step()
                    adapter.step()
                    expected, rng, rgb = adapter.latent_state, adapter.rng_state, adapter.decode()
                    adapter.restore(snapshot, kind + "-deferred", conditioning_id="fixed-prompt")
                    while not adapter.finished:
                        cursor = adapter.frame_cursor
                        handle = adapter.step(defer_publication=True)
                        self.assertEqual(adapter.frame_cursor, cursor)
                        self.assertEqual(adapter._phase, "publication_pending")
                        self.assertEqual(adapter.memory_report()["live_mutable_kv_bytes"], 0)
                        self.assertGreater(adapter.memory_report()["pending_native_kv_bytes"], 0)
                        for operation in (adapter.snapshot, adapter.close):
                            with self.assertRaises(PendingError):
                                operation()
                        self.assertEqual(adapter.finish_step(timeout=30), handle.result())
                    self.assertTrue(torch.equal(expected, adapter.latent_state))
                    self.assertTrue(torch.equal(rng, adapter.rng_state))
                    self.assertTrue(torch.equal(rgb, adapter.decode()))
                    self.assertEqual(self.runtime.async_stats()["pending_source_bytes"], 0)

    def test_overlapped_steps_have_no_publication_barrier(self):
        """The next chunk denoises while the previous chunk is still publishing (cross-chunk overlap).

        Its history is the published archive plus the deferred chunk's own native KV, so RAW is
        bitwise identical to synchronous sealing; lossy profiles are mixed-tier (newest chunk native).
        """
        coordinator = self.coordinator()
        for kind in ("raw", "svd_hevc"):
            with self.subTest(kind=kind):
                adapter = fixtures.CausVidAdapter(
                    self.pipe, self.runtime, kind + "-overlap", config=fixtures.CONFIG, commit_executor=coordinator,
                    specs=CodecSpec(kind=kind, rank=None if kind == "raw" else 3),
                )
                with adapter:
                    adapter.start(self.noise, self.conditioning, self.rng, conditioning_id="fixed-prompt")
                    adapter.step(clean_prefix=self.prefix)
                    snapshot = adapter.snapshot()
                    adapter.step()
                    adapter.step()
                    expected, rng = adapter.latent_state, adapter.rng_state
                    adapter.restore(snapshot, kind + "-overlap-run", conditioning_id="fixed-prompt")
                    entered, release = threading.Event(), threading.Event()
                    original = self.runtime._seal_chunk

                    def slow_seal(*args, **kwargs):
                        entered.set()
                        if not release.wait(30):
                            raise RuntimeError("test gate not released")
                        return original(*args, **kwargs)

                    with mock.patch.object(self.runtime, "_seal_chunk", side_effect=slow_seal):
                        first = adapter.step(defer_publication=True)
                        self.assertTrue(entered.wait(10))
                        self.assertFalse(first.done())
                        # No barrier: the next chunk's denoising forwards run while the first publishes.
                        forwards_before = len(self.pipe.generator.model.blocks[0].self_attn.seen_positions)
                        worker = threading.Thread(target=lambda: setattr(self, "_overlap_handle",
                                                                         adapter.step(defer_publication=True)))
                        worker.start()
                        deadline = time.monotonic() + 10
                        while (len(self.pipe.generator.model.blocks[0].self_attn.seen_positions)
                               < forwards_before + 3 and time.monotonic() < deadline):
                            time.sleep(0.01)
                        self.assertFalse(first.done())
                        self.assertGreaterEqual(
                            len(self.pipe.generator.model.blocks[0].self_attn.seen_positions), forwards_before + 3)
                        release.set()
                        worker.join(30)
                        self.assertFalse(worker.is_alive())
                    self.assertTrue(first.done())
                    adapter.finish_step(timeout=30)
                    self.assertTrue(adapter.finished)
                    if kind == "raw":
                        self.assertTrue(torch.equal(expected, adapter.latent_state))
                        self.assertTrue(torch.equal(rng, adapter.rng_state))
                    else:
                        self.assertEqual(adapter.latent_state.shape, expected.shape)
                    self.assertEqual(self.runtime.branch_info(kind + "-overlap-run")["token_end"],
                                     adapter.token_cursor)

    def test_timeout_and_queued_cancellation_restore_rng_and_retry(self):
        coordinator = self.coordinator()
        adapter = self.adapter(commit_executor=coordinator)
        adapter.step(clean_prefix=self.prefix)
        snapshot = adapter.snapshot()
        before_rng, before_latent = adapter.rng_state, adapter.latent_state
        entered, release = threading.Event(), threading.Event()
        def hold():
            entered.set()
            if not release.wait(15):
                raise RuntimeError("test blocker not released")
        blocker = coordinator.submit(hold)
        self.assertTrue(entered.wait(5))
        try:
            handle = adapter.step(defer_publication=True)
            with self.assertRaises(TimeoutError):
                adapter.finish_step(timeout=0.001)
            self.assertEqual(adapter._phase, "publication_pending")
            handle.cancel()
            with self.assertRaises(CancelledError):
                adapter.finish_step(timeout=5)
            self.assertEqual(adapter._phase, "clean")
            self.assertTrue(torch.equal(before_rng, adapter.rng_state))
            self.assertTrue(torch.equal(before_latent, adapter.latent_state))
        finally:
            release.set()
            blocker.result(timeout=10)
        adapter.step()
        expected, rng = adapter.latent_state, adapter.rng_state
        adapter.restore(snapshot, "cancel-replay", conditioning_id="fixed-prompt")
        adapter.step(defer_publication=True)
        adapter.finish_step(timeout=30)
        self.assertTrue(torch.equal(expected, adapter.latent_state))
        self.assertTrue(torch.equal(rng, adapter.rng_state))

    def test_encoder_failure_rolls_back_and_finish_without_pending_rejects(self):
        adapter = self.adapter(commit_executor=self.coordinator())
        adapter.step(clean_prefix=self.prefix)
        rng = adapter.rng_state
        with mock.patch.object(self.runtime, "encoder", side_effect=ValueError("failed encode")):
            adapter.step(defer_publication=True)
            with self.assertRaisesRegex(ValueError, "failed encode"):
                adapter.finish_step(timeout=30)
        self.assertEqual(adapter._phase, "clean")
        self.assertTrue(torch.equal(rng, adapter.rng_state))
        self.assertEqual(self.runtime.async_stats()["unobserved_failures"], 0)
        with self.assertRaises(StateError):
            adapter.finish_step()
        adapter.step()

    def test_missing_coordinator_rejected_before_native_forward(self):
        adapter = self.adapter()
        with mock.patch.object(adapter, "_generator_forward", side_effect=AssertionError("native call")):
            with self.assertRaises(ConfigurationError):
                adapter.step(defer_publication=True)


if __name__ == "__main__":
    unittest.main()
