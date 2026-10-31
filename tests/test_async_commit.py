from concurrent.futures import CancelledError
from dataclasses import replace
import multiprocessing
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import torch

from reelcache.codecs import encode_tensor
from reelcache.errors import CapacityError, ConfigurationError, IntegrityError, PendingError, StateError, UnsupportedError
from reelcache.execution import BoundedExecutor
from reelcache.manager import KVRuntime
from reelcache.types import CodecSpec
from tests.test_parallel_seal import KEYS, Sources, contexts
from tests.test_grouped_history import MODEL
from tests.test_runtime_integration import snapshot_metadata


def next_contexts():
    return {key: replace(context, token_start=16) for key, context in contexts().items()}


def crash_worker(root, connection):
    def encoder(tensor, context, spec, **kwargs):
        if context.key.target == "V":
            connection.send("K staged")
            threading.Event().wait(120)
            raise RuntimeError("crash test was not terminated")
        return encode_tensor(tensor, context, spec, **kwargs)

    with BoundedExecutor() as executor, KVRuntime(root, MODEL, encoder=encoder) as runtime:
        runtime.submit_chunk("main", "c1", Sources(), next_contexts(), commit_executor=executor).result()


class AsyncCommitTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.coordinator = BoundedExecutor(max_workers=1, max_pending=2, max_staging_bytes=32768)
        self.runtime = KVRuntime(self.root, MODEL, max_pending_source_bytes=4096)
        self.runtime.create_branch("main")
        self.runtime.seal_chunk("main", "c0", Sources(), contexts())
        self.before = self.runtime.branch_info("main")

    def tearDown(self):
        self.coordinator.close()
        self.runtime.close()
        self.directory.cleanup()

    def submit(self, sources=None, **kwargs):
        return self.runtime.submit_chunk(
            "main", "c1", Sources() if sources is None else sources, next_contexts(),
            commit_executor=self.coordinator, **kwargs,
        )

    def blocked_coordinator(self):
        entered, release = threading.Event(), threading.Event()

        def hold():
            entered.set()
            if not release.wait(15):
                raise RuntimeError("test coordinator was not released")

        job = self.coordinator.submit(hold)
        self.assertTrue(entered.wait(5))
        return release, job

    def assert_released(self):
        self.assertEqual(self.runtime.async_stats()["pending_commits"], 0)
        self.assertEqual(self.runtime.async_stats()["pending_source_bytes"], 0)
        self.assertEqual(self.coordinator.stats()["staging_bytes"], 0)

    def test_submission_copies_inputs_and_reserves_branch_before_worker_start(self):
        release, blocker = self.blocked_coordinator()
        try:
            originals = {key: torch.full((16, 8), float(i + 2)) for i, key in enumerate(KEYS)}
            expected = {key: value.clone() for key, value in originals.items()}
            handle = self.submit(originals)
            self.assertEqual(handle.status()["phase"], "queued")
            self.assertEqual(self.runtime.async_stats()["pending_source_bytes"], 1024)
            for value in originals.values():
                value.add_(100)
            with self.assertRaises(PendingError):
                self.submit()
            with self.assertRaises(TimeoutError):
                handle.result(timeout=0.001)
            self.assertFalse(handle.status()["cancel_requested"])
        finally:
            release.set()
            blocker.result(timeout=10)
        self.assertTrue(handle.result(timeout=15).startswith("manifest:"))
        self.assertTrue(handle.done())
        self.assertTrue(handle.status()["published"])
        self.assertFalse(handle.cancel())
        self.assert_released()
        for key in KEYS:
            with self.runtime.acquire_history("main", key, token_start=16) as lease:
                self.assertTrue(torch.equal(lease.tensor, expected[key]))

    def test_pending_snapshot_restore_close_reject_but_other_reader_sees_old_head(self):
        snapshot = self.runtime.snapshot(
            "main", metadata=snapshot_metadata("c0", 16),
            tensors={"rng": torch.Generator().manual_seed(1).get_state(), "latent": torch.zeros(4)},
        )
        release, blocker = self.blocked_coordinator()
        try:
            handle = self.submit()
            for operation in (
                lambda: self.runtime.snapshot("main", metadata=snapshot_metadata("c0", 16)),
                lambda: self.runtime.restore(snapshot, "main"),
                self.runtime.close,
            ):
                with self.assertRaises(PendingError):
                    operation()
            self.runtime.fork(snapshot, "independent")
            with KVRuntime(self.root, MODEL) as reader:
                self.assertEqual(reader.branch_info("main"), self.before)
            self.assertTrue(handle.cancel())
            with self.assertRaises(CancelledError):
                handle.result(timeout=5)
            self.assertEqual(self.runtime.branch_info("main"), self.before)
        finally:
            release.set()
            blocker.result(timeout=10)
        self.assert_released()

    def test_running_cancellation_keeps_reservation_until_encoder_exits(self):
        entered, release = threading.Event(), threading.Event()
        encoded_keys = []

        def encoder(tensor, context, spec, **kwargs):
            encoded_keys.append(context.key)
            entered.set()
            if not release.wait(10):
                raise RuntimeError("encoder was not released")
            return encode_tensor(tensor, context, spec, **kwargs)

        self.runtime.encoder = encoder
        handle = self.submit()
        try:
            self.assertTrue(entered.wait(5))
            self.assertTrue(handle.cancel())
            self.assertFalse(handle.done())
            self.assertEqual(self.runtime.async_stats()["pending_source_bytes"], 1024)
            with self.assertRaises(PendingError):
                self.runtime.close()
        finally:
            release.set()
        with self.assertRaises(CancelledError):
            handle.result(timeout=15)
        self.assertEqual(handle.status()["phase"], "cancelled")
        self.assertEqual(encoded_keys, [KEYS[0]])
        self.assertEqual(self.runtime.branch_info("main"), self.before)
        self.assert_released()

    def test_failed_commit_is_observable_and_never_advances_head(self):
        def encoder(tensor, context, spec, **kwargs):
            if context.key.target == "V":
                raise ValueError("injected V failure")
            return encode_tensor(tensor, context, spec, **kwargs)

        self.runtime.encoder = encoder
        handle = self.submit()
        self.coordinator.close()
        with self.assertRaisesRegex(StateError, "unobserved clean-commit"):
            self.runtime.close()
        with self.assertRaisesRegex(StateError, "unobserved clean-commit"):
            self.submit()
        with self.assertRaisesRegex(ValueError, "injected V failure"):
            handle.result(timeout=5)
        self.assertEqual(handle.status()["phase"], "failed")
        self.assertEqual(self.runtime.branch_info("main"), self.before)
        self.assert_released()

    def test_source_budget_is_checked_before_lazy_reads(self):
        sources = Sources()
        self.runtime.max_pending_source_bytes = 1023
        with self.assertRaises(CapacityError):
            self.submit(sources)
        self.assertEqual(sources.calls, [])
        self.assertEqual(self.runtime.branch_info("main"), self.before)
        self.assert_released()

    def test_queue_rejection_releases_copies_and_branch_reservation(self):
        release, blocker = self.blocked_coordinator()
        queued = self.coordinator.submit(lambda: None)
        try:
            with self.assertRaises(CapacityError):
                self.submit()
            self.assertEqual(self.runtime.branch_info("main"), self.before)
            self.assertEqual(self.runtime.async_stats()["pending_source_bytes"], 0)
        finally:
            release.set()
            blocker.result(timeout=10)
            queued.result(timeout=10)
        self.assert_released()

    def test_external_branch_change_cannot_be_overwritten_by_queued_commit(self):
        release, blocker = self.blocked_coordinator()
        try:
            handle = self.submit()
            with KVRuntime(self.root, MODEL) as other:
                other.seal_chunk("main", "other-c1", Sources(), next_contexts())
                changed = other.branch_info("main")
        finally:
            release.set()
            blocker.result(timeout=10)
        with self.assertRaisesRegex(StateError, "branch changed"):
            handle.result(timeout=15)
        self.assertEqual(self.runtime.branch_info("main"), changed)
        self.assert_released()

    def test_separate_branch_can_progress_while_another_commit_is_queued(self):
        release, blocker = self.blocked_coordinator()
        try:
            handle = self.submit()
            self.runtime.create_branch("other")
            self.runtime.seal_chunk("other", "c0", Sources(), contexts())
            self.assertEqual(self.runtime.branch_info("other")["token_end"], 16)
        finally:
            release.set()
            blocker.result(timeout=10)
        handle.result(timeout=15)
        self.assert_released()

    def test_raw_only_clean_only_and_distinct_coordinator_contract(self):
        with self.assertRaises(UnsupportedError):
            self.submit(specs=CodecSpec(kind="svd", rank=3))
        with self.assertRaises(PendingError):
            self.submit(clean=False)
        self.runtime.executor = self.coordinator
        try:
            with self.assertRaises(ConfigurationError):
                self.submit()
        finally:
            self.runtime.executor = None
        for timeout in (-1, True, float("nan")):
            release, blocker = self.blocked_coordinator()
            try:
                handle = self.submit()
                with self.assertRaises(ConfigurationError):
                    handle.result(timeout)
                handle.cancel()
                with self.assertRaises(CancelledError):
                    handle.result(timeout=5)
            finally:
                release.set()
                blocker.result(timeout=10)
        self.assert_released()

    def test_async_coordinator_can_use_distinct_parallel_packet_workers(self):
        with BoundedExecutor(max_workers=2) as encoders:
            self.runtime.executor = encoders
            self.runtime.encode_concurrency = 2
            self.runtime.encode_workspace_bytes = 8192
            try:
                handle = self.submit()
                handle.result(timeout=15)
                self.assertEqual(encoders.stats()["succeeded"], 2)
                self.assertEqual(encoders.stats()["pending_jobs"], 0)
            finally:
                self.runtime.executor = None
                self.runtime.encode_concurrency = 1
        self.assert_released()

    def test_cleanup_error_is_explicit_about_already_published_state(self):
        release, blocker = self.blocked_coordinator()
        handle = self.submit()
        real_release = handle._state.pin.release
        with mock.patch.object(handle._state.pin, "release", side_effect=OSError("pin cleanup failed")):
            release.set()
            blocker.result(timeout=10)
            with self.assertRaisesRegex(OSError, "pin cleanup failed"):
                handle.result(timeout=15)
        real_release()
        self.assertTrue(handle.status()["published"])
        self.assertEqual(handle.status()["phase"], "published")
        self.assertEqual(self.runtime.branch_info("main")["token_end"], 32)
        self.assert_released()

    def test_aliased_clone_is_rejected_without_releasing_unowned_storage(self):
        original = {key: torch.ones(16, 8) for key in KEYS}
        with mock.patch.object(torch.Tensor, "clone", lambda value, **kwargs: value):
            with self.assertRaisesRegex(IntegrityError, "own its exact admitted CPU storage"):
                self.submit(original)
        self.assertEqual(self.runtime.branch_info("main"), self.before)
        self.assert_released()

    def test_process_crash_reopens_previous_head_and_recovers_staging(self):
        self.runtime.close()
        ctx = multiprocessing.get_context("spawn")
        receive, send = ctx.Pipe(duplex=False)
        process = ctx.Process(target=crash_worker, args=(str(self.root), send))
        process.start()
        send.close()
        try:
            self.assertTrue(receive.poll(180), "async child did not reach staged K")
            self.assertEqual(receive.recv(), "K staged")
        finally:
            process.terminate()
            process.join(15)
            receive.close()
        self.assertFalse(process.is_alive())
        with KVRuntime(self.root, MODEL) as reopened:
            self.assertEqual(reopened.branch_info("main"), self.before)
            reopened.collect_garbage()
            with reopened.acquire_history("main", KEYS[0]) as lease:
                self.assertEqual(lease.tensor.shape, (16, 8))


if __name__ == "__main__":
    unittest.main()
