import gc
import threading
import unittest
import weakref
from concurrent.futures import CancelledError

from reelcache.errors import CapacityError, ConfigurationError, StateError
from reelcache.execution import BoundedExecutor


class ExecutionTests(unittest.TestCase):
    def test_result_and_metrics(self):
        with BoundedExecutor() as executor:
            job = executor.submit(lambda a, b: a + b, 2, 3, staging_bytes=7)
            self.assertEqual(job.result(timeout=5), 5)
        stats = executor.stats()
        self.assertEqual(stats["succeeded"], 1)
        self.assertEqual(stats["staging_bytes"], 0)
        self.assertEqual(stats["pending_jobs"], 0)
        self.assertGreaterEqual(job.metrics()["wall_seconds"], 0)

    def test_pending_capacity_and_wait_timeout(self):
        release = threading.Event()
        executor = BoundedExecutor(max_pending=1)
        try:
            job = executor.submit(release.wait, 5)
            with self.assertRaises(CapacityError):
                executor.submit(lambda: None)
            with self.assertRaises(CapacityError):
                executor.submit(lambda: None, block=True, timeout=0.01)
            with self.assertRaises(TimeoutError):
                job.result(timeout=0.01)
            self.assertFalse(job.done())
            release.set()
            self.assertTrue(job.result(timeout=5))
        finally:
            release.set()
            executor.close()

    def test_staging_budget(self):
        release = threading.Event()
        executor = BoundedExecutor(max_pending=3, max_staging_bytes=10)
        try:
            with self.assertRaises(CapacityError):
                executor.submit(lambda: None, staging_bytes=11)
            running = executor.submit(release.wait, 5, staging_bytes=8)
            with self.assertRaises(CapacityError):
                executor.submit(lambda: None, staging_bytes=3)
            release.set()
            running.result(timeout=5)
        finally:
            release.set()
            executor.close()
        self.assertEqual(executor.stats()["peak_staging_bytes"], 8)

    def test_queued_cancellation_and_running_non_cancellation(self):
        entered, release = threading.Event(), threading.Event()

        def hold():
            entered.set()
            return release.wait(5)

        executor = BoundedExecutor(max_workers=1, max_pending=2)
        try:
            running = executor.submit(hold, staging_bytes=3)
            self.assertTrue(entered.wait(5))
            queued = executor.submit(lambda: 4, staging_bytes=2)
            self.assertFalse(running.cancel())
            self.assertTrue(queued.cancel())
            with self.assertRaises(CancelledError):
                queued.result(timeout=5)
            release.set()
            running.result(timeout=5)
        finally:
            release.set()
            executor.close()
        self.assertEqual(executor.stats()["cancelled"], 1)
        self.assertEqual(executor.stats()["staging_bytes"], 0)

    def test_failure_is_rethrown_and_observed(self):
        def fail():
            raise ValueError("codec failed")

        with BoundedExecutor() as executor:
            job = executor.submit(fail)
            with self.assertRaisesRegex(ValueError, "codec failed"):
                job.result(timeout=5)
            executor.check_errors()
        self.assertEqual(executor.stats()["failed"], 1)

    def test_cancelled_queue_entry_drops_its_input_reference(self):
        class Buffer:
            pass

        entered, release = threading.Event(), threading.Event()

        def hold():
            entered.set()
            return release.wait(5)

        executor = BoundedExecutor(max_workers=1, max_pending=2)
        try:
            running = executor.submit(hold)
            self.assertTrue(entered.wait(5))
            buffer = Buffer()
            reference = weakref.ref(buffer)
            queued = executor.submit(lambda value: None, buffer, staging_bytes=100)
            del buffer
            self.assertIsNotNone(reference())
            self.assertTrue(queued.cancel())
            gc.collect()
            self.assertIsNone(reference())
            self.assertEqual(executor.stats()["staging_bytes"], 0)
            release.set()
            running.result(timeout=5)
        finally:
            release.set()
            executor.close()

    def test_unobserved_failure_prevents_silent_shutdown(self):
        def fail():
            raise ValueError("unobserved")

        executor = BoundedExecutor()
        job = executor.submit(fail)
        with self.assertRaisesRegex(StateError, "Unobserved background failure"):
            executor.close()
        with self.assertRaises(ValueError):
            job.result()
        executor.check_errors()
        executor.close()

    def test_observed_failure_drops_completed_worker_frame_inputs(self):
        class Buffer:
            pass

        def fail(value):
            raise ValueError("worker failed")

        with BoundedExecutor() as executor:
            value = Buffer()
            reference = weakref.ref(value)
            job = executor.submit(fail, value, staging_bytes=64)
            del value
            try:
                job.result(timeout=5)
            except ValueError as exc:
                observed = exc
            self.assertEqual(str(observed), "worker failed")
            gc.collect()
            self.assertIsNone(reference())
            self.assertEqual(executor.stats()["staging_bytes"], 0)

    def test_blocked_submit_released_by_shutdown(self):
        entered, release = threading.Event(), threading.Event()
        executor = BoundedExecutor(max_pending=1)
        job = executor.submit(release.wait, 5)
        errors = []

        def submitter():
            entered.set()
            try:
                executor.submit(lambda: None, block=True, timeout=5)
            except StateError as error:
                errors.append(str(error))

        thread = threading.Thread(target=submitter)
        thread.start()
        self.assertTrue(entered.wait(5))
        executor.close(wait_for_running=False)
        thread.join(5)
        release.set()
        job.result(timeout=5)
        executor.close()
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ["executor is closed"])

    def test_parallel_jobs_overlap_without_exceeding_limits(self):
        barrier = threading.Barrier(2)

        def rendezvous():
            barrier.wait(timeout=5)
            return 1

        with BoundedExecutor(max_workers=2, max_pending=2) as executor:
            first = executor.submit(rendezvous, staging_bytes=2)
            second = executor.submit(rendezvous, staging_bytes=3)
            self.assertEqual(first.result(timeout=5) + second.result(timeout=5), 2)
        self.assertEqual(executor.stats()["peak_pending_jobs"], 2)

    def test_invalid_configuration(self):
        for kwargs in ({"max_workers": 0}, {"max_pending": True},
                       {"max_staging_bytes": -1}, {"history_size": 0}):
            with self.assertRaises(ConfigurationError):
                BoundedExecutor(**kwargs)
        with BoundedExecutor() as executor:
            for kwargs in ({"staging_bytes": -1}, {"staging_bytes": True},
                           {"timeout": 1}, {"block": True, "timeout": float("nan")}):
                with self.assertRaises(ConfigurationError):
                    executor.submit(lambda: None, **kwargs)


if __name__ == "__main__":
    unittest.main()
