import gc
import threading
import unittest
from unittest.mock import Mock
import weakref

import torch

from reelcache.errors import CapacityError, ConfigurationError, IntegrityError, StateError
from reelcache.residency import ResidencyPool


class ResidencyTests(unittest.TestCase):
    def test_release_actually_drops_decoded_tensor(self):
        pool = ResidencyPool(1024)
        lease = pool.acquire(64, lambda: torch.zeros(16))
        reference = weakref.ref(lease.tensor)
        self.assertEqual(pool.resident_bytes, 64)
        self.assertEqual(pool.active_leases, 1)
        lease.release()
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(pool.resident_bytes, 0)
        self.assertEqual(pool.reserved_bytes, 0)
        with self.assertRaises(StateError):
            _ = lease.tensor
        lease.release()

    def test_context_exit_and_unreferenced_lease_release_pins(self):
        pool = ResidencyPool(100)
        pin = Mock()
        with pool.acquire(40, lambda: torch.zeros(10), pin=pin) as lease:
            self.assertEqual(lease.tensor.shape, (10,))
        pin.release.assert_called_once()
        second = pool.acquire(40, lambda: torch.zeros(10))
        reference = weakref.ref(second.tensor)
        del second
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(pool.resident_bytes, 0)

    def test_capacity_admission_precedes_decode_and_accounts_scratch(self):
        pool = ResidencyPool(100)
        loader = Mock(return_value=torch.zeros(20))
        pin = Mock()
        with self.assertRaises(CapacityError):
            pool.acquire(80, loader, scratch_bytes=21, pin=pin)
        loader.assert_not_called()
        pin.release.assert_called_once()
        with pool.acquire(80, loader, scratch_bytes=20):
            self.assertEqual(pool.resident_bytes, 80)
            self.assertEqual(pool.peak_reserved_bytes, 100)
            with self.assertRaises(CapacityError):
                pool.acquire(24, lambda: torch.zeros(6))
        self.assertEqual(pool.reserved_bytes, 0)

    def test_decode_failure_refunds_all_reservations(self):
        pool = ResidencyPool(100)
        pin = Mock()
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            pool.acquire(20, Mock(side_effect=RuntimeError("decode failed")),
                         scratch_bytes=80, pin=pin)
        self.assertEqual(pool.reserved_bytes, 0)
        self.assertEqual(pool.stats()["inflight"], 0)
        pin.release.assert_called_once()

    def test_rejects_hidden_larger_storage_and_wrong_tensor_size(self):
        pool = ResidencyPool(100)
        for loader in (lambda: torch.zeros(20)[:5], lambda: torch.zeros(10), lambda: None):
            with self.subTest(loader=loader), self.assertRaises(IntegrityError):
                pool.acquire(20, loader)
        self.assertEqual(pool.reserved_bytes, 0)

    def test_close_invalidates_active_leases(self):
        pool = ResidencyPool(100)
        lease = pool.acquire(20, lambda: torch.zeros(5))
        reference = weakref.ref(lease.tensor)
        pool.close()
        self.assertFalse(lease.active)
        self.assertIsNone(reference())
        with self.assertRaises(StateError):
            pool.acquire(0, lambda: torch.empty(0))
        self.assertEqual(pool.active_leases, 0)

    def test_inflight_admission_is_bounded(self):
        pool = ResidencyPool(100)
        started, finish = threading.Event(), threading.Event()
        errors, leases = [], []

        def loader():
            started.set()
            if not finish.wait(5):
                raise RuntimeError("test synchronization timeout")
            return torch.zeros(10)

        def restore():
            try:
                leases.append(pool.acquire(40, loader, scratch_bytes=60))
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=restore)
        thread.start()
        self.assertTrue(started.wait(5))
        with self.assertRaises(CapacityError):
            pool.acquire(4, lambda: torch.zeros(1))
        finish.set()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        leases[0].release()
        self.assertEqual(pool.reserved_bytes, 0)

    def test_close_during_inflight_drops_output(self):
        pool = ResidencyPool(100)
        started, finish = threading.Event(), threading.Event()
        errors = []
        references = []

        def loader():
            started.set()
            finish.wait(5)
            tensor = torch.zeros(10)
            references.append(weakref.ref(tensor))
            return tensor

        def restore():
            try:
                pool.acquire(40, loader)
            except StateError:
                errors.append("closed")

        thread = threading.Thread(target=restore)
        thread.start()
        self.assertTrue(started.wait(5))
        pool.close()
        finish.set()
        thread.join(5)
        gc.collect()
        self.assertEqual(errors, ["closed"])
        self.assertIsNone(references[0]())
        self.assertEqual(pool.reserved_bytes, 0)

    def test_invalid_capacities_and_zero_byte_tensor(self):
        with self.assertRaises(ConfigurationError):
            ResidencyPool(-1)
        with ResidencyPool(0) as pool:
            with pool.acquire(0, lambda: torch.empty(0)) as lease:
                self.assertEqual(lease.tensor.numel(), 0)


if __name__ == "__main__":
    unittest.main()
