"""Bounded decoded residency with explicit lifetimes and no tensor cache."""

from __future__ import annotations

import threading
from typing import Any, Callable
import weakref

import torch

from .errors import CapacityError, ConfigurationError, IntegrityError, StateError


class HistoryLease:
    def __init__(
        self, pool: ResidencyPool, tensor: torch.Tensor, nbytes: int, pin: Any = None
    ) -> None:
        self._pool: ResidencyPool | None = pool
        self._tensor: torch.Tensor | None = tensor
        self._pin = pin
        self.nbytes = nbytes

    @property
    def active(self) -> bool:
        return self._tensor is not None and self._pool is not None

    @property
    def tensor(self) -> torch.Tensor:
        if not self.active:
            raise StateError("history tensor is available only while its lease is active")
        assert self._tensor is not None
        return self._tensor

    def release(self) -> None:
        pool = self._pool
        if pool is None:
            return
        with pool._lock:
            if self._pool is None:
                return
            self._tensor = None
            self._pool = None
            pool._resident_bytes -= self.nbytes
            pool._reserved_bytes -= self.nbytes
            pin, self._pin = self._pin, None
        if pin is not None:
            pin.release()

    close = release

    def __enter__(self) -> HistoryLease:
        if not self.active:
            raise StateError("history lease has been released")
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


class ResidencyPool:
    """Reserves destination and caller-declared scratch before decoding.

    Leases own their tensors. The pool keeps weak references only. A caller must
    not retain a tensor (or a view of it) beyond the lease's active lifetime.
    Reservation peaks are not measurements of a device allocator or process RSS.
    """

    def __init__(self, max_bytes: int) -> None:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ConfigurationError("max resident bytes must be nonnegative")
        self.max_bytes = max_bytes
        self._lock = threading.RLock()
        self._leases: weakref.WeakSet[HistoryLease] = weakref.WeakSet()
        self._resident_bytes = 0
        self._reserved_bytes = 0
        self._peak_reserved_bytes = 0
        self._inflight = 0
        self._closed = False

    @property
    def resident_bytes(self) -> int:
        with self._lock:
            return self._resident_bytes

    @property
    def reserved_bytes(self) -> int:
        with self._lock:
            return self._reserved_bytes

    @property
    def peak_reserved_bytes(self) -> int:
        with self._lock:
            return self._peak_reserved_bytes

    @property
    def active_leases(self) -> int:
        with self._lock:
            return sum(lease.active for lease in self._leases)

    def acquire(
        self, nbytes: int, loader: Callable[[], torch.Tensor], *,
        scratch_bytes: int = 0, pin: Any = None,
    ) -> HistoryLease:
        admitted = False
        try:
            if type(nbytes) is not int or nbytes < 0:
                raise ConfigurationError("lease size must be a nonnegative integer")
            if type(scratch_bytes) is not int or scratch_bytes < 0:
                raise ConfigurationError("scratch size must be a nonnegative integer")
            if not callable(loader):
                raise ConfigurationError("lease loader must be callable")
            reservation = nbytes + scratch_bytes
            with self._lock:
                if self._closed:
                    raise StateError("residency pool is closed")
                if self._reserved_bytes + reservation > self.max_bytes:
                    raise CapacityError(
                        f"history admission needs {reservation} bytes with "
                        f"{self._reserved_bytes}/{self.max_bytes} already reserved"
                    )
                self._reserved_bytes += reservation
                self._peak_reserved_bytes = max(self._peak_reserved_bytes, self._reserved_bytes)
                self._inflight += 1
                admitted = True
            tensor = loader()
            if not isinstance(tensor, torch.Tensor):
                raise IntegrityError("history decoder did not return a tensor")
            actual = tensor.numel() * tensor.element_size()
            if (actual != nbytes or tensor.layout != torch.strided
                    or tensor.untyped_storage().nbytes() != nbytes):
                raise IntegrityError("history decoder returned unaccounted tensor storage")
            with self._lock:
                if self._closed:
                    raise StateError("residency pool closed during restoration")
                lease = HistoryLease(self, tensor, nbytes, pin)
                self._leases.add(lease)
                self._resident_bytes += nbytes
                self._reserved_bytes -= scratch_bytes
                self._inflight -= 1
                admitted = False
                pin = None
                return lease
        finally:
            if admitted:
                with self._lock:
                    self._reserved_bytes -= nbytes + scratch_bytes
                    self._inflight -= 1
            if pin is not None:
                pin.release()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "max_bytes": self.max_bytes, "resident_bytes": self._resident_bytes,
                "reserved_bytes": self._reserved_bytes,
                "peak_reserved_bytes": self._peak_reserved_bytes,
                "active_leases": self.active_leases, "inflight": self._inflight,
            }

    def close(self) -> None:
        with self._lock:
            self._closed = True
            leases = tuple(self._leases)
        for lease in leases:
            lease.release()

    def __enter__(self) -> ResidencyPool:
        with self._lock:
            if self._closed:
                raise StateError("residency pool is closed")
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
