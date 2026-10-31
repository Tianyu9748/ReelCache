"""Bounded CPU work with explicit admission, cancellation and failure semantics."""

from __future__ import annotations

import math
import threading
import time
import traceback
from collections import deque
from contextlib import nullcontext
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

from .errors import CapacityError, ConfigurationError, StateError
from .telemetry import RuntimeTelemetry, current_telemetry

T = TypeVar("T")


def _timeout(value: float | None) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value < 0
    ):
        raise ConfigurationError("timeout must be finite and nonnegative")


@dataclass
class _JobState:
    job_id: int
    label: str
    staging_bytes: int
    submitted: float
    started: float | None = None
    finished: float | None = None
    status: str = "queued"
    error: str | None = None
    observed: bool = False
    function: Callable | None = field(default=None, repr=False)
    args: tuple = field(default=(), repr=False)
    kwargs: dict = field(default_factory=dict, repr=False)
    telemetry: RuntimeTelemetry | None = field(default=None, repr=False)

    def summary(self) -> dict:
        return {
            "job_id": self.job_id,
            "label": self.label,
            "staging_bytes": self.staging_bytes,
            "status": self.status,
            "queue_seconds": (
                None if self.started is None else self.started - self.submitted
            ),
            "execution_seconds": (
                None if self.started is None or self.finished is None
                else self.finished - self.started
            ),
            "wall_seconds": (
                None if self.finished is None else self.finished - self.submitted
            ),
            "error": self.error,
        }


class Job(Generic[T]):
    """A timed-out result wait does not cancel an executing job."""

    def __init__(
        self, owner: BoundedExecutor, state: _JobState, future: Future[T]
    ) -> None:
        self._owner = owner
        self._state = state
        self._future = future
        ready = threading.Event()
        self._ready = ready
        future.add_done_callback(lambda completed: ready.set())

    @property
    def job_id(self) -> int:
        return self._state.job_id

    def done(self) -> bool:
        return self._future.done()

    def cancel(self) -> bool:
        return self._future.cancel()

    def add_done_callback(self, callback: Callable[[Job[T]], None]) -> None:
        if not callable(callback):
            raise ConfigurationError("job callback must be callable")
        self._future.add_done_callback(lambda completed: callback(self))

    def result(self, timeout: float | None = None) -> T:
        _timeout(timeout)
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError(
                f"Job {self.job_id} is not ready; a timed-out wait does not cancel work"
            )
        try:
            return self._future.result()
        finally:
            self._owner._observe(self._state)

    def metrics(self) -> dict:
        with self._owner._condition:
            return self._state.summary()


class BoundedExecutor:
    """Bound running plus queued jobs and their declared retained input bytes.

    Callers must include all retained staging buffers in ``staging_bytes``.
    Completed results or exception tracebacks retained by a caller are not input
    staging. CUDA staging and collective order belong to the adapter.
    """

    def __init__(
        self,
        *,
        max_workers: int = 1,
        max_pending: int = 2,
        max_staging_bytes: int = 1 << 30,
        history_size: int = 128,
    ) -> None:
        values = (max_workers, max_pending, max_staging_bytes, history_size)
        if any(type(value) is not int or value <= 0 for value in values):
            raise ConfigurationError("executor capacities must be positive integers")
        self.max_workers = max_workers
        self.max_pending = max_pending
        self.max_staging_bytes = max_staging_bytes
        self._condition = threading.Condition()
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="reelcache-codec"
        )
        self._closed = False
        self._next_id = 1
        self._pending: dict[int, _JobState] = {}
        self._unobserved: dict[int, str] = {}
        self._history: deque[dict] = deque(maxlen=history_size)
        self._staging_bytes = 0
        self._peak_staging_bytes = 0
        self._peak_pending = 0
        self._counts = {
            "submitted": 0, "succeeded": 0, "failed": 0, "cancelled": 0
        }

    def submit(
        self,
        function: Callable[..., T],
        *args,
        staging_bytes: int = 0,
        label: str | None = None,
        block: bool = False,
        timeout: float | None = None,
        **kwargs,
    ) -> Job[T]:
        if not callable(function):
            raise ConfigurationError("submitted function must be callable")
        if type(block) is not bool or (
            label is not None and (not isinstance(label, str) or not label)
        ):
            raise ConfigurationError("block must be boolean and label a nonempty string")
        if type(staging_bytes) is not int or staging_bytes < 0:
            raise ConfigurationError("staging_bytes must be a nonnegative integer")
        _timeout(timeout)
        if timeout is not None and not block:
            raise ConfigurationError("an admission timeout requires block=True")
        if staging_bytes > self.max_staging_bytes:
            raise CapacityError("one job exceeds the entire staging budget")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                if self._closed:
                    raise StateError("executor is closed")
                self._raise_unobserved()
                available = (
                    len(self._pending) < self.max_pending
                    and self._staging_bytes + staging_bytes <= self.max_staging_bytes
                )
                if available:
                    break
                if not block:
                    raise CapacityError("executor job or staging capacity is exhausted")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise CapacityError("timed out waiting for executor admission")
                self._condition.wait(remaining)
            state = _JobState(
                self._next_id,
                label or getattr(function, "__name__", type(function).__name__),
                staging_bytes,
                time.monotonic(),
                function=function, args=args, kwargs=kwargs, telemetry=current_telemetry(),
            )
            self._next_id += 1
            self._pending[state.job_id] = state
            self._staging_bytes += staging_bytes
            self._peak_pending = max(self._peak_pending, len(self._pending))
            self._peak_staging_bytes = max(
                self._peak_staging_bytes, self._staging_bytes
            )
            try:
                future = self._pool.submit(self._run, state)
            except RuntimeError:
                del self._pending[state.job_id]
                self._staging_bytes -= staging_bytes
                self._condition.notify_all()
                raise
            self._counts["submitted"] += 1
            future.add_done_callback(lambda completed: self._finish(state, completed))
            return Job(self, state, future)

    def _run(self, state):
        with self._condition:
            state.started = time.monotonic()
            state.status = "running"
            function, args, kwargs = state.function, state.args, state.kwargs
            state.function, state.args, state.kwargs = None, (), {}
        if function is None:
            raise StateError("job inputs were released before execution")
        with nullcontext() if state.telemetry is None else state.telemetry.capture():
            return function(*args, **kwargs)

    def _finish(self, state: _JobState, future: Future) -> None:
        with self._condition:
            state.finished = time.monotonic()
            if future.cancelled():
                state.status = "cancelled"
            else:
                failure = future.exception()
                state.status = "succeeded" if failure is None else "failed"
                if failure is not None:
                    traceback.clear_frames(failure.__traceback__)
                    state.error = f"{type(failure).__name__}: {failure}"
                    if not state.observed:
                        self._unobserved[state.job_id] = state.error
            self._counts[state.status] += 1
            state.function, state.args, state.kwargs = None, (), {}
            state.telemetry = None
            self._pending.pop(state.job_id)
            self._staging_bytes -= state.staging_bytes
            self._history.append(state.summary())
            self._condition.notify_all()

    def _observe(self, state: _JobState) -> None:
        with self._condition:
            state.observed = True
            self._unobserved.pop(state.job_id, None)

    def _raise_unobserved(self) -> None:
        if self._unobserved:
            detail = "; ".join(
                f"job {job_id}: {error}"
                for job_id, error in sorted(self._unobserved.items())
            )
            raise StateError(f"Unobserved background failure; retrieve job.result(): {detail}")

    def check_errors(self) -> None:
        with self._condition:
            self._raise_unobserved()

    def stats(self) -> dict:
        with self._condition:
            return {
                **self._counts,
                "pending_jobs": len(self._pending),
                "staging_bytes": self._staging_bytes,
                "peak_pending_jobs": self._peak_pending,
                "peak_staging_bytes": self._peak_staging_bytes,
                "max_workers": self.max_workers,
                "max_pending": self.max_pending,
                "max_staging_bytes": self.max_staging_bytes,
                "closed": self._closed,
                "unobserved_failures": dict(self._unobserved),
                "recent_jobs": list(self._history),
            }

    def close(
        self, *, wait_for_running: bool = True, cancel_pending: bool = False,
        raise_errors: bool = True,
    ) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._pool.shutdown(wait=wait_for_running, cancel_futures=cancel_pending)
        if raise_errors:
            self.check_errors()

    def __enter__(self) -> BoundedExecutor:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(raise_errors=exc_type is None)
