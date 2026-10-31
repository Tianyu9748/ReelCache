"""Opt-in, bounded CPU-wall attribution; no tensors or per-packet identities."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import threading
import time

from .errors import ConfigurationError


_CURRENT = ContextVar("reelcache_telemetry", default=None)
STAGES = (
    "store.read", "store.hash", "store.commit",
    "decode.validate", "decode.basis", "decode.coefficients",
    "decode.hevc", "decode.dequantize", "decode.reconstruct",
    "history.restore", "history.copy", "transport.collective",
    "native.h2d", "native.rope", "native.insert",
    "decode.checksum_readback", "prefetch.wait", "prefetch.restore",
    "quality.append",
    "nvdec.demux", "nvdec.decode", "nvdec.unpack", "decode.transfer",
    "history.packet", "sp.packets", "sp.decode", "sp.exchange", "seal.stream_add",
    "seal.prefit", "seal.encode", "seal.wait", "seal.persist", "seal.publish",
)


class RuntimeTelemetry:
    """Aggregate inclusive spans; overlapping/nested durations must not be added.

    Captures are context-local. BoundedExecutor explicitly carries the active
    collector into its workers; arbitrary threads/processes do not inherit it.
    CUDA operations are not synchronized, so durations are not device timings.
    """

    def __init__(self, *, cuda_timing=False):
        if type(cuda_timing) is not bool:
            raise ConfigurationError("cuda_timing must be boolean")
        self.cuda_timing = cuda_timing
        self._lock = threading.Lock()
        self._cuda = {name: {"calls": 0, "seconds": 0.0} for name in STAGES}
        self._stages = {
            name: {
                "calls": 0, "completed": 0, "failed": 0, "inflight": 0,
                "wall_seconds": 0.0, "attempted_bytes": 0, "completed_bytes": 0,
            }
            for name in STAGES
        }

    @contextmanager
    def capture(self):
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)

    @contextmanager
    def _span(self, stage, nbytes):
        if stage not in self._stages or type(nbytes) is not int or nbytes < 0:
            raise ConfigurationError("telemetry requires a known stage and nonnegative integer bytes")
        started = time.perf_counter()
        with self._lock:
            row = self._stages[stage]
            row["calls"] += 1
            row["inflight"] += 1
            row["attempted_bytes"] += nbytes
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            elapsed = time.perf_counter() - started
            with self._lock:
                row["inflight"] -= 1
                row["wall_seconds"] += elapsed
                row["completed" if succeeded else "failed"] += 1
                if succeeded:
                    row["completed_bytes"] += nbytes

    def snapshot(self):
        with self._lock:
            return {
                "schema": "reelcache.runtime_telemetry.v1",
                "timing": "inclusive CPU wall; nested/concurrent spans are not additive",
                "bytes": "stage-specific attempted/completed application bytes, not unique storage",
                "cuda_synchronized": any(row["calls"] for row in self._cuda.values()),
                "cuda_timing_enabled": self.cuda_timing,
                "cuda_stage_scope": "optional synchronized CUDA event spans; profiling changes scheduling",
                "cuda_stages": {name: dict(row) for name, row in self._cuda.items()},
                "physical_memory_measured": False,
                "stages": {name: dict(row) for name, row in self._stages.items()},
            }

    @contextmanager
    def _cuda_span(self, stage, device):
        import torch
        if stage not in self._cuda:
            raise ConfigurationError("unknown CUDA telemetry stage")
        with torch.cuda.device(device):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                end.synchronize()
                with self._lock:
                    self._cuda[stage]["calls"] += 1
                    self._cuda[stage]["seconds"] += start.elapsed_time(end) / 1000.


def current_telemetry():
    return _CURRENT.get()


def span(stage, *, nbytes=0):
    collector = current_telemetry()
    return nullcontext() if collector is None else collector._span(stage, nbytes)


def cuda_span(stage, device):
    collector = current_telemetry()
    if collector is None or not collector.cuda_timing or str(device).split(":")[0] != "cuda":
        return nullcontext()
    return collector._cuda_span(stage, device)
