"""Bounded K/V restoration on independently owned decoder sessions and streams."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
import os
import threading
import time
import traceback

import torch

from .errors import CapacityError, ConfigurationError, StateError
from .telemetry import current_telemetry
from .types import MatrixKey


class PairedDecoder:
    def __init__(self, provider, *, device, max_output_bytes, max_pairs=1):
        from .codecs import HEVCProvider
        if type(provider) is not HEVCProvider:
            raise ConfigurationError("paired decoding requires the built-in HEVCProvider")
        if type(max_output_bytes) is not int or max_output_bytes <= 0:
            raise ConfigurationError("paired output budget must be positive")
        if type(max_pairs) is not int or max_pairs not in (1, 2):
            raise ConfigurationError("paired decoder supports one pair or active-plus-next pairs")
        self.provider, self.device = provider, torch.device(device)
        self.max_output_bytes = max_output_bytes
        self.max_pairs = max_pairs
        self._active_pairs = self._output_bytes = 0
        self._owner_pid = os.getpid()
        self._pools = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"reelcache-{t}-decode")
                       for t in ("K", "V")]
        self._workers = [{}, {}]
        self._lock = threading.Lock()
        self._closed = False
        self._stats = {"pairs": 0, "failures": 0, "peak_output_bytes": 0,
                       "pair_wall_seconds": 0., "worker_interval_overlap_seconds": 0.,
                       "last_intervals": []}

    def _initialize(self, index):
        from .codecs import HEVCProvider
        from .nvdec import NVDECDecoder
        worker = self._workers[index]
        if not worker:
            old = self.provider.nvdec
            decoder = None if old is None else NVDECDecoder(
                gpu_id=old.gpu_id, gpu_uuid=old.gpu_uuid, max_sessions=old.max_sessions,
                max_output_bytes=old.max_output_bytes, max_bitstream_bytes=old.max_bitstream_bytes,
                device_output=old.device_output,
            )
            provider = HEVCProvider(
                self.provider.ffmpeg_path, nvdec=decoder,
                verify_decoded_coefficients=self.provider.verify_decoded_coefficients,
                verify_restored_packets=self.provider.verify_restored_packets,
            )
            stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
            worker.update(provider=provider, stream=stream)
        return worker

    def _check_owner(self):
        if os.getpid() != self._owner_pid:
            raise StateError("paired decoder cannot cross a process boundary")

    @contextmanager
    def _pair(self, operations, *, output_bytes, cleanup=None):
        self._check_owner()
        with self._lock:
            if self._closed:
                raise StateError("paired decoder is closed")
            if self._active_pairs >= self.max_pairs:
                raise CapacityError("all paired decoder slots are borrowed")
            if type(output_bytes) is not int or output_bytes < 0 or self._output_bytes + output_bytes > self.max_output_bytes:
                raise CapacityError("paired output exceeds the declared byte budget")
            self._active_pairs += 1
            self._output_bytes += output_bytes
            self._stats["peak_output_bytes"] = max(self._stats["peak_output_bytes"], self._output_bytes)
        results, futures, failures = [], [], []
        submitted = time.perf_counter()
        try:
            collector = current_telemetry()
            origin = None
            if self.device.type == "cuda":
                origin = torch.cuda.Event()
                origin.record(torch.cuda.current_stream(self.device))

            def run(index):
                worker = self._initialize(index)
                stream = worker["stream"]
                with (collector.capture() if collector else nullcontext()), \
                        (torch.cuda.stream(stream) if stream else nullcontext()):
                    if stream:
                        stream.wait_event(origin)
                    start = time.perf_counter()
                    try:
                        value = operations[index](worker["provider"])
                    finally:
                        if stream:
                            stream.synchronize()
                    return value, start, time.perf_counter()

            try:
                for index in range(2):
                    futures.append(self._pools[index].submit(run, index))
            except BaseException as exc:
                failures.append(exc)
            for future in futures:
                try:
                    results.append(future.result())
                except BaseException as exc:
                    traceback.clear_frames(exc.__traceback__)
                    failures.append(exc)
            if failures:
                with self._lock:
                    self._stats["failures"] += 1
                if len(failures) == 1:
                    raise failures[0]
                raise StateError("paired decoding failed: " + "; ".join(str(e) for e in failures)) from failures[0]
            intervals = [(start, end) for _, start, end in results]
            with self._lock:
                self._stats["pairs"] += 1
                self._stats["pair_wall_seconds"] += time.perf_counter() - submitted
                self._stats["last_intervals"] = intervals
                self._stats["worker_interval_overlap_seconds"] += max(
                    0., min(end for _, end in intervals) - max(start for start, _ in intervals),
                )
            yield tuple(value for value, _, _ in results)
        finally:
            try:
                if cleanup is not None:
                    cleanup_errors = []
                    for value, _, _ in results:
                        try:
                            cleanup(value)
                        except BaseException as exc:
                            cleanup_errors.append(exc)
                    if cleanup_errors:
                        raise StateError("paired cleanup failed: " + "; ".join(str(e) for e in cleanup_errors)) from cleanup_errors[0]
            finally:
                with self._lock:
                    self._active_pairs -= 1
                    self._output_bytes -= output_bytes

    def decode_pixels(self, pair):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ConfigurationError("decode_pixels requires exactly K and V stream/layout pairs")
        size = sum(layout["frames"] * layout["height"] * layout["width"] * layout["rank"]
                   for _, layout in pair)
        def operation(index):
            streams, layout = pair[index]
            def decode(provider):
                if provider.nvdec is None:
                    raise ConfigurationError("paired pixel decoding requires explicit NVDEC configuration")
                return provider.nvdec.decode_pixels(streams, layout)[0]
            return decode
        with self._pair((operation(0), operation(1)), output_bytes=size) as values:
            return values

    @contextmanager
    def acquire_history(self, runtime, branch, layer, *, token_end, native_dtype, pre_rope_keys,
                        expected_head=None):
        keys = (MatrixKey(layer, "K"), MatrixKey(layer, "V"))
        dtypes = [torch.float32 if key in pre_rope_keys else native_dtype for key in keys]
        size = sum(token_end * runtime.model.channels_for(key) * torch.empty((), dtype=dtype).element_size()
                   for key, dtype in zip(keys, dtypes))
        head = runtime.branch_info(branch)["head_id"] if expected_head is None else expected_head
        runtime.require_head(branch, head)
        def operation(index):
            def restore(provider):
                return runtime.acquire_history(
                    branch, keys[index], token_start=0, token_end=token_end,
                    device=self.device, dtype=dtypes[index], decoder=provider.decode_tensor,
                )
            return restore
        with self._pair((operation(0), operation(1)), output_bytes=size,
                        cleanup=lambda lease: lease.release()) as leases:
            runtime.require_head(branch, head)
            try:
                if self.device.type == "cuda":
                    stream = torch.cuda.current_stream(self.device)
                    for lease in leases:
                        lease.tensor.record_stream(stream)
                yield dict(zip(keys, leases))
            finally:
                if self.device.type == "cuda":
                    consumed = torch.cuda.Event()
                    consumed.record(stream)
                    consumed.synchronize()

    def stats(self):
        return {
            **self._stats, "max_output_bytes": self.max_output_bytes,
            "max_pairs": self.max_pairs, "active_pairs": self._active_pairs, "output_bytes": self._output_bytes,
            "closed": self._closed, "workers": [
                None if not worker else {
                    "stream": None if worker["stream"] is None else worker["stream"].cuda_stream,
                    "nvdec": None if worker["provider"].nvdec is None else worker["provider"].nvdec.stats(),
                } for worker in self._workers
            ],
            "scope": "two independent sessions/streams; wall-interval overlap is not physical-engine attribution",
        }

    def close(self):
        self._check_owner()
        with self._lock:
            if self._active_pairs:
                raise StateError("cannot close paired decoding while outputs are borrowed")
            if self._closed:
                return
            failures = []
            def cleanup(index):
                worker = self._workers[index]
                if worker:
                    if worker["stream"]:
                        worker["stream"].synchronize()
                    if worker["provider"].nvdec:
                        worker["provider"].nvdec.close()
                    worker.clear()
            for index, pool in enumerate(self._pools):
                try:
                    pool.submit(cleanup, index).result()
                except BaseException as exc:
                    failures.append(exc)
                finally:
                    pool.shutdown(wait=True)
            self._closed = True
            if failures:
                raise failures[0]

    def __enter__(self):
        if self._closed:
            raise StateError("paired decoder is closed")
        return self

    def __exit__(self, *_):
        self.close()
