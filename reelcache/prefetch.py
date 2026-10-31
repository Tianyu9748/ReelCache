"""One-layer-ahead restoration with charged leases and consumer-event retirement."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
import time

import torch

from .errors import CapacityError, ConfigurationError, StateError
from .telemetry import current_telemetry, span
from .types import MatrixKey


@dataclass
class _Layer:
    leases: dict
    stack: ExitStack
    ready: object
    start: object
    wall_seconds: float
    consumed_start: object = None
    consumed_end: object = None


class LayerPrefetcher:
    def __init__(self, runtime, *, device, layers, channels, native_dtype, pre_rope_keys,
                 max_bytes, parallel_kv_bytes=0):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ConfigurationError("prefetch requires a positive output byte budget")
        self.runtime, self.device = runtime, torch.device(device)
        self.layers, self.channels, self.native_dtype = layers, channels, native_dtype
        self.pre_rope_keys, self.max_bytes = pre_rope_keys, max_bytes
        if type(parallel_kv_bytes) is not int or parallel_kv_bytes < 0:
            raise ConfigurationError("parallel K/V budget must be a nonnegative integer")
        self.parallel_kv_bytes = parallel_kv_bytes
        self._paired = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reelcache-layer-prefetch")
        self._provider = self._stream = None
        self._pending, self._retirements, self._active = {}, [], {}
        self._binding = None
        self._closed = False
        self._stats = {"epochs": 0, "layers": 0, "wait_seconds": 0., "restore_seconds": 0.,
                       "peak_output_budget_bytes": 0, "stream_overlap_ms": 0., "traces": []}

    def _initialize_worker(self):
        if self._provider is not None:
            return
        from .codecs import HEVCProvider
        from .nvdec import NVDECDecoder

        original = getattr(self.runtime.decoder, "__self__", None)
        if type(original) is HEVCProvider:
            old = original.nvdec
            hardware = None if old is None else NVDECDecoder(
                gpu_id=old.gpu_id, gpu_uuid=old.gpu_uuid, max_sessions=old.max_sessions,
                max_output_bytes=old.max_output_bytes, max_bitstream_bytes=old.max_bitstream_bytes,
                device_output=old.device_output,
            )
            self._provider = HEVCProvider(
                original.ffmpeg_path, nvdec=hardware,
                verify_decoded_coefficients=original.verify_decoded_coefficients,
                verify_restored_packets=original.verify_restored_packets,
            )
        else:
            self._provider = self.runtime
        if self.device.type == "cuda":
            self._stream = torch.cuda.Stream(device=self.device)
        if self.parallel_kv_bytes:
            from .paired_decode import PairedDecoder
            self._paired = PairedDecoder(
                self._provider, device=self.device, max_output_bytes=self.parallel_kv_bytes, max_pairs=2,
            )

    def begin(self, branch, token_end):
        if self._closed or self._binding is not None:
            raise StateError("prefetch is closed or an earlier epoch is still active")
        if token_end <= 0:
            raise ConfigurationError("prefetch requires nonempty committed history")
        widths = [
            sum(4 if MatrixKey(layer, target) in self.pre_rope_keys else torch.empty(
                (), dtype=self.native_dtype,
            ).element_size() for target in ("K", "V"))
            for layer in range(self.layers)
        ]
        reserve = token_end * self.channels * sum(sorted(widths, reverse=True)[:2])
        if reserve > self.max_bytes:
            raise CapacityError("active plus next-layer output exceeds the prefetch byte budget")
        if self.parallel_kv_bytes and reserve > self.parallel_kv_bytes:
            raise CapacityError("active plus next-layer pairs exceed the aggregate paired output budget")
        info = self.runtime.branch_info(branch)
        if info["token_end"] != token_end:
            raise StateError("prefetch cursor differs from the committed immutable head")
        origin = None
        if self.device.type == "cuda":
            origin = torch.cuda.Event(enable_timing=True)
            origin.record(torch.cuda.current_stream(self.device))
        self._binding = (branch, info["head_id"], token_end, origin)
        self._stats["epochs"] += 1
        self._stats["peak_output_budget_bytes"] = max(self._stats["peak_output_budget_bytes"], reserve)
        self._schedule(0)

    def _schedule(self, layer):
        if layer >= self.layers:
            return
        binding = self._binding
        collector = current_telemetry()

        def restore():
            self._initialize_worker()
            stack = ExitStack()
            ready = start = None
            started = time.perf_counter()
            try:
                with (collector.capture() if collector else nullcontext()), \
                        (torch.cuda.stream(self._stream) if self._stream else nullcontext()), span("prefetch.restore"):
                    branch, head, end, origin = binding
                    self.runtime.require_head(branch, head)
                    if self._stream:
                        self._stream.wait_event(origin)
                        start, ready = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record(self._stream)
                    decoder = self.runtime.decoder if self._provider is self.runtime else self._provider.decode_tensor
                    if self._paired is not None:
                        leases = stack.enter_context(self._paired.acquire_history(
                            self.runtime, branch, layer, token_end=end, native_dtype=self.native_dtype,
                            pre_rope_keys=self.pre_rope_keys, expected_head=head,
                        ))
                    else:
                        leases = {}
                        for target in ("K", "V"):
                            key = MatrixKey(layer, target)
                            leases[key] = stack.enter_context(self.runtime.acquire_history(
                                branch, key, token_start=0, token_end=end, device=self.device,
                                dtype=torch.float32 if key in self.pre_rope_keys else self.native_dtype,
                                decoder=decoder,
                            ))
                    self.runtime.require_head(branch, head)
                    if ready:
                        ready.record(self._stream)
                    return _Layer(leases, stack, ready, start, time.perf_counter() - started)
            except BaseException:
                if self._stream:
                    self._stream.synchronize()
                stack.close()
                raise

        self._pending[layer] = self._pool.submit(restore)

    def acquire(self, layer):
        if self._binding is None or layer not in self._pending or layer in self._active:
            raise StateError("prefetch layer order differs from its native epoch")
        started = time.perf_counter()
        with span("prefetch.wait"):
            state = self._pending.pop(layer).result()
        self._stats["wait_seconds"] += time.perf_counter() - started
        self._stats["restore_seconds"] += state.wall_seconds
        if state.ready is not None:
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(state.ready)
            state.consumed_start = torch.cuda.Event(enable_timing=True)
            state.consumed_start.record(stream)
            for lease in state.leases.values():
                lease.tensor.record_stream(stream)
        self._active[layer] = state
        self._stats["layers"] += 1
        self._schedule(layer + 1)

    def tensor(self, layer, key):
        if layer not in self._active:
            raise StateError("prefetch operands are not acquired for this layer")
        return self._active[layer].leases[key].tensor

    def release(self, layer):
        state = self._active.pop(layer)
        if state.ready is not None:
            state.consumed_end = torch.cuda.Event(enable_timing=True)
            state.consumed_end.record(torch.cuda.current_stream(self.device))

        def retire():
            try:
                if state.consumed_end is not None:
                    state.consumed_end.synchronize()
            finally:
                state.stack.close()
            return state

        self._retirements.append(self._pool.submit(retire))

    def end(self):
        if self._binding is None:
            return
        failures, states = [], []
        for layer in list(self._active):
            self.release(layer)
        for future in self._pending.values():
            try:
                state = future.result()
                if state.ready is not None:
                    state.ready.synchronize()
                state.stack.close()
            except BaseException as exc:
                failures.append(exc)
        for future in self._retirements:
            try:
                states.append(future.result())
            except BaseException as exc:
                failures.append(exc)
        origin = self._binding[3]
        if origin is not None:
            trace = [{
                "restore_start_ms": origin.elapsed_time(s.start),
                "restore_end_ms": origin.elapsed_time(s.ready),
                "consume_start_ms": origin.elapsed_time(s.consumed_start),
                "consume_end_ms": origin.elapsed_time(s.consumed_end),
            } for s in states]
            overlap = sum(max(0., min(a["consume_end_ms"], b["restore_end_ms"]) -
                                 max(a["consume_start_ms"], b["restore_start_ms"]))
                          for a, b in zip(trace, trace[1:]))
            self._stats["stream_overlap_ms"] += overlap
            self._stats["traces"] = trace
        self._pending.clear()
        self._retirements.clear()
        self._binding = None
        if failures:
            raise failures[0]

    def stats(self):
        decoder = None if self._provider is self.runtime else getattr(self._provider, "nvdec", None)
        return {**self._stats, "active": self._binding is not None, "max_bytes": self.max_bytes,
                "worker_decoder": None if decoder is None else decoder.stats(),
                "worker_cuda_stream": None if self._stream is None else self._stream.cuda_stream,
                "paired_reconstruction": None if self._paired is None else self._paired.stats(),
                "scope": "one-layer lookahead; CUDA stream intervals include CPU preparation gaps, not kernel-only time"}

    def close(self):
        if self._closed:
            return
        try:
            self.end()
        finally:
            def cleanup():
                if self._stream:
                    self._stream.synchronize()
                if self._paired is not None:
                    self._paired.close()
                    self._paired = None
                if self._provider is not None and self._provider is not self.runtime and self._provider.nvdec is not None:
                    self._provider.nvdec.close()
                self._provider = self._stream = None
            try:
                self._pool.submit(cleanup).result()
            finally:
                self._pool.shutdown(wait=True)
                self._closed = True


class PackedLayerPrefetcher:
    """Layer-ahead device restore of attended history nodes from schema-v2 packets.

    ``begin(nodes, resident=...)`` plans only the missing (non-resident)
    entries and submits the decode jobs of the first ``depth + 1`` layers
    (keep ``depth`` at least the pool's worker count for single-node restores;
    the arena must hold ``depth + 1`` layers of canvases).
    ``restore(layer, destination, rope)`` first submits through ``layer +
    depth`` (NVDEC decodes ahead while this layer reconstructs and the model
    computes), then reconstructs ``layer`` straight into its native slots and
    returns the ``restore_kernels`` handle. Canvases are recycled after each
    handle's event; ``end`` waits for outstanding decodes and recycles the rest.
    """

    def __init__(self, restorer, *, depth=8):
        from .packed_decode import PackedRestorer

        if not isinstance(restorer, PackedRestorer):
            raise ConfigurationError("restorer must be a PackedRestorer")
        if type(depth) is not int or depth < 0:
            raise ConfigurationError("prefetch depth must be a nonnegative integer")
        self.restorer, self.depth = restorer, depth
        self._batch, self._layers = None, []
        self._stats = {"epochs": 0, "layers": 0, "reconstruct_issue_seconds": 0.0, "entries": 0,
                       "skipped_resident": 0}

    def begin(self, nodes, *, resident=(), keys=None):
        if self._batch is not None:
            raise StateError("an earlier packed prefetch epoch is still active")
        batch = self.restorer.plan(nodes, resident=resident, keys=keys)
        self._batch, self._layers = batch, batch.layers
        self._stats["epochs"] += 1
        self._stats["entries"] += len(batch)
        self._stats["skipped_resident"] += batch.skipped
        if self._layers:
            batch.submit_through(self._layers[min(self.depth, len(self._layers) - 1)])
        return batch

    @property
    def layers(self):
        return list(self._layers)

    def restore(self, layer, destination, rope=None, *, stream=None, **options):
        """Reconstruct every restored entry of ``layer``; None when the layer has none."""
        if self._batch is None:
            raise StateError("packed prefetch has no active epoch")
        if layer not in self._layers:
            return None
        position = self._layers.index(layer)
        self._batch.submit_through(self._layers[min(position + self.depth, len(self._layers) - 1)])
        started = time.perf_counter()
        with span("prefetch.packed_restore"):
            handle = self._batch.reconstruct(self._batch.entries(layer), destination, rope, stream=stream, **options)
        self._stats["reconstruct_issue_seconds"] += time.perf_counter() - started
        self._stats["layers"] += 1
        return handle

    def end(self):
        if self._batch is None:
            return
        batch, self._batch, self._layers = self._batch, None, []
        batch.close()

    def stats(self):
        return {**self._stats, "active": self._batch is not None, "depth": self.depth,
                "restorer": self.restorer.stats()}
