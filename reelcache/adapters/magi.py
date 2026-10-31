"""Instance-local MAGI Ulysses clean-history adapter.

The supported execution mode is **serial, batch-one clean history**, not MAGI's
fully pipelined scheduler. Native attention, GQA, masks, positions and the
near-clean duplicate chunk are unchanged. RAW stores native post-RoPE K bytes;
lossy providers see normalized FP32 pre-RoPE K and full global KV channels.
Neither importing this module nor constructing the geometry loads MAGI/CUDA.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import random
import shutil
import sys
import uuid

from ..distributed import DistributedOwner, KVLayout, PeerFailure
from ..errors import CapacityError, ConfigurationError, IntegrityError, StateError, UnsupportedError
from ..types import CodecSpec, MatrixKey, ModelIdentity, TensorContext, canonical_dtype
from ..telemetry import span


_MISSING = object()
NATIVE_TENSOR_HASH_FORMAT = "sha256-native-contiguous-bytes-no-header-v1"


@dataclass(frozen=True)
class MagiGeometry:
    layers: int = 48
    kv_heads: int = 8
    head_dim: int = 128
    frames: int = 6
    height: int = 68
    width: int = 120
    world_size: int = 2

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ConfigurationError("MAGI geometry dimensions must be positive integers")
        self.layout

    @property
    def tokens(self):
        return self.frames * self.height * self.width

    @property
    def channels(self):
        return self.kv_heads * self.head_dim

    @property
    def layout(self):
        return KVLayout(self.tokens, self.kv_heads, self.head_dim, self.world_size)

    @classmethod
    def for_cell(cls, cell_id):
        from ..study import study_cell

        cell = study_cell(cell_id)
        if cell.model != "magi_distill":
            raise ConfigurationError("A MAGI adapter requires a MAGI study cell")
        return cls(layers=cell.layers, frames=cell.frames, height=cell.token_height,
                   width=cell.token_width, world_size=cell.world_size)


@dataclass(frozen=True)
class MagiResourceBudget:
    """Explicit simultaneous request allowances, not a measured HBM claim.

    Workspace allowances include the resident model, attention/denoising,
    oracle/factorization and decoder work not owned by this adapter. The
    separately supplied packet allowance covers live encoded output.
    """

    host_bytes: int
    device_bytes: int
    staging_bytes: int
    workspace_host_bytes: int
    workspace_device_bytes: int
    packet_bytes: int
    headroom_bytes: int = 0

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in asdict(self).values()):
            raise ConfigurationError("MAGI resource allowances must be nonnegative integer bytes")
        if min(self.host_bytes, self.device_bytes, self.staging_bytes) <= 0:
            raise ConfigurationError("MAGI request resource limits must be positive")

    def admit(self, geometry, *, chunks, factor_keys, observer, quality_hot):
        if type(chunks) is not int or chunks < 1:
            raise ConfigurationError("Admission needs a positive clean/current chunk count")
        n, c, layers, world = geometry.tokens, geometry.channels, geometry.layers, geometry.world_size
        source = n * c * layers * (6 if factor_keys else 4)
        observed = n * c * layers * 6 if observer else 0
        row_count = (n + world - 1) // world
        local_source = row_count * c * layers * 6 if factor_keys else n * c * layers * 4 // world
        local_staging = local_source + (row_count * c * layers * 6 if observer else 0)
        staging = local_staging * world + source
        full_matrix = n * c * 4
        native_chunk = n * geometry.layout.local_channels * layers * 4
        native = native_chunk * chunks if quality_hot else native_chunk * chunks // layers
        growth = native_chunk * max(0, chunks - 1) // layers if quality_hot else 0
        # Original mmap pages, gather/receive copies and native RoPE intermediates
        # may coexist. Count page-backed sources as host memory, not only disk.
        transfer = 4 * full_matrix * world
        retained_rotary = n * geometry.head_dim * 8 * chunks * world
        native_temporaries = 6 * n * geometry.layout.local_channels * 4
        required = {
            "host_bytes": (staging + transfer + retained_rotary
                           + self.workspace_host_bytes + self.packet_bytes),
            "device_bytes": (native + growth + native_temporaries
                             + self.workspace_device_bytes + self.headroom_bytes),
            "staging_bytes": staging,
            "canonical_source_bytes": source,
            "observer_source_bytes": observed,
            "local_capture_bytes": local_staging,
            "global_capture_bytes": local_staging * world,
            "shard_transfer_bytes": transfer,
            "retained_cpu_rotary_bytes": retained_rotary,
            "native_history_current_bytes": native,
            "native_growth_copy_bytes": growth,
            "native_duplicate_and_rope_bytes": native_temporaries,
            "workspace_host_bytes": self.workspace_host_bytes,
            "workspace_device_bytes": self.workspace_device_bytes,
            "packet_bytes": self.packet_bytes,
            "headroom_bytes": self.headroom_bytes,
            "accounting": "conservative simultaneous allowances; global SP host/disk and per-rank device",
        }
        for name in ("host_bytes", "device_bytes", "staging_bytes"):
            if required[name] > getattr(self, name):
                raise CapacityError(f"MAGI admitted {name} exceeded: {required[name]} > {getattr(self, name)}")
        return required


def _tensor_hash(tensor):
    import torch

    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(memoryview(value.numpy())).hexdigest()


def _tensor_hash_format(dtype):
    return {
        "algorithm": "sha256", "header": "none", "byte_order": sys.byteorder,
        "layout": "C_contiguous_row_major", "dtype": canonical_dtype(dtype),
    }


def _tuple_tree(value):
    return tuple(_tuple_tree(item) for item in value) if isinstance(value, list) else value


def _tensor_context(model, key, token_start, geometry, key_space, source_dtype):
    source_dtype = canonical_dtype(source_dtype)
    context = TensorContext(
        model, key, token_start, geometry.frames, geometry.height, geometry.width, key_space,
        source_dtype=None if model.dtype_for(key) == source_dtype else source_dtype,
    )
    if context.dtype != source_dtype:
        raise ConfigurationError("Canonical matrix dtype differs from the captured native source")
    return context


def _portable_spec(spec):
    fields = spec.to_dict()
    fields.pop("ffmpeg_path", None)
    return fields


def _checked_reuse_threshold(value):
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ConfigurationError("MAGI reuse threshold must be a finite nonnegative residual limit")
    return float(value)


class _DiskMatrices(Mapping):
    """One-matrix staging, separate from canonical immutable runtime objects."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.entries = {}

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, key):
        import torch

        row = self.entries[key]
        value = torch.from_file(
            str(row["path"]), shared=False, size=row["numel"], dtype=getattr(torch, row["dtype"]),
        ).reshape(row["shape"])
        if _tensor_hash(value) != row["sha256"]:
            raise IntegrityError("MAGI transient matrix staging changed before publication")
        return value

    def put(self, key, tensor):
        import torch

        if key in self.entries:
            raise StateError(f"Native clean forward produced a matrix twice: {key}")
        value = tensor.detach().cpu().contiguous()
        path = self.directory / f"layer{key.layer:03d}_{key.target}.bin"
        with path.open("xb") as handle:
            value.reshape(-1).view(torch.uint8).numpy().tofile(handle)
        self.entries[key] = {
            "path": path, "numel": value.numel(), "shape": tuple(value.shape),
            "dtype": str(value.dtype).split(".")[-1], "sha256": _tensor_hash(value),
            "bytes": value.numel() * value.element_size(),
        }

    @property
    def nbytes(self):
        return sum(row["bytes"] for row in self.entries.values())


_STREAM_END = object()
_STREAM_ABORT = object()


class _LayerStream(Mapping):
    """Layer-wise clean-KV pipeline: gather each layer to the owner and encode it during the forward.

    ``put`` is called on the native execution thread as each layer produces its
    clean K/V shard. It snapshots the shard (device clone ordered on the current
    stream) and enqueues it. One pipeline thread per rank moves shards to the
    owner over a *separate* process group, in the deterministic put order, and
    the owner hands each full canonical matrix to a :class:`StreamingChunk`
    (optionally a parallel CPU encode pool) while later layers keep computing.
    Mapping access exposes only per-key metadata (for completeness checks).
    A failure on any rank is reported collectively on the stream group, so all
    ranks agree on the outcome returned by :meth:`finish`.
    """

    def __init__(self, adapter, directory, contexts, writer, *, depth):
        import queue
        import threading

        self.adapter, self.contexts, self.writer = adapter, contexts, writer
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.entries = {}
        self.hashes = {}
        self.error = None
        self.outcome = None
        self.stats = {"put": 0, "gathered": 0, "peak_queued_bytes": 0, "queued_bytes": 0,
                      "gather_seconds": 0.0, "hash_seconds": 0.0, "add_seconds": 0.0}
        self._queue = queue.Queue(maxsize=depth)
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="magi-layer-stream", daemon=True)
        self._thread.start()

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, key):
        raise UnsupportedError("streamed clean KV is not retained after its layer is gathered")

    @property
    def nbytes(self):
        return sum(row["bytes"] for row in self.entries.values())

    def put(self, key, tensor):
        import torch

        if self.error is not None or not self._thread.is_alive():
            raise PeerFailure(f"MAGI layer stream failed: {self.error}")
        try:
            if key in self.entries:
                raise StateError(f"Native clean forward produced a matrix twice: {key}")
            value = tensor.detach()
            if canonical_dtype(value.dtype) != self.contexts[key].dtype:
                raise IntegrityError("Streamed clean KV dtype differs from its declared context")
            snapshot = value.clone(memory_format=torch.contiguous_format)
            event = None
            if snapshot.device.type == "cuda":
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(snapshot.device))
        except BaseException as exc:
            # Keep every rank's pipeline sequence aligned: report this key's failure collectively.
            self._queue.put((key, None, exc, None, 0))
            raise
        nbytes = snapshot.numel() * snapshot.element_size()
        self.entries[key] = {"shape": tuple(snapshot.shape), "dtype": canonical_dtype(snapshot.dtype),
                             "bytes": nbytes}
        layout = "heads" if self.adapter._source_is_raw(key) else "rows"
        with self._lock:
            self.stats["put"] += 1
            self.stats["queued_bytes"] += nbytes
            self.stats["peak_queued_bytes"] = max(self.stats["peak_queued_bytes"], self.stats["queued_bytes"])
        self._queue.put((key, snapshot, event, layout, nbytes))

    def _run(self):
        import time

        import torch

        adapter = self.adapter
        transport = side = failure = None
        usable = False
        try:
            transport = DistributedOwner(
                adapter.stream_group, owner_rank=adapter.transport.owner_rank,
                max_wire_bytes=adapter.transport.max_wire_bytes,
                expected_world_size=adapter.geometry.world_size,
            )
            usable = True
            if adapter.stream_thread_init is not None and transport.is_owner:
                adapter.stream_thread_init()
        except BaseException as exc:
            failure = exc
        aborted = False
        while True:
            item = self._queue.get()
            if item is _STREAM_END or item is _STREAM_ABORT:
                aborted = item is _STREAM_ABORT
                break
            key, snapshot, event, layout, nbytes = item
            operation = f"magi_stream:{adapter._cursor}:{key.layer}:{key.target}"
            local = full = None
            try:
                if not usable:
                    continue
                if snapshot is None:
                    transport.checkpoint(operation, error=event)
                    continue
                if failure is not None:
                    transport.checkpoint(operation, error=failure)
                    continue
                if event is not None:
                    if side is None:
                        side = torch.cuda.Stream(snapshot.device)
                    with torch.cuda.stream(side):
                        side.wait_event(event)
                        local = torch.empty(snapshot.shape, dtype=snapshot.dtype, pin_memory=True)
                        local.copy_(snapshot, non_blocking=True)
                        snapshot.record_stream(side)
                    side.synchronize()
                else:
                    local = snapshot
                snapshot = None
                started = time.perf_counter()
                full = transport.gather_matrix(local, adapter.geometry.layout, source_layout=layout,
                                               operation=operation)
                self.stats["gather_seconds"] += time.perf_counter() - started
                local = None
                if transport.is_owner:
                    started = time.perf_counter()
                    self.hashes[key] = _tensor_hash(full)
                    self.stats["hash_seconds"] += time.perf_counter() - started
                    started = time.perf_counter()
                    try:
                        with span("seal.stream_add"):
                            self.writer.add(key, full)
                    except BaseException as exc:
                        failure = exc
                    self.stats["add_seconds"] += time.perf_counter() - started
                self.stats["gathered"] += 1
            except PeerFailure as exc:
                failure = failure or exc
                # An order mismatch means a peer may already have left the stream: no more collectives.
                if "Collective order mismatch" in str(exc) or transport._broken:
                    usable = False
            except BaseException as exc:
                failure = failure or exc
            finally:
                with self._lock:
                    self.stats["queued_bytes"] -= nbytes
                snapshot = local = full = None
        error = None if failure is None else f"{type(failure).__name__}: {failure}"
        self.outcome = "failed"
        try:
            if not usable:
                raise PeerFailure(error or "MAGI layer stream transport is unusable")
            transport.checkpoint("magi_stream_end", error=error, cancelled=aborted)
            self.outcome = "complete"
        except BaseException as exc:
            self.outcome = "aborted" if aborted else "failed"
            error = error or f"{type(exc).__name__}: {exc}"
        self.error = error

    def finish(self, *, abort=False, timeout=None):
        """Stop the pipeline after queued layers; return the collectively agreed outcome."""
        self._queue.put(_STREAM_ABORT if abort else _STREAM_END)
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise PeerFailure("MAGI layer stream did not finish")
        return self.outcome, self.error


class _SPCapture(_LayerStream):
    """``sp_tiered`` clean capture: inline device gather, owner-only background D2H and encode.

    ``put`` runs on the native execution thread of every rank in the same
    order. It gathers the shard to the owner's device over the SP data group
    (NCCL over NVLink on GPUs) with a preceding control checkpoint, so no rank
    enters the collective alone. Only the owner's pipeline thread then copies
    the full matrix to pinned host memory and hands it to the streaming
    chunk writer; it performs no collectives. The outcome is agreed at seal.
    """

    def put(self, key, tensor):
        import torch

        adapter = self.adapter
        layout = "heads" if adapter._source_is_raw(key) else "rows"
        error = None
        try:
            if key in self.entries:
                raise StateError(f"Native clean forward produced a matrix twice: {key}")
            if canonical_dtype(tensor.dtype) != self.contexts[key].dtype:
                raise IntegrityError("Captured clean KV dtype differs from its declared context")
            if self.error is not None or not self._thread.is_alive():
                raise PeerFailure(f"MAGI owner encode pipeline failed: {self.error}")
        except Exception as exc:
            error = exc
        from ..sp_kv import PendingGather

        full = adapter.sp_history.gather_to_owner(
            tensor.detach(), layout, operation=f"sp_capture:{adapter._cursor}:{key.layer}:{key.target}",
            error=error, async_op=adapter.sp_history.overlap,
        )
        nbytes = tensor.numel() * tensor.element_size()
        self.entries[key] = {"shape": tuple(tensor.shape), "dtype": canonical_dtype(tensor.dtype), "bytes": nbytes}
        with self._lock:
            self.stats["put"] += 1
        if not adapter.transport.is_owner:
            return
        event = None
        if not isinstance(full, PendingGather) and full.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(full.device))
        size = adapter.geometry.tokens * adapter.geometry.channels * tensor.element_size()
        with self._lock:
            self.stats["queued_bytes"] += size
            self.stats["peak_queued_bytes"] = max(self.stats["peak_queued_bytes"], self.stats["queued_bytes"])
        self._queue.put((key, full, event, layout, size))

    def _run(self):
        import time

        import torch

        from ..sp_kv import PendingGather

        side = failure = None
        aborted = False
        while True:
            item = self._queue.get()
            if item is _STREAM_END or item is _STREAM_ABORT:
                aborted = item is _STREAM_ABORT
                break
            key, full, event, _, size = item
            host = None
            try:
                if failure is not None or aborted:
                    if isinstance(full, PendingGather):
                        full.wait()
                    continue
                started = time.perf_counter()
                if isinstance(full, PendingGather):
                    device = full._pieces[0].device if full._pieces else torch.device("cpu")
                    if device.type == "cuda":
                        if side is None:
                            side = torch.cuda.Stream(device)
                        with torch.cuda.stream(side):
                            full = full.wait()
                            host = torch.empty(full.shape, dtype=full.dtype, pin_memory=True)
                            host.copy_(full, non_blocking=True)
                            full.record_stream(side)
                        side.synchronize()
                    else:
                        host = full.wait()
                elif event is not None:
                    if side is None:
                        side = torch.cuda.Stream(full.device)
                    with torch.cuda.stream(side):
                        side.wait_event(event)
                        host = torch.empty(full.shape, dtype=full.dtype, pin_memory=True)
                        host.copy_(full, non_blocking=True)
                        full.record_stream(side)
                    side.synchronize()
                elif host is None:
                    host = full
                full = None
                self.stats["gather_seconds"] += time.perf_counter() - started
                started = time.perf_counter()
                self.hashes[key] = _tensor_hash(host)
                self.stats["hash_seconds"] += time.perf_counter() - started
                started = time.perf_counter()
                with span("seal.stream_add"):
                    self.writer.add(key, host)
                self.stats["add_seconds"] += time.perf_counter() - started
                self.stats["gathered"] += 1
            except BaseException as exc:
                failure = failure or exc
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                with self._lock:
                    self.stats["queued_bytes"] -= size
                full = host = None
        self.error = None if failure is None else f"{type(failure).__name__}: {failure}"
        self.outcome = "aborted" if aborted else ("failed" if failure is not None else "complete")


class MagiAdapter:
    """A reversible seam around one native model and one InferenceParams.

    Every rank constructs this adapter; only the storage owner receives a
    ``runtime``. Storage traffic should use a separate ordered CPU/Gloo group.
    Call ``execution`` around a complete native clean forward or the whole
    denoising loop, then ``seal_chunk`` after a successful clean forward.

    ``rotary_apply`` is the *native* ``flash_apply_rotary_emb`` function,
    supplied by the external builder. No native module is imported here.

    RAW and factor branches share one native BF16 model identity. Captured
    normalized pre-RoPE K explicitly declares ``source_dtype="float32"`` in
    its tensor context, while RAW post-RoPE K and V retain the native default.
    Legacy declared dtype maps remain supported. No implicit tensor cast or
    RAW-post-to-pre-RoPE conversion is performed.

    Restore/fork preserve the requested portable codec profile and configured
    reuse threshold. Construct a new adapter with the saved profile explicitly;
    snapshots never silently replace the caller's profile or FFmpeg path.
    Adaptive restores use core's pre-deserialization policy gate and compare
    capsule metadata with typed policy/state derived from the verified head.

    ``rank_policy`` is a typed experimental alternative: only core prepares,
    resolves and advances it in an atomic whole-clean commit. ``source_observer``
    receives (portable_provenance, original_full_matrix) on the owner, including
    original FP32 pre-K on native RAW branches. It must not mutate/retain input
    outside its explicitly admitted source/workspace allowance. An adaptive
    observer may return a complete portable per-matrix source-provenance mapping;
    core binds that metadata before fitting. Returning None retains the generic
    numerical API without a study-provenance declaration.

    Default ``cache_mode="runtime"`` still releases every native layer buffer.
    Explicit ``quality_hot`` retains packet-decoded, native-ready shards for one
    request/branch and replaces each new clean slot before advancing. It requires
    a full simultaneous resource budget and separate native parity qualification;
    it is not an HBM-compression or speedup claim.

    Clean K/V capture failures are agreed before native Ulysses. A peer or
    transport failure aborts the epoch without another completion collective.

    ``hot_window_bytes=0`` preserves the original runtime path. A positive
    budget retains only owned, packet-decoded, native-dtype local-head history
    within one denoise execution. It never substitutes source KV or coefficients.
    ``hot_window_require_full`` rejects partial admission; ``hot_window_verify_hits``
    independently redecodes that many hits per epoch for explicit parity checks.
    HBM admission reserves ``hot_window_headroom_bytes`` in addition to the hot
    tensors. Epoch exit, failure, publication, restore and detach fence/release
    this pool. Physical allocator measurements remain separate from tensor bytes.
    """

    mode = "serial_clean_history"
    schema = "reelcache_magi_serial_clean_v1"

    def __init__(self, model, inference_params, runtime, *, model_identity: ModelIdentity,
                 branch_id: str, staging_dir, geometry=MagiGeometry(), coordinator=None,
                 specs=CodecSpec(), rotary_apply=None, device="cpu",
                 max_layer_buffer_bytes=2 * 1024**3, native_dtype=None,
                 reuse_threshold=None, hot_window_bytes=0,
                 hot_window_headroom_bytes=12 * 1024**3, hot_window_require_full=False,
                 hot_window_verify_hits=0, source_observer=None, source_context=None,
                 resource_budget=None, cache_mode="runtime", quality_hot_bytes=0,
                 quality_verify_history=True, rank_policy=None, shared_history_decode=False,
                 visible_history_only=False, layer_streaming=False, stream_group=None,
                 stream_executor=None, stream_queue_depth=4, stream_thread_init=None,
                 sp_history=None):
        import torch

        self.model = model
        self.cache = inference_params
        self.runtime = runtime
        self.identity = model_identity
        self.native_dtype = canonical_dtype(
            model_identity.dtype_for(MatrixKey(0, "V")) if native_dtype is None else native_dtype,
        )
        self.geometry = geometry
        self.transport = coordinator or DistributedOwner(expected_world_size=geometry.world_size)
        self.device = torch.device(device)
        if self.transport.world_size != geometry.world_size:
            raise ConfigurationError("Native SP and storage rank ownership differ")
        if self.transport.is_owner != (runtime is not None):
            raise ConfigurationError("Only the canonical owner may hold a KVRuntime")
        if self.transport.device.type != "cpu":
            raise UnsupportedError("Initial MAGI storage traffic uses CPU/Gloo; native attention retains NCCL")
        if (model_identity.layers != geometry.layers
                or any(model_identity.channels_for(target) != geometry.channels for target in ("K", "V"))):
            raise ConfigurationError("Model identity does not describe global MAGI KV channels")
        if getattr(inference_params, "max_batch_size", None) != 1:
            raise UnsupportedError("Only batch-one serial clean-history MAGI is supported")
        if getattr(inference_params, "kv_compressed", False):
            raise UnsupportedError("Native FlowCache-compressed state is not a clean full-KV boundary")
        if max_layer_buffer_bytes <= 0:
            raise ConfigurationError("The per-layer native buffer budget must be positive")
        self.branch_id = branch_id
        self.rotary_apply = rotary_apply
        self.staging_dir = Path(staging_dir)
        self.max_layer_buffer_bytes = max_layer_buffer_bytes
        self.reuse_threshold = _checked_reuse_threshold(reuse_threshold)
        if source_observer is not None and not callable(source_observer):
            raise ConfigurationError("source_observer must be callback(context, original_tensor)")
        if source_context is not None and not isinstance(source_context, Mapping):
            raise ConfigurationError("source_context must be portable request provenance")
        if resource_budget is not None and not isinstance(resource_budget, MagiResourceBudget):
            raise ConfigurationError("resource_budget must be a MagiResourceBudget")
        if cache_mode not in ("runtime", "quality_hot", "sp_tiered"):
            raise ConfigurationError("cache_mode must be runtime, explicit quality_hot, or sp_tiered")
        if type(quality_hot_bytes) is not int or quality_hot_bytes < 0:
            raise ConfigurationError("quality_hot_bytes must be nonnegative integer bytes")
        if type(quality_verify_history) is not bool:
            raise ConfigurationError("quality_verify_history must be boolean")
        if (source_observer is not None or cache_mode == "quality_hot") and resource_budget is None:
            raise ConfigurationError("Source capture and quality_hot require complete resource admission")
        if cache_mode == "quality_hot" and (quality_hot_bytes <= 0 or hot_window_bytes):
            raise ConfigurationError("quality_hot requires its own positive budget, not a runtime hot window")
        if cache_mode == "runtime" and quality_hot_bytes:
            raise ConfigurationError("A quality-hot budget must explicitly select quality_hot")
        self.source_observer = source_observer
        self.source_context = json.loads(json.dumps(source_context or {}, allow_nan=False))
        self.resource_budget = resource_budget
        self.cache_mode = cache_mode
        if type(shared_history_decode) is not bool:
            raise ConfigurationError("shared_history_decode must be boolean")
        if shared_history_decode and cache_mode != "runtime":
            raise ConfigurationError("shared_history_decode is an opt-in runtime restoration path")
        self.shared_history_decode = shared_history_decode
        if type(visible_history_only) is not bool:
            raise ConfigurationError("visible_history_only must be boolean")
        if visible_history_only and cache_mode not in ("runtime", "sp_tiered"):
            raise ConfigurationError("visible_history_only requires runtime restoration")
        self.visible_history_only = visible_history_only
        self.quality_hot_bytes = quality_hot_bytes
        self.quality_verify_history = quality_verify_history
        self._source_pending = None
        self._pending_source_provenance = {}
        self._admission = None
        self._quality_buffers = {}
        self._quality_hashes = {}
        self._quality_head = None
        self._quality_binding = None
        self._quality_pin = None
        self._quality_stats = {
            "packet_decodes_per_recipient": 0, "clean_slot_replacements": 0,
            "protected_history_checks": 0, "peak_native_bytes": 0,
            "native_parity_gate": "blocked_until_native_hot_vs_reload_KV_latent_RGB_both_resolutions",
            "HBM_compression_claim": False,
        }
        if any(type(value) is not int or value < 0 for value in
               (hot_window_bytes, hot_window_headroom_bytes, hot_window_verify_hits)):
            raise ConfigurationError("Hot-window byte budgets and verification count must be nonnegative integers")
        if type(hot_window_require_full) is not bool:
            raise ConfigurationError("hot_window_require_full must be boolean")
        self.hot_window_bytes = hot_window_bytes
        self.hot_window_headroom_bytes = hot_window_headroom_bytes
        self.hot_window_require_full = hot_window_require_full
        self.hot_window_verify_hits = hot_window_verify_hits
        self._hot = {}
        self._hot_bytes = 0
        self._hot_epoch = None
        self._hot_pin = None
        self._hot_event = None
        self._hot_verified = 0
        self._hot_stats = {
            "epochs": 0, "hits": 0, "misses": 0, "bypasses": 0, "verified_hits": 0,
            "peak_bytes": 0, "last_epoch_peak_bytes": 0, "last_epoch_hits": 0,
            "last_epoch_misses": 0, "last_epoch_required_bytes": 0,
        }
        self.keys = tuple(MatrixKey(layer, target)
                          for layer in range(geometry.layers) for target in ("K", "V"))
        self.rank_policy = None
        if rank_policy is not None:
            from ..online_rank import RankPolicy

            if not isinstance(rank_policy, RankPolicy):
                raise ConfigurationError("rank_policy must be the typed experimental RankPolicy")
            if specs != CodecSpec() or reuse_threshold is not None:
                raise ConfigurationError("An adaptive policy owns resolved specs and reuse; do not override them")
            if self.resource_budget is None:
                raise ConfigurationError("An adaptive MAGI capture requires complete resource admission")
            self.rank_policy = RankPolicy.from_dict(rank_policy.to_dict())
            self.specs = {}
        else:
            self.specs = {key: specs for key in self.keys} if isinstance(specs, CodecSpec) else dict(specs)
            if set(self.specs) != set(self.keys) or not all(isinstance(v, CodecSpec) for v in self.specs.values()):
                raise ConfigurationError("MAGI requires a complete per-layer K/V profile")
            if len({self.specs[MatrixKey(i, "K")].kind == "raw" for i in range(geometry.layers)}) > 1:
                raise UnsupportedError("Initial MAGI profiles cannot mix native post-RoPE RAW K with pre-RoPE K layers")
        for key in self.keys:
            expected = "float32" if key.target == "K" and not self._source_is_raw(key) else self.native_dtype
            space = ("post_rope" if self._source_is_raw(key) else "pre_rope") if key.target == "K" else "none"
            context = _tensor_context(model_identity, key, 0, geometry, space, expected)
            if self.rank_policy is not None:
                self.rank_policy.validate_context(context)
        if any(not self._source_is_raw(MatrixKey(i, "K")) for i in range(geometry.layers)):
            if rotary_apply is None:
                raise ConfigurationError("Lossy K restoration requires the native rotary function")
        if type(layer_streaming) is not bool:
            raise ConfigurationError("layer_streaming must be boolean")
        if layer_streaming:
            if self.rank_policy is not None or self.reuse_threshold is not None or source_observer is not None:
                raise UnsupportedError(
                    "Layer streaming publishes static profiles only (no rank policy, reuse or source observer)"
                )
            if geometry.world_size > 1 and (stream_group is None or stream_group is self.transport.group):
                raise ConfigurationError("Layer streaming needs its own process group, distinct from the coordinator's")
            if type(stream_queue_depth) is not int or stream_queue_depth < 1:
                raise ConfigurationError("stream_queue_depth must be a positive integer")
            if stream_thread_init is not None and not callable(stream_thread_init):
                raise ConfigurationError("stream_thread_init must be callable")
            if stream_executor is not None and not self.transport.is_owner:
                raise ConfigurationError("Only the canonical owner may hold a streaming encode executor")
        if cache_mode == "sp_tiered":
            from ..sp_kv import SPTieredHistory

            if not isinstance(sp_history, SPTieredHistory):
                raise ConfigurationError("sp_tiered requires an SPTieredHistory")
            if sp_history.coordinator is not self.transport or sp_history.layout != geometry.layout:
                raise ConfigurationError("sp_history must share this adapter's coordinator and SP layout")
            if (self.rank_policy is not None or self.reuse_threshold is not None or source_observer is not None
                    or layer_streaming or hot_window_bytes or shared_history_decode):
                raise UnsupportedError(
                    "sp_tiered publishes static profiles only; it replaces layer streaming, hot windows "
                    "and shared owner decode"
                )
            if type(stream_queue_depth) is not int or stream_queue_depth < 1:
                raise ConfigurationError("stream_queue_depth must be a positive integer")
            if stream_executor is not None and not self.transport.is_owner:
                raise ConfigurationError("Only the canonical owner may hold a streaming encode executor")
        elif sp_history is not None:
            raise ConfigurationError("sp_history requires cache_mode='sp_tiered'")
        self.sp_history = sp_history
        self.layer_streaming = layer_streaming
        self.stream_group = stream_group
        self.stream_executor = stream_executor
        self.stream_queue_depth = stream_queue_depth
        self.stream_thread_init = stream_thread_init
        self.last_stream_stats = None
        self._bound_rank_policy = self.rank_policy
        self._phase = "empty"
        self._attached = False
        self._patches = []
        self._handles = []
        self._cursor = 0
        self._chunks = []
        self._ropes = {}
        self.canonical_source_hashes = {}
        self.observed_source_hashes = {}
        self._pending = None
        self._pending_rope = None
        self._active_chunk = None
        self._live_layer = None
        self._seen_layers = set()
        self._clean_timestep = None
        self._branch_created = False
        self._stats = {
            "completed_clean_chunks": 0, "layer_acquisitions": 0,
            "skipped_archive_chunk_visits": 0,
            "peak_native_layer_buffer_bytes": 0, "peak_local_receive_bytes": 0,
            "released_native_layer_buffers": 0, "canonical_owner": self.transport.owner_rank,
            "canonical_channels": geometry.channels,
            "local_kv_heads": geometry.layout.local_heads,
            "native_cache_dtype": self.native_dtype,
            "query_head_expansion": False,
            "raw_key_space": "post_rope", "lossy_key_space": "pre_rope",
            "native_cache_was_already_head_sharded": True,
            "saving_claim_from_duplicate_full_KV": False,
            "peak_transient_disk_staging_bytes": 0,
        }

    @property
    def cursor(self):
        return self._cursor

    @property
    def phase(self):
        return self._phase

    def _source_is_raw(self, key):
        self._check_rank_binding()
        return self.rank_policy is None and self.specs[key].kind == "raw"

    def _check_rank_binding(self):
        if getattr(self, "_bound_rank_policy", self.rank_policy) is not self.rank_policy:
            raise ConfigurationError("The MAGI rank-policy binding is immutable; construct a new adapter")

    def _capture_before_ulysses(self, layer, target, function):
        value, error = None, None
        try:
            value = function()
        except BaseException as exc:
            error = exc
        try:
            self.transport.checkpoint(f"magi_pre_ulysses_capture:{layer}:{target}", error=error)
        except BaseException as exc:
            if isinstance(exc, PeerFailure):
                raise
            raise PeerFailure("MAGI pre-Ulysses capture agreement failed; abort both ranks") from exc
        return value

    def _patch(self, obj, name, function):
        self._patches.append((obj, name, obj.__dict__.get(name, _MISSING)))
        setattr(obj, name, function)

    def attach(self):
        if self._attached:
            raise StateError("MAGI adapter is already attached")
        if getattr(self.model, "_reelcache_magi_adapter", None) is not None:
            raise StateError("Another adapter already owns this model instance")
        if self.cache.key_value_memory_dict:
            raise StateError("Attach at an empty cache boundary; do not silently abandon native KV")
        layers = list(self.model.videodit_blocks.layers)
        if len(layers) != self.geometry.layers:
            raise ConfigurationError("Native layer count differs from the model identity")
        try:
            self._patch(self.model, "_reelcache_magi_adapter", self)
            for index, layer in enumerate(layers):
                attention = layer.self_attention
                original_adjust = attention.adjust_key_and_value_for_inference
                original_k, original_v = attention.get_k, attention.get_v

                def adjust(key_and_value, inference_params, meta_args,
                           _layer=index, _attention=attention, _original=original_adjust):
                    if inference_params is not self.cache:
                        return _original(key_and_value, inference_params, meta_args)
                    return self._adjust(_layer, _attention, _original, key_and_value, meta_args)

                def get_k(mixed, cos, sin, _layer=index, _original=original_k):
                    if self._phase != "active_clean":
                        return _original(mixed, cos, sin)

                    def capture():
                        if not self._source_is_raw(MatrixKey(_layer, "K")) or self.source_observer is not None:
                            self._capture_rope(cos, sin)
                        return _original(mixed, cos, sin)

                    return self._capture_before_ulysses(_layer, "K", capture)

                def get_v(mixed, _layer=index, _original=original_v):
                    if self._phase != "active_clean":
                        return _original(mixed)

                    def capture():
                        value = _original(mixed)
                        key = MatrixKey(_layer, "V")
                        if not self._source_is_raw(key) or self.source_observer is not None:
                            shape = self.geometry.layout.shape(self.transport.rank, "rows")
                            destination = self._source_pending if self._source_is_raw(key) else self._pending
                            destination.put(key, value.reshape(shape))
                        return value

                    return self._capture_before_ulysses(_layer, "V", capture)

                def normalized_key(module, args, output, _layer=index):
                    key = MatrixKey(_layer, "K")
                    if self._phase == "active_clean" and (
                        not self._source_is_raw(key) or self.source_observer is not None
                    ):
                        import torch

                        rows = self.geometry.layout.shape(self.transport.rank, "rows")[0]
                        expected = (rows, 1, self.geometry.kv_heads, self.geometry.head_dim)
                        if tuple(output.shape) != expected or output.dtype != torch.float32:
                            raise UnsupportedError("Native normalized pre-RoPE K seam changed")
                        destination = self._source_pending if self._source_is_raw(key) else self._pending
                        destination.put(key, output.reshape(rows, self.geometry.channels))

                self._patch(attention, "adjust_key_and_value_for_inference", adjust)
                self._patch(attention, "get_k", get_k)
                self._patch(attention, "get_v", get_v)
                self._handles.append(attention.k_layernorm.register_forward_hook(normalized_key))
            self._attached = True
        except BaseException:
            self._unpatch()
            raise
        return self

    def _unpatch(self):
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        for obj, name, old in reversed(self._patches):
            if old is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, old)
        self._patches.clear()
        self._attached = False

    def detach(self):
        if self._phase.startswith("active_") or self._live_layer is not None:
            raise StateError("Cannot remove MAGI hooks during a forward")
        self._clear_hot_window()
        self.release_quality_history()
        self._unpatch()
        self._abort_stream_local()
        if self.sp_history is not None:
            self.sp_history.clear()
        if self._pending is not None:
            shutil.rmtree(self._pending.directory)
        self._pending = self._source_pending = self._pending_rope = None
        self._pending_source_provenance.clear()

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.detach()

    def _capture_rope(self, cos, sin):
        import torch

        rows = self.geometry.layout.shape(self.transport.rank, "rows")[0]
        if cos.shape != sin.shape or cos.ndim != 2 or cos.shape[0] != rows:
            raise UnsupportedError("Native per-token rotary geometry changed")
        if self._pending_rope is None:
            # Keep device copies; the host copy and the cross-layer equality check are deferred to
            # seal (one sync per chunk) so capture never blocks the host on queued GPU work.
            self._pending_rope = (cos.detach().clone(), sin.detach().clone())
            self._rope_mismatch = None
        else:
            first_cos, first_sin = self._pending_rope
            if first_cos.device != cos.device:
                raise UnsupportedError("Native rotary tables moved between devices within one chunk")
            differs = torch.ne(first_cos, cos).any() | torch.ne(first_sin, sin).any()
            self._rope_mismatch = differs if self._rope_mismatch is None else self._rope_mismatch | differs

    def _materialize_rope(self):
        """Host copy of the chunk's rotary tables after checking every layer saw the same ones."""
        if self._pending_rope is None:
            return None
        mismatch = getattr(self, "_rope_mismatch", None)
        if mismatch is not None and bool(mismatch):
            raise UnsupportedError("Layer-dependent rotary embeddings require a separate contract")
        self._rope_mismatch = None
        self._pending_rope = tuple(value.cpu() for value in self._pending_rope)
        return self._pending_rope

    def _hot_binding(self):
        return {
            "branch": self.branch_id, "cursor": self._cursor,
            "profile": self._requested_restore_profile(), "chunks": self._chunks,
            "rotary": {name: [(id(value), value._version) for value in pair] for name, pair in self._ropes.items()},
            "cap": self.hot_window_bytes, "headroom": self.hot_window_headroom_bytes,
            "require_full": self.hot_window_require_full, "verify_hits": self.hot_window_verify_hits,
            "shared_history_decode": self.shared_history_decode,
            "visible_history_only": self.visible_history_only,
        }

    def _begin_hot_window(self):
        import torch

        if not self.hot_window_bytes or not self._chunks:
            return
        error = None
        required = self._cursor * self.geometry.tokens * self.geometry.layout.local_channels * len(
            self.keys,
        ) * torch.empty((), dtype=getattr(torch, self.native_dtype)).element_size()
        try:
            if self._hot or self._hot_epoch is not None:
                raise StateError("A previous MAGI hot window survived its denoise epoch")
            if self.hot_window_require_full and required > self.hot_window_bytes:
                raise CapacityError("The entire requested decoded history exceeds the hot-window budget")
            admitted = min(required, self.hot_window_bytes)
            if self.device.type == "cuda":
                if not torch.cuda.is_initialized():
                    raise StateError("The external builder must initialize CUDA before hot-window admission")
                free, _ = torch.cuda.mem_get_info(self.device)
                before_free = free
                released_allocator = False
                if admitted + self.hot_window_headroom_bytes > free:
                    torch.cuda.empty_cache()
                    free, _ = torch.cuda.mem_get_info(self.device)
                    released_allocator = True
                self._hot_stats["last_admission"] = {
                    "physical_free_before_bytes": before_free, "physical_free_after_bytes": free,
                    "released_unused_allocator_cache": released_allocator,
                    "torch_allocated_bytes": torch.cuda.memory_allocated(self.device),
                    "torch_reserved_bytes": torch.cuda.memory_reserved(self.device),
                    "admitted_hot_bytes": admitted, "additional_headroom_bytes": self.hot_window_headroom_bytes,
                }
                if admitted + self.hot_window_headroom_bytes > free:
                    raise CapacityError("Insufficient physical free HBM for hot history plus reserved headroom")
        except Exception as exc:
            error = exc
        policies = self.transport.checkpoint("magi_hot_admission", error=error, details={
            "required": required, "cap": self.hot_window_bytes,
            "headroom": self.hot_window_headroom_bytes, "full": self.hot_window_require_full,
            "verify_hits": self.hot_window_verify_hits,
        })
        if any(policy != policies[0] for policy in policies):
            raise StateError("Ranks disagree on hot-window admission")

        def pin_head():
            with self.runtime.store.transaction():
                info = self.runtime.branch_info(self.branch_id)
                if info["token_end"] != self._cursor * self.geometry.tokens:
                    raise StateError("Hot history cursor differs from the committed branch")
                self._hot_pin = self.runtime.store.lease(info["head_id"], verify=False)
                return info["head_id"]

        try:
            head = self.transport.owner_call("magi_hot_pin", pin_head)
            head = self.transport.broadcast_json(head, operation="magi_hot_head")
            error = None
            try:
                # Tracked CPU copies make in-place RoPE mutation observable even
                # when the external model executes under inference_mode().
                with torch.inference_mode(False):
                    self._ropes = {name: tuple(value.detach().clone() for value in pair)
                                   for name, pair in self._ropes.items()}
                self._hot_epoch = {
                    "head": head, "binding": json.dumps(self._hot_binding(), sort_keys=True),
                    "rotary": {name: [_tensor_hash(value) for value in pair] for name, pair in self._ropes.items()},
                    "stream": torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None,
                }
            except Exception as exc:
                error = exc
            self.transport.checkpoint("magi_hot_bound", error=error)
            self._hot_verified = 0
            self._hot_stats.update(
                last_epoch_peak_bytes=0, last_epoch_hits=0, last_epoch_misses=0,
                last_epoch_required_bytes=required,
            )
            self._hot_stats["epochs"] += 1
        except BaseException:
            self._clear_hot_window()
            raise

    def _check_hot_binding(self):
        if self._hot_epoch is None:
            return
        if json.dumps(self._hot_binding(), sort_keys=True) != self._hot_epoch["binding"]:
            raise StateError("Native branch/profile/positions changed during the decoded hot epoch")
        if self.transport.is_owner and self.runtime.store.get_ref(
            f"branches/{self.branch_id}",
        ) != self._hot_epoch["head"]:
            raise StateError("Committed branch head changed during the decoded hot epoch")
        if self.device.type == "cuda":
            import torch

            if torch.cuda.current_stream(self.device) != self._hot_epoch["stream"]:
                raise UnsupportedError("A hot epoch must use one ordered native CUDA stream")

    def _clear_hot_window(self):
        try:
            if self._hot_event is not None:
                self._hot_event.synchronize()
        finally:
            self._hot.clear()
            self._hot_bytes = 0
            self._hot_epoch = None
            self._hot_event = None
            if self._hot_pin is not None:
                self._hot_pin.release()
                self._hot_pin = None

    def _admit_resources(self, chunks):
        if self.resource_budget is None:
            return
        if self.rank_policy is not None and (
            self.rank_policy.workspace_limit_bytes > self.resource_budget.workspace_host_bytes
        ):
            raise CapacityError("The adaptive policy workspace exceeds the admitted MAGI host workspace")
        self._admission = self.resource_budget.admit(
            self.geometry, chunks=chunks,
            factor_keys=any(not self._source_is_raw(MatrixKey(i, "K"))
                            for i in range(self.geometry.layers)),
            observer=self.source_observer is not None, quality_hot=self.cache_mode == "quality_hot",
        )

    def _quality_contract(self):
        value = json.dumps({
            "branch": self.branch_id, "profile": self._requested_restore_profile(),
            "chunks": self._chunks, "budget": self.quality_hot_bytes,
        }, sort_keys=True)
        return hashlib.sha256(value.encode()).hexdigest()

    def _quality_pin_head(self):
        if self._quality_pin is not None:
            self._quality_pin.release()
            self._quality_pin = None

        def pin():
            with self.runtime.store.transaction():
                info = self.runtime.branch_info(self.branch_id)
                self._quality_pin = self.runtime.store.lease(info["head_id"], verify=False)
                return info["head_id"]

        head = self.transport.owner_call("magi_quality_pin", pin) if self._chunks else None
        self._quality_head = self.transport.broadcast_json(head, operation="magi_quality_head")
        self._quality_binding = self._quality_contract()

    def _check_quality_binding(self):
        if self.cache_mode != "quality_hot":
            return
        if self._quality_binding != self._quality_contract():
            raise StateError("Quality-hot branch/profile/history changed without invalidation")
        if self.transport.is_owner and self._quality_head is not None and self.runtime.store.get_ref(
            f"branches/{self.branch_id}",
        ) != self._quality_head:
            raise StateError("Quality-hot committed branch head changed")

    def _quality_digest(self, layer, buffer, tokens):
        digest = _tensor_hash(buffer[:tokens])
        if self.quality_verify_history:
            self._quality_stats["protected_history_checks"] += 1
            expected = self._quality_hashes.get(layer)
            if expected is not None and digest != expected:
                raise IntegrityError("Native attention changed protected archived quality-hot history")
        return digest

    def _prepare_quality_history(self):
        import torch

        if self.cache_mode != "quality_hot":
            return
        required = ((self._cursor + 1) * self.geometry.tokens
                    * self.geometry.layout.local_channels * 4 * self.geometry.layers)
        error = None
        try:
            self._admit_resources(self._cursor + 1)
            if required > self.quality_hot_bytes:
                raise CapacityError("The entire native quality-hot history/current slot exceeds its budget")
            if self._quality_buffers:
                self._check_quality_binding()
            elif self.cache.key_value_memory_dict:
                raise StateError("Quality-hot acquisition found unowned native buffers")
            if self.device.type == "cuda":
                free, _ = torch.cuda.mem_get_info(self.device)
                existing = sum(value.numel() * value.element_size()
                               for value in self._quality_buffers.values())
                growth = self._admission["native_growth_copy_bytes"]
                if required - existing + growth + self._admission["native_duplicate_and_rope_bytes"] + (
                    self.resource_budget.headroom_bytes
                ) > free:
                    raise CapacityError("Insufficient free HBM for quality-hot allocation and temporaries")
        except Exception as exc:
            error = exc
        rows = self.transport.checkpoint("magi_quality_admission", error=error, details={
            "required": required, "budget": self.quality_hot_bytes,
            "cursor": self._cursor, "contract": self._quality_contract(),
        })
        if any(row != rows[0] for row in rows):
            raise StateError("Ranks disagree on quality-hot admission")
        try:
            if self._quality_binding is None:
                self._quality_pin_head()
            for layer, module in enumerate(self.model.videodit_blocks.layers):
                attention = module.self_attention
                old = self._quality_buffers.get(layer)
                error = None
                buffer = old
                try:
                    shape = ((self._cursor + 1) * self.geometry.tokens,
                             self.geometry.layout.local_heads, 2 * self.geometry.head_dim)
                    if old is None or old.shape[0] != shape[0]:
                        buffer = torch.empty(shape, dtype=getattr(torch, self.native_dtype), device=self.device)
                        if old is not None:
                            self._quality_digest(layer, old, self._cursor * self.geometry.tokens)
                            buffer[:self._cursor * self.geometry.tokens].copy_(old[:self._cursor * self.geometry.tokens])
                except Exception as exc:
                    error = exc
                self.transport.checkpoint(f"magi_quality_allocate:{layer}", error=error)
                if old is None:
                    for chunk in self._chunks:
                        for target, offset in (("K", 0), ("V", self.geometry.head_dim)):
                            with self._decoded_history(MatrixKey(layer, target), chunk) as value:
                                buffer[chunk["token_start"]:chunk["token_end"], :,
                                       offset:offset + self.geometry.head_dim].copy_(value)
                            self._quality_stats["packet_decodes_per_recipient"] += 1
                self._quality_buffers[layer] = buffer
                self.cache.key_value_memory_dict[attention.layer_number] = buffer
                self._quality_hashes[layer] = _tensor_hash(buffer[:self._cursor * self.geometry.tokens])
                del old, buffer
            self._quality_stats["peak_native_bytes"] = max(self._quality_stats["peak_native_bytes"], required)
        except BaseException:
            self.release_quality_history()
            self._phase = "failed"
            raise

    def _replace_quality_clean_slot(self, chunk):
        """Decode committed packets, never promote native clean scratch to history."""
        if self.cache_mode != "quality_hot":
            return
        for layer in range(self.geometry.layers):
            buffer = self._quality_buffers[layer]
            self._quality_digest(layer, buffer, chunk["token_start"])
            for target, offset in (("K", 0), ("V", self.geometry.head_dim)):
                with self._decoded_history(MatrixKey(layer, target), chunk) as value:
                    buffer[chunk["token_start"]:chunk["token_end"], :,
                           offset:offset + self.geometry.head_dim].copy_(value)
                self._quality_stats["packet_decodes_per_recipient"] += 1
                self._quality_stats["clean_slot_replacements"] += 1
            self._quality_hashes[layer] = _tensor_hash(buffer[:chunk["token_end"]])
        self._quality_pin_head()

    def release_quality_history(self):
        """Drop only owned dense views at a quiescent snapshot/fork/request boundary."""
        if self._phase.startswith("active_") or self._live_layer is not None:
            raise StateError("Cannot release quality-hot history during a native forward")
        if self.device.type == "cuda" and self._quality_buffers:
            import torch

            torch.cuda.synchronize(self.device)
        for layer, buffer in self._quality_buffers.items():
            attention = self.model.videodit_blocks.layers[layer].self_attention
            if self.cache.key_value_memory_dict.get(attention.layer_number) is buffer:
                self.cache.key_value_memory_dict.pop(attention.layer_number)
        self._quality_buffers.clear()
        self._quality_hashes.clear()
        self._quality_head = self._quality_binding = None
        if self._quality_pin is not None:
            self._quality_pin.release()
            self._quality_pin = None

    @contextmanager
    def execution(self, *, chunk_index, clean, clean_timestep=None):
        """A whole denoise loop is in-flight until a completed clean recapture.

        A near-clean timestep is not a clean publication boundary. Only an
        explicit native prefix-feature forward at 0.9999 may be sealed.
        """
        error = None
        previous_pending, previous_rope, previous_source = self._pending, self._pending_rope, self._source_pending
        try:
            if not self._attached or self._live_layer is not None:
                raise StateError("MAGI adapter is detached or already executing")
            if self._phase not in ("empty", "ready", "noisy_complete"):
                raise StateError(f"Unsupported native boundary: {self._phase}")
            if chunk_index != self._cursor:
                raise StateError("Only the next contiguous clean-history chunk can execute")
            if not clean and self._phase == "noisy_complete":
                raise StateError("Clean-recapture or restore before beginning another noisy chunk")
            if clean and clean_timestep != 0.9999:
                raise UnsupportedError("MAGI clean capture requires timestep 0.9999, not near-clean")
            self._requested_restore_profile()
            self._admit_resources(self._cursor + 1)
            if self.cache_mode == "quality_hot" and (
                (self._cursor + 1) * self.geometry.tokens * self.geometry.layout.local_channels
                * self.geometry.layers * 4 > self.quality_hot_bytes
            ):
                raise CapacityError("The entire native quality-hot history/current slot exceeds its budget")
            if clean:
                self._pending_source_provenance = {}
                directory = self.staging_dir / f"rank{self.transport.rank}" / uuid.uuid4().hex
                self._pending = _DiskMatrices(directory)
                self._source_pending = _DiskMatrices(directory / "original") if self.source_observer else None
                self._pending_rope = None
            self._seen_layers.clear()
        except Exception as exc:
            error = exc
        try:
            rows = self.transport.checkpoint(
                "magi_execution_begin", error=error,
                details={
                    "chunk_index": chunk_index, "clean": clean, "clean_timestep": clean_timestep,
                    "branch_id": self.branch_id, "model": self.identity.to_dict(),
                    "specs": {f"{key.layer}:{key.target}": spec.to_dict() for key, spec in self.specs.items()},
                    "rank_policy": self.rank_policy.fingerprint if self.rank_policy is not None else None,
                    "reuse_threshold": self.reuse_threshold,
                    "hot_window_bytes": self.hot_window_bytes,
                    "shared_history_decode": self.shared_history_decode,
                    "visible_history_only": self.visible_history_only,
                    "cache_mode": self.cache_mode, "quality_hot_bytes": self.quality_hot_bytes,
                    "source_context": self.source_context, "observe_source": self.source_observer is not None,
                    "resource_budget": asdict(self.resource_budget) if self.resource_budget else None,
                },
            )
            if any(row != rows[0] for row in rows):
                raise StateError("Ranks disagree on the native boundary, branch, model, or profile")
        except BaseException:
            if self._pending is not previous_pending and self._pending is not None:
                shutil.rmtree(self._pending.directory)
            self._pending, self._pending_rope = previous_pending, previous_rope
            self._source_pending = previous_source
            raise
        self._prepare_quality_history()
        if clean and (self.layer_streaming or self.cache_mode == "sp_tiered"):
            if self.sp_history is not None:
                self.sp_history.discard_retained()
            try:
                self._start_stream()
            except BaseException:
                if self._pending is not None and self._pending is not previous_pending:
                    shutil.rmtree(self._pending.directory, ignore_errors=True)
                self._pending, self._pending_rope, self._source_pending = (
                    previous_pending, previous_rope, previous_source,
                )
                raise
        if not clean:
            self._begin_hot_window()
        self._active_chunk = chunk_index
        self._clean_timestep = clean_timestep
        self._checked_timestep = None
        self._phase = "active_clean" if clean else "active_noisy"
        failure = None
        try:
            yield self
            if clean and (self._seen_layers != set(range(self.geometry.layers))
                          or set(self._pending) != set(self.keys)):
                raise StateError("A clean forward did not complete every layer's K and V")
        except BaseException as exc:
            failure = exc
        finally:
            self._active_chunk = None
            try:
                self._clear_hot_window()
            except BaseException as exc:
                failure = failure or exc
        if isinstance(failure, PeerFailure) or getattr(self.transport, "_broken", False):
            self._phase = "failed"
            self.release_quality_history()
            self._abort_stream_local()
            raise failure or PeerFailure("MAGI transport is poisoned; abort both ranks")
        try:
            self.transport.checkpoint("magi_execution_complete", error=failure)
        except BaseException:
            self._phase = "failed"
            self.release_quality_history()
            self._abort_stream_local()
            raise
        self._phase = "clean_unsealed" if clean else "noisy_complete"

    def _stream_contexts(self):
        contexts = {}
        for key in self.keys:
            raw = self._source_is_raw(key)
            dtype = "float32" if key.target == "K" and not raw else self.native_dtype
            space = ("post_rope" if raw else "pre_rope") if key.target == "K" else "none"
            contexts[key] = _tensor_context(
                self.identity, key, self._cursor * self.geometry.tokens, self.geometry, space, dtype,
            )
        return contexts

    def _start_stream(self):
        """Open the owner's streaming chunk and start this rank's layer pipeline (collective)."""
        contexts = self._stream_contexts()
        if not self._branch_created:
            self.transport.owner_call("magi_create_branch", lambda: self.runtime.create_branch(self.branch_id))
            self._branch_created = True
        writer = self.transport.owner_call("magi_stream_open", lambda: self.runtime.open_chunk(
            self.branch_id, str(self._cursor), contexts, self.specs, executor=self.stream_executor,
        ))
        error = None
        try:
            directory = self._pending.directory
            kind = _SPCapture if self.cache_mode == "sp_tiered" else _LayerStream
            self._pending = kind(self, directory, contexts, writer, depth=self.stream_queue_depth)
        except BaseException as exc:
            error = exc
            if writer is not None:
                writer.abort()
        self.transport.checkpoint("magi_stream_started", error=error)

    def _drain_stream(self, stream):
        """Finish the layer pipeline after the clean forward; agree on its outcome on every rank."""
        if self.sp_history is not None:
            self.sp_history.drain_gathers()
        outcome, error = stream.finish()
        self.last_stream_stats = dict(stream.stats, outcome=outcome, error=error)
        failure = None if outcome == "complete" else (error or f"layer stream {outcome}")
        if failure is not None and stream.writer is not None:
            stream.writer.abort()
        try:
            self.transport.checkpoint("magi_stream_drained", error=failure)
        except BaseException:
            if stream.writer is not None:
                stream.writer.abort()
            if self.sp_history is not None:
                self.sp_history.discard_retained()
            self._phase = "failed"
            raise
        return dict(stream.contexts)

    def _abort_stream_local(self):
        """Stop a pending layer stream and drop its unpublished packets (no coordinator collectives)."""
        if self.sp_history is not None:
            self.sp_history.discard_retained()
        stream = self._pending
        if not isinstance(stream, _LayerStream):
            return
        try:
            stream.finish(abort=True)
        finally:
            if stream.writer is not None:
                stream.writer.abort()
            self.last_stream_stats = dict(stream.stats, outcome=stream.outcome, error=stream.error)

    def _validate_meta(self, key_and_value, meta):
        import torch

        if self._phase not in ("active_clean", "active_noisy"):
            raise StateError("Guard native forwards with adapter.execution()")
        if self._live_layer is not None:
            raise UnsupportedError("Pipelined or concurrent native layer acquisition is unsupported")
        if any(getattr(meta, name, False) for name in
               ("compress_kv", "save_kvcache_every_forward", "fwd_extra_1st_chunk", "enable_cuda_graph")):
            raise UnsupportedError("Only the native non-pipelined full-cache seam is supported")
        clean = self._phase == "active_clean"
        near = bool(getattr(meta, "distill_nearly_clean_chunk", False))
        if bool(meta.extract_prefix_video_feature) != clean or (clean and near):
            raise StateError("Native clean-feature flag differs from the adapter boundary")
        if clean:
            timestep = getattr(meta, "timestep", None)
            if not isinstance(timestep, torch.Tensor) or tuple(timestep.shape) != (1, 1):
                raise StateError("The actual native clean timestep is not 0.9999")
            # Checked once per forward (meta object/tensor version): a device tensor forces a host sync.
            token = (id(meta), id(timestep), timestep._version)
            if getattr(self, "_checked_timestep", None) != token:
                if not torch.equal(timestep, torch.full_like(timestep, 0.9999)):
                    raise StateError("The actual native clean timestep is not 0.9999")
                self._checked_timestep = token
        if (getattr(meta, "H", self.geometry.height) != self.geometry.height
                or getattr(meta, "W", self.geometry.width) != self.geometry.width
                or getattr(meta, "cp_pad_size", 0) != 0):
            raise UnsupportedError("Native spatial geometry or Ulysses padding differs from the supported layout")
        layout = KVLayout(self.geometry.tokens * (1 + int(near)), self.geometry.kv_heads,
                          self.geometry.head_dim, self.geometry.world_size)
        expected_splits = [layout.rows(rank).stop - layout.rows(rank).start
                           for rank in range(self.geometry.world_size)]
        if getattr(meta, "cp_split_sizes", expected_splits) != expected_splits:
            raise UnsupportedError("Native Ulysses row split differs from global-token ownership")
        if meta.slice_point != self._cursor or meta.clip_token_nums != self.geometry.tokens:
            raise UnsupportedError("Native token positions differ from the committed cursor")
        if self.cache.max_sequence_length < (self._cursor + 1) * self.geometry.tokens:
            raise CapacityError("The serial continuation exceeds the native configured cache capacity")
        if clean and not self.cache.update_kv_cache:
            raise StateError("A clean capture must enable native KV publication")
        if getattr(meta, "start_chunk_id", self._cursor) != self._cursor:
            raise UnsupportedError("Native start_chunk_id differs from the clean cursor")
        if getattr(meta, "end_chunk_id", self._cursor + 1 + near) != self._cursor + 1 + near:
            raise UnsupportedError("Native end_chunk_id does not preserve the near-clean duplicate")
        if meta.denoising_range_num != 1 + int(near):
            raise UnsupportedError("Only one noisy chunk plus its native near-clean duplicate is supported")
        expected = (self.geometry.tokens * (1 + int(near)),
                    self.geometry.layout.local_heads, 2 * self.geometry.head_dim)
        if tuple(key_and_value.shape) != expected:
            raise UnsupportedError("Post-Ulysses KV is not full tokens with local GQA heads")
        if str(key_and_value.dtype).split(".")[-1] != self.native_dtype:
            raise UnsupportedError("Native KV precision differs from the model identity")

    @contextmanager
    def _decoded_history(self, key, chunk):
        import torch

        pre_rope = key.target == "K" and chunk["spaces"][str(key.layer)] == "pre_rope"
        dtype = torch.float32 if pre_rope else getattr(torch, self.native_dtype)
        start, end = chunk["token_start"], chunk["token_end"]

        with ExitStack() as stack:
            grouped = None
            groups = [self.geometry.layout.columns(rank) for rank in range(self.transport.world_size)]
            if self.shared_history_decode:
                grouped = self.transport.owner_call(
                    f"magi_grouped_history:{chunk['chunk_id']}:{key.layer}:{key.target}",
                    lambda: stack.enter_context(self.runtime.acquire_history(
                        self.branch_id, key, token_start=start, token_end=end,
                        device="cpu", dtype=dtype, column_groups=groups,
                    )),
                )

            @contextmanager
            def load(columns):
                if grouped is not None:
                    yield grouped.tensor[groups.index(columns)]
                else:
                    with self.runtime.acquire_history(
                        self.branch_id, key, token_start=start, token_end=end,
                        device="cpu", dtype=dtype, columns=columns,
                    ) as lease:
                        yield lease.tensor

            with self.transport.acquire_columns(
                self.geometry.layout, load, dtype=dtype,
                operation=f"magi_history:{chunk['chunk_id']}:{key.layer}:{key.target}",
            ) as matrix:
                nbytes = matrix.numel() * matrix.element_size()
                self._stats["peak_local_receive_bytes"] = max(self._stats["peak_local_receive_bytes"], nbytes)
                with span("native.h2d", nbytes=nbytes):
                    value = matrix.to(self.device).reshape(
                        1, self.geometry.tokens, self.geometry.layout.local_heads, self.geometry.head_dim,
                    )
                if pre_rope:
                    cos, sin = self._ropes[chunk["chunk_id"]]
                    with span("native.rope"):
                        value = self.rotary_apply(value, cos.to(self.device), sin.to(self.device))
                value = value.to(getattr(torch, self.native_dtype)).reshape(
                    self.geometry.tokens, self.geometry.layout.local_heads, self.geometry.head_dim,
                )
                yield value

    @contextmanager
    def _history(self, key, chunk):
        import torch

        if self._hot_epoch is None:
            with self._decoded_history(key, chunk) as value:
                yield value
            return
        columns = self.geometry.layout.columns(self.transport.rank)
        token = (
            self._hot_epoch["head"], key, chunk["chunk_id"], chunk["token_start"], chunk["token_end"],
            columns.start, columns.stop, self.native_dtype, str(self.device),
            tuple(self._hot_epoch["rotary"].get(chunk["chunk_id"], ())),
        )
        cached = self._hot.get(token)
        count = self.geometry.tokens * self.geometry.layout.local_channels * torch.empty(
            (), dtype=getattr(torch, self.native_dtype),
        ).element_size()
        admit = cached is None and self._hot_bytes + count <= self.hot_window_bytes
        states = self.transport.checkpoint(f"magi_hot:{chunk['chunk_id']}:{key.layer}:{key.target}",
                                           details={"hit": cached is not None, "admit": admit})
        if any(state != states[0] for state in states):
            raise StateError("Ranks disagree on hot-window membership")
        if cached is not None:
            self._hot_stats["hits"] += 1
            self._hot_stats["last_epoch_hits"] += 1
            if self._hot_verified < self.hot_window_verify_hits:
                error = None
                try:
                    with self._decoded_history(key, chunk) as cold:
                        if not torch.equal(cold, cached):
                            raise IntegrityError("Attention-ready hot history differs from the same decoded packet")
                except Exception as exc:
                    error = exc
                self.transport.checkpoint("magi_hot_same_representation", error=error)
                self._hot_verified += 1
                self._hot_stats["verified_hits"] += 1
            yield cached
            return
        self._hot_stats["misses"] += 1
        self._hot_stats["last_epoch_misses"] += 1
        with self._decoded_history(key, chunk) as value:
            if not admit:
                self._hot_stats["bypasses"] += 1
                yield value
                return
            owned, error = None, None
            try:
                owned = value.detach().clone(memory_format=torch.contiguous_format)
                actual = owned.untyped_storage().nbytes()
                if actual != count or self._hot_bytes + actual > self.hot_window_bytes:
                    raise CapacityError("Actual attention-ready storage exceeds the admitted hot budget")
            except Exception as exc:
                error = exc
            self.transport.checkpoint("magi_hot_materialized", error=error)
            self._hot[token] = owned
            self._hot_bytes += actual
            self._hot_stats["peak_bytes"] = max(self._hot_stats["peak_bytes"], self._hot_bytes)
            self._hot_stats["last_epoch_peak_bytes"] = max(
                self._hot_stats["last_epoch_peak_bytes"], self._hot_bytes,
            )
            if (self.device.type == "cuda"
                    and self._hot_bytes == self._hot_stats["last_epoch_required_bytes"]):
                free, total = torch.cuda.mem_get_info(self.device)
                self._hot_stats["full_window_memory"] = {
                    "torch_allocated_bytes": torch.cuda.memory_allocated(self.device),
                    "torch_reserved_bytes": torch.cuda.memory_reserved(self.device),
                    "device_used_bytes": total - free,
                    "scope": "actual allocation at full window; not device-wide peak attribution",
                }
            yield owned

    def _adjust(self, layer, attention, original, key_and_value, meta):
        result, error = None, None
        try:
            result = self._adjust_local(layer, attention, original, key_and_value, meta)
        except BaseException as exc:
            error = exc
        if isinstance(error, PeerFailure) or getattr(self.transport, "_broken", False):
            raise error or PeerFailure("MAGI transport failed during native adjustment")
        try:
            self.transport.checkpoint(f"magi_post_ulysses_adjust:{layer}", error=error)
        except BaseException as exc:
            if error is not None:
                raise exc from error
            raise
        return result

    def _adjust_local(self, layer, attention, original, key_and_value, meta):
        import torch

        error = None
        visible_chunks, key_ranges = self._chunks, None
        try:
            self._check_rank_binding()
            self._validate_meta(key_and_value, meta)
            self._check_hot_binding()
            if self.cache_mode == "quality_hot":
                self._check_quality_binding()
                if self.cache.key_value_memory_dict.get(attention.layer_number) is not self._quality_buffers.get(layer):
                    raise StateError("Quality-hot native cache ownership changed")
            elif attention.layer_number in self.cache.key_value_memory_dict:
                raise StateError("Unexpected native raw buffer survived an earlier layer")
            if self.visible_history_only:
                visible_chunks, key_ranges = self._visible_history(meta)
        except Exception as exc:
            error = exc
        rows = self.transport.checkpoint(
            f"magi_layer:{layer}", error=error,
            details={"visible_history_only": self.visible_history_only, "key_ranges": key_ranges},
        )
        if any(row != rows[0] for row in rows):
            raise PeerFailure("Ranks disagree on native history visibility")
        self._live_layer = layer
        self._stats["layer_acquisitions"] += 1
        buffer = None
        try:
            shape = ((self._cursor + 1) * self.geometry.tokens,
                     self.geometry.layout.local_heads, 2 * self.geometry.head_dim)
            nbytes = shape[0] * shape[1] * shape[2] * key_and_value.element_size()
            error = None
            try:
                if nbytes > self.max_layer_buffer_bytes:
                    raise CapacityError("MAGI per-layer native history buffer budget exceeded")
                if self.cache_mode == "quality_hot":
                    buffer = self._quality_buffers[layer]
                    if tuple(buffer.shape) != shape or buffer.dtype != key_and_value.dtype:
                        raise StateError("Quality-hot native buffer shape/precision changed")
                    if self.quality_verify_history:
                        self._quality_digest(layer, buffer, self._cursor * self.geometry.tokens)
                else:
                    allocate = torch.zeros if self.visible_history_only else torch.empty
                    buffer = allocate(shape, dtype=key_and_value.dtype, device=key_and_value.device)
            except Exception as exc:
                error = exc
            self.transport.checkpoint(f"magi_layer_allocated:{layer}", error=error)
            self._stats["peak_native_layer_buffer_bytes"] = max(
                self._stats["peak_native_layer_buffer_bytes"], nbytes,
            )
            if self.cache_mode in ("runtime", "sp_tiered"):
                self._stats["skipped_archive_chunk_visits"] += len(self._chunks) - len(visible_chunks)
            if self.cache_mode == "sp_tiered" and visible_chunks:
                loaded = self.sp_history.load_layer(
                    layer, visible_chunks,
                    # Look one layer ahead; within a denoising loop, wrap the last layer to layer 0
                    # of the next step (or of the clean forward), which reads the same history.
                    prefetch_layer=(layer + 1 if layer + 1 < self.geometry.layers
                                    else 0 if self._phase == "active_noisy" else None),
                    packet_source=(lambda chunk, key: self.runtime.history_packet(
                        self.branch_id, key, token_start=chunk["token_start"], token_end=chunk["token_end"],
                    )) if self.transport.is_owner else None,
                    rope_for=lambda chunk: self._ropes[chunk["chunk_id"]],
                )
                head_dim = self.geometry.head_dim
                for chunk, (k_local, v_local) in zip(visible_chunks, loaded):
                    with span("native.insert", nbytes=2 * k_local.numel() * k_local.element_size()):
                        buffer[chunk["token_start"]:chunk["token_end"], :, :head_dim].copy_(k_local)
                        buffer[chunk["token_start"]:chunk["token_end"], :, head_dim:].copy_(v_local)
                del loaded
            for chunk in visible_chunks if self.cache_mode == "runtime" else ():
                for target, offset in (("K", 0), ("V", self.geometry.head_dim)):
                    with self._history(MatrixKey(layer, target), chunk) as value:
                        with span("native.insert", nbytes=value.numel() * value.element_size()):
                            buffer[chunk["token_start"]:chunk["token_end"], :,
                                   offset:offset + self.geometry.head_dim].copy_(value)
                    del value
            self.cache.key_value_memory_dict[attention.layer_number] = buffer
            result = original(key_and_value, self.cache, meta)
            if self.cache_mode == "quality_hot":
                if self.cache.key_value_memory_dict.get(attention.layer_number) is not buffer:
                    raise StateError("Native attention replaced an owned quality-hot buffer")
                if self.quality_verify_history:
                    self._quality_digest(layer, buffer, self._cursor * self.geometry.tokens)
            if self._phase == "active_clean":
                if self.cache_mode == "sp_tiered":
                    current = key_and_value[:self.geometry.tokens]
                    self.sp_history.retain(layer, current[:, :, :self.geometry.head_dim],
                                           current[:, :, self.geometry.head_dim:])
                for target, offset in (("K", 0), ("V", self.geometry.head_dim)):
                    key = MatrixKey(layer, target)
                    if self._source_is_raw(key):
                        value = key_and_value[:self.geometry.tokens, :, offset:offset + self.geometry.head_dim]
                        self._pending.put(key, value.reshape(self.geometry.tokens, -1))
                if layer in self._seen_layers:
                    raise StateError("A clean boundary traversed a native layer twice")
                self._seen_layers.add(layer)
            return result
        finally:
            self._live_layer = None
            if (self.cache_mode in ("runtime", "sp_tiered")
                    and self.cache.key_value_memory_dict.get(attention.layer_number) is buffer):
                self.cache.key_value_memory_dict.pop(attention.layer_number, None)
            del buffer
            if self.cache_mode in ("runtime", "sp_tiered"):
                self._stats["released_native_layer_buffers"] += 1
            if self._hot and self.device.type == "cuda":
                if self._hot_event is None:
                    self._hot_event = torch.cuda.Event()
                self._hot_event.record(torch.cuda.current_stream(self.device))

    def _visible_history(self, meta):
        params = getattr(meta, "core_attn_params", None)
        ranges = params.get("np_k_range") if isinstance(params, Mapping) else getattr(params, "np_k_range", None)
        if ranges is None:
            raise UnsupportedError("visible_history_only requires actual native np_k_range")
        maximum = (self._cursor + meta.denoising_range_num) * self.geometry.tokens
        try:
            if len(ranges) != meta.denoising_range_num:
                raise ConfigurationError("native key ranges must cover every query range")
            checked = []
            for pair in ranges:
                if len(pair) != 2 or any(isinstance(v, bool) or not isinstance(v, Integral) for v in pair):
                    raise ConfigurationError("native key ranges must be integer half-open intervals")
                start, stop = map(int, pair)
                if not 0 <= start < stop <= maximum:
                    raise ConfigurationError("native key range exceeds the absolute KV positions")
                checked.append([start, stop])
        except TypeError as exc:
            raise ConfigurationError("invalid native key range geometry") from exc
        return [
            chunk for chunk in self._chunks
            if any(chunk["token_start"] < stop and start < chunk["token_end"] for start, stop in checked)
        ], checked

    def _observe_clean_sources(self, canonical, chunk_id, global_rope):
        if self.source_observer is None:
            return
        hashes = {}
        position_hashes = [_tensor_hash(value) for value in global_rope]
        for key in self.keys:
            is_raw = self._source_is_raw(key)
            staged = self._source_pending if is_raw else self._pending
            local_metadata, local, error = None, None, None
            try:
                entry = staged.entries[key]
                local_metadata = {
                    "rank": self.transport.rank, "row_start": self.geometry.layout.rows(self.transport.rank).start,
                    "row_end": self.geometry.layout.rows(self.transport.rank).stop,
                    "channels": self.geometry.channels, "dtype": entry["dtype"], "sha256": entry["sha256"],
                    "hash_format": _tensor_hash_format(entry["dtype"]),
                }
                if is_raw:
                    local = staged[key]
            except Exception as exc:
                error = exc
            shards = self.transport.checkpoint(f"magi_original_shards:{key.layer}:{key.target}",
                                               error=error, details=local_metadata)
            full = self.transport.gather_matrix(
                local, self.geometry.layout, source_layout="rows",
                operation=f"magi_original_rows:{key.layer}:{key.target}",
            ) if is_raw else None

            def observe():
                matrix = full if is_raw else canonical[key]
                dtype = "float32" if key.target == "K" else self.native_dtype
                context = _tensor_context(
                    self.identity, key, self._cursor * self.geometry.tokens, self.geometry,
                    "pre_rope" if key.target == "K" else "none", dtype,
                )
                if tuple(matrix.shape) != context.shape or canonical_dtype(matrix.dtype) != dtype:
                    raise IntegrityError("Original clean observer source changed shape or dtype")
                if len(shards) != self.geometry.world_size:
                    raise IntegrityError("Original MAGI row-shard coverage is incomplete")
                for rank, shard in enumerate(shards):
                    rows = self.geometry.layout.rows(rank)
                    if (shard["rank"] != rank or shard["row_start"] != rows.start
                            or shard["row_end"] != rows.stop or shard["channels"] != self.geometry.channels
                            or shard["dtype"] != dtype or shard["hash_format"] != _tensor_hash_format(dtype)):
                        raise IntegrityError("Original MAGI row-shard coverage, channels or dtype disagrees")
                digest = _tensor_hash(matrix)
                description = {
                    **self.source_context, "schema": "reelcache.magi_original_clean.v1",
                    "tensor_context": context.to_dict(), "chunk_id": chunk_id,
                    "boundary": self._cursor, "layer": key.layer, "target": key.target,
                    "branch_id": self.branch_id, "history": json.loads(json.dumps(self._chunks)),
                    "history_head": self.runtime.branch_info(self.branch_id)["head_id"]
                    if self._branch_created else None,
                    "clean_timestep": 0.9999, "source_sha256": digest, "actual_dtype": dtype,
                    "source_hash_format": NATIVE_TENSOR_HASH_FORMAT,
                    "source_hash_descriptor": _tensor_hash_format(dtype),
                    "source_hash_byte_order": sys.byteorder,
                    "row_shard_hash_format": NATIVE_TENSOR_HASH_FORMAT,
                    "row_shard_hash_byte_order": sys.byteorder,
                    "shape": list(matrix.shape), "source_layout": "owner_assembled_original_token_rows",
                    "row_shards": shards, "global_channels": self.geometry.channels,
                    "token_positions": [context.token_start, context.token_end],
                    "rotary_sha256": position_hashes, "inverse_rope_source": False,
                    "rotary_sources": [
                        {"shape": list(value.shape), "dtype": canonical_dtype(value.dtype),
                         "sha256": digest, "hash_format": _tensor_hash_format(value.dtype)}
                        for value, digest in zip(global_rope, position_hashes)
                    ],
                }
                provenance = self.source_observer(description, matrix)
                if _tensor_hash(matrix) != digest:
                    raise IntegrityError("A clean-source observer mutated its original input")
                if self.rank_policy is not None and provenance is not None:
                    if not isinstance(provenance, Mapping):
                        raise ConfigurationError("An adaptive observer must return portable source provenance or None")
                    self._pending_source_provenance[key] = json.loads(json.dumps(
                        dict(provenance), sort_keys=True, allow_nan=False,
                    ))
                return digest

            digest = self.transport.owner_call(f"magi_observe:{key.layer}:{key.target}", observe)
            hashes[f"{key.layer}:{key.target}"] = digest
            del local, full
        self.transport.owner_call(
            "magi_source_provenance_complete",
            lambda: self._validate_pending_source_provenance(),
        )
        self.observed_source_hashes[chunk_id] = self.transport.broadcast_json(
            hashes if self.transport.is_owner else None, operation="magi_observed_original_hashes",
        )

    def _validate_pending_source_provenance(self):
        if self._pending_source_provenance and set(self._pending_source_provenance) != set(self.keys):
            raise ConfigurationError("Study source provenance must cover every matrix in the clean commit")

    def seal_chunk(self, chunk_id=None, *, reuse_threshold=_MISSING):
        """Publish atomically, then discard native transient staging.

        The configured reuse threshold is the default. An explicit per-call
        override, including None, applies only to this commit and does not
        replace the configured policy saved in snapshots.
        """
        error = None
        selected_reuse = None
        try:
            if self._phase != "clean_unsealed" or self._live_layer is not None:
                raise StateError("Publication requires a completed quiescent clean forward")
            self._requested_restore_profile()
            selected_reuse = _checked_reuse_threshold(
                self.reuse_threshold if reuse_threshold is _MISSING else reuse_threshold,
            )
            if self.rank_policy is not None and selected_reuse is not None:
                raise ConfigurationError("An adaptive policy owns reuse and must advance atomically in core")
            if self.layer_streaming and selected_reuse is not None:
                raise UnsupportedError("Layer-streamed publication does not support basis reuse")
            self._materialize_rope()
        except Exception as exc:
            error = exc
        policies = self.transport.checkpoint("magi_seal", error=error, details=selected_reuse)
        if any(policy != policies[0] for policy in policies):
            raise StateError("Ranks disagree on the publication reuse policy")
        self._clear_hot_window()
        chunk_id = str(self._cursor) if chunk_id is None else str(chunk_id)
        stream = self._pending if isinstance(self._pending, _LayerStream) else None
        canonical = None if stream is not None else self.transport.owner_call(
            "magi_canonical_staging",
            lambda: _DiskMatrices(self._pending.directory / f"canonical-{uuid.uuid4().hex}"),
        )
        contexts = {}
        try:
            if stream is not None:
                contexts = self._drain_stream(stream)
            for key in (self.keys if stream is None else ()):
                local, error = None, None
                try:
                    local = self._pending[key]
                except Exception as exc:
                    error = exc
                self.transport.checkpoint(f"magi_stage:{key.layer}:{key.target}", error=error)
                source_layout = "heads" if self._source_is_raw(key) else "rows"
                full = self.transport.gather_matrix(
                    local, self.geometry.layout, source_layout=source_layout,
                    operation=f"magi_publish:{key.layer}:{key.target}",
                )
                self.transport.owner_call("magi_stage_canonical", lambda: canonical.put(key, full))
                del local, full
                space = ("post_rope" if self._source_is_raw(key) else "pre_rope") if key.target == "K" else "none"
                contexts[key] = _tensor_context(
                    self.identity, key, self._cursor * self.geometry.tokens,
                    self.geometry, space, self._pending.entries[key]["dtype"],
                )
            global_rope = []
            if self._pending_rope is not None:
                for name, local in zip(("cos", "sin"), self._pending_rope):
                    rope_layout = KVLayout(
                        self.geometry.tokens, self.geometry.world_size,
                        local.shape[-1] // self.geometry.world_size, self.geometry.world_size,
                    )
                    full = self.transport.gather_matrix(
                        local, rope_layout, source_layout="rows", operation=f"magi_rope:{name}",
                    )
                    mine = full
                    for rank in range(self.transport.world_size):
                        received = self.transport.transfer_tensor(
                            full, source=self.transport.owner_rank, destination=rank,
                            operation=f"magi_rope:{name}:{rank}",
                        )
                        if self.transport.rank == rank:
                            mine = received
                    global_rope.append(mine)
            staged_bytes = self._pending.nbytes
            if self._source_pending is not None:
                staged_bytes += self._source_pending.nbytes
            if canonical is not None:
                staged_bytes += canonical.nbytes
            self._stats["peak_transient_disk_staging_bytes"] = max(
                self._stats["peak_transient_disk_staging_bytes"], staged_bytes,
            )
            if stream is not None:
                self._stats["peak_transient_disk_staging_bytes"] = max(
                    self._stats["peak_transient_disk_staging_bytes"], stream.stats["peak_queued_bytes"],
                )
            self._observe_clean_sources(canonical, chunk_id, global_rope)
            source_hashes = self.transport.broadcast_json(
                {f"{key.layer}:{key.target}": (stream.hashes[key] if stream is not None
                                                else canonical.entries[key]["sha256"]) for key in self.keys}
                if self.transport.is_owner else None,
                operation=f"magi_source_identity:{chunk_id}",
            )
            if not self._branch_created:
                self.transport.owner_call("magi_create_branch", lambda: self.runtime.create_branch(self.branch_id))
                self._branch_created = True
            def publish():
                if stream is not None:
                    return stream.writer.commit(chunk_id)
                if self.rank_policy is not None:
                    return self.runtime.seal_chunk(
                        self.branch_id, chunk_id, canonical, contexts, clean=True, rank_policy=self.rank_policy,
                        source_provenance=self._pending_source_provenance or None,
                    )
                return self.runtime.seal_chunk(
                    self.branch_id, chunk_id, canonical, contexts, self.specs,
                    clean=True, reuse_threshold=selected_reuse,
                )

            try:
                result = self.transport.owner_call("magi_atomic_publication", publish)
            except BaseException:
                if stream is not None:
                    self._phase = "failed"
                if self.sp_history is not None:
                    self.sp_history.discard_retained()
                raise
            if self.sp_history is not None:
                self.sp_history.commit_retained(chunk_id, self._cursor * self.geometry.tokens)
            if global_rope:
                self._ropes[chunk_id] = tuple(global_rope)
            self.canonical_source_hashes[chunk_id] = source_hashes
            chunk = {
                "chunk_id": chunk_id, "token_start": self._cursor * self.geometry.tokens,
                "token_end": (self._cursor + 1) * self.geometry.tokens,
                "spaces": {str(layer): contexts[MatrixKey(layer, "K")].key_space
                           for layer in range(self.geometry.layers)},
            }
            self._chunks.append(chunk)
            try:
                self._replace_quality_clean_slot(chunk)
            except BaseException:
                self._phase = "failed"
                self.release_quality_history()
                raise
            self._cursor += 1
            self._stats["completed_clean_chunks"] += 1
            self._phase = "ready"
            shutil.rmtree(self._pending.directory)
            self._pending = self._source_pending = self._pending_rope = None
            self._pending_source_provenance = {}
            return result
        finally:
            if canonical is not None and canonical.directory.exists():
                shutil.rmtree(canonical.directory)

    def memory_report(self):
        import torch

        tensors = list(self.cache.key_value_memory_dict.values())
        stores = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes() for tensor in tensors}
        row = {
            **self._stats, "mode": self.mode, "phase": self._phase,
            "live_native_cache_storage_bytes": sum(stores.values()),
            "native_cache_entries": len(tensors),
            "transient_disk_staging_bytes": self._pending.nbytes if self._pending is not None else 0,
            "retained_cpu_rotary_bytes": sum(t.numel() * t.element_size() for pair in self._ropes.values() for t in pair),
            "transport_payload": "owner-reconstructed local KV columns; not compressed coefficient wire transfer",
            "shared_history_decode": {
                "enabled": self.shared_history_decode,
                "scope": "one target/layer/chunk; owner charges all recipient outputs until consumption",
                "native_parity_qualified": False,
            },
            "visible_history_only": {
                "enabled": self.visible_history_only,
                "scope": "union of native np_k_range intervals, without index rebasing",
                "unused_history": "zero initialized when enabled; not byte-identical unused cache storage",
                "native_parity_qualified": False,
            },
            "wire": self.transport.counters.to_dict(),
            "decoded_hot_window": {
                **self._hot_stats, "resident_bytes": self._hot_bytes, "entries": len(self._hot),
                "cap_bytes": self.hot_window_bytes, "headroom_bytes": self.hot_window_headroom_bytes,
                "epoch_active": self._hot_epoch is not None,
                "representation": "attention-ready native dtype, exact packet decode plus original-position RoPE",
                "scope": "one denoise epoch; deterministic admission, no cross-epoch reuse",
            },
            "quality_hot": {
                **self._quality_stats, "enabled": self.cache_mode == "quality_hot",
                "cap_bytes": self.quality_hot_bytes, "entries": len(self._quality_buffers),
                "resident_bytes": sum(value.numel() * value.element_size()
                                      for value in self._quality_buffers.values()),
                "scope": "one request/branch; actual committed packet decode once per recipient",
                "verify_archived_hashes": self.quality_verify_history,
            },
            "resource_admission": self._admission,
            "original_source_hashes": self.observed_source_hashes,
        }
        if self.runtime is not None and hasattr(self.runtime, "residency"):
            row["owner_reconstruction"] = {
                "resident_bytes": self.runtime.resident_bytes,
                "active_leases": self.runtime.active_leases,
                "reserved_bytes": self.runtime.residency.reserved_bytes,
                "peak_reserved_bytes": self.runtime.residency.peak_reserved_bytes,
            }
            if hasattr(self.runtime, "async_stats"):
                row["owner_pending_sources"] = self.runtime.async_stats()
        if self.device.type == "cuda" and torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info(self.device)
            row["hbm"] = {
                "torch_allocated_bytes": torch.cuda.memory_allocated(self.device),
                "torch_reserved_bytes": torch.cuda.memory_reserved(self.device),
                "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "device_used_bytes": total - free, "device_total_bytes": total,
                "device_used_includes_other_processes_and_driver": True,
            }
        return row

    def _capture_rng(self, generators):
        import torch

        metadata = {
            "python": random.getstate(), "generators": sorted(generators),
            "generator_devices": {name: generator.device.type for name, generator in generators.items()},
        }
        tensors = {"torch_cpu": torch.get_rng_state().clone()}
        if self.device.type == "cuda":
            if not torch.cuda.is_initialized():
                raise StateError("Native CUDA RNG was not initialized by the external builder")
            tensors["torch_cuda"] = torch.cuda.get_rng_state(self.device).cpu()
        for name, generator in generators.items():
            tensors[f"generator/{name}"] = generator.get_state().cpu()
        try:
            import numpy as np
        except ImportError:
            metadata["numpy"] = None
        else:
            name, keys, position, has_gauss, cached = np.random.get_state()
            metadata["numpy"] = [name, position, has_gauss, cached]
            tensors["numpy_keys"] = torch.from_numpy(keys.astype("int64"))
        return metadata, tensors

    def _restore_rng(self, metadata, tensors, generators):
        import torch

        if sorted(generators) != metadata["generators"]:
            raise ConfigurationError("Restoration generator names differ from the snapshot")
        if {name: generator.device.type for name, generator in generators.items()} != metadata["generator_devices"]:
            raise ConfigurationError("Restoration generator device types differ from the snapshot")
        random.setstate(_tuple_tree(metadata["python"]))
        torch.set_rng_state(tensors["torch_cpu"].cpu())
        if "torch_cuda" in tensors:
            if self.device.type != "cuda" or not torch.cuda.is_initialized():
                raise StateError("CUDA RNG cannot be restored by a CPU-only adapter")
            torch.cuda.set_rng_state(tensors["torch_cuda"].cpu(), self.device)
        for name, generator in generators.items():
            generator.set_state(tensors[f"generator/{name}"].cpu())
        if metadata["numpy"] is not None:
            import numpy as np

            name, position, has_gauss, cached = metadata["numpy"]
            np.random.set_state((name, tensors["numpy_keys"].numpy().astype("uint32"),
                                 position, has_gauss, cached))

    def snapshot(self, *, metadata, tensors, generators=None):
        """Capture execution/RNG/latent/conditioning state only after publication.

        The caller supplies ``conditioning_identity``, ``execution_config`` and
        ``vae_contract`` metadata. VAE state is either explicitly named tensors
        or an explicit stateless-per-chunk decoding contract; omission is not
        treated as proof that a streaming VAE has no state.
        """
        generators = generators or {}
        error = None
        try:
            import torch

            self._check_rank_binding()
            if (self._phase != "ready" or self._live_layer is not None
                    or (self.cache_mode in ("runtime", "sp_tiered") and self.cache.key_value_memory_dict)
                    or self._hot or self._hot_epoch is not None):
                raise StateError("No snapshot of in-flight, noisy, unsealed, or resident native state")
            if self._quality_buffers:
                self._check_quality_binding()
                for layer, buffer in self._quality_buffers.items():
                    self._quality_digest(layer, buffer, self._cursor * self.geometry.tokens)
            if any(not isinstance(name, str) or not name or name.startswith("magi_rope/")
                   or not isinstance(value, torch.Tensor) for name, value in tensors.items()):
                raise ConfigurationError("Snapshot tensor names/types are invalid or use the reserved rotary prefix")
            json.dumps(metadata, allow_nan=False)
            _checked_reuse_threshold(self.reuse_threshold)
            if not metadata.get("conditioning_identity") or not metadata.get("execution_config"):
                raise ConfigurationError("Snapshot conditioning/configuration identity is required")
            vae = metadata.get("vae_contract", {})
            if vae.get("mode") not in ("stateless_chunk_decode", "explicit_state"):
                raise UnsupportedError("Declare the VAE state/decoding contract explicitly")
            if vae["mode"] == "explicit_state" and (
                not vae.get("tensor_names") or any(name not in tensors for name in vae["tensor_names"])
            ):
                raise ConfigurationError("The named streaming VAE state is incomplete")
            if not {"latent", "conditioning", "conditioning_mask"}.issubset(tensors):
                raise ConfigurationError("Snapshots require latent and actual conditioning tensors")
        except Exception as exc:
            error = exc
        descriptions = self.transport.checkpoint("magi_snapshot_validate", error=error, details=metadata)
        if any(row != descriptions[0] for row in descriptions):
            raise StateError("SP ranks disagree on the snapshot execution/conditioning/VAE contract")
        rank_state = None
        if self.rank_policy is not None:
            rank_state = self.transport.owner_call(
                "magi_snapshot_committed_rank_state",
                lambda: self.runtime.history_rank_state(self.branch_id).to_dict(),
            )
            rank_state = self.transport.broadcast_json(rank_state, operation="magi_snapshot_rank_state")
        common = dict(tensors)
        for chunk, pair in self._ropes.items():
            for name, tensor in zip(("cos", "sin"), pair):
                common[f"magi_rope/{chunk}/{name}"] = tensor
        description = {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": _tensor_hash(tensor)}
            for name, tensor in common.items()
        }
        rows = self.transport.checkpoint("magi_snapshot_common_state", details=description)
        if any(row != rows[0] for row in rows):
            raise StateError("SP ranks disagree on conditioning, latent, or rotary state")
        local_meta, local_rng = self._capture_rng(generators)
        rank_meta = self.transport.checkpoint("magi_snapshot_rng", details=local_meta)
        rank_names = self.transport.checkpoint("magi_snapshot_rng_names", details=sorted(local_rng))
        saved = {f"state/{name}": tensor.detach().cpu() for name, tensor in common.items()} if self.transport.is_owner else {}
        for rank, names in enumerate(rank_names):
            for name in names:
                value = self.transport.transfer_tensor(
                    local_rng.get(name), source=rank, destination=self.transport.owner_rank,
                    operation=f"magi_snapshot_rng:{rank}:{name}",
                )
                if self.transport.is_owner:
                    saved[f"rng/{rank}/{name}"] = value
        adapter_metadata = {
            **metadata, "adapter": self.schema, "mode": self.mode,
            "native_cache_dtype": self.native_dtype,
            "canonical_source_hashes": self.canonical_source_hashes,
            "observed_source_hashes": self.observed_source_hashes,
            "residency_mode": self.cache_mode,
            "clean_boundary": {
                "clean": True, "chunk_id": self._chunks[-1]["chunk_id"],
                "token_end": self._cursor * self.geometry.tokens,
            },
            "conditioning": {
                "identity": metadata["conditioning_identity"],
                "tensor_names": ["state/conditioning", "state/conditioning_mask"],
            },
            "execution": {
                "mode": self.mode, "cursor": self._cursor,
                "config": metadata["execution_config"], "pending": False,
            },
            "rng": {"mode": "rank_local", "world_size": self.transport.world_size},
            "latent": {"tensor_name": "state/latent", "next_chunk": self._cursor},
            "vae": metadata["vae_contract"], "clean_timestep": 0.9999,
            "native_cache_state": {
                "max_batch_size": self.cache.max_batch_size,
                "max_sequence_length": self.cache.max_sequence_length,
                "sequence_len_offset": getattr(self.cache, "sequence_len_offset", 0),
                "update_kv_cache": bool(self.cache.update_kv_cache),
                "kv_compressed": bool(getattr(self.cache, "kv_compressed", False)),
            },
            "cursor": self._cursor, "geometry": asdict(self.geometry),
            "chunks": self._chunks, "rng_by_rank": rank_meta,
            "specs": {f"{key.layer}:{key.target}": spec.to_dict() for key, spec in self.specs.items()}
            if self.rank_policy is None else None,
            "rank_policy": self.rank_policy.to_dict() if self.rank_policy is not None else None,
            "rank_policy_state": rank_state,
            "reuse_threshold": self.reuse_threshold,
        }
        result = self.transport.owner_call(
            "magi_snapshot_publish",
            lambda: self.runtime.snapshot(self.branch_id, metadata=adapter_metadata, tensors=saved),
        )
        return self.transport.broadcast_json(result, operation="magi_snapshot_id")

    def _requested_restore_profile(self):
        self._check_rank_binding()
        if self.rank_policy is None and set(self.specs) != set(self.keys):
            raise ConfigurationError("MAGI requires a complete requested K/V profile")
        if self.rank_policy is not None and self.specs:
            raise ConfigurationError("An adaptive MAGI policy cannot be replaced by a mutable fixed spec map")
        specs, domains = {}, {}
        for key in self.keys:
            label = f"{key.layer}:{key.target}"
            if self.rank_policy is None:
                spec = self.specs[key]
                if not isinstance(spec, CodecSpec):
                    raise ConfigurationError("Requested MAGI profile entries must be CodecSpec values")
                specs[label] = _portable_spec(spec)
            domains[label] = {
                "key_space": (("post_rope" if self._source_is_raw(key) else "pre_rope")
                              if key.target == "K" else "none"),
                "dtype": "float32" if key.target == "K" and not self._source_is_raw(key) else self.native_dtype,
            }
        return {
            "model": self.identity.to_dict(), "geometry": asdict(self.geometry),
            "native_cache_dtype": self.native_dtype,
            "specs": specs if self.rank_policy is None else None, "domains": domains,
            "rank_policy": self.rank_policy.to_dict() if self.rank_policy is not None else None,
            "reuse_threshold": _checked_reuse_threshold(self.reuse_threshold),
        }

    def _validate_restored_profile(self, state, record, requested):
        metadata = state.metadata
        saved = metadata.get("specs")
        committed_policy = getattr(state, "rank_policy", None)
        committed_state = getattr(state, "rank_state", None)
        if self.rank_policy is not None:
            from ..online_rank import RankPolicy, RankState

            if (not isinstance(committed_policy, RankPolicy)
                    or committed_policy.fingerprint != self.rank_policy.fingerprint):
                raise ConfigurationError("Snapshot committed rank policy differs from the requested MAGI policy")
            if (not isinstance(committed_state, RankState)
                    or committed_state.policy_hash != committed_policy.fingerprint
                    or metadata.get("rank_policy_state") != committed_state.to_dict()):
                raise ConfigurationError("Snapshot scalar rank state differs from its verified committed rank state")
        elif committed_policy is not None or committed_state is not None:
            raise ConfigurationError("A static MAGI profile cannot restore adaptive policy history")
        if metadata.get("rank_policy") != requested["rank_policy"]:
            raise ConfigurationError("Snapshot rank policy does not match the requested MAGI policy")
        if self.rank_policy is None:
            if not isinstance(saved, Mapping) or set(saved) != set(requested["specs"]):
                raise ConfigurationError("Snapshot codec profile does not match the requested MAGI profile")
            portable = {name: _portable_spec(CodecSpec.from_dict(value)) for name, value in saved.items()}
            if portable != requested["specs"]:
                raise ConfigurationError("Snapshot codec profile does not match the requested MAGI profile")
        elif saved is not None:
            raise ConfigurationError("Adaptive snapshots cannot assert a fixed resolved rank profile")
        if _checked_reuse_threshold(metadata.get("reuse_threshold")) != requested["reuse_threshold"]:
            raise ConfigurationError("Snapshot reuse policy does not match the requested MAGI profile")

        head = self.runtime.store.get_manifest(record["head"])
        native_chunks = metadata["chunks"]
        if (type(metadata["cursor"]) is not int or metadata["cursor"] != len(native_chunks)
                or len(head["chunks"]) != len(native_chunks)):
            raise ConfigurationError("Snapshot native cursor and committed history differ")
        expected_keys = set(self.keys)
        for reference, native_chunk in zip(head["chunks"], native_chunks):
            chunk = self.runtime.store.get_manifest(reference)
            if self.rank_policy is not None:
                if (chunk.get("version") != 2
                        or chunk.get("rank_policy_identity") != self.rank_policy.fingerprint):
                    raise ConfigurationError("Snapshot contains incompatible static or wrong-policy history")
            if any(native_chunk[name] != chunk[name] for name in ("chunk_id", "token_start", "token_end")):
                raise ConfigurationError("Snapshot native positions and committed history differ")
            if set(native_chunk["spaces"]) != {str(layer) for layer in range(self.geometry.layers)}:
                raise ConfigurationError("Snapshot native key-space declarations are incomplete")
            seen = set()
            for matrix in chunk["matrices"]:
                key = MatrixKey.from_dict(matrix["key"])
                packet = self.runtime.store.get_manifest(matrix["tensor"])
                context = TensorContext.from_dict(packet["context"])
                if key not in expected_keys or key in seen:
                    raise ConfigurationError("Snapshot native matrix table is incompatible")
                seen.add(key)
                if self.rank_policy is not None:
                    self.rank_policy.validate_context(context)
                    spec = CodecSpec.from_dict(packet["spec"])
                    if self.rank_policy.spec_for(key, spec.rank) != spec:
                        raise ConfigurationError("Snapshot packet spec differs from its bound adaptive policy")
                elif _portable_spec(CodecSpec.from_dict(packet["spec"])) != requested["specs"][
                    f"{key.layer}:{key.target}"
                ]:
                    raise ConfigurationError("Snapshot packet spec differs from the requested static MAGI profile")
                domain = requested["domains"][f"{key.layer}:{key.target}"]
                if (context.key != key or context.model != self.identity
                        or context.key_space != domain["key_space"] or context.dtype != domain["dtype"]
                        or (context.frames, context.height, context.width)
                        != (self.geometry.frames, self.geometry.height, self.geometry.width)
                        or context.token_start != chunk["token_start"] or context.token_end != chunk["token_end"]
                        or (key.target == "K" and native_chunk["spaces"][str(key.layer)] != context.key_space)):
                    raise ConfigurationError(f"Snapshot source domain does not match the requested MAGI profile: {key}")
            if seen != expected_keys:
                raise ConfigurationError("Snapshot native matrix table is incomplete")

    def _validate_restored_tensors(self, state, metadata):
        if state.metadata != metadata:
            raise IntegrityError("Snapshot metadata changed before native validation")
        if not {"state/latent", "state/conditioning", "state/conditioning_mask"}.issubset(state.tensors):
            raise ConfigurationError("Snapshot native latent/conditioning tensors are incomplete")

    def restore(self, snapshot_id, branch_id, *, generators=None, fork=False):
        """Restore into a new branch and return private native state tensors.

        The caller reinstalls its conditioning and latent references from the
        returned state. No native dense cache is reconstructed until a layer
        actually requests history. Portable codec/rank/normalization and the
        configured reuse policy must match; operational FFmpeg paths may differ.
        Pure profile/head/cursor validation precedes native state-blob reads.
        Full tensor validation still runs before branch publication or
        native/RNG mutation.
        """
        error = None
        if self._phase.startswith("active_") or self._live_layer is not None:
            error = StateError("Cannot restore while a native forward is in flight")
        self.transport.checkpoint("magi_restore_validate", error=error)
        record = self.transport.owner_call(
            "magi_inspect_snapshot", lambda: self.runtime.store.get_manifest(snapshot_id),
        )
        metadata = self.transport.broadcast_json(
            record["metadata"] if self.transport.is_owner else None, operation="magi_restore_metadata",
        )
        error = None
        generators = generators or {}
        requested = None
        try:
            if (metadata.get("adapter") != self.schema or metadata.get("mode") != self.mode
                    or metadata.get("clean_timestep") != 0.9999
                    or metadata.get("native_cache_dtype") != self.native_dtype
                    or metadata.get("geometry") != asdict(self.geometry)):
                raise UnsupportedError("Snapshot is not a compatible MAGI clean boundary")
            requested = self._requested_restore_profile()
            rng = metadata["rng_by_rank"][self.transport.rank]
            if sorted(generators) != rng["generators"] or {
                name: generator.device.type for name, generator in generators.items()
            } != rng["generator_devices"]:
                raise ConfigurationError("Restoration generators differ from the saved native execution")
        except Exception as exc:
            error = exc
        requested_hash = hashlib.sha256(json.dumps(requested, sort_keys=True).encode()).hexdigest()
        profiles = self.transport.checkpoint("magi_restore_contract", error=error, details=requested_hash)
        if any(profile != profiles[0] for profile in profiles):
            raise StateError("Ranks disagree on the requested MAGI restoration profile")

        def validate_metadata(state):
            if state.metadata != metadata:
                raise IntegrityError("Snapshot metadata changed before native validation")
            self._validate_restored_profile(state, record, requested)

        def validate(state):
            self._validate_restored_tensors(state, metadata)

        state = self.transport.owner_call(
            "magi_fork_branch" if fork else "magi_restore_new_branch",
            lambda: self.runtime.fork(
                snapshot_id, branch_id, metadata_validator=validate_metadata, validator=validate,
                **({"rank_policy": self.rank_policy} if self.rank_policy is not None else {}),
            ),
        )
        self._clear_hot_window()
        self.release_quality_history()
        names = self.transport.broadcast_json(
            sorted(state.tensors) if self.transport.is_owner else None, operation="magi_restore_names",
        )
        received, local_rng = {}, {}
        for name in names:
            destinations = range(self.transport.world_size) if name.startswith("state/") else (
                int(name.split("/", 2)[1]),
            )
            for rank in destinations:
                value = self.transport.transfer_tensor(
                    state.tensors[name] if self.transport.is_owner else None,
                    source=self.transport.owner_rank, destination=rank,
                    operation=f"magi_restore_tensor:{name}:{rank}",
                )
                if self.transport.rank == rank:
                    if name.startswith("state/"):
                        received[name[6:]] = value.clone()
                    else:
                        local_rng[name.split("/", 2)[2]] = value.clone()
        self._restore_rng(metadata["rng_by_rank"][self.transport.rank], local_rng, generators)
        self._abort_stream_local()
        if self.sp_history is not None:
            self.sp_history.clear()
        if self._pending is not None:
            shutil.rmtree(self._pending.directory)
        self._pending = self._source_pending = self._pending_rope = None
        self._pending_source_provenance = {}
        self._cursor = metadata["cursor"]
        self._chunks = json.loads(json.dumps(metadata["chunks"]))
        self.canonical_source_hashes = json.loads(json.dumps(metadata["canonical_source_hashes"]))
        self.observed_source_hashes = json.loads(json.dumps(metadata.get("observed_source_hashes", {})))
        self._stats["completed_clean_chunks"] = self._cursor
        self._ropes = {}
        for chunk in self._chunks:
            chunk_id = chunk["chunk_id"]
            cos_name, sin_name = f"magi_rope/{chunk_id}/cos", f"magi_rope/{chunk_id}/sin"
            if cos_name in received:
                self._ropes[chunk_id] = (received.pop(cos_name), received.pop(sin_name))
        for name, value in metadata["native_cache_state"].items():
            setattr(self.cache, name, value)
        self.branch_id = branch_id
        self._branch_created = True
        self._phase = "ready"
        return metadata, received

    def fork(self, snapshot_id, branch_id, *, generators=None):
        return self.restore(snapshot_id, branch_id, generators=generators, fork=True)
