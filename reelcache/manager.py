"""Clean-chunk publication, copy-on-write branches, and bounded restoration."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict, deque
from concurrent.futures import CancelledError
import contextlib
import hashlib
import json
import math
import os
import sqlite3
import threading
import time
import traceback
from types import MappingProxyType
from typing import Any, Callable, Mapping

import torch

from .errors import (
    CapacityError, ConfigurationError, IntegrityError, ReelCacheError, PendingError, QuarantinedError, StateError,
    UnsupportedError,
)
from .residency import HistoryLease, ResidencyPool
from .execution import BoundedExecutor, Job, _timeout
from .serialization import (
    deserialize_tensors, dtype_name, resolve_dtype, serialize_tensors,
    tensor_from_bytes, tensor_nbytes,
)
from .lineage import LINEAGE_PREFIX, LineageRuntimeMixin
from .store import ObjectLease, ObjectStore, ReadAhead
from .types import CodecSpec, EncodedTensor, MatrixKey, ModelIdentity, TensorContext, json_copy
from .telemetry import span


#: ``KVRuntime(on_open_error=...)``: a damaged head fails the open, or is quarantined while the others work.
OPEN_ERROR_POLICIES = ("raise", "quarantine")


def _rank_digest(value):
    return hashlib.sha256(json.dumps(
        json_copy(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _encode_with_metrics(encoder, tensor, context, spec, **kwargs):
    with span("seal.encode", nbytes=tensor.untyped_storage().nbytes()):
        return encoder(tensor, context, spec, **kwargs)


class _CommitSources(Mapping):
    """Load staged anchors once and release each source after its own encode."""

    def __init__(self, sources, contexts, keys, validator):
        self._sources, self._contexts = sources, contexts
        self._keys, self._validator = keys, validator
        self._cached = {}
        self._consumed = set()
        self.peak_bytes = 0
        self.load_seconds = 0.0

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def __getitem__(self, key):
        if key not in self._keys or key in self._consumed or self._sources is None:
            raise ConfigurationError("rank preparation requested an unavailable matrix source")
        if key not in self._cached:
            started = time.perf_counter()
            try:
                tensor = self._sources[key]
            except KeyError as exc:
                raise ConfigurationError(f"matrix disappeared during clean commit: {key}") from exc
            self._validator(tensor, self._contexts[key])
            self.load_seconds += time.perf_counter() - started
            self._cached[key] = tensor
            storages = {
                (item.device, item.untyped_storage().data_ptr()): item.untyped_storage().nbytes()
                for item in self._cached.values()
            }
            self.peak_bytes = max(self.peak_bytes, sum(storages.values()))
        return self._cached[key]

    def take(self, key):
        tensor = self[key]
        del self._cached[key]
        self._consumed.add(key)
        return tensor

    def close(self):
        self._cached.clear()
        self._sources = self._contexts = self._validator = None


@dataclass(frozen=True)
class SnapshotState:
    snapshot_id: str
    metadata: Mapping[str, Any]
    tensors: Mapping[str, torch.Tensor]
    rank_policy: Any = None
    rank_state: Any = None


@dataclass(frozen=True)
class RestoredState:
    snapshot_id: str
    branch_id: str
    metadata: Mapping[str, Any]
    tensors: Mapping[str, torch.Tensor]
    rank_policy: Any = None
    rank_state: Any = None


@dataclass
class _CommitState:
    branch_id: str
    parent_head: str
    source_bytes: int
    pin: ObjectLease
    phase: str = "queued"
    cancel_requested: bool = False
    result_id: str | None = None
    error: BaseException | None = None
    job: Job[str] | None = None
    finished: threading.Event = field(default_factory=threading.Event)


class PendingCommit:
    """A RAW clean commit; cancellation is acknowledged only when done."""

    def __init__(self, runtime: KVRuntime, state: _CommitState):
        self._runtime, self._state = runtime, state

    def done(self) -> bool:
        return self._state.finished.is_set()

    def cancel(self) -> bool:
        with self._runtime._lock:
            state = self._state
            if state.phase in ("published", "failed", "cancelled") or self.done():
                return False
            state.cancel_requested = True
            if state.job is not None:
                state.job.cancel()
            return True

    def status(self) -> dict:
        with self._runtime._lock:
            return {
                "phase": self._state.phase, "done": self.done(),
                "cancel_requested": self._state.cancel_requested,
                "published": self._state.result_id is not None,
                "source_bytes": self._state.source_bytes,
                "error": None if self._state.error is None else str(self._state.error),
            }

    def result(self, timeout: float | None = None) -> str:
        _timeout(timeout)
        if not self._state.finished.wait(timeout):
            raise TimeoutError("clean commit is not ready; waiting timeout does not cancel it")
        with self._runtime._lock:
            state = self._state
            if self._runtime._async_failures.get(state.branch_id) is state:
                del self._runtime._async_failures[state.branch_id]
            if state.error is not None:
                raise state.error
            if state.result_id is None:
                raise StateError("completed clean commit has no published chunk")
            return state.result_id


class StreamingChunk:
    """Layer-wise clean-chunk writer: encode each matrix when it arrives, publish atomically once.

    Obtain from :meth:`KVRuntime.open_chunk`. ``add`` may be called from any one
    producer thread (for example a layer pipeline overlapping native forwards);
    with an ``executor`` it returns once the encode is admitted. ``commit``
    waits for every encode and publishes the same packets and chunk record as the
    equivalent synchronous :meth:`KVRuntime.seal_chunk` (timing diagnostics
    aside). Nothing is visible
    before ``commit``; ``abort`` (or leaving the context without committing)
    reclaims every staged packet and releases the branch reservation.
    """

    def __init__(self, runtime, branch_id, chunk_id, contexts, specs, head_id, head, pin, executor,
                 committed_ids=()):
        self._runtime, self.branch_id, self.chunk_id = runtime, branch_id, chunk_id
        self._committed_ids = set(committed_ids)
        self._contexts, self._specs = contexts, specs
        self._head_id, self._head, self._pin = head_id, head, pin
        self._executor = executor
        self._lock = threading.Lock()
        self._records: dict[MatrixKey, dict[str, Any]] = {}
        self._jobs: dict[MatrixKey, Any] = {}
        self._staged_pins: list[ObjectLease] = []
        self._staged_objects: list[str] = []
        self._state = "open"
        self.stats = {"added": 0, "encoded": 0, "encode_seconds": 0.0, "persist_seconds": 0.0,
                      "commit_wait_seconds": 0.0, "first_add": None, "last_encoded": None}

    def _encode_and_stage(self, key, tensor):
        runtime = self._runtime
        started = time.perf_counter()
        encoded = _encode_with_metrics(runtime.encoder, tensor, self._contexts[key], self._specs[key])
        encoded_at = time.perf_counter()
        with span("seal.persist"), runtime.store.transaction():
            try:
                packet = runtime._save_tensor(encoded, self._contexts[key], self._specs[key])
                lease = runtime.store.lease(packet, purpose="staging")
            finally:
                with self._lock:
                    self._staged_objects.extend(runtime.store.touched_objects)
        with self._lock:
            self._staged_pins.append(lease)
            self._records[key] = {"key": key.to_dict(), "tensor": packet}
            self.stats["encoded"] += 1
            self.stats["encode_seconds"] += encoded_at - started
            self.stats["persist_seconds"] += time.perf_counter() - encoded_at
            self.stats["last_encoded"] = time.perf_counter()
        return packet

    def add(self, key: MatrixKey, tensor: torch.Tensor) -> None:
        runtime = self._runtime
        with self._lock:
            if self._state != "open":
                raise StateError(f"streaming chunk is {self._state}")
            if key not in self._contexts:
                raise ConfigurationError(f"unknown matrix key for this chunk: {key}")
            if key in self._jobs or key in self._records:
                raise StateError(f"matrix was already added to this chunk: {key}")
            self.stats["added"] += 1
            if self.stats["first_add"] is None:
                self.stats["first_add"] = time.perf_counter()
        runtime._validate_matrix(tensor, self._contexts[key])
        if self._executor is None:
            with self._lock:
                self._jobs[key] = None
            self._encode_and_stage(key, tensor)
            return
        if tensor.device.type != "cpu":
            raise UnsupportedError("stage streamed matrices on CPU before executor encoding")
        staging = tensor.untyped_storage().nbytes() + runtime.encode_workspace_bytes
        job = self._executor.submit(
            self._encode_and_stage, key, tensor, staging_bytes=staging, block=True,
            label=f"stream-encode:{self.branch_id}:{self.chunk_id}:{key.layer}:{key.target}",
        )
        with self._lock:
            self._jobs[key] = job

    def _drain(self):
        failures = []
        for job in list(self._jobs.values()):
            if job is None:
                continue
            try:
                job.result()
            except BaseException as exc:
                failures.append(exc)
        return failures

    def commit(self, chunk_id: Any = None) -> str:
        """Publish; ``chunk_id`` may replace the provisional id given to ``open_chunk``."""
        runtime = self._runtime
        with self._lock:
            if self._state != "open":
                raise StateError(f"streaming chunk is {self._state}")
            if chunk_id is not None:
                runtime._chunk_id(chunk_id)
                if chunk_id in self._committed_ids:
                    raise StateError("chunk_id has already been committed on this branch")
                self.chunk_id = chunk_id
            self._state = "committing"
        published = False
        try:
            waited = time.perf_counter()
            with span("seal.wait"):
                failures = self._drain()
            self.stats["commit_wait_seconds"] = time.perf_counter() - waited
            if failures:
                if len(failures) == 1:
                    raise failures[0]
                raise StateError("streamed encoding failed: " + "; ".join(
                    f"{type(error).__name__}: {error}" for error in failures)) from failures[0]
            if set(self._records) != set(runtime._keys):
                missing = sorted(set(runtime._keys) - set(self._records), key=lambda k: (k.layer, k.target))
                raise StateError(f"streaming chunk is missing matrices: {missing[:4]}")
            records = [self._records[key] for key in runtime._keys]
            with span("seal.publish"), runtime._lock:
                runtime._open()
                with runtime.store.transaction():
                    context = self._contexts[runtime._keys[0]]
                    chunk_data = {
                        "type": "chunk", "version": 1, "chunk_id": self.chunk_id,
                        "model": runtime.model.to_dict(), "clean": True,
                        "token_start": context.token_start, "token_end": context.token_end,
                        "frames": context.frames, "height": context.height, "width": context.width,
                        "matrices": records,
                    }
                    chunk_object = runtime.store.put_manifest(
                        chunk_data, references=[entry["tensor"] for entry in records],
                    )
                    next_head = runtime._new_head(self._head["chunks"] + [chunk_object])
                    runtime.store.set_ref(runtime._branch_name(self.branch_id), next_head, expected=self._head_id)
                published = True
            self._state = "committed"
            return chunk_object
        finally:
            if not published:
                self._state = "failed"
            self._release(published)

    def abort(self) -> None:
        with self._lock:
            if self._state in ("committed", "aborted", "failed"):
                return
            self._state = "aborting"
        for job in list(self._jobs.values()):
            if job is not None and not job.done():
                job.cancel()
        self._drain()
        self._state = "aborted"
        self._release(False)

    def _release(self, published):
        runtime = self._runtime
        try:
            with runtime.store.transaction():
                for lease in self._staged_pins:
                    lease.release()
                if self._pin is not None:
                    self._pin.release()
            if not published:
                runtime.store.collect_garbage(candidates=self._staged_objects)
        finally:
            self._staged_pins.clear()
            self._pin = None
            with runtime._lock:
                runtime._streams.discard(self.branch_id)

    @property
    def state(self) -> str:
        return self._state

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._state == "open":
            self.abort()


class KVRuntime(LineageRuntimeMixin):
    """A single-host durable runtime over canonical full-KV-channel matrices.

    The residency budget covers leased output and conservative reconstruction
    workspace reservations, not caller-owned native state or codec allocator
    overhead. It is not a measured or proven total peak-HBM bound.
    Executors are caller-owned; static matrix concurrency is explicitly opt-in.
    ``restore`` may replace a branch; ``fork`` requires an unused branch name.
    Both return caller-owned metadata/tensors and retain no decoded state.

    Store integrity: ``verify_reads`` hash-checks every object read for use;
    ``verify_references`` (None: follow ``verify_reads``) also hashes every
    object reached by graph operations (publication, leases, GC liveness) —
    the original strict mode; serving passes ``verify_reads=True,
    verify_references=False``. Opening checks every branch, snapshot and
    lineage head: ``audit_on_open`` (None: follow ``verify_reads``) chooses a
    full hash audit (cost grows with the stored bytes) over the graph check
    (manifests parse, every reference is indexed). ``on_open_error`` decides
    what a damaged head does: ``"raise"`` (default) fails the open;
    ``"quarantine"`` lists it in :attr:`quarantined`, raises
    :class:`~reelcache.errors.QuarantinedError` (an IntegrityError) whenever it
    is loaded, and lets every other head work; garbage collection then keeps
    damaged graphs (store ``on_damage="keep"``) instead of failing.

    History read-ahead: once two ``acquire_history`` calls of one head and
    token range ask for consecutive model keys (layer-major, K before V), the
    payloads of the following keys of that range are read ahead -- with the
    checks of the store's ``get`` (``verify_reads``: SHA-256 against the
    content address), several objects at a time on the store's read-ahead
    threads -- within ``history_read_ahead_bytes`` (None:
    :attr:`HISTORY_READ_AHEAD_BYTES`; 0 disables) of bytes read and not used
    yet. A read-ahead object that failed its read is read again where it is
    used, which raises exactly as without read-ahead; no unchecked byte is
    used. The window ends with a non-sequential call of its range, after
    :attr:`HISTORY_READ_AHEAD_IDLE_SECONDS` without a call, at garbage
    collection and at close.
    """

    #: Default bound of history payload bytes read ahead and not used yet (4 LongCat-sized K/V payloads).
    HISTORY_READ_AHEAD_BYTES = 512 << 20
    #: A read-ahead window unused this long gives way to another range's (dropped at the next history call).
    HISTORY_READ_AHEAD_IDLE_SECONDS = 5.0
    #: Bound of the (head, range) sequences tracked for sequential detection.
    HISTORY_READ_AHEAD_TRACKS = 64

    def __init__(
        self, root: str | os.PathLike[str], model: ModelIdentity, *,
        max_resident_bytes: int = 1 << 30, max_store_bytes: int | None = None,
        encoder: Callable[..., EncodedTensor] | None = None,
        decoder: Callable[..., torch.Tensor] | None = None, executor: Any = None,
        encode_concurrency: int = 1, encode_workspace_bytes: int = 0,
        max_pending_source_bytes: int = 1 << 30, verify_reads: bool = False,
        prefit: Callable[..., Any] | None = None, audit_on_open: bool | None = None,
        on_open_error: str = "raise", verify_references: bool | None = None,
        history_read_ahead_bytes: int | None = None,
    ) -> None:
        if not isinstance(model, ModelIdentity):
            raise ConfigurationError("runtime model must be a ModelIdentity")
        if history_read_ahead_bytes is not None and (
                type(history_read_ahead_bytes) is not int or history_read_ahead_bytes < 0):
            raise ConfigurationError("history_read_ahead_bytes must be a nonnegative integer or None (default)")
        if audit_on_open is not None and type(audit_on_open) is not bool:
            raise ConfigurationError("audit_on_open must be a bool or None (follow verify_reads)")
        if on_open_error not in OPEN_ERROR_POLICIES:
            raise ConfigurationError(f"on_open_error must be one of {OPEN_ERROR_POLICIES}")
        if encoder is None or decoder is None:
            from .codecs import decode_tensor, encode_tensor
            encoder = encode_tensor if encoder is None else encoder
            decoder = decode_tensor if decoder is None else decoder
        if not callable(encoder) or not callable(decoder):
            raise ConfigurationError("encoder and decoder must be callable")
        if prefit is not None and not callable(prefit):
            raise ConfigurationError("prefit must be a callable returning a context manager")
        if executor is not None and not callable(getattr(executor, "submit", None)):
            raise ConfigurationError("executor must implement bounded submit")
        if type(encode_concurrency) is not int or encode_concurrency < 1:
            raise ConfigurationError("encode_concurrency must be a positive integer")
        if type(encode_workspace_bytes) is not int or encode_workspace_bytes < 0:
            raise ConfigurationError("encode_workspace_bytes must be a nonnegative integer")
        if encode_concurrency > 1:
            if not isinstance(executor, BoundedExecutor) or not encode_workspace_bytes:
                raise ConfigurationError("parallel encoding requires BoundedExecutor and a positive workspace reservation")
            if encode_concurrency > min(executor.max_workers, executor.max_pending):
                raise ConfigurationError("encode_concurrency exceeds executor worker or pending capacity")
        if type(max_pending_source_bytes) is not int or max_pending_source_bytes < 0:
            raise ConfigurationError("max_pending_source_bytes must be a nonnegative integer")
        self.model = model
        self.encoder, self.decoder, self.executor = encoder, decoder, executor
        self.prefit = prefit
        self.encode_concurrency = encode_concurrency
        self.encode_workspace_bytes = encode_workspace_bytes
        self.residency = ResidencyPool(max_resident_bytes)
        self._lock = threading.RLock()
        self._pending: set[str] = set()
        self._streams: set[str] = set()
        self._async_states: dict[str, _CommitState] = {}
        self._async_failures: dict[str, _CommitState] = {}
        self.max_pending_source_bytes = max_pending_source_bytes
        self._pending_source_bytes = 0
        self._peak_pending_source_bytes = 0
        self._closed = False
        self._keys = tuple(MatrixKey(layer, target)
                           for layer in range(model.layers) for target in ("K", "V"))
        self._key_index = {key: index for index, key in enumerate(self._keys)}
        self.history_read_ahead_bytes = (self.HISTORY_READ_AHEAD_BYTES if history_read_ahead_bytes is None
                                         else history_read_ahead_bytes)
        # (head, token_start, token_end) -> the key index of its latest acquire_history (sequential detection).
        self._read_ahead_tracks: OrderedDict[tuple[str, int, int], int] = OrderedDict()
        self._read_ahead_window: tuple[tuple[str, int, int], ReadAhead] | None = None
        self._read_ahead_stats = {"windows": 0, "taken": 0, "taken_bytes": 0, "failed": 0, "dropped": 0,
                                  "plan_failures": 0}
        self.store = ObjectStore(root, max_bytes=max_store_bytes, verify_reads=verify_reads,
                                 verify_references=verify_references,
                                 on_damage="keep" if on_open_error == "quarantine" else "raise")
        self.verify_reads = verify_reads
        self.audit_on_open = verify_reads if audit_on_open is None else audit_on_open
        self.on_open_error = on_open_error
        # Heads found damaged (``on_open_error="quarantine"``): ref name -> (head object id, error); by object id.
        self._quarantine: dict[str, tuple[str, str]] = {}
        self._quarantined_objects: dict[str, str] = {}
        self._lineage_init()
        try:
            with self.store.transaction():
                identity = self.store.get_ref("runtime/model")
                if identity is None:
                    if self.store.list_refs():
                        raise IntegrityError("runtime identity is missing from a populated store")
                    identity = self.store.put_manifest(
                        {"type": "model", "version": 1, "model": model.to_dict()}
                    )
                    self.store.set_ref("runtime/model", identity, expected=None)
                else:
                    data = self._manifest(identity, "model")
                    self._same_model(data.get("model"))
                if on_open_error == "raise":
                    for head in self.store.list_refs("branches/").values():
                        self._load_head(head)
                        self._audit(head, deep=self.audit_on_open)
                    for snapshot_id in self.store.list_refs("snapshots/").values():
                        self._snapshot_record(snapshot_id, verify=False)
                        self._audit(snapshot_id, deep=self.audit_on_open)
                    for head in self.store.list_refs(LINEAGE_PREFIX).values():
                        self._node_entry(head)
                        self._audit(head, deep=self.audit_on_open)
            if on_open_error == "quarantine":
                for name, (head, error) in self._damaged_heads(deep=self.audit_on_open).items():
                    self._quarantine[name] = (head, error)
                    self._quarantined_objects[head] = error
        except BaseException:
            self.store.close()
            self.residency.close()
            raise

    @property
    def resident_bytes(self) -> int:
        return self.residency.resident_bytes

    def _audit(self, object_id: str, *, deep: bool | None = None) -> None:
        """Hash-audit a graph when ``deep`` (default: ``verify_reads``); otherwise check references exist."""
        if self.verify_reads if deep is None else deep:
            self.store.verify(object_id)
        else:
            self.store.check(object_id)

    def _damaged_heads(self, *, deep: bool) -> dict[str, tuple[str, str]]:
        """``{ref name: (head, error)}`` of damaged branch, snapshot and lineage heads (each head is checked in
        its own store transaction, so one damaged head cannot invalidate the others' checks)."""
        with self.store.transaction():
            heads = [(kind, name, ref) for kind, prefix in (("branch", "branches/"), ("snapshot", "snapshots/"),
                                                             ("lineage", LINEAGE_PREFIX))
                     for name, ref in self.store.list_refs(prefix).items()]
        damaged: dict[str, tuple[str, str]] = {}
        for kind, name, ref in heads:
            known = self._quarantine.get(name)
            if known is not None and known[0] == ref:
                damaged[name] = known
                continue
            try:
                with self.store.transaction():
                    if kind == "branch":
                        self._load_head(ref)
                    elif kind == "snapshot":
                        self._snapshot_record(ref, verify=False)
                    else:
                        self._node_entry(ref)
                    self._audit(ref, deep=deep)
            except IntegrityError as exc:
                if isinstance(exc.__cause__, sqlite3.Error):
                    raise  # the store index failed: not damage of this head
                if self.store.get_ref(name) != ref:
                    continue  # moved or deleted meanwhile (its old graph may have been collected)
                damaged[name] = (ref, f"{type(exc).__name__}: {exc}")
        return damaged

    @property
    def quarantined(self) -> dict[str, str]:
        """``{ref name: error}`` of the heads quarantined in this process (``on_open_error="quarantine"``):
        ``branches/<id>``, ``snapshots/<hex>``, ``lineages/<id>``. Deleting the head through the runtime lifts it."""
        with self._lock:
            return {name: error for name, (_, error) in self._quarantine.items()}

    def audit(self, *, deep: bool = True, quarantine: bool = False) -> dict[str, str]:
        """Explicit audit of every branch, snapshot and lineage head now (``deep``: full hash audit, otherwise the
        graph check); returns ``{ref name: error}`` of the damaged ones (quarantined heads with their recorded
        error) and never raises for damage. ``quarantine=True`` quarantines newly damaged heads (any policy)."""
        if type(deep) is not bool or type(quarantine) is not bool:
            raise ConfigurationError("audit deep and quarantine must be booleans")
        with self._lock:
            self._open()
            damaged = self._damaged_heads(deep=deep)
            if quarantine:
                for name, (head, error) in damaged.items():
                    self._quarantine[name] = (head, error)
                    self._quarantined_objects[head] = error
            return {name: error for name, (_, error) in damaged.items()}

    def _check_quarantined(self, object_id: str) -> None:
        error = self._quarantined_objects.get(object_id)
        if error is not None:
            names = sorted(name for name, (ref, _) in self._quarantine.items() if ref == object_id)
            raise QuarantinedError(f"{', '.join(names) or object_id} is quarantined: damaged when checked ({error}); "
                                   "delete it, or repair the store and reopen")

    def _lift_quarantine(self, name: str) -> None:
        """A deleted head's quarantine ends (its objects stay until garbage collection)."""
        entry = self._quarantine.pop(name, None)
        if entry is not None and all(ref != entry[0] for ref, _ in self._quarantine.values()):
            self._quarantined_objects.pop(entry[0], None)

    @property
    def active_leases(self) -> int:
        return self.residency.active_leases

    def _open(self) -> None:
        if self._closed:
            raise StateError("runtime is closed")

    def _branch_name(self, branch_id: str) -> str:
        if (not isinstance(branch_id, str) or not branch_id or "\x00" in branch_id
                or "/" in branch_id or branch_id in {".", ".."}):
            raise ConfigurationError("branch_id must be a nonempty single-component name")
        return "branches/" + branch_id

    def _chunk_id(self, chunk_id: Any) -> None:
        if ((not isinstance(chunk_id, str) or not chunk_id)
                and (type(chunk_id) is not int or chunk_id < 0)):
            raise ConfigurationError("chunk_id must be a nonempty string or nonnegative integer")

    def _check_not_streaming(self, branch_id: str) -> None:
        if branch_id in self._streams:
            raise PendingError(f"branch has an open streaming chunk: {branch_id}")

    def _ready(self, branch_id: str) -> str:
        self._open()
        name = self._branch_name(branch_id)
        if branch_id in self._pending:
            raise PendingError(f"branch has an uncommitted chunk: {branch_id}")
        head = self.store.get_ref(name)
        if head is None:
            raise StateError(f"branch does not exist: {branch_id}")
        return head

    def _manifest(self, object_id: str, kind: str) -> dict[str, Any]:
        data, references = self.store.get_manifest_with_references(object_id)
        if (data.get("type") != kind or type(data.get("version")) is not int
                or data["version"] not in ({1, 2} if kind == "chunk" else {1})):
            raise IntegrityError(f"unsupported {kind} manifest version")
        try:
            if kind == "model":
                expected = []
            elif kind == "head":
                expected = data["chunks"]
                if not isinstance(expected, list):
                    raise IntegrityError("invalid branch chunk references")
            elif kind == "chunk":
                expected = [entry["tensor"] for entry in data["matrices"]]
                if data["version"] == 2:
                    expected.append(data["parent_head"])
                elif any(name in data for name in (
                    "parent_head", "rank_policy", "rank_policy_identity", "rank_state", "rank_preparation",
                )):
                    raise IntegrityError("adaptive metadata requires a version-2 chunk")
            elif kind == "tensor":
                encoded = data["encoded"]
                expected = list(encoded["payloads"].values())
                if encoded["basis"] is not None:
                    expected.append(encoded["basis"])
            elif kind == "snapshot":
                expected = [data["head"], data["tensor_state"]]
            else:
                raise IntegrityError("unsupported runtime manifest type")
            if (any(not isinstance(reference, str) for reference in expected)
                    or references != tuple(sorted(set(expected)))):
                raise IntegrityError("manifest contains an incomplete or extraneous reference graph")
        except (KeyError, TypeError, AttributeError) as exc:
            raise IntegrityError("invalid runtime manifest reference fields") from exc
        return data

    def _same_model(self, value: Any) -> None:
        try:
            stored = ModelIdentity.from_dict(value)
        except (ConfigurationError, TypeError, ValueError) as exc:
            raise IntegrityError("invalid stored model identity") from exc
        if stored != self.model:
            raise ConfigurationError("stored model identity/layout/dtype/channels do not match")

    def _tensor_record(self, object_id: str) -> tuple[TensorContext, CodecSpec, dict[str, Any]]:
        data = self._manifest(object_id, "tensor")
        try:
            context = TensorContext.from_dict(data["context"])
            spec = CodecSpec.from_dict(data["spec"])
            encoded = data["encoded"]
            if (not isinstance(encoded, dict)
                    or set(encoded) != {"metadata", "payloads", "basis", "metrics"}
                    or not isinstance(encoded["metadata"], dict)
                    or not isinstance(encoded["payloads"], dict) or not encoded["payloads"]):
                raise IntegrityError("invalid encoded tensor record")
            references = list(encoded["payloads"].values())
            if any(not isinstance(name, str) or not name for name in encoded["payloads"]):
                raise IntegrityError("invalid encoded payload name")
            if any(not isinstance(oid, str) or not oid.startswith("blob:") for oid in references):
                raise IntegrityError("tensor payloads must reference immutable byte blobs")
            if encoded["basis"] is not None:
                if not isinstance(encoded["basis"], str) or not encoded["basis"].startswith("basis:"):
                    raise IntegrityError("tensor basis must reference a canonical basis blob")
                references.append(encoded["basis"])
            self._same_model(context.model.to_dict())
            if context.key.target == "K" and context.key_space == "post_rope" and spec.kind != "raw":
                raise IntegrityError("stored post-RoPE K uses a forbidden SVD representation")
            self._encoded_geometry(encoded["metadata"], spec, context,
                                   basis_present=encoded["basis"] is not None)
            self._stored_descriptors(encoded["metadata"], encoded["payloads"], encoded["basis"])
            return context, spec, encoded
        except (KeyError, TypeError, ValueError, ConfigurationError) as exc:
            raise IntegrityError("invalid stored tensor context or representation") from exc

    def _encoded_geometry(
        self, metadata: Mapping[str, Any], spec: CodecSpec, context: TensorContext, *,
        basis_present: bool,
    ) -> None:
        try:
            declared_context = TensorContext.from_dict(metadata["context"])
            declared_spec = CodecSpec.from_dict(metadata["codec"])
        except (KeyError, TypeError, ConfigurationError) as exc:
            raise IntegrityError("provider metadata lacks a valid context or codec specification") from exc
        portable_spec = spec.to_dict()
        portable_spec.pop("ffmpeg_path", None)
        if declared_context != context or metadata["context"] != context.to_dict():
            raise IntegrityError("provider context differs from the authoritative matrix request")
        declared_portable = declared_spec.to_dict()
        declared_portable.pop("ffmpeg_path", None)
        metadata_codec = dict(metadata["codec"])
        if "protected_fraction" not in metadata_codec:
            raise IntegrityError("provider codec metadata lacks its protected fraction")
        metadata_codec["protected_fraction"] = declared_spec.protected_fraction
        if (declared_portable != portable_spec or metadata_codec != portable_spec
                or metadata.get("kind") != spec.kind):
            raise IntegrityError("provider codec specification differs from the authoritative request")
        if metadata.get("shape") != list(context.shape):
            raise IntegrityError("encoded matrix shape does not match its context")
        if metadata.get("dtype") != context.dtype:
            raise IntegrityError("encoded matrix dtype does not match its model")
        if spec.kind == "raw":
            if basis_present or metadata.get("basis") is not None:
                raise IntegrityError("RAW matrices must not carry a basis")
            return
        if not basis_present:
            raise IntegrityError("SVD matrices require a separately stored canonical basis")
        self._basis_geometry(metadata, context, spec)

    def _basis_geometry(
        self, metadata: Mapping[str, Any], context: TensorContext, spec: CodecSpec
    ) -> tuple[int, int]:
        basis = metadata.get("basis")
        if (not isinstance(basis, Mapping)
                or set(basis) != {"bytes", "sha256", "dtype", "shape", "view_rank"}):
            raise IntegrityError("missing canonical FP16 basis descriptor")
        shape = basis.get("shape")
        if (basis.get("dtype") not in {"float16", "torch.float16"}
                or not isinstance(shape, list) or len(shape) != 2
                or any(type(size) is not int or size <= 0 for size in shape)
                or shape[1] != context.channels
                or type(basis.get("view_rank")) is not int
                or basis["view_rank"] != spec.rank or not spec.rank <= shape[0] <= context.channels
                or type(basis.get("bytes")) is not int
                or basis["bytes"] != shape[0] * shape[1] * 2
                or not isinstance(basis.get("sha256"), str)
                or len(basis["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in basis["sha256"])):
            raise IntegrityError("invalid canonical FP16 basis shape/dtype/view rank")
        return shape[0], shape[1]

    def _payload_descriptors(self, metadata: Mapping[str, Any], names: Any) -> Mapping[str, Any]:
        descriptors = metadata.get("payloads")
        if not isinstance(descriptors, Mapping) or set(descriptors) != set(names):
            raise IntegrityError("provider payload descriptors do not cover its byte payloads")
        for descriptor in descriptors.values():
            if (not isinstance(descriptor, Mapping)
                    or type(descriptor.get("bytes")) is not int or descriptor["bytes"] < 0
                    or not isinstance(descriptor.get("sha256"), str)
                    or len(descriptor["sha256"]) != 64
                    or any(character not in "0123456789abcdef" for character in descriptor["sha256"])):
                raise IntegrityError("invalid provider byte payload descriptor")
        return descriptors

    def _stored_descriptors(
        self, metadata: Mapping[str, Any], payloads: Mapping[str, str], basis: str | None
    ) -> None:
        descriptors = self._payload_descriptors(metadata, payloads)
        for name, object_id in payloads.items():
            descriptor = descriptors[name]
            if (descriptor["sha256"] != object_id.split(":")[1]
                    or descriptor["bytes"] != self.store.object_size(object_id)):
                raise IntegrityError("provider payload descriptor does not match the immutable stored bytes")
        if basis is not None:
            if (metadata["basis"]["sha256"] != basis.split(":")[1]
                    or metadata["basis"]["bytes"] != self.store.object_size(basis)):
                raise IntegrityError("canonical basis descriptor does not match the immutable stored bytes")

    def _load_head(
        self, head_id: str, *, matrix_key: MatrixKey | None = None, metadata_only: bool = False
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if self._quarantined_objects:
            self._check_quarantined(head_id)
        head = self._manifest(head_id, "head")
        self._same_model(head.get("model"))
        ids = head.get("chunks")
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            raise IntegrityError("invalid branch chunk sequence")
        chunks: list[dict[str, Any]] = []
        end = 0
        geometry = None
        spaces: dict[MatrixKey, str] = {}
        dtypes: dict[MatrixKey, str] = {}
        names: set[str | int] = set()
        adaptive = None
        for chunk_object in ids:
            chunk = self._manifest(chunk_object, "chunk")
            self._same_model(chunk.get("model"))
            try:
                current_adaptive = chunk["version"] == 2
                if adaptive is not None and adaptive != current_adaptive:
                    raise IntegrityError("static and adaptive chunks cannot share one history")
                adaptive = current_adaptive
                if current_adaptive:
                    required_fields = {
                        "type", "version", "chunk_id", "model", "clean",
                        "token_start", "token_end", "frames", "height", "width", "matrices",
                        "parent_head", "rank_policy", "rank_policy_identity", "rank_state", "rank_preparation",
                    }
                    if set(chunk) != required_fields:
                        raise IntegrityError("adaptive chunk has missing or extraneous policy fields")
                    parent = self._manifest(chunk["parent_head"], "head")
                    if parent != {
                        "type": "head", "version": 1, "model": self.model.to_dict(),
                        "chunks": ids[:len(chunks)],
                    }:
                        raise IntegrityError("adaptive chunk is bound to another parent history")
                self._chunk_id(chunk["chunk_id"])
                if (any(type(chunk.get(field)) is not int for field in
                        ("token_start", "token_end", "frames", "height", "width"))
                        or chunk["token_start"] < 0 or chunk["token_end"] <= chunk["token_start"]
                        or min(chunk["frames"], chunk["height"], chunk["width"]) <= 0
                        or chunk["token_end"] != chunk["token_start"]
                        + chunk["frames"] * chunk["height"] * chunk["width"]):
                    raise IntegrityError("invalid committed chunk geometry")
                if chunk["chunk_id"] in names or chunk.get("clean") is not True:
                    raise IntegrityError("branch contains duplicate or unclean chunks")
                names.add(chunk["chunk_id"])
                if chunk["token_start"] != end:
                    raise IntegrityError("branch history has a gap or overlap")
                matrix_entries = chunk["matrices"]
                if not isinstance(matrix_entries, list):
                    raise IntegrityError("invalid chunk matrix records")
                matrix_ids: dict[MatrixKey, str] = {}
                for entry in matrix_entries:
                    if not current_adaptive and any(
                        field in entry for field in ("decision", "binding", "factor_metrics", "reuse_source")
                    ):
                        raise IntegrityError("adaptive matrix evidence requires a version-2 chunk")
                    key = MatrixKey.from_dict(entry["key"])
                    if key in matrix_ids:
                        raise IntegrityError("duplicate matrix in an atomic chunk")
                    object_id = entry["tensor"]
                    if not isinstance(object_id, str) or not object_id.startswith("manifest:"):
                        raise IntegrityError("chunk matrix must reference an immutable tensor manifest")
                    self.store.path_for(object_id)
                    matrix_ids[key] = object_id
                if set(matrix_ids) != set(self._keys):
                    raise IntegrityError("committed chunk does not contain every layer's K and V")
                matrices: dict[MatrixKey, tuple[str, TensorContext, CodecSpec, dict[str, Any]]] = {}
                for key, object_id in matrix_ids.items():
                    if not current_adaptive and (
                        metadata_only or (matrix_key is not None and key != matrix_key)
                    ):
                        continue
                    context, spec, encoded = self._tensor_record(object_id)
                    if context.key != key:
                        raise IntegrityError("matrix key and tensor context disagree")
                    if (context.token_start != end or context.token_end != chunk["token_end"]
                            or (context.frames, context.height, context.width)
                            != (chunk["frames"], chunk["height"], chunk["width"])):
                        raise IntegrityError("chunk positions or geometry disagree across matrices")
                    if key in spaces and spaces[key] != context.key_space:
                        raise IntegrityError("key-space changes within a branch")
                    if key in dtypes and dtypes[key] != context.dtype:
                        raise IntegrityError("source dtype changes within a branch")
                    spaces[key] = context.key_space
                    dtypes[key] = context.dtype
                    matrices[key] = (object_id, context, spec, encoded)
                if current_adaptive:
                    policy, state = self._rank_chunk(chunk, chunks, matrices)
                current_geometry = (chunk["height"], chunk["width"])
                if geometry is not None and geometry != current_geometry:
                    raise IntegrityError("spatial geometry changes within a branch")
                geometry = current_geometry
                end = chunk["token_end"]
                chunks.append({
                    **chunk, "object_id": chunk_object, "records": matrices,
                    **({"parsed_rank_policy": policy, "parsed_rank_state": state}
                       if current_adaptive else {}),
                })
            except (AttributeError, KeyError, TypeError, ValueError, ConfigurationError, StateError) as exc:
                raise IntegrityError("invalid committed chunk") from exc
        return head, chunks

    def _new_head(self, chunks: list[str]) -> str:
        return self.store.put_manifest(
            {"type": "head", "version": 1, "model": self.model.to_dict(), "chunks": chunks},
            references=chunks,
        )

    def create_branch(self, branch_id: str) -> None:
        with self._lock:
            self._open()
            name = self._branch_name(branch_id)
            with self.store.transaction():
                if self.store.get_ref(name) is not None:
                    raise StateError(f"branch already exists: {branch_id}")
                self.store.set_ref(name, self._new_head([]), expected=None)

    def branch_info(self, branch_id: str) -> dict[str, Any]:
        with self._lock, self.store.transaction():
            head_id = self._ready(branch_id)
            _, chunks = self._load_head(head_id, metadata_only=True)
            return {
                "branch_id": branch_id, "head_id": head_id,
                "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
                "chunk_manifests": [chunk["object_id"] for chunk in chunks],
                "token_start": 0, "token_end": chunks[-1]["token_end"] if chunks else 0,
                "chunk_id": chunks[-1]["chunk_id"] if chunks else None,
                "clean": True,
            }

    def require_head(self, branch_id: str, expected_head: str) -> None:
        """Check an already validated epoch identity without rereading its metadata.

        Actual acquisitions still verify their selected manifests and payloads.
        """
        with self._lock, self.store.transaction():
            if self._ready(branch_id) != expected_head:
                raise StateError("branch changed during the bound restoration epoch")

    def history_contexts(self, branch_id: str, key: MatrixKey) -> tuple[TensorContext, ...]:
        if not isinstance(key, MatrixKey) or key not in self._keys:
            raise ConfigurationError("history key is outside the model")
        with self._lock, self.store.transaction():
            _, chunks = self._load_head(self._ready(branch_id), matrix_key=key)
            return tuple(chunk["records"][key][1] for chunk in chunks)

    def history_specs(self, branch_id: str, key: MatrixKey) -> tuple[CodecSpec, ...]:
        if not isinstance(key, MatrixKey) or key not in self._keys:
            raise ConfigurationError("history key is outside the model")
        with self._lock, self.store.transaction():
            _, chunks = self._load_head(self._ready(branch_id), matrix_key=key)
            return tuple(chunk["records"][key][2] for chunk in chunks)

    def history_rank_policy(self, branch_id: str):
        """Return only the immutable policy bound to committed adaptive history."""
        with self._lock, self.store.transaction():
            _, chunks = self._load_head(self._ready(branch_id))
            return chunks[-1].get("parsed_rank_policy") if chunks else None

    def history_rank_state(self, branch_id: str):
        """Return committed scalar selector state; rejected attempts have no state."""
        with self._lock, self.store.transaction():
            _, chunks = self._load_head(self._ready(branch_id))
            return chunks[-1].get("parsed_rank_state") if chunks else None

    def history_rank_decisions(self, branch_id: str, key: MatrixKey):
        if not isinstance(key, MatrixKey) or key not in self._keys:
            raise ConfigurationError("history key is outside the model")
        from .online_rank import RankDecision
        with self._lock, self.store.transaction():
            _, chunks = self._load_head(self._ready(branch_id))
            return tuple(
                RankDecision.from_dict(entry["decision"])
                for chunk in chunks if chunk["version"] == 2
                for entry in chunk["matrices"] if MatrixKey.from_dict(entry["key"]) == key
            )

    def _rank_policy(self, value, *, stored=False):
        from .online_rank import RankPolicy
        error = IntegrityError if stored else ConfigurationError
        try:
            if stored:
                policy = RankPolicy.from_dict(value)
                if policy.to_dict() != value:
                    raise error("adaptive policy has partial or noncanonical fields")
            else:
                if not isinstance(value, RankPolicy):
                    raise error("rank_policy must be an experimental RankPolicy")
                policy = RankPolicy.from_dict(value.to_dict())
            if policy.model != self.model:
                raise error("rank policy model/source identity does not match this runtime")
            from .codecs import _validate_spec
            _validate_spec(policy.codec)
            return policy
        except (AttributeError, TypeError, KeyError, ValueError, ReelCacheError) as exc:
            if not stored and isinstance(exc, (CapacityError, UnsupportedError)):
                raise
            raise error("invalid experimental rank policy") from exc

    def _rank_decision(self, decision, policy, context, spec, binding, chunk_id, parent_head):
        from .online_rank import RankDecision
        try:
            parsed = RankDecision.from_dict(decision)
            if parsed.to_dict() != decision:
                raise IntegrityError("adaptive decision has partial or noncanonical fields")
            policy.validate_decision(parsed, context, spec)
            if (
                parsed.input_sha256 != binding["source_sha256"]
                or parsed.context_sha256 != binding["context_sha256"]
                or parsed.chunk_id != chunk_id or parsed.parent_head != parent_head
                or parsed.basis_sha256 != binding["basis_sha256"]
                or parsed.coefficients_sha256 != binding["coefficients_sha256"]
                or _rank_digest(parsed.metrics) != binding["metrics_sha256"]
            ):
                raise IntegrityError("adaptive decision disagrees with its policy, source, or representation")
            band = 8 * torch.finfo(torch.float32).eps
            if (
                parsed.estimated_rank is None
                or parsed.metrics.get("guard_boundary_band") != band
                or (not parsed.zero_input and parsed.guard_relative_residual != 0
                    and parsed.guard_relative_residual >= parsed.guard_threshold - band)
            ):
                raise IntegrityError("adaptive decision has no resolved, conservative current-input guard")
            return parsed
        except (AttributeError, ReelCacheError, TypeError, ValueError, KeyError) as exc:
            raise IntegrityError("invalid bound adaptive rank decision") from exc

    def _rank_binding(self, binding, policy, context, spec, encoded, factor_metrics):
        if not isinstance(factor_metrics, dict):
            raise IntegrityError("adaptive factor diagnostics must be an object")
        from .codecs import _prepared_binding_tile_rows
        guard_tile_rows = _prepared_binding_tile_rows(binding, factor_metrics)
        if (
            binding["policy_identity"] != policy.fingerprint
            or binding["context_sha256"] != _rank_digest(context.to_dict())
            or binding["spec_sha256"] != _rank_digest(spec.to_dict())
            or binding["metrics_sha256"] != _rank_digest(factor_metrics)
            or guard_tile_rows != policy.tile_rows
            or type(factor_metrics.get("tile_rows")) is not int
            or factor_metrics["tile_rows"] != guard_tile_rows
        ):
            raise IntegrityError("adaptive factor binding is incomplete or incompatible")
        for name in set(binding) - {"schema", "policy_identity", "guard_tile_rows"}:
            value = binding[name]
            if (not isinstance(value, str) or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)):
                raise IntegrityError("adaptive factor binding requires complete SHA256 identities")
        metrics, metadata = encoded["metrics"], encoded["metadata"]
        # Version-2 headers are exactly the svd_hevc packets with an explicit GOP choice.
        expected_schema = (
            "reelcache.tensor.v2" if spec.kind == "svd_hevc" and spec.inter_prediction is not None
            else "reelcache.tensor.v1"
        )
        if (
            not isinstance(metrics, dict) or metrics.get("prepared_binding") != binding
            or metadata.get("schema") != expected_schema
            or metadata.get("coefficient_shape") != [context.token_count, spec.rank]
            or metadata.get("factorization_domain") != (
                "normalized_pre_rope_k" if context.key.target == "K" else "direct_v"
            )
            or encoded["basis"] != "basis:" + binding["basis_sha256"]
        ):
            raise IntegrityError("adaptive packet and accepted factor binding disagree")
        from .codecs import _json_bytes, tensor_metadata_fields
        if set(metadata) != tensor_metadata_fields(expected_schema) or metadata.get("byte_order") != "little":
            raise IntegrityError("adaptive packet has a malformed tensor header")
        if expected_schema == "reelcache.tensor.v2":
            from ._hevc import encoder_record
            if metadata.get("hevc") != encoder_record(spec.inter_prediction):
                raise IntegrityError("adaptive HEVC packet differs from its explicit GOP contract")
        metadata_bytes = len(_json_bytes(metadata))
        payload_bytes = sum(descriptor["bytes"] for descriptor in metadata["payloads"].values())
        basis_bytes = metadata["basis"]["bytes"]
        expected_bytes = {
            "source_bytes": tensor_nbytes(context.shape, context.dtype),
            "basis_bytes": basis_bytes, "payload_bytes": payload_bytes,
            "metadata_bytes": metadata_bytes,
            "representation_bytes": metadata_bytes + payload_bytes + basis_bytes,
        }
        if any(
            type(metrics.get(name)) is not int or metrics[name] != value
            for name, value in expected_bytes.items()
        ):
            raise IntegrityError("adaptive byte accounting disagrees with the immutable representation")
        if (
            factor_metrics.get("projection_convention") != "P=X.float()@B.float().T"
            or factor_metrics.get("source_dtype") != context.dtype
            or factor_metrics.get("basis_dtype") != "float16"
            or factor_metrics.get("coefficient_dtype") != "float32"
            or factor_metrics.get("guard_is_pre_codec") is not True
        ):
            raise IntegrityError("adaptive factors changed their source or precision convention")
        if spec.kind == "raw":
            raise IntegrityError("adaptive factor policy cannot become implicit RAW")
        if spec.kind == "svd" and (
            metadata.get("decoded_coefficients_sha256") != binding["coefficients_sha256"]
            or encoded["payloads"].get("coefficients.f32") != "blob:" + binding["coefficients_sha256"]
        ):
            raise IntegrityError("adaptive SVD packet changed the accepted projection")
        if spec.kind == "svd":
            coefficient = metadata["payloads"].get("coefficients.f32")
            if (
                set(encoded["payloads"]) != {"coefficients.f32"}
                or coefficient.get("dtype") != "float32"
                or coefficient.get("shape") != [context.token_count, spec.rank]
                or coefficient["bytes"] != context.token_count * spec.rank * 4
                or metadata["normalization"] is not None or metadata["video"] is not None
            ):
                raise IntegrityError("adaptive SVD coefficient header is incompatible")
        else:
            from . import _hevc
            normalization = _hevc.normalization_layout(
                spec.rank, spec.normalization, spec.protected_fraction,
            )
            video = _hevc.video_layout(
                context.frames, context.height, context.width, spec.rank, spec.pix_fmt,
            )
            names = {"affine.f32"} | {
                f"coefficients.{index:03d}.hevc" for index in range(video["segments"])
            }
            if (
                metadata["normalization"] != normalization or metadata["video"] != video
                or set(encoded["payloads"]) != names
                or metadata["payloads"]["affine.f32"].get("dtype") != "float32"
                or metadata["payloads"]["affine.f32"].get("shape") != normalization["affine_shape"]
            ):
                raise IntegrityError("adaptive HEVC normalization or native pixel layout is incompatible")
            digest = metadata["decoded_coefficients_sha256"]
            if (
                not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise IntegrityError("adaptive HEVC packet lacks its validation-decode checksum")
        if any(
            metrics.get(name) != value for name, value in factor_metrics.items()
        ):
            raise IntegrityError("adaptive packet changed the factorization diagnostics")

    def _rank_chunk(self, chunk, previous, matrices):
        from .online_rank import RankPreparation, RankState
        policy = self._rank_policy(chunk["rank_policy"], stored=True)
        if policy.fingerprint != chunk["rank_policy_identity"]:
            raise IntegrityError("stored adaptive policy identity is inconsistent")
        if previous and previous[-1]["rank_policy_identity"] != policy.fingerprint:
            raise IntegrityError("adaptive policy changes within a committed history")
        self._rank_contexts(
            policy,
            {key: record[1] for key, record in matrices.items()}, history_chunks=len(previous) + 1,
        )
        decisions = {}
        for entry in chunk["matrices"]:
            if set(entry) != {"key", "tensor", "spec", "decision", "binding", "factor_metrics", "reuse_source"}:
                raise IntegrityError("adaptive matrix lacks its complete resolved specification and evidence")
            key = MatrixKey.from_dict(entry["key"])
            _, context, spec, encoded = matrices[key]
            if spec.to_dict() != entry["spec"]:
                raise IntegrityError("adaptive tensor and committed resolved specification disagree")
            self._rank_binding(entry["binding"], policy, context, spec, encoded, entry["factor_metrics"])
            decision = self._rank_decision(
                entry["decision"], policy, context, spec, entry["binding"],
                chunk["chunk_id"], chunk["parent_head"] if previous else None,
            )
            decisions[key] = decision
            if (
                decision.canonical_basis_rank != encoded["metadata"]["basis"]["shape"][0]
                or decision.basis_bytes != encoded["metadata"]["basis"]["bytes"]
                or decision.coefficient_bytes != context.token_count * spec.rank * 4
                or decision.guard_relative_residual != entry["factor_metrics"].get("projection_relative_residual")
                or decision.zero_input != entry["factor_metrics"].get("zero_input")
            ):
                raise IntegrityError("adaptive decision does not describe its actual factor representation")
            origin = entry["reuse_source"]
            if not policy.reuse_enabled and (
                origin is not None or entry["factor_metrics"].get("reuse_attempts") != 0
                or entry["factor_metrics"].get("reuse_accepted") is not False
            ):
                raise IntegrityError("adaptive policy forbids this basis-reuse attempt")
            if origin is not None:
                if not isinstance(origin, dict) or set(origin) != {"chunk_id", "tensor", "basis"}:
                    raise IntegrityError("adaptive basis provenance is malformed")
                matching = [
                    prior for prior in previous if prior["chunk_id"] == origin["chunk_id"]
                    and prior["records"][key][0] == origin["tensor"]
                    and prior["records"][key][3]["basis"] == origin["basis"]
                ]
                if not matching or origin["basis"] != encoded["basis"]:
                    raise IntegrityError("adaptive reused basis is not from its declared committed ancestor")
            if bool(entry["factor_metrics"].get("reuse_accepted", False)) != (origin is not None):
                raise IntegrityError("adaptive reuse evidence is inconsistent")
        if any(decision.source_provenance for decision in decisions.values()):
            # Reuse the metadata-only admission checks for whole-chunk study identity.
            provenance_check = RankPreparation(
                policy, previous[-1]["parsed_rank_state"] if previous else None,
                chunk_id=chunk["chunk_id"], parent_head=chunk["parent_head"] if previous else None,
                token_start=chunk["token_start"],
                source_provenance={key: decision.source_provenance for key, decision in decisions.items()},
            )
            provenance_check.close()
        self._rank_predictions(
            policy, decisions, previous[-1]["parsed_rank_state"] if previous else RankState(policy.fingerprint),
        )
        saved = chunk["rank_state"]
        expected = {
            "schema": "reelcache.rank_commit_state.v1", "policy_identity": policy.fingerprint,
            "committed_chunks": len(previous) + 1, "chunk_id": chunk["chunk_id"],
            "token_end": chunk["token_end"],
            "prior_state_sha256": (
                _rank_digest(previous[-1]["rank_state"]) if previous else None
            ),
            "decisions_sha256": _rank_digest([entry["decision"] for entry in chunk["matrices"]]),
        }
        if (
            not isinstance(saved, dict) or set(saved) != set(expected) | {"state"}
            or any(
                type(saved[name]) is not type(value) or saved[name] != value
                for name, value in expected.items()
            )
        ):
            raise IntegrityError("adaptive scalar state is not bound to the committed decisions and parent")
        state = RankState.from_dict(saved["state"])
        expected_state = RankState(
            policy_hash=policy.fingerprint, last_chunk_id=chunk["chunk_id"],
            parent_head=chunk["parent_head"] if previous else None,
            token_end=chunk["token_end"], completed_chunks=len(previous) + 1,
            anchor_ranks={key: float(decisions[key].estimated_rank) for key in policy.anchors},
            anchor_chunks={key: len(previous) + 1 for key in policy.anchors},
            selected_ranks={key: decisions[key].rank for key in self._keys},
        )
        if state.to_dict() != saved["state"] or state.to_dict() != expected_state.to_dict():
            raise IntegrityError("adaptive scalar state is partial or noncanonical")
        preparation = chunk["rank_preparation"]
        if (
            not isinstance(preparation, dict) or preparation.get("policy_hash") != policy.fingerprint
            or preparation.get("prepared_matrix_count") != len(self._keys)
            or preparation.get("failed") is not False
            or preparation.get("logical_source_bytes_per_chunk") != policy.source_bytes_per_chunk
        ):
            raise IntegrityError("adaptive preparation summary is incomplete")
        for name in ("decomposition_count", "svd_count", "eigh_count", "qr_count", "projection_count"):
            if (
                type(preparation.get(name)) is not int
                or preparation[name] != sum(decision.metrics.get(name, 0) for decision in decisions.values())
            ):
                raise IntegrityError("adaptive preparation summary does not match its matrix work")
        return policy, state

    def _rank_contexts(self, policy, contexts, *, history_chunks):
        if not isinstance(contexts, Mapping) or set(contexts) != set(self._keys):
            raise ConfigurationError("adaptive policy requires the entire model's K/V context set")
        policy.validate_contexts(contexts, history_chunks=history_chunks)

    def _builtin_prepared_encoder(self, encoder):
        from .codecs import HEVCProvider, encode_tensor
        owner = getattr(encoder, "__self__", None)
        return encoder is encode_tensor or (
            type(owner) is HEVCProvider
            and getattr(encoder, "__func__", None) is HEVCProvider.encode_tensor
        )

    def _rank_predictions(self, policy, decisions, prior):
        ordinal = prior.completed_chunks + 1
        anchors = {key: float(decisions[key].estimated_rank) for key in policy.anchors}
        for key, decision in decisions.items():
            predicted = policy.predict_rank(
                key, current_anchors=anchors, prior_state=prior,
            )
            if (
                json_copy(decision.prediction_sources) != json_copy(predicted.prediction_sources)
                or decision.predicted_rank != predicted.predicted_rank
                or decision.initial_rank != predicted.initial_rank
                or decision.metrics.get("prediction_kind") != predicted.kind
                or decision.metrics.get("predictor_profile_fingerprint") != predicted.profile_fingerprint
                or decision.metrics.get("factorization_seed")
                != policy.factorization_seed_for(key, decision.context.token_start)
                or decision.metrics.get("chunk_ordinal") != ordinal
                or decision.metrics.get("prior_state_fingerprint") != prior.fingerprint
            ):
                raise IntegrityError("adaptive prediction or solver state uses incompatible history/anchors")

    def _validate_inputs(
        self, chunk_id: Any, matrices: Mapping[MatrixKey, torch.Tensor],
        contexts: Mapping[MatrixKey, TensorContext],
        specs: Mapping[MatrixKey, CodecSpec] | CodecSpec, chunks: list[dict[str, Any]],
    ) -> dict[MatrixKey, CodecSpec]:
        self._chunk_id(chunk_id)
        if not isinstance(matrices, Mapping) or not isinstance(contexts, Mapping):
            raise ConfigurationError("matrices and contexts must be keyed mappings")
        required = set(self._keys)
        if (set(matrices) != required or set(contexts) != required
                or len(matrices) != len(required) or len(contexts) != len(required)):
            raise ConfigurationError("a clean chunk requires exactly all 2*layers K/V matrices")
        if isinstance(specs, CodecSpec):
            selected = {key: specs for key in self._keys}
        elif isinstance(specs, Mapping) and set(specs) == required:
            selected = dict(specs)
        else:
            raise ConfigurationError("specs must be a CodecSpec or a complete matrix-keyed mapping")
        if any(not isinstance(spec, CodecSpec) for spec in selected.values()):
            raise ConfigurationError("each matrix requires a CodecSpec")
        if any(chunk["chunk_id"] == chunk_id for chunk in chunks):
            raise StateError("chunk_id has already been committed on this branch")
        end = chunks[-1]["token_end"] if chunks else 0
        reference = None
        for key in self._keys:
            context, spec = contexts[key], selected[key]
            if not isinstance(context, TensorContext) or context.model != self.model or context.key != key:
                raise ConfigurationError("matrix context model identity or key is incompatible")
            geometry = (context.token_start, context.frames, context.height, context.width)
            if reference is None:
                reference = geometry
            if geometry != reference or context.token_start != end:
                raise ConfigurationError("chunk positions/geometry must match with no history gaps or overlaps")
            if chunks:
                previous = chunks[-1]["records"][key][1]
                if (context.height, context.width, context.key_space, context.dtype) != (
                    previous.height, previous.width, previous.key_space, previous.dtype
                ):
                    raise ConfigurationError("chunk spatial geometry, key-space, or source dtype is incompatible")
            if key.target == "K" and context.key_space == "post_rope" and spec.kind != "raw":
                raise ConfigurationError("SVD of post-RoPE K is forbidden; store RAW or pre-RoPE K")
            if spec.rank is not None and spec.rank > min(context.shape):
                raise ConfigurationError("matrix rank exceeds its full-channel factorization domain")
        return selected

    def _validate_matrix(self, tensor: Any, context: TensorContext) -> None:
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.device.type == "meta" or tuple(tensor.shape) != context.shape
                or dtype_name(tensor.dtype) != context.dtype):
            raise ConfigurationError("matrix tensor shape/dtype/layout does not match its context")
        if self.executor is not None and tensor.device.type != "cpu":
            raise UnsupportedError("stage native tensors on CPU before unordered codec executor work")

    def _reuse_basis(
        self, key: MatrixKey, spec: CodecSpec, chunks: list[dict[str, Any]]
    ) -> torch.Tensor | None:
        if spec.kind == "raw":
            return None
        for chunk in reversed(chunks):
            _, context, previous_spec, encoded = chunk["records"][key]
            if encoded["basis"] is None:
                continue
            shape = self._basis_geometry(encoded["metadata"], context, previous_spec)
            if shape[0] < spec.rank:
                continue
            return tensor_from_bytes(self.store.get(encoded["basis"]), dtype="float16", shape=shape)
        return None

    def _rank_reuse_basis(self, key, chunks):
        for chunk in reversed(chunks):
            object_id, context, spec, encoded = chunk["records"][key]
            if encoded["basis"] is not None:
                shape = self._basis_geometry(encoded["metadata"], context, spec)
                return (
                    tensor_from_bytes(self.store.get(encoded["basis"]), dtype="float16", shape=shape),
                    {"chunk_id": chunk["chunk_id"], "tensor": object_id, "basis": encoded["basis"]},
                )
        return None, None

    def _save_tensor(self, encoded: EncodedTensor, context: TensorContext, spec: CodecSpec) -> str:
        if not isinstance(encoded, EncodedTensor):
            raise IntegrityError("representation provider must return EncodedTensor")
        self._encoded_geometry(encoded.metadata, spec, context, basis_present=encoded.basis is not None)
        if encoded.basis is not None:
            shape = self._basis_geometry(encoded.metadata, context, spec)
            if len(encoded.basis) != tensor_nbytes(shape, "float16"):
                raise IntegrityError("canonical basis payload does not match its full stored geometry")
        self._payload_descriptors(encoded.metadata, encoded.payloads)
        payloads = {name: self.store.put(payload) for name, payload in encoded.payloads.items()}
        basis = None if encoded.basis is None else self.store.put(encoded.basis, kind="basis")
        self._stored_descriptors(encoded.metadata, payloads, basis)
        references = list(payloads.values()) + ([] if basis is None else [basis])
        return self.store.put_manifest(
            {
                "type": "tensor", "version": 1, "context": context.to_dict(), "spec": spec.to_dict(),
                "encoded": {"metadata": encoded.metadata, "payloads": payloads,
                            "basis": basis, "metrics": encoded.metrics},
            },
            references=references,
        )

    def seal_chunk(
        self, branch_id: str, chunk_id: Any, matrices: Mapping[MatrixKey, torch.Tensor],
        contexts: Mapping[MatrixKey, TensorContext],
        specs: Mapping[MatrixKey, CodecSpec] | CodecSpec = CodecSpec(), *,
        clean: bool = True, reuse_threshold: float | None = None,
        rank_policy=None, source_provenance: Mapping[MatrixKey, Mapping[str, Any]] | None = None,
    ) -> str:
        """Synchronously publish a complete clean chunk; inputs stay unchanged until return."""
        with self._lock:
            self._check_async_errors()
        return self._seal_chunk(
            branch_id, chunk_id, matrices, contexts, specs, clean=clean,
            reuse_threshold=reuse_threshold, rank_policy=rank_policy, source_provenance=source_provenance,
        )

    def open_chunk(
        self, branch_id: str, chunk_id: Any, contexts: Mapping[MatrixKey, TensorContext],
        specs: Mapping[MatrixKey, CodecSpec] | CodecSpec = CodecSpec(), *, executor: BoundedExecutor | None = None,
    ) -> StreamingChunk:
        """Reserve ``branch_id`` and return a :class:`StreamingChunk` for layer-wise publication.

        Static specs only (no rank policy or basis reuse). ``executor`` (optional,
        distinct from any commit coordinator) runs encodes in parallel; matrices
        must then be CPU tensors. Until commit/abort the branch rejects other
        writers (seal, submit, open, restore-onto, delete); reads keep seeing
        the committed head, so native forwards may consume existing history.
        """
        if executor is not None and not isinstance(executor, BoundedExecutor):
            raise ConfigurationError("streaming encode executor must be a BoundedExecutor")
        if not isinstance(contexts, Mapping):
            raise ConfigurationError("contexts must be a matrix-keyed mapping")
        contexts = dict(contexts)
        with self._lock:
            self._check_async_errors()
            with self.store.transaction():
                head_id = self._ready(branch_id)
                self._check_not_streaming(branch_id)
                head, chunks = self._load_head(head_id)
                if chunks and chunks[-1]["version"] == 2:
                    raise ConfigurationError("adaptive history requires its bound rank policy on every chunk")
                selected = self._validate_inputs(chunk_id, {key: None for key in self._keys}, contexts, specs, chunks)
                pin = self.store.lease(head_id)
                self._streams.add(branch_id)
        return StreamingChunk(self, branch_id, chunk_id, contexts, selected, head_id, head, pin, executor,
                              committed_ids=[chunk["chunk_id"] for chunk in chunks])

    def submit_chunk(
        self, branch_id, chunk_id, matrices, contexts, specs=CodecSpec(), *,
        commit_executor: BoundedExecutor, clean=True, allow_lossy=False, reuse_threshold=None,
    ) -> PendingCommit:
        """Reserve a branch, admit/copy CPU inputs, then enqueue static publication.

        Copying completes before this method returns. The caller must not mutate
        inputs during submission, but may release/reuse them afterward. This API
        requires explicit lossy opt-in and never admits adaptive history or
        switches the committed domain/precision for an existing matrix.
        """
        if not isinstance(commit_executor, BoundedExecutor) or commit_executor is self.executor:
            raise ConfigurationError("use a separate BoundedExecutor for clean-commit coordination")
        if clean is not True:
            raise PendingError("only complete clean chunks can be submitted")
        if type(allow_lossy) is not bool:
            raise ConfigurationError("allow_lossy must be boolean")
        if reuse_threshold is not None:
            from .reuse import validate_threshold
            reuse_threshold = validate_threshold(reuse_threshold)
        if not isinstance(contexts, Mapping):
            raise ConfigurationError("contexts must be a matrix-keyed mapping")
        contexts = dict(contexts)
        with self._lock:
            self._check_async_errors()
            with self.store.transaction():
                parent = self._ready(branch_id)
                _, chunks = self._load_head(parent)
                selected = self._validate_inputs(chunk_id, matrices, contexts, specs, chunks)
                if any(chunk["version"] != 1 for chunk in chunks):
                    raise UnsupportedError("asynchronous clean publication requires static history")
                if not allow_lossy and any(spec.kind != "raw" for spec in selected.values()):
                    raise UnsupportedError("lossy asynchronous publication requires explicit allow_lossy=True")
                required = sum(tensor_nbytes(ctx.shape, resolve_dtype(ctx.dtype)) for ctx in contexts.values())
                if self._pending_source_bytes + required > self.max_pending_source_bytes:
                    raise CapacityError("pending clean-source copies exceed the runtime byte budget")
                if required > commit_executor.max_staging_bytes:
                    raise CapacityError("clean-source copies exceed the coordinator staging budget")
                pin = self.store.lease(parent)
            self._check_not_streaming(branch_id)
            state = _CommitState(branch_id, parent, required, pin)
            self._async_states[branch_id] = state
            self._pending.add(branch_id)
            self._pending_source_bytes += required
            self._peak_pending_source_bytes = max(self._peak_pending_source_bytes, self._pending_source_bytes)

        owned = {}

        def run():
            try:
                with self._lock:
                    self._check_commit_cancel(state)
                    state.phase = "encoding"
                return self._seal_chunk(
                    branch_id, chunk_id, owned, contexts, selected, reservation=state,
                    reuse_threshold=reuse_threshold,
                )
            finally:
                owned.clear()

        def finalize(job):
            error = None
            try:
                job.result()
            except BaseException as exc:
                error = exc
            finally:
                owned.clear()
                try:
                    pin.release()
                except BaseException as exc:
                    if error is None:
                        error = exc
                    else:
                        cleanup = StateError(f"commit failed ({error}); pin cleanup also failed ({exc})")
                        cleanup.__cause__ = error
                        error = cleanup
                with self._lock:
                    self._pending_source_bytes -= required
                    self._async_states.pop(branch_id, None)
                    self._pending.discard(branch_id)
                    state.error = error
                    if error is not None:
                        if state.result_id is None:
                            state.phase = "cancelled" if isinstance(error, CancelledError) else "failed"
                        if not isinstance(error, CancelledError):
                            self._async_failures[branch_id] = state
                    state.finished.set()

        try:
            for key in self._keys:
                tensor = copied = None
                try:
                    tensor = matrices[key]
                    self._validate_matrix(tensor, contexts[key])
                    if tensor.device.type != "cpu":
                        raise UnsupportedError("stage RAW sources on CPU before asynchronous submission")
                    copied = tensor.detach().clone(memory_format=torch.contiguous_format)
                    self._validate_matrix(copied, contexts[key])
                    if (
                        copied.device.type != "cpu" or not copied.is_contiguous()
                        or copied.untyped_storage().nbytes() != tensor_nbytes(contexts[key].shape, copied.dtype)
                        or copied.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
                    ):
                        raise IntegrityError("asynchronous source copy must own its exact admitted CPU storage")
                    owned[key] = copied
                finally:
                    tensor = copied = None
            job = commit_executor.submit(
                run, staging_bytes=required, label=f"commit:{branch_id}:{chunk_id}",
            )
            state.job = job
        except BaseException as exc:
            owned.clear()
            traceback.clear_frames(exc.__traceback__)
            try:
                pin.release()
            finally:
                with self._lock:
                    self._pending_source_bytes -= required
                    self._async_states.pop(branch_id, None)
                    self._pending.discard(branch_id)
            raise
        job.add_done_callback(finalize)
        return PendingCommit(self, state)

    def _check_commit_cancel(self, state):
        if state is not None:
            with self._lock:
                if state.cancel_requested:
                    raise CancelledError("clean commit cancelled before publication")

    def _check_async_errors(self):
        if self._async_failures:
            detail = "; ".join(f"{branch}: {state.error}" for branch, state in self._async_failures.items())
            raise StateError(f"unobserved clean-commit failure; retrieve handle.result(): {detail}")

    def async_stats(self):
        with self._lock:
            return {
                "pending_commits": len(self._async_states),
                "pending_source_bytes": self._pending_source_bytes,
                "peak_pending_source_bytes": self._peak_pending_source_bytes,
                "max_pending_source_bytes": self.max_pending_source_bytes,
                "unobserved_failures": len(self._async_failures),
            }

    def _seal_chunk(
        self, branch_id: str, chunk_id: Any, matrices: Mapping[MatrixKey, torch.Tensor],
        contexts: Mapping[MatrixKey, TensorContext],
        specs: Mapping[MatrixKey, CodecSpec] | CodecSpec = CodecSpec(), *,
        clean: bool = True, reuse_threshold: float | None = None,
        rank_policy=None, source_provenance: Mapping[MatrixKey, Mapping[str, Any]] | None = None,
        reservation: _CommitState | None = None,
    ) -> str:
        """Stream each matrix exactly once and atomically publish all layers.

        This call is synchronous, including when an executor is supplied. Keep
        source tensors unchanged until it returns. Enqueuing this entire method
        externally does not reserve the branch while the call is still queued
        and is not a supported asynchronous commit protocol.

        ``matrices`` may be a lazy disk-backed mapping. Encoded packets are
        persisted and GC-pinned individually; neither decoded matrices nor byte
        packets accumulate across layers. Only the final complete head is made
        visible. Failed staging is reclaimed without changing the old branch.
        ``rank_policy`` is an explicit experimental alternative to ``specs``;
        it cannot switch an existing static history or override its source
        domain. Only successfully published decisions advance selector state.
        Experimental HEVC uses the built-in encoder or ``HEVCProvider``: a
        third-party bitstream cannot prove which QP/source pixels produced it.
        ``source_provenance`` is an optional complete adaptive-only mapping,
        frozen and validated before source fitting or pending-state mutation.
        It binds original study identities to the same exact-oracle solve.
        """
        if clean is not True:
            raise PendingError("only complete clean chunks can be committed")
        if rank_policy is not None and self.encode_concurrency > 1:
            raise UnsupportedError("adaptive rank preparation requires ordered serial encoding")
        if reuse_threshold is not None and (
            isinstance(reuse_threshold, bool) or not isinstance(reuse_threshold, (int, float))
            or not math.isfinite(reuse_threshold) or not 0 <= reuse_threshold <= 1
        ):
            raise ConfigurationError("reuse_threshold must be a finite relative residual in [0, 1]")
        pin = None
        staged_pins = []
        staged_objects: list[str] = []
        published = False
        preparation = sources = checked_policy = None
        encoder = self.encoder
        if not isinstance(contexts, Mapping):
            raise ConfigurationError("contexts must be a matrix-keyed mapping")
        contexts = dict(contexts)
        if source_provenance is not None:
            if (
                rank_policy is None or not isinstance(source_provenance, Mapping)
                or set(source_provenance) != set(self._keys)
            ):
                raise ConfigurationError("source provenance requires an adaptive policy and every matrix key")
            source_provenance = {
                key: json_copy(source_provenance[key], name="source provenance") for key in self._keys
            }
            if any(
                not isinstance(row, Mapping)
                or ("branch_id" in row and row["branch_id"] != branch_id)
                for row in source_provenance.values()
            ):
                raise ConfigurationError("source provenance must describe this runtime branch")
        with self._lock, self.store.transaction():
            if reservation is None:
                head_id = self._ready(branch_id)
            else:
                self._open()
                if self._async_states.get(branch_id) is not reservation:
                    raise StateError("clean-commit reservation was lost")
                self._check_commit_cancel(reservation)
                head_id = self.store.get_ref(self._branch_name(branch_id))
                if head_id != reservation.parent_head:
                    raise StateError("branch changed after asynchronous clean submission")
            head, chunks = self._load_head(head_id)
            if rank_policy is not None:
                if not isinstance(specs, CodecSpec) or specs != CodecSpec() or reuse_threshold is not None:
                    raise ConfigurationError("rank_policy owns resolved specs and reuse; do not supply overrides")
                if not isinstance(chunk_id, str) or not chunk_id:
                    raise ConfigurationError("experimental rank decisions require a nonempty string chunk_id")
                checked_policy = self._rank_policy(rank_policy)
                if checked_policy.codec.kind == "svd_hevc" and not self._builtin_prepared_encoder(encoder):
                    raise UnsupportedError(
                        "adaptive HEVC requires the built-in encoder or exact HEVCProvider"
                    )
                if chunks and (
                    chunks[-1]["version"] != 2
                    or chunks[-1]["rank_policy_identity"] != checked_policy.fingerprint
                ):
                    raise ConfigurationError("rank policy cannot switch an incompatible committed history")
                self._rank_contexts(checked_policy, contexts, history_chunks=len(chunks) + 1)
                input_specs = {
                    key: checked_policy.spec_for(key, checked_policy.allowed_ranks[key][0])
                    for key in self._keys
                }
            else:
                if chunks and chunks[-1]["version"] == 2:
                    raise ConfigurationError("adaptive history requires its bound rank policy on every chunk")
                input_specs = specs
            selected = self._validate_inputs(chunk_id, matrices, contexts, input_specs, chunks)
            if checked_policy is not None:
                from .online_rank import RankPreparation
                preparation = RankPreparation(
                    checked_policy,
                    prior_state=chunks[-1]["parsed_rank_state"] if chunks else None,
                    chunk_id=chunk_id, parent_head=head_id if chunks else None,
                    token_start=contexts[self._keys[0]].token_start,
                    source_provenance=source_provenance,
                )
            self._check_not_streaming(branch_id)
            pin = self.store.lease(head_id)
            self._pending.add(branch_id)
        prefit_scope = contextlib.ExitStack()
        try:
            if checked_policy is not None:
                from .online_rank import PreparedRankMatrix
                sources = _CommitSources(matrices, contexts, self._keys, self._validate_matrix)
                if checked_policy.anchor_mode == "staged":
                    preparation.prepare_staged(sources, contexts)
            records = []
            if (
                self.prefit is not None and preparation is None and sources is None
                and self.encode_concurrency == 1 and reuse_threshold is None
            ):
                lossy = [key for key in self._keys if selected[key].kind != "raw"]
                if lossy:
                    with span("seal.prefit"):
                        prefit_scope.enter_context(self.prefit(
                            {key: matrices[key] for key in lossy},
                            {key: contexts[key] for key in lossy},
                            {key: selected[key] for key in lossy},
                        ))
            if self.encode_concurrency > 1:
                records = self._parallel_static_packets(
                    matrices, contexts, selected, chunks, reuse_threshold, staged_pins, staged_objects,
                )
            for key in (() if self.encode_concurrency > 1 else self._keys):
                self._check_commit_cancel(reservation)
                tensor = encoded = basis = kwargs = job = prepared = binding = None
                origin = decision = factor_metrics = None
                binding_seconds = 0.0
                try:
                    if sources is None:
                        try:
                            tensor = matrices[key]
                        except KeyError as exc:
                            raise ConfigurationError(f"matrix disappeared during clean commit: {key}") from exc
                    else:
                        tensor = sources.take(key)
                    self._validate_matrix(tensor, contexts[key])
                    if preparation is None:
                        basis = self._reuse_basis(key, selected[key], chunks) if reuse_threshold is not None else None
                        kwargs = {
                            "reuse_basis": basis,
                            "reuse_threshold": reuse_threshold if selected[key].kind != "raw" else None,
                        }
                    else:
                        from .codecs import (
                            factor_binding, prepared_validation_workspace_bytes,
                            validate_prepared_factors, validate_prepared_packet,
                        )
                        if checked_policy.reuse_enabled:
                            basis, origin = self._rank_reuse_basis(key, chunks)
                        prepared = preparation.prepare_matrix(tensor, contexts[key], reuse_basis=basis)
                        if not isinstance(prepared, PreparedRankMatrix) or not isinstance(prepared.spec, CodecSpec):
                            raise IntegrityError("rank preparation returned no resolved CodecSpec")
                        selected[key] = prepared.spec
                        verification_bytes = prepared_validation_workspace_bytes(
                            contexts[key], prepared.spec, tile_rows=checked_policy.tile_rows,
                        )
                        verification_bytes += prepared.factors.basis.numel() * prepared.factors.basis.element_size()
                        verification_bytes += prepared.factors.coefficients.numel() * 4
                        verification_bytes += int(preparation.metrics["cached_factor_bytes"])
                        if verification_bytes > checked_policy.workspace_limit_bytes:
                            raise CapacityError("prepared-factor verification exceeds the rank-policy workspace")
                        binding_started = time.perf_counter()
                        binding = factor_binding(
                            tensor, contexts[key], prepared.spec, prepared.factors, checked_policy.fingerprint,
                        )
                        binding_seconds = time.perf_counter() - binding_started
                        factor_metrics = json_copy(prepared.factors.metrics)
                        decision = json_copy(prepared.decision.to_dict())
                        self._rank_decision(
                            decision, checked_policy, contexts[key], prepared.spec, binding,
                            chunk_id, head_id if chunks else None,
                        )
                        reused = bool(factor_metrics.get("reuse_accepted", False))
                        if reused and (
                            origin is None or origin["basis"] != "basis:" + binding["basis_sha256"]
                        ):
                            raise IntegrityError("rank preparation did not use its declared ancestor basis")
                        if not reused:
                            origin = None
                        kwargs = {
                            "prepared_factors": prepared.factors,
                            "prepared_binding": json_copy(binding),
                        }
                    if self.executor is None:
                        encoded = _encode_with_metrics(encoder, tensor, contexts[key], selected[key], **kwargs)
                    else:
                        staged_tensors = [tensor] + ([] if basis is None else [basis])
                        if prepared is not None:
                            staged_tensors += [prepared.factors.basis, prepared.factors.coefficients]
                        storages = {
                            (value.device, value.untyped_storage().data_ptr()): value.untyped_storage().nbytes()
                            for value in staged_tensors
                        }
                        staging = sum(storages.values())
                        del staged_tensors, storages
                        job = self.executor.submit(
                            _encode_with_metrics, encoder, tensor, contexts[key], selected[key],
                            staging_bytes=staging,
                            label=f"encode:{branch_id}:{chunk_id}:{key.layer}:{key.target}", **kwargs,
                        )
                        try:
                            with span("seal.wait"):
                                encoded = job.result()
                        except BaseException:
                            done = getattr(job, "done", None)
                            cancel = getattr(job, "cancel", None)
                            if callable(done) and callable(cancel) and not done():
                                if not cancel():
                                    try:
                                        job.result()
                                    except BaseException:
                                        pass
                            raise
                    if prepared is not None:
                        validation = {
                            "scope": "builtin_provider_already_verified",
                            "projection_count": 0, "decomposition_count": 0,
                        } if self._builtin_prepared_encoder(encoder) else validate_prepared_factors(
                            tensor, contexts[key], selected[key], prepared.factors, binding,
                        )
                        provider = getattr(encoder, "__self__", None)
                        packet_validation = validate_prepared_packet(
                            encoded, contexts[key], selected[key], prepared.factors, binding,
                            source=tensor, ffmpeg_path=getattr(provider, "ffmpeg_path", None),
                        )
                        encoded = EncodedTensor(
                            encoded.metadata, encoded.payloads, encoded.basis,
                            {
                                **encoded.metrics, "runtime_prepared_validation": validation,
                                "runtime_prepared_packet_validation": packet_validation,
                                "rank_binding_seconds": binding_seconds,
                                "rank_source_buffer_peak_bytes": sources.peak_bytes,
                            },
                        )
                    with span("seal.persist"), self.store.transaction():
                        packet = self._save_tensor(encoded, contexts[key], selected[key])
                        stage_pin = self.store.lease(packet, purpose="staging")
                        staged_pins.append(stage_pin)
                        staged_objects.extend(self.store.touched_objects)
                        records.append({
                            "key": key.to_dict(), "tensor": packet,
                            **({
                                "spec": selected[key].to_dict(), "decision": decision,
                                "binding": binding, "factor_metrics": factor_metrics,
                                "reuse_source": origin,
                            } if preparation is not None else {}),
                        })
                finally:
                    tensor = encoded = basis = kwargs = job = prepared = binding = None
                    origin = decision = factor_metrics = None
            rank_state = rank_preparation = None
            if preparation is not None:
                rank_state = {
                    "schema": "reelcache.rank_commit_state.v1", "policy_identity": checked_policy.fingerprint,
                    "committed_chunks": len(chunks) + 1, "chunk_id": chunk_id,
                    "token_end": contexts[self._keys[0]].token_end,
                    "prior_state_sha256": _rank_digest(chunks[-1]["rank_state"]) if chunks else None,
                    "decisions_sha256": _rank_digest([entry["decision"] for entry in records]),
                    "state": json_copy(preparation.next_state().to_dict()),
                }
                rank_preparation = {
                    **json_copy(preparation.metrics),
                    "runtime_source_buffer_peak_bytes": sources.peak_bytes,
                    "runtime_source_load_seconds": sources.load_seconds,
                }
                preparation.close()
                preparation = None
                sources.close()
                sources = None
            with span("seal.publish"), self._lock:
                self._open()
                self._check_commit_cancel(reservation)
                if reservation is not None:
                    reservation.phase = "ready"
                with self.store.transaction():
                    context = contexts[self._keys[0]]
                    chunk_data = {
                        "type": "chunk", "version": 1 if checked_policy is None else 2, "chunk_id": chunk_id,
                        "model": self.model.to_dict(), "clean": True,
                        "token_start": context.token_start, "token_end": context.token_end,
                        "frames": context.frames, "height": context.height, "width": context.width,
                        "matrices": records,
                    }
                    references = [entry["tensor"] for entry in records]
                    if checked_policy is not None:
                        chunk_data.update(
                            parent_head=head_id, rank_policy=checked_policy.to_dict(),
                            rank_policy_identity=checked_policy.fingerprint, rank_state=rank_state,
                            rank_preparation=rank_preparation,
                        )
                        references.append(head_id)
                        verified = {}
                        for entry in records:
                            stored_context, stored_spec, stored_encoded = self._tensor_record(entry["tensor"])
                            verified[stored_context.key] = (
                                entry["tensor"], stored_context, stored_spec, stored_encoded,
                            )
                        self._rank_chunk(chunk_data, chunks, verified)
                    chunk_object = self.store.put_manifest(chunk_data, references=references)
                    next_head = self._new_head(head["chunks"] + [chunk_object])
                    self.store.set_ref(self._branch_name(branch_id), next_head, expected=head_id)
                published = True
                if reservation is not None:
                    reservation.result_id = chunk_object
                    reservation.phase = "published"
            return chunk_object
        finally:
            try:
                try:
                    if preparation is not None:
                        preparation.close()
                finally:
                    prefit_scope.close()
                    if sources is not None:
                        sources.close()
                    with self.store.transaction():
                        for staged_pin in staged_pins:
                            staged_pin.release()
                        if pin is not None:
                            pin.release()
                    if not published:
                        self.store.collect_garbage(candidates=staged_objects)
            finally:
                with self._lock:
                    if reservation is None:
                        self._pending.discard(branch_id)

    def _parallel_static_packets(
        self, matrices, contexts, specs, chunks, reuse_threshold, staged_pins, staged_objects,
    ):
        from .codecs import encoded_nbytes

        def encode_and_stage(tensor, key, basis):
            encoded = _encode_with_metrics(
                self.encoder, tensor, contexts[key], specs[key], reuse_basis=basis,
                reuse_threshold=reuse_threshold if specs[key].kind != "raw" else None,
            )
            if encoded_nbytes(encoded, include_metrics=True) > self.encode_workspace_bytes:
                raise CapacityError("encoded packet exceeds its declared parallel workspace reservation")
            with span("seal.persist"), self.store.transaction():
                try:
                    packet = self._save_tensor(encoded, contexts[key], specs[key])
                    staged_pins.append(self.store.lease(packet, purpose="staging"))
                    return {"key": key.to_dict(), "tensor": packet}
                finally:
                    staged_objects.extend(self.store.touched_objects)

        pending = deque()
        records, failures = [], []

        def collect():
            key, job = pending[0]
            try:
                with span("seal.wait"):
                    record = job.result()
                records.append(record)
            finally:
                if job.done():
                    pending.popleft()

        try:
            for key in self._keys:
                if len(pending) == self.encode_concurrency:
                    collect()
                tensor = basis = None
                try:
                    tensor = matrices[key]
                    self._validate_matrix(tensor, contexts[key])
                    if reuse_threshold is not None:
                        basis = self._reuse_basis(key, specs[key], chunks)
                    values = [tensor] + ([] if basis is None else [basis])
                    backing = {
                        (value.device, value.untyped_storage().data_ptr()): value.untyped_storage().nbytes()
                        for value in values
                    }
                    staging = sum(backing.values()) + self.encode_workspace_bytes
                    del values, backing
                    job = self.executor.submit(
                        encode_and_stage, tensor, key, basis, staging_bytes=staging,
                        label=f"encode-static:{key.layer}:{key.target}",
                    )
                    pending.append((key, job))
                finally:
                    tensor = basis = None
        except BaseException as exc:
            failures.append(exc)
        finally:
            while pending:
                try:
                    collect()
                except BaseException as exc:
                    failures.append(exc)
        if failures:
            if len(failures) == 1:
                raise failures[0]
            detail = "; ".join(f"{type(error).__name__}: {error}" for error in failures)
            raise StateError(f"parallel encoding failed: {detail}") from failures[0]
        return records

    def _snapshot_metadata(
        self, metadata: Mapping[str, Any], chunks: list[dict[str, Any]], *, stored: bool = False
    ) -> dict[str, Any]:
        error = IntegrityError if stored else ConfigurationError
        if not isinstance(metadata, Mapping):
            raise error("snapshot metadata must be a mapping")
        try:
            copied = json_copy(metadata)
        except ConfigurationError as exc:
            raise error("snapshot metadata must be portable JSON") from exc
        required = {"clean_boundary", "conditioning", "execution", "rng", "latent", "vae"}
        if not required <= set(copied):
            raise error("snapshot metadata must record boundary/conditioning/execution/RNG/latent/VAE contracts")
        boundary = copied["clean_boundary"]
        end = chunks[-1]["token_end"] if chunks else 0
        chunk_id = chunks[-1]["chunk_id"] if chunks else None
        if (not isinstance(boundary, dict) or boundary.get("token_end") != end
                or boundary.get("chunk_id") != chunk_id or "chunk_id" not in boundary
                or type(boundary.get("token_end")) is not int
                or boundary.get("clean", True) is not True):
            raise error("snapshot boundary does not match the committed clean history")
        if copied.get("pending", False) or (
            isinstance(copied["execution"], dict) and copied["execution"].get("pending", False)
        ):
            raise PendingError("pending native execution cannot be snapshotted")
        for field in required - {"clean_boundary"}:
            if copied[field] is None:
                raise error(f"snapshot {field} contract must be explicit, not null")
        return copied

    def snapshot(
        self, branch_id: str, *, metadata: Mapping[str, Any],
        tensors: Mapping[str, torch.Tensor] | None = None,
    ) -> str:
        """Persist a clean history root and portable named native tensor state.

        Metadata must explicitly record clean_boundary, conditioning, execution,
        rng, latent, and vae. The boundary's token_end and chunk_id are checked
        against committed history. Adapter-specific contracts remain the native
        adapter's responsibility, not an assumed RAW continuation guarantee.
        """
        with self._lock, self.store.transaction():
            head_id = self._ready(branch_id)
            _, chunks = self._load_head(head_id)
            checked = self._snapshot_metadata(metadata, chunks)
            state = serialize_tensors({} if tensors is None else tensors)
            state_id = self.store.put(state)
            snapshot_id = self.store.put_manifest(
                {
                    "type": "snapshot", "version": 1, "model": self.model.to_dict(),
                    "head": head_id, "metadata": checked, "tensor_state": state_id,
                },
                references=[head_id, state_id],
            )
            self.store.set_ref("snapshots/" + snapshot_id.split(":")[1], snapshot_id)
            return snapshot_id

    def _snapshot_record(
        self, snapshot_id: str, *, verify: bool = True, include_rank_contract: bool = False
    ) -> tuple:
        if self._quarantined_objects:
            self._check_quarantined(snapshot_id)
        data = self._manifest(snapshot_id, "snapshot")
        self._same_model(data.get("model"))
        try:
            if not isinstance(data["tensor_state"], str) or not data["tensor_state"].startswith("blob:"):
                raise IntegrityError("snapshot tensor state must be an immutable blob")
            _, chunks = self._load_head(data["head"])
            metadata = self._snapshot_metadata(data["metadata"], chunks, stored=True)
            if verify:
                self._audit(snapshot_id)
            if include_rank_contract:
                return (
                    data, metadata,
                    chunks[-1].get("parsed_rank_policy") if chunks else None,
                    chunks[-1].get("parsed_rank_state") if chunks else None,
                )
            return data, metadata
        except (KeyError, TypeError, ValueError) as exc:
            raise IntegrityError("invalid snapshot record") from exc

    def load_snapshot(self, snapshot_id: str) -> SnapshotState:
        """Read verified native and committed rank state without changing a branch."""
        with self._lock, self.store.transaction():
            self._open()
            record, metadata, policy, state = self._snapshot_record(snapshot_id, include_rank_contract=True)
            tensors = deserialize_tensors(self.store.get(record["tensor_state"]))
            return SnapshotState(snapshot_id, metadata, tensors, policy, state)

    def _restore(
        self, snapshot_id: str, branch_id: str, *, new_only: bool,
        validator: Callable[[RestoredState], None] | None,
        metadata_validator: Callable[[RestoredState], None] | None,
        rank_policy=None,
    ) -> RestoredState:
        if validator is not None and not callable(validator):
            raise ConfigurationError("snapshot validator must be callable")
        if metadata_validator is not None and not callable(metadata_validator):
            raise ConfigurationError("snapshot metadata validator must be callable")
        with self._lock, self.store.transaction():
            self._open()
            name = self._branch_name(branch_id)
            if branch_id in self._pending or branch_id in self._streams:
                raise PendingError("cannot replace a branch while a clean commit is pending")
            existing = self.store.get_ref(name)
            if new_only and existing is not None:
                raise StateError(f"fork target already exists: {branch_id}")
            record, metadata, policy, state = self._snapshot_record(
                snapshot_id, verify=False, include_rank_contract=True,
            )
            if rank_policy is not None:
                expected_policy = self._rank_policy(rank_policy)
                if policy is None or policy.fingerprint != expected_policy.fingerprint:
                    raise ConfigurationError("snapshot is not bound to the requested rank policy")
            if metadata_validator is not None:
                preview = RestoredState(
                    snapshot_id, branch_id, json_copy(metadata), MappingProxyType({}), policy, state,
                )
                self._run_restore_validator(metadata_validator, preview, metadata_only=True)
            self._audit(snapshot_id)
            tensors = deserialize_tensors(self.store.get(record["tensor_state"]))
            restored = RestoredState(snapshot_id, branch_id, metadata, tensors, policy, state)
            self._run_restore_validator(validator, restored)
            self.store.set_ref(name, record["head"], expected=existing)
            return restored

    def _run_restore_validator(self, validator, restored, *, metadata_only=False):
        if validator is None:
            return
        qualifier = "metadata " if metadata_only else ""
        try:
            validated = validator(restored)
        except ReelCacheError:
            raise
        except Exception as exc:
            raise StateError(f"native snapshot {qualifier}validation failed") from exc
        if validated is not None:
            raise ConfigurationError(
                f"snapshot {qualifier}validator must return None or raise a domain error"
            )

    def restore(
        self, snapshot_id: str, branch_id: str, *,
        validator: Callable[[RestoredState], None] | None = None,
        metadata_validator: Callable[[RestoredState], None] | None = None,
        rank_policy=None,
    ) -> RestoredState:
        """Restore atomically after metadata and full-state validation.

        ``metadata_validator`` receives a detached JSON metadata copy, verified
        rank contracts and an empty read-only tensor mapping, before payload
        reads or deserialization. It must only validate metadata, without
        changing native state or RNG. ``validator`` receives verified tensors
        afterward. Both callbacks must return None; final publication is CAS.
        """
        return self._restore(
            snapshot_id, branch_id, new_only=False, validator=validator,
            metadata_validator=metadata_validator, rank_policy=rank_policy,
        )

    def fork(
        self, snapshot_id: str, branch_id: str, *,
        validator: Callable[[RestoredState], None] | None = None,
        metadata_validator: Callable[[RestoredState], None] | None = None,
        rank_policy=None,
    ) -> RestoredState:
        """Fork with restore's validation stages; never replace an existing branch."""
        return self._restore(
            snapshot_id, branch_id, new_only=True, validator=validator,
            metadata_validator=metadata_validator, rank_policy=rank_policy,
        )

    def _columns(self, columns: Any, channels: int) -> tuple[slice | tuple[int, ...], int]:
        if columns is None:
            return slice(0, channels), channels
        if isinstance(columns, tuple):
            if len(columns) != 2:
                raise ConfigurationError("tuple columns must be a (start, stop) interval; use a list for indices")
            columns = slice(*columns)
        elif isinstance(columns, list):
            if (not columns or any(type(column) is not int or not 0 <= column < channels for column in columns)
                    or len(set(columns)) != len(columns)):
                raise ConfigurationError("channel indices must be nonempty, unique, and inside the canonical domain")
            return tuple(columns), len(columns)
        if not isinstance(columns, slice) or columns.step not in (None, 1):
            raise ConfigurationError("columns must be a contiguous slice or unique channel indices")
        start = 0 if columns.start is None else columns.start
        stop = channels if columns.stop is None else columns.stop
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= channels:
            raise ConfigurationError("channel slice is outside the canonical KV domain")
        return slice(start, stop), stop - start

    def _encoded(self, record: dict[str, Any], read_ahead: ReadAhead | None = None) -> EncodedTensor:
        def read(object_id: str) -> bytes:
            data = None if read_ahead is None else read_ahead.take(object_id)
            return self.store.get(object_id) if data is None else data
        return EncodedTensor(
            record["metadata"], {name: read(oid) for name, oid in record["payloads"].items()},
            basis=None if record["basis"] is None else read(record["basis"]),
            metrics=record["metrics"],
        )

    def _history_read_ahead(self, head_id: str, key: MatrixKey, start: int, stop: int) -> ReadAhead | None:
        """The read-ahead window serving this ``acquire_history`` call (see the class docstring), or None.

        Caller holds the runtime lock, outside its store transaction. Never raises: a failed plan only means no
        read-ahead (each object is still read, and fails, where it is used), and it runs in its own store
        transaction, never inside a caller's (whose failure state it must not change).
        """
        if not self.history_read_ahead_bytes or start >= stop:
            return None
        index, track, now = self._key_index[key], (head_id, start, stop), time.monotonic()
        last = self._read_ahead_tracks.pop(track, None)
        self._read_ahead_tracks[track] = index
        while len(self._read_ahead_tracks) > self.HISTORY_READ_AHEAD_TRACKS:
            self._read_ahead_tracks.popitem(last=False)
        sequential = last is not None and index == last + 1
        if self._read_ahead_window is not None:
            owner, window = self._read_ahead_window
            if owner == track and sequential:
                return window
            if owner == track or now - window.last_used > self.HISTORY_READ_AHEAD_IDLE_SECONDS:
                self._drop_read_ahead()
            else:
                return None  # another range's sequence owns the bound
        if not sequential or index + 1 >= len(self._keys):
            return None
        # The second consecutive key: read the following keys' payloads of this range ahead, in key order.
        try:
            with self.store._lock:
                if self.store._depth or self.store.closed:
                    return None
                with self.store.transaction():
                    _, chunks = self._load_head(head_id)
                    object_ids = []
                    for following in self._keys[index + 1:]:
                        for chunk in chunks:
                            if chunk["token_end"] > start and chunk["token_start"] < stop:
                                record = chunk["records"][following][3]
                                object_ids.extend(record["payloads"].values())
                                if record["basis"] is not None:
                                    object_ids.append(record["basis"])
                    window = self.store.read_ahead(object_ids, max_bytes=self.history_read_ahead_bytes)
        except Exception:
            self._read_ahead_stats["plan_failures"] += 1
            return None
        self._read_ahead_window = (track, window)
        self._read_ahead_stats["windows"] += 1
        return window

    def _drop_read_ahead(self) -> None:
        """Close the history read-ahead window (its unused bytes are dropped). Caller holds the runtime lock."""
        if self._read_ahead_window is not None:
            _, window = self._read_ahead_window
            self._read_ahead_window = None
            window.close()
            stats = window.stats()
            for name in ("taken", "taken_bytes", "failed", "dropped"):
                self._read_ahead_stats[name] += stats[name]

    def read_ahead_stats(self) -> dict[str, Any]:
        """History read-ahead counters: windows opened; objects taken (bytes), failed (read again where used) and
        dropped unused, over closed windows plus the open one (``open``: its counters, or None)."""
        with self._lock:
            result: dict[str, Any] = dict(self._read_ahead_stats)
            current = None if self._read_ahead_window is None else self._read_ahead_window[1].stats()
            if current is not None:
                for name in ("taken", "taken_bytes", "failed"):
                    result[name] += current[name]
            result["open"] = current
            result["max_bytes"] = self.history_read_ahead_bytes
            return result

    def acquire_history(
        self, branch_id: str, key: MatrixKey, *, token_start: int | None = None,
        token_end: int | None = None, device: Any = "cpu", dtype: Any = None,
        columns: Any = None, column_groups: Any = None,
        decoder: Callable[..., torch.Tensor] | None = None, head_id: str | None = None,
    ) -> HistoryLease:
        """Restore a fully covered token range into a non-cached active lease.

        ``head_id`` reads an explicit immutable published head of the branch
        instead of its current head; it is legal while a deferred commit is
        pending on the branch (cross-chunk overlap reads the published prefix).

        ``columns`` is a slice or a list of ordered unique channel indices.
        Legacy two-item tuples denote intervals and are converted to slices
        before provider calls; use lists for explicit indices. Neither expands
        query heads. Do not retain tensors/views beyond the lease.

        Static histories read only this key's tensor manifests. Experimental
        histories additionally audit all policy/decision metadata on replay.
        Restored bytes are hash-verified (``verify_reads``) before use, after
        memory admission -- except payloads the history read-ahead (class
        docstring) read and checked ahead, within its own host-byte bound; no
        persistent metadata or decoded-tensor cache hides other layer history.

        Explicit ``column_groups`` returns [groups, tokens, columns], decoding
        each packet once for equal-width disjoint groups. It is incompatible
        with ``columns`` and requires the built-in provider. All output groups
        remain charged until the lease closes; no cross-call cache is retained.
        """
        if not isinstance(key, MatrixKey) or key not in self._keys:
            raise ConfigurationError("history key is outside the model")
        requested_dtype = None if dtype is None else resolve_dtype(dtype)
        if requested_dtype is not None and not requested_dtype.is_floating_point:
            raise ConfigurationError("KV history must be restored to a floating-point dtype")
        decoder = self.decoder if decoder is None else decoder
        if not callable(decoder):
            raise ConfigurationError("history decoder override must be callable")
        if column_groups is not None:
            from .codecs import HEVCProvider, decode_column_views, decode_tensor
            if columns is not None or not isinstance(column_groups, (tuple, list)) or not column_groups:
                raise ConfigurationError("use nonempty column_groups without columns")
            selected = [self._columns(group, self.model.channels_for(key)) for group in column_groups]
            selected_columns = tuple(group for group, _ in selected)
            channels = selected[0][1]
            indices = [
                column for group in selected_columns
                for column in (range(group.start, group.stop) if isinstance(group, slice) else group)
            ]
            if any(width != channels for _, width in selected) or len(set(indices)) != len(indices):
                raise ConfigurationError("column groups must have equal widths and no repeated channels")
            provider = getattr(decoder, "__self__", None)
            if decoder is decode_tensor:
                decoder = decode_column_views
            elif type(provider) is HEVCProvider and getattr(decoder, "__func__", None) is HEVCProvider.decode_tensor:
                decoder = provider.decode_column_views
            else:
                raise UnsupportedError("grouped history requires the built-in decoder or HEVCProvider")
            group_count = len(selected_columns)
        else:
            selected_columns, channels = self._columns(columns, self.model.channels_for(key))
            group_count = 1
        with self._lock, self.store.transaction():
            if head_id is None:
                head_id = self._ready(branch_id)
            else:
                self._open()
                state = self._async_states.get(branch_id)
                current = self.store.get_ref(self._branch_name(branch_id))
                if head_id != current or (branch_id in self._pending and (state is None or state.parent_head != head_id)):
                    raise StateError("an explicit history head must be the branch's published head")
            _, chunks = self._load_head(head_id, matrix_key=key)
            source_dtype = chunks[0]["records"][key][1].dtype if chunks else self.model.dtype_for(key)
            output_dtype = resolve_dtype(source_dtype) if requested_dtype is None else requested_dtype
            end = chunks[-1]["token_end"] if chunks else 0
            start = 0 if token_start is None else token_start
            stop = end if token_end is None else token_end
            if type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= end:
                raise StateError("requested history is not completely covered by committed chunks")
            selected = [chunk["records"][key] for chunk in chunks
                        if start < stop and chunk["token_end"] > start and chunk["token_start"] < stop]
            destination_shape = ((stop - start, channels) if column_groups is None
                                 else (group_count, stop - start, channels))
            destination_bytes = tensor_nbytes(destination_shape, output_dtype)
            scratch_bytes = 0
            for _, context, spec, record in selected:
                # Providers reconstruct a whole chunk, never the entire history.
                # Reserve source/output conversions and FP32 factor workspaces.
                reconstruction_channels = context.channels if spec.kind == "raw" else channels * group_count
                matrix_bytes = context.token_count * reconstruction_channels * max(
                    4, torch.empty((), dtype=output_dtype).element_size(),
                    torch.empty((), dtype=resolve_dtype(context.dtype)).element_size(),
                )
                workspace = 2 * matrix_bytes
                if spec.kind != "raw":
                    shape = self._basis_geometry(record["metadata"], context, spec)
                    workspace += 4 * (math.prod(shape) + context.token_count * spec.rank)
                scratch_bytes = max(scratch_bytes, workspace)
            pin = self.store.lease([record[0] for record in selected], verify=False) if selected else None
        with self._lock:
            read_ahead = self._history_read_ahead(head_id, key, start, stop)

        def load() -> torch.Tensor:
            try:
                result = torch.empty(destination_shape, device=device, dtype=output_dtype)
            except (RuntimeError, ValueError, TypeError) as exc:
                raise ConfigurationError(f"cannot allocate history on device {device!r}") from exc
            covered = start
            for _, context, _, record in selected:
                left, right = max(start, context.token_start), min(stop, context.token_end)
                if left != covered:
                    raise IntegrityError("history restoration encountered a gap or overlap")
                encoded = self._encoded(record, read_ahead)
                options = ({"columns": None if columns is None else selected_columns}
                           if column_groups is None else {"column_groups": selected_columns})
                decoded = decoder(
                    encoded, device=device, dtype=output_dtype,
                    **options,
                )
                expected = ((context.token_count, channels) if column_groups is None
                            else (group_count, context.token_count, channels))
                if (not isinstance(decoded, torch.Tensor)
                        or tuple(decoded.shape) != expected
                        or decoded.dtype != output_dtype):
                    raise IntegrityError("decoder shape/dtype does not match the committed tensor context")
                with span("history.copy", nbytes=tensor_nbytes((group_count, right - left, channels), output_dtype)):
                    if column_groups is None:
                        result[left - start:right - start].copy_(
                            decoded[left - context.token_start:right - context.token_start]
                        )
                    else:
                        result[:, left - start:right - start].copy_(
                            decoded[:, left - context.token_start:right - context.token_start]
                        )
                covered = right
                del decoded, encoded
            if covered != stop:
                raise IntegrityError("history restoration did not cover its complete requested range")
            return result

        with span("history.restore", nbytes=destination_bytes):
            return self.residency.acquire(
                destination_bytes, load, scratch_bytes=scratch_bytes, pin=pin
            )

    def history_packet(
        self, branch_id: str, key: MatrixKey, *, token_start: int, token_end: int,
    ) -> EncodedTensor:
        """Return the committed packet of exactly one chunk (for off-owner decoding)."""
        if not isinstance(key, MatrixKey) or key not in self._keys:
            raise ConfigurationError("history key is outside the model")
        with self._lock, self.store.transaction():
            head_id = self._ready(branch_id)
            _, chunks = self._load_head(head_id, matrix_key=key)
            matches = [chunk["records"][key] for chunk in chunks
                       if chunk["token_start"] == token_start and chunk["token_end"] == token_end]
            if len(matches) != 1:
                raise StateError("requested range is not exactly one committed chunk")
            with span("history.packet"):
                return self._encoded(matches[0][3])

    def delete_branch(self, branch_id: str) -> None:
        with self._lock:
            with self.store.transaction():
                head = self._ready(branch_id)
                self._check_not_streaming(branch_id)
                self.store.delete_ref(self._branch_name(branch_id), expected=head)
            self._lift_quarantine(self._branch_name(branch_id))

    def delete_snapshot(self, snapshot_id: str) -> None:
        with self._lock:
            with self.store.transaction():
                self._open()
                if snapshot_id not in self._quarantined_objects:  # a quarantined snapshot is deleted unvalidated
                    self._snapshot_record(snapshot_id)
                self.store.delete_ref("snapshots/" + snapshot_id.split(":")[1], expected=snapshot_id)
            self._lift_quarantine("snapshots/" + snapshot_id.split(":")[1])

    def collect_garbage(self) -> dict[str, Any]:
        """Store garbage collection; then the parsed manifests of reclaimed objects leave the runtime's caches
        (result key ``manifests_forgotten``, CR-17-3)."""
        with self._lock:
            self._open()
            self._drop_read_ahead()
            result = self.store.collect_garbage()
            result["manifests_forgotten"] = self.forget_unstored_manifests()
            return result

    gc = collect_garbage

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._pending or self._streams or self._lineage_pending:
                raise PendingError("cannot close a runtime with encoding still in flight")
            self._check_async_errors()
            self._closed = True
            self._drop_read_ahead()
            self.residency.close()
            self.store.close()

    def __enter__(self) -> KVRuntime:
        self._open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
