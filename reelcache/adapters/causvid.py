"""Instance-local, clean-boundary integration with native CausVid.

The native self-attention forward is *not* replaced by a new attention kernel.
Its two absolute cache writes and two prefix reads operate on checked cache
slots instead of a preallocated, all-layer dense history. Only one layer's
history leases exist at a time. A separate t=0 forward precedes publication.
An opt-in byte-budgeted cache owns decoded attention-ready copies for selected
complete layer prefixes within one fixed-head chunk epoch, never original KV.

Supported execution is single-sample, full-prefix, serial inference with an
explicit Generator, fixed conditioning within a chunk, and full-prefix VAE
decode. Training, a partially denoised snapshot, streaming VAE state, cache
eviction, concurrent forwards, and adopting an already-populated native cache
are deliberately rejected. No external model package is imported here.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any
import time
import uuid

import torch

from reelcache.errors import CapacityError, ConfigurationError, IntegrityError, PendingError, StateError, UnsupportedError
from reelcache.reuse import validate_threshold
from reelcache.types import CodecSpec, MatrixKey, TensorContext, json_copy


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _identity(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _version_stamp(tensor: torch.Tensor):
    """Mutation stamp: the version counter of a normal tensor; an inference tensor (``torch.inference_mode``)
    tracks no version counter, so its stamp is its storage location -- it is treated as immutable while borrowed,
    and the content digests of the audit paths still apply."""
    if tensor.is_inference():
        return ("inference", tensor.untyped_storage().data_ptr(), tensor.storage_offset())
    return tensor._version


def _comparable_freqs(freqs: torch.Tensor) -> torch.Tensor:
    """The rotary table to compare later reads against: the tensor itself, or a normal-tensor snapshot of an
    inference tensor (whose in-place updates no version counter would reveal)."""
    if not freqs.is_inference():
        return freqs
    with torch.inference_mode(False):
        return freqs.clone()


def unique_tensor_bytes(tensors) -> int:
    """Charge backing allocations, not logical views or duplicate references."""
    storages = {}
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            storages[key] = storage.nbytes()
    return sum(storages.values())


@dataclass(frozen=True)
class CausVidConfig:
    frames_per_chunk: int = 3
    grid_height: int = 30
    grid_width: int = 52
    heads: int = 12
    head_dim: int = 128
    latent_channels: int = 16
    patch_height: int = 2
    patch_width: int = 2
    vae_mode: str = "full_prefix"

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key != "vae_mode" and (type(value) is not int or value <= 0):
                raise ConfigurationError(f"{key} must be a positive integer")
        if self.head_dim % 2:
            raise ConfigurationError("RoPE requires an even head dimension")
        if self.vae_mode != "full_prefix":
            raise UnsupportedError("CausVid snapshots require full-prefix VAE decode")

    @property
    def frame_tokens(self):
        return self.grid_height * self.grid_width

    @property
    def chunk_tokens(self):
        return self.frames_per_chunk * self.frame_tokens

    @property
    def channels(self):
        return self.heads * self.head_dim

    @property
    def latent_tail(self):
        return (
            self.latent_channels,
            self.grid_height * self.patch_height,
            self.grid_width * self.patch_width,
        )


@dataclass
class _LayerRead:
    layer: int
    freqs: torch.Tensor
    stack: ExitStack
    writes: dict[str, int] = field(default_factory=lambda: {"K": 0, "V": 0})
    reads: dict[str, int] = field(default_factory=lambda: {"K": 0, "V": 0})
    tensors: list = field(default_factory=list)
    native_reads: dict = field(default_factory=dict)
    paired_leases: dict = field(default_factory=dict)


@dataclass
class _HotHistory:
    tensor: torch.Tensor
    version: Any  # _version_stamp
    freqs: torch.Tensor
    freqs_version: Any
    event: Any = None


@dataclass
class _QualityHistory:
    tensor: torch.Tensor
    version: Any  # _version_stamp
    valid_tokens: int = 0
    freqs: torch.Tensor | None = None
    freqs_version: Any = 0


class CausVidCleanStaging(Mapping):
    """Owned, bounded clean matrices, optionally spilled to an explicit directory.

    Disk-backed reads materialize only one matrix. The host budget includes a
    conservative serialization/read workspace, not only retained tensor bytes.
    Closing removes only files created by this instance.
    """

    def __init__(self, *, max_host_bytes, max_disk_bytes=0, directory=None):
        for name, value, minimum in (
            ("max_host_bytes", max_host_bytes, 1),
            ("max_disk_bytes", max_disk_bytes, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ConfigurationError(f"{name} must be an integer >= {minimum}")
        if directory is not None and not max_disk_bytes:
            raise CapacityError("Disk clean staging requires an explicit disk budget")
        self.max_host_bytes, self.max_disk_bytes = max_host_bytes, max_disk_bytes
        self.directory = None if directory is None else Path(directory)
        self._records, self._tensors = {}, {}
        self.host_bytes = self.disk_bytes = self.peak_host_bytes = self.peak_disk_bytes = 0
        self.closed = False
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=False)

    def __iter__(self):
        return iter(self._records)

    def __len__(self):
        return len(self._records)

    def put(self, key, tensor):
        if self.closed or key in self._records:
            raise StateError("Clean staging is closed or this matrix was already captured")
        if not isinstance(key, MatrixKey) or not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ConfigurationError("Clean staging requires a matrix key and a two-dimensional tensor")
        size = tensor.numel() * tensor.element_size()
        workspace = 4 * size + 65536
        if self.host_bytes + workspace > self.max_host_bytes:
            raise CapacityError("Clean capture exceeds its explicit host byte budget")
        self.peak_host_bytes = max(self.peak_host_bytes, self.host_bytes + workspace)
        with torch.inference_mode(False):  # owned copy: a normal tensor whatever the caller's mode
            value = tensor.detach().to("cpu", copy=True).contiguous()
        record = {
            "shape": tuple(value.shape), "dtype": value.dtype,
            "sha256": tensor_digest(value), "bytes": size,
        }
        if self.directory is None:
            self._tensors[key] = value
            record["version"] = _version_stamp(value)
            self.host_bytes += size
        else:
            from reelcache.serialization import serialize_tensors

            raw = serialize_tensors({"source": value})
            if self.disk_bytes + len(raw) > self.max_disk_bytes:
                raise CapacityError("Clean capture exceeds its explicit disk byte budget")
            path = self.directory / f"{key.layer:04d}-{key.target}.reelcache"
            try:
                with path.open("xb") as stream:
                    stream.write(raw)
                    stream.flush()
                record.update(path=path, file_sha256=hashlib.sha256(raw).hexdigest(), file_bytes=len(raw))
            except BaseException:
                path.unlink(missing_ok=True)
                raise
            self.disk_bytes += len(raw)
            self.peak_disk_bytes = max(self.peak_disk_bytes, self.disk_bytes)
        self._records[key] = record

    def __getitem__(self, key):
        if self.closed:
            raise StateError("Clean staging is closed")
        record = self._records[key]
        if self.directory is None:
            value = self._tensors[key]
            if _version_stamp(value) != record["version"]:
                raise IntegrityError("An owned clean source was mutated before publication")
        else:
            from reelcache.serialization import deserialize_tensors

            path = record["path"]
            if path.stat().st_size != record["file_bytes"]:
                raise IntegrityError("Staged clean source size changed")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != record["file_sha256"]:
                raise IntegrityError("Staged clean source failed its file hash")
            fields = deserialize_tensors(raw)
            if set(fields) != {"source"}:
                raise IntegrityError("Staged clean source fields changed")
            value = fields["source"]
        if (
            tuple(value.shape) != record["shape"] or value.dtype != record["dtype"]
            or tensor_digest(value) != record["sha256"]
        ):
            raise IntegrityError("Staged clean source failed its tensor identity")
        return value

    def close(self):
        if self.closed:
            return
        self._tensors.clear()
        for record in self._records.values():
            if "path" in record:
                record["path"].unlink()
        self._records.clear()
        if self.directory is not None:
            self.directory.rmdir()
        self.host_bytes = self.disk_bytes = 0
        self.closed = True


class _CacheSlot:
    """The deliberately small indexing protocol used by native cache inference."""

    def __init__(self, adapter, layer, target):
        self.adapter, self.layer, self.target = adapter, layer, target

    def _slice(self, index):
        if (
            not isinstance(index, tuple) or len(index) != 2
            or index[0] != slice(None) or not isinstance(index[1], slice)
            or index[1].step is not None
        ):
            raise UnsupportedError("Unsupported native CausVid cache indexing")
        frame = self.adapter._active_layer
        if frame is None or frame.layer != self.layer:
            raise StateError("Native cache access outside its layer-scoped forward")
        return frame, index[1]

    def __setitem__(self, index, value):
        frame, span = self._slice(index)
        owner = self.adapter
        position = owner._position_tokens()
        if span.start != position or span.stop != position + owner.config.chunk_tokens:
            raise UnsupportedError("Native write must address exactly the mutable current chunk")
        if frame.writes[self.target]:
            raise UnsupportedError("Multiple native writes to one cache slot in one forward")
        expected = (1, owner.config.chunk_tokens, owner.config.heads, owner.config.head_dim)
        if tuple(value.shape) != expected or value.device != owner.device or value.dtype != owner.dtype:
            raise ConfigurationError("Native cache write has incompatible geometry, device or source dtype")
        key = MatrixKey(self.layer, self.target)
        buffer = owner._mutable.get(key)
        if buffer is None:
            required = value.numel() * owner._element_bytes
            if unique_tensor_bytes(owner._mutable.values()) + required > owner.max_current_bytes:
                raise CapacityError("Native current KV exceeds its explicit byte budget")
            buffer = torch.empty(expected, device=value.device, dtype=owner.dtype)
            owner._mutable[key] = buffer
        buffer.copy_(value)
        if owner._phase == "clean_forward":
            if self.target == "V":
                owner._observe_clean(key, value[0].reshape(owner.config.chunk_tokens, owner.config.channels))
            if self.target == "V" or key not in owner._pre_rope_keys:
                owner._clean_staging.put(
                    key, buffer[0].reshape(owner.config.chunk_tokens, owner.config.channels),
                )
        frame.writes[self.target] += 1
        owner._peak_mutable_bytes = max(owner._peak_mutable_bytes, unique_tensor_bytes(owner._mutable.values()))

    def __getitem__(self, index):
        frame, span = self._slice(index)
        owner = self.adapter
        if span.start not in (None, 0) or span.stop != owner._position_tokens() + owner.config.chunk_tokens:
            raise UnsupportedError("Native history reads must preserve the complete visible prefix")
        if frame.writes != {"K": 1, "V": 1} or frame.reads[self.target]:
            raise StateError("Native history must be read once, after both current writes")
        frame.reads[self.target] += 1
        key = MatrixKey(self.layer, self.target)
        current = owner._mutable[key]
        if not owner._position_tokens():
            owner._audit_native_read(frame, key, current)
            return current
        history = owner._history_for(frame, key)
        visible = torch.cat((history, current), dim=1)
        frame.tensors.append(visible)
        owner._peak_read_bytes = max(owner._peak_read_bytes, unique_tensor_bytes(frame.tensors))
        owner._audit_native_read(frame, key, visible)
        return visible


class CausVidAdapter:
    """Own one native pipeline's inference state without owning model loading.

    ``start`` creates a fresh runtime branch. ``step`` atomically advances one
    chunk (or primes a supplied clean prefix at position zero). ``snapshot`` is
    legal only after successful sealing. ``restore``/``fork`` switch this
    adapter to a new branch; the old branch and its snapshots stay immutable.

    Static specs remain fixed, including on restore. Alternatively a typed
    experimental ``rank_policy`` owns resolved specs and committed scalar state;
    it cannot be mixed with a static profile. Static RAW uses native post-RoPE
    keys; factor histories always use direct normalized pre-RoPE keys, including
    after a fork. ``quality_hot`` retains only separately admitted reconstructions
    of committed packets and is not a compressed-HBM benchmark.

    ``rank_source_provenance(context)`` optionally returns source-bound JSON
    metadata for the adaptive commit. It is synchronous and separate from policy
    features; the worker resolves it from the actual pristine clean observation.

    ``native_history_observer(record, tensor)`` is a development-only, synchronous
    read-only borrow of actual attention operands / published native history.
    It must not retain tensors. Hashing streams one native chunk at a time through
    the existing clean-host workspace; only scalar committed hashes survive.
    This opt-in diagnostic requires a fresh run (no restore), includes device
    readback and extra reload decoding, and is not a performance measurement.
    """

    SCHEMA = "reelcache.causvid.clean.v1"

    def __init__(
        self, pipeline, runtime, branch_id: str, *, config: CausVidConfig | None = None,
        specs: CodecSpec | Mapping[MatrixKey, CodecSpec] | None = None,
        rope=None, max_attention_bytes: int = 512 * 1024**2, reuse_threshold=None,
        max_hot_history_bytes: int = 0,
        clean_observer=None, source_provenance=None,
        max_current_bytes: int = 1024**3, max_clean_host_bytes: int = 1024**3,
        max_clean_disk_bytes: int = 0, clean_staging_root=None,
        history_mode: str = "reload", max_quality_history_bytes: int = 0,
        rank_policy=None, rank_source_provenance=None,
        native_history_observer=None, commit_executor=None, prefetch_history_bytes=0, parallel_kv_bytes=0,
    ):
        if (
            not isinstance(branch_id, str) or not branch_id or "/" in branch_id
            or "\x00" in branch_id or branch_id in (".", "..")
        ):
            raise ConfigurationError("CausVid branch_id must be a nonempty single-component name")
        self.pipeline, self.runtime, self.branch_id = pipeline, runtime, branch_id
        if commit_executor is not None:
            from reelcache.execution import BoundedExecutor
            if not isinstance(commit_executor, BoundedExecutor) or commit_executor is runtime.executor:
                raise ConfigurationError("commit_executor must be a distinct BoundedExecutor")
            if rank_policy is not None or history_mode != "reload" or native_history_observer is not None:
                raise UnsupportedError("deferred publication requires static reload mode without native history auditing")
        self.commit_executor = commit_executor
        self._pending_step = None
        # Cross-chunk overlap: native KV of a deferred, not yet published chunk serves the next chunk.
        self._overlay = None
        if type(prefetch_history_bytes) is not int or prefetch_history_bytes < 0:
            raise ConfigurationError("prefetch_history_bytes must be a nonnegative integer")
        if prefetch_history_bytes and (max_hot_history_bytes or history_mode != "reload" or native_history_observer):
            raise UnsupportedError("layer prefetch requires reload mode without hot history or diagnostic observers")
        self.prefetch_history_bytes = prefetch_history_bytes
        self._prefetch = None
        self._prefetch_active = False
        if type(parallel_kv_bytes) is not int or parallel_kv_bytes < 0:
            raise ConfigurationError("parallel_kv_bytes must be a nonnegative integer")
        if parallel_kv_bytes and (max_hot_history_bytes or history_mode != "reload"
                                  or native_history_observer):
            raise UnsupportedError("parallel K/V requires reload mode without hot cache or diagnostics")
        self.parallel_kv_bytes = parallel_kv_bytes
        self._paired_decoder = None
        self.config = config or CausVidConfig()
        self.model = pipeline.generator.model
        self.identity = runtime.model
        self.device = next(self.model.parameters()).device
        self.dtype = getattr(torch, self.identity.dtype)
        self._element_bytes = torch.empty((), dtype=self.dtype).element_size()
        if type(max_attention_bytes) is not int or max_attention_bytes <= 0:
            raise ConfigurationError("max_attention_bytes must be a positive integer")
        self.max_attention_bytes = max_attention_bytes
        for name, value, minimum in (
            ("max_current_bytes", max_current_bytes, 1),
            ("max_clean_host_bytes", max_clean_host_bytes, 1),
            ("max_clean_disk_bytes", max_clean_disk_bytes, 0),
            ("max_quality_history_bytes", max_quality_history_bytes, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ConfigurationError(f"{name} must be an integer >= {minimum}")
            setattr(self, name, value)
        if clean_observer is not None and not callable(clean_observer):
            raise ConfigurationError("clean_observer must be a synchronous callable")
        if native_history_observer is not None and not callable(native_history_observer):
            raise ConfigurationError("native_history_observer must be a synchronous callable")
        if clean_staging_root is not None and not max_clean_disk_bytes:
            raise CapacityError("An explicit disk budget is required for clean staging")
        self.clean_observer = clean_observer
        self.native_history_observer = native_history_observer
        self._native_committed_hashes = {}
        self._native_event_ordinal = self._native_forward_ordinal = 0
        self._native_hash_bytes = self._native_hash_peak_host_bytes = 0
        self._native_hash_seconds = self._native_observer_seconds = self._native_extra_decode_seconds = 0.0
        self._native_extra_decodes = 0
        self.source_provenance = json_copy({} if source_provenance is None else source_provenance)
        self.clean_staging_root = None if clean_staging_root is None else Path(clean_staging_root)
        self._clean_staging = None
        self._clean_observations = []
        self._peak_clean_host_bytes = self._peak_clean_disk_bytes = 0
        if history_mode not in ("reload", "quality_hot"):
            raise ConfigurationError("history_mode must be reload or quality_hot")
        if history_mode == "quality_hot" and (not max_quality_history_bytes or max_hot_history_bytes):
            raise ConfigurationError("quality_hot needs its own full-history budget, not a partial hot cache")
        if history_mode == "reload" and max_quality_history_bytes:
            raise ConfigurationError("Full decoded-history admission is only valid in quality_hot mode")
        self.history_mode = history_mode
        self._quality_history, self._quality_freqs = {}, {}
        self._quality_epoch = None
        self._peak_quality_bytes = self._quality_reads = self._quality_replacements = 0
        if type(max_hot_history_bytes) is not int or max_hot_history_bytes < 0:
            raise ConfigurationError("max_hot_history_bytes must be a nonnegative integer")
        self.max_hot_history_bytes = max_hot_history_bytes
        self._hot_history = {}
        self._hot_epoch = None
        self._hot_layer_count = 0
        self._peak_hot_bytes = self._hot_hits = self._hot_misses = 0
        self.reuse_threshold = None if reuse_threshold is None else validate_threshold(reuse_threshold)
        self.keys = tuple(MatrixKey(layer, target) for layer in range(self.identity.layers) for target in ("K", "V"))
        self.rank_policy = None
        if rank_source_provenance is not None and (
            rank_policy is None or not callable(rank_source_provenance)
        ):
            raise ConfigurationError("rank_source_provenance requires an adaptive policy and synchronous callable")
        self.rank_source_provenance = rank_source_provenance
        if rank_policy is not None:
            from reelcache.online_rank import RankPolicy

            if not isinstance(rank_policy, RankPolicy):
                raise ConfigurationError("rank_policy must be the typed experimental RankPolicy")
            if specs is not None or reuse_threshold is not None:
                raise ConfigurationError("An adaptive policy owns resolved specs and reuse; do not override them")
            self.rank_policy = RankPolicy.from_dict(rank_policy.to_dict())
            self.specs = {}
        else:
            specs = CodecSpec() if specs is None else specs
            self.specs = {key: specs for key in self.keys} if isinstance(specs, CodecSpec) else dict(specs)
            if set(self.specs) != set(self.keys) or any(not isinstance(spec, CodecSpec) for spec in self.specs.values()):
                raise ConfigurationError("CausVid requires one codec spec for every layer's K and V")
            if len({spec.kind == "raw" for key, spec in self.specs.items() if key.target == "K"}) != 1:
                raise UnsupportedError("All CausVid layers must use the same K factorization domain")
        self.specs = MappingProxyType(self.specs)
        self._pre_rope_keys = frozenset(
            key for key in self.keys
            if key.target == "K" and (self.rank_policy is not None or self.specs[key].kind != "raw")
        )
        self._rank_fingerprint = None if self.rank_policy is None else self.rank_policy.fingerprint
        self.rope = rope
        self._patches, self._handles = [], []
        self._mutable, self._pre_keys = {}, {}
        self._verified_pre_keys = set()
        self._active_layer = None
        self._installed = self._started = False
        self._phase = "uninitialized"
        self._noise = self._latents = self._rng = None
        self._conditioning = {}
        self._conditioning_id = None
        self.frame_cursor = 0
        self.audit = []
        self._seen_layers = set()
        self._peak_mutable_bytes = self._peak_read_bytes = 0
        self._validate_rank_policy()
        self._validate_native()

    def _validate_rank_policy(self, *, ordinal=None):
        if self._rank_fingerprint is None:
            if self.rank_policy is not None:
                raise ConfigurationError("A static CausVid adapter cannot acquire a rank policy mid-history")
            return
        from reelcache.online_rank import RankPolicy

        if not isinstance(self.rank_policy, RankPolicy) or self.rank_policy.fingerprint != self._rank_fingerprint:
            raise ConfigurationError("The bound CausVid rank policy cannot change during execution")
        self.rank_policy.validate_contexts(
            {key: self._context(key) for key in self.keys}, history_chunks=ordinal,
        )

    def _chunk_id(self, ordinal):
        return ordinal if self.rank_policy is None else f"chunk-{ordinal:06d}"

    def _begin_hot_epoch(self):
        if self.history_mode == "quality_hot":
            self._validate_quality_epoch()
            return
        if self._hot_history or self._hot_epoch is not None:
            raise StateError("A decoded-history epoch survived its clean chunk")
        if not self.max_hot_history_bytes or not self.token_cursor:
            return
        info = self.runtime.branch_info(self.branch_id)
        if info["token_end"] != self.token_cursor:
            raise StateError("Committed branch position differs from native execution")
        layer_bytes = 2 * self.token_cursor * self.config.channels * self._element_bytes
        self._hot_layer_count = min(self.identity.layers, self.max_hot_history_bytes // layer_bytes)
        self._hot_epoch = (self.branch_id, info["head_id"], self.token_cursor)

    def _validate_hot_epoch(self):
        if self.history_mode == "quality_hot":
            self._validate_quality_epoch()
            return
        if self._hot_epoch is not None:
            branch, head, cursor = self._hot_epoch
            if (
                (self.branch_id, self.token_cursor) != (branch, cursor)
                or self.runtime.branch_info(branch)["head_id"] != head
            ):
                raise StateError("Immutable decoded-history epoch changed during native execution")
            if any(_version_stamp(entry.tensor) != entry.version for entry in self._hot_history.values()):
                raise IntegrityError("An attention-ready decoded-history tensor was mutated")

    def _history_for(self, frame, key):
        if self.history_mode == "quality_hot":
            return self._quality_read(frame, key)
        entry = self._hot_history.get(key)
        if entry is not None:
            if _version_stamp(entry.tensor) != entry.version:
                raise IntegrityError("An attention-ready decoded-history tensor was mutated")
            if key.target == "K" and (
                _version_stamp(entry.freqs) != entry.freqs_version
                or (frame.freqs is not entry.freqs and not torch.equal(frame.freqs, entry.freqs))
            ):
                raise IntegrityError("Native rotary frequencies changed inside the decoded-history epoch")
            if entry.event is not None:
                torch.cuda.current_stream(self.device).wait_event(entry.event)
            self._hot_hits += 1
            frame.tensors.append(entry.tensor)
            return entry.tensor
        self._hot_misses += 1
        pre_rope = key in self._pre_rope_keys
        if self._prefetch_active:
            history = self._prefetch.tensor(frame.layer, key)
        elif self.parallel_kv_bytes:
            if self._paired_decoder is None:
                from reelcache.paired_decode import PairedDecoder
                self._paired_decoder = PairedDecoder(
                    getattr(self.runtime.decoder, "__self__", None), device=self.device,
                    max_output_bytes=self.parallel_kv_bytes,
                )
            if not frame.paired_leases:
                frame.paired_leases = frame.stack.enter_context(self._paired_decoder.acquire_history(
                    self.runtime, self.branch_id, frame.layer, token_end=self.token_cursor,
                    native_dtype=self.dtype, pre_rope_keys=self._pre_rope_keys,
                ))
            history = frame.paired_leases[key].tensor
        elif self._overlay is not None and not self.token_cursor:
            history = None
        else:
            lease = frame.stack.enter_context(self.runtime.acquire_history(
                self.branch_id, key, token_start=0, token_end=self.token_cursor,
                device=self.device, dtype=torch.float32 if pre_rope else self.dtype,
                **({} if self._overlay is None else {"head_id": self._overlay["published_head"]}),
            ))
            history = lease.tensor
        if history is not None:
            if tuple(history.shape) != (self.token_cursor, self.config.channels):
                raise IntegrityError("Restored history does not cover the native visible prefix")
            frame.tensors.append(history)
            history = history.reshape(1, self.token_cursor, self.config.heads, self.config.head_dim)
            if pre_rope:
                grid = torch.tensor([[self.frame_cursor, self.config.grid_height, self.config.grid_width]])
                history = self.rope(history, grid, frame.freqs, start_frame=0).to(self.dtype)
                frame.tensors.append(history)
        if self._overlay is not None:
            # Published archive history, then the deferred chunk's own native (post-RoPE) KV.
            native = self._overlay["native"][key]
            history = native if history is None else torch.cat((history, native), dim=1)
            frame.tensors.append(history)
        if self._hot_epoch is not None and key.layer < self._hot_layer_count:
            required = history.numel() * history.element_size()
            occupied = unique_tensor_bytes(item.tensor for item in self._hot_history.values())
            if occupied + required > self.max_hot_history_bytes:
                raise CapacityError("Decoded-history cache exceeds its explicit byte budget")
            # Own a separate allocation: retaining a released lease or a view of
            # one would violate the shared residency contract.
            with torch.inference_mode(False):  # owned: a normal tensor whatever the caller's mode
                history = history.detach().clone(memory_format=torch.contiguous_format)
            frame.tensors.append(history)
            freqs = _comparable_freqs(frame.freqs)
            self._hot_history[key] = _HotHistory(history, _version_stamp(history), freqs, _version_stamp(freqs))
            actual = unique_tensor_bytes(item.tensor for item in self._hot_history.values())
            if actual > self.max_hot_history_bytes:
                raise CapacityError("Decoded-history backing allocation exceeded admission")
            self._peak_hot_bytes = max(self._peak_hot_bytes, actual)
        return history

    def _mark_hot_used(self, layer):
        if self.device.type == "cuda":
            for key, entry in self._hot_history.items():
                if key.layer == layer:
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(self.device))
                    entry.event = event

    def _release_hot_tensors(self):
        try:
            for entry in self._hot_history.values():
                if entry.event is not None:
                    entry.event.synchronize()
        finally:
            self._hot_history.clear()

    def _clear_hot_history(self):
        try:
            self._release_hot_tensors()
        finally:
            self._hot_epoch = None
            self._hot_layer_count = 0

    def _new_quality_storage(self, total_frames):
        if self.history_mode != "quality_hot":
            return {}
        tokens = total_frames * self.config.frame_tokens
        required = len(self.keys) * tokens * self.config.channels * self._element_bytes
        if required > self.max_quality_history_bytes:
            raise CapacityError("Complete decoded native history exceeds its explicit quality-only budget")
        occupied = unique_tensor_bytes(entry.tensor for entry in self._quality_history.values())
        if self._quality_history and all(
            entry.tensor.shape[1] >= tokens for entry in self._quality_history.values()
        ):
            return {
                key: _QualityHistory(entry.tensor, _version_stamp(entry.tensor))
                for key, entry in self._quality_history.items()
            }
        if occupied + required > self.max_quality_history_bytes:
            raise CapacityError("Growing a restored quality epoch would exceed its live allocation budget")
        result = {}
        for key in self.keys:
            with torch.inference_mode(False):  # owned: a normal tensor whatever the caller's mode
                value = torch.empty(
                    (1, tokens, self.config.heads, self.config.head_dim), device=self.device, dtype=self.dtype,
                )
            result[key] = _QualityHistory(value, _version_stamp(value))
        self._peak_quality_bytes = max(self._peak_quality_bytes, occupied + required)
        return result

    def _validate_quality_epoch(self):
        if self._quality_epoch is None or set(self._quality_history) != set(self.keys):
            raise StateError("No complete decoded-history quality epoch is attached")
        info = self.runtime.branch_info(self.branch_id)
        if self._quality_epoch != (self.branch_id, info["head_id"], self.token_cursor):
            raise StateError("Committed history differs from the decoded-history quality epoch")
        if any(_version_stamp(entry.tensor) != entry.version for entry in self._quality_history.values()):
            raise IntegrityError("An owned decoded quality-history tensor was mutated")

    @contextmanager
    def _committed_native(self, key, start, stop, freqs):
        first, last = start * self.config.frame_tokens, stop * self.config.frame_tokens
        pre_rope = key in self._pre_rope_keys
        with self.runtime.acquire_history(
            self.branch_id, key, token_start=first, token_end=last, device=self.device,
            dtype=torch.float32 if pre_rope else self.dtype,
        ) as lease:
            value = lease.tensor
            if tuple(value.shape) != (last - first, self.config.channels):
                raise IntegrityError("Committed quality reconstruction does not cover its exact clean range")
            value = value.reshape(1, last - first, self.config.heads, self.config.head_dim)
            if pre_rope:
                grid = torch.tensor([[stop - start, self.config.grid_height, self.config.grid_width]])
                value = self.rope(value, grid, freqs, start_frame=start).to(self.dtype)
            if not bool(torch.isfinite(value).all()):
                raise IntegrityError("Committed quality reconstruction contains nonfinite values")
            yield value

    def _copy_committed_quality(self, key, start, stop, freqs, *, replace_current):
        entry = self._quality_history[key]
        first, last = start * self.config.frame_tokens, stop * self.config.frame_tokens
        if entry.valid_tokens != first:
            raise StateError("Decoded quality history has a gap or overlap")
        with self._committed_native(key, start, stop, freqs) as value:
            measured = self._measure_native(value, first) if self.native_history_observer is not None else None
            if replace_current:
                self._mutable[key].copy_(value)
                self._quality_replacements += 1
            entry.tensor[:, first:last].copy_(value)
        entry.version, entry.valid_tokens = _version_stamp(entry.tensor), last
        entry.freqs = _comparable_freqs(freqs)
        entry.freqs_version = _version_stamp(entry.freqs)
        return measured

    def _quality_read(self, frame, key):
        entry = self._quality_history[key]
        if entry.valid_tokens == 0 and self.token_cursor:
            self._copy_committed_quality(key, 0, self.frame_cursor, frame.freqs, replace_current=False)
        if entry.valid_tokens != self.token_cursor or _version_stamp(entry.tensor) != entry.version:
            raise IntegrityError("Decoded quality history does not match the committed visible prefix")
        if key.target == "K" and (
            entry.freqs is None or _version_stamp(entry.freqs) != entry.freqs_version
            or (entry.freqs is not frame.freqs and not torch.equal(entry.freqs, frame.freqs))
        ):
            raise IntegrityError("Native rotary frequencies changed in the quality-history epoch")
        self._quality_reads += 1
        history = entry.tensor[:, :self.token_cursor]
        frame.tensors.append(history)
        return history

    def _append_quality_history(self, start, stop):
        if self.history_mode != "quality_hot":
            if self.native_history_observer is not None:
                for key in self.keys:
                    tick = time.perf_counter()
                    with self._committed_native(key, 0, stop, self._quality_freqs[key.layer]) as value:
                        self._native_extra_decode_seconds += time.perf_counter() - tick
                        self._native_extra_decodes += 1
                        self._audit_native_commit(key, start, stop, value)
                    del value
            return
        info = self.runtime.branch_info(self.branch_id)
        if info["token_end"] != stop * self.config.frame_tokens:
            raise StateError("Only committed packets may replace a quality-history current chunk")
        from reelcache.telemetry import span
        with span("quality.append"):
            for key in self.keys:
                measured = self._copy_committed_quality(
                    key, start, stop, self._quality_freqs[key.layer], replace_current=True,
                )
                if self.native_history_observer is not None:
                    self._audit_native_commit(
                        key, start, stop, self._quality_history[key].tensor[:, :info["token_end"]],
                        decoded=measured,
                    )
        self._quality_epoch = (self.branch_id, info["head_id"], info["token_end"])
        self._peak_quality_bytes = max(
            self._peak_quality_bytes, unique_tensor_bytes(entry.tensor for entry in self._quality_history.values()),
        )

    def _measure_native(self, tensor, first=0, *, archived_end=0):
        """Hash actual native bytes, never a canonical/pre-RoPE surrogate."""
        if (
            tensor.ndim != 4 or tensor.shape[0] != 1
            or tuple(tensor.shape[2:]) != (self.config.heads, self.config.head_dim)
            or tensor.dtype != self.dtype or tensor.device != self.device
            or first % self.config.chunk_tokens or tensor.shape[1] % self.config.chunk_tokens
        ):
            raise IntegrityError("Native-history audit received an incompatible attention-ready tensor")
        tick = time.perf_counter()
        digest, archived, segments = hashlib.sha256(), hashlib.sha256(), []
        try:
            for offset in range(0, tensor.shape[1], self.config.chunk_tokens):
                end = offset + self.config.chunk_tokens
                size = self.config.chunk_tokens * self.config.channels * self._element_bytes
                workspace = 2 * size + 65536
                staged = 0 if self._clean_staging is None else self._clean_staging.host_bytes
                if staged + workspace > self.max_clean_host_bytes:
                    raise CapacityError("Native-history readback exceeds the clean-host workspace admission")
                self._native_hash_peak_host_bytes = max(self._native_hash_peak_host_bytes, workspace)
                value = tensor[:, offset:end].detach().to("cpu", copy=True).contiguous()
                raw = memoryview(value.reshape(-1).view(torch.uint8).numpy())
                digest.update(raw)
                if first + end <= archived_end:
                    archived.update(raw)
                segments.append({
                    "chunk_id": (first + offset) // self.config.chunk_tokens,
                    "token_start": first + offset, "token_end": first + end,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                })
                self._native_hash_bytes += size
                del raw, value
            return {
                "sha256": digest.hexdigest(), "archived_prefix_sha256": archived.hexdigest(),
                "segments": segments,
            }
        finally:
            self._native_hash_seconds += time.perf_counter() - tick

    def _check_native_archive(self, key, measured, archived_end):
        for segment in measured["segments"]:
            if segment["token_end"] <= archived_end:
                committed = self._native_committed_hashes.get((key, segment["chunk_id"]))
                if committed is None or segment["sha256"] != committed["sha256"]:
                    raise IntegrityError("An archived native attention operand differs from its committed packet")
                segment["publication_head_id"] = committed["head_id"]

    def _native_record(self, key, event, first, last):
        info = self.runtime.branch_info(self.branch_id)
        if info["token_end"] != (self.token_cursor if event == "attention_read" else last):
            raise IntegrityError("Native-history audit lost its exact committed head")
        return {
            "schema": "reelcache.causvid.native_history_event.v1", "event": event,
            "event_ordinal": self._native_event_ordinal,
            "forward_ordinal": self._native_forward_ordinal, "phase": self._phase,
            "layer": key.layer, "target": key.target,
            "chunk_id": self.frame_cursor // self.config.frames_per_chunk,
            "token_start": first, "token_end": last,
            "frame_start": first // self.config.frame_tokens, "frame_end": last // self.config.frame_tokens,
            "current_token_start": self.token_cursor, "history_mode": self.history_mode,
            "branch_id": self.branch_id, "committed_head_id": info["head_id"],
            "committed_token_end": info["token_end"],
            "native_domain": "post_rope" if key.target == "K" else "direct_v",
            "packet_domain": ("pre_rope" if key in self._pre_rope_keys else "post_rope")
            if key.target == "K" else "direct_v",
            "dtype": str(self.dtype).removeprefix("torch."),
            "shape": [1, last - first, self.config.heads, self.config.head_dim],
        }

    def _emit_native(self, record, tensor, digest):
        version, tick = _version_stamp(tensor), time.perf_counter()
        borrowed = json_copy(record)
        try:
            result = self.native_history_observer(borrowed, tensor.detach())
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise UnsupportedError("Native-history observers must finish synchronously")
        finally:
            self._native_observer_seconds += time.perf_counter() - tick
        if (
            _version_stamp(tensor) != version or self._measure_native(tensor, record["token_start"])["sha256"] != digest
            or borrowed != record
            or self.runtime.branch_info(self.branch_id)["head_id"] != record["committed_head_id"]
        ):
            raise IntegrityError("A native-history observer mutated its borrowed tensor, record or committed head")
        self._native_event_ordinal += 1

    def _audit_native_read(self, frame, key, tensor):
        if self.native_history_observer is None:
            return
        measured = self._measure_native(tensor, archived_end=self.token_cursor)
        self._check_native_archive(key, measured, self.token_cursor)
        frame.native_reads[key] = (
            tensor, _version_stamp(tensor), measured, self.runtime.branch_info(self.branch_id)["head_id"],
        )

    def _finish_native_reads(self, frame):
        for key, (tensor, version, measured, head) in frame.native_reads.items():
            after = self._measure_native(tensor)["sha256"]
            record = self._native_record(key, "attention_read", 0, self.token_cursor + self.config.chunk_tokens)
            if _version_stamp(tensor) != version or after != measured["sha256"] or record["committed_head_id"] != head:
                raise IntegrityError("Native attention mutated an audited operand or its committed head")
            record.update(
                native_kv_sha256=measured["sha256"], after_attention_sha256=after,
                archived_prefix_sha256=measured["archived_prefix_sha256"],
                archived_token_end=self.token_cursor, segments=measured["segments"],
            )
            self._emit_native(record, tensor, measured["sha256"])

    def _audit_native_commit(self, key, start, stop, tensor, *, decoded=None):
        first, last = start * self.config.frame_tokens, stop * self.config.frame_tokens
        if self.history_mode == "quality_hot" and (
            not isinstance(decoded, dict) or len(decoded.get("segments", [])) != 1
            or decoded["segments"][0]["token_start"] != first or decoded["segments"][0]["token_end"] != last
        ):
            raise IntegrityError("Quality publication is missing its freshly decoded native packet measurement")
        measured = self._measure_native(tensor, archived_end=first)
        self._check_native_archive(key, measured, first)
        newest = measured["segments"][-1]["sha256"]
        current = None if decoded is None else self._measure_native(self._mutable[key], first)["sha256"]
        expected = newest if decoded is None else decoded["sha256"]
        if newest != expected or (decoded is not None and current != expected):
            raise IntegrityError("Newest native current/retained slot differs from its freshly decoded committed packet")
        record = self._native_record(key, "clean_publication", 0, last)
        record.update(
            native_kv_sha256=measured["sha256"], archived_prefix_sha256=measured["archived_prefix_sha256"],
            archived_token_end=first, segments=measured["segments"],
            newest_token_start=first, newest_token_end=last,
            decoded_current_sha256=expected, retained_current_sha256=None if decoded is None else newest,
            mutable_current_sha256=current,
            source="fresh_committed_native_reload" if decoded is None else "quality_retained_native_history",
        )
        self._emit_native(record, tensor, measured["sha256"])
        if decoded is not None and self._measure_native(self._mutable[key], first)["sha256"] != expected:
            raise IntegrityError("A native-history observer mutated the newest mutable slot")
        self._native_committed_hashes[(key, first // self.config.chunk_tokens)] = {
            "sha256": expected, "head_id": record["committed_head_id"],
        }

    @property
    def token_cursor(self):
        return self.frame_cursor * self.config.frame_tokens

    def _position_tokens(self):
        """Native absolute position of the chunk being generated (the generated frontier)."""
        return self.token_cursor if self._overlay is None else self._overlay["position_tokens"]

    @property
    def latent_state(self):
        if not self._started:
            raise StateError("No inference state is attached")
        return self._latents.clone()

    @property
    def rng_state(self):
        if self._rng is None:
            raise StateError("No inference RNG is attached")
        return self._rng.get_state().clone()

    @property
    def finished(self):
        return self._started and self.frame_cursor == self._noise.shape[1]

    def _validate_native(self):
        cfg, model, pipe = self.config, self.model, self.pipeline
        if model.training:
            raise UnsupportedError("Put the native CausVid model in evaluation mode before attaching")
        if self.identity.channels_for("K") != cfg.channels or self.identity.channels_for("V") != cfg.channels:
            raise ConfigurationError("Runtime full KV channels do not match native heads")
        if (
            len(model.blocks) != self.identity.layers
            or pipe.num_transformer_blocks != self.identity.layers
            or model.num_heads != cfg.heads or model.dim != cfg.channels
            or model.num_frame_per_block != cfg.frames_per_chunk
            or pipe.num_frame_per_block != cfg.frames_per_chunk
            or pipe.frame_seq_length != cfg.frame_tokens
        ):
            raise ConfigurationError("CausVid model, runtime identity and chunk geometry disagree")
        if getattr(model, "model_type", "t2v") != "t2v":
            raise UnsupportedError("Only native CausVid text-to-video inference is supported")
        if tuple(getattr(model, "patch_size", (1, cfg.patch_height, cfg.patch_width))) != (
            1, cfg.patch_height, cfg.patch_width
        ):
            raise ConfigurationError("Native patch geometry differs from the adapter")
        names = ("x", "seq_lens", "grid_sizes", "freqs", "block_mask", "kv_cache", "current_start", "current_end")
        for layer, block in enumerate(model.blocks):
            attention = block.self_attn
            if tuple(inspect.signature(attention.forward).parameters) != names:
                raise UnsupportedError("Unrecognized native CausVid self-attention signature")
            if (
                attention.num_heads != cfg.heads or attention.head_dim != cfg.head_dim
                or tuple(attention.window_size) != (-1, -1)
            ):
                raise UnsupportedError("CausVid adapter requires full-prefix native attention without GQA")
            if (
                (MatrixKey(layer, "K") in self._pre_rope_keys or self.clean_observer is not None)
                and not getattr(attention, "qk_norm", True)
            ):
                raise UnsupportedError("Lossy CausVid K requires native normalized pre-RoPE keys")
            if self.rope is None:
                method = attention.forward
                self.rope = method.__func__.__globals__.get("causal_rope_apply")
        if not callable(self.rope):
            raise ConfigurationError("Supply the native causal_rope_apply callable")
        steps = pipe.denoising_step_list.detach().to("cpu")
        if steps.ndim != 1 or not steps.numel() or not bool(torch.isfinite(steps).all()) or bool((steps <= 0).any()):
            raise ConfigurationError("Denoising steps must be positive; the clean forward is separate")

    def _binding(self):
        scheduler = self.pipeline.scheduler
        schedule = {
            name: tensor_digest(getattr(scheduler, name))
            for name in ("timesteps", "sigmas")
            if isinstance(getattr(scheduler, name, None), torch.Tensor)
        }
        return {
            "model": self.identity.to_dict(), "geometry": asdict(self.config),
            "specs": [
                {"key": key.to_dict(), "spec": self.specs[key].to_dict()}
                for key in sorted(self.specs)
            ],
            "steps": self.pipeline.denoising_step_list.detach().cpu().tolist(),
            "scheduler": f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
            "schedule_hashes": schedule,
            "seq_len": getattr(self.pipeline.generator, "seq_len", None),
            "uniform_timestep": getattr(self.pipeline.generator, "uniform_timestep", None),
            "reuse_threshold": self.reuse_threshold,
            "vae_mode": "full_prefix",
            **({"rank_policy": self.rank_policy.to_dict()} if self.rank_policy is not None else {}),
        }

    def install(self):
        if self._installed:
            raise StateError("CausVid adapter is already installed")
        if getattr(self.pipeline, "_reelcache_causvid_owner", None) is not None:
            raise StateError("The native pipeline already has a ReelCache owner")
        if getattr(self.pipeline, "kv_cache1", None) is not None:
            raise UnsupportedError("Attach to an empty native cache, or restore a ReelCache snapshot; dense adoption is unsupported")
        try:
            for layer, block in enumerate(self.model.blocks):
                attention = block.self_attn
                had = "forward" in attention.__dict__
                previous = attention.__dict__.get("forward")
                original = attention.forward
                signature = inspect.signature(original)
                self._patches.append((attention, had, previous))
                attention.forward = self._wrap_layer(layer, original, signature)
                if MatrixKey(layer, "K") in self._pre_rope_keys or self.clean_observer is not None:
                    self._handles.append(attention.norm_k.register_forward_hook(self._capture_key(layer)))
            self.pipeline.kv_cache1 = [
                {name: _CacheSlot(self, layer, target) for name, target in (("k", "K"), ("v", "V"))}
                for layer in range(self.identity.layers)
            ]
            self.pipeline._reelcache_causvid_owner = self
            self._installed = True
            return self
        except BaseException:
            self.close()
            raise

    def _wrap_layer(self, layer, original, signature):
        def forward(*args, **kwargs):
            if self._phase not in ("denoising", "clean_forward") or self._active_layer is not None:
                raise StateError("Concurrent, training or unmanaged CausVid forward is unsupported")
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            fields = bound.arguments
            if fields["kv_cache"] is not self.pipeline.kv_cache1[layer]:
                raise StateError("Native forward bypassed the owned cache")
            position = self._position_tokens()
            if fields["current_start"] != position or fields["current_end"] != position + self.config.chunk_tokens:
                raise StateError("Native absolute position does not match the committed execution cursor")
            if (
                fields["grid_sizes"].tolist() != [[self.config.frames_per_chunk, self.config.grid_height, self.config.grid_width]]
                or fields["seq_lens"].tolist() != [self.config.chunk_tokens]
                or tuple(fields["x"].shape) != (1, self.config.chunk_tokens, self.config.channels)
            ):
                raise UnsupportedError("Padding, ragged batches or different native token geometry are unsupported")
            if layer in self._seen_layers:
                raise StateError("Native layer executed more than once in a generator forward")
            # Includes leases, native FP64 rotary workspace and concatenated K/V.
            current_factor = (
                32 if MatrixKey(layer, "K") in self._pre_rope_keys or self.clean_observer is not None
                else 2 * self._element_bytes
            )
            history_factor = 32 + (2 * self._element_bytes if layer < self._hot_layer_count else 0)
            conservative = (
                self._position_tokens() * self.config.channels * history_factor
                + self.config.chunk_tokens * self.config.channels * current_factor
            )
            if conservative > self.max_attention_bytes:
                raise CapacityError("Complete native attention-visible history exceeds max_attention_bytes")
            if self._prefetch_active:
                self._prefetch.acquire(layer)
            with ExitStack() as stack:
                frame = _LayerRead(layer, fields["freqs"], stack)
                self._active_layer = frame
                try:
                    output = original(*args, **kwargs)
                    if frame.writes != {"K": 1, "V": 1} or frame.reads != {"K": 1, "V": 1}:
                        raise UnsupportedError("Native cache seam changed its read/write behavior")
                    self._finish_native_reads(frame)
                    if self._phase == "clean_forward" and (
                        MatrixKey(layer, "K") in self._pre_rope_keys or self.clean_observer is not None
                    ):
                        source = self._pre_keys.pop(layer).reshape(
                            1, self.config.chunk_tokens, self.config.heads, self.config.head_dim,
                        )
                        restored = self.rope(
                            source, fields["grid_sizes"], frame.freqs,
                            start_frame=self._position_tokens() // self.config.frame_tokens,
                        ).to(self.dtype)
                        if not torch.equal(restored, self._mutable[MatrixKey(layer, "K")]):
                            raise IntegrityError("Direct normalized pre-RoPE K does not reproduce the native clean cache")
                        self._verified_pre_keys.add(layer)
                    if self._phase == "clean_forward" and (
                        self.history_mode == "quality_hot" or self.native_history_observer is not None
                    ):
                        self._quality_freqs[layer] = frame.freqs
                    self._seen_layers.add(layer)
                    return output
                finally:
                    if self._prefetch_active:
                        self._prefetch.release(layer)
                    self._mark_hot_used(layer)
                    frame.native_reads.clear()
                    frame.tensors.clear()
                    frame.paired_leases.clear()
                    self._active_layer = None
        return forward

    def _capture_key(self, layer):
        def capture(_module, _inputs, output):
            if self._phase != "clean_forward":
                return
            if self._active_layer is None or self._active_layer.layer != layer or layer in self._pre_keys:
                raise StateError("Clean normalized-key capture was not one-to-one with the native layer")
            if (
                tuple(output.shape) != (1, self.config.chunk_tokens, self.config.channels)
                or output.dtype != self.dtype
            ):
                raise ConfigurationError("Native normalized pre-RoPE K has unexpected shape or source dtype")
            key = MatrixKey(layer, "K")
            self._pre_keys[layer] = output[0].detach()
            self._observe_clean(key, output[0])
            if key in self._pre_rope_keys:
                self._clean_staging.put(key, output[0])
        return capture

    def _context(self, key, *, canonical=False, source_dtype=None):
        return TensorContext(
            self.identity, key, self._position_tokens(), self.config.frames_per_chunk,
            self.config.grid_height, self.config.grid_width,
            ("pre_rope" if canonical or key in self._pre_rope_keys else "post_rope")
            if key.target == "K" else "none",
            source_dtype=source_dtype,
        )

    def _observe_clean(self, key, tensor):
        if self.clean_observer is None:
            return
        if (
            self._phase != "clean_forward" or tuple(tensor.shape) != (
                self.config.chunk_tokens, self.config.channels,
            )
            or tensor.dtype != self.dtype or not bool(torch.isfinite(tensor).all())
        ):
            raise IntegrityError("The clean observer requires finite pristine native-dtype source matrices")
        if any(row["context"]["key"] == key.to_dict() for row in self._clean_observations):
            raise StateError("A clean observer saw the same source more than once")
        context = self._context(key, canonical=True, source_dtype=str(tensor.dtype).removeprefix("torch."))
        digest, version = tensor_digest(tensor), _version_stamp(tensor)
        result = self.clean_observer(context, tensor.detach())
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise UnsupportedError("Clean observers must finish synchronously before native execution resumes")
        if _version_stamp(tensor) != version or tensor_digest(tensor) != digest:
            raise IntegrityError("A clean observer mutated the pristine native source")
        self._clean_observations.append({
            "context": context.to_dict(), "sha256": digest,
            "shape": list(tensor.shape), "dtype": context.dtype,
            "branch_id": self.branch_id,
            "chunk_id": self._position_tokens() // self.config.chunk_tokens,
            "conditioning_identity": self._conditioning_id,
            "history_head_id": self._observation_head if self.token_cursor else None,
            "runtime_head_id": self._observation_head,
            "source": "native_clean_t0_norm_k_before_rope" if key.target == "K" else "native_clean_t0_direct_v",
            "available_keys": [
                *(row["context"]["key"] for row in self._clean_observations), key.to_dict(),
            ],
            "provenance": json_copy(self.source_provenance),
        })

    def _reset_cross_attention(self):
        self.pipeline.crossattn_cache = [
            {"k": None, "v": None, "is_init": False} for _ in range(self.identity.layers)
        ]

    def _check_ready(self, *, allow_pending=False):
        if self._pending_step is not None and not allow_pending:
            raise PendingError("finish_step() must resolve publication before using the next clean boundary")
        idle = ("ready", "clean", "publication_pending") if allow_pending else ("ready", "clean")
        if not self._installed or not self._started or self._phase not in idle:
            raise StateError("Operation requires an attached, idle CausVid execution")
        if self.model.training:
            raise UnsupportedError("Training-mode native execution cannot use a resumable inference branch")
        self._validate_rank_policy()

    def _validate_noise(self, noise):
        if (
            not isinstance(noise, torch.Tensor) or noise.ndim != 5 or noise.shape[0] != 1
            or noise.shape[1] <= 0 or noise.shape[1] % self.config.frames_per_chunk
            or tuple(noise.shape[2:]) != self.config.latent_tail or noise.dtype != self.dtype
            or not bool(torch.isfinite(noise).all())
        ):
            raise ConfigurationError("Expected finite [1, chunks*frames, C, H, W] native-dtype noise")
        freqs = getattr(self.model, "freqs", None)
        if isinstance(freqs, torch.Tensor) and max(noise.shape[1], self.config.grid_height, self.config.grid_width) > freqs.shape[0]:
            raise UnsupportedError("Execution horizon or spatial grid exceeds the native absolute RoPE table")

    def _admit_clean_storage(self):
        matrix_bytes = self.config.chunk_tokens * self.config.channels * self._element_bytes
        chunk_bytes = len(self.keys) * matrix_bytes
        if self.max_current_bytes < chunk_bytes:
            raise CapacityError("The entire native current chunk must be admitted before starting")
        if self.clean_staging_root is None:
            if self.max_clean_host_bytes < chunk_bytes + 3 * matrix_bytes + 65536:
                raise CapacityError("The whole clean chunk and hashing workspace exceed host admission")
        elif (
            self.max_clean_host_bytes < 4 * matrix_bytes + 65536
            or self.max_clean_disk_bytes < chunk_bytes + len(self.keys) * 65536
        ):
            raise CapacityError("The whole clean chunk and bounded disk I/O workspace were not admitted")

    def _prepare_conditioning(self, conditioning, conditioning_id):
        if not isinstance(conditioning_id, str) or not conditioning_id:
            raise ConfigurationError("An explicit conditioning identity is required")
        if not isinstance(conditioning, Mapping) or set(conditioning) != {"prompt_embeds"}:
            raise UnsupportedError("CausVid conditioning must contain only prompt_embeds")
        value = conditioning["prompt_embeds"]
        if (
            not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1
            or not value.shape[1] or not value.shape[2] or value.dtype != self.dtype
            or not bool(torch.isfinite(value).all())
        ):
            raise ConfigurationError("Invalid native prompt embeddings")
        return {"prompt_embeds": value.detach().to(device=self.device, copy=True)}

    def start(self, noise, conditioning, rng, *, conditioning_id: str):
        if not self._installed or self._started:
            raise StateError("Install a fresh adapter before starting an execution")
        self._validate_rank_policy(ordinal=1)
        self._validate_noise(noise)
        if self.native_history_observer is not None and (
            noise.shape[1] > 3 * self.config.frames_per_chunk or len(self.pipeline.denoising_step_list) > 4
        ):
            raise CapacityError("Native-history diagnostics admit at most three chunks and four noisy passes")
        self._admit_clean_storage()
        if not isinstance(rng, torch.Generator) or torch.device(rng.device).type != self.device.type:
            raise ConfigurationError("An explicit device-matched native RNG is required")
        prepared = self._prepare_conditioning(conditioning, conditioning_id)
        quality = self._new_quality_storage(noise.shape[1])
        cloned_rng = torch.Generator(device=self.device)
        cloned_rng.set_state(rng.get_state())
        noise_cpu = noise.detach().to("cpu", copy=True).contiguous()
        self.runtime.create_branch(self.branch_id)
        self._noise = noise_cpu
        self._latents = torch.empty((1, 0, *self.config.latent_tail), dtype=self.dtype)
        self._rng, self._conditioning, self._conditioning_id = cloned_rng, prepared, conditioning_id
        self.frame_cursor = 0
        self._quality_history = quality
        self._quality_freqs.clear()
        if self.history_mode == "quality_hot":
            self._quality_epoch = (self.branch_id, self.runtime.branch_info(self.branch_id)["head_id"], 0)
            self._peak_quality_bytes = unique_tensor_bytes(entry.tensor for entry in quality.values())
        self._reset_cross_attention()
        self._started, self._phase = True, "ready"
        return self

    def set_conditioning(self, conditioning, *, conditioning_id: str):
        self._check_ready()
        prepared = self._prepare_conditioning(conditioning, conditioning_id)
        self._conditioning, self._conditioning_id = prepared, conditioning_id
        self._reset_cross_attention()

    def _generator_forward(self, latents, timestep, *, clean):
        self._validate_hot_epoch()
        self._phase = "clean_forward" if clean else "denoising"
        if self.native_history_observer is not None:
            self._native_forward_ordinal += 1
        self._seen_layers.clear()
        if self.prefetch_history_bytes and self.token_cursor:
            if self._prefetch is None:
                from reelcache.prefetch import LayerPrefetcher
                self._prefetch = LayerPrefetcher(
                    self.runtime, device=self.device, layers=self.identity.layers, channels=self.config.channels,
                    native_dtype=self.dtype, pre_rope_keys=self._pre_rope_keys, max_bytes=self.prefetch_history_bytes,
                    parallel_kv_bytes=self.parallel_kv_bytes,
                )
            self._prefetch.begin(self.branch_id, self.token_cursor)
            self._prefetch_active = True
        try:
            output = self.pipeline.generator(
                noisy_image_or_video=latents, conditional_dict=self._conditioning, timestep=timestep,
                kv_cache=self.pipeline.kv_cache1, crossattn_cache=self.pipeline.crossattn_cache,
                current_start=self._position_tokens(),
                current_end=self._position_tokens() + self.config.chunk_tokens,
            )
        finally:
            if self._prefetch_active:
                try:
                    self._prefetch.end()
                finally:
                    self._prefetch_active = False
        if self._seen_layers != set(range(self.identity.layers)):
            raise StateError("Native generator did not execute every cache layer")
        self._validate_hot_epoch()
        if not clean and (
            not isinstance(output, torch.Tensor) or output.shape != latents.shape
            or output.dtype != self.dtype or output.device != self.device
            or not bool(torch.isfinite(output).all())
        ):
            raise IntegrityError("Native denoising produced invalid clean latents")
        return output

    @torch.no_grad()
    def step(self, *, clean_prefix=None, defer_publication=False):
        """Denoise, separately clean-forward, seal, then release dense current KV.

        ``clean_prefix`` is accepted only for the first chunk. It is a real
        native clean prefill, not an invented continuation or a cache adoption.
        Failed computation/encoding rolls back the RNG and mutable chunk.

        With a deferred step still publishing, the next step overlaps it
        (cross-chunk overlap, reload mode without hot/prefetch/paired caches):
        its forwards read the published archive history plus the deferred
        chunk's retained native KV, and the deferred publication is resolved
        (``finish_step``) only before this chunk is submitted or sealed. A
        failure of the overlapped predecessor poisons the execution (restore a
        snapshot); no chunk is ever published on top of an unpublished parent.
        """
        overlapped = self._pending_step
        self._check_ready(allow_pending=True)
        if overlapped is not None and (
            self.max_hot_history_bytes or self.prefetch_history_bytes or self.parallel_kv_bytes
            or self.history_mode != "reload" or clean_prefix is not None
        ):
            raise PendingError("finish_step() first: overlapped steps need plain reload mode without "
                               "hot, prefetched or paired history")
        if type(defer_publication) is not bool:
            raise ConfigurationError("defer_publication must be boolean")
        if defer_publication and self.commit_executor is None:
            raise ConfigurationError("deferred publication requires an explicit commit_executor")
        base = self.frame_cursor if overlapped is None else overlapped["end"]
        if base == self._noise.shape[1]:
            raise StateError("All configured chunks have already completed")
        if clean_prefix is not None and self.frame_cursor:
            raise UnsupportedError("A supplied clean prefix is only accepted at absolute position zero")
        start, end = base, base + self.config.frames_per_chunk
        self._validate_rank_policy(ordinal=start // self.config.frames_per_chunk + 1)
        if overlapped is None:
            info = self.runtime.branch_info(self.branch_id)
            if info["token_end"] != self.token_cursor:
                raise StateError("The committed branch cursor differs from native execution")
        if self.rank_policy is not None and self.token_cursor:
            committed_policy = self.runtime.history_rank_policy(self.branch_id)
            if committed_policy is None or committed_policy.fingerprint != self._rank_fingerprint:
                raise ConfigurationError("The committed history belongs to a different rank policy")
        prior_rng, prior_phase = self.rng_state, self._phase
        self._verified_pre_keys.clear()
        self._clean_observations = []
        staged = matrices = contexts = None
        committed = poisoned = False
        if overlapped is not None:
            retained = sum(value.numel() * self._element_bytes for value in overlapped["native"].values())
            if 2 * retained > self.max_current_bytes:
                raise CapacityError("an overlapped step needs the deferred chunk's KV plus the current chunk "
                                    "within max_current_bytes")
            self._overlay = {"native": overlapped["native"], "position_tokens": base * self.config.frame_tokens,
                             "published_head": overlapped["parent"]}
        try:
            self._observation_head = (self.runtime.branch_info(self.branch_id)["head_id"] if overlapped is None
                                      else overlapped["parent"])
            directory = (
                None if self.clean_staging_root is None else
                self.clean_staging_root / f"chunk-{start // self.config.frames_per_chunk:06d}-{uuid.uuid4().hex}"
            )
            self._clean_staging = CausVidCleanStaging(
                max_host_bytes=self.max_clean_host_bytes,
                max_disk_bytes=self.max_clean_disk_bytes, directory=directory,
            )
            self._begin_hot_epoch()
            if clean_prefix is None:
                noisy = self._noise[:, start:end].to(self.device)
                steps = self.pipeline.denoising_step_list
                for index, value in enumerate(steps):
                    timestep = torch.ones((1, self.config.frames_per_chunk), device=self.device, dtype=torch.int64) * value
                    staged = self._generator_forward(noisy, timestep, clean=False)
                    if index + 1 < len(steps):
                        noisy = self.pipeline.scheduler.add_noise(
                            staged.flatten(0, 1),
                            torch.empty_like(staged.flatten(0, 1)).normal_(generator=self._rng),
                            steps[index + 1] * torch.ones(
                                (self.config.frames_per_chunk,), device=self.device, dtype=torch.long,
                            ),
                        ).unflatten(0, staged.shape[:2])
            else:
                if (
                    tuple(clean_prefix.shape) != (1, self.config.frames_per_chunk, *self.config.latent_tail)
                    or clean_prefix.dtype != self.dtype or not bool(torch.isfinite(clean_prefix).all())
                ):
                    raise ConfigurationError("The clean prefix must have exactly one native chunk")
                staged = clean_prefix.detach().to(self.device)
            zeros = torch.zeros((1, self.config.frames_per_chunk), device=self.device, dtype=torch.int64)
            self._generator_forward(staged, zeros, clean=True)
            self._release_hot_tensors()
            if overlapped is not None:
                # The predecessor must be durable before this chunk is submitted on top of it.
                self._overlay = None
                try:
                    self.finish_step()
                except BaseException:
                    poisoned = True
                    raise
                self._observation_head = self.runtime.branch_info(self.branch_id)["head_id"]
                for record in self._clean_observations:
                    record["history_head_id"] = record["runtime_head_id"] = self._observation_head
            self._phase = "clean_unsealed"
            matrices, contexts = self._clean_staging, {}
            if set(matrices) != set(self.keys):
                raise IntegrityError("The separate clean forward did not stage every native K/V")
            for key in self.keys:
                if not bool(torch.isfinite(self._mutable[key]).all()):
                    raise IntegrityError("Native clean K/V contains nonfinite values; refusing publication")
                contexts[key] = self._context(
                    key, source_dtype=self.identity.dtype if self.rank_policy is not None else None,
                )
            # Prepare all possibly failing local allocations before publication.
            next_latents = torch.cat((self._latents, staged.detach().to("cpu")), dim=1)
            self._validate_hot_epoch()
            self._validate_rank_policy(ordinal=start // self.config.frames_per_chunk + 1)
            chunk_id = self._chunk_id(start // self.config.frames_per_chunk)
            audit = {
                "chunk": start // self.config.frames_per_chunk,
                "token_start": start * self.config.frame_tokens, "token_end": end * self.config.frame_tokens,
                "separate_clean_forward": True, "supplied_clean_prefix": clean_prefix is not None,
                "history_visible_from_token": 0, "all_layers_written_and_read": True,
                "denoising_steps": 0 if clean_prefix is not None else len(self.pipeline.denoising_step_list),
                "pre_rope_source_verified_layers": sorted(self._verified_pre_keys),
                "history_mode": self.history_mode,
                "committed_current_reconstructed": self.history_mode == "quality_hot",
                "rank_policy_identity": self._rank_fingerprint,
                "rank_anchor_availability": None if self.rank_policy is None else self.rank_policy.anchor_mode,
                "clean_sources": list(self._clean_observations),
                "clean_staging": {
                    "mode": "owned_host" if self.clean_staging_root is None else "owned_disk_lazy_mapping",
                    "peak_host_bytes": self._clean_staging.peak_host_bytes,
                    "peak_disk_bytes": self._clean_staging.peak_disk_bytes,
                    "whole_clean_chunk_buffered": True,
                },
                "deferred_publication": defer_publication,
            }
            if defer_publication:
                pending_state = {
                    "next_latents": next_latents, "end": end, "prior_rng": prior_rng,
                    "prior_phase": "clean" if overlapped is not None else prior_phase,
                    "parent": self._observation_head, "audit": audit,
                    "native": {key: self._mutable[key] for key in self.keys},
                }
                pending_state["handle"] = self.runtime.submit_chunk(
                    self.branch_id, chunk_id, matrices, contexts, self.specs,
                    commit_executor=self.commit_executor, allow_lossy=True, reuse_threshold=self.reuse_threshold,
                )
                self._pending_step = pending_state
                self._mutable = {}
                self._phase = "publication_pending"
                return pending_state["handle"]
            if self.rank_policy is None:
                result = self.runtime.seal_chunk(
                    self.branch_id, chunk_id, matrices, contexts, self.specs,
                    clean=True, reuse_threshold=self.reuse_threshold,
                )
            else:
                source_provenance = None
                if self.rank_source_provenance is not None:
                    source_provenance = {}
                    for key in self.keys:
                        binding = self.rank_source_provenance(contexts[key])
                        if inspect.isawaitable(binding):
                            if inspect.iscoroutine(binding):
                                binding.close()
                            raise UnsupportedError("Rank source provenance must finish synchronously before fitting")
                        if not isinstance(binding, Mapping):
                            raise ConfigurationError("Rank source provenance must return a finite JSON object")
                        source_provenance[key] = json_copy(binding)
                result = self.runtime.seal_chunk(
                    self.branch_id, chunk_id, matrices, contexts, clean=True, rank_policy=self.rank_policy,
                    source_provenance=source_provenance,
                )
            committed = True
            self._append_quality_history(start, end)
        except BaseException:
            try:
                head_unchanged = (self._pending_step is not None
                                  or self.runtime.branch_info(self.branch_id)["head_id"] == self._observation_head)
            except BaseException:
                head_unchanged = False
            if committed or poisoned or not head_unchanged:
                self._phase = "poisoned"
            elif self._pending_step is not None:
                # The overlapped predecessor is still pending: only this chunk's work is rolled back.
                self._rng.set_state(prior_rng)
                self._phase = "publication_pending"
            else:
                self._rng.set_state(prior_rng)
                self._phase = prior_phase
            self._reset_cross_attention()
            raise
        else:
            self._latents, self.frame_cursor, self._phase = next_latents, end, "clean"
            self.audit.append(audit)
            return result
        finally:
            self._overlay = None
            self._mutable.clear()
            self._pre_keys.clear()
            self._verified_pre_keys.clear()
            if self._clean_staging is not None:
                self._peak_clean_host_bytes = max(self._peak_clean_host_bytes, self._clean_staging.peak_host_bytes)
                self._peak_clean_disk_bytes = max(self._peak_clean_disk_bytes, self._clean_staging.peak_disk_bytes)
                self._clean_staging.close()
                self._clean_staging = None
            self._active_layer = None
            self._clear_hot_history()

    def finish_step(self, timeout=None):
        """Resolve a deferred commit before advancing the native clean cursor.

        A wait timeout retains pending execution. Failure/cancellation rolls
        back RNG only if the durable head is unchanged; otherwise restore is
        required. No denoising forward uses unpublished or alternate history.
        """
        pending = self._pending_step
        if pending is None:
            raise StateError("there is no deferred CausVid step to finish")
        handle = pending["handle"]
        try:
            result = handle.result(timeout=timeout)
        except BaseException:
            if not handle.done():
                raise
            try:
                unchanged = self.runtime.branch_info(self.branch_id)["head_id"] == pending["parent"]
            except BaseException:
                unchanged = False
            if unchanged:
                self._rng.set_state(pending["prior_rng"])
                self._phase = pending["prior_phase"]
            else:
                self._phase = "poisoned"
            self._pending_step = None
            self._reset_cross_attention()
            raise
        self._latents, self.frame_cursor, self._phase = pending["next_latents"], pending["end"], "clean"
        self.audit.append(pending["audit"])
        self._pending_step = None
        return result

    def _metadata(self):
        self._check_ready()
        if self._phase != "clean" or not self.frame_cursor or self._mutable or self._pre_keys:
            raise StateError("Only a separately clean-forwarded, sealed boundary can be snapshotted")
        return {
            "schema": self.SCHEMA, "binding": self._binding(),
            "clean_boundary": {
                "clean": True, "token_end": self.token_cursor,
                "chunk_id": self._chunk_id(self.frame_cursor // self.config.frames_per_chunk - 1),
            },
            "conditioning": {
                "identity": self._conditioning_id,
                "sha256": tensor_digest(self._conditioning["prompt_embeds"]),
            },
            "execution": {
                "frame_cursor": self.frame_cursor, "token_cursor": self.token_cursor,
                "total_frames": self._noise.shape[1], "pending": False,
            },
            "rng": {"device_type": self.device.type, "contract": "explicit_generator_with_saved_upfront_noise"},
            "latent": {"layout": "B,F,C,H,W", "dtype": self.identity.dtype, "shape": list(self._latents.shape)},
            "vae": {"contract": "reset_and_decode_entire_saved_latent_prefix"},
        }

    def snapshot(self):
        metadata = self._metadata()
        return self.runtime.snapshot(self.branch_id, metadata=metadata, tensors={
            "noise": self._noise, "latents": self._latents, "rng_state": self.rng_state,
            "prompt_embeds": self._conditioning["prompt_embeds"].detach().to("cpu"),
        })

    def _validate_restore_metadata(self, saved, *, conditioning_id):
        if self.rank_policy is None:
            if saved.rank_policy is not None or saved.rank_state is not None:
                raise ConfigurationError("A static CausVid adapter cannot restore committed adaptive history")
        else:
            from reelcache.online_rank import RankPolicy, RankState

            if (
                not isinstance(saved.rank_policy, RankPolicy)
                or saved.rank_policy.fingerprint != self._rank_fingerprint
            ):
                raise ConfigurationError("The verified snapshot rank policy differs from this CausVid adapter")
            if not isinstance(saved.rank_state, RankState):
                raise IntegrityError("The adaptive snapshot is missing verified committed rank state")
        metadata = saved.metadata
        if (
            metadata.get("schema") != self.SCHEMA
            or metadata.get("clean_boundary", {}).get("clean") is not True
            or metadata.get("binding") != self._binding()
            or metadata.get("rng") != {
                "device_type": self.device.type, "contract": "explicit_generator_with_saved_upfront_noise",
            }
            or metadata.get("vae") != {"contract": "reset_and_decode_entire_saved_latent_prefix"}
        ):
            raise ConfigurationError("Snapshot model, codec, native schedule, RNG device or VAE contract differs")
        if conditioning_id is not None and conditioning_id != metadata["conditioning"].get("identity"):
            raise ConfigurationError("Snapshot conditioning identity differs from the requested conditioning")

    def _prepare_restore(self, saved, *, seed, conditioning_id):
        self._validate_restore_metadata(saved, conditioning_id=conditioning_id)
        metadata, tensors = saved.metadata, saved.tensors
        condition, execution = metadata["conditioning"], metadata["execution"]
        if set(tensors) != {"noise", "latents", "rng_state", "prompt_embeds"}:
            raise IntegrityError("Incomplete CausVid snapshot tensor state")
        noise, latents = tensors["noise"], tensors["latents"]
        self._validate_noise(noise)
        self._admit_clean_storage()
        cursor = execution.get("frame_cursor")
        if (
            type(cursor) is not int or not 0 < cursor <= noise.shape[1]
            or cursor % self.config.frames_per_chunk or execution.get("token_cursor") != cursor * self.config.frame_tokens
            or execution.get("total_frames") != noise.shape[1]
            or metadata["clean_boundary"]["token_end"] != cursor * self.config.frame_tokens
            or metadata["clean_boundary"]["chunk_id"] != self._chunk_id(cursor // self.config.frames_per_chunk - 1)
            or tuple(latents.shape) != (1, cursor, *self.config.latent_tail) or latents.dtype != self.dtype
            or not bool(torch.isfinite(latents).all())
            or tensor_digest(tensors["prompt_embeds"]) != condition.get("sha256")
            or metadata["latent"] != {
                "layout": "B,F,C,H,W", "dtype": self.identity.dtype, "shape": list(latents.shape),
            }
        ):
            raise IntegrityError("CausVid snapshot cursor, latents or conditioning failed verification")
        if self.rank_policy is not None:
            saved.rank_state.validate(
                self.rank_policy, token_end=cursor * self.config.frame_tokens,
                completed_chunks=cursor // self.config.frames_per_chunk,
                last_chunk_id=self._chunk_id(cursor // self.config.frames_per_chunk - 1),
            )
            self.rank_policy.validate_contexts(
                {
                    key: replace(
                        self._context(key),
                        token_start=cursor * self.config.frame_tokens - self.config.chunk_tokens,
                    )
                    for key in self.keys
                },
                history_chunks=cursor // self.config.frames_per_chunk,
            )
        prepared = self._prepare_conditioning(
            {"prompt_embeds": tensors["prompt_embeds"]}, condition["identity"],
        )
        rng = torch.Generator(device=self.device)
        rng.set_state(tensors["rng_state"])
        noise, latents = noise.clone(), latents.clone()
        if seed is not None:
            if type(seed) is not int or seed < 0:
                raise ConfigurationError("Fork seed must be a nonnegative integer")
            rng.manual_seed(seed)
            if cursor < noise.shape[1]:
                future = torch.empty(
                    (1, noise.shape[1] - cursor, *self.config.latent_tail), device=self.device, dtype=self.dtype,
                ).normal_(generator=rng)
                noise[:, cursor:] = future.cpu()
        quality = self._new_quality_storage(noise.shape[1])
        return {
            "cursor": cursor, "noise": noise, "latents": latents, "rng": rng, "quality": quality,
            "conditioning": prepared, "conditioning_id": condition["identity"],
        }

    def restore(self, snapshot_id, branch_id, *, seed=None, conditioning_id=None):
        """Validate the actual committed policy and native state before publication.

        Metadata-only checks reject incompatible conditioning/configuration and
        committed policy before snapshot tensor payload reads or deserialization.
        Loaded tensors still pass the full validator before new-only publication.
        With ``seed=None``, RNG and remaining noise are exact snapshot state.
        With a seed, only the child's remaining upfront noise is redrawn, then
        its scheduler noise uses that new generator. The parent is untouched.
        """
        if self.native_history_observer is not None:
            raise UnsupportedError("Native-history diagnostics require a fresh run, not a restored baseline")
        if not self._installed or self._phase not in ("uninitialized", "ready", "clean"):
            raise StateError("Restore requires an installed idle adapter")
        self._validate_rank_policy()
        prepared = {}

        def validate_metadata(saved):
            self._validate_restore_metadata(saved, conditioning_id=conditioning_id)

        def validate(saved):
            prepared.update(self._prepare_restore(saved, seed=seed, conditioning_id=conditioning_id))

        # New-only publication prevents a restore/seed operation from replacing
        # an existing parent, even if another writer races this adapter.
        self.runtime.fork(
            snapshot_id, branch_id, rank_policy=self.rank_policy,
            metadata_validator=validate_metadata, validator=validate,
        )
        self.branch_id, self.frame_cursor = branch_id, prepared["cursor"]
        self._noise, self._latents, self._rng = prepared["noise"], prepared["latents"], prepared["rng"]
        self._quality_history = prepared["quality"]
        self._quality_freqs.clear()
        if self.history_mode == "quality_hot":
            self._quality_epoch = (
                self.branch_id, self.runtime.branch_info(self.branch_id)["head_id"], self.token_cursor,
            )
            self._peak_quality_bytes = max(
                self._peak_quality_bytes,
                unique_tensor_bytes(entry.tensor for entry in self._quality_history.values()),
            )
        self._conditioning, self._conditioning_id = prepared["conditioning"], prepared["conditioning_id"]
        self._mutable.clear()
        self._pre_keys.clear()
        self._clear_hot_history()
        self._reset_cross_attention()
        self._started, self._phase = True, "clean"
        self.audit = []
        return self

    def fork(self, snapshot_id, branch_id, *, seed=None, conditioning_id=None):
        return self.restore(snapshot_id, branch_id, seed=seed, conditioning_id=conditioning_id)

    @torch.no_grad()
    def decode(self):
        self._metadata()
        vae = self.pipeline.vae
        clear = getattr(getattr(vae, "model", None), "clear_cache", None)
        if not callable(clear):
            raise UnsupportedError("Native VAE must support explicit cache reset for full-prefix decode")
        self._phase = "decoding"
        try:
            clear()
            return (vae.decode_to_pixel(self._latents.to(self.device)) * 0.5 + 0.5).clamp(0, 1)
        finally:
            clear()
            self._phase = "clean"

    def memory_report(self):
        cross = getattr(self.pipeline, "crossattn_cache", None) or []
        return {
            "live_mutable_kv_bytes": unique_tensor_bytes(self._mutable.values()),
            "live_clean_capture_cpu_bytes": 0 if self._clean_staging is None else self._clean_staging.host_bytes,
            "live_clean_capture_disk_bytes": 0 if self._clean_staging is None else self._clean_staging.disk_bytes,
            "peak_clean_capture_host_bytes": self._peak_clean_host_bytes,
            "peak_clean_capture_disk_bytes": self._peak_clean_disk_bytes,
            "current_kv_budget_bytes": self.max_current_bytes,
            "clean_capture_host_budget_bytes": self.max_clean_host_bytes,
            "clean_capture_disk_budget_bytes": self.max_clean_disk_bytes,
            "live_attention_read_bytes": unique_tensor_bytes(
                self._active_layer.tensors if self._active_layer is not None else []
            ),
            "peak_mutable_kv_bytes": self._peak_mutable_bytes,
            "peak_tracked_attention_read_bytes": self._peak_read_bytes,
            "live_cross_attention_bytes": unique_tensor_bytes(
                value for row in cross for value in row.values() if isinstance(value, torch.Tensor)
            ),
            "live_conditioning_bytes": unique_tensor_bytes(self._conditioning.values()),
            "cpu_noise_and_latents_bytes": unique_tensor_bytes((self._noise, self._latents)),
            "pending_publication": self._pending_step is not None,
            "pending_native_kv_bytes": 0 if self._pending_step is None else unique_tensor_bytes(
                self._pending_step.get("native", {}).values()),
            "pending_next_latent_bytes": 0 if self._pending_step is None else unique_tensor_bytes(
                (self._pending_step["next_latents"],)
            ),
            "runtime_pending_sources": (
                self.runtime.async_stats() if callable(getattr(self.runtime, "async_stats", None)) else None
            ),
            "layer_prefetch": None if self._prefetch is None else self._prefetch.stats(),
            "parallel_kv": None if self._paired_decoder is None else self._paired_decoder.stats(),
            "history_leases_scope": "active_and_next_layer" if self.prefetch_history_bytes else "one_layer_only",
            "native_full_history_preallocation": self.history_mode == "quality_hot",
            "retained_reconstruction_dictionary": bool(
                self._hot_history or self._quality_history or self._prefetch_active
                or (self._paired_decoder is not None and self._paired_decoder.stats()["active_pairs"])
            ),
            "history_mode": self.history_mode,
            "quality_history_budget_bytes": self.max_quality_history_bytes,
            "live_quality_history_bytes": unique_tensor_bytes(entry.tensor for entry in self._quality_history.values()),
            "peak_quality_history_bytes": self._peak_quality_bytes,
            "quality_history_reads": self._quality_reads,
            "committed_current_replacements": self._quality_replacements,
            "compressed_hbm_benchmark": self.history_mode != "quality_hot" and self.native_history_observer is None,
            "native_history_diagnostic": {
                "enabled": self.native_history_observer is not None,
                "event_count": self._native_event_ordinal,
                "hash_readback_bytes": self._native_hash_bytes,
                "peak_hash_host_workspace_bytes": self._native_hash_peak_host_bytes,
                "hash_workspace_charged_to": "clean_capture_host_budget_bytes",
                "hash_seconds": self._native_hash_seconds,
                "observer_seconds": self._native_observer_seconds,
                "extra_committed_reload_decodes": self._native_extra_decodes,
                "extra_decode_seconds": self._native_extra_decode_seconds,
                "timing_is_diagnostic_not_benchmark": self.native_history_observer is not None,
            },
            "hot_history_budget_bytes": self.max_hot_history_bytes,
            "live_hot_history_bytes": unique_tensor_bytes(item.tensor for item in self._hot_history.values()),
            "live_attention_and_hot_unique_bytes": unique_tensor_bytes([
                *(item.tensor for item in self._hot_history.values()),
                *(self._active_layer.tensors if self._active_layer is not None else []),
            ]),
            "peak_hot_history_bytes": self._peak_hot_bytes,
            "hot_history_hits": self._hot_hits,
            "hot_history_misses": self._hot_misses,
            "hot_history_scope": "fixed_head_one_chunk_epoch; complete layer prefixes; separately owned decoded tensors",
            "accounting_note": "Owned backing allocations; attention views, core leases, paired/prefetch outputs and hot-cache bytes may alias, so do not add overlapping ledgers. Allocator peaks include model/kernel workspace.",
        }

    def close(self):
        if self._pending_step is not None:
            raise PendingError("resolve deferred publication with finish_step() before uninstalling")
        if self._active_layer is not None:
            raise StateError("Cannot uninstall during a native attention forward")
        if self._prefetch is not None:
            self._prefetch.close()
            self._prefetch = None
        if self._paired_decoder is not None:
            self._paired_decoder.close()
            self._paired_decoder = None
        self._clear_hot_history()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for attention, had, previous in reversed(self._patches):
            if had:
                attention.forward = previous
            else:
                attention.__dict__.pop("forward", None)
        self._patches.clear()
        if getattr(self.pipeline, "_reelcache_causvid_owner", None) is self:
            self.pipeline.kv_cache1 = None
            self.pipeline.crossattn_cache = None
            del self.pipeline._reelcache_causvid_owner
        self._mutable.clear()
        self._pre_keys.clear()
        if self._clean_staging is not None:
            self._clean_staging.close()
            self._clean_staging = None
        self._clean_observations.clear()
        self._quality_history.clear()
        self._quality_freqs.clear()
        self._quality_epoch = None
        self._native_committed_hashes.clear()
        self._verified_pre_keys.clear()
        self._seen_layers.clear()
        self._conditioning.clear()
        self._noise = self._latents = self._rng = None
        self._installed = self._started = False
        self._phase = "uninitialized"
        self.frame_cursor = 0

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc, traceback):
        if exc is not None and self._pending_step is not None:
            self._pending_step["handle"].cancel()
            try:
                self.finish_step()
            except BaseException as cleanup_error:
                exc.add_note(f"Deferred-publication cleanup outcome: {cleanup_error!r}")
        self.close()


class _CaptureSlot:
    """One clean chunk's owned pre-RoPE K clones (the native cache stores post-RoPE K)."""

    def __init__(self, index):
        self.index = index
        self.keys = {}
        self.release_event = None
        self.free = True


class CausVidArchiveNativeAdapter:
    """``archive_native`` CausVid: native dense active KV serves generation; archival runs behind it.

    Generation is the native CausVid chunk loop over ordinary dense per-layer
    ``kv_cache1`` tensors (attention reads and writes them exactly as without
    ReelCache). The adapter owns the explicit noise generator, the dense caches,
    the clean-forward capture of normalized pre-RoPE K (owned clones in a
    bounded slot pool) and V (a lease on the dense cache rows of the finished
    chunk, which later chunks never write), and submits every clean chunk to an
    :class:`~reelcache.archive_pipeline.ArchivePipeline`. There is no barrier
    between chunks: the next denoising forward starts while earlier chunks are
    still fitted/encoded; generation waits only when the capture slots or the
    pipeline's bounded queues are exhausted (explicit backpressure).

    ``archive=None`` runs the identical native loop without archival (the
    parity reference). :meth:`snapshot` drains the published frontier to the
    generated frontier; :meth:`restore` forks the lineage at a published node
    and restores its attended history into fresh dense caches through the
    packed device path (``restorer``), then generation continues natively.
    Each node stores its chunk's clean latents and the generator state
    (chunk 0 also the noise and conditioning), so a restored execution needs
    no other session state.
    """

    SCHEMA = "reelcache.causvid.archive_native.v1"

    def __init__(self, pipeline, runtime, branch_id: str, *, config: CausVidConfig | None = None, archive=None,
                 restorer=None, capture_slots: int = 2, slot_timeout: float | None = None, rope=None,
                 verify_pre_rope: bool = False):
        if (not isinstance(branch_id, str) or not branch_id or "/" in branch_id or "\x00" in branch_id
                or branch_id in (".", "..")):
            raise ConfigurationError("CausVid branch_id must be a nonempty single-component name")
        if type(capture_slots) is not int or capture_slots < 1:
            raise ConfigurationError("capture_slots must be a positive integer")
        if archive is not None and capture_slots > archive.max_source_chunks:
            raise ConfigurationError("capture slots exceed the pipeline's source-lease bound")
        if type(verify_pre_rope) is not bool:
            raise ConfigurationError("verify_pre_rope must be boolean")
        self.pipeline, self.runtime, self.branch_id = pipeline, runtime, branch_id
        self.archive, self.restorer = archive, restorer
        self.config = config or CausVidConfig()
        self.model = pipeline.generator.model
        self.identity = runtime.model
        self.device = next(self.model.parameters()).device
        self.dtype = getattr(torch, self.identity.dtype)
        self.keys = tuple(MatrixKey(layer, target) for layer in range(self.identity.layers) for target in ("K", "V"))
        self.rope = rope
        self.verify_pre_rope = verify_pre_rope
        self.slot_timeout = slot_timeout
        self._slots = [_CaptureSlot(index) for index in range(capture_slots)]
        import threading
        self._slot_condition = threading.Condition()
        self._active_slot = None
        self._captured = set()
        self._rope_inputs = {}
        self._handles = []
        self._installed = self._started = False
        self._phase = "uninitialized"
        self._k, self._v = [], []
        self._noise = self._latents = self._rng = None
        self._conditioning, self._conditioning_id = {}, None
        self._conditioning_stored = None
        self.frame_cursor = 0
        self.tickets = []
        self.stats_ = {"chunks": 0, "slot_wait_seconds": 0.0, "slot_waits": 0, "submit_seconds": 0.0,
                       "chunk_seconds": [], "restores": []}
        self._validate_native()

    # -- validation and installation -----------------------------------------------
    def _validate_native(self):
        cfg, model, pipe = self.config, self.model, self.pipeline
        if model.training:
            raise UnsupportedError("Put the native CausVid model in evaluation mode before attaching")
        if self.identity.channels_for("K") != cfg.channels or self.identity.channels_for("V") != cfg.channels:
            raise ConfigurationError("Runtime full KV channels do not match native heads")
        if (len(model.blocks) != self.identity.layers or pipe.num_transformer_blocks != self.identity.layers
                or model.num_heads != cfg.heads or model.num_frame_per_block != cfg.frames_per_chunk
                or pipe.num_frame_per_block != cfg.frames_per_chunk or pipe.frame_seq_length != cfg.frame_tokens):
            raise ConfigurationError("CausVid model, runtime identity and chunk geometry disagree")
        for block in model.blocks:
            attention = block.self_attn
            if attention.num_heads != cfg.heads or attention.head_dim != cfg.head_dim or tuple(
                    attention.window_size) != (-1, -1):
                raise UnsupportedError("archive_native CausVid requires full-prefix attention without GQA")
            if not getattr(attention, "qk_norm", True):
                raise UnsupportedError("archive_native CausVid factorizes normalized pre-RoPE K")
            if self.rope is None:
                self.rope = attention.forward.__func__.__globals__.get("causal_rope_apply")
        if self.verify_pre_rope and not callable(self.rope):
            raise ConfigurationError("pre-RoPE verification needs the native causal_rope_apply callable")

    def install(self, *, prewarm: bool = False, prewarm_archive: bool = False):
        """Attach to the native pipeline. ``prewarm=True`` (session open over an existing store) also prepares
        the stored history of every lineage head for restore (:meth:`prewarm`); ``prewarm_archive=True`` pays
        the archive pipeline's cold start before the first chunk (:meth:`prewarm_archive`)."""
        if self._installed:
            raise StateError("CausVid adapter is already installed")
        if getattr(self.pipeline, "_reelcache_causvid_owner", None) is not None:
            raise StateError("The native pipeline already has a ReelCache owner")
        if getattr(self.pipeline, "kv_cache1", None) is not None:
            raise UnsupportedError("Attach to an empty native cache; dense adoption is unsupported")
        if type(prewarm) is not bool or type(prewarm_archive) is not bool:
            raise ConfigurationError("prewarm and prewarm_archive must be boolean")
        try:
            for layer, block in enumerate(self.model.blocks):
                self._handles.append(block.self_attn.norm_k.register_forward_hook(self._capture(layer)))
                if self.verify_pre_rope:
                    self._handles.append(block.self_attn.register_forward_pre_hook(
                        self._record_rope(layer), with_kwargs=True))
            self.pipeline._reelcache_causvid_owner = self
            self._installed = True
            if prewarm:
                self.prewarm()
            if prewarm_archive:
                self.prewarm_archive()
            return self
        except BaseException:
            self.close()
            raise

    def prewarm(self, *, branch_ids=None, node_refs=None):
        """Session-open restore preparation of stored history (off the resume path; see
        :meth:`reelcache.archive_restore.LineageRestorer.prewarm_lineages`). Returns its report."""
        if self.restorer is None:
            raise ConfigurationError("prewarm needs a LineageRestorer")
        method = getattr(self.restorer, "prewarm_lineages", None)
        if method is None:
            report = {"heads": 0, "packets": 0, "seconds": 0.0, "note": "restorer needs no preparation"}
        else:
            report = method(branch_ids, node_refs=node_refs)
        self.stats_.setdefault("prewarm", []).append(report)
        return report

    def prewarm_archive(self, *, encode: bool = True, max_frames: int | None = None):
        """Warm the archive pipeline's fitter (and encoder) for this model's chunk geometry before generation
        (:meth:`reelcache.archive_pipeline.ArchivePipeline.prewarm`; synthetic data, nothing stored)."""
        method = getattr(self.archive, "prewarm", None)
        if method is None:
            report = {"seconds": 0.0, "note": "no archive pipeline to warm"}
        else:
            cfg = self.config
            contexts = {key: TensorContext(self.identity, key, 0, cfg.frames_per_chunk, cfg.grid_height,
                                           cfg.grid_width, "pre_rope" if key.target == "K" else "none")
                        for key in self.keys}
            report = method(contexts, encode=encode, max_frames=max_frames)
        self.stats_.setdefault("prewarm_archive", []).append(report)
        return report

    def _native_rope(self, values, job):
        """The engine's own RoPE (``causal_rope_apply``) of one restored chunk: exact native post-RoPE K."""
        cfg = self.config
        frames, height, width = job.grid
        rows = job.rope.rows
        start_frame = rows[0] if isinstance(rows, tuple) else rows // cfg.frame_tokens
        grid = torch.tensor([[frames, height, width]], dtype=torch.long)
        rotated = self.rope(values.reshape(1, frames * height * width, cfg.heads, cfg.head_dim), grid,
                            self.model.freqs.to(values.device), start_frame=start_frame)
        return rotated.to(self.dtype)[0]

    def _record_rope(self, layer):
        names = ("x", "seq_lens", "grid_sizes", "freqs", "block_mask", "kv_cache", "current_start", "current_end")

        def hook(_module, args, kwargs):
            if self._phase == "clean_forward":
                fields = dict(zip(names, args))
                fields.update(kwargs)
                self._rope_inputs[layer] = (fields["grid_sizes"], fields["freqs"])
        return hook

    def _capture(self, layer):
        def hook(_module, _inputs, output):
            if self._phase != "clean_forward":
                return
            slot = self._active_slot
            if (tuple(output.shape) != (1, self.config.chunk_tokens, self.config.channels)
                    or output.dtype != self.dtype):
                raise ConfigurationError("Native normalized pre-RoPE K has unexpected shape or dtype")
            if layer in self._captured:
                raise StateError("Clean normalized-key capture was not one-to-one with the native layer")
            self._captured.add(layer)
            if slot is None:
                return
            buffer = slot.keys.get(layer)
            if buffer is None:
                buffer = torch.empty((self.config.chunk_tokens, self.config.channels), device=output.device,
                                     dtype=self.dtype)
                slot.keys[layer] = buffer
            buffer.copy_(output[0])
        return hook

    # -- execution -----------------------------------------------------------------------
    @property
    def token_cursor(self):
        return self.frame_cursor * self.config.frame_tokens

    @property
    def finished(self):
        return self._started and self.frame_cursor == self._noise.shape[1]

    @property
    def latents(self):
        return self._latents.clone()

    def _allocate(self, total_frames, *, zero=True):
        tokens = total_frames * self.config.frame_tokens
        shape = (1, tokens, self.config.heads, self.config.head_dim)
        if self._k and tuple(self._k[0].shape) == shape:
            # A restore overwrites [0, frontier) and native forwards write every later slot before reading it.
            if zero:
                for tensor in (*self._k, *self._v):
                    tensor.zero_()
        else:
            self._k = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in range(self.identity.layers)]
            self._v = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in range(self.identity.layers)]
        self.pipeline.kv_cache1 = [{"k": k, "v": v} for k, v in zip(self._k, self._v)]
        self.pipeline.crossattn_cache = [{"k": None, "v": None, "is_init": False}
                                         for _ in range(self.identity.layers)]

    def _prepare_conditioning(self, conditioning, conditioning_id):
        if not isinstance(conditioning_id, str) or not conditioning_id:
            raise ConfigurationError("An explicit conditioning identity is required")
        if not isinstance(conditioning, Mapping) or set(conditioning) != {"prompt_embeds"}:
            raise UnsupportedError("CausVid conditioning must contain only prompt_embeds")
        value = conditioning["prompt_embeds"]
        if (not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1
                or value.dtype != self.dtype or not bool(torch.isfinite(value).all())):
            raise ConfigurationError("Invalid native prompt embeddings")
        return {"prompt_embeds": value.detach().to(device=self.device, copy=True)}

    def start(self, noise, conditioning, rng, *, conditioning_id: str):
        if not self._installed or self._started:
            raise StateError("Install a fresh adapter before starting an execution")
        if (not isinstance(noise, torch.Tensor) or noise.ndim != 5 or noise.shape[0] != 1
                or noise.shape[1] % self.config.frames_per_chunk or tuple(noise.shape[2:]) != self.config.latent_tail
                or noise.dtype != self.dtype):
            raise ConfigurationError("Expected [1, chunks*frames, C, H, W] native-dtype noise")
        if not isinstance(rng, torch.Generator) or torch.device(rng.device).type != self.device.type:
            raise ConfigurationError("An explicit device-matched native RNG is required")
        self._conditioning = self._prepare_conditioning(conditioning, conditioning_id)
        self._conditioning_id = conditioning_id
        self._rng = torch.Generator(device=self.device)
        self._rng.set_state(rng.get_state())
        self._noise = noise.detach().to("cpu", copy=True).contiguous()
        self._latents = torch.empty((1, 0, *self.config.latent_tail), dtype=self.dtype)
        self._allocate(self._noise.shape[1])
        if self.archive is not None:
            self.runtime.create_lineage(self.branch_id)
        self._conditioning_stored = None
        self.frame_cursor = 0
        self._started, self._phase = True, "ready"
        return self

    def set_conditioning(self, conditioning, *, conditioning_id: str):
        self._check_ready()
        self._conditioning = self._prepare_conditioning(conditioning, conditioning_id)
        self._conditioning_id = conditioning_id
        self.pipeline.crossattn_cache = [{"k": None, "v": None, "is_init": False}
                                         for _ in range(self.identity.layers)]

    def _check_ready(self):
        if not self._installed or not self._started or self._phase not in ("ready", "clean"):
            raise StateError("Operation requires an attached, idle CausVid execution")

    def _acquire_slot(self):
        started = time.perf_counter()
        waited = False
        deadline = None if self.slot_timeout is None else time.monotonic() + self.slot_timeout
        with self._slot_condition:
            while True:
                slot = next((item for item in self._slots if item.free), None)
                if slot is not None:
                    break
                waited = True
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise CapacityError("archive capture slots stayed busy (explicit backpressure)")
                self._slot_condition.wait(remaining)
            slot.free = False
        waited_seconds = time.perf_counter() - started
        if waited:
            self.stats_["slot_waits"] += 1
        self.stats_["slot_wait_seconds"] += waited_seconds
        self.stats_.setdefault("slot_wait_per_chunk", []).append(waited_seconds)
        if slot.release_event is not None and self.device.type == "cuda":
            torch.cuda.current_stream(self.device).wait_event(slot.release_event)
        return slot

    def _release_slot(self, slot, event):
        with self._slot_condition:
            slot.release_event = event
            slot.free = True
            self._slot_condition.notify_all()

    def _forward(self, latents, timestep, start, end):
        return self.pipeline.generator(
            noisy_image_or_video=latents, conditional_dict=self._conditioning, timestep=timestep,
            kv_cache=self.pipeline.kv_cache1, crossattn_cache=self.pipeline.crossattn_cache,
            current_start=start, current_end=end,
        )

    @torch.no_grad()
    def step(self):
        """Denoise one chunk natively, clean-forward it, submit its clean K/V; no publication barrier."""
        self._check_ready()
        if self.finished:
            raise StateError("All configured chunks have already completed")
        started = time.perf_counter()
        cfg = self.config
        start, end = self.frame_cursor, self.frame_cursor + cfg.frames_per_chunk
        token_start, token_end = start * cfg.frame_tokens, end * cfg.frame_tokens
        slot = self._acquire_slot() if self.archive is not None else None
        submitted = False
        try:
            noisy = self._noise[:, start:end].to(self.device)
            steps = self.pipeline.denoising_step_list
            self._phase = "denoising"
            for index, value in enumerate(steps):
                timestep = torch.ones((1, cfg.frames_per_chunk), device=self.device, dtype=torch.int64) * value
                staged = self._forward(noisy, timestep, token_start, token_end)
                if index + 1 < len(steps):
                    noisy = self.pipeline.scheduler.add_noise(
                        staged.flatten(0, 1), torch.empty_like(staged.flatten(0, 1)).normal_(generator=self._rng),
                        steps[index + 1] * torch.ones((cfg.frames_per_chunk,), device=self.device, dtype=torch.long),
                    ).unflatten(0, staged.shape[:2])
            self._phase, self._active_slot, self._captured = "clean_forward", slot, set()
            zeros = torch.zeros((1, cfg.frames_per_chunk), device=self.device, dtype=torch.int64)
            self._forward(staged, zeros, token_start, token_end)
            if self._captured != set(range(self.identity.layers)):
                raise StateError("The clean forward did not capture every layer's normalized key")
            if self.verify_pre_rope and slot is not None:
                for layer, value in slot.keys.items():
                    grid, freqs = self._rope_inputs[layer]
                    restored = self.rope(value.reshape(1, cfg.chunk_tokens, cfg.heads, cfg.head_dim), grid, freqs,
                                         start_frame=start).to(self.dtype)
                    if not torch.equal(restored, self._k[layer][:, token_start:token_end]):
                        raise IntegrityError("captured pre-RoPE K does not reproduce the native clean cache")
            chunk_latents = staged.detach().to("cpu")
            ticket = None
            if self.archive is not None:
                event = None
                if self.device.type == "cuda":
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(self.device))
                matrices, contexts = {}, {}
                for key in self.keys:
                    if key.target == "K":
                        matrices[key] = slot.keys[key.layer]
                    else:
                        matrices[key] = self._v[key.layer][0, token_start:token_end].reshape(
                            cfg.chunk_tokens, cfg.channels)
                    contexts[key] = TensorContext(self.identity, key, token_start, cfg.frames_per_chunk,
                                                  cfg.grid_height, cfg.grid_width,
                                                  "pre_rope" if key.target == "K" else "none")
                state = {"latents": chunk_latents, "rng_state": self._rng.get_state().clone()}
                if self._conditioning_stored != self._conditioning_id:
                    state["prompt_embeds"] = self._conditioning["prompt_embeds"].detach().to("cpu")
                    state["conditioning_id"] = torch.tensor(list(self._conditioning_id.encode()), dtype=torch.uint8)
                if start == 0:
                    state["noise"] = self._noise
                submit_started = time.perf_counter()
                ticket = self.archive.submit(
                    self.branch_id, f"chunk-{start // cfg.frames_per_chunk:06d}", matrices, contexts,
                    execution_state=state, ready_event=event,
                    release=lambda released, slot=slot: self._release_slot(slot, released),
                )
                submitted = True
                self.stats_["submit_seconds"] += time.perf_counter() - submit_started
                self._conditioning_stored = self._conditioning_id
                self.tickets.append(ticket)
        except BaseException:
            self._phase = "poisoned"
            if slot is not None and not submitted:
                self._release_slot(slot, None)
            raise
        finally:
            self._active_slot = None
        self._latents = torch.cat((self._latents, chunk_latents), dim=1)
        self.frame_cursor = end
        self._phase = "clean"
        self.stats_["chunks"] += 1
        self.stats_["chunk_seconds"].append(time.perf_counter() - started)
        return ticket

    # -- checkpoints -------------------------------------------------------------------
    def drain(self, timeout=None):
        """Wait until the published frontier reaches the generated frontier; returns the head node."""
        if self.archive is None:
            raise StateError("archival is disabled for this execution")
        return self.archive.drain(self.branch_id, timeout=timeout)

    def snapshot(self, timeout=None):
        """A durable checkpoint at the generated frontier (drains the archive first)."""
        self._check_ready()
        head = self.drain(timeout)
        node = self.runtime.load_node(head)
        if node.frontier != self.token_cursor:
            raise IntegrityError("published frontier differs from the generated frontier after drain")
        return {"schema": self.SCHEMA, "branch_id": self.branch_id, "node_ref": head,
                "frame_cursor": self.frame_cursor}

    def _load_state(self, node_ref):
        from reelcache.serialization import deserialize_tensors
        latents, rng_state, noise, prompt, conditioning_id = [], None, None, None, None
        for _, node, _ in self.runtime.node_path(node_ref):
            if node.node_kind != "chunk":
                continue
            if node.execution_state_ref is None:
                raise IntegrityError("archive_native nodes carry their execution state")
            state = deserialize_tensors(self.runtime.store.get(node.execution_state_ref))
            latents.append(state["latents"])
            rng_state = state["rng_state"]
            if "noise" in state:
                noise = state["noise"]
            if "prompt_embeds" in state:
                prompt = state["prompt_embeds"]
                conditioning_id = bytes(state["conditioning_id"].tolist()).decode()
        if noise is None or prompt is None or rng_state is None:
            raise IntegrityError("restored path lacks its noise, conditioning or generator state")
        return torch.cat(latents, dim=1), rng_state, noise, prompt, conditioning_id

    @torch.no_grad()
    def restore(self, node_ref, branch_id, *, rope_table=None, layer_ahead=None):
        """Fork ``node_ref`` as ``branch_id`` and restore its attended history into the dense caches.

        ``layer_ahead`` selects the restorer's bounded layer-ahead mode (``LineageRestorer.restore``;
        histories whose canvases do not fit next to the dense caches)."""
        if not self._installed:
            raise StateError("install the adapter before restoring")
        if self.restorer is None:
            raise ConfigurationError("restore needs a LineageRestorer")
        from reelcache import restore_kernels as rk
        started = time.perf_counter()
        if self._started and self.archive is not None:
            self.archive.drain(self.branch_id)
        if branch_id in self.runtime.list_lineages():
            if self.runtime.lineage_head(branch_id) != node_ref:
                raise StateError("restore target branch exists at another head; choose a new branch id")
        else:
            self.runtime.fork_lineage(node_ref, branch_id)
        forked = time.perf_counter()
        latents, rng_state, noise, prompt, conditioning_id = self._load_state(node_ref)
        node = self.runtime.load_node(node_ref)
        loaded = time.perf_counter()
        cfg = self.config
        self._allocate(noise.shape[1], zero=False)
        allocated = time.perf_counter()
        if rope_table is None:
            # Kernel tables are cached on the RopeTable (keyed by tensor versions): build it from normal tensors
            # even when the caller restores inside torch.inference_mode.
            with torch.inference_mode(False):
                table = rk.RopeTable.from_complex(self.model.freqs)
        else:
            table = rope_table

        def destination(node_id, key, tile):
            matrix_key = key if isinstance(key, MatrixKey) else key[0]
            cache = self._k if matrix_key.target == "K" else self._v
            return rk.NativeView(cache[matrix_key.layer], "BTHD", token_offset=tile.token_start)

        def rope(node_id, key, tile):
            matrix_key = key if isinstance(key, MatrixKey) else key[0]
            if matrix_key.target != "K":
                return None
            if table.sections is None:
                return rk.RopePositions(table, tile.token_start)
            return rk.RopePositions(table, (tile.token_start // cfg.frame_tokens, 0, 0))

        options = {"native_rope": self._native_rope} if getattr(self.restorer, "exact_rope", False) else {}
        if layer_ahead is not None:
            options["layer_ahead"] = layer_ahead
        report = self.restorer.restore(node_ref, destination, rope, **options) if node.frontier else {
            "nodes": 0, "entries": 0, "seconds": 0.0, "plan_seconds": 0.0, "handles": [], "history": []}
        restored = time.perf_counter()
        self.branch_id = branch_id
        self._noise, self._latents = noise, latents
        self._rng = torch.Generator(device=self.device)
        self._rng.set_state(rng_state)
        self._conditioning = {"prompt_embeds": prompt.to(self.device)}
        self._conditioning_id = self._conditioning_stored = conditioning_id
        self.frame_cursor = node.frontier // cfg.frame_tokens
        self._started, self._phase = True, "clean" if self.frame_cursor else "ready"
        self.tickets = []
        result = {"node_ref": node_ref, "branch_id": branch_id, "frame_cursor": self.frame_cursor,
                  "restore_seconds": restored - allocated, "allocate_seconds": allocated - loaded,
                  "fork_seconds": forked - started, "execution_state_seconds": loaded - forked,
                  "total_seconds": time.perf_counter() - started, "nodes": report["nodes"],
                  "entries": report["entries"], "plan_seconds": report["plan_seconds"],
                  "breakdown": report.get("breakdown")}
        self.stats_["restores"].append(result)
        return result

    @torch.no_grad()
    def decode(self):
        vae = self.pipeline.vae
        clear = getattr(getattr(vae, "model", None), "clear_cache", None)
        if not callable(clear):
            raise UnsupportedError("Native VAE must support explicit cache reset for full-prefix decode")
        clear()
        try:
            return (vae.decode_to_pixel(self._latents.to(self.device)) * 0.5 + 0.5).clamp(0, 1)
        finally:
            clear()

    def memory_report(self):
        return {
            "native_active_kv_bytes": unique_tensor_bytes([*self._k, *self._v]),
            "capture_slot_bytes": unique_tensor_bytes(value for slot in self._slots for value in slot.keys.values()),
            "capture_slots": len(self._slots),
            "busy_capture_slots": sum(not slot.free for slot in self._slots),
            "archive": None if self.archive is None else self.archive.frontiers(self.branch_id),
            "accounting_note": "native dense KV is the active cache (archive_native); capture slots hold owned "
                               "pre-RoPE K clones until the archive releases them; V is leased from the dense cache",
        }

    def close(self, *, drain=True):
        if self._phase in ("denoising", "clean_forward"):
            raise StateError("Cannot uninstall during a native forward")
        errors = []
        if self.archive is not None and self._started:
            try:
                (self.archive.drain if drain else self.archive.cancel)(self.branch_id)
            except BaseException as exc:  # noqa: BLE001 - close still retires everything
                errors.append(exc)
                self.archive.cancel(self.branch_id)
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if getattr(self.pipeline, "_reelcache_causvid_owner", None) is self:
            self.pipeline.kv_cache1 = None
            self.pipeline.crossattn_cache = None
            del self.pipeline._reelcache_causvid_owner
        self._k, self._v = [], []
        for slot in self._slots:
            slot.keys.clear()
        self._installed = self._started = False
        self._phase = "uninitialized"
        if errors:
            raise errors[0]

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc, traceback):
        self.close(drain=exc_type is None)
