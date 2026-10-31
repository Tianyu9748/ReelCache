"""Tiered sequence-parallel KV history: hot local shards plus distributed cold decode.

Under Ulysses SP each rank owns a head slice of every chunk's KV. This module
keeps three mechanisms separate from the model adapter:

* **Compression gather.** Each clean matrix is gathered once to the storage
  owner's device (NCCL over NVLink in production; Gloo on CPU in tests) and
  compressed there. The rank's own attention-ready head slice is *retained* on
  its device, so following generations keep using it without decoding.
* **Hot tier.** Retained (native) and optionally promoted (decoded) head slices
  live in a per-rank byte budget. Eviction removes the *oldest* chunks first
  (not LRU, which thrashes when every forward visits every chunk). All ranks
  apply the same operations to equal-sized entries, so membership is agreed
  and checked collectively on every load.
* **Cold tier.** A chunk that is not hot is decoded on exactly one GPU and its
  head slices are scattered to the other ranks with point-to-point NCCL. When
  several chunks are cold in one layer, they are assigned round-robin (rotated
  by layer) to different ranks and decoded concurrently, each on its own GPU
  and NVDEC engine. Compressed packets stay authoritative on the owner; only
  the few-MB packet bytes travel to a non-owner decoder over the control group.

Every payload collective is preceded by a control-group checkpoint, so a local
failure is reported on all ranks before anyone enters an NCCL operation.

With ``overlap=True`` (default) communication runs beside computation:
capture gathers are asynchronous (the forward stream never waits for them);
cold decodes run on a dedicated decode thread and CUDA stream, the scatter
runs on a separate comm stream, and the next layer's cold chunks are decoded
while the current layer computes (one-layer lookahead). The forward stream
waits only for exactly the tensors it is about to read.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import math
import struct
import time
from typing import Any, Callable, Mapping, Sequence

from .distributed import DistributedOwner, KVLayout, PeerFailure
from .errors import ConfigurationError, IntegrityError
from .telemetry import span
from .types import EncodedTensor, MatrixKey

_HEADER = struct.Struct("<Q")


def pack_encoded(encoded: EncodedTensor) -> bytes:
    """Byte-exact, self-describing wire form of a packet (metadata, payloads, basis)."""
    names = list(encoded.payloads)
    header = json.dumps({
        "metadata": encoded.metadata,
        "payloads": [[name, len(encoded.payloads[name])] for name in names],
        "basis": None if encoded.basis is None else len(encoded.basis),
    }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    parts = [_HEADER.pack(len(header)), header, *(encoded.payloads[name] for name in names)]
    if encoded.basis is not None:
        parts.append(encoded.basis)
    return b"".join(parts)


def unpack_encoded(data: bytes) -> EncodedTensor:
    if not isinstance(data, (bytes, bytearray)) or len(data) < _HEADER.size:
        raise IntegrityError("truncated packet wire form")
    (length,) = _HEADER.unpack_from(data)
    offset = _HEADER.size + length
    if offset > len(data):
        raise IntegrityError("truncated packet wire header")
    try:
        header = json.loads(bytes(data[_HEADER.size:offset]))
    except ValueError as exc:
        raise IntegrityError("invalid packet wire header") from exc
    payloads = {}
    for name, size in header["payloads"]:
        payloads[name] = bytes(data[offset:offset + size])
        offset += size
    basis = None
    if header["basis"] is not None:
        basis = bytes(data[offset:offset + header["basis"]])
        offset += header["basis"]
    if offset != len(data) or any(len(value) != size for (_, size), value in zip(header["payloads"],
                                                                               payloads.values())):
        raise IntegrityError("packet wire form length mismatch")
    return EncodedTensor(header["metadata"], payloads, basis)


@dataclass(frozen=True)
class HotEntry:
    k: Any
    v: Any
    nbytes: int
    priority: tuple
    source: str


class HotShards:
    """Per-rank attention-ready head slices keyed by (chunk_id, layer), newest-first retention."""

    def __init__(self, budget_bytes: int):
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ConfigurationError("hot budget must be nonnegative integer bytes")
        self.budget_bytes = budget_bytes
        self.entries: OrderedDict[tuple, HotEntry] = OrderedDict()
        self.bytes = 0
        self.peak_bytes = 0
        self.evictions = 0

    def get(self, key):
        return self.entries.get(key)

    def admit(self, key, k, v, *, priority, source) -> bool:
        """Insert, evicting strictly older entries only; returns whether the entry is now hot."""
        nbytes = k.untyped_storage().nbytes() + v.untyped_storage().nbytes()
        if key in self.entries:
            return True
        if nbytes > self.budget_bytes:
            return False
        victims, freed = [], 0
        if self.bytes + nbytes > self.budget_bytes:
            for other, entry in sorted(self.entries.items(), key=lambda item: item[1].priority):
                if entry.priority >= priority:
                    break
                victims.append(other)
                freed += entry.nbytes
                if self.bytes - freed + nbytes <= self.budget_bytes:
                    break
            if self.bytes - freed + nbytes > self.budget_bytes:
                return False
        for other in victims:
            self.bytes -= self.entries.pop(other).nbytes
            self.evictions += 1
        self.entries[key] = HotEntry(k, v, nbytes, priority, source)
        self.bytes += nbytes
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        return True

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def membership(self):
        return sorted([list(map(str, key)), entry.source] for key, entry in self.entries.items())


class PendingGather:
    """An in-flight gather to the owner; ``wait`` returns the canonical matrix on the owner (else None)."""

    def __init__(self, history, work, piece, pieces, layout):
        self._history, self._work, self._piece, self._pieces, self._layout = history, work, piece, pieces, layout
        self._result = None
        self._done = False

    def done(self) -> bool:
        return self._done or self._work is None or self._work.is_completed()

    def wait(self):
        """Order the *current* stream after the gather and assemble the owner's matrix on it."""
        import torch

        if self._done:
            return self._result
        if self._work is not None:
            self._work.wait()
        history = self._history
        if self._pieces is not None:
            if self._pieces[0].device.type == "cuda":
                stream = torch.cuda.current_stream(self._pieces[0].device)
                for piece in self._pieces:
                    piece.record_stream(stream)
            if self._layout == "rows":
                self._result = torch.cat([self._pieces[rank][:history.layout.shape(rank, "rows")[0]]
                                          for rank in range(history.world_size)])
            else:
                self._result = torch.cat(self._pieces, dim=1)
        self._done = True
        self._work = self._piece = self._pieces = None
        return self._result


class SPTieredHistory:
    """One rank's tiered history service; construct identically on every rank.

    ``coordinator``: the adapter's CPU/Gloo :class:`DistributedOwner` (control
    messages and packet bytes). ``data_group``: a separate process group for
    KV payloads (NCCL on CUDA devices, or Gloo on CPU). ``decoder``: this rank's
    packet decoder (e.g. ``HEVCProvider(nvdec=...)``); ``decode_column_views``
    is used when present so each rank's slice matches owner restoration bits.
    """

    PLACEMENTS = ("round_robin", "owner")

    def __init__(self, *, coordinator: DistributedOwner, data_group, layout: KVLayout, device,
                 native_dtype: str, decoder: Callable | None = None, rotary_apply: Callable | None = None,
                 hot_budget_bytes: int = 0, promote_decoded: bool = True,
                 decode_placement: str = "round_robin", column_decoder: Callable | None = None,
                 packet_cache_bytes: int = 1 << 30, decoder_factory: Callable | None = None,
                 overlap: bool = True):
        import torch
        import torch.distributed as dist

        if not isinstance(coordinator, DistributedOwner):
            raise ConfigurationError("coordinator must be the adapter's DistributedOwner")
        if coordinator.world_size != layout.world_size:
            raise ConfigurationError("layout and coordinator world sizes differ")
        if (decoder is None) == (decoder_factory is None):
            raise ConfigurationError("supply exactly one of decoder or decoder_factory")
        if (decoder is not None and not callable(decoder) and not hasattr(decoder, "decode_tensor")) or (
            decoder_factory is not None and not callable(decoder_factory)
        ) or (column_decoder is not None and not callable(column_decoder)):
            raise ConfigurationError("decoders must be callable")
        if type(overlap) is not bool:
            raise ConfigurationError("overlap must be boolean")
        if decoder_factory is not None and not overlap:
            raise ConfigurationError("decoder_factory builds the decoder on the overlap decode thread")
        if decode_placement not in self.PLACEMENTS:
            raise ConfigurationError(f"decode_placement must be one of {self.PLACEMENTS}")
        if type(promote_decoded) is not bool:
            raise ConfigurationError("promote_decoded must be boolean")
        self.coordinator, self.layout = coordinator, layout
        self.device = torch.device(device)
        self.dist = dist
        self.initialized = coordinator.initialized
        if self.initialized:
            if data_group is None or data_group is coordinator.group:
                raise ConfigurationError("KV payloads need their own process group, distinct from the coordinator")
            backend = str(dist.get_backend(data_group))
            if backend == "nccl" and self.device.type != "cuda":
                raise ConfigurationError("an NCCL data group requires CUDA tensors")
            if backend == "gloo" and self.device.type != "cpu":
                raise ConfigurationError("a Gloo data group requires CPU tensors")
            if dist.get_world_size(data_group) != layout.world_size:
                raise ConfigurationError("data group size differs from the SP layout")
        self.data_group = data_group
        self.rank, self.world_size = coordinator.rank, coordinator.world_size
        self.native_dtype = getattr(torch, native_dtype)
        self.overlap = overlap
        self.decoder = self.column_decoder = None
        self._closer = None
        self._decoder_factory = decoder_factory
        self._explicit_column_decoder = column_decoder
        if decoder is not None:
            self._bind_decoder(decoder)
            if overlap and self._closer is not None:
                raise ConfigurationError(
                    "NVDEC sessions are thread-owned; with overlap pass decoder_factory so the decoder is "
                    "built on the decode thread")
        self._pool = None
        self._decode_stream = self._comm_stream = None
        self._prefetched: dict[int, dict] = {}
        self._pending_gathers: list[PendingGather] = []
        self.rotary_apply = rotary_apply
        self.hot = HotShards(hot_budget_bytes)
        self.promote_decoded = promote_decoded
        self.decode_placement = decode_placement
        self._retained: dict[int, tuple] = {}
        if type(packet_cache_bytes) is not int or packet_cache_bytes < 0:
            raise ConfigurationError("packet_cache_bytes must be nonnegative integer bytes")
        self.packet_cache_bytes = packet_cache_bytes
        self._packets: OrderedDict[tuple, bytes] = OrderedDict()
        self._packet_bytes = 0
        rows = coordinator.checkpoint("sp_tiered_config", details={
            "budget": hot_budget_bytes, "promote": promote_decoded, "placement": decode_placement,
            "packet_cache": packet_cache_bytes, "overlap": overlap,
            "layout": [layout.tokens, layout.kv_heads, layout.head_dim, layout.world_size],
            "native_dtype": native_dtype,
        })
        if any(row != rows[0] for row in rows):
            raise ConfigurationError("SP ranks disagree on the tiered-history configuration")
        self.stats = {
            "hot_hits": 0, "cold_loads": 0, "decoded_here": 0, "promoted": 0,
            "retained_native": 0, "packets_received": 0, "packet_bytes_received": 0, "packet_cache_hits": 0,
            "slice_bytes_sent": 0, "slice_bytes_received": 0,
            "gathers": 0, "gather_bytes": 0, "gather_seconds": 0.0,
            "decode_seconds": 0.0, "exchange_seconds": 0.0, "packet_seconds": 0.0, "load_seconds": 0.0,
            "cold_by_decoder_rank": [0] * self.world_size,
            "prefetched_layers": 0, "prefetch_hits": 0, "prefetch_wait_seconds": 0.0,
        }

    # ------------------------------------------------------------------ helpers
    def _global(self, rank):
        return self.dist.get_global_rank(self.data_group, rank) if self.data_group is not None else rank

    def _agree(self, operation, *, error=None, details=None):
        rows = self.coordinator.checkpoint(operation, error=error, details=details)
        if any(row != rows[0] for row in rows):
            raise PeerFailure(f"{operation}: SP ranks disagree: {rows}")
        return rows[0]

    def _bind_decoder(self, decoder):
        if hasattr(decoder, "decode_tensor"):
            provider = decoder
            decode = provider.decode_tensor
        else:
            provider, decode = getattr(decoder, "__self__", None), decoder
        self.decoder = decode
        self.column_decoder = self._explicit_column_decoder or getattr(provider, "decode_column_views", None)
        nvdec = getattr(provider, "nvdec", None)
        self._closer = None if nvdec is None else nvdec.close

    def _worker(self):
        """The single decode thread; NVDEC sessions are owned by (and built on) it."""
        from concurrent.futures import ThreadPoolExecutor
        import torch

        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sp-decode")
            if self.device.type == "cuda":
                # High priority: the model's large GEMMs otherwise hold every SM and delay the
                # decode's small kernels (and its per-frame NVDEC syncs) until compute drains.
                high = torch.cuda.Stream.priority_range()[1] if hasattr(torch.cuda.Stream, "priority_range") else -1
                self._decode_stream = torch.cuda.Stream(self.device, priority=high)
                self._comm_stream = torch.cuda.Stream(self.device, priority=high)
            if self._decoder_factory is not None:
                self._pool.submit(lambda: self._bind_decoder(self._decoder_factory())).result()
        return self._pool

    def _stream_ctx(self, stream):
        import contextlib
        import torch

        return torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()

    def close(self):
        """Finish lookahead work and release the decode thread (closing an owned NVDEC on it)."""
        prefetched, self._prefetched = self._prefetched, {}
        for entry in prefetched.values():
            try:
                entry["future"].result()
            except Exception:
                pass
        self.drain_gathers()
        if self._pool is not None:
            if self._closer is not None:
                self._pool.submit(self._closer).result()
            self._pool.shutdown(wait=True)
            self._pool = None
        elif self._closer is not None:
            self._closer()
        self._closer = None

    def drain_gathers(self):
        """Wait (current stream) for this rank's outstanding async capture sends."""
        pending, self._pending_gathers = self._pending_gathers, []
        for gather in pending:
            gather.wait()

    # --------------------------------------------------------------- compression
    def gather_to_owner(self, local, source_layout: str, *, operation="sp_gather", error=None, async_op=False):
        """Collective: return the canonical [tokens, channels] matrix on the owner's device.

        A caller-side ``error`` is reported collectively instead of entering the gather.
        With ``async_op`` a :class:`PendingGather` is returned on every rank and the
        caller's stream does not wait; the owner calls ``wait()`` on the stream that
        consumes the matrix. Non-owner sends are tracked and pruned here.
        """
        import torch

        try:
            if error is not None:
                raise error
            if tuple(local.shape) != self.layout.shape(self.rank, source_layout):
                raise ConfigurationError("local KV shard does not match its declared layout")
            if local.device.type != self.device.type:
                raise ConfigurationError("local KV shard is on the wrong device type")
        except Exception as exc:
            error = exc
        self._agree(operation, error=error, details={"layout": source_layout, "dtype": str(local.dtype)})
        started = time.perf_counter()
        owner = self.coordinator.owner_rank
        if self.world_size == 1:
            full = local.contiguous()
        else:
            if source_layout == "rows":
                size = math.ceil(self.layout.tokens / self.world_size)
                piece = torch.zeros((size, self.layout.channels), dtype=local.dtype, device=local.device)
                piece[:local.shape[0]].copy_(local)
            else:
                piece = local.contiguous()
            pieces = ([torch.empty_like(piece) for _ in range(self.world_size)]
                      if self.rank == owner else None)
            work = self.dist.gather(piece, pieces, dst=self._global(owner), group=self.data_group,
                                    async_op=async_op)
            if async_op:
                pending = PendingGather(self, work, piece, pieces if self.rank == owner else None, source_layout)
                self.stats["gathers"] += 1
                self.stats["gather_bytes"] += local.numel() * local.element_size()
                self.stats["gather_seconds"] += time.perf_counter() - started
                if self.rank != owner:
                    self._pending_gathers = [g for g in self._pending_gathers if not g.done()]
                    self._pending_gathers.append(pending)
                return pending
            full = None
            if self.rank == owner:
                if source_layout == "rows":
                    full = torch.cat([pieces[rank][:self.layout.shape(rank, "rows")[0]]
                                      for rank in range(self.world_size)])
                else:
                    full = torch.cat(pieces, dim=1)
            del pieces, piece
        nbytes = local.numel() * local.element_size()
        self.stats["gathers"] += 1
        self.stats["gather_bytes"] += nbytes
        self.stats["gather_seconds"] += time.perf_counter() - started
        return full

    def retain(self, layer: int, k_local, v_local):
        """Keep this rank's clean attention-ready head slice pending until the chunk commits."""
        if self.hot.budget_bytes == 0:
            return
        k, v = k_local.detach().clone(), v_local.detach().clone()
        if k.untyped_storage().nbytes() + v.untyped_storage().nbytes() > self.hot.budget_bytes:
            return
        self._retained[layer] = (k, v)

    def commit_retained(self, chunk_id: str, token_start: int):
        """Promote retained native slices of a published chunk into the hot tier (newest priority)."""
        for layer in sorted(self._retained):
            k, v = self._retained[layer]
            if self.hot.admit((chunk_id, layer), k, v, priority=(token_start, layer), source="native"):
                self.stats["retained_native"] += 1
        self._retained.clear()

    def discard_retained(self):
        self._retained.clear()

    def clear(self):
        """Drop all hot state and cached packets (e.g. after restore/fork to another branch)."""
        self.hot.clear()
        self._retained.clear()
        self._packets.clear()
        self._packet_bytes = 0
        prefetched, self._prefetched = self._prefetched, {}
        for entry in prefetched.values():
            try:
                entry["future"].result()
            except Exception:
                pass

    # ------------------------------------------------------------ restoration
    def _placement(self, layer, cold):
        """Sticky chunk -> decoder rank: chunk ordinal modulo SP size (or the owner only).

        Stickiness lets a non-owner decoder keep that chunk's compressed packets
        cached, so each packet crosses the wire at most once.
        """
        if self.decode_placement == "owner":
            return {chunk["chunk_id"]: self.coordinator.owner_rank for chunk in cold}
        return {chunk["chunk_id"]: (chunk["token_start"] // self.layout.tokens) % self.world_size
                for chunk in cold}

    def _cache_packet(self, name, blob):
        if len(blob) > self.packet_cache_bytes:
            return
        while self._packet_bytes + len(blob) > self.packet_cache_bytes:
            _, old = self._packets.popitem(last=False)
            self._packet_bytes -= len(old)
        self._packets[name] = blob
        self._packet_bytes += len(blob)

    def _distribute_packets(self, layer, cold, assignment, packet_source):
        """Deliver each decoder its chunks' K/V packets; non-owners cache them for later loads.

        Transfers use the data group (host->device, NCCL/NVLink, device->host)
        when it is on CUDA, otherwise the Gloo control group.
        """
        import torch

        owner = self.coordinator.owner_rank
        names = [(chunk["chunk_id"], target) for chunk in cold for target in ("K", "V")]
        have = sorted(f"{cid}|{target}" for cid, target in names
                      if assignment[cid] == self.rank and self.rank != owner and (cid, layer, target) in self._packets)
        # The owner also caches packed packets, avoiding repeated store reads for chunks it decodes.
        rows = self.coordinator.checkpoint(f"sp_packet_cache:{layer}", details=have)
        needed = [(cid, target) for cid, target in names
                  if assignment[cid] != owner and f"{cid}|{target}" not in rows[assignment[cid]]]
        blobs, sizes, error = {}, None, None
        if self.rank == owner:
            try:
                for cid, target in names:
                    if assignment[cid] == owner or (cid, target) in needed:
                        cached = self._packets.get((cid, layer, target))
                        if cached is None:
                            chunk = next(c for c in cold if c["chunk_id"] == cid)
                            cached = pack_encoded(packet_source(chunk, MatrixKey(layer, target)))
                            if assignment[cid] == owner:
                                self._cache_packet((cid, layer, target), cached)
                        else:
                            self._packets.move_to_end((cid, layer, target))
                            self.stats["packet_cache_hits"] += 1
                        blobs[(cid, target)] = cached
                sizes = {f"{cid}|{target}": len(blobs[(cid, target)]) for cid, target in needed}
            except Exception as exc:
                error = exc
        self._agree(f"sp_packets:{layer}", error=error)
        sizes = self.coordinator.broadcast_json(sizes, operation=f"sp_packet_sizes:{layer}") if needed else {}
        received = {}
        if needed:
            if self.device.type == "cuda":
                self._worker() if self.overlap else None
                ops, buffers = [], {}
                for cid, target in needed:
                    decoder = assignment[cid]
                    if self.rank == owner:
                        host = torch.frombuffer(bytearray(blobs[(cid, target)]), dtype=torch.uint8)
                        with self._stream_ctx(self._comm_stream):
                            buffer = host.pin_memory().to(self.device, non_blocking=True)
                        ops.append(self.dist.P2POp(self.dist.isend, buffer, self._global(decoder), self.data_group))
                    elif self.rank == decoder:
                        with self._stream_ctx(self._comm_stream):
                            buffer = torch.empty(sizes[f"{cid}|{target}"], dtype=torch.uint8, device=self.device)
                        ops.append(self.dist.P2POp(self.dist.irecv, buffer, self._global(owner), self.data_group))
                    else:
                        continue
                    buffers[(cid, target)] = buffer
                with self._stream_ctx(self._comm_stream):
                    if ops:
                        for work in self.dist.batch_isend_irecv(ops):
                            work.wait()
                    for name, buffer in buffers.items():
                        if self.rank != owner:
                            received[name] = buffer.cpu().numpy().tobytes()
                del buffers
            else:
                for cid, target in needed:
                    decoder = assignment[cid]
                    if self.rank == owner:
                        buffer = torch.frombuffer(bytearray(blobs[(cid, target)]), dtype=torch.uint8)
                    elif self.rank == decoder:
                        buffer = torch.empty(sizes[f"{cid}|{target}"], dtype=torch.uint8)
                    else:
                        continue
                    self.coordinator._move(buffer, owner, decoder)
                    if self.rank == decoder:
                        received[(cid, target)] = buffer.numpy().tobytes()
        mine = {}
        for cid, target in names:
            if assignment[cid] != self.rank:
                continue
            if self.rank == owner:
                mine[(cid, target)] = blobs[(cid, target)]
            elif (cid, target) in received:
                blob = received[(cid, target)]
                mine[(cid, target)] = blob
                self.stats["packets_received"] += 1
                self.stats["packet_bytes_received"] += len(blob)
                self._cache_packet((cid, layer, target), blob)
            else:
                mine[(cid, target)] = self._packets[(cid, layer, target)]
                self._packets.move_to_end((cid, layer, target))
                self.stats["packet_cache_hits"] += 1
        return mine

    def _decode_slices(self, chunk, target, blob, rope_for):
        """Decode one packet into per-rank attention-ready [tokens, local_heads, head_dim] slices."""
        import torch

        layout = self.layout
        encoded = unpack_encoded(blob)
        pre_rope = target == "K" and chunk["spaces"][str(chunk["layer"])] == "pre_rope"
        dtype = torch.float32 if pre_rope else self.native_dtype
        groups = [layout.columns(rank) for rank in range(self.world_size)]
        if self.column_decoder is not None and encoded.metadata.get("kind") != "raw":
            views = self.column_decoder(encoded, column_groups=groups, device=self.device, dtype=dtype)
            views = [views[index] for index in range(self.world_size)]
        else:
            full = self.decoder(encoded, device=self.device, dtype=dtype)
            views = [full[:, group] for group in groups]
        slices = []
        for view in views:
            value = view.reshape(1, layout.tokens, layout.local_heads, layout.head_dim)
            if pre_rope:
                cos, sin = rope_for(chunk)
                value = self.rotary_apply(value, cos.to(self.device), sin.to(self.device))
            value = value.to(self.native_dtype).reshape(layout.tokens, layout.local_heads, layout.head_dim)
            if not value.is_contiguous() or value.untyped_storage().nbytes() != value.numel() * value.element_size():
                value = value.clone(memory_format=torch.contiguous_format)
            slices.append(value)
        return slices

    def _decode_assigned(self, layer, cold, assignment, packets, rope_for):
        """Decode this rank's assigned cold chunks; returns (slices by (chunk, target), ready event)."""
        import torch

        decoded = {}
        started = time.perf_counter()
        with self._stream_ctx(self._decode_stream):
            with span("sp.decode"):
                for chunk in cold:
                    if assignment[chunk["chunk_id"]] != self.rank:
                        continue
                    for target in ("K", "V"):
                        decoded[(chunk["chunk_id"], target)] = self._decode_slices(
                            chunk, target, packets[(chunk["chunk_id"], target)], rope_for)
                    self.stats["decoded_here"] += 1
            event = None
            if self.device.type == "cuda":
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(self.device))
                if not self.overlap:
                    torch.cuda.current_stream(self.device).synchronize()
        self.stats["decode_seconds"] += time.perf_counter() - started
        return decoded, event

    def _start_decode(self, layer, cold, assignment, packets, rope_for):
        from concurrent.futures import Future

        if self.overlap:
            return self._worker().submit(self._decode_assigned, layer, cold, assignment, packets, rope_for)
        future = Future()
        try:
            future.set_result(self._decode_assigned(layer, cold, assignment, packets, rope_for))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def _plan(self, layer, chunks):
        return [{**chunk, "layer": layer} for chunk in chunks if self.hot.get((chunk["chunk_id"], layer)) is None]

    def _prefetch(self, layer, chunks, packet_source, rope_for):
        """Collective: ship packets for ``layer``'s cold chunks now and decode them in the background."""
        cold = self._plan(layer, chunks)
        self._agree(f"sp_prefetch:{layer}", details=[chunk["chunk_id"] for chunk in cold])
        if not cold:
            return
        assignment = self._placement(layer, cold)
        tick = time.perf_counter()
        with span("sp.packets"):
            packets = self._distribute_packets(layer, cold, assignment, packet_source)
        self.stats["packet_seconds"] += time.perf_counter() - tick
        self._prefetched[layer] = {
            "ids": tuple(chunk["chunk_id"] for chunk in cold), "assignment": assignment,
            "future": self._start_decode(layer, cold, assignment, packets, rope_for),
        }
        self.stats["prefetched_layers"] += 1

    def load_layer(self, layer: int, chunks: Sequence[Mapping[str, Any]], *,
                   packet_source: Callable | None, rope_for: Callable | None,
                   prefetch_layer: int | None = None):
        """Collective: return [(k_local, v_local)] for ``chunks`` in order, on this rank's device.

        ``chunks`` rows need ``chunk_id``, ``token_start`` and ``spaces`` (layer ->
        "pre_rope"/"post_rope"). ``packet_source(chunk, MatrixKey)`` is only
        called on the owner. ``rope_for(chunk)`` returns the chunk's full-token
        (cos, sin) and is only needed for pre-RoPE K. With ``overlap``,
        ``prefetch_layer`` (normally ``layer + 1``) starts decoding that layer's
        cold chunks now, so they decode while this layer computes. The returned
        tensors are ordered on the caller's current stream.
        """
        import torch

        started = time.perf_counter()
        result: dict[str, tuple] = {}
        cold = self._plan(layer, chunks)
        for chunk in chunks:
            entry = self.hot.get((chunk["chunk_id"], layer))
            if entry is not None:
                result[chunk["chunk_id"]] = (entry.k, entry.v)
        self._agree(f"sp_load:{layer}", details={
            "cold": [chunk["chunk_id"] for chunk in cold], "hot": sorted(result),
            "hot_bytes": self.hot.bytes,
        })
        self.stats["hot_hits"] += len(result)
        prefetched = self._prefetched.pop(layer, None)
        if cold:
            assignment = self._placement(layer, cold)
            for chunk in cold:
                self.stats["cold_by_decoder_rank"][assignment[chunk["chunk_id"]]] += 1
            self.stats["cold_loads"] += len(cold)
            covered = set(prefetched["ids"]) if prefetched is not None else set()
            missing = [chunk for chunk in cold if chunk["chunk_id"] not in covered]
            futures = []
            if prefetched is not None:
                futures.append(prefetched["future"])
                self.stats["prefetch_hits"] += len(cold) - len(missing)
            if missing:
                tick = time.perf_counter()
                with span("sp.packets"):
                    packets = self._distribute_packets(layer, missing, assignment, packet_source)
                self.stats["packet_seconds"] += time.perf_counter() - tick
                futures.append(self._start_decode(layer, missing, assignment, packets, rope_for))
            decoded, events, error = {}, [], None
            tick = time.perf_counter()
            for future in futures:
                try:
                    part, event = future.result()
                    decoded.update(part)
                    if event is not None:
                        events.append(event)
                except Exception as exc:
                    error = error or exc
            self.stats["prefetch_wait_seconds"] += time.perf_counter() - tick
            self._agree(f"sp_decoded:{layer}", error=error)
            tick = time.perf_counter()
            with span("sp.exchange"):
                received = self._exchange(cold, assignment, decoded, events)
            self.stats["exchange_seconds"] += time.perf_counter() - tick
            for chunk in cold:
                cid = chunk["chunk_id"]
                k, v = received[(cid, "K")], received[(cid, "V")]
                result[cid] = (k, v)
                if self.promote_decoded and self.hot.admit(
                    (cid, layer), k, v, priority=(chunk["token_start"], layer), source="decoded",
                ):
                    self.stats["promoted"] += 1
        elif prefetched is not None:
            try:
                prefetched["future"].result()
            except Exception:
                pass
        if self.overlap and prefetch_layer is not None:
            self._prefetch(prefetch_layer, chunks, packet_source, rope_for)
        self.stats["load_seconds"] += time.perf_counter() - started
        return [result[chunk["chunk_id"]] for chunk in chunks]

    def _exchange(self, cold, assignment, decoded, events=()):
        """Scatter decoded head slices from each chunk's decoder rank to every other rank.

        Sends/receives run on the comm stream (after the decode event); only the
        caller's current stream is then made to wait for the received tensors.
        """
        import torch

        layout, ops, received = self.layout, [], {}
        shape = (layout.tokens, layout.local_heads, layout.head_dim)
        cuda = self.device.type == "cuda"
        current = torch.cuda.current_stream(self.device) if cuda else None
        comm = self._comm_stream if (cuda and self.overlap) else None
        with self._stream_ctx(comm):
            if cuda:
                for event in events:
                    torch.cuda.current_stream(self.device).wait_event(event)
            for chunk in cold:
                decoder = assignment[chunk["chunk_id"]]
                for target in ("K", "V"):
                    name = (chunk["chunk_id"], target)
                    if self.rank == decoder:
                        slices = decoded[name]
                        received[name] = slices[self.rank]
                        for destination in range(self.world_size):
                            if destination != self.rank:
                                ops.append(self.dist.P2POp(self.dist.isend, slices[destination],
                                                           self._global(destination), self.data_group))
                                self.stats["slice_bytes_sent"] += (
                                    slices[destination].numel() * slices[destination].element_size())
                    else:
                        buffer = torch.empty(shape, dtype=self.native_dtype, device=self.device)
                        received[name] = buffer
                        ops.append(self.dist.P2POp(self.dist.irecv, buffer, self._global(decoder), self.data_group))
                        self.stats["slice_bytes_received"] += buffer.numel() * buffer.element_size()
            works = self.dist.batch_isend_irecv(ops) if ops else []
        # Order only the consumer (current) stream after the transfers and local decodes.
        for work in works:
            work.wait()
        if cuda:
            for event in events:
                current.wait_event(event)
            for tensor in received.values():
                tensor.record_stream(current)
            if comm is not None:
                for group in decoded.values():
                    for tensor in group:
                        tensor.record_stream(comm)
        return received

    def report(self):
        return {**self.stats, "hot_bytes": self.hot.bytes, "hot_peak_bytes": self.hot.peak_bytes,
                "hot_entries": len(self.hot.entries), "hot_evictions": self.hot.evictions,
                "hot_budget_bytes": self.hot.budget_bytes, "decode_placement": self.decode_placement,
                "packet_cache_bytes": self._packet_bytes, "overlap": self.overlap}
