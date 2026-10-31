"""Ordered, owner-only KV transport; this module never creates process groups.

Counts are application-level tensor bytes, not NIC/PCIe protocol overhead.
Use a separate Gloo group for CPU storage traffic alongside native NCCL
attention. All ranks must call the same operations on their execution thread.
Process death is bounded by the caller's process-group timeout.

Owner-only dense transport (one canonical owner gathers/scatters reconstructed
KV) is not factor sharding: non-replicated factor ownership, where every rank
stores and decodes only its own basis rows and coefficient columns, lives in
:mod:`reelcache.factor_sharding`, which uses :class:`DistributedOwner` for its
ordered control checkpoints and :class:`KVLayout` for native KV ownership.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, dataclass
import json
import threading
from typing import Callable

from .errors import ConfigurationError, StateError


class PeerFailure(StateError):
    """A local validation, peer operation, or collective failed."""


class CollectiveCancelled(PeerFailure):
    """Cancellation was agreed before starting a payload transfer."""


@dataclass(frozen=True)
class KVLayout:
    tokens: int
    kv_heads: int
    head_dim: int
    world_size: int = 2

    def __post_init__(self):
        values = (self.tokens, self.kv_heads, self.head_dim, self.world_size)
        if any(type(value) is not int or value <= 0 for value in values):
            raise ConfigurationError("KV layout dimensions must be positive integers")
        if self.kv_heads % self.world_size:
            raise ConfigurationError("KV heads must divide evenly across ranks")
        if self.tokens < self.world_size:
            raise ConfigurationError("Every row owner needs at least one token")

    @property
    def channels(self):
        return self.kv_heads * self.head_dim

    @property
    def local_heads(self):
        return self.kv_heads // self.world_size

    @property
    def local_channels(self):
        return self.local_heads * self.head_dim

    def _check_rank(self, rank):
        if type(rank) is not int or not 0 <= rank < self.world_size:
            raise ConfigurationError("Rank is outside the KV layout")

    def rows(self, rank):
        self._check_rank(rank)
        count, remainder = divmod(self.tokens, self.world_size)
        start = rank * count + min(rank, remainder)
        return slice(start, start + count + int(rank < remainder))

    def columns(self, rank):
        self._check_rank(rank)
        return slice(rank * self.local_channels, (rank + 1) * self.local_channels)

    def shape(self, rank, layout):
        self._check_rank(rank)
        if layout == "rows":
            rows = self.rows(rank)
            return (rows.stop - rows.start, self.channels)
        if layout == "heads":
            return (self.tokens, self.local_channels)
        raise ConfigurationError("Layout must be 'rows' or 'heads', never query heads")


@dataclass
class WireCounters:
    payload_sent_bytes: int = 0
    payload_received_bytes: int = 0
    control_sent_bytes: int = 0
    control_received_bytes: int = 0
    payload_messages_sent: int = 0
    payload_messages_received: int = 0
    checkpoints: int = 0

    def to_dict(self):
        return {
            **asdict(self),
            "accounting": "application tensor bytes; control includes lengths and padding",
            "transport_protocol_overhead_included": False,
        }


class DistributedOwner:
    """An explicit collective domain with one canonical storage owner.

    ``runtime`` objects belong only on ``owner_rank``. Peers receive one
    reconstructed local column view at a time, not replicated global KV.
    The optional group must have the same rank order as native Ulysses.
    """

    def __init__(self, group=None, *, owner_rank=0, max_wire_bytes=16 * 1024**2,
                 device="cpu", expected_world_size=None):
        import torch
        import torch.distributed as dist

        self.group = group
        self.dist = dist
        self.initialized = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size(group) if self.initialized else 1
        self.rank = dist.get_rank(group) if self.initialized else 0
        if not 0 <= owner_rank < self.world_size:
            raise ConfigurationError("Invalid canonical owner rank")
        if expected_world_size is not None and self.world_size != expected_world_size:
            raise ConfigurationError("Storage and native SP world sizes differ")
        if type(max_wire_bytes) is not int or max_wire_bytes < 8:
            raise ConfigurationError("max_wire_bytes must be at least eight")
        self.owner_rank = owner_rank
        self.max_wire_bytes = max_wire_bytes
        self.device = torch.device(device)
        if self.initialized:
            backend = str(dist.get_backend(group))
            if backend == "gloo" and self.device.type != "cpu":
                raise ConfigurationError("Use CPU tensors for the storage Gloo group")
            if backend == "nccl" and self.device.type != "cuda":
                raise ConfigurationError("NCCL requires an explicitly granted CUDA device")
        self.counters = WireCounters()
        self.sequence = 0
        self._thread = threading.get_ident()
        self._broken = False

    @property
    def is_owner(self):
        return self.rank == self.owner_rank

    def _guard(self):
        if self._thread != threading.get_ident():
            raise StateError("Collectives must remain on the ordered execution thread")
        if self._broken:
            raise PeerFailure("The storage process group failed; create a new session")

    def _global_rank(self, rank):
        return self.dist.get_global_rank(self.group, rank) if self.group is not None else rank

    def _collective(self, function, *args, **kwargs):
        from .telemetry import span

        try:
            with span("transport.collective"):
                return function(*args, **kwargs)
        except Exception as exc:
            self._broken = True
            raise PeerFailure(f"Storage transport failed: {type(exc).__name__}: {exc}") from exc

    def _all_json(self, value):
        import torch

        self._guard()
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if len(raw) > 1024 * 1024:
            raise ConfigurationError("Control messages must be bounded to one MiB")
        if self.world_size == 1:
            return [json.loads(raw)]
        length = torch.tensor([len(raw)], dtype=torch.int64, device=self.device)
        lengths = [torch.empty_like(length) for _ in range(self.world_size)]
        self._collective(self.dist.all_gather, lengths, length, group=self.group)
        counts = [int(item.item()) for item in lengths]
        size = max(counts)
        if min(counts) < 0 or size > 1024 * 1024:
            self._broken = True
            raise PeerFailure("A peer supplied an invalid control message length")
        encoded = torch.zeros(size, dtype=torch.uint8, device=self.device)
        encoded[:len(raw)] = torch.tensor(list(raw), dtype=torch.uint8, device=self.device)
        gathered = [torch.empty_like(encoded) for _ in range(self.world_size)]
        self._collective(self.dist.all_gather, gathered, encoded, group=self.group)
        wire = (8 + size) * (self.world_size - 1)
        self.counters.control_sent_bytes += wire
        self.counters.control_received_bytes += wire
        return [
            json.loads(bytes(item[:count].cpu().tolist()))
            for item, count in zip(gathered, counts)
        ]

    def checkpoint(self, operation, *, error=None, cancelled=False, details=None):
        """Report failures on *every* rank before any next payload collective."""
        self._guard()
        if error is not None:
            error = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
        try:
            if len(json.dumps(details, allow_nan=False).encode()) > 512 * 1024:
                raise ConfigurationError("Checkpoint details exceed 512 KiB")
        except Exception as exc:
            error, details = f"Invalid checkpoint details: {exc}", None
        rows = self._all_json({
            "sequence": self.sequence, "operation": str(operation),
            "rank": self.rank, "error": error, "cancelled": bool(cancelled),
            "details": details,
        })
        self.sequence += 1
        self.counters.checkpoints += 1
        if len({(row["sequence"], row["operation"]) for row in rows}) != 1:
            raise PeerFailure(f"Collective order mismatch: {rows}")
        if any(row["cancelled"] for row in rows):
            raise CollectiveCancelled(f"Collective cancelled: {operation}")
        failures = [f"rank {row['rank']}: {row['error']}" for row in rows if row["error"] is not None]
        if failures:
            raise PeerFailure(f"{operation}: " + "; ".join(failures))
        return [row["details"] for row in rows]

    def owner_call(self, operation: str, function: Callable):
        """Execute owner work, propagating failure before returning on peers."""
        self.checkpoint(f"{operation}:begin")
        result, error = None, None
        if self.is_owner:
            try:
                result = function()
            except BaseException as exc:
                error = exc
        try:
            self.checkpoint(f"{operation}:complete", error=error)
        except PeerFailure as failure:
            if error is not None:
                raise failure from error
            raise
        return result

    def broadcast_json(self, value, *, operation="broadcast_json", source=None):
        source = self.owner_rank if source is None else source
        error = None
        try:
            if not 0 <= source < self.world_size:
                raise ConfigurationError("Invalid JSON source")
            json.dumps(value, allow_nan=False)
        except Exception as exc:
            error = exc
        self.checkpoint(operation, error=error)
        return self._all_json(value if self.rank == source else None)[source]

    def _move(self, tensor, source, destination):
        """Move an already allocated flat-contiguous tensor, in bounded pieces."""
        self._guard()
        if source == destination or self.world_size == 1:
            return
        flat = tensor.reshape(-1)
        elements = max(1, self.max_wire_bytes // tensor.element_size())
        for start in range(0, flat.numel(), elements):
            block = flat[start:start + elements]
            if self.rank == source:
                self._collective(
                    self.dist.send, block, dst=self._global_rank(destination), group=self.group,
                )
                self.counters.payload_sent_bytes += block.numel() * block.element_size()
                self.counters.payload_messages_sent += 1
            elif self.rank == destination:
                self._collective(
                    self.dist.recv, block, src=self._global_rank(source), group=self.group,
                )
                self.counters.payload_received_bytes += block.numel() * block.element_size()
                self.counters.payload_messages_received += 1

    def gather_matrix(self, local, layout: KVLayout, *, source_layout, operation="gather_matrix"):
        """Return canonical [global tokens, KV channels] on the owner only."""
        import torch

        error, details = None, None
        try:
            if layout.world_size != self.world_size:
                raise ConfigurationError("Layout and group world sizes differ")
            if tuple(local.shape) != layout.shape(self.rank, source_layout):
                raise ConfigurationError("Local KV shape does not match its declared layout")
            if local.device.type != self.device.type:
                raise ConfigurationError("Transfer tensor is on the wrong device")
            if local.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                raise ConfigurationError("KV transfers require a floating point tensor")
            details = {"dtype": str(local.dtype), "layout": source_layout,
                       "geometry": asdict(layout)}
        except Exception as exc:
            error = exc
        rows = self.checkpoint(operation, error=error, details=details)
        if any(row != rows[0] for row in rows):
            raise PeerFailure("Peers disagree on canonical KV layout or dtype")
        canonical, error = None, None
        try:
            local = local.detach().contiguous()
            if self.is_owner:
                canonical = torch.empty(
                    (layout.tokens, layout.channels), dtype=local.dtype, device=self.device,
                )
        except Exception as exc:
            error = exc
        self.checkpoint(f"{operation}:allocated", error=error)
        for source in range(self.world_size):
            value, error = None, None
            try:
                if self.rank == source:
                    value = local
                elif self.is_owner:
                    value = torch.empty(
                        layout.shape(source, source_layout), dtype=local.dtype, device=self.device,
                    )
            except Exception as exc:
                error = exc
            self.checkpoint(f"{operation}:rank{source}", error=error)
            if self.rank in (source, self.owner_rank):
                self._move(value, source, self.owner_rank)
            if self.is_owner:
                if source_layout == "rows":
                    canonical[layout.rows(source)].copy_(value)
                else:
                    canonical[:, layout.columns(source)].copy_(value)
            del value
        return canonical

    def local_columns(self, layout: KVLayout, loader: Callable, *, dtype,
                      operation="local_columns"):
        """Decode only one rank's column slice at a time on the owner.

        ``loader(slice)`` must return a CPU tensor for a Gloo storage group.
        Only the recipient keeps the returned local view. No full-channel
        reconstruction or query-head repeat is needed.
        """
        with self.acquire_columns(layout, loader, dtype=dtype, operation=operation) as local:
            return local.clone()

    @contextmanager
    def acquire_columns(self, layout: KVLayout, loader: Callable, *, dtype,
                        operation="acquire_columns"):
        """Hold the owner's local lease until consumption; release peer staging.

        The loader may return a context manager yielding a tensor. This allows
        runtime leases to stay charged while transport and native copies use
        them. Peers own one bounded receive allocation during this context.
        """
        import torch

        rows = self.checkpoint(operation, error=(
            None if layout.world_size == self.world_size
            else ConfigurationError("Layout and group world sizes differ")
        ), details={"layout": asdict(layout), "dtype": str(dtype)})
        if any(row != rows[0] for row in rows):
            raise PeerFailure("Peers disagree on the restoration layout or dtype")
        with ExitStack() as local_stack:
            local = None
            for destination in range(self.world_size):
                with ExitStack() as peer_stack:
                    value, error = None, None
                    try:
                        if self.is_owner:
                            loaded = loader(layout.columns(destination))
                            stack = local_stack if destination == self.rank else peer_stack
                            value = stack.enter_context(
                                loaded if hasattr(loaded, "__enter__") else nullcontext(loaded)
                            )
                            if tuple(value.shape) != layout.shape(destination, "heads"):
                                raise ConfigurationError("Decoder returned a nonlocal channel view")
                            if value.dtype != dtype or value.device.type != self.device.type:
                                raise ConfigurationError("Decoder changed transfer dtype or device")
                            if not value.is_contiguous():
                                raise ConfigurationError("Decoder must return a contiguous column view")
                        elif self.rank == destination:
                            value = torch.empty(
                                layout.shape(destination, "heads"), dtype=dtype, device=self.device,
                            )
                    except BaseException as exc:
                        error = exc
                    self.checkpoint(f"{operation}:rank{destination}", error=error)
                    if self.rank in (self.owner_rank, destination):
                        self._move(value, self.owner_rank, destination)
                    if self.rank == destination:
                        local = value
                    del value
            yield local

    def transfer_tensor(self, value, *, source, destination, operation="transfer_tensor"):
        """Transfer state without pickle, returning a tensor only on its owner."""
        import torch

        error, description = None, None
        try:
            if not 0 <= source < self.world_size or not 0 <= destination < self.world_size:
                raise ConfigurationError("Invalid tensor source/destination")
            if self.rank == source:
                value = value.detach().to(self.device).contiguous()
                description = {"shape": list(value.shape), "dtype": str(value.dtype).split(".")[-1]}
        except Exception as exc:
            error = exc
        rows = self.checkpoint(operation, error=error, details=description)
        description = rows[source]
        received, error = None, None
        try:
            if self.rank == source:
                received = value
            elif self.rank == destination:
                received = torch.empty(
                    description["shape"], dtype=getattr(torch, description["dtype"]), device=self.device,
                )
        except Exception as exc:
            error = exc
        self.checkpoint(f"{operation}:allocated", error=error)
        if self.rank in (source, destination):
            self._move(received, source, destination)
        return received if self.rank == destination else None
