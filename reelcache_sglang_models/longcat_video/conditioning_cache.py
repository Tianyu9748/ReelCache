"""Typed seam for the LongCat-Video conditioning cache (ReelCache capture / restore install).

Each continuation segment re-encodes its last 13 RGB frames into 4 conditioning latents and runs one
timestep-zero DiT forward over them with cross-attention skipped. The keys and values of all 48 layers
of that forward are the segment's *conditioning cache*: keys are captured after ``k_norm`` and before
3D RoPE (pre-RoPE, the only ReelCache key domain), values directly. The 16 denoising steps of the segment
read the cache and apply segment-local RoPE at read time; the next segment builds a new cache from its
own last 13 frames (a new :class:`ConditioningCacheEpoch`, never appended to the previous one).

This module is pure torch. The SGLang stages call a :class:`ConditioningCacheBackend` (looked up by
name in a process-local registry, so requests only carry a plain string) right after a native build
(``on_cache_built``: capture) and before building (``restore``: install a restored cache instead of
running the forward). ReelCache compression and compressed restore live in :mod:`.reelcache_backend`.

Per-rank seam (context parallelism, :mod:`.context_parallel`): under native LongCat CP every rank builds and reads
only its token rows of the cache. The epoch then carries a :class:`~.context_parallel.RowShard` (``row_shard``) and
each rank's :class:`LongCatConditioningCache` holds ``[1, T_local, H, D]`` pre-RoPE K and V of exactly the rows
``row_shard.rows(token_grid)`` (= ``reelcache.factor_sharding.NativeOwners.spatial_rows(**row_shard.native_owners_kwargs
(...)).rows(rank)``). Capture: ``on_cache_built`` sees the rank's shard (``layer_matrices`` gives the ``[T_local, C]``
row matrices; summing ``row_shard_gram`` of them over ranks is the full Gram). Install: ``restore`` returns the
rank's shard built with :meth:`LongCatConditioningCache.from_matrices` (factor-sharded restore output ``[T_local, C]``,
keys restored with ``rope=None``) or :meth:`LongCatConditioningCache.from_native_layout` (``[1, H, T_local, D]``).
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch

from .context_parallel import RowShard

LAYOUT_BTHD = "BTHD"  # [B, T_tokens, H, D]: SGLang / FlashAttention layout used by this port
LAYOUT_NATIVE_BHTD = "BHTD"  # [B, H, T_tokens, D]: native LongCat cache layout
SOURCE_NATIVE_BUILD = "native_build"
SOURCE_RESTORED = "restored"
EPOCH_CAUSE = "longcat_segment"


class ConditioningCacheError(RuntimeError):
    """A conditioning cache or backend violates the seam contract."""


def tensor_sha256(t: torch.Tensor) -> str:
    """SHA256 of a tensor's bytes (BF16/FP16 hashed through an int16 view, as the native harnesses do)."""
    x = t.detach().cpu().contiguous()
    if x.dtype in (torch.bfloat16, torch.float16):
        x = x.view(torch.int16)
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class ConditioningCacheEpoch:
    """Identity of one segment's conditioning cache (immutable, plain data)."""

    segment_index: int
    token_grid: tuple[int, int, int]  # (conditioning latent frames, token rows, token cols), e.g. (4, 44, 82)
    cond_latents_sha256: str
    model_fingerprint: str
    session_id: str | None = None
    parent_segment_index: int | None = None
    cause: str = EPOCH_CAUSE
    position_offset: int = 0  # segment-local: the cache occupies temporal positions [0, frames)
    key_domain: str = "pre_rope"
    layout: str = LAYOUT_BTHD
    row_shard: RowShard | None = None  # context parallelism: this rank's token rows of the epoch (None: all rows)

    def __post_init__(self) -> None:
        if self.segment_index < 0:
            raise ConditioningCacheError("segment_index must be non-negative")
        if len(self.token_grid) != 3 or min(self.token_grid) <= 0:
            raise ConditioningCacheError(f"invalid token grid {self.token_grid}")
        if self.position_offset != 0:
            raise ConditioningCacheError("LongCat conditioning caches use segment-local positions (offset 0)")
        if self.key_domain != "pre_rope":
            raise ConditioningCacheError("conditioning-cache keys must be pre-RoPE")
        if self.layout not in (LAYOUT_BTHD, LAYOUT_NATIVE_BHTD):
            raise ConditioningCacheError(f"unknown layout {self.layout!r}")
        if self.parent_segment_index is not None and self.parent_segment_index != self.segment_index - 1:
            raise ConditioningCacheError("a segment's parent epoch is the previous segment")
        object.__setattr__(self, "token_grid", tuple(int(x) for x in self.token_grid))
        if self.row_shard is not None:
            if not isinstance(self.row_shard, RowShard):
                raise ConditioningCacheError("row_shard must be a RowShard")
            try:
                self.row_shard.check_grid(self.token_grid)
            except RuntimeError as exc:
                raise ConditioningCacheError(str(exc)) from exc

    @property
    def tokens(self) -> int:
        t, h, w = self.token_grid
        return t * h * w

    @property
    def tokens_per_frame(self) -> int:
        return self.token_grid[1] * self.token_grid[2]

    @property
    def local_tokens(self) -> int:
        """Token rows a cache of this epoch holds (this rank's rows under context parallelism)."""
        return self.tokens if self.row_shard is None else self.row_shard.local_tokens(self.token_grid)

    def row_index(self) -> torch.Tensor:
        """Native token rows (int64, CPU) held by a cache of this epoch, in storage order."""
        if self.row_shard is None:
            return torch.arange(self.tokens, dtype=torch.long)
        return self.row_shard.rows(self.token_grid)

    def full_epoch(self) -> "ConditioningCacheEpoch":
        """The same epoch without the row shard (identity of the whole cache across ranks)."""
        if self.row_shard is None:
            return self
        data = dict(self.__dict__)
        data["row_shard"] = None
        return ConditioningCacheEpoch(**data)

    def as_dict(self) -> dict:
        out = {"segment_index": self.segment_index, "token_grid": list(self.token_grid), "tokens": self.tokens,
               "cond_latents_sha256": self.cond_latents_sha256, "model_fingerprint": self.model_fingerprint,
               "session_id": self.session_id, "parent_segment_index": self.parent_segment_index,
               "cause": self.cause, "position_offset": self.position_offset, "key_domain": self.key_domain,
               "layout": self.layout}
        if self.row_shard is not None:
            out["row_shard"] = self.row_shard.as_dict()
            out["local_tokens"] = self.local_tokens
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "ConditioningCacheEpoch":
        """Inverse of :meth:`as_dict` (derived fields ``tokens`` / ``local_tokens`` are recomputed)."""
        values = {k: v for k, v in data.items() if k not in ("tokens", "local_tokens")}
        values["token_grid"] = tuple(values["token_grid"])
        if values.get("row_shard") is not None:
            values["row_shard"] = RowShard.from_dict(values["row_shard"])
        return cls(**values)


class LayerReadiness:
    """Per-layer readiness of a conditioning cache whose layers are still being written (overlapped restore).

    The producer (a restore thread) calls :meth:`mark` after it has issued layer ``i``'s writes on its stream (a CUDA
    event is recorded there; on the CPU the writes are complete when ``mark`` is called), or :meth:`fail` once with
    the exception (every waiter then raises it). The consumer (the DiT forward) calls :meth:`wait` before reading
    layer ``i``: a host wait until the layer was issued, then a device-side wait of the consumer's stream on the
    layer's event -- the GPU never reads a layer before its writes complete, and the host is never blocked longer than
    until the writes were issued. :meth:`join` waits for the producer thread and re-raises its failure; its record
    adds the consumer's host waits (``consumer_waits``: count, seconds, max, waits over 1 ms -- how much of the restore
    stayed exposed in the forward; d5).

    A producer failure that is a ``reelcache.errors.IntegrityError`` (a damaged archive, e.g. a payload the asynchronous
    verification found corrupt after step 0 started) is re-raised as an ``IntegrityError`` (``from`` the original), so
    the request fails typed (``integrity``) and a fork or rewind is rolled back exactly as in the decode-failure path;
    any other failure as :class:`ConditioningCacheError`.
    """

    def __init__(self, layers: int, device: torch.device | str | None = None, *, timeout: float = 600.0) -> None:
        if type(layers) is not int or layers < 1:
            raise ConditioningCacheError("readiness needs a positive layer count")
        self.layers = layers
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.timeout = float(timeout)
        self._flags = [threading.Event() for _ in range(layers)]
        self._events = [torch.cuda.Event() if self.device.type == "cuda" else None for _ in range(layers)]
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self.marked_at: list[float | None] = [None] * layers
        self.info: dict = {}
        self.consumer_waits: list[tuple[int, float]] = []  # (layer, host seconds) of every wait

    def mark(self, layer: int, stream=None) -> None:
        import time
        if self._events[layer] is not None:
            self._events[layer].record(stream)  # stream=None: the producer thread's current stream
        self.marked_at[layer] = time.perf_counter()
        self._flags[layer].set()

    def fail(self, error: BaseException) -> None:
        if self._error is None:
            self._error = error
        for flag in self._flags:
            flag.set()

    def _failure(self) -> BaseException:
        """The producer's failure as waiters raise it (see the class docstring)."""
        message = f"overlapped conditioning-cache restore failed: {self._error!r}"
        try:
            from reelcache.errors import IntegrityError
        except ImportError:  # without reelcache the producer's failure cannot be a reelcache IntegrityError
            return ConditioningCacheError(message)
        if isinstance(self._error, IntegrityError):
            return IntegrityError(message)
        return ConditioningCacheError(message)

    def wait(self, layer: int, stream=None) -> None:
        import time
        started = time.perf_counter()
        if not self._flags[layer].wait(self.timeout):
            raise ConditioningCacheError(f"conditioning-cache layer {layer} was not restored within {self.timeout} s")
        self.consumer_waits.append((layer, time.perf_counter() - started))
        if self._error is not None:
            raise self._failure() from self._error
        event = self._events[layer]
        if event is not None:
            (stream if stream is not None else torch.cuda.current_stream(self.device)).wait_event(event)

    def ready(self, layer: int) -> bool:
        return self._flags[layer].is_set() and self._error is None

    def attach(self, thread: threading.Thread) -> None:
        self._thread = thread

    def join(self) -> dict:
        if self._thread is not None:
            self._thread.join(self.timeout)
            if self._thread.is_alive():
                raise ConditioningCacheError("overlapped conditioning-cache restore did not finish")
        if self._error is not None:
            raise self._failure() from self._error
        return {**self.info, "consumer_waits": self.wait_summary()}

    def wait_summary(self) -> dict:
        """The consumer's host waits so far: how much of the restore stayed exposed in the forward."""
        waits = [seconds for _, seconds in self.consumer_waits]
        return {"count": len(waits), "seconds": sum(waits), "max_seconds": max(waits, default=0.0),
                "over_1ms": sum(1 for seconds in waits if seconds > 1e-3),
                "layers_over_1ms": [layer for layer, seconds in self.consumer_waits if seconds > 1e-3][:8]}


@dataclass
class LongCatConditioningCache:
    """Live per-layer pre-RoPE keys and values of one epoch (process-local resource owner).

    ``cond_latents`` optionally carries the epoch's 4 normalized conditioning latents (``[1, 16, 4, h, w]``, the
    input of the cache build) so an archive can store them as execution state; ``metadata`` stays JSON-safe.
    """

    epoch: ConditioningCacheEpoch
    keys_pre_rope: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]
    source: str = SOURCE_NATIVE_BUILD
    metadata: dict = field(default_factory=dict)
    cond_latents: torch.Tensor | None = None
    ready: LayerReadiness | None = None  # overlapped restore: layers become readable progressively (see LayerReadiness)

    def __post_init__(self) -> None:
        self.keys_pre_rope = tuple(self.keys_pre_rope)
        self.values = tuple(self.values)
        if self.source not in (SOURCE_NATIVE_BUILD, SOURCE_RESTORED):
            raise ConditioningCacheError(f"unknown cache source {self.source!r}")
        if self.cond_latents is not None:
            if not isinstance(self.cond_latents, torch.Tensor) or self.cond_latents.ndim != 5:
                raise ConditioningCacheError("cond_latents must be a [B, C, T, H, W] tensor")
            if self.cond_latents.shape[2] != self.epoch.token_grid[0]:
                raise ConditioningCacheError("cond_latents frames differ from the epoch's token grid")

    @property
    def num_layers(self) -> int:
        return len(self.keys_pre_rope)

    @property
    def layout(self) -> str:
        return self.epoch.layout

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (*self.keys_pre_rope, *self.values))

    def validate(self, *, layers: int, heads: int, head_dim: int, dtype: torch.dtype | None = None,
                 batch: int = 1, device: torch.device | str | None = None) -> None:
        if len(self.keys_pre_rope) != layers or len(self.values) != layers:
            raise ConditioningCacheError(
                f"cache has {len(self.keys_pre_rope)}/{len(self.values)} K/V layers, expected {layers}")
        tokens = self.epoch.local_tokens
        expect = (batch, tokens, heads, head_dim) if self.layout == LAYOUT_BTHD else (batch, heads, tokens, head_dim)
        for i, (k, v) in enumerate(zip(self.keys_pre_rope, self.values)):
            for name, t in (("K", k), ("V", v)):
                if tuple(t.shape) != expect:
                    raise ConditioningCacheError(f"layer {i} {name} shape {tuple(t.shape)} != {expect} ({self.layout})")
                if dtype is not None and t.dtype != dtype:
                    raise ConditioningCacheError(f"layer {i} {name} dtype {t.dtype} != {dtype}")
                if device is not None and t.device != torch.device(device):
                    raise ConditioningCacheError(f"layer {i} {name} on {t.device}, expected {device}")
                if not t.is_contiguous():
                    raise ConditioningCacheError(f"layer {i} {name} must be contiguous")

    def layer(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.keys_pre_rope[index], self.values[index]

    def layer_native(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer ``index`` in the native LongCat layout ``[B, H, T, D]`` (views; for parity checks)."""
        k, v = self.layer(index)
        if self.layout == LAYOUT_NATIVE_BHTD:
            return k, v
        return k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

    def layer_matrices(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Canonical ReelCache matrices ``[N, C]`` (C = H*D) of layer ``index`` (batch 1).

        For the ``BTHD`` layout these are views of the cache storage; the native layout needs a copy. Under
        context parallelism ``N`` is this rank's rows (``epoch.row_index()``), the native-owner shard layout of
        ``reelcache.factor_sharding``.
        """
        k, v = self.layer(index)
        if k.shape[0] != 1:
            raise ConditioningCacheError("canonical matrices are defined for batch 1")
        if self.layout == LAYOUT_BTHD:
            return k[0].reshape(k.shape[1], -1), v[0].reshape(v.shape[1], -1)
        return (k[0].permute(1, 0, 2).reshape(k.shape[2], -1), v[0].permute(1, 0, 2).reshape(v.shape[2], -1))

    def per_layer_sha256(self) -> list[dict]:
        """Hashes in the native layout so they compare with native references."""
        out = []
        for i in range(self.num_layers):
            k, v = self.layer_native(i)
            out.append({"K": tensor_sha256(k), "V": tensor_sha256(v)})
        return out

    def summary(self) -> dict:
        first = self.keys_pre_rope[0] if self.keys_pre_rope else None
        return {"epoch": self.epoch.as_dict(), "source": self.source, "layers": self.num_layers,
                "bytes": self.nbytes, "dtype": None if first is None else str(first.dtype).replace("torch.", ""),
                "shape_per_tensor": None if first is None else list(first.shape), **self.metadata}

    @classmethod
    def from_native_layout(cls, epoch: ConditioningCacheEpoch, keys, values, *, source: str = SOURCE_RESTORED,
                           metadata: dict | None = None) -> "LongCatConditioningCache":
        """Convert native ``[B, H, T, D]`` tensors (e.g. a ReelCache restore) to this port's ``BTHD`` layout."""
        if epoch.layout != LAYOUT_BTHD:
            raise ConditioningCacheError("target epoch must use the BTHD layout")
        ks = tuple(k.permute(0, 2, 1, 3).contiguous() for k in keys)
        vs = tuple(v.permute(0, 2, 1, 3).contiguous() for v in values)
        return cls(epoch=epoch, keys_pre_rope=ks, values=vs, source=source, metadata=dict(metadata or {}))

    @classmethod
    def from_matrices(cls, epoch: ConditioningCacheEpoch, key_matrices, value_matrices, *, heads: int,
                      source: str = SOURCE_RESTORED, metadata: dict | None = None) -> "LongCatConditioningCache":
        """Build a ``BTHD`` cache from canonical ``[N, C]`` matrices (one K and one V per layer).

        With ``epoch.row_shard`` the matrices are this rank's ``[T_local, C]`` rows in ``epoch.row_index()`` order:
        exactly what ``FactorShardDomain.restore_node(node_map, NativeOwners.spatial_rows(...), rope=None)`` returns
        per entry on the native owner (pre-RoPE keys stay unrotated; RoPE is applied at read time).
        """
        if epoch.layout != LAYOUT_BTHD:
            raise ConditioningCacheError("target epoch must use the BTHD layout")
        ks, vs = [], []
        tokens = epoch.local_tokens
        for km, vm in zip(key_matrices, value_matrices, strict=True):
            if km.ndim != 2 or km.shape != vm.shape or km.shape[0] != tokens or km.shape[1] % heads:
                raise ConditioningCacheError(f"bad matrix shape {tuple(km.shape)} for {tokens} tokens")
            ks.append(km.reshape(1, km.shape[0], heads, km.shape[1] // heads).contiguous())
            vs.append(vm.reshape(1, vm.shape[0], heads, vm.shape[1] // heads).contiguous())
        return cls(epoch=epoch, keys_pre_rope=tuple(ks), values=tuple(vs), source=source, metadata=dict(metadata or {}))


@runtime_checkable
class ConditioningCacheBackend(Protocol):
    """Hooks a ReelCache session controller implements for LongCat conditioning-cache epochs."""

    def restore(self, epoch: ConditioningCacheEpoch) -> LongCatConditioningCache | None:
        """Return a restored cache for ``epoch`` to install instead of the native build, or ``None``."""

    def on_cache_built(self, cache: LongCatConditioningCache) -> None:
        """Observe a natively built cache (capture/archive). Must not mutate the tensors."""


class CacheRowSeam:
    """Backend adapter for capturing/installing a cache as per-layer ``[T_local, C]`` token-row matrices.

    The per-rank seam of context parallelism (whole caches work the same way, ``T_local = T``): ``capture(epoch, keys,
    values)`` receives this rank's pre-RoPE K and V row matrices in ``epoch.row_index()`` order (views of the live
    cache: must not be mutated or kept beyond the call without a copy; summing ``row_shard_gram`` over ranks gives the
    full Gram for an owner fit); ``install(epoch)`` returns ``(keys, values)`` row matrices for exactly that epoch and
    rank -- e.g. ``FactorShardDomain.restore_node(node_map, NativeOwners.spatial_rows(**epoch.row_shard.
    native_owners_kwargs(epoch.token_grid, kv_heads=H, head_dim=D)), rope=None)`` per entry -- or ``None`` to build
    natively. RoPE is never applied here: keys stay pre-RoPE and are rotated at read time.
    """

    def __init__(self, *, heads: int, capture=None, install=None) -> None:
        if capture is None and install is None:
            raise ConditioningCacheError("a row seam needs a capture and/or an install callable")
        self.heads = int(heads)
        self.capture = capture
        self.install = install
        self.events: list[dict] = []

    def on_cache_built(self, cache: LongCatConditioningCache) -> None:
        if self.capture is None:
            return
        if cache.layout != LAYOUT_BTHD:
            raise ConditioningCacheError("the row seam captures BTHD caches")
        pairs = [cache.layer_matrices(i) for i in range(cache.num_layers)]
        self.capture(cache.epoch, [k for k, _ in pairs], [v for _, v in pairs])
        self.events.append({"event": "captured", "segment_index": cache.epoch.segment_index,
                            "row_shard": None if cache.epoch.row_shard is None else cache.epoch.row_shard.as_dict(),
                            "rows": cache.epoch.local_tokens})

    def restore(self, epoch: ConditioningCacheEpoch) -> LongCatConditioningCache | None:
        if self.install is None:
            return None
        rows = self.install(epoch)
        if rows is None:
            return None
        keys, values = rows
        cache = LongCatConditioningCache.from_matrices(epoch, keys, values, heads=self.heads,
                                                       metadata={"installed_by": "CacheRowSeam"})
        self.events.append({"event": "installed", "segment_index": epoch.segment_index,
                            "row_shard": None if epoch.row_shard is None else epoch.row_shard.as_dict(),
                            "rows": epoch.local_tokens})
        return cache


_REGISTRY: dict[str, ConditioningCacheBackend] = {}
_LOCK = threading.Lock()


def register_cache_backend(name: str, backend: ConditioningCacheBackend, *, replace: bool = False) -> None:
    if not name or not isinstance(name, str):
        raise ConditioningCacheError("backend name must be a non-empty string")
    if not isinstance(backend, ConditioningCacheBackend):
        raise ConditioningCacheError("backend must implement restore() and on_cache_built()")
    with _LOCK:
        if name in _REGISTRY and not replace:
            raise ConditioningCacheError(f"conditioning-cache backend {name!r} already registered")
        _REGISTRY[name] = backend


def unregister_cache_backend(name: str) -> None:
    with _LOCK:
        _REGISTRY.pop(name, None)


def get_cache_backend(name: str) -> ConditioningCacheBackend:
    with _LOCK:
        backend = _REGISTRY.get(name)
    if backend is None:
        raise ConditioningCacheError(
            f"conditioning-cache backend {name!r} was requested but is not registered in this worker process")
    return backend


def registered_cache_backends() -> list[str]:
    with _LOCK:
        return sorted(_REGISTRY)


def accept_restored_cache(cache: object, epoch: ConditioningCacheEpoch, *, layers: int, heads: int, head_dim: int,
                          dtype: torch.dtype, device) -> LongCatConditioningCache:
    """Validate a backend's restore result for ``epoch`` (no silent fallback on a malformed restore)."""
    if not isinstance(cache, LongCatConditioningCache):
        raise ConditioningCacheError(f"restore() returned {type(cache).__name__}, expected LongCatConditioningCache")
    if cache.epoch != epoch:
        raise ConditioningCacheError(f"restored cache epoch {cache.epoch} does not match requested {epoch}")
    if cache.source != SOURCE_RESTORED:
        raise ConditioningCacheError("restored caches must be labeled source='restored'")
    cache.validate(layers=layers, heads=heads, head_dim=head_dim, dtype=dtype, device=device)
    return cache
